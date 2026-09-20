"""The Prithvi-EO-1.0-100M multi-temporal crop-classification network in plain PyTorch.

Vendored from the upstream training code (`NASA-IMPACT/hls-foundation-os`, `geospatial_fm/geospatial_fm.py` at
commit `3b6d401f3b4527059af0e44bd640225285e1933d`, Apache-2.0) and from the parts of `mmsegmentation` 0.30 it was
trained with (`FCNHead` with one conv, the `ConvModule` conv→BN→ReLU order), rewritten without `mmcv`, `mmseg`,
`timm` or `einops` so that the served checkpoint is rebuilt from a dependency-free module. The parameter and
buffer names reproduce the upstream state dict exactly (`backbone.*`, `neck.*`, `decode_head.*`), which is how
the conversion can load it with `strict=True`; the training-only `auxiliary_head` is not part of the network.

Architecture (from the pinned `multi_temporal_crop_classification_Prithvi_100M.py` config):

* `TemporalViTEncoder` — a 3-D patch embedding (`Conv3d`, tubelet 1 × 16 × 16) over (bands, dates, H, W), a
  fixed 3-D sin/cos positional embedding plus a class token, 6 pre-norm transformer blocks of width 768 with 8
  heads (MLP ratio 4), and a final LayerNorm; the first 6 blocks of the 12-block Prithvi-EO-1.0-100M masked
  autoencoder, fine-tuned end to end.
* `ConvTransformerTokensToEmbeddingNeck` — drops the class token, folds the (dates × 14 × 14) tokens into a
  (768 × 3, 14, 14) map, and upsamples it 16× through four `ConvTranspose2d` layers (two with a LayerNorm + GELU).
* `FCNHead` — one 3 × 3 conv (2304 → 256) with BatchNorm and ReLU, Dropout2d(0.1), and a 1 × 1 conv to the 13
  crop / land-cover classes, at the input resolution.

The whole network takes a (B, 6, 3, 224, 224) standardised stack — six HLS bands × three dates, in the exact
layout the upstream data pipeline produced (see `pipeline._normalise`) — and returns (B, 13, 224, 224) logits.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _sincos_1d(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    """(M,) positions -> (M, embed_dim) sin/cos table (float64, as the upstream numpy code)."""
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / torch.pow(10000.0, omega)
    out = positions.to(torch.float64).reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def sincos_pos_embed_3d(embed_dim: int, grid_size: tuple[int, int, int], *, cls_token: bool) -> torch.Tensor:
    """The fixed 3-D positional embedding of the upstream `get_3d_sincos_pos_embed`: 6/16 of the width for the
    column, 6/16 for the row and 4/16 for the date, concatenated per token in (date, row, column) order, with a
    zero row prepended for the class token."""
    if embed_dim % 16:
        raise ValueError("embed_dim must be a multiple of 16")
    t_size, h_size, w_size = grid_size
    w_dim = h_dim = embed_dim // 16 * 6
    t_dim = embed_dim // 16 * 4
    w_pos = _sincos_1d(w_dim, torch.arange(w_size)).repeat(t_size * h_size, 1)
    h_pos = _sincos_1d(h_dim, torch.arange(h_size)).repeat_interleave(w_size, dim=0).repeat(t_size, 1)
    t_pos = _sincos_1d(t_dim, torch.arange(t_size)).repeat_interleave(h_size * w_size, dim=0)
    pos = torch.cat([w_pos, h_pos, t_pos], dim=1)
    if cls_token:
        pos = torch.cat([torch.zeros(1, embed_dim, dtype=pos.dtype), pos], dim=0)
    return pos.to(torch.float32)


class PatchEmbed(nn.Module):
    """Frames of images to patch embeddings: a Conv3d with kernel = stride = (tubelet, patch, patch)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 3,
        tubelet_size: int = 1,
        in_chans: int = 6,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (num_frames // tubelet_size, img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, _t, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(f"input spatial size {(height, width)} does not match the model's {self.img_size}")
        x = self.proj(x)  # (B, D, T', H', W')
        return x.flatten(2).transpose(1, 2)  # (B, T'·H'·W', D), date-major token order


class Attention(nn.Module):
    """Multi-head self-attention with a fused qkv projection (timm's `Attention`, no q/k norms)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block (timm's `Block` with `qkv_bias=True`, no layer scale, no drop path)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # upstream passes nn.LayerNorm unchanged: eps 1e-5
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TemporalViTEncoder(nn.Module):
    """The fine-tuned Prithvi encoder: patch embedding, fixed 3-D positional embedding, class token, blocks."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 3,
        tubelet_size: int = 1,
        in_chans: int = 6,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_embed = PatchEmbed(img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            sincos_pos_embed_3d(embed_dim, self.patch_embed.grid_size, cls_token=True).unsqueeze(0), requires_grad=False
        )
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, 1 + T·H'·W', D) normalised tokens, class token first."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class Norm2d(nn.Module):
    """LayerNorm over the channel axis of an NCHW map."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """Tokens -> (B, embed_dim, Hp, Wp) map -> 16× upsampled (B, output_embed_dim, 16·Hp, 16·Wp) map through
    four stride-2 transposed convolutions. `embed_dim` is the token width × the number of dates (2304): the
    (dates, Hp, Wp) tokens are folded into the channel axis by a plain reshape, exactly as upstream."""

    def __init__(self, embed_dim: int = 2304, output_embed_dim: int = 2304, hp: int = 14, wp: int = 14) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.hp, self.wp = hp, wp
        self.h_out, self.w_out = hp * 16, wp * 16
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, output_embed_dim, kernel_size=2, stride=2),
            Norm2d(output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, kernel_size=2, stride=2),
            Norm2d(output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, kernel_size=2, stride=2),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens[:, 1:, :]  # drop the class token
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.hp, self.wp)
        x = self.fpn2(self.fpn1(x))
        return x.reshape(-1, self.output_embed_dim, self.h_out, self.w_out)


class ConvModule(nn.Module):
    """mmcv's ConvModule in its default order: conv (no bias) -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class FCNHead(nn.Module):
    """mmseg's FCNHead with `num_convs=1`, `concat_input=False`, `dropout_ratio=0.1`."""

    def __init__(self, in_channels: int = 2304, channels: int = 256, num_classes: int = 13, dropout_ratio: float = 0.1) -> None:
        super().__init__()
        self.convs = nn.Sequential(ConvModule(in_channels, channels, kernel_size=3, padding=1))
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_seg(self.dropout(self.convs(x)))


class PrithviCropSegmenter(nn.Module):
    """Encoder + neck + head: (B, bands, dates, 224, 224) standardised -> (B, num_classes, 224, 224) logits."""

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 3,
        in_chans: int = 6,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        head_channels: int = 256,
        num_classes: int = 13,
    ) -> None:
        super().__init__()
        grid = img_size // patch_size
        self.backbone = TemporalViTEncoder(img_size, patch_size, num_frames, 1, in_chans, embed_dim, depth, num_heads)
        self.neck = ConvTransformerTokensToEmbeddingNeck(embed_dim * num_frames, embed_dim * num_frames, grid, grid)
        self.decode_head = FCNHead(embed_dim * num_frames, head_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decode_head(self.neck(self.backbone(x)))
        if tuple(logits.shape[-2:]) != tuple(x.shape[-2:]):  # mmseg resizes the head output to the input size
            logits = nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def check_shapes(module: PrithviCropSegmenter) -> dict[str, int]:
    """Sanity numbers used by the tests and the conversion: tensors, parameters and buffers."""
    state = module.state_dict()
    return {
        "tensors": len(state),
        "elements": sum(int(math.prod(v.shape)) for v in state.values()),
        "parameters": parameter_count(module),
    }
