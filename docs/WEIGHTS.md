# Weight provenance, the pickle audit, the conversion, the vendored network, the pinned dataset tarball and DIMER hosting

This repository pins **one** model snapshot with its own `dimer-base-manifest.json` and **one** dataset tarball. The checkpoint is an mmsegmentation pickle, which this pipeline audits and converts but never serves; the network is vendored in plain PyTorch; the tarball is streamed once for exactly the pinned members and never `extractall`-ed.

## Prithvi-EO-1.0-100M multi-temporal crop classification weights

- Upstream: `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification`
- Immutable revision: `b53a88b8da673800b67c34a98a527b77076e7035` (2025-09-05, "Update README.md", the repository head at pinning time); the checkpoint was first published in commit `389998ba` (2023-07-30, "Upload multi_temporal_crop_classification_Prithvi_100M.pth"); the LFS object is identical at both revisions.
- Source format: `multi_temporal_crop_classification_Prithvi_100M.pth` — torch zip archive (`archive/data.pkl`, 318 members) holding an mmsegmentation 0.30.0 / mmcv 1.6.2 checkpoint: `state_dict` (112 tensors: 109 float32, 3 int64 `num_batches_tracked`), `optimizer` (Adam state for 102 parameter tensors — two thirds of the file), `meta` (epoch 80, iter 30800, seed, the config text, `hook_msgs.best_score` as a numpy float64 scalar, `CLASSES` 1..13, `PALETTE` None).
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter).
- Local layout: `weights/prithvi-eo-1.0-100m-crop/` holds the 3 manifest entries (the checkpoint, the upstream `README.md`, the mmseg config `multi_temporal_crop_classification_Prithvi_100M.py`; 1,680,479,459 bytes total) with byte size and SHA-256 for each, plus the converted file described below. `verify_snapshot()` in `src/prithvi_crop_classification_pipeline/pipeline.py` re-hashes every entry and, when present, the converted file.
- Cross-check: the manifest's checkpoint digest `37ed41637eccccec65ca2031324e2c03a4f168e1ea0ea71ad180910589fa018c` equals the `oid sha256` of the Hub LFS pointer at the pinned revision (size 1680468041).

## What the pickle would execute, and how it is audited

Under the fleet asset specification (§11) a pickle is executable serialization. `audit_pickle()` disassembles the file with `pickletools.genops` — every `.pkl` inside the torch zip archive — collects every `GLOBAL` / `STACK_GLOBAL` it would import, and refuses anything outside the allow-list, executing nothing:

| File | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|
| `multi_temporal_crop_classification_Prithvi_100M.pth` | `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2` (the state dict) and `numpy.core.multiarray.scalar`, `numpy.dtype`, `_codecs.encode` (the one numpy scalar in `meta`) | exactly those seven | `465513635354b8c8a9c4bfd4a37b39d5306eb8ec63527905ce0195075d04108d` |

The audit reports 0 violations and its digest is pinned in `PICKLE_AUDIT_SHA256`; `convert_model()` refuses a file whose audit digest differs. The three extra names are the standard way numpy pickles a scalar (`scalar(dtype("f8"), encode("<8 latin-1 bytes>"))`): they construct a value, not code, and the audit lists them separately under `meta_globals`. Tests craft a torch archive carrying `os.system`, a plain pickle of a `complex` number and an mmseg-shaped stream naming the three `meta` globals, and assert that the audit refuses the first two, accepts the third and refuses it again under the fleet's torch-only allow-list.

An allow-list bounds what the unpickler can name; the loader below bounds what it can construct. The digest pins tie the audited bytes to the loaded bytes, and the unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs size check → SHA-256 check against the package constant → static audit and audit-digest check, and only then:

- `restricted_load()`: `torch.load(map_location="cpu", weights_only=True)` — torch's restricted unpickler, which constructs tensors and containers and nothing else — inside `torch.serialization.safe_globals` with the three `meta` names bound to **inert stand-ins** (`_MetaValue`, a class that records its arguments and ignores its state) rather than to numpy. Nothing from numpy is imported or executed by the pickle; the scalar unpickles to a stand-in and is discarded. Under numpy 2 the legacy name `numpy.core.multiarray.scalar` would not even resolve to an allow-listable object without this indirection, and `numpy.dtype` builds a `Float64DType` that the weights-only loader rejects — the stand-ins avoid both.
- the result must be a dict with a `state_dict` of exactly 112 tensors; the 14 `auxiliary_head.*` tensors (mmsegmentation's training-only auxiliary FCN head) are dropped, and the remaining 98 are loaded with `strict=True` into `PrithviCropSegmenter()` from `modeling.py` (encoder: 6 blocks, width 768, 8 heads, 3 dates × 6 bands at 224 × 224; neck: 2304 channels, four stride-2 transposed convolutions; head: 256 channels, 13 classes) — 0 missing, 0 unexpected, 0 shape mismatches;
- the network's own state dict (98 tensors, 134,428,174 elements, of which 134,427,661 are parameters and 513 are the head's BatchNorm buffers) is saved with `safetensors.torch.save_file`; the `optimizer` and `meta` entries are read and discarded (the conversion record keeps the epoch, iteration and framework versions).

Serving file (both identities recorded, `derived_from_sha256` = the source digest above):

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `prithvi-eo-1.0-100m-crop.safetensors` | 537,722,508 | 98 (134,428,174 elements; 134,427,661 parameters) | `d1df8044700a0d1e00b11b1fbac66e0495cf6647e1632a858c003eaf4b5ce36d` | no (regenerated) |

The conversion is deterministic: the digest was reproduced on two consecutive build conversions and on the executed tutorial notebook, which converts the file it downloads. `verify_converted()` checks size and digest; `from_pretrained()` loads the safetensors with `strict=True` and asserts the parameter count.

## The vendored network

`modeling.py` reimplements, in plain PyTorch, `PatchEmbed`, `TemporalViTEncoder` (with the 3-D sin/cos positional table of `get_3d_sincos_pos_embed`), `Norm2d` and `ConvTransformerTokensToEmbeddingNeck` from `geospatial_fm/geospatial_fm.py` of `NASA-IMPACT/hls-foundation-os` at commit `3b6d401f3b4527059af0e44bd640225285e1933d` (file SHA-256 `5e16a033253c358d…`, Apache-2.0), timm's pre-norm `Block` / `Attention` / `Mlp` as that file uses them (fused qkv with bias, no layer scale, LayerNorm eps 1e-5), and mmsegmentation 0.30's `FCNHead` with `num_convs=1`, `concat_input=False`, `dropout_ratio=0.1` (mmcv's `ConvModule` order conv → BatchNorm → ReLU, no conv bias) plus the `encode_decode` resize to the input size. Parameter and buffer names reproduce the checkpoint's exactly (`backbone.*`, `neck.*`, `decode_head.*`), which is what lets the conversion load with `strict=True`. No mmcv, mmseg, timm or einops is imported anywhere in the package.

Two checks tie the vendored code to the checkpoint. The positional table computed by `sincos_pos_embed_3d(768, (3, 14, 14), cls_token=True)` equals `backbone.pos_embed` with a maximum absolute difference of 0.0. And the input layout: the upstream data pipeline (`geospatial_pipelines.py` at the same commit, `TorchNormalize` then `Reshape(new_shape=(6, 3, 224, 224))`) standardised the 18 date-major channels of the chip with the six-band statistics repeated three times and then applied a plain `torch.reshape` — so channel `c` of the network's "band" axis holds input channels `3c`, `3c + 1`, `3c + 2`, not band `c` at three dates. The checkpoint learned that layout: on the tutorial's 12 test chips the network reaches 59.0 % pixel accuracy with the reshape and 15.1 % with a bands-by-dates layout. `pipeline._normalise` reproduces the reshape and says so; `tests/test_adaptation.py::test_normalise_reproduces_the_upstream_reshape` pins it.

## Fidelity

No upstream regression fixture is published for this checkpoint. The evidence is the strict key-and-shape match, the bit-exact positional table, and the agreement with the upstream card on the chips the card was scored on: the upstream card reports a mean IoU of 0.4269, an overall accuracy of 60.64 % and a mean class accuracy of 64.06 % on the full validation split; on the 12 pinned test chips (all from that split) the vendored network reaches 0.4379 / 59.05 % / 62.39 % with Open Water the easiest class here as there (IoU 0.81 against the card's 0.68) and Natural Vegetation and Other among the hardest (0.25 and 0.28). Sample-sanity evidence, not a reproduction of the benchmark.

## Runtime facts

- The model is float32 as shipped; `predict` runs under `torch.inference_mode()` with float16 autocast on CUDA and moves results to the CPU; `adapt` trains with gradients only on the selected tensors, with BatchNorm running statistics frozen. Float32 on a GPU is much slower than float16 here (the 2304-channel transposed convolutions of the neck fall back to slow kernels), and the first forward on a new GPU may pay a long kernel warm-up.
- Inputs are standardised with the per-band means and standard deviations of the pinned mmseg config, in digital numbers (HLS reflectance × 10 000); a chip whose values all lie in [0, 1.5] is taken as reflectance and scaled by 10 000 first. The dataset's chips are int16 in the same units, so nothing else is applied.
- The package depends on `torch`, `numpy`, `tifffile`, `safetensors` and `huggingface-hub` only; chips are read with `tifffile` (no rasterio, no georeferencing).

## The tutorial data: the pinned multi-temporal crop classification tarball

`samples.py` fetches `validation_chips.tgz` from the Hugging Face dataset `ibm-nasa-geospatial/multi-temporal-crop-classification` at the immutable revision `f285bb27c8f623a0fb6a44a6fd953c3ad34007d6` (1,179,542,384 bytes, SHA-256 `d6e616cc008858a1935a8937e0cdf852d574754273b0655fb696bd29aebd2fd3`, hashed once per `fetch_tarball` call and refused on a mismatch), then streams through it **once** with `extract_pinned_members`, copying out exactly the 120 members pinned in `SAMPLE_RECORDS` (60 `_merged.tif` chips of 1,808,174 bytes — 224 × 224 × 18 int16, pixel-interleaved — and 60 `.mask.tif` masks of 224 × 224 uint8), each refused on a size or digest mismatch and written under its base name in `weights/multi-temporal-crop/chips/`. Nothing else in the archive is written: not the other 1,422 real members (771 chips with their 771 masks in all), and not the 1,543 macOS `._` resource-fork twins the archive carries (which are not TIFFs).

Roles: each chip key `chip_<row>_<col>` is mapped to a 4 × 4 block of the chip grid (`chip_block`), each block to a role by a seeded SHA-256 hash (60 / 20 / 20 %), and 36 / 12 / 12 chips were drawn per role with a fixed seed, round-robin over dominant classes so every one of the 13 classes occurs in every role. Chips of one block never straddle roles; `check_split_disjoint` refuses a chip (by pixel digest) or a block (by `region`) in two splits. The 771 chips span 600 blocks, so the grouping seldom binds — the split is close to a split by chip, and neighbouring blocks may share fields; real data must be split by region. Every chip belongs to the upstream validation split, on which the published checkpoint's best epoch was chosen.

## Files deliberately not staged

The upstream repository at the pinned revision also carries `config.yaml` (a TerraTorch configuration added in 2025) and `multi_temporal_crop_classification.png`; neither is listed in the manifest. The pretrained (non-fine-tuned) `Prithvi-100M` backbone is not fetched: the fine-tuned checkpoint carries the encoder. The dataset's `training_chips.tgz`, `chips_df.csv` and split lists are not fetched. Of the validation tarball, only the 120 pinned members are ever written to disk.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the licence and notices. DIMER may host the converted safetensors in its model store under those terms; it is derived from, and recorded beside, the unmodified upstream checkpoint.
- Upload set: `prithvi-eo-1.0-100m-crop.safetensors`. **The `.pth` file must not be uploaded** — a profile that carries it would reintroduce the executable-serialization boundary this conversion removes, and two thirds of its bytes are optimizer state nobody serves.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no mmcv or mmsegmentation, no pickle on the serving path; the network is this repository's `modeling.py`, the served state dict is safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified file without the manifest or the checkpoint.
- Serving shape: a chip needs the 538 MB weights and one (3, 6, 224, 224) array; on an RTX 5070 Ti laptop GPU 12 chips take 0.7 s in float16 autocast once warm; a CPU takes seconds per chip. An adapted profile needs the weights plus a 21 MB adapter (the head) or 50 MB (head + last block).
- One review item is open: whether the one-time restricted unpickle (in the build and, for the tutorial, in the runtime) meets the DIMER deserialization-trust bar or whether DIMER hosts only maintainer-converted files. The served artifact is the same file either way.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
