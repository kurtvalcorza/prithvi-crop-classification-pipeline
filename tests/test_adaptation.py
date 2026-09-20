"""Adaptation, evaluation and artifact tests on a stub model (torch required, no weights): the scope of the
trainable tensors, epoch selection, the transactional guarantee and the artifact round trip. Plus the vendored
architecture's shape and key contract on a randomly initialised instance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import synthetic_records  # noqa: E402
from prithvi_crop_classification_pipeline import ADAPTATION_MODES, NUM_CLASSES, PrithviCropPipeline, build_model  # noqa: E402
from prithvi_crop_classification_pipeline import pipeline as pl  # noqa: E402
from prithvi_crop_classification_pipeline.modeling import check_shapes, sincos_pos_embed_3d  # noqa: E402


class _StubModel(torch.nn.Module):
    """Parameter names follow the vendored layout; the logits depend on the head tensors so training moves them."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(6)])
        for block in self.backbone.blocks:
            block.w = torch.nn.Parameter(torch.ones(1))
        self.neck = torch.nn.Module()
        self.neck.scale = torch.nn.Parameter(torch.ones(1))
        self.decode_head = torch.nn.Module()
        self.decode_head.weight = torch.nn.Parameter(torch.zeros(6))  # frozen stub predicts class 0 everywhere
        self.decode_head.bias = torch.nn.Parameter(torch.zeros(NUM_CLASSES))

    def forward(self, x):
        feature = (x[:, :, 0] * self.decode_head.weight[None, :, None, None]).sum(dim=1) * self.neck.scale
        feature = feature * self.backbone.blocks[5].w
        ramp = torch.linspace(-1.0, 1.0, NUM_CLASSES, device=x.device)
        return feature[:, None] * ramp[None, :, None, None] + self.decode_head.bias[None, :, None, None]


def _pipeline() -> PrithviCropPipeline:
    return PrithviCropPipeline(model=_StubModel(), device="cpu", weights_dir=Path("unused"), source="stub")


def test_trainable_scopes():
    pipe = _pipeline()
    assert pipe._trainable("head") == ["decode_head.bias", "decode_head.weight"]
    assert pipe._trainable("head+last_block") == ["backbone.blocks.5.w", "decode_head.bias", "decode_head.weight"]
    assert ADAPTATION_MODES == ("head", "head+last_block")
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe._trainable("everything")


def test_predict_and_evaluate_shapes():
    pipe = _pipeline()
    records = synthetic_records(3)
    result = pipe.predict(records, batch_size=2)
    assert len(result["predictions"]) == 3 and result["predictions"][0]["mask"].shape == (224, 224)
    assert result["predictions"][0]["scores"].shape == (13, 224, 224) and result["decision_rule"].startswith("argmax")
    assert abs(sum(result["predictions"][0]["class_fraction"].values()) - 1.0) < 1e-3
    assert result["predictions"][0]["class_fraction"]["Natural Vegetation"] == 1.0
    report = pipe.evaluate(records)
    assert set(report["model"]) >= {"iou", "mean_iou", "mean_accuracy", "accuracy", "recall"}
    assert report["baseline_majority"]["majority_class"] in pl.CLASS_NAMES and report["model"]["classes_scored"] == 13
    with pytest.raises(ValueError, match="batch_size"):
        pipe.predict(records, batch_size=0)


def test_adapt_trains_only_the_scope_and_keeps_the_best_epoch():
    pipe = _pipeline()
    train, val = synthetic_records(6), synthetic_records(2, seed=50)
    block_before = pipe.model.backbone.blocks[0].w.clone()
    before = pipe.evaluate(val)["model"]["mean_iou"]
    seen = []
    result = pipe.adapt(train, val, epochs=3, lr=1e-2, batch_size=2, seed=0, progress=seen.append)
    assert [e["epoch"] for e in seen] == [0, 1, 2, 3] and seen[0]["note"] == "frozen model"
    assert seen[0]["val"]["mean_iou"] == before
    assert result["best_epoch"] == min(range(4), key=lambda i: result["history"][i]["val_loss"])
    assert result["n_trainable"] == 19 and result["n_steps"] == 9 and result["precision"] == "float32"
    assert result["loss"].startswith("cross-entropy with the upstream class weights")
    assert torch.equal(pipe.model.backbone.blocks[0].w, block_before)
    assert pipe.adapter is not None and all(not p.requires_grad for p in pipe.model.parameters())
    assert pipe.evaluate(val)["model"] == result["history"][result["best_epoch"]]["val"]
    assert result["history"][-1]["train_loss"] < result["history"][1]["train_loss"] or result["best_epoch"] >= 1


def test_adapt_is_transactional_when_the_progress_callback_raises():
    pipe = _pipeline()
    initial = {k: v.clone() for k, v in pipe.model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipe.adapt(synthetic_records(4), None, epochs=2, lr=1e-2, progress=boom)
    assert pipe.adapter is None
    assert all(torch.equal(initial[k], v) for k, v in pipe.model.state_dict().items())
    assert all(not p.requires_grad for p in pipe.model.parameters())


def test_adapt_refusals():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(synthetic_records(4), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(synthetic_records(4), lr=1.0)
    with pytest.raises(ValueError, match="4..2000"):
        pipe.adapt(synthetic_records(3))
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe.adapt(synthetic_records(4), trainable="all")
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact("unused")


def test_artifact_round_trip_and_refusals(tmp_path):
    pipe = _pipeline()
    records = synthetic_records(4)
    pipe.adapt(records, epochs=1, lr=1e-2, trainable="head+last_block")
    adapted = pipe.evaluate(records)["model"]
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"tutorial": "test"})
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["tensors"] == ["backbone.blocks.5.w", "decode_head.bias", "decode_head.weight"]
    assert manifest["adapter"]["trainable"] == "head+last_block" and manifest["metadata"] == {"tutorial": "test"}
    fresh = _pipeline()
    assert fresh.evaluate(records)["model"] != adapted
    fresh.load_artifact(out)
    assert fresh.evaluate(records)["model"] == adapted and fresh.adapter["best_epoch"] == 1
    # a scope narrower than the tensor list is refused before deserialising
    narrowed = dict(manifest, adapter={**manifest["adapter"], "trainable": "head"})
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(narrowed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _pipeline().load_artifact(out)
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (out / pl.ARTIFACT_WEIGHTS_NAME).write_bytes((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest or size"):
        _pipeline().load_artifact(out)
    assert np.isfinite(adapted["mean_iou"])


# --- the vendored architecture (random weights) ----------------------------------------------------------


def test_vendored_architecture_matches_the_checkpoint_contract():
    model = build_model()
    shapes = check_shapes(model)
    assert shapes == {"tensors": pl.STATE_TENSORS, "elements": pl.STATE_NUMEL, "parameters": pl.PARAMETER_COUNT}
    state = model.state_dict()
    prefixes = {k.split(".")[0] for k in state}
    assert prefixes == {"backbone", "neck", "decode_head"} and not any(k.startswith("auxiliary_head.") for k in state)
    assert tuple(state["backbone.pos_embed"].shape) == (1, 589, 768)
    assert tuple(state["decode_head.conv_seg.weight"].shape) == (13, 256, 1, 1)
    assert tuple(state["backbone.patch_embed.proj.weight"].shape) == (768, 6, 1, 16, 16)
    assert tuple(state["neck.fpn2.3.weight"].shape) == (2304, 2304, 2, 2)
    assert "decode_head.convs.0.bn.num_batches_tracked" in state
    assert not model.backbone.pos_embed.requires_grad
    # the fixed positional table: zero row for the class token, unit-norm sin/cos pairs, date-major token order
    table = sincos_pos_embed_3d(768, (3, 14, 14), cls_token=True)
    assert tuple(table.shape) == (589, 768) and torch.all(table[0] == 0)
    assert torch.allclose(table[1, :288] ** 2 + torch.roll(table[1, :288], 144) ** 2, torch.ones(288), atol=1e-5)
    assert torch.allclose(table[1, 576:], table[2, 576:]) and not torch.allclose(table[1, 576:], table[1 + 14 * 14, 576:])
    with torch.inference_mode():
        logits = model.eval()(torch.zeros(1, 6, 3, 224, 224))
    assert tuple(logits.shape) == (1, 13, 224, 224)
    with pytest.raises(ValueError, match="does not match the model's"):
        model(torch.zeros(1, 6, 3, 256, 256))


def test_normalise_reproduces_the_upstream_reshape():
    """18 date-major channels standardised per band, then folded into (6, 3, H, W) by a plain reshape (not by
    bands and dates): channel c of the output holds input channels 3c, 3c + 1, 3c + 2."""
    images = np.zeros((1, 3, 6, 4, 4), dtype=np.float32)
    for t in range(3):
        for b in range(6):
            images[0, t, b] = pl.MEANS[b] + pl.STDS[b] * (t * 6 + b)  # standardises to the channel index
    out = pl._normalise(images)
    assert out.shape == (1, 6, 3, 4, 4)
    for c in range(6):
        for k in range(3):
            assert np.allclose(out[0, c, k], 3 * c + k, atol=1e-4)
