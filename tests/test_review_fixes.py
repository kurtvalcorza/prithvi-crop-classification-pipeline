"""Regression tests for the Notebook Review Framework v1 findings on prithvi_crop_classification_colab.ipynb
(review of 2026-10-02, prefix CR): CR-M1, CR-M3(a) and CR-M4 were fixed by the 2026-10-05 sweep (tests in
test_sweep_fixes.py); this file covers CR-M2, CR-M3(b), CR-m1..m6 and the trivial suggestions CR-S5/S6.

Every test needs only CI's dependencies (numpy, tifffile): synthetic chips stand in for HLS chips, a NumPy stand-in
pipeline stands in for the model, and the notebook's own cells are executed against them. Stand-in evidence is
plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import struct
import types
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_chip, synthetic_records
from prithvi_crop_classification_pipeline import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    PngImage,
    byod_minimum_records,
    class_map_panel,
    class_palette,
    false_colour_composite,
    load_byod_dataset,
    majority_baseline,
    per_chip_metrics,
    render_png,
    segmentation_metrics,
    split_dataset,
    write_dataset_csv,
    write_sample_pair,
)
from prithvi_crop_classification_pipeline import metrics as mt
from prithvi_crop_classification_pipeline import pipeline as pl
from prithvi_crop_classification_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "prithvi_crop_classification_colab.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _write_zip(path: Path, records, *, group=None, missing_member=False, bad_image=False):
    columns = "id,image,label" + (",group" if group else "")
    rows = [columns]
    folder = path.parent / f"chips_{path.stem}"
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for i, record in enumerate(records):
            image_name, label_name = f"{record['id']}_merged.tif", f"{record['id']}.mask.tif"
            rows.append(f"{record['id']},{image_name},{label_name}" + (f",{group(i)}" if group else ""))
            if missing_member and i == 0:
                continue
            write_sample_pair(record, folder / image_name, folder / label_name)
            archive.writestr(image_name, b"not a tiff" if (bad_image and i == 0) else (folder / image_name).read_bytes())
            archive.writestr(label_name, (folder / label_name).read_bytes())
        archive.writestr("pairs.csv", "\n".join(rows) + "\n")
    return path


# --- CR-m1: the true minimum is stated and refused by name; failures name the row, the file and the fix ---------------


def test_cr_m1_minimum_is_seven_and_a_six_chip_set_is_refused_with_the_minimum():
    assert byod_minimum_records() == 7
    records = synthetic_records(7)
    splits = split_dataset(records, seed=0)
    assert {k: len(v) for k, v in splits.items()} == {"test": 2, "validation": 1, "train": 4}
    with pytest.raises(ValueError, match=r"6 distinct labelled chips split into .* bring at least 7 chips"):
        split_dataset(records[:6], seed=0)
    with pytest.raises(ValueError, match="bring at least 7 chips"):
        split_dataset(records[:4], seed=0)


def test_cr_m1_missing_member_and_non_tiff_name_the_row_file_and_fix(tmp_path):
    records = synthetic_records(2)
    with pytest.raises(ValueError, match=r"pairs.csv row 'chip-000': image file 'chip-000_merged.tif' is listed but not in the zip; add the file"):
        load_byod_dataset(_write_zip(tmp_path / "missing.zip", records, missing_member=True))
    with pytest.raises(ValueError, match=r"pairs.csv row 'chip-000': image file 'chip-000_merged.tif' is not a readable GeoTIFF .*18-band 224 × 224"):
        load_byod_dataset(_write_zip(tmp_path / "bad.zip", records, bad_image=True))
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "pairs.csv").write_text("id,image,label\nx,x.tif,x_mask.tif\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is listed but not in the folder"):
        load_byod_dataset(folder)


def test_cr_m1_notebook_states_the_true_minimum_and_pads_the_probes(notebook):
    md = _markdown(notebook)
    assert "at least four chips" not in md and "at least **7** labelled chips" in md and "bring at least 7 chips" in md
    source = _cell(notebook, "if USE_BYOD:")
    assert "'minimum_chips': byod_minimum_records()" in source
    assert "probe_fill = (test_records[1:] + train_records + val_records)[:MIN_RECORDS - 1]" in source and "*test_records[1:4]" not in source


# --- CR-m6: a `group` column never puts one group in two roles; Section 4 documents it -------------------------------


def test_cr_m6_group_column_keeps_every_group_in_one_role(tmp_path):
    records = synthetic_records(12)
    loaded = load_byod_dataset(_write_zip(tmp_path / "grouped.zip", records, group=lambda i: f"field-{i % 4}"))
    assert all(r["group"] == r["region"] == f"field-{i % 4}" for i, r in enumerate(loaded))
    for seed in range(5):
        splits = split_dataset(loaded, seed=seed)
        roles = {}
        for role, members in splits.items():
            for record in members:
                assert roles.setdefault(record["group"], role) == role, f"group {record['group']} in two roles"
        assert set(roles) == {f"field-{i}" for i in range(4)} and len(splits["train"]) >= 4
        sm.check_split_disjoint(splits)  # the block/region leakage check agrees with the group split
    assert sum(len(v) for v in split_dataset(records, seed=0).values()) == 12
    with pytest.raises(ValueError, match="chips have no 'group'"):
        split_dataset([{**r, "group": "a"} if i % 2 else r for i, r in enumerate(records)], seed=0)
    with pytest.raises(ValueError, match="at least 3 groups"):
        split_dataset([{**r, "group": "a" if i < 6 else "b"} for i, r in enumerate(records)], seed=0)
    rows = write_dataset_csv([{**records[0], "region": "r001c002"}], tmp_path / "t.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "id,image,label,group,source" and rows[1].startswith("chip-000,chip-000_merged.tif,chip-000.mask.tif,r001c002,")


def test_cr_m6_section_4_documents_the_group_column(notebook):
    md = _markdown(notebook)
    assert "the `group` column of `pairs.csv` applies it" in md and "'grouped_split': byod_grouped" in _cell(notebook, "if USE_BYOD:")


# --- CR-M3(b): an earlier head+last_block run is undone before a head run, so the exported adapter is the model -------


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.value.copy())


class _FakeModel:
    def __init__(self):
        self.state = {"decode_head.weight": _Tensor([0.0]), "backbone.blocks.5.w": _Tensor([1.0])}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state, strict=True):
        assert strict and set(state) == set(self.state)
        self.state = {k: _Tensor(v.value.copy()) for k, v in state.items()}

    def eval(self):
        pass


def test_cr_m3b_last_block_from_an_earlier_scope_is_restored_before_a_head_run():
    model = _FakeModel()
    pipe = pl.PrithviCropPipeline(model=model, device="cpu", weights_dir=Path("."), source="stand-in")
    # run 1: head+last_block "trains" both tensors
    pl._remember_base(pipe._base_state, model, ["backbone.blocks.5.w", "decode_head.weight"])
    model.state["backbone.blocks.5.w"] = _Tensor([1.0244])
    model.state["decode_head.weight"] = _Tensor([0.7])
    # run 2 (head only) begins as adapt() does: restore everything an earlier run changed, then remember the head
    restored = pipe.restore_base()
    assert restored == ["backbone.blocks.5.w", "decode_head.weight"]
    assert model.state["backbone.blocks.5.w"].value.tolist() == [1.0] and model.state["decode_head.weight"].value.tolist() == [0.0]
    pl._remember_base(pipe._base_state, model, ["decode_head.weight"])
    assert pipe._base_state["backbone.blocks.5.w"].value.tolist() == [1.0]  # the base copy is never overwritten
    text = (ROOT / "src" / "prithvi_crop_classification_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def restore_base(")]
    assert adapt.index("restored = self.restore_base()") < adapt.index("_remember_base(self._base_state, model, names)")


# --- CR-m3: the tensor count ------------------------------------------------------------------------------------------


def test_cr_m3_tensor_count_is_five_everywhere(notebook):
    md = _markdown(notebook)
    assert "8 tensors" not in md and "8-tensor" not in md
    assert "the 5 trainable tensors of the FCN head" in md and "the 5-tensor, 21 MB adapter" in md
    docs = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "(8 tensors, about 21 MB)" not in docs and "(5 tensors" in docs


# --- CR-m4: the class maps are shown inline (no plotting library) ---------------------------------------------------


def _decode_png(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        assert struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])[0] == zlib.crc32(kind + payload) & 0xFFFFFFFF
        chunks.setdefault(kind, b"")
        chunks[kind] += payload
        pos += 12 + length
    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, colour) == (8, 2)
    raw = zlib.decompress(chunks[b"IDAT"])
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, 1 + width * 3)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(height, width, 3)


def test_cr_m4_panel_and_png_round_trip():
    image, label = synthetic_chip(seed=2, size=32)
    frozen = np.clip(label, 0, NUM_CLASSES - 1)
    adapted = np.where(label == 3, 4, frozen)
    panel = class_map_panel({"image": image, "label": label}, frozen, adapted)
    assert panel["rgb"].shape == (32, 4 * 32 + 3 * 6, 3) and panel["rgb"].dtype == np.uint8 and len(panel["panels"]) == 4
    palette = class_palette()
    assert palette.shape == (NUM_CLASSES, 3) and len({tuple(c) for c in palette.tolist()}) == NUM_CLASSES
    assert [e["class"] for e in panel["legend"]] == list(CLASS_NAMES)
    reference = panel["rgb"][:, 38:70]
    assert (reference[0, 0] == 0).all()  # the no-data corner is black
    assert (reference[20, 20] == palette[label[20, 20]]).all()
    assert (panel["rgb"][:, 32:38] == 255).all()  # the gutter
    png = render_png(panel["rgb"])
    assert np.array_equal(_decode_png(png), panel["rgb"])
    assert PngImage(png, caption="x")._repr_png_() == png and "bytes: x" in repr(PngImage(png, caption="x"))
    composite = false_colour_composite(image)
    assert composite.shape == (32, 32, 3) and composite.min() >= 0.0 and composite.max() <= 1.0
    with pytest.raises(ValueError, match="expected a"):
        false_colour_composite(image[:2])
    with pytest.raises(ValueError, match="mask shape"):
        class_map_panel({"image": image, "label": label}, frozen[:10], adapted)


# --- CR-S5: per-chip metrics ------------------------------------------------------------------------------------------


def test_cr_s5_per_chip_metrics_match_the_pooled_metric_on_one_chip():
    _, label = synthetic_chip(seed=0, size=32)
    pred = np.clip(label, 0, NUM_CLASSES - 1)
    rows = per_chip_metrics([pred, np.zeros_like(pred)], [label, label], class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX, ids=["a", "b"])
    assert rows[0]["id"] == "a" and rows[0]["mean_iou"] == 1.0 and rows[0]["accuracy"] == 1.0
    assert rows[1]["mean_iou"] == segmentation_metrics([np.zeros_like(pred)], [label], class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX)["mean_iou"]
    assert rows[0]["pixels"] == 32 * 32 - 64


# --- CR-M2 + CR-m4 + CR-S5/S6: Sections 4-8 run end to end on BYOD records (no source_id / region) with stand-ins -----


class _StandInPipe:
    """A model-free pipeline: the frozen stand-in mislabels a band of pixels, the adapted one reproduces the label."""

    device = "cpu"
    source = "stand-in"

    def __init__(self, adapted=None):
        self.adapter = adapted
        self._base_state = {}

    def _masks(self, records):
        out = []
        for r in records:
            mask = np.clip(r["label"], 0, NUM_CLASSES - 1).astype(np.uint8)
            if self.adapter is None:
                mask[40:80] = (mask[40:80] + 1) % NUM_CLASSES
            out.append(mask)
        return out

    def predict(self, records, batch_size=4):
        return {
            "model": {"id": "stand-in", "adapted": self.adapter is not None}, "classes": list(CLASS_NAMES),
            "decision_rule": "argmax over the 13 class scores (no threshold)",
            "predictions": [{"id": r["id"], "mask": m, "scores": np.zeros((NUM_CLASSES, 4, 4), np.float32), "class_fraction": {n: round(float((m == c).mean()), 4) for c, n in enumerate(CLASS_NAMES)}} for r, m in zip(records, self._masks(records), strict=True)],
        }

    def evaluate(self, records, batch_size=4):
        labels = [r["label"] for r in records]
        return {"n_records": len(records), "metric": "stand-in", "model": segmentation_metrics(self._masks(records), labels, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX), "baseline_majority": majority_baseline(labels, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX), "adapted": self.adapter is not None}

    def restore_base(self):
        self.adapter = None
        return []

    def adapt(self, train, val, *, epochs, lr, batch_size, trainable, progress=None):
        frozen_val = self.evaluate(val)["model"]
        self.adapter = {"trainable": trainable, "best_epoch": 1}
        adapted_val = self.evaluate(val)["model"]
        history = [{"epoch": 0, "train_loss": None, "val_loss": 1.0, "val": frozen_val, "note": "frozen model"}, {"epoch": 1, "train_loss": 0.9, "val_loss": 0.9, "val": adapted_val}]
        for entry in history:
            if progress:
                progress(entry)
        self.adapter.update({"n_trainable": 5, "n_total": 100, "n_steps": 9, "loss": "ce", "precision": "fp32", "batchnorm": "frozen", "history": history, "trainable_names": ["decode_head.weight"], "started_from": "pinned base"})
        return dict(self.adapter)

    def save_artifact(self, output_dir, metadata=None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter.safetensors").write_bytes(b"stand-in")
        (out / "manifest.json").write_text(json.dumps({"format": "stand-in", "adapter": {"trainable": self.adapter["trainable"], "best_epoch": 1}, "tensors": ["decode_head.bias", "decode_head.weight"], "files": [{"path": "adapter.safetensors", "bytes": 8, "sha256": "0" * 64}]}), encoding="utf-8")
        return out

    @classmethod
    def from_artifact(cls, artifact_dir, *, weights_dir, device):
        return cls(adapted={"best_epoch": 1})


def _tutorial_namespace(tmp_path, monkeypatch, displayed):
    namespace = {k: v for mod in (pl, mt, sm) for k, v in vars(mod).items() if not k.startswith("_")}
    namespace.update({
        "Path": Path, "np": np, "json": json, "display": lambda obj: displayed.append(obj), "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), __version__="stand-in"),
        "PrithviCropPipeline": _StandInPipe, "pipe": _StandInPipe(), "WEIGHTS_DIR": tmp_path / "weights", "verify_converted": lambda p: {"files": []},
        "MANIFEST": {"files": []}, "NOTEBOOK_SOURCE": {"repository_revision": "stand-in"}, "SAMPLE_LABEL_SOURCE": "stand-in",
    })
    monkeypatch.setitem(__import__("sys").modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    return namespace


def test_cr_m2_byod_records_without_source_id_run_sections_4_to_8_and_export(notebook, tmp_path, monkeypatch, capsys):
    displayed: list = []
    namespace = _tutorial_namespace(tmp_path, monkeypatch, displayed)
    byod_zip = _write_zip(tmp_path / "byod7.zip", synthetic_records(7, seed=300))
    for marker in ("if USE_BYOD:", "frozen_test = pipe.evaluate(test_records)", "adapt_result = pipe.adapt(", "adapted_test = pipe.evaluate(test_records)", "shown_predictions = pipe.predict(shown_records)"):
        source = _cell(notebook, marker).replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", f"BYOD_PATH = {str(byod_zip)!r}")
        exec(compile(source, f"<{marker}>", "exec"), namespace)
    out = capsys.readouterr().out
    assert "'source_id'" not in out and namespace["data_source"] == "BYOD (byod7.zip)"
    assert "'byod_chips': 7, 'minimum_chips': 7, 'grouped_split': False" in out
    assert "'splits': {'train': 4, 'validation': 1, 'test': 2}" in out
    # CR-m1: the padded probes fail on their intended check even with 2 test chips
    assert "'probe': 'two-date chip', 'rejected': \"records[0]: image must have shape" in out or "'probe': 'two-date chip', 'rejected': 'records[0]: image must have shape" in out
    assert "'unknown label class', 'rejected': 'records[0]: label values [13]" in out
    assert "4..2000 are required" not in out
    # CR-M2: Section 5 and 8 name chips by id, the artifact, predictions.json and result.json are written
    assert "'chip': 'chip-" in out and "'block': '—'" in out
    outputs = tmp_path / "outputs"
    assert (outputs / "prithvi_crop_classification_adapter" / "manifest.json").is_file()
    assert (outputs / "prithvi_crop_classification_predictions.json").is_file()
    result = json.loads((outputs / "prithvi_crop_classification_result.json").read_text(encoding="utf-8"))
    assert result["data_source"] == "BYOD (byod7.zip)" and result["reload_parity"]["mean_iou_diff"] == 0.0 and result["adaptation_gpu_peak_gb"] is None
    assert sorted(p.name for p in outputs.glob("*_map_*.tif")) == sorted(f"prithvi_crop_classification_map_{kind}_{r['id']}.tif" for kind in ("adapted", "reference") for r in namespace["test_records"][:2])
    # CR-m4: two strips displayed inline and written as PNG; CR-S5: per-chip rows and ranges printed and reported
    assert len(displayed) == 2 and all(isinstance(d, PngImage) for d in displayed)
    assert sorted(p.name for p in outputs.glob("*_maps_*.png")) == sorted(f"prithvi_crop_classification_maps_{r['id']}.png" for r in namespace["test_records"][:2])
    assert _decode_png(displayed[0].data).shape == (224, 4 * 224 + 18, 3)
    assert "'per_chip_mean_iou_frozen'" in out and "'per_chip_mean_iou_range'" in out and "'legend'" in out
    report = json.loads((outputs / "prithvi_crop_classification_evaluation_report.json").read_text(encoding="utf-8"))
    assert len(report["per_chip"]["frozen"]) == 2 and report["per_chip"]["adapted"][0]["mean_iou"] == 1.0
    assert result["comparison"]["per_chip_mean_iou_range"]["adapted"] == [1.0, 1.0]


# --- CR-m2 / CR-M4: interpretation, experiments and the GDL10 activity --------------------------------------------------


def test_cr_m2_interpretation_and_experiments(notebook):
    md = _markdown(notebook)
    assert "leaves the mean IoU where it was" not in md
    assert "Natural Vegetation, the majority class, lost a quarter of its IoU (0.2494 → 0.1884)" in md and "0.3229 → 0.3784" in md
    assert "to see the frozen model win every epoch" not in md
    assert "## Optional experiment: Predict → Change → Run → Observe → Explain" in md
    assert "run Sections 6, 7 and 8 in that order" in md and "There is no recorded outcome for this setting" in md
    assert "{{id, image, label}}" not in md and "`{id, image, label}`" in md


# --- CR-m5: decision-rule ownership -------------------------------------------------------------------------------------


def test_cr_m5_section_5_states_the_default_rule_and_the_deployment_owns_the_threshold(notebook):
    md = _markdown(notebook)
    section_5 = md[md.index("## 5. The frozen model") : md.index("## 6. Bounded fine-tuning")]
    assert "The decision rule is the argmax over the 13 class scores, and it is a default, not a tuned operating point" in section_5
    assert "is the deployment's to set on its own validation data" in section_5 and "owns the calibration" in section_5


# --- CR-S6: GPU peak -------------------------------------------------------------------------------------------------


def test_cr_s6_adapt_cell_records_gpu_peak(notebook):
    assert "adapt_result['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None" in _cell(notebook, "adapt_result = pipe.adapt(")
    assert "'adaptation_gpu_peak_gb': adapt_result['gpu_peak_gb']" in _cell(notebook, "result_payload = {")
