"""Offline tests for the snapshot manifest, staging, the static pickle audit, the chip contract, the metrics and
the artifact-manifest rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_chip, synthetic_records
from prithvi_crop_classification_pipeline import (
    BANDS,
    CKPT_ALLOWED_GLOBALS,
    CLASS_NAMES,
    CLASS_WEIGHTS,
    IMAGE_SIZE,
    INPUT_SCHEMA,
    META_GLOBALS,
    MODEL_ID,
    MODEL_REVISION,
    NUM_CLASSES,
    PrithviCropPipeline,
    audit_pickle,
    dataset_digest,
    majority_baseline,
    read_chip,
    read_mask,
    segmentation_metrics,
    stage_missing_files,
    validate_dataset,
    validate_inputs,
    verify_converted,
    verify_snapshot,
)
from prithvi_crop_classification_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes], *, pin_source: bool = True) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if pin_source and pl.SOURCE_CKPT_NAME not in files:
        entries.append({"path": pl.SOURCE_CKPT_NAME, "bytes": pl.SOURCE_CKPT_BYTES, "sha256": pl.SOURCE_CKPT_SHA256})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": pl.MODEL_KEY,
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and the committed manifest ------------------------------------------------------------------


def test_identity_is_immutable_and_the_manifest_agrees():
    assert len(MODEL_REVISION) == 40
    manifest = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, pl.MODEL_KEY)
    assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    source = [e for e in manifest["files"] if e["path"] == pl.SOURCE_CKPT_NAME]
    assert source and (source[0]["bytes"], source[0]["sha256"]) == (pl.SOURCE_CKPT_BYTES, pl.SOURCE_CKPT_SHA256)
    assert len(pl.CONVERTED_SHA256) == 64 and pl.CONVERTED_BYTES > 0 and len(pl.PICKLE_AUDIT_SHA256) == 64
    assert pl.PARAMETER_COUNT > 100_000_000 and pl.STATE_TENSORS == 98 and pl.SOURCE_STATE_TENSORS == 112
    assert len(CLASS_NAMES) == NUM_CLASSES == len(CLASS_WEIGHTS) == 13


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x", pl.SOURCE_CKPT_NAME: b"ckpt"}, pin_source=False)
    with pytest.raises(ValueError, match="disagrees with the package constant"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).write_bytes(b"ckpt-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x"}, pin_source=False)
    with pytest.raises(ValueError, match="does not list"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_converted_checks_the_pinned_digest(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        verify_converted(tmp_path)
    (tmp_path / pl.CONVERTED_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="size|sha256"):
        verify_converted(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"fetched")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [pl.SOURCE_CKPT_NAME]
    assert calls == [pl.SOURCE_CKPT_NAME]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=downloader)


# --- static pickle audit ----------------------------------------------------------------------------------


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def _torch_like_archive(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


def test_audit_pickle_lists_globals_and_refuses_code(tmp_path, forbid_model_imports):
    benign = tmp_path / "benign.pth"
    benign.write_bytes(_torch_like_archive(pickle.dumps({"state_dict": {"a": 1}, "meta": {"epoch": 3}})))
    report = audit_pickle(benign)
    assert report["torch_archive"] and report["pickles"] == 1 and report["globals"] == [] and report["violations"] == []
    assert report["meta_globals"] == []
    evil = tmp_path / "evil.pth"
    evil.write_bytes(_torch_like_archive(pickle.dumps({"state_dict": _Evil()})))
    with pytest.raises(ValueError, match="pickle audit failed.*os.system|posix.system|nt.system"):
        audit_pickle(evil)
    plain = tmp_path / "plain.pickle"
    plain.write_bytes(pickle.dumps(complex(1, 2)))
    with pytest.raises(ValueError, match="pickle audit failed"):
        audit_pickle(plain)
    assert "collections.OrderedDict" in CKPT_ALLOWED_GLOBALS and len(CKPT_ALLOWED_GLOBALS) == 7
    assert {"numpy.core.multiarray.scalar", "numpy.dtype", "_codecs.encode"} == META_GLOBALS
    assert META_GLOBALS < CKPT_ALLOWED_GLOBALS
    with pytest.raises(FileNotFoundError):
        audit_pickle(tmp_path / "missing.pth")


def test_audit_pickle_reports_the_meta_globals_of_an_mmseg_checkpoint(tmp_path, forbid_model_imports):
    """A pickle stream that references what the upstream file references (the numpy scalar globals of `meta`)
    passes the audit and lists the three data-only names under `meta_globals`; with the fleet's torch-only
    allow-list the same stream is refused, naming them."""
    stream = b"".join(
        [
            pickle.PROTO + b"\x02",
            pickle.GLOBAL + b"numpy.core.multiarray\nscalar\n",
            pickle.GLOBAL + b"numpy\ndtype\n",
            pickle.GLOBAL + b"_codecs\nencode\n",
            pickle.GLOBAL + b"collections\nOrderedDict\n",
            pickle.STOP,
        ]
    )
    source = tmp_path / "mmseg.pth"
    source.write_bytes(_torch_like_archive(stream))
    report = audit_pickle(source)
    assert report["meta_globals"] == ["_codecs.encode", "numpy.core.multiarray.scalar", "numpy.dtype"]
    assert report["violations"] == [] and set(report["globals"]) == META_GLOBALS | {"collections.OrderedDict"}
    with pytest.raises(ValueError, match="_codecs.encode.*numpy.core.multiarray.scalar.*numpy.dtype"):
        audit_pickle(source, allowed=pl.TORCH_GLOBALS)


# --- chip contract ----------------------------------------------------------------------------------------


def test_validate_dataset_reports_balance_and_digest(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["n_labelled"] == 6 and report["bands"] == list(BANDS) and report["dates"] == 3
    assert report["classes_present"] == 13 and report["ignored_pixels"] == 6 * 64
    assert abs(sum(report["class_pixel_fraction"].values()) - 1.0) < 1e-2
    assert report["digest"] == dataset_digest(records) and len(report["digest"]) == 64
    assert INPUT_SCHEMA["image_size"] == IMAGE_SIZE == 224 and INPUT_SCHEMA["classes"]["12"] == "Other"
    assert report["value_range"][0] >= 0.0 and report["value_range"][1] <= 10_000.0


def test_validate_dataset_refusals_name_the_rule(forbid_model_imports):
    records = synthetic_records(6)
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"})
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="missing 'image'"):
        validate_dataset([{"id": "a", "label": records[0]["label"]}, *records[1:]])
    with pytest.raises(ValueError, match="must have shape \\(3, 6, 224, 224\\)"):
        validate_dataset([{**records[0], "image": records[0]["image"][:, :5]}, *records[1:]])
    with pytest.raises(ValueError, match="must have shape \\(3, 6, 224, 224\\)"):
        validate_dataset([{**records[0], "image": records[0]["image"][:2]}, *records[1:]])
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset([{**records[0], "image": records[0]["image"] * np.nan}, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records])
    with pytest.raises(ValueError, match="has no label"):
        validate_dataset([{"id": "u", "image": records[0]["image"]}, *records[1:]])
    with pytest.raises(ValueError, match="label values \\[13\\] outside"):
        validate_dataset([{**records[0], "label": records[0]["label"] + 1}, *records[1:]])
    with pytest.raises(ValueError, match="label must have shape"):
        validate_dataset([{**records[0], "label": records[0]["label"][:100]}, *records[1:]])
    with pytest.raises(ValueError, match="fewer than two classes"):
        validate_dataset([{**r, "label": np.zeros_like(r["label"])} for r in records])
    assert validate_dataset([{"id": r["id"], "image": r["image"]} for r in records], require_labels=False)["n_labelled"] == 0


def test_reflectance_scaling_and_range(forbid_model_imports):
    image, label = synthetic_chip()
    reflectance = (image / 10_000.0).astype(np.float32)
    report = validate_inputs({"id": "s", "image": reflectance, "label": label})
    assert report["value_range"][1] > 100.0  # scaled back to digital numbers
    assert report["label_fraction"]["Corn"] >= 0 and report["ignored_pixels"] == 64
    with pytest.raises(ValueError, match="plausible range"):
        validate_inputs({"id": "s", "image": image * 50.0})
    with pytest.raises(ValueError, match="plausible range"):
        validate_inputs({"id": "s", "image": image - 5_000.0})


def test_geotiff_round_trip(tmp_path, forbid_model_imports):
    import tifffile

    image, label = synthetic_chip()
    flat = image.reshape(18, IMAGE_SIZE, IMAGE_SIZE).astype(np.int16)
    tifffile.imwrite(tmp_path / "chip.tif", np.moveaxis(flat, 0, -1), photometric="minisblack", planarconfig="contig")
    tifffile.imwrite(tmp_path / "planar.tif", flat, photometric="minisblack", planarconfig="separate")
    raw = np.where(label == -1, 0, label + 1).astype(np.uint8)
    tifffile.imwrite(tmp_path / "label.tif", raw)
    chip = read_chip(tmp_path / "chip.tif")
    assert chip.shape == (3, 6, IMAGE_SIZE, IMAGE_SIZE) and chip.dtype == np.float32
    assert np.array_equal(chip, flat.reshape(3, 6, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32))
    assert np.array_equal(read_chip(tmp_path / "planar.tif"), chip)
    mask = read_mask(tmp_path / "label.tif")
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE) and mask.dtype == np.int64 and np.array_equal(mask, label)
    assert validate_inputs({"id": "p", "image": str(tmp_path / "chip.tif"), "label": str(tmp_path / "label.tif")})["has_label"]
    tifffile.imwrite(tmp_path / "six.tif", flat[:6], photometric="minisblack", planarconfig="separate")
    with pytest.raises(ValueError, match="18-band"):
        read_chip(tmp_path / "six.tif")
    with pytest.raises(ValueError, match="image file not found"):
        validate_inputs({"id": "p", "image": str(tmp_path / "none.tif")})


# --- metrics ----------------------------------------------------------------------------------------------


def test_segmentation_metrics_and_baseline(forbid_model_imports):
    _, label = synthetic_chip()
    perfect = segmentation_metrics([np.where(label < 0, 0, label)], [label], class_names=CLASS_NAMES, ignore_index=-1)
    assert perfect["accuracy"] == 1.0 and perfect["mean_iou"] == 1.0 and perfect["mean_f1"] == 1.0
    assert perfect["pixels"] == IMAGE_SIZE * IMAGE_SIZE - 64 and perfect["classes_scored"] == 13
    baseline = majority_baseline([label], class_names=CLASS_NAMES, ignore_index=-1)
    assert baseline["majority_class"] in CLASS_NAMES and baseline["accuracy"] == max(baseline["label_fraction"].values())
    assert baseline["mean_iou"] < 0.2 and baseline["iou"][baseline["majority_class"]] == baseline["accuracy"]
    shifted = segmentation_metrics([np.where(label < 0, 0, (label + 1) % 13)], [label], class_names=CLASS_NAMES, ignore_index=-1)
    assert shifted["accuracy"] == 0.0 and shifted["mean_iou"] == 0.0
    with pytest.raises(ValueError, match="argument 2 is shorter|shape"):
        segmentation_metrics([label], [label[:10]], class_names=CLASS_NAMES, ignore_index=-1)
    partial = segmentation_metrics([np.zeros_like(label)], [np.where(label < 0, -1, 0)], class_names=CLASS_NAMES, ignore_index=-1)
    assert partial["classes_scored"] == 1 and partial["iou"]["Forest"] is None and partial["mean_iou"] == 1.0


# --- artifact manifest (static checks, no weights) -------------------------------------------------------


def _good_manifest(root: Path) -> dict:
    (root / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    return {
        "format": pl.ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": pl.MODEL_KEY, "converted_sha256": pl.CONVERTED_SHA256},
        "adapter": {"trainable": "head"},
        "tensors": ["decode_head.x"],
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
    }


def test_artifact_manifest_static_checks(tmp_path, forbid_model_imports):
    good = _good_manifest(tmp_path)
    path, mode = PrithviCropPipeline.check_artifact_manifest(tmp_path, good)
    assert path == (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).resolve() and mode == "head"
    cases = {
        "format": ({**good, "format": "other"}, "artifact format"),
        "version": ({**good, "format_version": "2.0"}, "format_version"),
        "base": ({**good, "base_model": {**good["base_model"], "revision": "0" * 40}}, "different base model"),
        "digest": ({**good, "base_model": {**good["base_model"], "converted_sha256": "0" * 64}}, "converted-base digest"),
        "two files": ({**good, "files": good["files"] * 2}, "exactly one weights file"),
        "other name": ({**good, "files": [{**good["files"][0], "path": "weights.safetensors"}]}, "must be named"),
        "mode": ({**good, "adapter": {"trainable": "everything"}}, "adapter.trainable"),
        "tensors": ({**good, "tensors": "all"}, "list its tensors"),
    }
    for name, (manifest, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            PrithviCropPipeline.check_artifact_manifest(tmp_path, manifest)
        del name
