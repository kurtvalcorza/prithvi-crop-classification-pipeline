"""Offline tests for the pinned multi-temporal crop sample (tarball + member pins), role assignment by block, BYOD
loaders and sample export."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile

import numpy as np
import pytest

from conftest import synthetic_chip, synthetic_records
from prithvi_crop_classification_pipeline import (
    IMAGE_SIZE,
    SAMPLE_RECORDS,
    TAR_SHA256,
    check_split_disjoint,
    chip_block,
    dataset_manifest,
    extract_pinned_members,
    fetch_corpus,
    fetch_sample_dataset,
    fetch_tarball,
    load_byod_dataset,
    read_corpus,
    split_dataset,
    write_dataset_csv,
    write_sample_pair,
)
from prithvi_crop_classification_pipeline import samples as sm


def test_pinned_records_are_consistent(forbid_model_imports):
    assert len(SAMPLE_RECORDS) == 60 and len(TAR_SHA256) == 64 and sm.TAR_BYTES > 1_000_000_000
    roles: dict[str, int] = {}
    blocks: dict[str, str] = {}
    for name, role, image, image_bytes, image_sha, label, label_bytes, label_sha in SAMPLE_RECORDS:
        roles[role] = roles.get(role, 0) + 1
        assert role in sm.ROLES and name in image and name in label and name.startswith("chip_")
        assert image.endswith("_merged.tif") and label.endswith(".mask.tif") and not image.startswith("/") and ".." not in image
        assert "/._" not in image and "/._" not in label  # the archive's AppleDouble twins are never pinned
        assert image_bytes == 1_808_174 and 0 < label_bytes < image_bytes and len(image_sha) == 64 and len(label_sha) == 64
        block = chip_block(name)
        assert blocks.setdefault(block, role) == role, f"block {block} straddles roles"
    assert roles == {"train": 36, "validation": 12, "test": 12}
    assert len({r[0] for r in SAMPLE_RECORDS}) == 60 and len({r[4] for r in SAMPLE_RECORDS}) == 60
    assert sum(r[3] + r[6] for r in SAMPLE_RECORDS) == sm.CORPUS_BYTES
    assert sm.CORPUS_BASE_URL.startswith("https://huggingface.co/datasets/") and sm.DATASET_REVISION in sm.CORPUS_BASE_URL
    assert chip_block("chip_143_480") == "r35c120" and chip_block("chip_002_017") == "r00c04"


def _tiff_bytes(array, **kwargs) -> bytes:
    import tifffile

    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array, photometric="minisblack", **kwargs)
    return buffer.getvalue()


def _fake_tarball(monkeypatch, n_per_role=(4, 2, 2)):
    """Replace the record table and the tarball pins with a small synthetic gzipped tarball (plus a decoy member
    and an AppleDouble twin)."""
    records, members = [], {}
    index = 0
    for role, count in zip(sm.ROLES, n_per_role, strict=True):
        for _ in range(count):
            image, label = synthetic_chip(seed=index)
            flat = image.reshape(18, IMAGE_SIZE, IMAGE_SIZE).astype(np.int16)
            image_bytes = _tiff_bytes(np.moveaxis(flat, 0, -1), planarconfig="contig")  # the dataset layout: pixel-interleaved
            label_bytes = _tiff_bytes(np.where(label == -1, 0, label + 1).astype(np.uint8))
            key = f"chip_{index * 8:03d}_{100 + index * 8:03d}"  # one block per chip
            image_member = f"validation_chips/{key}_merged.tif"
            label_member = f"validation_chips/{key}.mask.tif"
            members[image_member] = image_bytes
            members[label_member] = label_bytes
            image_sha, label_sha = hashlib.sha256(image_bytes).hexdigest(), hashlib.sha256(label_bytes).hexdigest()
            records.append((key, role, image_member, len(image_bytes), image_sha, label_member, len(label_bytes), label_sha))
            index += 1
    members["validation_chips/decoy.txt"] = b"not pinned"
    members["validation_chips/._chip_000_100_merged.tif"] = b"\x00\x05\x16\x07 apple double"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_bytes = buffer.getvalue()
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(records))
    monkeypatch.setattr(sm, "TAR_BYTES", len(tar_bytes))
    monkeypatch.setattr(sm, "TAR_SHA256", hashlib.sha256(tar_bytes).hexdigest())
    return tar_bytes


def test_fetch_tarball_verifies_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    tar_bytes = _fake_tarball(monkeypatch)
    calls = []

    def fetcher(url):
        calls.append(url)
        return tar_bytes

    path = fetch_tarball(cache_dir=tmp_path, fetcher=fetcher)
    assert path.name == sm.TAR_NAME and len(calls) == 1 and calls[0].endswith(sm.TAR_NAME)
    fetch_tarball(cache_dir=tmp_path, fetcher=fetcher)
    assert len(calls) == 1  # cached
    with pytest.raises(ValueError, match="pinned"):
        fetch_tarball(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_extract_pinned_members_only(tmp_path, monkeypatch, forbid_model_imports):
    tar_bytes = _fake_tarball(monkeypatch)
    tar_path = tmp_path / "fake.tgz"
    tar_path.write_bytes(tar_bytes)
    out = extract_pinned_members(tar_path, cache_dir=tmp_path)
    assert len(out) == 16 and not (tmp_path / "chips" / "decoy.txt").exists()
    assert not (tmp_path / "chips" / "._chip_000_100_merged.tif").exists()
    assert sorted(p.name for p in (tmp_path / "chips").iterdir()) == sorted(p.rsplit("/", 1)[-1] for p in out)
    extra = (
        "chip_999_999", "train", "validation_chips/chip_999_999_merged.tif", 1, "0" * 64,
        "validation_chips/chip_999_999.mask.tif", 1, "0" * 64,
    )
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", (*sm.SAMPLE_RECORDS, extra))
    with pytest.raises(ValueError, match="does not contain"):
        extract_pinned_members(tar_path, cache_dir=tmp_path / "again")


def test_sample_dataset_roles_and_manifest(tmp_path, monkeypatch, forbid_model_imports):
    tar_bytes = _fake_tarball(monkeypatch)
    splits = fetch_sample_dataset(cache_dir=tmp_path, fetcher=lambda url: tar_bytes)
    assert {k: len(v) for k, v in splits.items()} == {"train": 4, "validation": 2, "test": 2}
    assert splits["test"][0]["id"] == "test-000" and splits["test"][0]["image"].shape == (3, 6, IMAGE_SIZE, IMAGE_SIZE)
    assert 0.0 <= float(splits["test"][0]["image"].max()) <= 10_000.0
    assert set(np.unique(splits["test"][0]["label"])) <= set(range(-1, 13))
    manifest = dataset_manifest(splits)
    assert manifest["disjoint"] == {"train": 4, "validation": 2, "test": 2} and len(manifest["digest"]) == 64
    assert manifest["splits"]["train"]["regions"] == ["r00c25", "r02c27", "r04c29", "r06c31"]
    assert manifest["splits"]["train"]["classes_present"] == 13
    files = fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: (_ for _ in ()).throw(AssertionError(url)))
    assert len(files) == 8  # served from the extracted cache, nothing fetched
    assert read_corpus(files)["train"][0]["source"].startswith(sm.CORPUS_BASE_URL)


def test_check_split_disjoint_detects_leakage(forbid_model_imports):
    records = synthetic_records(4)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:2], "test": [records[0]]})
    blocked = [{**r, "region": "r01c01"} for r in records]
    with pytest.raises(ValueError, match="block 'r01c01' has chips in both"):
        check_split_disjoint({"train": blocked[:2], "test": blocked[2:]})


def test_split_dataset(forbid_model_imports):
    records = synthetic_records(12)
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 12 and len(splits["train"]) >= 4 and splits["test"]
    check_split_disjoint(splits)
    assert sum(len(v) for v in split_dataset(records + [{**records[0], "id": "dup"}]).values()) == 12
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, test_fraction=0.9)


def test_byod_directory_and_zip_loaders(tmp_path, forbid_model_imports):
    records = synthetic_records(5)
    folder = tmp_path / "byod"
    folder.mkdir()
    for record in records:
        write_sample_pair(record, folder / f"{record['id']}_merged.tif", folder / f"{record['id']}.mask.tif")
    write_dataset_csv(records, folder / "pairs.csv")
    loaded = load_byod_dataset(folder)
    assert [r["id"] for r in loaded] == [r["id"] for r in records]
    assert np.allclose(loaded[0]["image"], np.rint(records[0]["image"]), atol=0.5)
    assert np.array_equal(loaded[0]["label"], records[0]["label"])
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"nested/{path.name}")
    assert len(load_byod_dataset(archive)) == 5
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "no table")
    with pytest.raises(ValueError, match="pairs.csv"):
        load_byod_dataset(bad)
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "missing.tar")
    (folder / "pairs.csv").write_text("id,image\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(folder)
