"""Offline tests for the public validation-stage helpers and the package surface."""

from __future__ import annotations

import prithvi_crop_classification_pipeline as pkg
from conftest import synthetic_records
from prithvi_crop_classification_pipeline import BANDS, IGNORE_INDEX, IMAGE_SIZE, INPUT_SCHEMA, NUM_FRAMES, validate_inputs


def test_input_schema_names_the_contract():
    assert INPUT_SCHEMA["bands"] == list(BANDS) and len(BANDS) == 6 and INPUT_SCHEMA["dates"] == NUM_FRAMES == 3
    assert INPUT_SCHEMA["image_size"] == IMAGE_SIZE == 224 and INPUT_SCHEMA["ignore_index"] == IGNORE_INDEX == -1
    assert "any (3, 6, 224, 224) array is classified" in INPUT_SCHEMA["validation"]
    assert len(INPUT_SCHEMA["classes"]) == 13 and INPUT_SCHEMA["classes"]["0"] == "Natural Vegetation"


def test_validate_inputs_reports_the_record():
    report = validate_inputs(synthetic_records(1)[0])
    assert report["id"] == "chip-000" and report["shape"] == (3, 6, 224, 224) and report["has_label"]
    assert report["ignored_pixels"] == 64 and abs(sum(report["label_fraction"].values()) - 1.0) < 1e-2


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
    assert "PrithviCropPipeline" in pkg.__all__ and "audit_pickle" in pkg.__all__ and "restricted_load" in pkg.__all__
