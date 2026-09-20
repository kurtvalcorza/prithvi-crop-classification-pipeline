"""Model-backed smoke on the real converted weights: skipped unless the digest-verified safetensors file is
staged (`weights/prithvi-eo-1.0-100m-crop/`). Loads strictly, classifies a synthetic chip, runs one adaptation
step on the head and checks reload parity — evidence that the vendored architecture and the converted file
agree, not a quality claim."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import synthetic_records  # noqa: E402
from prithvi_crop_classification_pipeline import DEFAULT_WEIGHTS_DIR, PARAMETER_COUNT, PrithviCropPipeline  # noqa: E402
from prithvi_crop_classification_pipeline import pipeline as pl  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DEFAULT_WEIGHTS_DIR / pl.CONVERTED_WEIGHTS_NAME).is_file(), reason="converted weights are not staged"
)


def test_converted_weights_load_classify_adapt_and_reload(tmp_path):
    pipe = PrithviCropPipeline.from_pretrained(require_source=False)
    assert sum(p.numel() for p in pipe.model.parameters()) == PARAMETER_COUNT
    records = synthetic_records(4)
    result = pipe.predict(records[:2], batch_size=2)
    masks = [p["mask"] for p in result["predictions"]]
    assert masks[0].shape == (224, 224) and masks[0].dtype == np.uint8 and int(masks[0].max()) < 13
    scores = result["predictions"][0]["scores"]
    assert np.allclose(scores.sum(axis=0), 1.0, atol=1e-3)
    frozen = pipe.evaluate(records[2:])["model"]
    adapt = pipe.adapt(records, records[2:], epochs=1, lr=1e-4, batch_size=2, trainable="head")
    assert adapt["n_trainable"] == 5_312_269 and adapt["n_steps"] == 2 and adapt["history"][0]["val"] == frozen
    out = pipe.save_artifact(tmp_path / "adapter")
    again = PrithviCropPipeline.from_artifact(out, require_source=False)
    assert again.evaluate(records[2:])["model"] == pipe.evaluate(records[2:])["model"]
