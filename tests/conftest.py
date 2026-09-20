import builtins

import numpy as np
import pytest

MODEL_LIBRARIES = {"torch", "safetensors", "huggingface_hub", "mmcv", "mmseg", "timm", "einops"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_chip(*, seed: int = 0, size: int = 224, dates: int = 3, bands: int = 6, classes: int = 13):
    """A smooth three-date six-band chip in HLS digital numbers (0..10 000) whose label is a coarse mosaic of
    class patches driven by the same field; the patches shift between dates so the three frames differ."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    field = np.sin(4 * x + seed * 0.3) * np.cos(3 * y) + 0.5 * np.sin(9 * x * y)
    label = (np.floor(x * classes) + np.floor(y * 5) + seed).astype(np.int64) % classes  # diagonal stripes: every class present
    stack = np.zeros((dates, bands, size, size), dtype=np.float32)
    for t in range(dates):
        for b in range(bands):
            base = 400 + 450 * b + 300 * t + 1200 * (field + 1.6) / 3.2 + 60 * label
            stack[t, b] = base + rng.normal(0, 20, (size, size))
    image = np.clip(stack, 0, 10_000).astype(np.float32)
    label[:8, :8] = -1  # a no-data corner
    return image, label


def synthetic_records(n: int = 6, *, seed: int = 0, labels: bool = True):
    out = []
    for i in range(n):
        image, label = synthetic_chip(seed=seed + i)
        record = {"id": f"chip-{i:03d}", "image": image}
        if labels:
            record["label"] = label
        out.append(record)
    return out
