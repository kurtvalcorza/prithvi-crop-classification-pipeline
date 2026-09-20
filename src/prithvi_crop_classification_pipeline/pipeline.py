"""Prithvi-EO-1.0-100M multi-temporal crop classification
(`ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification`) DIMER pipeline: verified snapshot,
one-time conversion of the pickled mmsegmentation checkpoint into safetensors, 13-class crop / land-cover
segmentation of three-date six-band HLS chips, held-out evaluation against a majority-class baseline, and bounded
fine-tuning of the segmentation head to a user's labelled chips with a portable adapter.

Prithvi-EO-1.0-100M (Jakubik et al., 2023) is a ViT-B masked-autoencoder foundation model for Harmonized Landsat
Sentinel-2 imagery, pretrained on three-date stacks. The checkpoint packaged here is the upstream authors'
fine-tune for multi-temporal crop classification: the first six encoder blocks, a transposed-convolution neck and
an FCN head over 13 USDA Cropland Data Layer classes, trained on 224 × 224 chips of three HLS dates (six bands
each) over the contiguous United States in 2022.

The upstream asset is an mmsegmentation checkpoint — a torch zip archive whose pickle holds `meta`, `state_dict`
and the Adam `optimizer` state, and references, besides `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`
and two storage classes, three data-only globals that rebuild the numpy scalar `meta.hook_msgs.best_score`
(`numpy.core.multiarray.scalar`, `numpy.dtype`, `_codecs.encode`). `audit_pickle` verifies statically that nothing
else is referenced; the conversion then unpickles the file **once** with torch's weights-only unpickler, in which
those three names are bound to inert stand-ins (no numpy code runs and the value is discarded), keeps the 98
inference tensors (the training-only auxiliary head and the optimizer are dropped) after a strict load into the
vendored architecture (`modeling.py`, plain PyTorch, no mmcv / mmseg / timm), and writes safetensors with a
pinned digest — the only file the model is ever loaded from.

Everything model-related is imported lazily so that snapshot verification, the pickle audit and input validation
run (and can refuse) before `torch` is imported (fleet RTM-001). `numpy` and `tifffile` are used for chips and are
imported freely.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickletools
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification"
MODEL_REVISION = "b53a88b8da673800b67c34a98a527b77076e7035"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "prithvi-eo-1.0-100m-crop"
ARTIFACT_FORMAT = "org.valcorza.prithvi-crop-classification.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Immutable upstream source asset (an mmsegmentation checkpoint, i.e. a pickle; see docs/WEIGHTS.md).
SOURCE_CKPT_NAME = "multi_temporal_crop_classification_Prithvi_100M.pth"
SOURCE_CKPT_BYTES = 1_680_468_041
SOURCE_CKPT_SHA256 = "37ed41637eccccec65ca2031324e2c03a4f168e1ea0ea71ad180910589fa018c"
# Code-free serving file produced deterministically by `convert_model` (asset spec §11.2).
CONVERTED_WEIGHTS_NAME = "prithvi-eo-1.0-100m-crop.safetensors"
CONVERTED_SHA256 = "d1df8044700a0d1e00b11b1fbac66e0495cf6647e1632a858c003eaf4b5ce36d"
CONVERTED_BYTES = 537_722_508
# Static-audit digest of the source pickle (sorted global names), see `audit_pickle`.
PICKLE_AUDIT_SHA256 = "465513635354b8c8a9c4bfd4a37b39d5306eb8ec63527905ce0195075d04108d"
TORCH_GLOBALS = frozenset(
    {"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage", "torch.LongStorage"}
)
# The three data-only globals that rebuild the numpy scalar `meta.hook_msgs.best_score` (the best validation
# mIoU mmseg logged). They construct values, not code, and the conversion binds them to inert stand-ins.
META_GLOBALS = frozenset({"numpy.core.multiarray.scalar", "numpy.dtype", "_codecs.encode"})
CKPT_ALLOWED_GLOBALS = TORCH_GLOBALS | META_GLOBALS
TRAINING_ONLY_PREFIX = "auxiliary_head."  # mmseg's auxiliary FCN head: trained with, never used at inference

# Architecture (the pinned multi_temporal_crop_classification_Prithvi_100M.py config) and data-contract facts.
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_FRAMES = 3
EMBED_DIM = 768
DEPTH = 6
NUM_HEADS = 8
HEAD_CHANNELS = 256
SOURCE_STATE_TENSORS = 112  # 78 backbone + 12 neck + 8 decode_head + 14 auxiliary_head
STATE_TENSORS = 98
STATE_NUMEL = 134_428_174
PARAMETER_COUNT = 134_427_661  # nn.Parameters (the fixed pos_embed included); the rest are BatchNorm buffers
CLASS_NAMES: tuple[str, ...] = (
    "Natural Vegetation",
    "Forest",
    "Corn",
    "Soybeans",
    "Wetlands",
    "Developed/Barren",
    "Open Water",
    "Winter Wheat",
    "Alfalfa",
    "Fallow/Idle Cropland",
    "Cotton",
    "Sorghum",
    "Other",
)
NUM_CLASSES = len(CLASS_NAMES)
# The upstream training loss weights (CrossEntropyLoss class_weight of the pinned config), reused by `adapt`.
CLASS_WEIGHTS: tuple[float, ...] = (
    0.386375,
    0.661126,
    0.548184,
    0.640482,
    0.876862,
    0.925186,
    3.249462,
    1.542289,
    2.175141,
    2.272419,
    3.062762,
    3.626097,
    1.198702,
)
IGNORE_INDEX = -1
# Label rasters of the dataset use 0 = no data and 1..13 = class; records use 0..12 and IGNORE_INDEX (mmseg's
# `reduce_zero_label`), and `read_mask` / `write_sample_pair` translate between the two.
BANDS: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
# Per-band statistics of the pinned config, in the dataset's digital-number units (HLS reflectance × 10 000),
# repeated for each of the three dates.
MEANS: tuple[float, ...] = (494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917)
STDS: tuple[float, ...] = (284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808)
REFLECTANCE_SCALE = 10_000.0  # applied only when a chip arrives as reflectance in [0, 1]
VALUE_RANGE = (-2_000.0, 20_000.0)  # plausible HLS digital numbers (clouds and snow exceed 10 000)
MIN_RECORDS = 4
MAX_RECORDS = 2_000
ADAPTATION_MODES = ("head", "head+last_block")  # the only scopes an adapter may declare
LAST_BLOCK_PREFIX = f"backbone.blocks.{DEPTH - 1}."
TRAINABLE_PREFIXES: dict[str, tuple[str, ...]] = {
    "head": ("decode_head.",),
    "head+last_block": ("decode_head.", LAST_BLOCK_PREFIX),
}


# --------------------------------------------------------------------------------------------------
# manifest, staging, static pickle audit and conversion
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    if SOURCE_CKPT_NAME not in listed:
        raise ValueError(f"manifest does not list {SOURCE_CKPT_NAME}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == SOURCE_CKPT_NAME and (size, digest) != (SOURCE_CKPT_BYTES, SOURCE_CKPT_SHA256):
            raise ValueError(f"{entry['path']}: manifest digest disagrees with the package constant")
    return manifest


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check the converted serving file (safetensors) against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    file_path = root / CONVERTED_WEIGHTS_NAME
    if not file_path.is_file():
        raise FileNotFoundError(f"converted file missing: {file_path}")
    size = file_path.stat().st_size
    if size != CONVERTED_BYTES:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: size {size} != pinned {CONVERTED_BYTES}")
    digest = _sha256_file(file_path)
    if digest != CONVERTED_SHA256:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: sha256 {digest} != pinned {CONVERTED_SHA256}")
    return {"files": [{"path": CONVERTED_WEIGHTS_NAME, "bytes": size, "sha256": digest}]}


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed Hub file) and, when the
    converted serving file is present, that against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    converted = (root / CONVERTED_WEIGHTS_NAME).is_file()
    if converted:
        verify_converted(root)
    return {**manifest, "converted": converted}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest and git-ignores the
    1.68 GB checkpoint and the safetensors it converts to)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_globals(data: bytes) -> dict[str, int]:
    """Every global a pickle stream would import, collected with `pickletools.genops` (no execution)."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
    return found


def audit_pickle(path: str | Path, *, allowed: frozenset[str] = CKPT_ALLOWED_GLOBALS) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, or inside a torch zip archive) would import and refuse any
    outside `allowed`. Executes nothing. Returns the sorted globals, which of them are the data-only meta
    globals, and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    nested = 0
    if data[:4] == b"PK\x03\x04":
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                nested += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "torch_archive": data[:4] == b"PK\x03\x04",
        "pickles": nested if nested else 1,
        "globals": sorted(found),
        "meta_globals": sorted(name for name in found if name in META_GLOBALS),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}")
    return summary


def _check_pinned_source(root: Path) -> dict[str, Any]:
    source = root / SOURCE_CKPT_NAME
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != SOURCE_CKPT_BYTES:
        raise ValueError(f"{SOURCE_CKPT_NAME}: size {size} != pinned {SOURCE_CKPT_BYTES}")
    digest = _sha256_file(source)
    if digest != SOURCE_CKPT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: sha256 {digest} != pinned {SOURCE_CKPT_SHA256}")
    audit = audit_pickle(source)
    if audit["audit_sha256"] != PICKLE_AUDIT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: pickle audit digest {audit['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256}")
    return {"path": SOURCE_CKPT_NAME, "bytes": size, "sha256": digest, "audit": audit}


class _MetaValue:
    """Inert stand-in for the numpy dtype / scalar objects mmseg stored in the checkpoint's `meta`: the
    weights-only unpickler builds these instead of importing numpy, and the conversion discards them."""

    def __init__(self, kind: str, *args: Any) -> None:
        self.kind = kind
        self.args = args

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __repr__(self) -> str:
        return f"<meta {self.kind}>"


def _meta_scalar(dtype: Any, payload: Any) -> _MetaValue:
    return _MetaValue("numpy.core.multiarray.scalar", dtype, payload)


def _meta_dtype(*args: Any) -> _MetaValue:
    return _MetaValue("numpy.dtype", *args)


def _meta_encode(text: str, encoding: str = "latin1", *_rest: Any) -> bytes:
    return text.encode(encoding)


def restricted_load(path: str | Path) -> dict[str, Any]:
    """Unpickle the checkpoint once through torch's weights-only unpickler, with the three meta globals bound to
    inert stand-ins (nothing from numpy is imported or executed by the pickle). Only the tensors and plain
    containers survive; the stand-ins are what `meta.hook_msgs.best_score` unpickles to."""
    import torch

    bindings = [
        (_meta_scalar, "numpy.core.multiarray.scalar"),
        (_meta_dtype, "numpy.dtype"),
        (_meta_encode, "_codecs.encode"),
        _MetaValue,
    ]
    source = Path(path)
    with torch.serialization.safe_globals(bindings):
        payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{source.name} did not unpickle to a checkpoint dict")
    return payload


def build_model() -> Any:
    """Instantiate the fine-tuned architecture from the vendored module (no pretrained download)."""
    from .modeling import PrithviCropSegmenter

    return PrithviCropSegmenter(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        in_chans=len(BANDS),
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        head_channels=HEAD_CHANNELS,
        num_classes=NUM_CLASSES,
    )


def convert_model(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the pinned mmsegmentation checkpoint into safetensors, deterministically, after size, digest and
    static-audit checks: the restricted weights-only unpickle, the `state_dict` entry with the training-only
    auxiliary head dropped, a strict load into the vendored architecture, and the model's own state dict saved.
    The optimizer state and `meta` are read and discarded."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    source = _check_pinned_source(root)
    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    payload = restricted_load(root / SOURCE_CKPT_NAME)
    if "state_dict" not in payload:
        raise ValueError(f"{SOURCE_CKPT_NAME} did not unpickle to an mmsegmentation checkpoint with a state_dict")
    state = payload["state_dict"]
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{SOURCE_CKPT_NAME}: state_dict is not a dict of tensors")
    if len(state) != SOURCE_STATE_TENSORS:
        raise ValueError(f"{SOURCE_CKPT_NAME}: state_dict has {len(state)} tensors, expected {SOURCE_STATE_TENSORS}")
    dropped = sorted(k for k in state if k.startswith(TRAINING_ONLY_PREFIX))
    kept = {k: v for k, v in state.items() if not k.startswith(TRAINING_ONLY_PREFIX)}
    model = build_model()
    model.load_state_dict(kept, strict=True)
    canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
    n_elements = sum(v.numel() for v in canonical.values())
    if len(canonical) != STATE_TENSORS or n_elements != STATE_NUMEL:
        raise ValueError(
            f"converted state dict has {len(canonical)} tensors / {n_elements} elements; expected {STATE_TENSORS} / {STATE_NUMEL}"
        )
    save_file(canonical, str(root / CONVERTED_WEIGHTS_NAME), metadata={"format": "pt"})
    report = verify_converted(root)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return {
        "source": {k: v for k, v in source.items() if k != "audit"},
        "audit": source["audit"],
        "checkpoint": {
            "epoch": meta.get("epoch"),
            "iter": meta.get("iter"),
            "mmseg_version": meta.get("mmseg_version"),
            "mmcv_version": meta.get("mmcv_version"),
            "entries": sorted(payload),
            "state_dict_tensors": len(state),
            "dropped_training_only_tensors": len(dropped),
            "optimizer_state_discarded": "optimizer" in payload,
        },
        "converted": report["files"],
        "seconds": round(time.perf_counter() - started, 2),
    }


# --------------------------------------------------------------------------------------------------
# chips, labels and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": (
        "{id, image, label?}: image = (3, 6, 224, 224) float32 stack of three dates × six HLS bands in digital "
        "numbers (reflectance × 10 000; or a GeoTIFF path with 18 bands, date-major); "
        "label = (224, 224) int mask with 0..12 / -1 (or a GeoTIFF path with 0 = no data, 1..13 = class), optional"
    ),
    "bands": list(BANDS),
    "dates": NUM_FRAMES,
    "image_size": IMAGE_SIZE,
    "value_units": (
        "HLS surface-reflectance digital numbers (reflectance × 10 000, int16 in the dataset) in "
        f"[{VALUE_RANGE[0]:.0f}, {VALUE_RANGE[1]:.0f}]; a chip whose values all lie in [0, 1.5] is taken as reflectance "
        "and scaled by 10 000"
    ),
    "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
    "ignore_index": IGNORE_INDEX,
    "label_files": "0 = no data (ignored), 1..13 = the classes above in order (the dataset's CDL reclassification)",
    "records": [MIN_RECORDS, MAX_RECORDS],
    "validation": (
        "record shape, date and band counts, chip size, finiteness, value range and label values only. Nothing "
        "checks that the bands are the six HLS bands in the right order, that the three dates are the growing-"
        "season dates the model was trained on, or that the label was drawn for this chip -- any (3, 6, 224, 224) "
        "array is classified without complaint"
    ),
}


def _read_tiff(path: Path) -> Any:
    """Read a GeoTIFF's pixel array with tifffile as (bands, H, W) or (H, W); no georeferencing is used."""
    import numpy as np
    import tifffile

    with tifffile.TiffFile(path) as tf:
        array = tf.asarray()
    if array.ndim == 3 and array.shape[-1] <= 32 and array.shape[0] > 32:
        array = np.moveaxis(array, -1, 0)  # pixel-interleaved (H, W, bands) -> band-sequential
    return np.asarray(array)


def read_chip(path: str | Path) -> Any:
    """Load a chip from a GeoTIFF as float32 (3, 6, H, W): the file's 18 bands are read date-major (date 1's six
    bands, then date 2's, then date 3's — the dataset's layout)."""
    import numpy as np

    array = _read_tiff(Path(path))
    if array.ndim != 3 or array.shape[0] != NUM_FRAMES * len(BANDS):
        raise ValueError(f"{Path(path).name}: expected an 18-band raster (3 dates × 6 bands), got shape {array.shape}")
    stack = array.reshape(NUM_FRAMES, len(BANDS), *array.shape[1:])
    return np.ascontiguousarray(stack.astype(np.float32))


def read_mask(path: str | Path) -> Any:
    """Load a label raster (0 = no data, 1..13 = class) from a GeoTIFF as int64 (H, W) with classes 0..12 and
    no data as IGNORE_INDEX."""
    import numpy as np

    array = _read_tiff(Path(path))
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{Path(path).name}: a label raster must have one band, got shape {array.shape}")
        array = array[0]
    raw = array.astype(np.int64)
    return np.ascontiguousarray(np.where(raw == 0, IGNORE_INDEX, raw - 1))


def _check_record(record: Any, index: int) -> dict[str, Any]:
    import numpy as np

    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image[/label]")
    for key in ("id", "image"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image = record["id"], record["image"]
    if not isinstance(rid, str) or not rid or len(rid) > 128:
        raise ValueError(f"{label_name}: id must be a non-empty string of at most 128 characters")
    if isinstance(image, str | Path):
        if not Path(image).is_file():
            raise ValueError(f"{label_name}: image file not found: {image}")
        image = read_chip(image)
    try:
        array = np.asarray(image, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_name}: image must be a numeric array") from exc
    expected = (NUM_FRAMES, len(BANDS), IMAGE_SIZE, IMAGE_SIZE)
    if array.shape != expected:
        raise ValueError(f"{label_name}: image must have shape {expected}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label_name}: image contains non-finite values")
    if float(array.min()) >= 0.0 and float(array.max()) <= 1.5:
        array = array * REFLECTANCE_SCALE
    if float(array.min()) < VALUE_RANGE[0] or float(array.max()) > VALUE_RANGE[1]:
        span = (float(array.min()), float(array.max()))
        raise ValueError(f"{label_name}: digital numbers outside the plausible range {VALUE_RANGE}: {span}")
    item: dict[str, Any] = {"id": rid, "image": np.ascontiguousarray(array.astype(np.float32))}
    label = record.get("label")
    if label is not None:
        if isinstance(label, str | Path):
            if not Path(label).is_file():
                raise ValueError(f"{label_name}: label file not found: {label}")
            label = read_mask(label)
        try:
            mask = np.asarray(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label_name}: label must be an integer array") from exc
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{label_name}: label must have shape {(IMAGE_SIZE, IMAGE_SIZE)}, got {mask.shape}")
        if not np.issubdtype(mask.dtype, np.integer) and not np.all(mask == np.round(mask)):
            raise ValueError(f"{label_name}: label values must be integers")
        allowed = set(range(NUM_CLASSES)) | {IGNORE_INDEX}
        found = set(np.unique(mask).astype(int).tolist())
        if not found <= allowed:
            raise ValueError(f"{label_name}: label values {sorted(found - allowed)} outside {sorted(allowed)}")
        item["label"] = np.ascontiguousarray(mask.astype(np.int64))
    for key in ("split", "region", "source", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def check_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record and return its normalised copy (float32 digital numbers, int64 label)."""
    return _check_record(record, 0)


def chip_digest(record: Mapping[str, Any]) -> str:
    checked = _check_record(record, 0)
    digest = hashlib.sha256(checked["image"].tobytes())
    if "label" in checked:
        digest.update(checked["label"].tobytes())
    return digest.hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], chip_digest(r)] for r in records]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Structural validation of a chip dataset; raises ValueError before any model import."""
    import numpy as np

    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        if require_labels and "label" not in item:
            raise ValueError(f"records[{index}] has no label; every record of a labelled dataset needs one")
        if "label" in item:
            valid = item["label"][item["label"] != IGNORE_INDEX]
            class_pixels += np.bincount(valid, minlength=NUM_CLASSES)[:NUM_CLASSES]
            ignored += int(item["label"].size - valid.size)
        checked.append(item)
    labelled = sum("label" in r for r in checked)
    if require_labels and labelled and int((class_pixels > 0).sum()) < 2:
        raise ValueError("fewer than two classes are labelled in the dataset; nothing to learn or evaluate")
    total = int(class_pixels.sum())
    return {
        "records": checked,
        "n_records": len(checked),
        "n_labelled": labelled,
        "image_size": IMAGE_SIZE,
        "dates": NUM_FRAMES,
        "bands": list(BANDS),
        "class_pixel_fraction": {
            CLASS_NAMES[c]: round(float(class_pixels[c]) / total, 4) if total else None for c in range(NUM_CLASSES)
        },
        "classes_present": int((class_pixels > 0).sum()),
        "ignored_pixels": ignored,
        "value_range": [
            round(float(min(r["image"].min() for r in checked)), 1),
            round(float(max(r["image"].max() for r in checked)), 1),
        ],
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record; returns its id, shape, value range and label class fractions."""
    item = _check_record(record, 0)
    report = {
        "id": item["id"],
        "shape": tuple(item["image"].shape),
        "value_range": [round(float(item["image"].min()), 1), round(float(item["image"].max()), 1)],
        "has_label": "label" in item,
    }
    if "label" in item:
        label = item["label"]
        valid = int((label != IGNORE_INDEX).sum())
        report["label_fraction"] = {
            CLASS_NAMES[c]: round(float((label == c).sum()) / max(valid, 1), 4) for c in range(NUM_CLASSES)
        }
        report["ignored_pixels"] = int((label == IGNORE_INDEX).sum())
    return report


# --------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------


def _normalise(images: Any) -> Any:
    """(B, 3, 6, H, W) digital numbers -> the (B, 6, 3, H, W) standardised tensor the network was trained on.

    The upstream data pipeline standardised the 18 date-major channels with the per-band statistics repeated
    three times, then reshaped the (18, H, W) array to (6, 3, H, W) with a plain `reshape` — which splits the 18
    channels into six groups of three consecutive channels, not into bands and dates. The checkpoint learned
    that layout (it scores 62 % pixel accuracy this way and 17 % with a bands-by-dates layout on the tutorial
    chips), so this function reproduces the reshape exactly rather than the layout the axis names suggest."""
    import numpy as np

    mean = np.asarray(MEANS * NUM_FRAMES, dtype=np.float32)[None, :, None, None]
    std = np.asarray(STDS * NUM_FRAMES, dtype=np.float32)[None, :, None, None]
    batch = np.asarray(images, dtype=np.float32)
    flat = batch.reshape(batch.shape[0], NUM_FRAMES * len(BANDS), *batch.shape[-2:])  # date-major channels
    flat = (flat - mean) / std
    return np.ascontiguousarray(flat.reshape(batch.shape[0], len(BANDS), NUM_FRAMES, *batch.shape[-2:]))


@dataclass
class PrithviCropPipeline:
    """Crop / land-cover segmentation and bounded head fine-tuning on top of the verified Prithvi crop model."""

    model: Any
    device: str
    weights_dir: Path
    source: str
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
        report: Callable[[dict[str, Any]], None] | None = None,
    ) -> PrithviCropPipeline:
        """Verify, convert if needed, rebuild from the vendored module and strictly load. With
        `require_source=False` the checkpoint may be absent (the DIMER-hosted case) as long as the converted file
        verifies. `report` receives the audit and conversion records when a conversion happens."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if require_source:
            stage_missing_files(root, allow_download=allow_download)
            snapshot = verify_snapshot(root)
            if not snapshot["converted"]:
                conversion = convert_model(root)
                if report is not None:
                    report({"conversion": conversion})
                snapshot = verify_snapshot(root)
            elif report is not None:
                report({"conversion": "converted file already present and digest-verified"})
            source = "converted from the manifest-verified source checkpoint"
        else:
            verify_converted(root)
            source = "converted file, pinned digest (source checkpoint not required)"
        import torch
        from safetensors.torch import load_file

        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if chosen.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        model = build_model()
        state = load_file(str(root / CONVERTED_WEIGHTS_NAME))
        model.load_state_dict(state, strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != PARAMETER_COUNT:
            raise ValueError(f"rebuilt model has {n_params} parameters, expected {PARAMETER_COUNT}")
        model.to(torch.device(chosen)).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model=model, device=chosen, weights_dir=root, source=source)

    # ---- forward ---------------------------------------------------------------------------------------

    def _logits(self, images: Any, *, grad: bool = False) -> Any:
        """(B, 3, 6, H, W) float32 digital numbers -> (B, 13, H, W) float32 logits at input resolution."""
        import torch

        batch = torch.from_numpy(_normalise(images)).to(self.device)
        use_amp = self.device.startswith("cuda")
        context = torch.enable_grad() if grad else torch.inference_mode()
        with context, torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            logits = self.model(batch)
        return logits.float()

    # ---- inference -------------------------------------------------------------------------------------

    def predict(self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 4) -> dict[str, Any]:
        """Classify every pixel: per record the argmax map (H, W) uint8 with classes 0..12, the softmax scores
        (13, H, W) float32 and the fraction of pixels in each class. Softmax scores are the model's own outputs,
        not calibrated probabilities."""
        import numpy as np
        import torch

        checked = validate_dataset(records, min_records=1, require_labels=False)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        started = time.perf_counter()
        predictions = []
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            logits = self._logits(np.stack([r["image"] for r in batch]))
            scores = torch.softmax(logits, dim=1).cpu().numpy()
            masks = scores.argmax(axis=1).astype(np.uint8)
            for record, score, mask in zip(batch, scores, masks, strict=True):
                predictions.append(
                    {
                        "id": record["id"],
                        "mask": mask,
                        "scores": score.astype(np.float32),
                        "class_fraction": {CLASS_NAMES[c]: round(float((mask == c).mean()), 4) for c in range(NUM_CLASSES)},
                    }
                )
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "classes": list(CLASS_NAMES),
            "decision_rule": "argmax over the 13 class scores (no threshold)",
            "predictions": predictions,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 4) -> dict[str, Any]:
        """Pixel-level metrics on labelled chips (ignore index excluded): per-class IoU and recall, mean IoU,
        mean class accuracy, mean F1 and overall accuracy, with the majority-class baseline scored on the same
        pixels."""
        from .metrics import majority_baseline, segmentation_metrics

        checked = validate_dataset(records, min_records=1)["records"]
        started = time.perf_counter()
        result = self.predict(checked, batch_size=batch_size)
        masks = [p["mask"] for p in result["predictions"]]
        labels = [r["label"] for r in checked]
        metrics = segmentation_metrics(masks, labels, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX)
        return {
            "n_records": len(checked),
            "metric": "pixel IoU / accuracy over the labelled pixels of the held-out chips (ignore index excluded)",
            "model": metrics,
            "baseline_majority": majority_baseline(labels, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX),
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 3),
        }

    # ---- adaptation ------------------------------------------------------------------------------------

    def _trainable(self, mode: str) -> list[str]:
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"trainable must be one of {ADAPTATION_MODES}")
        prefixes = TRAINABLE_PREFIXES[mode]
        return sorted(name for name, _param in self.model.named_parameters() if name.startswith(prefixes))

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 6,
        lr: float = 1e-4,
        batch_size: int = 4,
        trainable: str = "head",
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the segmentation head (`trainable="head"`; `"head+last_block"` also unfreezes the
        last encoder block) on labelled chips: the upstream class-weighted cross-entropy over the labelled pixels
        (ignore index excluded), AdamW at a fixed learning rate, seeded horizontal/vertical flips, float16
        autocast with loss scaling on CUDA. Epoch 0 records the frozen model; the epoch with the lowest validation
        loss is kept."""
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        names = self._trainable(trainable)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        import numpy as np
        import torch

        torch.manual_seed(seed)
        started = time.perf_counter()
        model = self.model
        name_set = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in name_set)
        params = [p for n, p in model.named_parameters() if n in name_set]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        use_amp = self.device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(seed)
        weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=self.device)

        def loss_fn(logits: Any, target: Any) -> Any:
            return torch.nn.functional.cross_entropy(logits, target, weight=weight, ignore_index=IGNORE_INDEX)

        def val_loss() -> float | None:
            if val_checked is None:
                return None
            model.eval()
            losses = []
            for start in range(0, len(val_checked), batch_size):
                batch = val_checked[start : start + batch_size]
                logits = self._logits(np.stack([r["image"] for r in batch]))
                target = torch.from_numpy(np.stack([r["label"] for r in batch])).to(self.device)
                losses.append(float(loss_fn(logits, target)))
            return sum(losses) / len(losses)

        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        try:
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val_loss": val_loss(), "note": "frozen model"}
            if val_checked is not None:
                entry["val"] = self.evaluate(val_checked, batch_size=batch_size)["model"]
            history.append(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            best_epoch = 0
            if progress:
                progress(entry)
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                for module in model.modules():  # BatchNorm statistics stay frozen: tiny batches would corrupt them
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                order = rng.permutation(len(train_checked)).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    images = np.stack([r["image"] for r in batch])
                    labels = np.stack([r["label"] for r in batch])
                    if rng.random() < 0.5:
                        images, labels = images[..., ::-1], labels[..., ::-1]
                    if rng.random() < 0.5:
                        images, labels = images[..., ::-1, :], labels[..., ::-1, :]
                    logits = self._logits(np.ascontiguousarray(images), grad=True)
                    target = torch.from_numpy(np.ascontiguousarray(labels)).to(self.device)
                    loss = loss_fn(logits, target)
                    optimiser.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_loss()}
                if val_checked is not None:
                    entry["val"] = self.evaluate(val_checked, batch_size=batch_size)["model"]
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the model as it
            # was before adapt() (trained tensors restored), frozen, with no adapter attached.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable": trainable,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "loss": "cross-entropy with the upstream class weights, ignore index excluded",
            "augmentation": "seeded horizontal/vertical flips",
            "batchnorm": "running statistics frozen (eval mode) during adaptation",
            "precision": "float16 autocast + GradScaler" if use_amp else "float32",
            "n_train_records": len(train_checked),
            "n_steps": n_steps,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts -------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors as safetensors with a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "converted_sha256": CONVERTED_SHA256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> tuple[Path, str]:
        """Static checks on an adapter manifest, before any model or weights work: format and version, the pinned
        base and converted digest, exactly one weights entry named `adapter.safetensors` inside the artifact
        directory, and an adaptation mode that is one of the declared scopes. Returns the weights path and mode."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("converted_sha256") != CONVERTED_SHA256:
            raise ValueError("artifact records a different converted-base digest")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        mode = adapter.get("trainable") if isinstance(adapter, Mapping) else None
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"artifact adapter.trainable must be one of {ADAPTATION_MODES}")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path, mode

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, scope and digest, then overwrite exactly the tensors the scope allows."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path, mode = self.check_artifact_manifest(root, manifest)
        expected = self._trainable(mode)
        if sorted(manifest["tensors"]) != expected:
            raise ValueError(
                f"artifact tensor list does not match the {len(expected)} tensors that trainable={mode!r} may change"
            )
        entry = manifest["files"][0]
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the validated manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, model has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
    ) -> PrithviCropPipeline:
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.check_artifact_manifest(root, manifest)
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download, require_source=require_source
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
