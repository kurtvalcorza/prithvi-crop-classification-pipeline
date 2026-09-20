# Prithvi Crop Classification Pipeline

DIMER-oriented pipeline for **Prithvi-EO-1.0-100M multi-temporal crop classification** (`ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification` — NASA and IBM's fine-tune of their first HLS foundation model to 13 Cropland-Data-Layer-derived classes), pinned to an immutable Hugging Face revision. The repository exposes 13-class per-pixel classification of 224 × 224 three-date six-band HLS chips, per-class IoU and accuracy against a majority-class baseline, a labelled-chip contract with explicit ceilings, a bounded fine-tuning contract for the segmentation head with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification`
- Revision: `b53a88b8da673800b67c34a98a527b77076e7035`
- Source asset: `multi_temporal_crop_classification_Prithvi_100M.pth` (1,680,468,041 bytes, SHA-256 `37ed4163…`) — an mmsegmentation checkpoint (a torch pickle with the state dict, the optimizer state and a `meta` record), converted once to safetensors and never served (see below)
- Upstream code: `NASA-IMPACT/hls-foundation-os` at `3b6d401f3b4527059af0e44bd640225285e1933d` (`geospatial_fm/geospatial_fm.py`) and the mmsegmentation 0.30 FCN head, vendored in plain PyTorch as `src/prithvi_crop_classification_pipeline/modeling.py`
- Upstream weight license: Apache-2.0
- Upstream task: multi-temporal crop classification — the first six blocks of the Prithvi-EO-1.0-100M encoder over three dates, a transposed-convolution neck and an FCN head over 13 classes, trained for 80 epochs on 2022 HLS chips with CDL labels (Jakubik et al., 2023)
- Runtime: `torch==2.14.0` + `numpy` + `tifffile` + `safetensors` + `huggingface-hub` — **no mmcv, no mmsegmentation, no timm, no Hub-hosted code, no served pickle**
- Repository adaptation: **E2E** (bounded fine-tuning of the segmentation head — optionally the last encoder block — on labelled chips with the upstream class-weighted loss, with a portable safetensors adapter)

## Three things to know before you start

**The checkpoint is a pickle, and the pipeline converts it once.** The upstream file is a torch zip archive whose pickle references, besides the four state-dict globals, three data-only names that rebuild one numpy scalar in `meta` (`numpy.core.multiarray.scalar`, `numpy.dtype`, `_codecs.encode`). `audit_pickle` lists every global with `pickletools` (no execution) and refuses anything else; `convert_model` unpickles it once through `torch.load(weights_only=True)` with those three names bound to inert stand-ins, drops the training-only auxiliary head and the optimizer, loads the 98 inference tensors strictly into the vendored network and writes `prithvi-eo-1.0-100m-crop.safetensors` (537,722,508 bytes, SHA-256 `d1df8044…`), the only file the model is ever loaded from.

**The network is vendored, and the input layout is the one it learned.** `modeling.py` rebuilds the temporal ViT encoder, the neck and the head without the OpenMMLab stack; its computed positional table equals the checkpoint's exactly. The upstream data pipeline standardised the 18 date-major channels and then folded them into (6, 3, H, W) with a plain `reshape` — which mixes the band and date axes, not the layout the names suggest. The checkpoint learned that layout (59.0 % pixel accuracy on the tutorial's test chips against 15.1 % the "semantic" way), so `pipeline._normalise` reproduces the reshape exactly and the contract is stated in terms of the file layout: three dates × six bands, date-major, as digital numbers.

**The model was selected on the chips the tutorial samples.** Every chip in the dataset's validation archive belongs to the split the published checkpoint's best epoch was chosen on. The frozen model scores a mean IoU of 0.438 / accuracy 0.591 on 12 held-out chips against 0.011 / 0.137 for the majority class; the bounded head fine-tuning lowers the class-weighted validation loss a little (0.937 → 0.911) and leaves the mean IoU where it was (0.438 → 0.430). That is the contract demonstrated, not a claim that this sample improves the model.

## Quick start

```python
from prithvi_crop_classification_pipeline import PrithviCropPipeline, fetch_sample_dataset

pipe = PrithviCropPipeline.from_pretrained(allow_download=True)  # stages + verifies the snapshot, audits + converts the pickle once, loads safetensors
splits = fetch_sample_dataset()                                 # 36 / 12 / 12 labelled chips extracted from the pinned tarball
print(pipe.evaluate(splits["test"])["model"]["mean_iou"])       # frozen model (the majority baseline is under ["baseline_majority"])
pipe.adapt(splits["train"], splits["validation"], epochs=4, lr=1e-5)  # bounded fine-tuning of the head, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["model"]["mean_iou"])       # adapted model, same chips
maps = pipe.predict(splits["test"][:2])["predictions"]          # argmax class maps (224, 224) uint8 with classes 0..12 and softmax scores
pipe.save_artifact("outputs/adapter")
```

`predict()` takes records `{id, image}` with a (3, 6, 224, 224) float32 array of digital numbers (or an 18-band GeoTIFF path); `evaluate()` and `adapt()` take `{id, image, label}` records with a (224, 224) label of classes 0..12 and −1 for no data (or a GeoTIFF path in the dataset's 0 / 1..13 convention). Validation is structural: nothing checks that the bands are the six HLS bands in the right order, that the dates are growing-season dates, or that a label belongs to its chip.

## Weights layout

```
weights/prithvi-eo-1.0-100m-crop/   multi_temporal_crop_classification_Prithvi_100M.pth  (git-ignored, the pinned source)
                                    multi_temporal_crop_classification_Prithvi_100M.py   (the upstream mmseg config)
                                    README.md                                            (the upstream card)
                                    dimer-base-manifest.json
                                    prithvi-eo-1.0-100m-crop.safetensors                 (git-ignored, converted)
weights/multi-temporal-crop/        validation_chips.tgz + chips/                        (git-ignored, the pinned dataset tarball and the 120 extracted members)
```

`from_pretrained()` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of every manifest entry and of the converted file when present), converts the pickle when the safetensors file is absent, and refuses on the first mismatch. With `require_source=False` the digest-verified converted file is accepted without the checkpoint — the DIMER-hosted shape. `docs/WEIGHTS.md` records the provenance, the audit, the conversion, the vendored code, the data pins and the DIMER hosting notes.

## Sample data

`fetch_sample_dataset()` fetches the dataset's `validation_chips.tgz` (1.18 GB, CC BY 4.0) from the Hub at an immutable revision, verifies it by size and SHA-256, streams through it once and extracts exactly the 120 pinned members (each verified again) into `weights/multi-temporal-crop/chips/`, then reads them as `{id, image, label}` records with roles by 4 × 4-chip block (36 train / 12 validation / 12 test, every class present in every role). `dataset_manifest` validates the splits, refuses a chip or a block in two splits and records a digest. Nothing is vendored under `weights/`.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — the head, about 21 MB; with `trainable="head+last_block"` also encoder block 5, about 50 MB) and a `manifest.json` recording the artifact format, the exact base model id and revision, the converted-base digest, the adaptation scope, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `PrithviCropPipeline.from_artifact(dir)` re-verifies the base file, checks the manifest, scope and digest before deserialising, rebuilds the network and overlays the tensors.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: crafted pickles (an mmseg-shaped stream with the three `meta` globals, a stream with executable globals), temporary manifests, synthetic chips, a synthetic tarball with a decoy member and an AppleDouble twin, a stub model with the vendored parameter layout, and the vendored network at random initialisation (shapes, keys, positional table, the reshape contract), never the weights; `tests/test_model_backed.py` runs the real converted weights when they are staged (strict load, classification, one adaptation epoch, reload parity) and skips otherwise. `tifffile` is required by the chip tests. The model-backed smoke and the GPU sweep are recorded in `MODEL_CARD.md` (*Runtime*).

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/prithvi-crop-classification-pipeline/blob/main/tutorials/prithvi_crop_classification_colab.ipynb)

`tutorials/prithvi_crop_classification_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 4 package modules (`modeling.py`, `metrics.py`, `pipeline.py`, `samples.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It needs a GPU runtime. It converts the pinned checkpoint in the runtime (the audit and conversion records are printed before the model loads), extracts the pinned chips from the digest-verified tarball, and runs the sample path: validation, the frozen model against the majority-class baseline, bounded fine-tuning of the head, held-out evaluation, class maps, adapter export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Candidate** — the `E2E` notebook has executed top-to-bottom on the local pre-flight harness only (WSL, RTX 5070 Ti, weights and tarball pre-staged); the clean-runtime Kaggle execution that promotes it is pending and will be recorded in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but are not the evidence; the hosted run is.

## Licensing

- Upstream weights: Apache-2.0, staged from the pinned Hub revision and converted, not modified, into the served safetensors.
- Upstream code: Apache-2.0 (`NASA-IMPACT/hls-foundation-os`), vendored as `modeling.py` with the commit and file digest recorded in `docs/WEIGHTS.md`; the mmsegmentation head it reproduces is Apache-2.0 as well.
- Tutorial data: the HLS multi-temporal crop classification dataset is CC BY 4.0 (NASA IMPACT / IBM); the tutorial fetches it at run time and this repository redistributes no chip.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository's code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
