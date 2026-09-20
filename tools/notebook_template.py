"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (modeling.py, metrics.py, pipeline.py, samples.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E crop-classification workflow: the pinned Prithvi-EO-1.0-100M crop checkpoint
(an mmsegmentation pickle) is digest-verified, statically audited and converted once into safetensors, 60 labelled
three-date HLS chips are extracted from the digest-pinned dataset tarball, validated and assigned roles by spatial
block, the frozen model is scored against the majority-class baseline, a bounded fine-tuning of the segmentation
head runs in the kernel, the held-out chips are scored again, class maps are written, and the adapter is exported
and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "prithvi-crop-classification-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/prithvi_crop_classification_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ibm--nasa--geospatial%2FPrithvi--EO--1.0--100M--multi--temporal--crop--classification-ffcc4d?style=flat",
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-NASA--IMPACT%2Fhls--foundation--os-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/NASA-IMPACT/hls-foundation-os",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2310.18660-b31b1b.svg", "https://arxiv.org/abs/2310.18660"),
]

TEMPLATE = {
    "package": "prithvi_crop_classification_pipeline",
    "repo_name": REPO,
    "stem": "prithvi_crop_classification",
    "notebook_name": "prithvi_crop_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime installs the pinned dependencies (torch, tifffile, numpy, safetensors, "
        "huggingface-hub — no mmcv, mmsegmentation or timm: the network is carried in this notebook), stages and digest-verifies "
        "the pinned Prithvi crop-classification checkpoint (1.68 GB) from the Hub, statically audits the mmsegmentation pickle "
        "against an allow-list, converts it once into safetensors with a pinned digest (keeping the 98 inference tensors and "
        "dropping the training-only auxiliary head and the optimizer state), rebuilds the network from the carried module and "
        "loads it strictly, fetches the digest-pinned dataset tarball (1.18 GB, no credential) and extracts exactly the 120 pinned "
        "chip and mask members, validates them and assigns roles by spatial block (36 training, 12 validation, 12 test), classifies "
        "the held-out chips with the frozen model and scores them against the majority-class baseline, runs a bounded fine-tuning "
        "of the segmentation head, scores the same chips again, writes class maps for two held-out chips, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On a T4 the whole path takes a few minutes of model time after the downloads; the tarball "
        "and the adaptation are the slowest steps."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled chips as a zip holding `pairs.csv` (columns `id`, `image`, `label`) beside 18-band 224 × 224 GeoTIFF chips "
        "(three dates × six HLS bands — blue, green, red, narrow NIR, SWIR 1, SWIR 2 — date-major, surface reflectance × 10 000) "
        "and single-band label rasters (0 = no data, 1..13 = the classes in the order the model uses); at least four chips with "
        "at least two classes. Your chips are split by seed into training, validation and test sets and flow through the same "
        "contract — validation, frozen baseline, adaptation, held-out evaluation, class maps, artifact export and reload parity. "
        "The expected schema, the ceilings and the privacy guidance are stated in the Prerequisites and in Section 4, and "
        "uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "PrithviCropPipeline",
    "model_load": "PrithviCropPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "prithvi-eo-1.0-100m-crop",
    "modules": ["modeling.py", "metrics.py", "pipeline.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "tifffile"],
    "title": "Prithvi-EO-1.0 multi-temporal crop classification — DIMER E2E segmentation fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "13-class crop and land-cover segmentation of three-date six-band HLS chips with a Prithvi-EO-1.0 temporal ViT encoder, held-out IoU/accuracy against a majority-class baseline, and bounded fine-tuning of the segmentation head to labelled chips",
    "intro": (
        "Prithvi-EO-1.0-100M (Jakubik et al., 2023) is NASA and IBM's first foundation model for Harmonized Landsat Sentinel-2 "
        "imagery: a ViT-B masked autoencoder pretrained on three-date stacks of six bands over the contiguous United States. The "
        "checkpoint packaged here is the upstream authors' fine-tune for multi-temporal crop classification — the first six "
        "encoder blocks, a transposed-convolution neck that folds the three dates into one 16×-upsampled feature map, and an FCN "
        "head over 13 classes derived from the USDA Cropland Data Layer — trained with mmsegmentation on 224 × 224 chips of "
        "three 2022 growing-season dates.\n\n"
        "Three things about this row are handled in the open. **The upstream asset is a pickle** — an mmsegmentation checkpoint "
        "holding the state dict, the Adam optimizer state and a `meta` record. Section 3 downloads and digest-verifies it, "
        "statically lists every global the pickle would import (a state dict of tensors, plus the three data-only names that "
        "rebuild one numpy scalar in `meta`), refuses anything outside that allow-list, unpickles it exactly once through torch's "
        "weights-only loader with those three names bound to inert stand-ins, keeps the 98 inference tensors and writes a "
        "safetensors file whose digest is pinned in the carried module; the network you run is rebuilt from `modeling.py`, "
        "carried in this notebook in plain PyTorch, and loads that file strictly. **The dataset ships as one 1.18 GB tarball**, "
        "so Section 4 pins it by size and digest, streams through it once and copies out exactly the 120 pinned members (each "
        "pinned again by size and digest, no `extractall`, no paths taken from the archive), and leaves the other 1,422 alone. "
        "**The model was selected on these chips**: every chip in the archive belongs to the upstream validation split, on which "
        "the published checkpoint's best epoch was chosen, so the bounded adaptation in Section 6 is a demonstration of the "
        "contract, selected by validation loss with the frozen model as epoch 0; the point of the contract is the same recipe "
        "applied to *your* labelled chips."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried network, pipeline, dataset and metrics modules; stage and digest-verify "
        "a pickled checkpoint, read its static audit and see it converted into safetensors; extract pinned members from a "
        "digest-verified tarball and validate real labelled multispectral time series with an ignore class; read per-class IoU, "
        "mean IoU, mean class accuracy and overall accuracy against a majority-class baseline; run a bounded fine-tuning of the "
        "segmentation head with the upstream class-weighted loss, explicit hyperparameters and frozen BatchNorm statistics; "
        "compare the adapted and frozen models on the same held-out chips; write class maps; and export a safetensors adapter "
        "that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "yield estimation, field delineation, dates other than three, chips other than 224 × 224 (tiling is the caller's), "
        "cloud masking, the published benchmark scores, the 12-block Prithvi-EO-1.0 encoder (this fine-tune keeps six blocks), "
        "and any claim that a 60-chip sample stands in for an operational evaluation. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12): the network runs in float16 autocast and the default adaptation needs about 4 GB of GPU memory; on CPU one chip takes several seconds and the adaptation would take an hour. About 4.5 GB of disk is needed for the checkpoint, its conversion and the tarball.",
        "- **Knowledge:** what a multispectral surface-reflectance chip is (bands, dates, digital numbers, no-data), what a pixel-wise class map and an ignore class are, and how per-class IoU and mean IoU are read against a majority baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader — with the three data-only numpy names of its `meta` record bound to inert stand-ins — to produce the safetensors the network is actually loaded from. No Hub-hosted Python module is imported and no mmsegmentation code runs; the network is the carried `modeling.py`.",
        "- **Data contract:** a record is `{{id, image, label}}` — a (3, 6, 224, 224) array of three dates × six HLS bands in digital numbers (reflectance × 10 000; an array in [0, 1] is scaled) or an 18-band date-major GeoTIFF, and a (224, 224) mask with classes 0..12 and −1 for no data (or a GeoTIFF with 0 = no data, 1..13 = class). Validation is structural: nothing checks that the bands are the right six in the right order, that the three dates are growing-season dates, or that the label belongs to the chip.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — field-level records tied to a producer or commercial imagery under licence are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches one pinned object — the 1.18 GB `validation_chips.tgz` of the Hugging Face dataset `ibm-nasa-geospatial/multi-temporal-crop-classification` at an immutable revision — over HTTPS, digest-verified before any member is read; the dataset is CC BY 4.0 (NASA IMPACT / IBM).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample chips, validation and roles\n\n"
                "The default dataset is 60 labelled 224 × 224 chips of the HLS multi-temporal crop classification dataset — three "
                "2022 HLS dates of six bands each, with a 13-class label derived from the USDA Cropland Data Layer — drawn from "
                "the 771 chips of the archive. Roles are assigned per 4 × 4-chip block of the "
                "chip grid (36 training, 12 validation, 12 test, each stratified by dominant class so all 13 classes occur in every "
                "role): chips of one block never straddle roles, but neighbouring blocks may, so this is a split by block, not by "
                "region. `fetch_corpus` downloads the dataset tarball from the Hub at its immutable revision, refuses it on any "
                "size or SHA-256 mismatch, streams through it once and copies out exactly the 120 pinned members — each refused on "
                "its own size or digest mismatch and written under its base name, never at a path taken from the archive — then "
                "reads the 18-band int16 chips as (3, 6, 224, 224) digital numbers and the masks, mapping 0 (no data) to −1 and "
                "1..13 to 0..12. `dataset_manifest` validates every split, checks that no chip or block appears twice and records "
                "a digest.\n\n"
                "Look for: 36 / 12 / 12 chips with all 13 classes present in each role, the block ids per split, a written sample "
                "pair (`outputs/{stem}_sample_chip.tif` + `_sample_label.tif`, the BYOD shape), and three refusal probes — a "
                "two-date chip, a mask with an unknown class, a chip with digital numbers far outside range — each rejected before "
                "the model runs. The tarball takes about a minute to fetch and two to stream."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/multi-temporal-crop')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    top = sorted(part['class_pixel_fraction'].items(), key=lambda kv: -kv[1])[:4]\n"
                "    print({{name: {{'classes_present': part['classes_present'], 'largest_classes': top, 'ignored_pixels': part['ignored_pixels'], 'blocks': len(part['regions'])}}}})\n"
                "print({{'first_test_chip': validate_inputs(test_records[0])}})\n"
                "sample_pair = write_sample_pair(test_records[0], 'outputs/{stem}_sample_chip.tif', 'outputs/{stem}_sample_label.tif')\n"
                "print({{'sample_pair': sample_pair, 'pairs_csv': str(write_dataset_csv(test_records, 'outputs/{stem}_sample_pairs.csv'))}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'two-date chip': [{{**test_records[0], 'image': test_records[0]['image'][:2]}}, *test_records[1:4]],\n"
                "    'unknown label class': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 2, 13, test_records[0]['label'])}}, *test_records[1:4]],\n"
                "    'digital numbers out of range': [{{**test_records[0], 'image': test_records[0]['image'] * 50.0}}, *test_records[1:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. The frozen model against the majority-class baseline\n\n"
                "`pipe.predict` standardises each chip with the band statistics of the upstream training configuration and folds "
                "the 18 channels exactly as the upstream data pipeline did (a plain reshape that the checkpoint learned — see the "
                "note in `pipeline._normalise`), runs the encoder, neck and head in float16 autocast, and returns the argmax map "
                "(classes 0..12), the softmax scores (the model's outputs, not calibrated probabilities) and the class fractions "
                "per chip. `pipe.evaluate` pools the labelled pixels of every held-out chip into one 13 × 13 confusion matrix "
                "(−1 pixels excluded) and reports the per-class IoU and recall, the mean IoU and mean class accuracy over the "
                "classes present, the mean F1 and the overall accuracy; the **majority-class baseline** — every pixel named with "
                "the most frequent class of the scored labels, the best any constant map can do — is scored on the same pixels.\n\n"
                "Look for: a mean IoU near 0.44 and an accuracy near 0.59 on the test chips (in the build record 0.438 and 0.591, "
                "against 0.011 and 0.137 for the majority class, Natural Vegetation; the model card reports a mean IoU of 0.427, an overall "
                "accuracy of 60.6 % and a mean class accuracy of 64.1 % on the full validation split), with Open Water and Winter Wheat the easiest classes and Natural Vegetation and "
                "Other the hardest. These are sample-sanity numbers on 12 chips, not the benchmark."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_majority_test': {{k: frozen_test['baseline_majority'][k] for k in ('majority_class', 'accuracy', 'mean_iou')}}}})\n"
                "print({{'frozen_test': {{k: frozen_test['model'][k] for k in ('mean_iou', 'mean_accuracy', 'mean_f1', 'accuracy', 'classes_scored')}}}})\n"
                "print({{'frozen_test_iou': frozen_test['model']['iou']}})\n"
                "print({{'frozen_validation': {{k: frozen_val['model'][k] for k in ('mean_iou', 'accuracy')}}}})\n"
                "frozen_predictions = pipe.predict(test_records)\n"
                "for record, pred in list(zip(test_records, frozen_predictions['predictions']))[:6]:\n"
                "    labelled = record['label'] >= 0\n"
                "    dominant = CLASS_NAMES[int(np.bincount(record['label'][labelled], minlength=NUM_CLASSES).argmax())]\n"
                "    predicted = max(pred['class_fraction'], key=pred['class_fraction'].get)\n"
                "    print({{'chip': record['source_id'], 'block': record['region'], 'dominant_label': dominant, 'dominant_prediction': predicted, 'agreement': round(float((pred['mask'] == record['label'])[labelled].mean()), 3)}})\n"
                "print({{'decision_rule': frozen_predictions['decision_rule'], 'scores_shape': frozen_predictions['predictions'][0]['scores'].shape}})"
            ),
        },
        {
            "md": (
                "## 6. Bounded fine-tuning of the segmentation head\n\n"
                "`pipe.adapt` trains the 8 tensors of the FCN head (5.3 M parameters — 4 % of the model) and nothing else: the "
                "encoder and the neck are frozen (no gradient is stored for them), and the head's BatchNorm layer keeps its "
                "running statistics, because batches of four chips would corrupt them. Each step takes four chips with a seeded "
                "horizontal or vertical flip, computes the cross-entropy over the labelled pixels (−1 ignored) with the upstream "
                "class weights (rare classes such as Open Water and Sorghum count up to 9× more than Natural Vegetation) and takes "
                "an AdamW step at a small fixed learning rate with gradient-norm clipping and float16 loss scaling. Epoch 0 records "
                "the frozen model's validation loss and metrics; the epoch with the lowest validation loss is kept — which can be "
                "epoch 0, since the packaged checkpoint was selected on the very split these chips come from.\n\n"
                "Watch the validation loss: in the build record it fell from 0.937 to 0.911 over four epochs while the validation mean IoU slipped from 0.461 to 0.445 — the class-weighted loss and the mean IoU the checkpoint was selected by do not rank the same head, which is exactly why the kept epoch is chosen on the loss you declare and reported beside the metric you care about. Four epochs (36 steps) take a few minutes on a T4, the validation pass after each epoch included. "
                "`TRAINABLE = 'head+last_block'` also unfreezes the last encoder block (7.1 M more parameters)."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE = 'head'  # @param [\"head\", \"head+last_block\"]\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_mean_iou'] = entry['val']['mean_iou']\n"
                "        row['val_accuracy'] = entry['val']['accuracy']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable=TRAINABLE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'loss': adapt_result['loss'], 'precision': adapt_result['precision'], 'batchnorm': adapt_result['batchnorm'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison\n\n"
                "The test chips were never used for training or epoch selection (their blocks were assigned to the test role "
                "before anything ran). The adapted model is scored exactly as the frozen model was in Section 5, and the table "
                "puts the baseline, the frozen and the adapted numbers side by side. The cell asserts what the procedure "
                "guarantees — the kept epoch's validation loss is no higher than the frozen model's, and re-scoring the validation "
                "chips reproduces the kept epoch's mean IoU within 0.01 (float16 kernels are not bit-reproducible across batch "
                "sizes) — and prints the test numbers without asserting a direction: on this sample the test mean IoU moved from 0.438 to 0.430 and the accuracy from 0.591 to 0.581 in the build record (the kept epoch lowered the class-weighted validation loss, not the mean IoU), a sample-sanity observation on 12 chips with no dispersion estimate, not a quality claim. With your own chips from "
                "another region or year, the gap between frozen and adapted is the number to watch."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{}}\n"
                "for key in ('mean_iou', 'mean_accuracy', 'mean_f1', 'accuracy'):\n"
                "    comparison[key] = {{'baseline_majority': frozen_test['baseline_majority'][key], 'frozen': frozen_test['model'][key], 'adapted': adapted_test['model'][key]}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "moved = {{name: (frozen_test['model']['iou'][name], adapted_test['model']['iou'][name]) for name in CLASS_NAMES if frozen_test['model']['iou'][name] != adapted_test['model']['iou'][name]}}\n"
                "print({{'per_class_iou_changes': moved if moved else 'none (the kept epoch is the frozen model)'}})\n"
                "print({{'validation_mean_iou': {{'frozen': frozen_val['model']['mean_iou'], 'adapted': adapted_val['model']['mean_iou']}}, 'validation_loss': {{'frozen': adapt_result['history'][0]['val_loss'], 'kept_epoch': adapt_result['history'][adapt_result['best_epoch']]['val_loss']}}}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'frozen': {{'test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'test': adapted_test, 'validation': adapted_val}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "assert adapt_result['history'][adapt_result['best_epoch']]['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['model']['mean_iou'] - adapt_result['history'][adapt_result['best_epoch']]['val']['mean_iou']) < 1e-2\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 8. Class maps, artifact export and fresh reload\n\n"
                "The adapted model's class maps of two held-out chips are written as single-band GeoTIFFs in the dataset's label "
                "convention (0 = no data, 1..13 = class) beside the reference masks, so they can be opened side by side: the "
                "agreement per chip printed here is a sanity check, not an evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors (about 21 MB) as `adapter.safetensors`, with a `manifest.json` "
                "recording the artifact format, the base model id and revision, the digest of the converted base file, the "
                "adaptation scope, the tensor names, the file size and SHA-256, the training configuration and the epoch history "
                "(OUT8). `PrithviCropPipeline.from_artifact` re-verifies the base file, checks the artifact manifest, scope and "
                "digest **before** deserialising, rebuilds the network and overlays the tensors — a fresh object from files, not "
                "the in-memory model (VER2). The cell asserts the same held-out mean IoU within 0.001 and score maps within 0.01 "
                "(VER4: float16 tolerances; on one device they are usually identical)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "import tifffile\n\n"
                "shown_records = test_records[:2]\n"
                "shown_predictions = pipe.predict(shown_records)\n"
                "for record, pred in zip(shown_records, shown_predictions['predictions']):\n"
                "    tifffile.imwrite(f'outputs/{stem}_map_adapted_' + record['source_id'] + '.tif', (pred['mask'].astype(np.int64) + 1).astype(np.uint8))\n"
                "    tifffile.imwrite(f'outputs/{stem}_map_reference_' + record['source_id'] + '.tif', np.where(record['label'] < 0, 0, record['label'] + 1).astype(np.uint8))\n"
                "    labelled = record['label'] >= 0\n"
                "    print({{'chip': record['source_id'], 'agreement': round(float((pred['mask'] == record['label'])[labelled].mean()), 3), 'largest_predicted': max(pred['class_fraction'], key=pred['class_fraction'].get), 'note': 'sanity check on two chips'}})\n"
                "with open('outputs/{stem}_predictions.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'model': shown_predictions['model'], 'classes': shown_predictions['classes'], 'decision_rule': shown_predictions['decision_rule'], 'predictions': [{{'id': p['id'], 'class_fraction': p['class_fraction']}} for p in shown_predictions['predictions']]}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable': artifact_manifest['adapter']['trainable'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PrithviCropPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "before = pipe.predict(test_records[:2])['predictions']\n"
                "after = reloaded.predict(test_records[:2])['predictions']\n"
                "parity = {{'mean_iou_diff': round(abs(reloaded_test['model']['mean_iou'] - adapted_test['model']['mean_iou']), 6), 'metrics_identical': reloaded_test['model'] == adapted_test['model'], 'max_abs_score_diff': max(float(np.abs(a['scores'] - b['scores']).max()) for a, b in zip(before, after))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['mean_iou_diff'] < 1e-3 and parity['max_abs_score_diff'] < 1e-2\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': [e for e in MANIFEST['files'] if e['path'] == SOURCE_CKPT_NAME],\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'meta_globals_bound_to_stand_ins': sorted(META_GLOBALS),\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'network_source': 'modeling.py carried in this notebook (plain PyTorch)',\n"
                "        'data_tarball': {{'name': TAR_NAME, 'sha256': TAR_SHA256, 'pinned_members': 2 * len(SAMPLE_RECORDS)}},\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'tifffile': tifffile.__version__, 'numpy': np.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 12 held-out chips the packaged crop-classification model reaches a mean IoU near 0.44 and an accuracy near 0.59 "
        "against a majority-class baseline of 0.01 and 0.14; a bounded fine-tuning of its segmentation head on 36 chips, "
        "selected by validation loss with the frozen model as a candidate, lowers the class-weighted validation loss a little and leaves the mean IoU where it was (0.438 → 0.430 on the test chips in the build record). That is the claim: the "
        "adaptation contract runs end to end on real labelled multispectral time series drawn from a digest-verified tarball, "
        "the pickle is audited and converted rather than served, the network is carried in plain PyTorch, and the artifact that "
        "carries the change is 21 MB and reloads with the same outputs. It is not a claim that this sample improves the model — "
        "the checkpoint was selected on the split these chips come from — nor that 12 chips measure its skill.\n\n"
        "The numbers are sample-sanity evidence: one seeded run, 12 test chips, no dispersion estimate, pixel-pooled metrics "
        "that let large fields dominate, and labels derived from the Cropland Data Layer with their own errors at field edges "
        "and for minor crops. Nothing here measures the model outside the contiguous United States, outside 2022, on other "
        "dates, or on chips larger than 224 × 224.\n\n"
        "Three things to carry to real data. **The 18 channels and their scaling are the contract:** three dates × six bands "
        "— blue, green, red, narrow NIR, SWIR 1, SWIR 2 — date-major, as digital numbers; and the network folds them exactly "
        "as its training pipeline did, which is not the (bands, dates) layout the axis names suggest — a different band order "
        "or date order is classified without complaint and silently wrong. **Split by region, not by chip:** neighbouring chips "
        "share fields, and a random split makes memorisation look like skill. **Read the baseline and the per-class IoU first:** "
        "the majority class alone is right on a seventh of the pixels, and a mean IoU hides that Natural Vegetation is barely found while "
        "Open Water is easy.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify a pickled upstream checkpoint, audit and convert it into safetensors without "
        "executing anything outside the audited allow-list, rebuild the network from the carried module, fetch a digest-pinned "
        "tarball and extract exactly the pinned labelled chips, execute bounded fine-tuning, evaluate against a baseline and the "
        "frozen model on held-out chips, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, production fitness, or crop-mapping skill beyond the "
        "checks shown.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE = 'head+last_block'`; raise `EPOCHS` "
        "and watch the validation loss; try `LEARNING_RATE = 1e-4` to see the frozen model win every epoch; or bring your own "
        "labelled chips through BYOD and read the baseline before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/prithvi-crop-classification-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/prithvi-crop-classification-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/prithvi-crop-classification-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification (revision `{MODEL_REVISION}`)\n"
        "- Multi-temporal crop classification dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification (CC BY 4.0)\n"
        "- Jakubik, J., Roy, S., Phillips, C. E., et al. (2023). Foundation models for generalist geospatial artificial intelligence. arXiv:2310.18660: https://arxiv.org/abs/2310.18660\n"
        "- Upstream fine-tuning code: https://github.com/NASA-IMPACT/hls-foundation-os (the network is vendored in `modeling.py`)\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
