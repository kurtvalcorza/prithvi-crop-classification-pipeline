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
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime builds an isolated environment from the hash-locked pins (torch, tifffile, "
        "numpy, safetensors, huggingface-hub — no mmcv, mmsegmentation or timm: the network is carried in this notebook); nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass. It then stages and digest-verifies "
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
        "labelled chips — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, "
        "through the Colab upload dialog — as a zip (or folder) holding `pairs.csv` (columns `id`, `image`, `label`) beside 18-band 224 × 224 GeoTIFF chips "
        "(three dates × six HLS bands — blue, green, red, narrow NIR, SWIR 1, SWIR 2 — date-major, surface reflectance × 10 000) "
        "and single-band label rasters (0 = no data, 1..13 = the classes in the order the model uses); at least **7** labelled chips with "
        "at least two classes (the seeded 25 % test / 20 % validation split must leave the 4 training chips adaptation needs; Section 4 prints the minimum and "
        "refuses a smaller set by name). An optional `group` column (a field, scene or region id) keeps every chip of one group in one role, so neighbouring "
        "chips of one field cannot sit on both sides of the split. Your chips flow through the same "
        "contract — validation, frozen baseline, adaptation, held-out evaluation, class maps, artifact export and reload parity — and Section 5 puts the model "
        "back to the pinned base first, so the frozen numbers on your chips are the packaged model's. "
        "The expected schema, the ceilings and the privacy guidance are stated in the Prerequisites and in Section 4, and "
        "uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see how a '
            'multi-temporal Earth-observation model that names 13 crop and land-cover classes is evaluated and adapted honestly: how its class maps are scored per '
            'class against the best constant map, what a bounded fine-tuning of its head does to a model selected on the very chips used here, and how the change '
            'is exported and reloaded. No prior experience with Prithvi, mmsegmentation or remote sensing models is assumed; terms are explained where they first '
            'matter and again in the **Glossary** at the end. A GPU runtime (T4 or better) is expected.\n\n**Input → Model → Output.**\n\n| | Classification | Bounded '
            'fine-tuning |\n|---|---|---|\n| Input | 224 × 224 chips of three 2022 dates × six HLS bands (digital numbers) | labelled chips with 13 classes and −1 '
            'for no data (36 training, 12 validation, 12 test, split by 4 × 4-chip block) |\n| Model | the Prithvi-EO-1.0 temporal ViT encoder (six blocks), a '
            'transposed-convolution neck and an FCN head, carried in plain PyTorch and loaded from audited, converted safetensors | the same network; only the FCN '
            'head (5.3 M parameters) is trained with the upstream class weights, encoder, neck and BatchNorm statistics stay frozen |\n| Output | a per-pixel class '
            "map (0..12), softmax scores and class fractions; per-class IoU, mean IoU and accuracy beside the majority-class baseline | the adapted model's numbers "
            "beside the frozen model's on the same held-out chips, and a 21 MB safetensors adapter that reloads with parity |\n\n**How to use this notebook.** Choose "
            'a GPU runtime (**Runtime → Change runtime type → T4 GPU**), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into '
            "the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried network and package, "
            'and the audited model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form '
            'fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook '
            'asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded run (the '
            'Kaggle Tesla T4 run of 20 September 2026 recorded in `docs/release-verification.md`). Every adaptation starts from the pinned base, so re-running '
            'Section 6 with other settings is a fresh experiment. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your '
            'predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the labelled chips, block split, validation and refusals *(evaluation practice)* → '
            '5 the frozen model against the majority-class baseline *(core concept)* → 6 bounded fine-tuning of the segmentation head *(core concept)* → 7 the '
            'held-out paired comparison with per-chip numbers *(evaluation practice)* → 8 class maps shown inline, export and fresh reload *(engineering)* → interpretation, an optional experiment, troubleshooting, glossary '
            'and your conclusion.'
        ],
    },
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
        "- **Data contract:** a record is `{id, image, label}` — a (3, 6, 224, 224) array of three dates × six HLS bands in digital numbers (reflectance × 10 000; an array in [0, 1] is scaled) or an 18-band date-major GeoTIFF, and a (224, 224) mask with classes 0..12 and −1 for no data (or a GeoTIFF with 0 = no data, 1..13 = class). Validation is structural: nothing checks that the bands are the right six in the right order, that the three dates are growing-season dates, or that the label belongs to the chip.",
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
                "the model runs. The tarball takes about a minute to fetch and two to stream.\n\n"
                "For your own chips the same advice applies and the `group` column of `pairs.csv` applies it: give every chip a field, scene or region id and "
                "`split_dataset` keeps every chip of one group in one role (the cell reports `grouped_split`); without the column the split is a seeded shuffle by "
                "chip, and neighbouring chips of one field can land on both sides. The exported `pairs.csv` of the sample carries the block id in that column. "
                "The smallest BYOD dataset the split accepts is 7 distinct labelled chips (the cell prints it as `minimum_chips`).\n\n"
                "*Evaluation practice.* **Predict before running:** neighbouring chips share fields. Why are roles assigned per 4 × 4-chip "
                "block rather than per chip, and what leakage can remain?"
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "# A .zip or a folder already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding pairs.csv and the GeoTIFF files.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding pairs.csv and the GeoTIFF files.')\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    byod_records = load_byod_dataset(byod_path)\n"
                "    byod_grouped = bool(byod_records) and all(r.get('group') for r in byod_records)\n"
                "    splits = split_dataset(byod_records, seed=0)\n"
                "    print({{'byod_chips': len(byod_records), 'minimum_chips': byod_minimum_records(), 'grouped_split': byod_grouped, 'groups': sorted({{r['group'] for r in byod_records}}) if byod_grouped else 'no group column: chips of one field or scene may land in different roles'}})\n"
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
                "# CR-m1: pad each probe to the dataset minimum from the other roles, so a small BYOD set still fails on the intended check.\n"
                "probe_fill = (test_records[1:] + train_records + val_records)[:MIN_RECORDS - 1]\n"
                "probes = {{\n"
                "    'two-date chip': [{{**test_records[0], 'image': test_records[0]['image'][:2]}}, *probe_fill],\n"
                "    'unknown label class': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 2, 13, test_records[0]['label'])}}, *probe_fill],\n"
                "    'digital numbers out of range': [{{**test_records[0], 'image': test_records[0]['image'] * 50.0}}, *probe_fill],\n"
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
                '**What to notice:** 36 / 12 / 12 chips, `classes_present` (13 in every role), the largest classes per split, the block counts, and the three refusals.\n\n<details><summary>Check '
                'your reasoning</summary>A field that spans two chips would otherwise appear in training and test, and the model would be rewarded for remembering it. '
                'Blocks keep each 4 × 4 group of chips in one role, but two neighbouring blocks can still land in different roles, so this is a split by block, not by '
                'region — leakage at block edges remains possible. The refusals (two dates, an unknown class, digital numbers far out of range) stop before any model '
                'call and name the rule.</details>'
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
                "the most frequent class of the scored labels, the best any constant map can do — is scored on the same pixels. "
                "The decision rule is the argmax over the 13 class scores, and it is a default, not a tuned operating point: any minimum-confidence rule, "
                "class-specific threshold or post-processing (majority filtering, field-level voting) is the deployment's to set on its own validation data, "
                "and the deployment owns the calibration of the scores; this notebook sets none of them. The cell also prints the mean IoU and accuracy "
                "**per chip**, because the pooled numbers let large fields dominate.\n\n"
                "Look for: a mean IoU near 0.44 and an accuracy near 0.59 on the test chips (in the build record 0.438 and 0.591, "
                "against 0.011 and 0.137 for the majority class, Natural Vegetation; the model card reports a mean IoU of 0.427, an overall "
                "accuracy of 60.6 % and a mean class accuracy of 64.1 % on the full validation split), with Open Water and Winter Wheat the easiest classes and Natural Vegetation and "
                "Other the hardest. These are sample-sanity numbers on 12 chips, not the benchmark. If you re-run this cell after Section "
                "6, it first puts the adapted head back to the pinned base, so *frozen* always means the packaged model.\n\n"
                "**Predict before running:** the most frequent class covers about one pixel in seven. Will the packaged model's mean IoU "
                "over 13 classes be closer to 0.1, 0.4 or 0.8?"
            ),
            "code": (
                "import time\n\n"
                "t0 = time.perf_counter()\n"
                "# SWP-F: the frozen numbers are always the pinned base. On a re-run after Section 6 the adapted tensors are put back\n"
                "# to the base first (then re-run Sections 6-8 in order); adapt() itself also starts from the base on every call.\n"
                "restored_tensors = pipe.restore_base()\n"
                "if restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'adapted tensors put back to the pinned base; re-run Sections 6-8 in order'}})\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_majority_test': {{k: frozen_test['baseline_majority'][k] for k in ('majority_class', 'accuracy', 'mean_iou')}}}})\n"
                "print({{'frozen_test': {{k: frozen_test['model'][k] for k in ('mean_iou', 'mean_accuracy', 'mean_f1', 'accuracy', 'classes_scored')}}}})\n"
                "print({{'frozen_test_iou': frozen_test['model']['iou']}})\n"
                "print({{'frozen_validation': {{k: frozen_val['model'][k] for k in ('mean_iou', 'accuracy')}}}})\n"
                "frozen_predictions = pipe.predict(test_records)\n"
                "# CR-M2: BYOD records carry id / image / label only; source_id and region (block) exist for the sample alone.\n"
                "chip_name = lambda record: record.get('source_id', record['id'])\n"
                "frozen_per_chip = per_chip_metrics([p['mask'] for p in frozen_predictions['predictions']], [r['label'] for r in test_records], class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX, ids=[chip_name(r) for r in test_records])\n"
                "for record, pred, row in list(zip(test_records, frozen_predictions['predictions'], frozen_per_chip))[:6]:\n"
                "    labelled = record['label'] >= 0\n"
                "    dominant = CLASS_NAMES[int(np.bincount(record['label'][labelled], minlength=NUM_CLASSES).argmax())]\n"
                "    predicted = max(pred['class_fraction'], key=pred['class_fraction'].get)\n"
                "    print({{'chip': chip_name(record), 'block': record.get('region', '—'), 'dominant_label': dominant, 'dominant_prediction': predicted, 'agreement': round(float((pred['mask'] == record['label'])[labelled].mean()), 3), 'mean_iou': row['mean_iou']}})\n"
                "frozen_iou_range = [row['mean_iou'] for row in frozen_per_chip]\n"
                "print({{'per_chip_mean_iou_frozen': {{'min': min(frozen_iou_range), 'max': max(frozen_iou_range), 'n_chips': len(frozen_iou_range)}}}})\n"
                "print({{'decision_rule': frozen_predictions['decision_rule'], 'scores_shape': frozen_predictions['predictions'][0]['scores'].shape}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `baseline_majority_test` (class, accuracy, mean IoU) beside `frozen_test`, the per-class IoU, and the per-chip dominant label '
                'against prediction.\n\n<details><summary>Check your reasoning</summary>Near 0.4. In the recorded run the frozen test mean IoU was 0.4379 with accuracy '
                '0.5905, against 0.011 and 0.137 for the majority class (Natural Vegetation). The per-class IoU explains the mean: Open Water and Winter Wheat are '
                'easy, Natural Vegetation and Other are barely found. A single mean IoU hides that spread; read the per-class row.</details>'
            ),
        },
        {
            "md": (
                "## 6. Bounded fine-tuning of the segmentation head\n\n"
                "`pipe.adapt` trains the 5 trainable tensors of the FCN head (5.3 M parameters — 4 % of the model) and nothing else: the "
                "encoder and the neck are frozen (no gradient is stored for them), and the head's BatchNorm layer keeps its "
                "running statistics (its three buffers are not trained, which is why the exported adapter holds 5 tensors, not the head's 8 state-dict entries), "
                "because batches of four chips would corrupt them. Each step takes four chips with a seeded "
                "horizontal or vertical flip, computes the cross-entropy over the labelled pixels (−1 ignored) with the upstream "
                "class weights (rare classes such as Open Water and Sorghum count up to 9× more than Natural Vegetation) and takes "
                "an AdamW step at a small fixed learning rate with gradient-norm clipping and float16 loss scaling. Epoch 0 records "
                "the frozen model's validation loss and metrics; the epoch with the lowest validation loss is kept — which can be "
                "epoch 0, since the packaged checkpoint was selected on the very split these chips come from.\n\n"
                "Watch the validation loss: in the build record it fell from 0.937 to 0.911 over four epochs while the validation mean IoU slipped from 0.461 to 0.445 — the class-weighted loss and the mean IoU the checkpoint was selected by do not rank the same head, which is exactly why the kept epoch is chosen on the loss you declare and reported beside the metric you care about. Four epochs (36 steps) take a few minutes on a T4, the validation pass after each epoch included; the cell prints the peak GPU memory it used. "
                "`TRAINABLE = 'head+last_block'` also unfreezes the last encoder block (7.1 M more parameters). Every call starts "
                "from the pinned base (`started_from` in the printed result), so a re-run with other settings is a fresh experiment, "
                "not continued training, and epoch 0 is always the frozen model.\n\n"
                "**Predict before running:** the head is trained on the class-weighted loss. If the validation loss falls, will the "
                "validation mean IoU rise with it?"
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
                "adapt_result['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'loss': adapt_result['loss'], 'precision': adapt_result['precision'], 'batchnorm': adapt_result['batchnorm'], 'seconds': adapt_seconds, 'gpu_peak_gb': adapt_result['gpu_peak_gb']}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (`note: frozen model`), `val_loss` against `val_mean_iou` per epoch, `best_epoch`, and the trainable share.\n\n<details><summary>Check '
                'your reasoning</summary>Not necessarily. In the recorded run the class-weighted validation loss fell from 0.9368 to 0.9114 and epoch 4 was kept, while '
                'the validation mean IoU slipped from 0.4609 to 0.4451. The loss rewards getting rare, heavily weighted classes right; the mean IoU weighs every class '
                'equally over pixels. Selection follows the loss you declare, so report the metric you care about beside it.</details>'
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
                "another region or year, the gap between frozen and adapted is the number to watch — and in a BYOD run *frozen* is the packaged model, "
                "because Section 5 restored the pinned base before scoring it. The cell also prints the adapted model's mean IoU per chip beside the frozen "
                "model's, so a change hidden by the pooled number is visible.\n\n"
                "*Evaluation practice.* **Predict before running:** given Section 6, will the test mean IoU of the adapted head be above "
                "or below the frozen model's — and which class will move most?"
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
                "# CR-S5: per-chip mean IoU, frozen beside adapted (the pooled metric lets large fields dominate).\n"
                "adapted_predictions = pipe.predict(test_records)\n"
                "adapted_per_chip = per_chip_metrics([p['mask'] for p in adapted_predictions['predictions']], [r['label'] for r in test_records], class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX, ids=[chip_name(r) for r in test_records])\n"
                "for f_row, a_row in zip(frozen_per_chip, adapted_per_chip):\n"
                "    print({{'chip': f_row['id'], 'frozen_mean_iou': f_row['mean_iou'], 'adapted_mean_iou': a_row['mean_iou'], 'classes_scored': a_row['classes_scored']}})\n"
                "comparison['per_chip_mean_iou_range'] = {{'frozen': [min(r['mean_iou'] for r in frozen_per_chip), max(r['mean_iou'] for r in frozen_per_chip)], 'adapted': [min(r['mean_iou'] for r in adapted_per_chip), max(r['mean_iou'] for r in adapted_per_chip)]}}\n"
                "print({{'per_chip_mean_iou_range': comparison['per_chip_mean_iou_range']}})\n"
                "print({{'validation_mean_iou': {{'frozen': frozen_val['model']['mean_iou'], 'adapted': adapted_val['model']['mean_iou']}}, 'validation_loss': {{'frozen': adapt_result['history'][0]['val_loss'], 'kept_epoch': adapt_result['history'][adapt_result['best_epoch']]['val_loss']}}}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'frozen': {{'test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'test': adapted_test, 'validation': adapted_val}},\n"
                "    'per_chip': {{'frozen': frozen_per_chip, 'adapted': adapted_per_chip}},\n"
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
                '**What to notice:** the `mean_iou`, `mean_accuracy` and `accuracy` rows (baseline, frozen, adapted), `per_class_iou_changes`, and the per-chip range.\n\n<details><summary>Check '
                'your reasoning</summary>Slightly below, and the majority class moved most. In the recorded run the test mean IoU moved from 0.4379 to 0.4302 and accuracy from 0.5905 to 0.5813, while '
                'the mean class accuracy rose a hair (0.6239 → 0.6250). Underneath, Natural Vegetation — the most frequent class — lost a quarter of its IoU (0.2494 → 0.1884) '
                'while Fallow/Idle Cropland gained (0.3229 → 0.3784) and Cotton slipped (0.432 → 0.399): the class-weighted loss, which counts rare classes up to 9× more, traded '
                'the common class for rarer ones, which is what it is built to do and what the per-class row is for. Twelve chips and one seed cannot resolve '
                'differences this small; the frozen model was selected on this split, so no gain was expected.</details>'
            ),
        },
        {
            "md": (
                "## 8. Class maps, artifact export and fresh reload\n\n"
                "The adapted model's class maps of two held-out chips are written as single-band GeoTIFFs in the dataset's label "
                "convention (0 = no data, 1..13 = class) beside the reference masks, and **shown inline**: for each chip one strip — a SWIR 2 / NIR / red "
                "composite of the middle date, the reference map (black = no data), the frozen class map and the adapted class map, in a fixed 13-colour "
                "palette whose legend is printed under the strip (`class_map_panel` + `render_png`, no plotting library). Look at where the frozen and adapted "
                "maps differ and whether those pixels moved towards the reference; the agreement per chip printed here is a sanity check, not an evaluation. "
                "The strips are also written as `outputs/{stem}_maps_<chip>.png`.\n\n"
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
                "# CR-M2: chip_name() works for sample and BYOD records alike. CR-m4: the maps are shown inline, not only written.\n"
                "for record, pred, frozen_pred in zip(shown_records, shown_predictions['predictions'], frozen_predictions['predictions']):\n"
                "    name = chip_name(record)\n"
                "    tifffile.imwrite(f'outputs/{stem}_map_adapted_' + name + '.tif', (pred['mask'].astype(np.int64) + 1).astype(np.uint8))\n"
                "    tifffile.imwrite(f'outputs/{stem}_map_reference_' + name + '.tif', np.where(record['label'] < 0, 0, record['label'] + 1).astype(np.uint8))\n"
                "    labelled = record['label'] >= 0\n"
                "    print({{'chip': name, 'agreement': round(float((pred['mask'] == record['label'])[labelled].mean()), 3), 'largest_predicted': max(pred['class_fraction'], key=pred['class_fraction'].get), 'note': 'sanity check on two chips'}})\n"
                "    panel = class_map_panel(record, frozen_pred['mask'], pred['mask'])\n"
                "    png = render_png(panel['rgb'])\n"
                "    with open(f'outputs/{stem}_maps_' + name + '.png', 'wb') as f:\n"
                "        f.write(png)\n"
                "    print({{'chip': name, 'panels': panel['panels']}})\n"
                "    display(PngImage(png, caption=name))\n"
                "print({{'legend': {{entry['class']: 'rgb' + str(tuple(entry['rgb'])) for entry in panel['legend']}}}})\n"
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
                "    'adaptation_gpu_peak_gb': adapt_result['gpu_peak_gb'],\n"
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
        {
            "md": (
                "**What to notice:** in each strip, where the frozen and adapted maps differ and whether those pixels agree with the reference; the agreement of the two class maps with their references; the artifact's size and tensor count; and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>Agreement on two chips is a sanity check, not an evaluation: the strips show the shape of the errors (field edges, the classes the model confuses), and the GeoTIFFs can be opened beside the reference masks in any GIS. In '
                'the recorded run the 5-tensor, 21 MB adapter (the head\'s weights and biases; its BatchNorm buffers are not trained and not exported) reloaded into a fresh pipeline with identical held-out metrics (`mean_iou_diff` 0.0, `metrics_identical` '
                'True, `max_abs_score_diff` 0.0).</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 12 held-out chips the packaged crop-classification model reaches a mean IoU near 0.44 and an accuracy near 0.59 "
        "against a majority-class baseline of 0.01 and 0.14; a bounded fine-tuning of its segmentation head on 36 chips, "
        "selected by validation loss with the frozen model as a candidate, lowers the class-weighted validation loss a little (0.9368 → 0.9114) and lowers the mean IoU a little "
        "(0.4379 → 0.4302 on the test chips, 0.4609 → 0.4451 on validation, in the build record) — and underneath the mean, the per-class table shows the trade the "
        "class-weighted loss makes: Natural Vegetation, the majority class, lost a quarter of its IoU (0.2494 → 0.1884) while Fallow/Idle Cropland gained (0.3229 → "
        "0.3784); the loss counts rare classes up to 9× more than common ones, so it buys the rare classes with the common one. That is the claim: the "
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
        "## Optional experiment: Predict → Change → Run → Observe → Explain\n\n"
        "None of this affects the default path, and every adaptation starts from the pinned base — including after a `head+last_block` run, whose encoder block "
        "is put back before the next run — so each run is a fresh experiment rather than continued training. **Scope of a re-run:** change the form field in "
        "Section 6, then run Sections 6, 7 and 8 in that order (Section 5 need not be re-run; if you do, it restores the base and prints `restored_pinned_base`). "
        "Pick one:\n\n"
        "1. **Learning rate.** *Predict:* with `LEARNING_RATE = 1e-4` (ten times the default) on a checkpoint selected on this very split, will any epoch beat the "
        "frozen model's class-weighted validation loss (0.9368 in the record), or will epoch 0 be kept? *Change* the field, *run* 6–8, *observe* `best_epoch`, the "
        "per-epoch `val_loss` and `val_mean_iou`, and the `per_class_iou_changes`, and *explain* the result in terms of what a larger step does to a head near a "
        "minimum. There is no recorded outcome for this setting; your run is the evidence.\n"
        "2. **Scope.** *Predict:* does `TRAINABLE = 'head+last_block'` (7.1 M more parameters) move the validation mean IoU with the loss, or against it as the "
        "head-only run did? *Observe* `trainable_parameters`, `gpu_peak_gb`, the per-epoch rows and the artifact's tensor count (now more than 5).\n"
        "3. **Epochs.** *Predict:* with `EPOCHS = 10`, does the validation loss keep falling while the validation mean IoU keeps slipping? *Observe* the kept epoch "
        "and the per-class changes.\n"
        "4. **Your own chips.** Set `USE_BYOD = True` with `BYOD_PATH` (a `group` column keeps one field in one role) and re-run from Section 4: read the "
        "majority-class baseline and the per-class IoU before the adapted number, and compare the frozen column — the packaged model — with the adapted one.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 3 stops during the pickle audit or '
        'conversion** — the audit refuses any global outside the allow-list and names it; the downloaded checkpoint is not the pinned one. Delete the snapshot '
        "folder and run Section 3 again.\n- **Section 4 stops on the tarball's size or SHA-256** — the 1.18 GB download was cut short or altered. Run Section 4 "
        'again; chips already extracted and verified are reused. If the digest still fails, delete `weights/multi-temporal-crop/` and run it once more.\n- '
        "**CUDA out of memory in Section 6** — another notebook holds the GPU, or `TRAINABLE = 'head+last_block'` with a larger `BATCH_SIZE` exceeds a T4. "
        'Restart the session, keep `BATCH_SIZE = 4`, and choose **Run all**.\n- **Section 5 prints `restored_pinned_base`** — you re-ran it after Section 6; the '
        'adapted head was put back to the base. Re-run Sections 6–8 in order.\n- **BYOD: a band, date, shape or label refusal** — the message names the rule; '
        'chips must be 18-band 224 × 224 GeoTIFFs (three dates × six bands, date-major, digital numbers) with masks of 0 = no data and 1..13 = class, and the '
        'dataset needs at least two classes.\n- **BYOD: "… bring at least 7 chips"** — the seeded split takes 25 % for test and 20 % for validation and must leave 4 '
        'training chips, so 7 distinct labelled chips is the minimum (more if a `group` column puts many chips in one group).\n- **BYOD: "pairs.csv row … is '
        'listed but not in the zip"** or **"… is not a readable GeoTIFF"** — the message names the row and the file; fix the name in `pairs.csv` or replace the '
        'file. The sample pair written by Section 4 shows the expected files.\n- **BYOD: "chips have no \'group\'"** — either give every row a `group` value (a field, '
        'scene or region id) or leave the column out.\n- **Section 8 shows no strips** — they are also written as `outputs/{stem}_maps_<chip>.png`; on Jupyter make sure the '
        'notebook is trusted.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- '
        '**BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip (or folder) in the runtime and set `BYOD_PATH` to its path.\n- '
        '**BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n\n## Glossary\n\n- **HLS '
        "(Harmonized Landsat Sentinel-2):** NASA's surface-reflectance product on one 30 m grid; the six bands here are blue, green, red, narrow NIR, SWIR 1 "
        'and SWIR 2.\n- **Multi-temporal chip:** the same 224 × 224 area at three dates of one growing season; crops are told apart by how they change over the season.\n- '
        "**Digital numbers:** reflectance × 10 000 stored as integers, the scale the model's band statistics expect.\n- **Cropland Data Layer (CDL):** the "
        "USDA's annual crop map, from which the 13 labels were derived; it has its own errors at field edges.\n- **Class map / ignore class:** one class per "
        'pixel; −1 (0 in the GeoTIFFs) marks no-data pixels that are excluded from training and scoring.\n- **Majority-class baseline:** naming every pixel with '
        'the most frequent class; the best any constant map can do.\n- **Per-class IoU, mean IoU, mean class accuracy:** overlap of predicted and true pixels '
        'for one class; its average over the classes present; the average per-class recall.\n- **Class weights:** the upstream loss counts rare classes up to 9× '
        'more than common ones.\n- **Encoder, neck, head:** the temporal ViT that turns patches into features; the transposed convolutions that fold the dates '
        'and upsample; the FCN layer that names each pixel.\n- **Frozen / adapted / pinned base:** the packaged model; the model after Section 6; the verified '
        'packaged weights every adaptation starts from (`restore_base`).\n- **Block split / group split:** roles assigned per 4 × 4-chip block so chips of one block never '
        'straddle training and test; for your own chips the `group` column of `pairs.csv` does the same for a field, scene or region.\n- **Class-weighted loss trade:** a loss that '
        'counts rare classes more can lower itself by giving up the common class; the mean IoU barely moves while the per-class row does.\n- **Pickle audit / safetensors:** the upstream checkpoint is a pickle that could run code when loaded; it is statically '
        'checked against an allow-list and converted once to safetensors, a format that stores only tensors.\n- **Isolated environment:** the separate Python '
        'environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\nComplete these in your own words; the '
        "recorded run's values are in the **Check your reasoning** answers above.\n\n- The frozen model's test mean IoU was ___ against the majority-class "
        "baseline's ___; the easiest and hardest classes were ___ and ___.\n- Head fine-tuning lowered the validation loss but moved the test mean IoU to ___, and the class that moved most was ___, "
        'which I read as ___.\n- The number I would not trust on its own is ___, because ___.\n- Before adapting on my own chips I would check the band and date '
        'order, split by ___, and read ___ first.\n\n'
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
