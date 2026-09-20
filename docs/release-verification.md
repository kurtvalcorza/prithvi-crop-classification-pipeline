# Release verification

`tutorials/prithvi_crop_classification_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`modeling.py`, `metrics.py`, `pipeline.py`, `samples.py`), each equal to its
  source after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest
  and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions; the dataset
  revision `f285bb27…` and the vendored upstream commit `3b6d401f…` are the only other 40-hex commits the documents
  may name;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `PrithviCropPipeline.from_pretrained(weights_dir=..., device=..., report=print)` so the pickle audit and the
  conversion are printed before the model loads, `fetch_sample_dataset` from the pinned cache path,
  `load_byod_dataset`, `dataset_manifest`, `write_sample_pair`, `validate_dataset` with the refusal probes,
  `pipe.evaluate` on the frozen model with the majority-class baseline and after the adaptation with the procedural
  assertions, `pipe.adapt` with its explicit hyperparameters, `pipe.predict` class maps written beside the reference
  masks, `pipe.save_artifact`, `PrithviCropPipeline.from_artifact` and the reload-parity assertion, and the provenance
  fields `served_from_pickle: False`, `remote_code_executed: False`, `meta_globals_bound_to_stand_ins` and the
  `data_tarball` record), the seven expected `outputs/` paths, the learner-facing statements (the asset is a pickle
  unpickled once with the meta globals bound to inert stand-ins, the model was selected on these chips, the
  majority-class baseline, the tarball is streamed with no `extractall`, split by region, CC BY 4.0) and the
  gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import
  on the primary path, a mutable `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` /
  `tarfile.open(` / `Unpickler` / `safe_globals(` / `PrithviCropSegmenter(` / mmcv / mmseg / timm use or
  `torch.load(` / `pickle.load` **outside the carried module cells**, `trust_remote_code=True`, `pickle.load` or
  `torch.load(` without `weights_only=True` anywhere, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub model and the randomly
initialised vendored network, skipped without torch), `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`, `tests/test_model_backed.py` (skipped without the staged converted weights); crafted
pickles, temporary manifests, synthetic chips, a synthetic tarball with a decoy member and an AppleDouble twin and an
injected fetcher, no weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/prithvi-eo-1.0-100m-crop/` or the data cache `weights/multi-temporal-crop/` (the standalone path
   writes the manifest itself, stages all three listed files from the Hub, audits and converts the checkpoint,
   fetches the 1.18 GB tarball from the Hub dataset at its immutable revision, hashes it and streams out the 120
   pinned members, so neither directory may be seeded); the runtime needs about 4.5 GB of free disk for the
   snapshot, the converted file, the tarball and the extracted members;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `EPOCHS = 4`, `LEARNING_RATE = 1e-5`, `BATCH_SIZE = 4`, `TRAINABLE = 'head'`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `tifffile==2026.9.15`, `numpy==2.5.3`, `safetensors==0.8.0`,
   `huggingface-hub==1.32.0` (an interpreter restart after the install is expected where the runtime's preinstalled
   torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the four carried module cells execute (defining `PrithviCropSegmenter`, `PrithviCropPipeline`, `audit_pickle`,
     `restricted_load`, `convert_model`, `build_model`, `verify_snapshot`, `verify_converted`, `stage_missing_files`,
     `validate_inputs`, `validate_dataset`, `read_chip`, `read_mask`, `fetch_tarball`, `extract_pinned_members`,
     `fetch_sample_dataset`, `load_byod_dataset`, `write_sample_pair`, `write_dataset_csv`, `dataset_manifest`,
     `check_split_disjoint`, `chip_block`, `segmentation_metrics`, `majority_baseline` and the constants) with no
     repository import;
   - the model cell writing `weights/prithvi-eo-1.0-100m-crop/dimer-base-manifest.json`, staging the three files
     from the Hub at the pinned revision and `verify_snapshot` reporting 3 verified files;
   - the model cell printing the **conversion record** with the static audit (the four torch globals and the three
     `meta` globals, 0 violations, audit digest `46551363…`), the checkpoint record (epoch 80, iter 30800, mmseg
     0.30.0, mmcv 1.6.2, 112 state-dict tensors, 14 training-only tensors dropped, optimizer discarded) and the
     converted file (`d1df8044…`, 537,722,508 bytes), then the load report on `cuda` with source "converted from
     the manifest-verified source checkpoint";
   - the tarball fetched and hashed (`d6e616cc…`, 1,179,542,384 bytes), the 120 pinned members extracted under
     `weights/multi-temporal-crop/chips/`, the dataset manifest with 36 / 12 / 12 chips and 13 classes present in
     every role, the written sample pair and `outputs/prithvi_crop_classification_sample_pairs.csv`, and three
     refusals (two-date chip, unknown label class, digital numbers out of range);
   - the majority-class baseline and the frozen model on the test chips (on the sample: baseline accuracy ≈ 0.14,
     mean IoU ≈ 0.01; frozen mean IoU ≈ 0.44, accuracy ≈ 0.59) and the validation chips (mean IoU ≈ 0.46);
   - `pipe.adapt` printing epoch 0 as the frozen model, 5,312,269 trainable of 134,427,661 parameters, 36 steps,
     the class-weighted loss, frozen BatchNorm statistics, and a four-epoch history with validation loss ≈ 0.94 →
     ≈ 0.91 at the kept epoch;
   - `pipe.evaluate` on the test chips with the three-way comparison and
     `outputs/prithvi_crop_classification_evaluation_report.json` written (the cell asserts the kept epoch's
     validation loss is no higher than the frozen model's and that the validation mean IoU matches the history
     within 0.01);
   - the class maps of two held-out chips written beside their reference masks with
     `outputs/prithvi_crop_classification_predictions.json`;
   - `pipe.save_artifact` writing `outputs/prithvi_crop_classification_adapter/{adapter.safetensors,manifest.json}`
     (8 tensors, about 21 MB), and `PrithviCropPipeline.from_artifact` reloading it with held-out metrics and score
     maps matching the adapted model (the cell asserts a mean-IoU difference below 10⁻³ and a maximum score
     difference below 10⁻²);
   - `outputs/prithvi_crop_classification_result.json` written with `NOTEBOOK_SOURCE`, the model identity, the
     provenance block (`served_from_pickle: false`, `remote_code_executed: false`, the audit digest, the three meta
     globals bound to stand-ins, the converted digest, the `data_tarball` record), the runtime versions, the
     comparison and the reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the data cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `prithvi_crop_classification_colab.ipynb` | generated, pre-commit | 2026-09-20 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti, `google.colab` shim, pins pre-installed) | PASS 11/11 code cells, 142.1 s — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/prithvi_crop_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/prithvi_crop_classification_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `acb9323` / `9e19fb22` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-prithvi-crop-classification` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 290.5 s | **FAILED** at Section 4 (6/11 code cells; the install, the 3-file snapshot, the pickle audit and conversion and the model load all passed): `validation_chips.tgz: 1179542384 bytes with sha256 d6e616cc…, pinned 1179542384 / 59407373…`. The pinned digest was wrong, not the Hub file: the builder's local copy of the tarball, a download resumed after a machine restart, had the right size but a corrupt gzip stream, and `huggingface_hub` had recorded the correct LFS digest in its cache metadata without verifying the bytes. The archive was re-downloaded, verified against the Hub LFS pointer (`d6e616cc…`) and re-indexed (1,542 real members, not the 787 the corrupt copy exposed), the 120 members were re-pinned and the notebook regenerated. Finding, not evidence |
| 2026-09-20 | generated, pre-commit | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti) | Default sample path (stage → verify → load the already-converted file → tarball hash → pinned-member extraction → validate → refusal probes → majority baseline → frozen evaluation → head adaptation → held-out evaluation → class maps → artifact export → reload parity), `Run all` in a fresh interpreter with the Hub files, the converted safetensors and the tarball pre-staged | 142.1 s | **PASS** — 11/11 code cells; snapshot verified (3 files), the already-converted safetensors loaded on `cuda`; the tarball hashed (`d6e616cc…`) and the 120 pinned members streamed out of it; probes refused (two-date chip, unknown label class 13, digital numbers out of range); majority baseline (Natural Vegetation) accuracy 0.1371 / mean IoU 0.0105; frozen test mean IoU 0.4379, accuracy 0.5905, mean class accuracy 0.6239, validation mean IoU 0.4609; head adaptation 36 steps in 36.7 s, class-weighted validation loss 0.9368 → 0.9114 at the kept epoch 4 (validation mean IoU 0.4609 → 0.4452); adapted test mean IoU 0.4302, accuracy 0.5813, mean class accuracy 0.6250 (Fallow/Idle Cropland 0.323 → 0.379, Wetlands 0.403 → 0.406, Natural Vegetation 0.250 → 0.188, Cotton 0.432 → 0.399); two class maps written beside their reference masks; adapter 21,249,580 bytes (8 tensors); reload parity exact (mean-IoU difference 0.0, maximum score difference 0.0). Pre-flight only, **not** promotion evidence |

## Current status

**Candidate.** The notebook has run top-to-bottom on the local pre-flight harness only. The clean-runtime execution of
the committed notebook blob (REL1/REL10) is pending; when it is recorded here the registry moves to
**Release-grade**. Any later change to the carried modules or to the notebook produces a new blob, and the registry
returns to **Candidate** until a clean run of that blob is recorded here.
