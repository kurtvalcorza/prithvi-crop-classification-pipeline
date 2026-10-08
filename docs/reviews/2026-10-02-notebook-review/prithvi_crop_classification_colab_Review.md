# Prithvi-EO-1.0 Multi-Temporal Crop Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/prithvi-crop-classification-pipeline`  
**Notebook:** `tutorials/prithvi_crop_classification_colab.ipynb`  
**Reviewed commit:** `95f678fc0d5be02b4a6e40a3a90fe72a084c50e1` (`main`, confirmed with `gh api repos/kurtvalcorza/prithvi-crop-classification-pipeline/commits/main`)  
**Notebook Git blob:** `7f88cfa0cc55dd54d54b8bb41a97da4dceaa44a0`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-20 (commit `0f1284d`); the only later change to `src/`, `tools/` or `tutorials/` (`f6525ca`) adds a weight-facts check to the validator and does not touch the notebook or the carried modules.  
**Finding prefix:** `CR`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` `b1cfe13`. The notebook declares 2.0.

## Executive assessment

The default path is carefully engineered and honestly framed. The notebook statically audits the pickled mmsegmentation checkpoint against a seven-global allow-list, unpickles it once through torch's weights-only loader with the three `meta` globals bound to inert stand-ins, converts it into a digest-pinned safetensors file, rebuilds the network from carried plain-PyTorch code, streams exactly 120 digest-pinned members out of a digest-pinned 1.18 GB tarball without `extractall`, assigns roles by 4 × 4-chip block, scores every model number against a majority-class baseline on the same pixels, says plainly that the checkpoint was selected on the very split these chips come from, selects the adapted epoch on validation loss with the frozen model as epoch 0, and reloads the exported adapter with exact parity. Section 6's prose matches the record (class-weighted validation loss 0.9368 → 0.9114 while validation mean IoU slipped 0.4609 → 0.4451).

| Measure | This review (CPU, stub model, synthetic chips) | Kaggle T4 record (blob `7f88cfa0`) |
|---|---|---|
| Code cells completed | default-path cells 15–23 executed against the repository's stub model: 5/5 ok | 11/11 on pass 2; pass 1 stopped at the install guard |
| Test mean IoU, baseline / frozen / adapted | not meaningful on a stub | 0.0105 / 0.4379 / 0.4302 (accuracy 0.137 / 0.5905 / 0.5813) |
| Natural Vegetation (majority class) IoU, frozen → adapted | — | **0.2494 → 0.1884** (not mentioned in the interpretation) |
| BYOD path, `USE_BYOD = True` as documented | **crashes** in Section 5 and Section 8 with `KeyError: 'source_id'` (the actual cells, executed) | not run |
| BYOD smallest dataset accepted | **7 chips** (stated minimum of 4 refused; 5 and 6 refused too) | not run |
| Model state when Section 4 is re-run | **not reset**: Section 5 "frozen" = the adapted model; Section 6 epoch 0 labelled "frozen model" | not run |
| `TRAINABLE = 'head+last_block'`, then back to `'head'` | Section 8 reload-parity **assertion fails** (adapter omits the still-modified last block) | not run |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CR-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.2, `numpy` 2.0.2 → 2.5.3) and passed only after a restart. `docs/release-verification.md` calls the restart "expected", and the repository marks the blob `Release-grade` on that run.
2. **The documented BYOD path cannot finish (CR-M2).** Sections 5 and 8 index `record['source_id']` (and `record['region']`), which only the sample loader sets. BYOD records carry `id`, `image`, `label`, so the frozen per-chip table, the class maps, the artifact export, the reload and `result.json` never happen for the learner's own data.
3. **Re-running without a reset mislabels the model (CR-M3).** `adapt()` trains from whatever state the model is in and always labels epoch 0 "frozen model"; nothing reloads the base. Re-running from Section 4 (the documented BYOD route) makes Section 5's "frozen" numbers and Section 7's frozen column the sample-adapted model, and the optional `head+last_block` experiment leaves a modified encoder block in memory that a later `head` run neither resets nor exports.
4. **Guided layer largely absent (CR-M4).** The notebook is declared `GUIDED` but has no audience statement, how-to-use section, roadmap, glossary, prediction, checkpoint, troubleshooting or conclusion template; 2,356 carried-module lines sit in four unlabelled, uncollapsed cells.

CR-M1, CR-M3 and CR-M4 match the sibling `prithvi-burnscar-segmentation-pipeline` review (BS-M1/M2/M3), which uses the same template family; CR-M2 was not reported there; the burn-scar notebook's Section 5 also indexes `record['source_id']`, so it may share the defect (not checked in this review).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. The Prerequisites (cell 1, "Knowledge") assume the reader knows multispectral surface reflectance (bands, dates, digital numbers, no-data), pixel-wise class maps with an ignore class, and how per-class IoU and mean IoU are read against a majority baseline |
| Supported runtime | "a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12)"; about 4.5 GB of disk; "about 4 GB of GPU memory" for the default adaptation |
| Promised outcomes | Pinned install; carried package (4 modules); 3-file snapshot staged and digest-verified; pickle audited and converted once; 120 pinned members extracted, 60 chips validated and assigned 36 / 12 / 12 by block with three refusals; frozen model vs majority-class baseline; bounded fine-tuning of the FCN head (5.3 M parameters); paired held-out comparison; class maps of two chips beside reference masks; safetensors adapter export and fresh reload with parity; BYOD zip through "the same contract — validation, frozen baseline, adaptation, held-out evaluation, class maps, artifact export and reload parity" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `9b25a261` |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code). Cells 5, 7, 9 and 11 carry `metrics.py` (87 lines), `modeling.py` (285), `pipeline.py` (1,053) and `samples.py` (931). Also read: `pipeline.py` (`_check_record`, `validate_dataset`, `adapt`, `save_artifact`, `load_artifact`, `from_artifact`), `samples.py` (`SAMPLE_RECORDS`, `fetch_corpus`, `read_corpus`, `split_dataset`, `load_byod_dataset`, `write_sample_pair`, `write_dataset_csv`, `dataset_manifest`), `README.md`, `tutorials/README.md`, `docs/release-verification.md`, `STATUS.md`, `tests/test_adaptation.py`, `tests/conftest.py`. The repository has no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md`, row 2026-09-20 (`0f1284d` / `7f88cfa0`), and the workspace archive it cites (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-prithvi-crop-classification/v2/evidence/`: `run_summary.json`, `executed.ipynb`, `executed-pass1.ipynb`, `outputs/`). Kaggle Tesla T4, **the reviewed blob** (`fetched_blob_verified: true`), clean Hugging Face cache. Pass 1 (04:28:42Z, 158.4 s) raised `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` Pass 2 (04:31:21Z, 120.1 s) completed 11/11.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), the shared `eo-notebook-test` conda env (Python 3.12.14, torch 2.13.0+cpu, numpy 2.5.3, tifffile 2026.9.20, safetensors 0.8.0; nothing installed). No Prithvi weights and no tarball were downloaded.
  - **How the cells were run:** the four carried-module cells were executed from the notebook itself; Section 1 ran with the documented `DIMER_NOTEBOOK_CI_PREINSTALLED=1` switch (install skipped); the inline `MANIFEST` of Section 3 was executed but not its download. `PrithviCropPipeline.from_pretrained` was replaced by a factory returning the repository's own stub model (copied from `tests/test_adaptation.py`) and `verify_converted` by a dummy; `fetch_sample_dataset` returned 8 / 4 / 4 synthetic chips (`tests/conftest.py`) carrying `source_id` and `region` like the real loader; `google.colab.files.upload` returned a synthetic 8-chip BYOD zip. Cells 15, 17, 19, 21 and 23 were then executed verbatim, except `EPOCHS = 2` and `LEARNING_RATE = 1e-2` (so the stub moves) and the form substitutions named per probe.
  - **Probes (about 86 s in total):**
    - P1: notebook parse, every code cell compiles, guided-layer markers, display calls, tensor-count and threshold claims, `source_id` uses.
    - P2: `tools/build_notebook.py --check` (OK), `tools/validate_release_assets.py` (PASS), offline suite with `PYTHONPATH=src` (exit 0, 43 passed, 1 skipped: `test_model_backed.py`, weights not staged).
    - P3: block geometry of the 60 pinned chips (8-neighbour adjacency of validation/test blocks to training blocks).
    - P4: the carried `load_byod_dataset` + `split_dataset(seed=0)`, exactly as Section 4 calls them, on 8 synthetic zips.
    - P5: default cells 15–23 on the stub (5/5 ok), then the documented BYOD route (`USE_BYOD = True`, re-run from Section 4).
    - P6: re-run from Section 4 on the sample after a complete run; compare Section 5 "frozen" and Section 6 epoch 0 with the true frozen model.
    - P7: Section 6–8 with `TRAINABLE = 'head+last_block'`, then again with `'head'`.
    - P8: the exported `…_sample_pairs.csv` names vs the extracted member names and the written pair.
- **Not verified:** any notebook cell end to end on real weights here; any Colab run (the stated runtime has no record); the real upload dialog; the size of the effects of CR-M3 on the real network; the optional experiments on real weights; the "about 4 GB of GPU memory" figure.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| **Technical correctness** | Strong on the default path: provenance, audit, conversion, extraction, block split, metrics, epoch selection and reload are sound and recorded. Three state/branch defects: the install needs a restart (CR-M1), the BYOD branch crashes on a sample-only key (CR-M2), and re-runs adapt from an already-adapted model while labelling it frozen, which can also break reload parity (CR-M3). BYOD error handling is partly unactionable (CR-m1). |
| **Promise fulfilment** | The default promises are delivered on the recorded run. The BYOD promise ("flow through the same contract … class maps, artifact export and reload parity") is not delivered (CR-M2), and once it runs it would compare two adapted models (CR-M3). |
| **Learner experience** | The prose explains each stage and gives "Look for" notes with recorded values, but there is no guided layer (CR-M4), no chip, mask or class map is ever shown in a segmentation tutorial (CR-m4), and the interpretation says the mean IoU stays where it was while the majority class lost a quarter of its IoU (CR-m2). |
| **Spec conformance** | Open `MUST`s: RUN1, RUN10, ENV6, REL2/REL11 (CR-M1); DAT10, DAT14, REL12 (CR-M2, CR-M3); DAT12, DAT19 (CR-m1); SPL5 (CR-m6, BYOD); UNC4 (CR-m5); OUT8 accuracy of the stated scope (CR-m3, prose only). Open `SHOULD`s: GDL1–GDL14, UX8 (CR-M4); UX3, UX11 (CR-m4); UX10 (CR-m1); UNC3 (CR-m5); SPL10 (CR-m6); EXE5 (CR-S3). |

## 3. Findings

### CR-M1 — Major: `Run all` needs a manual restart after the install cell

**Location:** Section 1, install cell (cell 3); `docs/release-verification.md` procedure step 4 and Current status; `STATUS.md`; generator `tools/notebook_template.py` (install cell).

**Observed issue:** The install cell `pip install`s the five pins into the running kernel and raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')` whenever a pin replaced an already-imported distribution. In a fresh Kaggle T4 image that always happens (`numpy` 2.0.2 preloaded vs pin 2.5.3; `cuda-bindings` 12.9.4 vs 13.4.2 pulled in by `torch==2.14.0`).

**Consequence:** A learner who chooses `Run all` stops at cell 3 after about 2.5 minutes of installation and must restart and run again. The release record treats this as expected ("an interpreter restart after the install is expected") and marks the blob `Release-grade` on a two-pass run, which RUN1, RUN10 and ENV6 forbid.

**Evidence:** Documented execution evidence: `run_summary.json` passes `[{attempt 1, ok false, 158.4 s}, {attempt 2, ok true, 120.1 s}]`, `restarted_after_install_cell: true`, error text above; `executed-pass1.ipynb`. Source inspection: the guard in cell 3. Release record: "11/11 ok (1 restart after install cell)".

**Recommended correction:** Replace the in-kernel install with the uv isolated-environment pattern: a carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart is needed. Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` (origin/main). Make the change in `tools/notebook_template.py` and regenerate; remove "restart is expected" from the release procedure and return the status to `Candidate` until a one-pass run is recorded.

**Acceptance check:** On a fresh Colab or Kaggle GPU runtime, `Run all` on the regenerated blob completes every code cell in one pass with no error output and no restart; the release record shows one pass for that blob.

**Spec:** RUN1, RUN10, ENV6, REL2, REL11.

### CR-M2 — Major: the BYOD branch crashes in Section 5 and Section 8 (`KeyError: 'source_id'`)

**Location:** Section 5 (cell 17) per-chip loop: `print({'chip': record['source_id'], 'block': record['region'], …})`; Section 8 (cell 23): `'…_map_adapted_' + record['source_id'] + '.tif'` and two more uses; `load_byod_dataset` / `split_dataset` in `samples.py`; generator `tools/notebook_template.py`.

**Observed issue:** Only the sample loader (`read_corpus`) sets `source_id` and `region`. `load_byod_dataset` returns `{id, image, label}` and `_check_record` copies `source_id`/`region` only when present, so in BYOD mode Section 5 raises `KeyError: 'source_id'` after the frozen metrics print, and Section 8 raises it before writing any class map. The artifact is never saved, the reload parity is never checked, `predictions.json` and `result.json` are never written.

**Consequence:** The opening cell promises that BYOD chips "flow through the same contract — validation, frozen baseline, adaptation, held-out evaluation, class maps, artifact export and reload parity". A learner who follows the BYOD instruction gets two tracebacks and no exported adapter for their own data, i.e. no reusable result. The default sample path is unaffected.

**Evidence:** Direct execution (P5b): the actual cells 15 → 23 with `USE_BYOD = True` and a valid synthetic 8-chip zip: cell 15 ok, cell 17 `KeyError: 'source_id'`, cells 19 and 21 ok, cell 23 `KeyError: 'source_id'` with no output written. P4: BYOD record keys are `['id', 'image', 'label']`. P1: `source_id` is indexed once in cell 17 and three times in cell 23. The offline tests never run the tutorial cells on BYOD records.

**Recommended correction:** In the template, use `record.get('source_id', record['id'])` and `record.get('region', '—')` (or have `load_byod_dataset` set `source_id = id` and an optional `region` column); add a parity/unit test that executes the Section 5 and Section 8 cell bodies on `load_byod_dataset` output with a stub pipeline (the repository already has the stub model in `tests/test_adaptation.py`).

**Acceptance check:** With a stub pipeline and a valid 8-chip BYOD zip, executing cells 15–23 with `USE_BYOD = True` completes without error and writes the adapter, `predictions.json` and `result.json`; a hosted BYOD run (REL12) reaches the reload-parity print.

**Spec:** DAT10, DAT14, REL12.

### CR-M3 — Major: re-runs adapt from the already-adapted model and label it "frozen"; a scope change can break the export

**Location:** Opening cell ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); Section 4 (cell 15); Section 5 (cell 17); Section 6 (cell 19) and `PrithviCropPipeline.adapt` in `pipeline.py` (epoch 0 note `"frozen model"`); Section 7 markdown ("the gap between frozen and adapted is the number to watch"); Section 8 (cell 23) and `save_artifact`; Interpretation "Optional experiments".

**Observed issue:** `adapt()` trains from the current tensors and keeps the best epoch in place; `initial_state` is restored only on an exception, and no cell reloads the base model. Two consequences:
(a) After a complete run, re-running from Section 4 (the documented BYOD route, once CR-M2 is fixed, or a plain rerun) scores the adapted model in Section 5 as `frozen_test` / `frozen_val`, starts Section 6 from it and labels its epoch 0 "frozen model", and reports Section 7's `frozen` column and the evaluation report's `frozen` block from it.
(b) The optional experiment `TRAINABLE = 'head+last_block'` changes encoder block 5. A later `'head'` run (for example the next optional experiment, "raise `EPOCHS`") leaves that block modified in memory, but `save_artifact` writes only the head tensors, so the exported adapter does not reproduce the evaluated model and Section 8's reload-parity assertion can fail.

**Consequence:** On the learner's own chips the "frozen versus adapted" comparison the notebook tells them to watch is between two adapted models, and an exploring learner can end with an `AssertionError` or, below the tolerance, an adapter that silently differs from what they evaluated.

**Evidence:** Source inspection of `adapt()` and `save_artifact()`. Direct execution on the stub model with the actual cells: (P6) after a complete run, re-running Sections 4–6 gives Section 5 frozen test mean IoU 0.0088 (= the adapted value; true frozen 0.0059) and epoch 0 `{'val_loss': 2.3872, 'note': 'frozen model'}` (= the first run's kept epoch; true frozen 2.5649). (P7) `head+last_block` then `head`: block-5 weight stays at 1.0244 after the head run, the artifact lists only `decode_head.bias`, `decode_head.weight`, and cell 23 fails `assert parity['mean_iou_diff'] < 1e-3` (`mean_iou_diff 0.0011`, `metrics_identical False`). The stub's magnitudes are not the real network's; the mechanism is.

**Recommended correction:** Snapshot the base state right after `from_pretrained` and restore it at the start of Section 4 (or give `adapt()` a `reset_to_base=True` default that reloads the trainable *and* previously-trained tensors from the verified safetensors); make epoch 0's note "frozen base" only when the tensors equal the base. Alternatively instruct "Runtime → Restart and run all" with the changed form values, and state the rerun scope for each optional experiment. Fix in `pipeline.py` and `tools/notebook_template.py`, then regenerate.

**Acceptance check:** (1) After a complete default run, re-running from Section 4 gives Section 5 frozen metrics identical to a fresh run and Section 6 epoch 0 validation loss equal to the fresh run's epoch 0. (2) Running Section 6–8 with `head+last_block` and then with `head` passes the reload-parity assertion, and the second artifact reproduces the evaluated model.

**Spec:** DAT14, REL12 (GDL10 for the rerun instruction).

### CR-M4 — Major (learner-facing): the declared `GUIDED` layer is largely absent

**Location:** Whole notebook; carried module cells 5, 7, 9, 11; generator `tools/notebook_template.py`.

**Observed issue:** The notebook declares mode `GUIDED` but has no intended-audience statement (GDL1), no **How to use this notebook** (GDL2), no roadmap (GDL3), no Input → Model → Output task contract near the opening (GDL4; the Prerequisites' data contract is the closest), no glossary although IoU, mean class accuracy, ignore class, HLS, digital numbers, CDL, transposed-convolution neck, FCN head, BatchNorm statistics, float16 autocast, GradScaler and safetensors all appear (GDL6), no prediction before any principal result (GDL7), no interpretation checkpoints or sample answers (GDL9), no Predict → Change → Run → Observe → Explain activity (GDL10; the optional experiments are one sentence), no Infrastructure labelling or collapsed carrier cells (GDL11; 2,356 lines in four cells, `cellView` unset on all 11 code cells), no troubleshooting (GDL13) and no conclusion template (GDL14). Sections end without a synthesis (UX8). "Look for" notes exist for Sections 1, 4, 5 and 6 (GDL8 partly met).

**Consequence:** A self-paced learner meets 2,356 lines of carrier code before any lesson, gets no help separating essentials from infrastructure, and is never asked to predict, check or explain anything, so "read per-class IoU … against a majority-class baseline" and "compare the adapted and frozen models" are exercised only by reading printed dictionaries.

**Evidence:** Source inspection; P1 markers (`How to use`, `Glossary`, `Troubleshoot`, `Roadmap`, `Infrastructure`, `Check your reasoning`, `audience`, `What to notice` all absent; the only "predict" hits are the `pipe.predict` API name).

**Recommended correction:** Add the NOTEBOOK_SPEC 2.2 guided layer in `tools/notebook_template.py`: audience, how-to-use, roadmap, task contract, collapsible glossary, a prediction before Sections 5 and 7 (for example "will a head fine-tuned on 36 chips beat a checkpoint already selected on this split?"), "What to notice" notes, collapsible "Check your reasoning" answers, one bounded PCROE activity (for example `TRAINABLE = 'head+last_block'` with its rerun scope), `# @title Infrastructure: …` with `cellView: form` on the carrier cells, a troubleshooting section (install/restart, Hub download, disk, GPU memory, BYOD errors) and a conclusion template.

**Acceptance check:** The GDL1–GDL14 checklist in NOTEBOOK_SPEC 2.2 passes item by item on the regenerated notebook, and the four carrier cells open collapsed in Colab.

**Spec:** GDL1–GDL14, UX8.

### CR-m1 — Minor: BYOD stated minimum is wrong, several failures are not actionable, and the refusal probes misfire on small BYOD sets

**Location:** Opening cell ("at least four chips with at least two classes"); Section 4 (cell 15) upload branch and refusal probes; `split_dataset` and `load_byod_dataset` in `samples.py`.

**Observed issue:** `split_dataset(seed=0)` takes 25 % for test and 20 % for validation and requires 4 training chips, so 4, 5 and 6 chips are refused and 7 is the true minimum. A `pairs.csv` row naming a file missing from the zip raises a bare `KeyError: 'c3.tif'`; a non-TIFF file raises `TiffFileError: not a TIFF file`; cancelling the upload dialog raises `StopIteration` from `next(iter(uploaded.items()))` (source). In BYOD mode the three refusal probes build their lists from `test_records[1:4]`; with fewer than 4 test chips (any BYOD set under 14 chips) all three are rejected with `2 records; 4..2000 are required` instead of the shape, label or range check they are meant to show.

**Consequence:** A learner following the stated contract with 4–6 chips is refused after uploading; three common mistakes give errors that do not name the contract or the fix; and the refusal demonstration silently stops demonstrating anything while still printing "rejected".

**Evidence:** Direct execution (P4): `valid_n4/5/6` → `ValueError: split leaves 2/3/3 training chips; at least 4 are required`; `valid_n7` → 2 / 1 / 4; `missing_member` → `KeyError`; `non_tiff_image` → `TiffFileError`; `no_pairs_csv` → clear `ValueError`. P5b: the three probe lines in BYOD mode. Source inspection for the cancel path.

**Recommended correction:** State "at least 7 labelled chips" (or derive and print the minimum); wrap member lookup and TIFF decode in `load_byod_dataset` with `ValueError`s naming the row id, the file and the expected format; handle an empty upload; build the probes from `validate_dataset`'s own minimum (e.g. pad with training records) so each probe fails on its intended check.

**Acceptance check:** A 7-chip zip is accepted and a 6-chip zip is refused with a message giving the minimum; a missing member, a non-TIFF image and a cancelled upload each produce a `ValueError` naming the row/file and the corrective action; with a 7-chip BYOD zip each probe's rejection message names shape, label value or range respectively.

**Spec:** DAT12, DAT19, UX10.

### CR-m2 — Minor: the interpretation understates what the adaptation changed and pre-states an experiment's result

**Location:** Interpretation (cell 24); Section 7 markdown (cell 20); Prerequisites (cell 1).

**Observed issue:** (a) The interpretation says the head fine-tuning "leaves the mean IoU where it was". In the record, test mean IoU fell 0.4379 → 0.4302, validation mean IoU 0.4609 → 0.4451, and the majority class, Natural Vegetation, lost a quarter of its IoU (0.2494 → 0.1884) while Fallow/Idle Cropland gained (0.3229 → 0.3784). The same section then advises "read … the per-class IoU first", but never reads this one. (b) "try `LEARNING_RATE = 1e-4` to see the frozen model win every epoch" states a result with no record, and on a rerun epoch 0 is no longer the frozen model (CR-M3). (c) The optional experiments give no rerun scope. (d) The data contract reads `{{id, image, label}}` (template brace escape leaked into markdown).

**Consequence:** The learner is told nothing moved while the per-class table shows the class-weighted loss trading the most frequent class for rarer ones — the exact point the notebook says the per-class IoU is for — and is promised an experimental outcome that is not established.

**Evidence:** Documented execution evidence (cell 21 output: `per_class_iou_changes`); source inspection; P1 (`brace_typo_present: true`).

**Recommended correction:** Name the per-class trade (Natural Vegetation down, Fallow up) and connect it to the upstream class weights; phrase the learning-rate experiment as a prediction to test; state the rerun scope for each experiment; fix the brace escape in the template.

**Acceptance check:** The interpretation mentions the mean-IoU decrease and the largest per-class changes; no optional experiment states its outcome as fact; each experiment names the cells to rerun; the rendered markdown shows `{id, image, label}`.

**Spec:** GDL8, GDL10, GDL14.

### CR-m3 — Minor: "8 tensors" is wrong; the head adapter has 5

**Location:** Section 6 markdown ("`pipe.adapt` trains the 8 tensors of the FCN head"); `docs/release-verification.md` procedure step 5 ("8 tensors, about 21 MB") and the pre-flight row ("adapter 21,249,580 bytes (8 tensors)").

**Observed issue:** `_trainable('head')` selects the parameters under `decode_head.`, and `save_artifact` writes exactly those. The recorded run prints `'trainable': 'head', 'tensors': 5` for the 21,249,580-byte adapter. Eight is the head's state-dict entry count including the BatchNorm buffers, which are deliberately not trained.

**Consequence:** The learner reads that 8 tensors are trained and sees 5 exported, in the section that also says BatchNorm statistics are frozen; the release procedure checks for a number the run never prints.

**Evidence:** Documented execution evidence (cell 23 output); source inspection (`_trainable`, `save_artifact`).

**Recommended correction:** Say "the 5 trainable tensors of the FCN head (its BatchNorm running statistics stay frozen)" in the template and the release procedure.

**Acceptance check:** The Section 6 markdown and `docs/release-verification.md` state the same tensor count the artifact print reports.

**Spec:** OUT8 (accuracy of the stated adaptation scope).

### CR-m4 — Minor: a segmentation tutorial that never shows a chip or a class map

**Location:** Sections 4, 5, 7 and 8.

**Observed issue:** No code cell displays a chip composite, a label, a predicted class map or an error map. Class maps are written to GeoTIFFs "so they can be opened side by side", outside the notebook.

**Consequence:** The learner cannot see what the 13 classes look like, where the model confuses Natural Vegetation with neighbouring classes, or what the adaptation changed spatially, so per-class IoU stays abstract.

**Evidence:** Source inspection; P1 (`image_display_calls: 0`).

**Recommended correction:** Add one figure in Section 8: a false-colour composite of the middle date, the reference mask, and the frozen and adapted class maps for the two written chips, with a shared 13-class legend.

**Acceptance check:** Running the default path displays at least one composite + reference + prediction figure inline.

**Spec:** UX3, UX11.

### CR-m5 — Minor: decision-rule ownership is not stated, though the docs say it is

**Location:** Section 5 markdown (cell 16); `tutorials/README.md` conformance note "Score semantics (UNC1–UNC4)".

**Observed issue:** The notebook says the softmax scores are "not calibrated probabilities" and prints `decision_rule: argmax over the 13 class scores (no threshold)`, but never says that a minimum-confidence rule, post-processing or calibration belongs to the deployment. `tutorials/README.md` claims "the notebook says any minimum-confidence rule or post-processing is the deployment's to set".

**Consequence:** A learner reusing the class maps operationally is not told that argmax is a default, not a tuned operating point; the conformance note overstates the notebook.

**Evidence:** Source inspection; P1 (`threshold` and `deployment` absent from all markdown).

**Recommended correction:** Add one sentence to Section 5: the decision rule is argmax; a deployment chooses any minimum-confidence rule or post-processing on its own validation data and owns calibration.

**Acceptance check:** Section 5 markdown states the default rule and the deployment's ownership; the README note then matches.

**Spec:** UNC3, UNC4.

### CR-m6 — Minor: the notebook's own split advice cannot be followed in BYOD

**Location:** `split_dataset` (`samples.py`), Section 4 BYOD branch; Interpretation ("Split by region, not by chip").

**Observed issue:** The BYOD path is a seeded shuffle by chip; `pairs.csv` has no group or split column, so a learner cannot apply the notebook's central split advice inside the notebook (the `split_dataset` docstring says "group them yourself", but nothing in the notebook says how). On the sample the block split is stated honestly ("neighbouring blocks may" straddle roles); by the review's count 1 of 12 test blocks and 1 of 12 validation blocks touch a training block.

**Consequence:** BYOD held-out numbers on chips cut from one field or scene will look better than they are — the failure the interpretation warns about.

**Evidence:** Source inspection (`split_dataset`, `load_byod_dataset` read only `id,image,label`); direct execution (P3).

**Recommended correction:** Accept an optional `group` (or `split`) column in `pairs.csv` and split by group when present, warning when absent; say so in Section 4.

**Acceptance check:** A BYOD zip with a `group` column never puts one group in two roles, and Section 4 documents the column.

**Spec:** SPL5, SPL10.

### Suggestions

- **CR-S1 — Regenerate against NOTEBOOK_SPEC 2.2.** The notebook, `metadata.dimer` and the docs declare 2.0. Acceptance: metadata and opening cell declare 2.2 and the validator checks 2.2.
- **CR-S2 — Record a Colab run.** Only Kaggle T4 runs exist although Colab is the stated runtime. Acceptance: a Colab row for the regenerated blob in `docs/release-verification.md`.
- **CR-S3 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`.** Cell 3 reads it silently. Acceptance: a sentence in Section 1 names it and says it only skips the install. (EXE5)
- **CR-S4 — Record a generating revision on `main`.** `metadata.dimer.generated_from.revision` and `NOTEBOOK_SOURCE.repository_revision` are `9b25a261`, a PR #1 branch commit that is not an ancestor of `main` (squash merge); it is reachable on GitHub and its four module files are byte-identical to `main`, so nothing is lost today, but a branch clean-up would orphan the provenance link. Acceptance: on regeneration the recorded revision is a `main` commit.
- **CR-S5 — Show per-chip dispersion.** The interpretation rightly says pooled metrics let large fields dominate; printing per-chip mean IoU (and its range) for frozen and adapted lets the learner see it. Acceptance: Section 5/7 print a per-chip column. (EVAL6, ENV8)
- **CR-S6 — Back the GPU-memory figure.** Print `torch.cuda.max_memory_allocated()` after adaptation so "about 4 GB" is evidenced in each run. Acceptance: the value appears in the adapt output and the result JSON.

## 4. Readiness

**Needs revision.** No Blocker. Four Majors are open (CR-M1 one-pass `Run all`; CR-M2 BYOD crash; CR-M3 rerun state; CR-M4 guided layer), and the open `MUST`s RUN1, RUN10, ENV6, REL2/REL11, DAT10, DAT12, DAT14, DAT19, SPL5, UNC4 and REL12 fail release under the spec regardless of severity. The repository's `Release-grade` status rests on a two-pass run and should return to `Candidate`. Remaining gates after fixes: a one-pass hosted run of the regenerated blob (Colab preferred, CR-S2), and a hosted BYOD run that reaches export and reload parity (REL12).

## 5. Verified vs inferred

- **Verified by direct execution (CPU, stub model, synthetic chips; the notebook's own cells):** generator parity, validator PASS, 43 offline tests passed / 1 skipped; default cells 15–23 wire up end to end on the stub; BYOD `KeyError: 'source_id'` in Sections 5 and 8; BYOD minimum of 7, unactionable `KeyError`/`TiffFileError`, misfiring refusal probes; Section 5 "frozen" and epoch 0 "frozen model" equal to the adapted model after a rerun; reload-parity assertion failing after `head+last_block` → `head`; block adjacency counts; CSV names match the extracted members.
- **Verified from documented execution evidence (Kaggle T4, this blob):** pass-1 restart error; default-path metrics, per-class IoU changes, history, 5-tensor artifact and exact reload parity.
- **Inferred, not executed:** the size of CR-M3's effects on the real network (the stub's magnitudes are not the model's; whether the real parity assertion trips depends on how far block 5 moved); the cancel-upload `StopIteration`; GPU memory; behaviour on Colab.
- **Most likely to be wrong:** CR-M3(b)'s claim that the real reload-parity assertion fails. On the stub the mean-IoU difference was 0.0011 against a 0.001 tolerance — a narrow miss. With the real 7.1 M-parameter block trained at the default `1e-5` for 4 epochs, the drift could stay under tolerance, in which case the outcome is a silently non-reproducing adapter rather than an `AssertionError`. The mechanism (the block is neither reset nor exported) is certain from source either way.
