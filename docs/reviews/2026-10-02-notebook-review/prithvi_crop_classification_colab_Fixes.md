# Review fixes: `prithvi_crop_classification_colab.ipynb` (review of 2026-10-02, prefix CR)

Fixes for the Notebook Review Framework v1 review `prithvi_crop_classification_colab_Review.md` (reviewed commit `95f678f`,
review PR #8). CR-M1, CR-M3(a) and CR-M4 were fixed by the 2026-10-05 fleet sweep (commit `efb9941`, record
`docs/reviews/2026-10-05-fleet-sweep/prithvi_crop_classification_colab_Fixes.md`) and are verified here against the review's
acceptance checks; CR-M2, CR-M3(b), the Minors and two trivial Suggestions are fixed in this pass. All changes are made in
the generator (`tools/notebook_template.py`, `tools/validate_release_assets.py`) and the carried modules (`pipeline.py`,
`samples.py`, `metrics.py`); the notebook is regenerated with `tools/build_notebook.py`. Status and release labels are
unchanged.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| CR-M1 (Run all needs a restart) | Fixed (sweep `efb9941`) — hosted confirmation pending | Isolated hash-locked `uv` environment (`build_notebook.py/2.2`): nothing installed into the kernel, no restart requested, lock-keyed reuse, idempotent Section 1. The one-pass hosted run and the release-procedure wording are the maintainer's. | Section 1; generator | `test_swp_r_*` (3); P1 re-run: 12 code cells compile, 7 collapsed infrastructure cells |
| CR-M2 (BYOD crashes on `source_id` in Sections 5 and 8) | Fixed | `chip_name = lambda record: record.get('source_id', record['id'])` and `record.get('region', '—')` in Sections 5 and 8; no tutorial cell indexes `source_id` or `region` any more. | Sections 5, 8 | `test_cr_m2_byod_records_without_source_id_run_sections_4_to_8_and_export`: the notebook's own Section 4–8 cells executed with `USE_BYOD = True` on a 7-chip synthetic zip (records carry only id/image/label) and a NumPy stand-in pipeline — adapter, `predictions.json`, `result.json` and the class maps are written, reload parity 0. P1 re-run: `source_id_uses_in_tutorial_cells` {} |
| CR-M3 (re-runs adapt from the adapted model; a scope change breaks export) | Fixed (sweep `efb9941` for (a); (b) verified here) | (a) `restore_base()`; `adapt()` / `load_artifact()` start from the pinned base; Section 5 restores it first. (b) `_base_state` keeps the pinned value of every tensor any earlier run changed, so a `head` run after `head+last_block` first puts encoder block 5 back; the exported head adapter then is the evaluated model. The closing says so and states the re-run scope. Acceptance on real weights needs the hosted run. | `pipeline.py`; Sections 5–6; closing | `test_swp_f_*` (3), `test_cr_m3b_last_block_from_an_earlier_scope_is_restored_before_a_head_run`. The review's P6/P7 need torch (not a CI dependency) and could not re-run here. |
| CR-M4 (guided layer absent) | Fixed (sweep `efb9941`) + completed here | Sweep: audience, Input → Model → Output, How to use, roadmap, Predict prompts, What to notice / Check your reasoning with the recorded numbers, Troubleshooting, Glossary, Conclusion, infrastructure collapsed. This pass adds the GDL10 **Predict → Change → Run → Observe → Explain** activity with the re-run scope (four bounded experiments) and extends the checkpoints (per-class trade, figure). | opening; closing | `test_swp_g_*` (2), `test_cr_m2_interpretation_and_experiments`; P1 re-run: guided markers 10/10 |
| CR-m1 (BYOD minimum wrong; unactionable failures; probes misfire) | Fixed | `byod_minimum_records()` (7); `split_dataset` refuses a smaller set naming the counts and the minimum; `load_byod_dataset` names the row, file and fix for a missing member or an unreadable TIFF; the cancelled upload was fixed by the sweep; the three refusal probes are padded from the other roles to `MIN_RECORDS`, so each fails on its intended check with a 7-chip BYOD set. Prose says 7. | `samples.py`; Section 4; Troubleshooting | `test_cr_m1_*` (3); in the BYOD end-to-end test the probes report the shape / label / range rule, not "4..2000 are required" |
| CR-m2 (interpretation understates; pre-states an experiment) | Fixed | Interpretation and the Section 7 checkpoint name the mean-IoU decrease and the per-class trade from the record (Natural Vegetation 0.2494 → 0.1884, Fallow/Idle Cropland 0.3229 → 0.3784, Cotton 0.432 → 0.399) and connect it to the class weights; the learning-rate experiment is a prediction with "no recorded outcome"; every experiment names the cells to re-run; the brace escape was fixed by the sweep. | closing; Section 7 markdown | `test_cr_m2_interpretation_and_experiments` |
| CR-m3 ("8 tensors") | Fixed | Section 6 says "the 5 trainable tensors of the FCN head" and why the state dict has 8; the Section 8 checkpoint says 5-tensor; `docs/release-verification.md` (procedure step and the 2026-09-20 row, with a correction note) and the model card's selection record say 5. | Sections 6, 8; docs | `test_cr_m3_tensor_count_is_five_everywhere` |
| CR-m4 (no chip or class map shown) | Fixed — hosted confirmation pending | Section 8 shows one strip per written chip — middle-date SWIR 2 / NIR / red composite, reference (black = no data), frozen map, adapted map — in a fixed 13-colour palette with the legend printed, through `class_map_panel` + `render_png` (a 30-line PNG encoder; matplotlib is not in the lock) and `display(PngImage(...))`, which the isolated runtime forwards; also written as `outputs/<stem>_maps_<chip>.png`. | `samples.py`; Section 8 | `test_cr_m4_panel_and_png_round_trip` (PNG decoded back byte-exact), the BYOD end-to-end test (two strips displayed, 224 × 912 × 3). Inline rendering on Colab is hosted evidence. |
| CR-m5 (decision-rule ownership) | Fixed | Section 5 states the argmax default and that any minimum-confidence rule, threshold or post-processing is the deployment's to set and that it owns calibration; `tutorials/README.md`'s note now matches. | Section 5 markdown | `test_cr_m5_*`; P1 `threshold` / `deployment's` true |
| CR-m6 (split advice cannot be followed in BYOD) | Fixed | Optional `group` column in `pairs.csv` (also set as `region`, so `check_split_disjoint` and the block list see it); `split_dataset(group_key="group")` assigns whole groups to one role, refuses a half-grouped table or fewer than 3 groups; `_check_record` keeps `group`; the exported `pairs.csv` carries the block id as `group`; Section 4 documents it and reports `grouped_split`. | `samples.py`, `pipeline.py`; Section 4 | `test_cr_m6_*` (2): 5 seeds, no group in two roles |
| CR-S1 (declare 2.2) | Not fixed — generator-wide; maintainer decision | — | — | — |
| CR-S2 (Colab run) | Not fixed — needs the hosted run | — | — | — |
| CR-S3 (document `DIMER_NOTEBOOK_CI_PREINSTALLED`) | Fixed (sweep) | Section 1 markdown names it. | Section 1 | source |
| CR-S4 (generating revision on `main`) | Not fixed — the recorded revision is this branch's head until the PR is merged; a regeneration on `main` after the merge records a `main` commit | — | — | — |
| CR-S5 (per-chip dispersion) | Fixed | `per_chip_metrics()`; Section 5 prints mean IoU per chip and the frozen range; Section 7 prints frozen beside adapted per chip and the ranges; stored in the evaluation report and `result.json`. | `metrics.py`; Sections 5, 7 | `test_cr_s5_*`, BYOD end-to-end test |
| CR-S6 (GPU-memory figure) | Fixed — hosted confirmation pending | `gpu_peak_gb` in the adapt output, the evaluation report and `result.json`. | Sections 6, 8 | `test_cr_s6_adapt_cell_records_gpu_peak` |

## User-visible changes

- BYOD works end to end (it crashed in Sections 5 and 8 before); the stated minimum is 7 chips (unchanged behaviour, now stated and refused by name); optional `group` column; a missing or unreadable file names the row and file; the refusal probes pad themselves from the other roles.
- `split_dataset()` gains `group_key="group"`; its refusal message changed. `write_dataset_csv()` writes a `group` column instead of `region`. New `byod_minimum_records()`, `per_chip_metrics()`, `class_map_panel()`, `class_palette()`, `false_colour_composite()`, `render_png()`, `PngImage`.
- Outputs: new `outputs/<stem>_maps_<chip>.png`; the map GeoTIFFs are named by `source_id` (sample) or `id` (BYOD); the evaluation report carries `per_chip`; `result.json` carries `adaptation_gpu_peak_gb` and `per_chip_mean_iou_range`.
- Sections 5 and 7 print per-chip mean IoU; Section 8 shows the strips inline; the closing has a Predict → Change → Run → Observe → Explain activity in place of the one-sentence optional experiments; the tensor count reads 5.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable; torch is not a CI dependency). Everything below is plumbing evidence on synthetic chips and a NumPy stand-in pipeline, not model evidence.
- `tools/build_notebook.py --check`: up to date; `tools/validate_release_assets.py`: PASS; `ruff check src tests tools`: clean.
- `pytest` with CI's dependencies (numpy, tifffile; torch absent): 48 passed, 2 skipped (sweep head `efb9941`) → **61 passed, 2 skipped**. New `tests/test_review_fixes.py` (13 tests mapped to CR IDs, including the Section 4–8 BYOD end-to-end run); `test_sweep_fixes.py` stand-ins updated for the new Section 4 cell.
- The review's probe script re-run on the regenerated notebook: P1 guided markers 10/10, `brace_typo_present` false, `markdown_claims_8_tensors` [], `source_id_uses_in_tutorial_cells` {}, `probe_slice_test_records_1_4` false, 13 image-display calls, `threshold` / `deployment's` present; P2 tooling OK. P3–P8 import torch at module level and could not run here; their subjects are covered by `test_cr_m1_*`, `test_cr_m6_*`, `test_cr_m2_byod_*` and `test_cr_m3b_*`.

## Remaining gates

- A hosted **Run all in one pass** on a fresh Colab T4 (no restart), including the inline strips in Section 8 and the `gpu_peak_gb` value, then a re-run of the Section 8 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with `BYOD_PATH`): Section 5's frozen metrics must equal a fresh-runtime BYOD run's, and the run must reach the reload-parity print.
- CR-M3(b) on real weights: `head+last_block` then `head` should pass the reload-parity assertion (the mechanism is fixed; the hosted check is the evidence).
- CR-S1, CR-S4 and the release-procedure wording are maintainer decisions.
