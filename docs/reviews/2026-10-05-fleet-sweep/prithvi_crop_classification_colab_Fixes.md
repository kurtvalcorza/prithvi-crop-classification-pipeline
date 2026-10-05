# Fleet-sweep fixes: `prithvi_crop_classification_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report; each flag was first
confirmed in the cell source at `main` `95f678f`. Changes are made in the generator (`tools/build_notebook.py`,
`tools/notebook_template.py`) and, for SWP-F, in the carried `pipeline.py`; the notebook is regenerated. Status and release
labels are unchanged.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 pip-installed the pins into the kernel and raised "Restart the runtime" on stale modules. Generator → `build_notebook.py/2.2`; the template opts in. One kernel cell verifies and runs the pinned `uv` 0.12.15, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (44 packages compiled from the unchanged pyproject pins, `--require-hashes --only-binary :all:`), keys the folder on the lock digest and reuses it, keeps a live worker on re-run, forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1; `tools/build_notebook.py`, `tools/notebook_template.py`, `tools/validate_release_assets.py`, new lock | `test_swp_r_*` (3 tests) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED with 1 of 9 guided markers. Added audience, Input → Model → Output, How to use, roadmap, Predict prompts (Sections 4–7), What to notice + Check your reasoning after Sections 4–8 quoting the recorded Kaggle T4 run of 2026-09-20 (frozen test mean IoU 0.4379 → adapted 0.4302, accuracy 0.5905 → 0.5813 vs majority 0.011 / 0.137; class-weighted validation loss 0.9368 → 0.9114 while validation mean IoU 0.4609 → 0.4451), Troubleshooting, Glossary, Conclusion template; infrastructure labelled and collapsed. The literal `{{id, image, label}}` in the Prerequisites now renders as `{id, image, label}`. | opening, Sections 4–8 markdown, closing, Prerequisites | `test_swp_g_*` (2 tests) |
| SWP-A (quality asserts) | Not flagged | The sweep found no quality assert here; Section 7's asserts are procedure invariants (kept epoch's validation loss ≤ epoch 0's; re-scoring reproduces the kept epoch) and print the test direction without asserting it. Unchanged. | — | — |
| SWP-F (frozen re-run) | Fixed | Confirmed: `adapt()` trained the head in place from whatever weights the model held, so a Section 6 re-run (the closing's optional experiments) continued training while epoch 0 was labelled "frozen model", and a Section 5 re-run scored the adapted model as frozen. `pipeline.py` now keeps the pinned-base value of every tensor adapt() or load_artifact() changes and restores it first (siglip-v1's `restore_base` pattern); a failed adapt leaves the weights as before; the result records `started_from`. Section 5 calls `pipe.restore_base()` before its frozen evaluation. | `src/…/pipeline.py`, Sections 5–6 | `test_swp_f_*` (3 tests) |
| SWP-B (BYOD upload only) | Fixed | Confirmed: BYOD used only `files.upload()`. Added `BYOD_PATH` (zip or folder; Kaggle/Jupyter); guarded upload fallback (off Colab, cancelled, multi-file, non-.zip each name the file or rule). | Section 4 | `test_swp_b_*` (3 tests) |

## User-visible changes

- Section 1 installs nothing into the kernel and never asks for a restart (first build takes a few minutes; reused afterwards). Linux x86_64 only.
- `PrithviCropPipeline.adapt()` and `load_artifact()` always start from the pinned base; new `restore_base()`; `adapt()` result has `started_from`. A re-run of Section 5 after Section 6 puts the base back (re-run 6–8 next).
- New `BYOD_PATH` field; guided-layer cells; infrastructure collapsed.

## Verification (offline; not clean-runtime evidence)

- No model stage can run here (Hub unreachable). The Section 1 cell runs for real against a stand-in environment; the BYOD block runs with stand-ins; `restore_base` runs on a NumPy stand-in model. Plumbing evidence, not model evidence.
- `build_notebook.py --check` up to date; `validate_release_assets.py` PASS; `ruff check src tests tools` clean.
- `pytest` with CI's dependencies only (torch absent): 35 passed, 2 skipped before → 48 passed, 2 skipped after.
- Sweep re-check on the regenerated notebook: isolated runtime, guided markers 9/9, quality asserts 0.

## Remaining gates

- A hosted **Run all in one pass** in a fresh Colab T4 runtime (no restart expected), then a re-run of the Section 8 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with `BYOD_PATH`).
- A full Notebook Review Framework v1 review has not been done.
