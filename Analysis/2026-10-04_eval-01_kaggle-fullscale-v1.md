# Evaluation 01 — Kaggle Full-Scale Pipeline, Version 1 (internet ON)

**Date:** 2026-10-04
**Notebook:** `noname7n11/notebooka927747cc9`, committed Version 1
**Scope:** First fully successful end-to-end run of the full ~2.5M-row pipeline on Kaggle, after replacing `matchms` and `AutoGluon` with dependency-light alternatives (see LOG.md for the full debugging arc that led here).

---

## Run conditions

- Internet: **ON** (pip installs `rdkit` directly from PyPI)
- Accelerator: GPU T4 x2
- Sample size: `load_sampled_train(sample_size=2_600_000, random_state=42)`
- Confirmed via both an interactive run and the official **Save & Run All (Commit)**, which reproduced identical numbers on a completely fresh container (`2514.2 second run - successful`, ~42 min including pip install from scratch).

## Data loaded

- 2,539,608 training spectra, 275,810 unique structures
- 1,213 test spectra, 400 unique molecules

## Results

| Metric | Value |
|---|---|
| Baseline-only MRR@25 (double-holdout, random_state=99) | **0.6740** |
| Reranked MRR@25 | **0.8393** |
| Delta | **+0.1653** |
| `submission.csv` | 400 rows |
| `submission_reranked.csv` | 400 rows |

Both the interactive run and the official committed run reproduced these numbers exactly — confirms determinism, not a fluke of warm kernel state.

## What changed to get here (summary; see LOG.md for full detail)

1. **Dropped `matchms`.** Its import chain (`matchms` → `exporting` → `scipy.sparse` → `numpy._core.strings` → `numpy._core.umath`) reproducibly broke numpy's own `_core.umath` module on Kaggle's base image (`ImportError: cannot import name '_center'`), independent of kernel freshness, CPU/GPU image, or numpy/scipy pinning. Replaced with a hand-written numpy implementation of `CosineGreedy`'s greedy tolerance-matched peak-pairing cosine similarity (`src/baseline.py::cosine_similarity`). `Spectrum` became a plain `{"mz": ..., "intensities": ...}` dict.
2. **Dropped `AutoGluon`.** Hit the *identical* `_center` ImportError independently, on a fully fresh kernel, with a plain unpinned `pip install`. Replaced `TabularPredictor` with `sklearn.ensemble.GradientBoostingClassifier` (`src/reranker.py::train_reranker`/`rerank_candidates`) — sklearn is pre-installed and stable on Kaggle's image.
3. **Fixed the competition data mount path.** Actual path is `/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/`, not the flatter `/kaggle/input/<slug>/` layout assumed earlier.
4. **Fixed a self-inflicted notebook-rebuild bug.** An earlier loose-substring-matching rebuild script had silently overwritten the `src/baseline.py` cell with pip-install text, deleting `build_submission` without any error at build time — only surfaced as `NameError` mid-run. Rebuilt via a clean from-scratch assembly script that asserts every expected function name is present before writing the notebook JSON.

## Validity caveat — IMPORTANT, carried into Evaluation 02's deeper analysis

This run's offline validation score (0.6740 baseline / 0.8393 reranked) was **not yet scrutinized against the competition's actual test-set composition** at the time this run completed. That scrutiny happened in Evaluation 02 (2026-10-04, same day, later) and surfaced a significant scope problem — see that file for the full analysis. Short version: `make_validation_split` holds out *some spectra* of a molecule while deliberately leaving at least one other spectrum of the *same structure* in the candidate pool, so the measured score reflects pure library-retrieval performance (OVERVIEW.md's "Class 1" novelty tier) and says nothing about Class 2 (known structure, no public spectra) or Class 3 (novel structure, de novo generation) molecules, which the hidden test set is confirmed to contain.

## Status at end of this evaluation

- Pipeline runs correctly and completely at full scale, internet ON.
- Not yet verified under internet OFF (competition's actual scoring condition) — that became the focus of Evaluation 02.
- Numbers here should be read as "Class-1-only upper bound," not an estimate of real leaderboard performance.
