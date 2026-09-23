# AutoGluon Reranker — Design Spec

**Status:** Approved for implementation planning
**Date:** 2026-09-23
**Sub-project:** 2 of the CASMI 2026 modeling roadmap (Phase 5 per WORKFLOW.md)

## Purpose

The baseline retrieval pipeline (sub-project 1, shipped 2026-09-22) filters
candidates by precursor mass + adduct and ranks them by raw cosine spectral
similarity. This works but throws away signal: cosine alone ignores mass
accuracy, candidate structural properties, and formula plausibility. A
gradient-boosted tree ensemble (AutoGluon) trained on richer per-candidate
features can learn to combine these signals and push more correct answers
to rank 1, improving MRR@25 without touching the retrieval/filtering step.

**Success criterion:** offline MRR@25 (via `make_validation_split`) with
reranking applied is higher than the baseline-only MRR@25 on the same
held-out split. This is a comparison, not a fixed target — the notebook
must report both numbers side by side.

## Constraints carried from the baseline pipeline (still binding)

- Metric: MRR@25, InChIKey14-based (see `src/metric.py`, unchanged).
- Submission format: `molecule_id,smiles`, ≤25 semicolon-separated
  candidates, every molecule_id exactly once, no NaN.
- Local development only (CPU) — per WORKFLOW.md, AutoGluon training stays
  local; no cloud GPU needed for tree ensembles.
- `dataset/` never committed; model artifacts (AutoGluon's saved predictor
  directory) also never committed — already covered by `.gitignore`'s
  `models/`/`checkpoints/` patterns, verify the AutoGluon save path falls
  under one of those or add a new gitignore entry for it.

## New constraint: LOG.md

A repo-root `LOG.md` file, organized by dated sections (`## YYYY-MM-DD`),
records every meaningful implementation decision going forward: what was
built/changed and *why* (the reasoning, not just a diff summary). This
serves three purposes: (1) raw material for the competition's required
winner methodology writeup (Rule 2.5.b — reproducible description of
architecture, preprocessing, training details), (2) a debugging aid when
something breaks later and nobody remembers why a choice was made, (3)
context for future chat sessions that don't have this conversation's
history.

Every implementation task in this plan appends its own dated entry (or
adds to the current date's section if same-day) as part of that task's
work — not a separate cleanup task. The first entry backfills 2026-09-22
covering the baseline pipeline, written once as part of Task 1 of this
plan (before new reranker work begins), so the log's history is complete
from the project's actual start.

## Architecture

```
score_candidates_for_molecule (existing, untouched)
        |
        v (NOT reused directly — reranker needs richer per-row data)
extract_candidate_features (new, src/reranker.py)
        |
        v
build_training_examples (new) --uses--> make_validation_split (existing, src/data.py)
        |
        v
train_reranker (new, wraps AutoGluon TabularPredictor.fit)
        |
        v
rerank_candidates (new, uses trained predictor at inference time)
        |
        v
notebook: side-by-side MRR@25 comparison (baseline ranking vs reranked ranking)
```

`src/baseline.py` is not modified. `extract_candidate_features` duplicates
the filter+cosine-scoring loop from `score_candidates_for_molecule`
rather than extending that function's return shape, to avoid touching
already-shipped, already-reviewed, already-tested code. The duplication
cost (two similar loops) is accepted as the simpler, lower-risk option.

## Components

### 1. `extract_candidate_features(test_spectra_rows, train_df, ppm_tolerance=15.0) -> pd.DataFrame`

For one molecule's test spectra against the candidate train pool, returns
one row per candidate `inchikey14` (deduplicated, same max-aggregation
convention as the baseline) with columns:

- `inchikey14`, `smiles` (carried through, same as baseline)
- `cosine_score` — max cosine similarity across all (test spectrum,
  candidate spectrum) pairs for this candidate (same computation as
  baseline's `score`, recomputed here since we need the intermediate
  ppm/peak values the baseline discards)
- `ppm_error` — precursor ppm error of the best-matching (highest-cosine)
  test-spectrum/candidate-spectrum pair
- `num_peaks_candidate` — number of peaks in the candidate's spectrum
  (from `num_peaks` column if present, else `len(ms2_mzs)`)
- `mol_wt`, `log_p`, `num_rings`, `num_rotatable_bonds`, `num_hbd`,
  `num_hba` — RDKit descriptors computed from `smiles` via
  `rdkit.Chem.Descriptors` and `rdkit.Chem.Lipinski`; `None`/NaN if the
  SMILES fails to parse (should not happen for `normalized_smiles` from
  the training data, but guarded defensively since AutoGluon handles NaN
  natively)

Molecular-formula match indicator is deferred out of scope for v0 — deriving
a candidate formula comparison reliably from adduct + precursor_mz adds
real complexity (adduct mass tables, multiple charge states) that doesn't
clearly pay for itself when `ppm_error` already captures most of that
signal at this stage. Can be added in a later iteration if feature
importance analysis shows the reranker wants it.

### 2. `build_training_examples(train_df, n_held_out=200, random_state=42, ppm_tolerance=15.0) -> pd.DataFrame`

Calls `make_validation_split(train_df, n_held_out, random_state)` to get
`(val_query_df, val_train_df, val_ground_truth)`. For each held-out
molecule (grouped by `molecule_id` in `val_query_df`), calls
`extract_candidate_features` against `val_train_df`, then labels each
resulting row `is_correct = 1 if inchikey14 == val_ground_truth[molecule_id]
else 0`. Concatenates all molecules' labeled candidate rows into one flat
training dataframe with an added `molecule_id` column (for grouping
during evaluation, not used as a training feature).

### 3. `train_reranker(training_df, model_path, feature_columns, label_column="is_correct", time_limit=120) -> TabularPredictor`

Thin wrapper: `TabularPredictor(label=label_column, path=model_path,
problem_type="binary").fit(training_df[feature_columns + [label_column]],
time_limit=time_limit)`. `time_limit` defaults to 120 seconds (v0 —
AutoGluon's own quality/time tradeoff, not hand-tuned further now) to keep
local iteration fast; the notebook cell that calls this can override it
for a longer run once the pipeline is validated end-to-end.

### 4. `rerank_candidates(predictor, candidate_features_df, top_k=25) -> list[str]`

Given a trained predictor and one molecule's `extract_candidate_features`
output, computes `predictor.predict_proba(candidate_features_df)[1]`
(probability of `is_correct=1`), sorts descending, returns the top_k
`smiles` values as a list (caller joins with `;` for submission format,
matching `build_submission`'s existing convention).

## Data flow / integration point

The notebook adds cells (after the existing baseline cells, not replacing
them):
1. Build training examples from the sampled train set's own held-out
   split (same `train_df` already loaded, reusing `make_validation_split`
   — note this means training data and the offline-eval held-out split
   are the same split; this is acceptable for v0 since AutoGluon has its
   own internal train/validation holdout during `fit()`, but the printed
   offline MRR@25 "after reranking" number on this same split should be
   read as optimistic/in-sample and flagged as such in a markdown cell —
   a fully separate double-holdout is a future-iteration refinement, not
   this sub-project's job).
2. Train the reranker.
3. Re-score the same held-out molecules' candidates with
   `rerank_candidates`, compute MRR@25, print alongside the original
   baseline-only MRR@25 for direct comparison.
4. (Optional, not required for v0 completion) apply the trained reranker
   to the real test set's candidates and produce a second `submission.csv`
   variant — deferred: the plan's primary deliverable is proving the
   reranker measurably helps offline; producing a reranked real submission
   is a natural but separate follow-up once the comparison is validated.

## Testing strategy

- `tests/test_reranker.py`: unit tests on tiny synthetic fixtures (few
  rows, no real parquet needed) covering:
  - `extract_candidate_features` returns expected columns and correct
    `cosine_score`/`ppm_error` values for a known synthetic case
  - `build_training_examples` correctly labels the true match as
    `is_correct=1` and others as `0`
  - `rerank_candidates` correctly orders by predicted probability
    descending
- One integration test training a real (but tiny, low `time_limit`)
  AutoGluon model against a small real data sample (mirrors Task 4's
  pattern from the baseline plan) to catch real-data surprises (e.g.
  AutoGluon-specific dtype requirements, NaN handling) before the
  notebook run.
- AutoGluon's `fit()` is inherently slow even at low `time_limit` (model
  ensemble overhead) — tests must use `time_limit` in the single-digit
  seconds range and a small `presets` value (e.g. `"medium_quality"` or
  lower) to keep the test suite's runtime reasonable; document this choice
  in the test file.

## Explicitly out of scope for this sub-project

- Molecular-formula match feature (noted above, deferred).
- Fully separate double-holdout (train split vs. offline-eval split) for
  a non-optimistic reranked MRR@25 — noted above, deferred.
- Producing a reranked real `submission.csv` — optional stretch, not
  required for completion.
- DreaMS embeddings as additional features — that's sub-project 3
  (Phase 6), comes after this one per the agreed roadmap.
- Any change to `src/baseline.py`, `src/metric.py`, or `src/data.py`
  beyond calling their existing, already-shipped functions.
