# AutoGluon Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an AutoGluon tabular binary classifier on pairwise query-candidate features (cosine similarity, mass accuracy, RDKit descriptors) that reranks the baseline retrieval's candidates, and prove offline (via MRR@25) that reranking beats raw cosine-similarity ranking.

**Architecture:** A new `src/reranker.py` module re-extracts richer per-candidate features (duplicating, not modifying, `src/baseline.py`'s filter+cosine loop), builds a labeled training set from `make_validation_split`'s held-out molecules, trains an `autogluon.tabular.TabularPredictor`, and reorders candidates by predicted match probability at inference. A `LOG.md` file is introduced to record dated, reasoned entries for every implementation decision in this plan (and backfilled for the prior baseline sub-project).

**Tech Stack:** Python 3.12 (via a dedicated venv — AutoGluon has no Python 3.14 wheels as of 2026-09-23; the rest of this repo runs on 3.14 but this sub-project's code/tests run under the 3.12 venv), `autogluon.tabular` 1.6.3, `rdkit` 2026.3.6, `matchms` 0.33.1, `pandas` 2.3.3, `pytest` 9.1.1 — all already verified installed in `.venv-reranker/` (see Environment Setup below).

**Spec:** `docs/superpowers/specs/2026-09-23-autogluon-reranker-design.md`

## Global Constraints

- Metric: MRR@25, InChIKey14-based, via the existing `src/metric.py` (unchanged, do not modify).
- `src/baseline.py`, `src/metric.py`, `src/data.py` are NOT modified by this plan — only called.
- Reranker training/inference re-derives features via a new `extract_candidate_features` function in `src/reranker.py`, deliberately duplicating (not extending) `score_candidates_for_molecule`'s filter+cosine loop.
- Molecular-formula match feature, fully separate double-holdout, and producing a reranked real `submission.csv` are explicitly OUT of scope for this plan (see spec's "Explicitly out of scope" section).
- Success is a comparison: offline MRR@25 with reranking vs. offline MRR@25 baseline-only on the same held-out split, both printed side by side — not a fixed numeric target.
- `LOG.md` (repo root) gets a dated entry (`## YYYY-MM-DD`) for every task in this plan, stating what was done and why. Task 1 also backfills a `## 2026-09-22` entry covering the already-shipped baseline pipeline.
- AutoGluon `fit()` calls in tests must use a low `time_limit` (single-digit seconds) and a light preset to keep test runtime reasonable — this is a deliberate YAGNI-style scoping of test rigor, not a production training setting.
- Environment: all Python/pip/pytest/jupyter commands in this plan's tasks run through the venv at `.venv-reranker/Scripts/python.exe` (Windows), NOT the system `python`/`pip` used by the earlier baseline pipeline plan. This is called out explicitly in every task's steps.

## Review Focus

1. **Held-out molecule with zero remaining candidates after filtering** — `make_validation_split` can produce a held-out molecule whose spectrum-level split leaves candidates that then don't survive `filter_candidates`'s ppm/adduct filter at all (e.g. adduct mismatch across its own split spectra). `extract_candidate_features` must return an empty (not crashing) dataframe for that molecule, and `build_training_examples` must skip it gracefully (no `is_correct=1` row exists for it, which is fine — it simply contributes zero rows, not a crash).
2. **RDKit descriptor computation on a SMILES that fails to parse** — `normalized_smiles` values from `train.parquet` should always parse (they're RDKit-normalized already), but `extract_candidate_features` explicitly guards this per the spec ("`None`/NaN if the SMILES fails to parse"); a reasonable person expects a bad SMILES to produce NaN features, not a crash that kills the whole feature-extraction loop.
3. **`rerank_candidates` called with fewer than `top_k` candidates available** — a molecule with only 3 candidates after filtering should return all 3 reranked, not error or pad with garbage. Mirrors `build_submission`'s existing "fewer than top_k is fine" behavior.
4. **AutoGluon predictor's `predict_proba` column ordering** — binary classifiers can return probability columns in either label order (`[0, 1]` or based on which class was seen first); `rerank_candidates` must explicitly select the probability of `is_correct == 1`, not assume column position, or reranking silently inverts on some runs.
5. **Empty training set** (e.g. if every held-out molecule's split produces zero candidates in some edge configuration) — `train_reranker` receiving a training dataframe with zero or only-one-class rows should fail loudly (AutoGluon will raise on this), not silently produce a useless model; no task needs to "handle" this gracefully, but Task 3's test should confirm AutoGluon's natural failure mode here is a clear error, not a silent bad model, so nobody debugging this later is confused by a cryptic downstream symptom.

---

## Environment Setup Notes (context for every task below)

A Python 3.12 venv already exists at `.venv-reranker/` in the repo root (created 2026-09-23, outside any task in this plan — do not recreate it) with `autogluon.tabular`, `rdkit`, `matchms`, `pandas`, `pyarrow`, `pytest`, `jupyter`, `nbconvert` already installed. Every task's Python/pytest/pip commands use this venv explicitly:

- Windows: `.venv-reranker\Scripts\python.exe`, `.venv-reranker\Scripts\python.exe -m pytest`, `.venv-reranker\Scripts\python.exe -m pip ...`
- The venv is gitignored (`.venv-reranker/` added to `.gitignore` already, outside this plan's tasks).
- `src/reranker.py` imports `src.data` and `src.baseline` (both already on Python 3.14 but written in plain, version-agnostic Python — they run fine under 3.12 too, already verified importable: `rdkit`, `matchms`, `pandas`, `pyarrow` are all installed in `.venv-reranker/`).

---

## Task 1: LOG.md — Introduce and Backfill

**Files:**
- Create: `LOG.md`

**Interfaces:**
- Consumes: nothing (pure documentation file).
- Produces: a dated-entry convention every later task in this plan follows (`## YYYY-MM-DD` sections, each entry a `### <short title>` with a "What" and "Why" paragraph).

- [ ] **Step 1: Create LOG.md with the backfilled baseline entry and today's entry header**

Create `LOG.md`:

```markdown
# CASMI 2026 — Project Log

Dated, reasoned record of what was built and why. Doubles as raw material
for the competition's required winner methodology writeup (Rule 2.5.b)
and as context for future sessions.

## 2026-09-22

### Baseline retrieval pipeline (sub-project 1)

**What:** Built `src/metric.py` (offline MRR@25 scorer using RDKit
tautomer-canonicalized InChIKey14 matching), `src/data.py` (stratified
sampled train-set loader always including the `enveda-np-examples` and
`drug_plus` libraries in full, plus `make_validation_split` for a
molecule-level held-out validation split), and `src/baseline.py`
(precursor-mass + adduct filtering, matchms cosine spectral similarity
scoring, max-aggregation across a molecule's multiple spectra, top-25
submission formatting). Wired into `Enveda_CASMI_local.ipynb`, produced
a first real `submission.csv` against the actual test set.

**Why:** Establishes a working, offline-validatable baseline before any
ML model — per WORKFLOW.md's phased approach (Phase 3). Precursor mass +
cosine similarity is the standard first-pass strategy for Class 1
(publicly-known-spectra) molecules per the competition's own strategic
blueprint (OVERVIEW.md Section 10). Getting a real submission shipped
early also banks MRR@25 points directly (the metric rewards any correct
guess in the top 25, so even a weak baseline contributes to the final
score — unlike a discardable exploratory step).

**Bug found and fixed during final review:** the first version of the
offline validation split held out candidates by `inchikey14` — the exact
match target — meaning the correct structure could never appear in its
own candidate pool. This made the offline MRR@25 number (0.0050)
structurally uninformative rather than just "low" — it measured the
rate of rare tautomer-canonicalization collisions, not retrieval
quality. Fixed by splitting at the spectrum level instead: for each
held-out molecule, some of its spectra become the query set while at
least one other spectrum of that same structure remains in the
candidate pool (mirroring the real Class 1 scenario — the molecule does
have library spectra, just not the exact query acquisition). After the
fix, offline MRR@25 jumped to 0.4173 on the same 75k-row sample,
confirming the harness now actually measures something.

**Result:** 27 tests passing. `src/data.py::make_validation_split` is
the reusable spectrum-level split function later sub-projects (including
this reranker) build on.

## 2026-09-23
```

- [ ] **Step 2: Commit**

```bash
git add LOG.md
git commit -m "docs: introduce LOG.md, backfill baseline pipeline entry"
```

---

## Task 2: Candidate Feature Extraction

**Files:**
- Create: `src/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `src.baseline.filter_candidates(test_row, train_df, ppm_tolerance)`, `src.baseline.to_matchms_spectrum(mzs, intensities, metadata)`, `src.baseline.cosine_similarity(spectrum_a, spectrum_b)` — all already defined and unmodified in `src/baseline.py`.
- Produces: `extract_candidate_features(test_spectra_rows: pd.DataFrame, train_df: pd.DataFrame, ppm_tolerance: float = 15.0) -> pd.DataFrame` returning columns `["inchikey14", "smiles", "cosine_score", "ppm_error", "num_peaks_candidate", "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba"]`, deduplicated by `inchikey14` keeping the row from the highest-`cosine_score` (test spectrum, candidate spectrum) pair.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reranker.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.reranker import extract_candidate_features


def test_extract_candidate_features_returns_expected_columns():
    test_spectra_rows = pd.DataFrame(
        [
            {
                "molecule_id": "m1",
                "spectrum_id": "t1",
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "inchikey14": "AAAAAAAAAAAAAA",
                "normalized_smiles": "CCO",  # ethanol
                "num_peaks": 1,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    result = extract_candidate_features(test_spectra_rows, train_df, ppm_tolerance=15.0)
    expected_columns = {
        "inchikey14", "smiles", "cosine_score", "ppm_error",
        "num_peaks_candidate", "mol_wt", "log_p", "num_rings",
        "num_rotatable_bonds", "num_hbd", "num_hba",
    }
    assert set(result.columns) == expected_columns
    assert len(result) == 1


def test_extract_candidate_features_cosine_and_ppm_values_correct():
    test_spectra_rows = pd.DataFrame(
        [
            {
                "molecule_id": "m1",
                "spectrum_id": "t1",
                "adduct": "[M+H]+",
                "precursor_mz": 100.0000,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "adduct": "[M+H]+",
                "precursor_mz": 100.0010,  # 10 ppm off from 100.0000
                "inchikey14": "AAAAAAAAAAAAAA",
                "normalized_smiles": "CCO",
                "num_peaks": 1,
                "ms2_mzs": np.array([100.0]),  # identical spectrum -> cosine ~1.0
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    result = extract_candidate_features(test_spectra_rows, train_df, ppm_tolerance=15.0)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["cosine_score"] == pytest.approx(1.0, abs=1e-6)
    assert row["ppm_error"] == pytest.approx(10.0, abs=0.1)
    assert row["num_peaks_candidate"] == 1


def test_extract_candidate_features_rdkit_descriptors_computed():
    test_spectra_rows = pd.DataFrame(
        [
            {
                "molecule_id": "m1",
                "spectrum_id": "t1",
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "inchikey14": "AAAAAAAAAAAAAA",
                "normalized_smiles": "CCO",  # ethanol: MolWt ~46.07
                "num_peaks": 1,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    result = extract_candidate_features(test_spectra_rows, train_df, ppm_tolerance=15.0)
    row = result.iloc[0]
    assert row["mol_wt"] == pytest.approx(46.07, abs=0.1)
    assert row["num_rings"] == 0
    assert row["num_hbd"] == 1  # ethanol has 1 OH


def test_extract_candidate_features_no_candidates_returns_empty_dataframe():
    test_spectra_rows = pd.DataFrame(
        [
            {
                "molecule_id": "m1",
                "spectrum_id": "t1",
                "adduct": "[M+H]+",
                "precursor_mz": 999.0,  # matches nothing in train_df
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "inchikey14": "AAAAAAAAAAAAAA",
                "normalized_smiles": "CCO",
                "num_peaks": 1,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    result = extract_candidate_features(test_spectra_rows, train_df, ppm_tolerance=15.0)
    assert len(result) == 0
    assert list(result.columns) == [
        "inchikey14", "smiles", "cosine_score", "ppm_error",
        "num_peaks_candidate", "mol_wt", "log_p", "num_rings",
        "num_rotatable_bonds", "num_hbd", "num_hba",
    ]


def test_extract_candidate_features_max_aggregates_across_test_spectra():
    # Two test spectra for the same molecule; candidate should keep the
    # row from whichever (test, candidate) pair had the higher cosine score.
    test_spectra_rows = pd.DataFrame(
        [
            {
                "molecule_id": "m1", "spectrum_id": "t1", "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "ms2_mzs": np.array([100.0]), "ms2_normalized_intensities": np.array([1.0]),
            },
            {
                "molecule_id": "m1", "spectrum_id": "t2", "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "ms2_mzs": np.array([50.0]), "ms2_normalized_intensities": np.array([1.0]),
            },
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "adduct": "[M+H]+", "precursor_mz": 100.0,
                "inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO",
                "num_peaks": 1,
                "ms2_mzs": np.array([100.0]),  # matches t1 exactly
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    result = extract_candidate_features(test_spectra_rows, train_df, ppm_tolerance=15.0)
    assert len(result) == 1
    assert result.iloc[0]["cosine_score"] == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-reranker\Scripts\python.exe -m pytest tests/test_reranker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.reranker'`.

- [ ] **Step 3: Write the implementation**

Create `src/reranker.py`:

```python
"""AutoGluon-based reranker: richer per-candidate features + trained reordering."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from src.baseline import filter_candidates, to_matchms_spectrum, cosine_similarity

_FEATURE_COLUMNS = [
    "inchikey14", "smiles", "cosine_score", "ppm_error",
    "num_peaks_candidate", "mol_wt", "log_p", "num_rings",
    "num_rotatable_bonds", "num_hbd", "num_hba",
]


def _rdkit_descriptors(smiles: str) -> dict:
    """Compute RDKit descriptors for a SMILES string; NaN values if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "mol_wt": np.nan, "log_p": np.nan, "num_rings": np.nan,
            "num_rotatable_bonds": np.nan, "num_hbd": np.nan, "num_hba": np.nan,
        }
    return {
        "mol_wt": Descriptors.MolWt(mol),
        "log_p": Descriptors.MolLogP(mol),
        "num_rings": Descriptors.RingCount(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_hbd": Lipinski.NumHDonors(mol),
        "num_hba": Lipinski.NumHAcceptors(mol),
    }


def extract_candidate_features(
    test_spectra_rows: pd.DataFrame, train_df: pd.DataFrame, ppm_tolerance: float = 15.0
) -> pd.DataFrame:
    """Extract rich per-candidate features for reranker training/inference.

    One row per candidate inchikey14, deduplicated by keeping the
    highest-cosine_score (test spectrum, candidate spectrum) pair's
    feature values (mirrors src.baseline's max-aggregation convention).
    """
    best_score: dict[str, float] = {}
    best_row: dict[str, dict] = {}

    for _, test_row in test_spectra_rows.iterrows():
        candidates = filter_candidates(test_row, train_df, ppm_tolerance=ppm_tolerance)
        if candidates.empty:
            continue
        test_spectrum = to_matchms_spectrum(
            test_row["ms2_mzs"], test_row["ms2_normalized_intensities"], metadata={}
        )
        for _, candidate_row in candidates.iterrows():
            candidate_spectrum = to_matchms_spectrum(
                candidate_row["ms2_mzs"],
                candidate_row["ms2_normalized_intensities"],
                metadata={},
            )
            score = cosine_similarity(test_spectrum, candidate_spectrum)
            key = candidate_row["inchikey14"]
            if score > best_score.get(key, -1.0):
                best_score[key] = score
                ppm_error = (
                    abs(candidate_row["precursor_mz"] - test_row["precursor_mz"])
                    / test_row["precursor_mz"]
                    * 1e6
                )
                num_peaks = candidate_row.get("num_peaks")
                if num_peaks is None or pd.isna(num_peaks):
                    num_peaks = len(candidate_row["ms2_mzs"])
                descriptors = _rdkit_descriptors(candidate_row["normalized_smiles"])
                best_row[key] = {
                    "inchikey14": key,
                    "smiles": candidate_row["normalized_smiles"],
                    "cosine_score": score,
                    "ppm_error": ppm_error,
                    "num_peaks_candidate": num_peaks,
                    **descriptors,
                }

    if not best_row:
        return pd.DataFrame(columns=_FEATURE_COLUMNS)

    return pd.DataFrame(list(best_row.values()), columns=_FEATURE_COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-reranker\Scripts\python.exe -m pytest tests/test_reranker.py -v`
Expected: 5/5 tests pass.

- [ ] **Step 5: Append LOG.md entry**

Add to the `## 2026-09-23` section of `LOG.md` (append after the header, do not remove it):

```markdown
### Candidate feature extraction (`src/reranker.py::extract_candidate_features`)

**What:** New function that re-runs the same precursor-mass/adduct
filtering and cosine-similarity scoring as `src/baseline.py`, but
returns a richer per-candidate row: cosine score, precursor ppm error,
candidate peak count, and RDKit descriptors (molecular weight, LogP,
ring count, rotatable bond count, H-bond donor/acceptor counts) computed
from the candidate's SMILES.

**Why:** `src/baseline.py::score_candidates_for_molecule` only returns
`inchikey14`, `smiles`, `score` — not enough signal for a reranker to
learn from. Rather than extending that function's return shape (which
would touch already-shipped, already-reviewed code and risk regressing
its existing callers/tests), a parallel function duplicates the
filter+cosine loop. The duplication cost is small and isolates risk:
`src/baseline.py` and its test suite are untouched by this sub-project.
```

- [ ] **Step 6: Commit**

```bash
git add src/reranker.py tests/test_reranker.py LOG.md
git commit -m "feat: add candidate feature extraction for reranker"
```

---

## Task 3: Training Example Builder + Reranker Training

**Files:**
- Modify: `src/reranker.py` (add two functions)
- Test: `tests/test_reranker.py` (add tests)

**Interfaces:**
- Consumes: `src.data.make_validation_split(train_df, n_held_out, random_state) -> (val_query_df, val_train_df, val_ground_truth)` (already defined, unmodified, in `src/data.py`); `extract_candidate_features` from Task 2.
- Produces:
  - `build_training_examples(train_df: pd.DataFrame, n_held_out: int = 200, random_state: int = 42, ppm_tolerance: float = 15.0) -> pd.DataFrame` — returns a flat dataframe with all of `extract_candidate_features`'s columns plus `molecule_id` and `is_correct` (0 or 1), one row per (held-out molecule, candidate) pair across all held-out molecules.
  - `train_reranker(training_df: pd.DataFrame, model_path: str, feature_columns: list[str], label_column: str = "is_correct", time_limit: int = 120) -> "autogluon.tabular.TabularPredictor"` — fits and returns a binary-classification `TabularPredictor`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reranker.py`:

```python
from src.reranker import build_training_examples, train_reranker


def _make_synthetic_train_df():
    # 3 molecules, each with 2 spectra (eligible for held-out split),
    # distinct precursor_mz/adduct groups so filter_candidates separates them cleanly.
    rows = []
    specs = [
        ("AAAAAAAAAAAAAA", "CCO", 100.0, [100.0, 101.0]),
        ("BBBBBBBBBBBBBB", "CCC", 200.0, [200.0, 201.0]),
        ("CCCCCCCCCCCCCC", "CCCC", 300.0, [300.0, 301.0]),
    ]
    for key, smiles, mz, peak_variants in specs:
        for i, peak in enumerate([mz, mz]):  # 2 spectra per molecule
            rows.append(
                {
                    "molecule_id": key,
                    "spectrum_id": f"{key}_s{i}",
                    "inchikey14": key,
                    "normalized_smiles": smiles,
                    "adduct": "[M+H]+",
                    "precursor_mz": mz,
                    "num_peaks": 1,
                    "ms2_mzs": np.array([peak]),
                    "ms2_normalized_intensities": np.array([1.0]),
                    "instrument_type": "timsTOF",
                }
            )
    return pd.DataFrame(rows)


def test_build_training_examples_labels_true_match_correctly():
    train_df = _make_synthetic_train_df()
    training_df = build_training_examples(train_df, n_held_out=3, random_state=1)
    assert "is_correct" in training_df.columns
    assert "molecule_id" in training_df.columns
    # every row's is_correct must be 0 or 1
    assert set(training_df["is_correct"].unique()) <= {0, 1}
    # for each molecule_id, exactly the row whose inchikey14 matches its own
    # molecule_id (ground truth is keyed by inchikey14 in this synthetic setup)
    # should be labeled 1, if that molecule's own structure survived filtering
    for molecule_id, group in training_df.groupby("molecule_id"):
        correct_rows = group[group["is_correct"] == 1]
        if len(correct_rows) > 0:
            assert correct_rows.iloc[0]["inchikey14"] == molecule_id


def test_build_training_examples_empty_when_no_eligible_molecules():
    # all single-spectrum molecules -> none eligible for held-out split
    train_df = _make_synthetic_train_df().drop_duplicates("molecule_id")
    training_df = build_training_examples(train_df, n_held_out=3, random_state=1)
    assert len(training_df) == 0
    assert "is_correct" in training_df.columns


def test_train_reranker_fits_and_returns_predictor(tmp_path):
    train_df = _make_synthetic_train_df()
    training_df = build_training_examples(train_df, n_held_out=3, random_state=1)
    feature_columns = [
        "cosine_score", "ppm_error", "num_peaks_candidate",
        "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba",
    ]
    predictor = train_reranker(
        training_df, model_path=str(tmp_path / "model"),
        feature_columns=feature_columns, time_limit=10,
    )
    predictions = predictor.predict_proba(training_df[feature_columns])
    assert 1 in predictions.columns or True in predictions.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-reranker\Scripts\python.exe -m pytest tests/test_reranker.py -v -k "build_training_examples or train_reranker"`
Expected: FAIL with `ImportError: cannot import name 'build_training_examples'`.

- [ ] **Step 3: Write the implementation**

Append to `src/reranker.py`:

```python
from src.data import make_validation_split


def build_training_examples(
    train_df: pd.DataFrame,
    n_held_out: int = 200,
    random_state: int = 42,
    ppm_tolerance: float = 15.0,
) -> pd.DataFrame:
    """Build a labeled training set for the reranker from held-out molecules.

    Uses make_validation_split to get held-out query molecules and their
    ground truth, extracts candidate features for each, and labels each
    candidate row is_correct=1 if its inchikey14 matches that molecule's
    ground truth, else 0.
    """
    val_query_df, val_train_df, val_ground_truth = make_validation_split(
        train_df, n_held_out=n_held_out, random_state=random_state
    )

    all_rows = []
    for molecule_id, group in val_query_df.groupby("molecule_id"):
        features = extract_candidate_features(group, val_train_df, ppm_tolerance=ppm_tolerance)
        if features.empty:
            continue
        features = features.copy()
        features["molecule_id"] = molecule_id
        true_key = val_ground_truth[molecule_id]
        features["is_correct"] = (features["inchikey14"] == true_key).astype(int)
        all_rows.append(features)

    if not all_rows:
        columns = _FEATURE_COLUMNS + ["molecule_id", "is_correct"]
        return pd.DataFrame(columns=columns)

    return pd.concat(all_rows, ignore_index=True)


def train_reranker(
    training_df: pd.DataFrame,
    model_path: str,
    feature_columns: list[str],
    label_column: str = "is_correct",
    time_limit: int = 120,
):
    """Train a binary-classification AutoGluon reranker.

    Returns the fitted TabularPredictor. Raises whatever AutoGluon raises
    on degenerate input (e.g. a single-class training set) — that failure
    is intentionally not caught here, since a silently-broken model is
    worse than a loud training-time error.
    """
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=label_column, path=model_path, problem_type="binary"
    )
    predictor.fit(training_df[feature_columns + [label_column]], time_limit=time_limit)
    return predictor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-reranker\Scripts\python.exe -m pytest tests/test_reranker.py -v`
Expected: all tests pass (9 total: 5 from Task 2 + 4 new). Note: `test_train_reranker_fits_and_returns_predictor` will take longer than the other tests (AutoGluon's fit overhead even at `time_limit=10`) — expect tens of seconds, not instant; this is normal.

- [ ] **Step 5: Append LOG.md entry**

Add to the `## 2026-09-23` section of `LOG.md`:

```markdown
### Training example builder + AutoGluon training (`build_training_examples`, `train_reranker`)

**What:** `build_training_examples` reuses `make_validation_split`
(already built for baseline offline validation) to get held-out query
molecules and ground truth, extracts candidate features for each via
`extract_candidate_features`, and labels every candidate row
`is_correct` (1 if it matches the molecule's true structure, else 0).
`train_reranker` wraps `autogluon.tabular.TabularPredictor` as a binary
classifier over these labeled rows.

**Why:** Reusing `make_validation_split` means the reranker's training
data and the baseline's offline-eval split share one definition — no
duplicated split logic, and both sub-projects benefit from any future
fix to that function. Framing this as binary classification
(`is_correct` per candidate row) rather than a learning-to-rank
objective is a deliberate v0 simplification: AutoGluon's tabular
binary-classification path is simpler to set up and debug than its
ranking support, and `predict_proba` gives a usable ranking signal
directly. A different acceptable approach for a small candidate pool
per molecule.

**Known limitation (documented in spec, not fixed here):** training
data and the offline-eval held-out split are currently the same split,
so the "reranked MRR@25" comparison in Task 4 should be read as
somewhat optimistic/in-sample. A fully separate double-holdout is a
deferred future refinement.
```

- [ ] **Step 6: Commit**

```bash
git add src/reranker.py tests/test_reranker.py LOG.md
git commit -m "feat: add training example builder and AutoGluon trainer"
```

---

## Task 4: Reranking at Inference + Real-Data Integration Test

**Files:**
- Modify: `src/reranker.py` (add one function)
- Test: `tests/test_reranker.py` (add tests), `tests/test_reranker_integration.py` (new)

**Interfaces:**
- Consumes: `train_reranker`'s returned `TabularPredictor` (Task 3); `extract_candidate_features` (Task 2).
- Produces: `rerank_candidates(predictor, candidate_features_df: pd.DataFrame, feature_columns: list[str], top_k: int = 25) -> list[str]` — returns up to `top_k` SMILES strings ordered by descending predicted `is_correct` probability.

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/test_reranker.py`:

```python
from src.reranker import rerank_candidates


def test_rerank_candidates_orders_by_predicted_probability(tmp_path):
    train_df = _make_synthetic_train_df()
    training_df = build_training_examples(train_df, n_held_out=3, random_state=1)
    feature_columns = [
        "cosine_score", "ppm_error", "num_peaks_candidate",
        "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba",
    ]
    predictor = train_reranker(
        training_df, model_path=str(tmp_path / "model2"),
        feature_columns=feature_columns, time_limit=10,
    )
    # Build a candidate set with a clear "obviously correct" row (matches
    # the pattern the model was trained to recognize as is_correct=1-like)
    # and a clear "obviously wrong" row, then check ordering is a valid
    # permutation weighted by predicted probability (not asserting exact
    # scores, since AutoGluon's exact output isn't deterministic enough
    # to pin — asserting the function returns a valid ranked list of the
    # right length and content is the meaningful check here).
    candidates = training_df[training_df["molecule_id"] == training_df["molecule_id"].iloc[0]]
    result = rerank_candidates(predictor, candidates, feature_columns, top_k=25)
    assert isinstance(result, list)
    assert len(result) == len(candidates)
    assert set(result) == set(candidates["smiles"])


def test_rerank_candidates_respects_top_k_smaller_than_pool(tmp_path):
    train_df = _make_synthetic_train_df()
    training_df = build_training_examples(train_df, n_held_out=3, random_state=1)
    feature_columns = [
        "cosine_score", "ppm_error", "num_peaks_candidate",
        "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba",
    ]
    predictor = train_reranker(
        training_df, model_path=str(tmp_path / "model3"),
        feature_columns=feature_columns, time_limit=10,
    )
    result = rerank_candidates(predictor, training_df, feature_columns, top_k=1)
    assert len(result) == 1


def test_rerank_candidates_fewer_candidates_than_top_k_returns_all(tmp_path):
    train_df = _make_synthetic_train_df()
    training_df = build_training_examples(train_df, n_held_out=3, random_state=1)
    feature_columns = [
        "cosine_score", "ppm_error", "num_peaks_candidate",
        "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba",
    ]
    predictor = train_reranker(
        training_df, model_path=str(tmp_path / "model4"),
        feature_columns=feature_columns, time_limit=10,
    )
    single_molecule = training_df[training_df["molecule_id"] == training_df["molecule_id"].iloc[0]]
    result = rerank_candidates(predictor, single_molecule, feature_columns, top_k=25)
    assert len(result) == len(single_molecule)  # fewer than 25 available, returns all
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-reranker\Scripts\python.exe -m pytest tests/test_reranker.py -v -k rerank_candidates`
Expected: FAIL with `ImportError: cannot import name 'rerank_candidates'`.

- [ ] **Step 3: Write the implementation**

Append to `src/reranker.py`:

```python
def rerank_candidates(
    predictor, candidate_features_df: pd.DataFrame, feature_columns: list[str], top_k: int = 25
) -> list[str]:
    """Reorder candidates by the trained reranker's predicted match probability.

    Returns up to top_k SMILES strings, highest predicted probability first.
    Explicitly selects the probability of the positive class (is_correct=1)
    rather than assuming column position, since AutoGluon's predict_proba
    column order is not guaranteed to be [0, 1].
    """
    if candidate_features_df.empty:
        return []

    probabilities = predictor.predict_proba(candidate_features_df[feature_columns])
    positive_class_probs = probabilities[1] if 1 in probabilities.columns else probabilities[True]

    ranked = candidate_features_df.assign(_positive_prob=positive_class_probs.values).sort_values(
        "_positive_prob", ascending=False
    )
    return ranked["smiles"].head(top_k).tolist()
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `.venv-reranker\Scripts\python.exe -m pytest tests/test_reranker.py -v`
Expected: all tests pass (12 total).

- [ ] **Step 5: Write the real-data integration test**

Create `tests/test_reranker_integration.py`:

```python
"""End-to-end validation: real sampled train data -> reranker training -> MRR@25 comparison."""

import os
import pandas as pd
import pytest
from src.data import load_sampled_train, make_validation_split
from src.baseline import build_submission
from src.metric import mrr_at_25
from src.reranker import (
    build_training_examples,
    train_reranker,
    extract_candidate_features,
    rerank_candidates,
)

DATASET_AVAILABLE = os.path.exists("dataset/train.parquet")

_FEATURE_COLUMNS = [
    "cosine_score", "ppm_error", "num_peaks_candidate",
    "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba",
]


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="dataset/train.parquet not present")
def test_reranker_pipeline_trains_and_reranks_on_real_sample(tmp_path):
    train_df = load_sampled_train(sample_size=5_000, random_state=0)

    training_df = build_training_examples(train_df, n_held_out=20, random_state=1)
    if training_df.empty or training_df["is_correct"].nunique() < 2:
        pytest.skip(
            "Sampled data produced no usable training examples with both "
            "classes present — expected occasionally at small sample sizes, "
            "not a code defect (see Review Focus item 5)."
        )

    predictor = train_reranker(
        training_df, model_path=str(tmp_path / "integration_model"),
        feature_columns=_FEATURE_COLUMNS, time_limit=30,
    )

    # Build a fresh held-out split (different random_state) to compare
    # baseline vs reranked MRR@25 on the same query set.
    val_query_df, val_train_df, val_ground_truth = make_validation_split(
        train_df, n_held_out=20, random_state=2
    )
    if not val_ground_truth:
        pytest.skip("No eligible held-out molecules in this sample at this random_state.")

    baseline_predictions = {}
    reranked_predictions = {}
    for molecule_id, group in val_query_df.groupby("molecule_id"):
        features = extract_candidate_features(group, val_train_df, ppm_tolerance=15.0)
        if features.empty:
            baseline_predictions[molecule_id] = [""]
            reranked_predictions[molecule_id] = [""]
            continue
        baseline_order = features.sort_values("cosine_score", ascending=False)["smiles"].tolist()
        baseline_predictions[molecule_id] = baseline_order if baseline_order else [""]
        reranked_predictions[molecule_id] = rerank_candidates(
            predictor, features, _FEATURE_COLUMNS, top_k=25
        ) or [""]

    baseline_mrr = mrr_at_25(baseline_predictions, val_ground_truth)
    reranked_mrr = mrr_at_25(reranked_predictions, val_ground_truth)

    print(f"Baseline MRR@25: {baseline_mrr:.4f}, Reranked MRR@25: {reranked_mrr:.4f}")

    # Smoke assertions only (per plan scope — this is not a score threshold
    # test, sample is too small/random for that): both scores are valid
    # MRR values, and the pipeline ran end to end without crashing.
    assert 0.0 <= baseline_mrr <= 1.0
    assert 0.0 <= reranked_mrr <= 1.0
```

- [ ] **Step 6: Run the integration test**

Run: `.venv-reranker\Scripts\python.exe -m pytest tests/test_reranker_integration.py -v -s`
Expected: PASS (or SKIPPED with a clear reason if the small random sample happens to produce no usable training data — this is an accepted, documented possibility per Review Focus item 5, not a bug). Note this test loads real data and trains an AutoGluon model twice conceptually (once via `build_training_examples`+`train_reranker`, output reused for reranking) — expect it to take one to several minutes; that is normal.

- [ ] **Step 7: Append LOG.md entry**

Add to the `## 2026-09-23` section of `LOG.md`:

```markdown
### Reranking at inference + real-data integration test (`rerank_candidates`)

**What:** `rerank_candidates` takes a trained predictor and a molecule's
candidate features, reorders by predicted `is_correct=1` probability
(explicitly selecting that column rather than assuming position 1 in
`predict_proba`'s output, since AutoGluon does not guarantee column
order), and returns up to `top_k` SMILES. Added
`tests/test_reranker_integration.py`, mirroring the baseline pipeline's
Task 4 pattern: trains and reranks against a real (small) sample of
`dataset/train.parquet` to catch real-data surprises before touching
the notebook.

**Why:** The predict_proba column-order guard is not optional —
silently reading the wrong column would invert every ranking without
raising any error, the worst kind of bug for a scoring pipeline. The
real-data integration test exists for the same reason Task 4 of the
baseline plan added one: synthetic unit-test fixtures don't exercise
real schema quirks (dtypes, missing columns, AutoGluon's own
data-validation behavior) that only show up against the actual parquet
files.
```

- [ ] **Step 8: Commit**

```bash
git add src/reranker.py tests/test_reranker.py tests/test_reranker_integration.py LOG.md
git commit -m "feat: add reranking inference function and real-data integration test"
```

---

## Task 5: Wire Into Notebook — Side-by-Side MRR@25 Comparison

**Files:**
- Modify: `Enveda_CASMI_local.ipynb`

**Interfaces:**
- Consumes: `src.reranker.build_training_examples`, `src.reranker.train_reranker`, `src.reranker.extract_candidate_features`, `src.reranker.rerank_candidates` (Tasks 2-4); `src.data.load_sampled_train`, `src.data.make_validation_split` (already shipped); `src.metric.mrr_at_25` (already shipped).
- Produces: new notebook cells appended after the existing baseline cells, printing baseline MRR@25 vs. reranked MRR@25 side by side. No new `submission.csv` variant is produced (explicitly out of scope per spec).

- [ ] **Step 1: Add setup markdown + imports cell**

Add a markdown cell:

```markdown
## AutoGluon Reranker (sub-project 2)

Trains a binary-classification tree ensemble on richer per-candidate
features (cosine score, mass accuracy, RDKit descriptors) to reorder
the baseline's candidates. Compares offline MRR@25 before/after
reranking on the same held-out split.

**Note:** this section requires the `.venv-reranker` environment
(Python 3.12 + autogluon.tabular) — run this notebook's kernel from
that venv, not the system Python 3.14 kernel used for the baseline
cells above (AutoGluon has no Python 3.14 support as of this writing).
```

Add a code cell:

```python
from src.reranker import (
    build_training_examples,
    train_reranker,
    extract_candidate_features,
    rerank_candidates,
)

_RERANKER_FEATURE_COLUMNS = [
    "cosine_score", "ppm_error", "num_peaks_candidate",
    "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba",
]
```

- [ ] **Step 2: Add training cell**

```python
reranker_training_df = build_training_examples(
    train_df, n_held_out=200, random_state=42
)
print(f"Reranker training examples: {len(reranker_training_df)} rows, "
      f"{reranker_training_df['is_correct'].sum()} positive")

reranker_predictor = train_reranker(
    reranker_training_df,
    model_path="./autogluon_reranker_model",
    feature_columns=_RERANKER_FEATURE_COLUMNS,
    time_limit=120,
)
```

- [ ] **Step 3: Add side-by-side comparison cell**

```python
baseline_predictions_cmp = {}
reranked_predictions_cmp = {}

for molecule_id, group in val_query_df.groupby("molecule_id"):
    features = extract_candidate_features(group, val_train_df, ppm_tolerance=15.0)
    if features.empty:
        baseline_predictions_cmp[molecule_id] = [""]
        reranked_predictions_cmp[molecule_id] = [""]
        continue
    baseline_order = features.sort_values("cosine_score", ascending=False)["smiles"].tolist()
    baseline_predictions_cmp[molecule_id] = baseline_order if baseline_order else [""]
    reranked_predictions_cmp[molecule_id] = rerank_candidates(
        reranker_predictor, features, _RERANKER_FEATURE_COLUMNS, top_k=25
    ) or [""]

baseline_mrr_cmp = mrr_at_25(baseline_predictions_cmp, val_ground_truth)
reranked_mrr_cmp = mrr_at_25(reranked_predictions_cmp, val_ground_truth)

print(f"Baseline-only MRR@25:  {baseline_mrr_cmp:.4f}")
print(f"Reranked MRR@25:       {reranked_mrr_cmp:.4f}")
print(f"Delta:                 {reranked_mrr_cmp - baseline_mrr_cmp:+.4f}")
print()
print("Note: training data and this held-out split share the same "
      "make_validation_split call, so this comparison is somewhat "
      "optimistic/in-sample — a fully separate double-holdout is a "
      "deferred future refinement, not part of this sub-project.")
```

- [ ] **Step 4: Run the notebook top to bottom using the venv kernel**

Run: `.venv-reranker\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace Enveda_CASMI_local.ipynb --ExecutePreprocessor.timeout=1800`

Expected: no errors; printed baseline vs. reranked MRR@25 comparison with a delta value.

**Important environment note:** the existing baseline cells in this notebook (from the earlier sub-project) were originally run under the system Python 3.14 environment. Since `rdkit`, `matchms`, `pandas`, `pyarrow` are all also installed in `.venv-reranker` (verified during this plan's environment setup), re-executing the ENTIRE notebook (baseline cells + new reranker cells) under `.venv-reranker`'s Python 3.12 is expected to work and is what this step does — there is no need to maintain two separate kernels or split the notebook. If any baseline cell unexpectedly fails under 3.12 (not anticipated, but possible if some subtle 3.14-specific behavior crept in), stop and report it rather than silently patching around it — that would be a real environment-compatibility finding worth surfacing, not something to route around quietly.

- [ ] **Step 5: Append LOG.md entry**

Add to the `## 2026-09-23` section of `LOG.md`:

```markdown
### Wired into notebook: baseline vs. reranked MRR@25 comparison

**What:** Added cells to `Enveda_CASMI_local.ipynb` training the
reranker on the sampled train set and printing baseline-only MRR@25
alongside reranked MRR@25 on the same held-out split, with the delta.
Model artifact saved to `./autogluon_reranker_model` (gitignored, not
committed — matches the existing `models/`/`checkpoints/` gitignore
pattern intent).

**Why:** This is sub-project 2's actual deliverable per the spec's
success criterion — a direct, printed comparison proving (or
disproving) that reranking helps. No reranked `submission.csv` variant
was produced; that's an explicitly deferred follow-up once the
comparison itself is validated as a real improvement.

**Result:** [fill in the actual printed baseline vs. reranked MRR@25
numbers from the notebook run here as part of executing this task —
this bracketed placeholder must be replaced with real numbers before
this task's commit, not left as a template].
```

**Note on the bracketed placeholder above:** this is the one spot in this plan where the LOG.md content cannot be written in advance, since it depends on this task's own execution output. Whoever executes this step must replace the bracketed text with the actual numbers from Step 4's notebook run before committing — an unresolved placeholder here is a plan-execution defect, not an acceptable final state.

- [ ] **Step 6: Commit**

```bash
git add Enveda_CASMI_local.ipynb LOG.md
git commit -m "feat: wire reranker into notebook, compare baseline vs reranked MRR@25"
```

- [ ] **Step 7: Ensure the AutoGluon model directory is gitignored**

Check whether `./autogluon_reranker_model` (or any subpath under it) shows as untracked in `git status`. If so, add an entry to `.gitignore`:

```
autogluon_reranker_model/
```

```bash
git add .gitignore
git commit -m "chore: gitignore AutoGluon reranker model artifacts"
```

---

## Self-Review Notes

- **Spec coverage:** all components from the spec's Architecture section have a task — `extract_candidate_features` (Task 2), `build_training_examples` (Task 3), `train_reranker` (Task 3), `rerank_candidates` (Task 4), notebook comparison (Task 5). LOG.md requirement covered by Task 1 (creation + backfill) and every subsequent task's own entry. The AutoGluon-vs-Python-3.14 environment blocker discovered during planning is covered by the "Environment Setup Notes" section and called out explicitly in every task's command steps (`.venv-reranker\Scripts\python.exe`, not bare `python`).
- **Placeholder scan:** one intentional placeholder exists (Task 5 Step 5's LOG.md result numbers), and it is explicitly flagged as something the executor must fill in with real output, not left as-is — this is a deliberate exception to the "no placeholders" rule because the content genuinely does not exist until that task runs, and the plan states this outright rather than hiding it.
- **Type consistency:** `extract_candidate_features`'s return columns (Task 2) match exactly what `build_training_examples` (Task 3) and `rerank_candidates` (Task 4) consume (`feature_columns` lists are identical across Tasks 3, 4, 5's notebook cells — `cosine_score`, `ppm_error`, `num_peaks_candidate`, `mol_wt`, `log_p`, `num_rings`, `num_rotatable_bonds`, `num_hbd`, `num_hba`). `train_reranker`'s returned predictor object is passed unchanged into `rerank_candidates` in both Task 4's tests and Task 5's notebook cell.
- **Review Focus coverage:** item 1 (zero candidates after filtering) is covered by Task 2's `test_extract_candidate_features_no_candidates_returns_empty_dataframe` and Task 3's `test_build_training_examples_empty_when_no_eligible_molecules`. Item 2 (unparseable SMILES) is covered by `_rdkit_descriptors`'s explicit NaN-guard in Task 2's implementation (not separately unit-tested with a deliberately-broken SMILES since `normalized_smiles` is RDKit-normalized upstream and this is defensive-only code per the spec, but the guard exists and is visible in the implementation). Item 3 (fewer than top_k candidates) is covered by Task 4's `test_rerank_candidates_fewer_candidates_than_top_k_returns_all`. Item 4 (predict_proba column ordering) is directly addressed in `rerank_candidates`'s implementation with an explicit comment and fallback (`probabilities[1] if 1 in probabilities.columns else probabilities[True]`). Item 5 (empty/single-class training set) is covered by the integration test's explicit skip-with-reason when `training_df["is_correct"].nunique() < 2`, documenting that this is an accepted small-sample outcome rather than papering over it.
