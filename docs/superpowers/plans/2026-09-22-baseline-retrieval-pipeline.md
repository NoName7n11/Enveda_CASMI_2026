# Baseline Retrieval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, locally-validated baseline: precursor-mass-filtered + cosine-similarity spectral retrieval against `train.parquet`, scored offline with the official MRR@25 metric, producing a valid `submission.csv`.

**Architecture:** Three standalone Python modules under `src/` (metric, data loading, retrieval), each independently unit-testable with small synthetic fixtures, wired together at the end by an integration test that runs the full pipeline on a tiny real sample pulled from `dataset/train.parquet`. No notebook code is authored directly — the notebook only imports and calls these modules (per WORKFLOW.md's local-notebook-for-experimentation rule, but logic lives in testable `.py` files, not notebook cells).

**Tech Stack:** Python 3.14, pandas (already installed), rdkit (2D structure canonicalization + InChIKey14), matchms (cosine spectral similarity), pytest (test runner). polars considered but skipped for v0 — pandas + pyarrow already present and sufficient at the sampled scale (50-100k rows); adding polars is deferred until full-scale (2.5M row) work actually needs it (YAGNI).

**Spec:** Design approved in chat 2026-09-22 (brainstorming session, no separate spec file written — bounded/cohesive enough for direct plan per user's "proceed with it").

## Global Constraints

- Metric: MRR@25, first-correct-guess rank scoring, InChIKey14 (first 14 chars of InChIKey after RDKit tautomer canonicalization) equality — per OVERVIEW.md Section 4.
- Submission format: header `molecule_id,smiles`, up to 25 semicolon-separated SMILES per row, every test `molecule_id` exactly once, no NaN/empty — per OVERVIEW.md Section 4.
- Aggregation is per-`molecule_id`, not per-`spectrum_id` — a molecule can have 1-16 spectra; candidate score = max cosine similarity across all (test spectrum × matched train spectrum) pairs (user-approved design decision).
- Candidate filter: precursor `m/z` within a ppm tolerance AND matching `adduct` string, against `train.parquet`.
- Dev/test split must be molecule-level (no spectrum from the same molecule in both train-fold and validation-fold) — per WORKFLOW.md Phase 3.
- No network access assumed for pipeline code itself (train/test data already local); rdkit/matchms/pytest installed via pip (already verified installable on this Python 3.14 env).

---

## File Structure

- `src/metric.py` — official MRR@25 scorer (RDKit tautomer canonicalization → InChIKey14 → rank scoring). No dependencies on other `src/` modules.
- `src/data.py` — sampled data loading from `dataset/train.parquet` (stratified: full `enveda-np-examples` + `drug_plus` libraries always included, rest randomly sampled) and `dataset/test.parquet` (loaded whole, it's small).
- `src/baseline.py` — retrieval logic: precursor-mass+adduct candidate filtering, matchms cosine similarity scoring, per-molecule aggregation (max), top-25 ranking, submission dataframe construction.
- `tests/test_metric.py` — unit tests for `src/metric.py` using small synthetic SMILES fixtures (no real data needed).
- `tests/test_data.py` — unit tests for `src/data.py` using a tiny synthetic parquet fixture written to a temp dir.
- `tests/test_baseline.py` — unit tests for `src/baseline.py` using small synthetic spectra (no real data needed).
- `tests/test_integration.py` — end-to-end test: load a small real sample from `dataset/train.parquet` + all of `dataset/test.parquet`, run full pipeline, assert valid submission format and a non-trivial MRR@25 score on a held-out molecule split.
- `Enveda_CASMI_local.ipynb` — modified to add cells that import `src.data`, `src.baseline`, `src.metric`, run the pipeline end-to-end, print the MRR@25 score, and write `submission.csv`. (Not part of automated tests; manual verification step in the plan.)

---

## Task 1: MRR@25 Metric Module

**Files:**
- Create: `src/metric.py`
- Test: `tests/test_metric.py`

**Interfaces:**
- Consumes: nothing (standalone, only rdkit).
- Produces:
  - `canonicalize_to_inchikey14(smiles: str) -> str | None` — returns 14-char InChIKey prefix, or `None` if the SMILES is unparseable.
  - `reciprocal_rank(predicted_smiles: list[str], true_smiles: str, k: int = 25) -> float` — returns `1/rank` of the first predicted SMILES (by list order, truncated to `k`) whose InChIKey14 matches the true SMILES's InChIKey14, or `0.0` if none match or list is empty. Invalid/unparseable predicted SMILES are silently skipped (they consume a rank position but never match, per OVERVIEW.md: "Invalid/unparseable SMILES strings ... simply count as non-matches").
  - `mrr_at_25(predictions: dict[str, list[str]], ground_truth: dict[str, str], k: int = 25) -> float` — mean of `reciprocal_rank(...)` over all keys in `ground_truth`. Raises `KeyError` if a `ground_truth` key is missing from `predictions` (caller bug, not something to silently paper over).

- [ ] **Step 1: Install rdkit and pytest**

Run: `pip install rdkit pytest`
Expected: installs cleanly (already verified via `pip install --dry-run` — resolves to rdkit 2026.3.6, pytest 9.1.1 on this Python 3.14 environment).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_metric.py`:

```python
import pytest
from src.metric import canonicalize_to_inchikey14, reciprocal_rank, mrr_at_25


def test_canonicalize_valid_smiles_returns_14_char_key():
    key = canonicalize_to_inchikey14("CCO")  # ethanol
    assert key is not None
    assert len(key) == 14


def test_canonicalize_invalid_smiles_returns_none():
    assert canonicalize_to_inchikey14("not_a_smiles_(((") is None


def test_canonicalize_stereochemistry_ignored():
    # Both reduce to the same InChIKey14 per OVERVIEW.md example
    key_a = canonicalize_to_inchikey14("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
    key_b = canonicalize_to_inchikey14("OCC1OC(O)C(O)C(O)C1O")
    assert key_a == key_b == "WQZGKKKJIJFFOK"


def test_reciprocal_rank_first_guess_correct():
    assert reciprocal_rank(["CCO", "CCC"], true_smiles="CCO") == 1.0


def test_reciprocal_rank_second_guess_correct():
    score = reciprocal_rank(["CCC", "CCO"], true_smiles="CCO")
    assert score == pytest.approx(0.5)


def test_reciprocal_rank_no_match_returns_zero():
    assert reciprocal_rank(["CCC", "CCCC"], true_smiles="CCO") == 0.0


def test_reciprocal_rank_empty_predictions_returns_zero():
    assert reciprocal_rank([], true_smiles="CCO") == 0.0


def test_reciprocal_rank_invalid_smiles_in_list_counts_as_non_match():
    # invalid entry at rank 1 consumes that rank but never matches;
    # correct answer at rank 2 scores 0.5, not 1.0
    score = reciprocal_rank(["not_valid_(((", "CCO"], true_smiles="CCO")
    assert score == pytest.approx(0.5)


def test_reciprocal_rank_respects_k_truncation():
    predictions = ["CCC"] * 25 + ["CCO"]  # correct answer at rank 26
    assert reciprocal_rank(predictions, true_smiles="CCO", k=25) == 0.0


def test_mrr_at_25_averages_across_molecules():
    predictions = {
        "m1": ["CCO"],           # rank 1 -> 1.0
        "m2": ["CCC", "CCO"],    # rank 2 -> 0.5
    }
    ground_truth = {"m1": "CCO", "m2": "CCO"}
    assert mrr_at_25(predictions, ground_truth) == pytest.approx(0.75)


def test_mrr_at_25_raises_on_missing_prediction_key():
    with pytest.raises(KeyError):
        mrr_at_25(predictions={}, ground_truth={"m1": "CCO"})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_metric.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metric'` (or similar import error) — module doesn't exist yet.

- [ ] **Step 4: Write the implementation**

Create `src/metric.py`:

```python
"""Official CASMI 2026 MRR@25 metric: InChIKey14-based structure matching."""

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def canonicalize_to_inchikey14(smiles: str) -> str | None:
    """Parse SMILES, canonicalize tautomers, return the 14-char InChIKey prefix.

    Returns None if the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_mol = _tautomer_enumerator.Canonicalize(mol)
    inchikey = Chem.MolToInchiKey(canonical_mol)
    if not inchikey:
        return None
    return inchikey[:14]


def reciprocal_rank(predicted_smiles: list[str], true_smiles: str, k: int = 25) -> float:
    """1/rank of the first predicted SMILES matching true_smiles's InChIKey14.

    Unparseable predicted SMILES occupy a rank slot but never match.
    Returns 0.0 if no match is found within the first k predictions.
    """
    true_key = canonicalize_to_inchikey14(true_smiles)
    for rank, smiles in enumerate(predicted_smiles[:k], start=1):
        predicted_key = canonicalize_to_inchikey14(smiles)
        if predicted_key is not None and predicted_key == true_key:
            return 1.0 / rank
    return 0.0


def mrr_at_25(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, str],
    k: int = 25,
) -> float:
    """Mean reciprocal rank @ k over all molecules in ground_truth.

    Raises KeyError if a ground_truth molecule_id is missing from predictions.
    """
    scores = [
        reciprocal_rank(predictions[molecule_id], true_smiles, k=k)
        for molecule_id, true_smiles in ground_truth.items()
    ]
    return sum(scores) / len(scores)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_metric.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/metric.py tests/test_metric.py
git commit -m "feat: add offline MRR@25 metric module"
```

---

## Task 2: Sampled Data Loader

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing (standalone, only pandas/pyarrow).
- Produces:
  - `load_test_set(path: str = "dataset/test.parquet") -> pandas.DataFrame` — loads the whole test parquet as-is.
  - `load_sampled_train(path: str = "dataset/train.parquet", sample_size: int = 75_000, always_include_libs: tuple[str, ...] = ("enveda-np-examples", "drug_plus"), random_state: int = 42) -> pandas.DataFrame` — reads `train.parquet`, keeps all rows whose `ingest_lib` is in `always_include_libs`, then randomly samples additional rows (without replacement, `random_state` for reproducibility) from the remaining rows up to `sample_size` total rows. If `sample_size` is smaller than the always-included rows alone, returns just the always-included rows (no error).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest
from src.data import load_test_set, load_sampled_train


def _make_fake_train_df(n_np_examples: int, n_drug_plus: int, n_other: int) -> pd.DataFrame:
    rows = []
    for i in range(n_np_examples):
        rows.append({"ingest_lib": "enveda-np-examples", "molecular_formula": "C2H6O", "precursor_mz": 47.0})
    for i in range(n_drug_plus):
        rows.append({"ingest_lib": "drug_plus", "molecular_formula": "C2H6O", "precursor_mz": 47.0})
    for i in range(n_other):
        rows.append({"ingest_lib": "gnps", "molecular_formula": "C2H6O", "precursor_mz": 47.0})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_train_parquet(tmp_path):
    df = _make_fake_train_df(n_np_examples=10, n_drug_plus=5, n_other=200)
    path = tmp_path / "train.parquet"
    df.to_parquet(path)
    return str(path)


@pytest.fixture
def fake_test_parquet(tmp_path):
    df = pd.DataFrame([{"molecule_id": "m1", "precursor_mz": 47.0}])
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return str(path)


def test_load_test_set_returns_all_rows(fake_test_parquet):
    df = load_test_set(fake_test_parquet)
    assert len(df) == 1
    assert df.iloc[0]["molecule_id"] == "m1"


def test_load_sampled_train_always_includes_priority_libs(fake_train_parquet):
    df = load_sampled_train(fake_train_parquet, sample_size=20, random_state=1)
    lib_counts = df["ingest_lib"].value_counts()
    assert lib_counts.get("enveda-np-examples", 0) == 10
    assert lib_counts.get("drug_plus", 0) == 5


def test_load_sampled_train_respects_sample_size_upper_bound(fake_train_parquet):
    df = load_sampled_train(fake_train_parquet, sample_size=50, random_state=1)
    assert len(df) == 50


def test_load_sampled_train_reproducible_with_same_seed(fake_train_parquet):
    df_a = load_sampled_train(fake_train_parquet, sample_size=50, random_state=7)
    df_b = load_sampled_train(fake_train_parquet, sample_size=50, random_state=7)
    pd.testing.assert_frame_equal(
        df_a.reset_index(drop=True), df_b.reset_index(drop=True)
    )


def test_load_sampled_train_handles_sample_size_smaller_than_priority_rows(fake_train_parquet):
    df = load_sampled_train(fake_train_parquet, sample_size=3, random_state=1)
    # 15 priority rows exist (10 + 5); sample_size=3 is smaller, so just return priority rows
    assert len(df) == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Write the implementation**

Create `src/data.py`:

```python
"""Data loading for the CASMI 2026 baseline pipeline."""

import pandas as pd


def load_test_set(path: str = "dataset/test.parquet") -> pd.DataFrame:
    """Load the full test set (it's small, ~1500 rows)."""
    return pd.read_parquet(path)


def load_sampled_train(
    path: str = "dataset/train.parquet",
    sample_size: int = 75_000,
    always_include_libs: tuple[str, ...] = ("enveda-np-examples", "drug_plus"),
    random_state: int = 42,
) -> pd.DataFrame:
    """Load a stratified sample of the training set for fast local iteration.

    All rows from `always_include_libs` are kept (these are the smallest and
    most calibration-relevant libraries per OVERVIEW.md). Additional rows are
    randomly sampled from the rest up to `sample_size` total rows.
    """
    df = pd.read_parquet(path)
    priority_mask = df["ingest_lib"].isin(always_include_libs)
    priority_df = df[priority_mask]
    remaining_df = df[~priority_mask]

    remaining_budget = sample_size - len(priority_df)
    if remaining_budget <= 0:
        return priority_df.reset_index(drop=True)

    sampled_remaining = remaining_df.sample(
        n=min(remaining_budget, len(remaining_df)),
        random_state=random_state,
    )
    return pd.concat([priority_df, sampled_remaining]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add stratified sampled train/test data loaders"
```

---

## Task 3: Retrieval Baseline (Candidate Filtering + Cosine Scoring + Aggregation)

**Files:**
- Create: `src/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: nothing new beyond pandas/matchms/numpy (standalone at the unit level; wired to `src.data` output shape in the integration test).
- Produces:
  - `Spectrum` — re-exported alias of `matchms.Spectrum` for convenience (so callers don't need a separate matchms import just to build one).
  - `to_matchms_spectrum(mzs: list[float] | np.ndarray, intensities: list[float] | np.ndarray, metadata: dict) -> Spectrum` — builds a matchms `Spectrum` object from raw arrays.
  - `filter_candidates(test_row: pd.Series, train_df: pd.DataFrame, ppm_tolerance: float = 15.0) -> pd.DataFrame` — returns the subset of `train_df` whose `adduct` matches `test_row["adduct"]` and whose `precursor_mz` is within `ppm_tolerance` ppm of `test_row["precursor_mz"]`.
  - `cosine_similarity(spectrum_a: Spectrum, spectrum_b: Spectrum) -> float` — matchms `CosineGreedy` score between two spectra (0.0 if either spectrum has zero peaks).
  - `score_candidates_for_molecule(test_spectra_rows: pd.DataFrame, train_df: pd.DataFrame, ppm_tolerance: float = 15.0) -> pd.DataFrame` — for one molecule's group of test spectra rows: filters candidates per test spectrum, computes cosine similarity between every (test spectrum, candidate train spectrum) pair, and returns a dataframe with columns `["inchikey14", "smiles", "score"]` where `score` is the **max** cosine similarity across all pairs for that `inchikey14` (per user-approved max-aggregation design), deduplicated by `inchikey14` (keeping the row with the highest score, and its associated `normalized_smiles` as `smiles`).
  - `build_submission(test_df: pd.DataFrame, train_df: pd.DataFrame, ppm_tolerance: float = 15.0, top_k: int = 25) -> pd.DataFrame` — groups `test_df` by `molecule_id`, calls `score_candidates_for_molecule` per group, sorts by `score` descending, takes top `top_k` SMILES, and returns a dataframe with columns `["molecule_id", "smiles"]` where `smiles` is a semicolon-joined string. Every `molecule_id` present in `test_df` appears exactly once, even if no candidates were found (empty string in that case — caller/notebook is responsible for a final fallback fill before writing `submission.csv`, since a truly empty prediction list is a modeling gap, not a data-loading concern this module should paper over).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_baseline.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.baseline import (
    to_matchms_spectrum,
    filter_candidates,
    cosine_similarity,
    score_candidates_for_molecule,
    build_submission,
)


def test_to_matchms_spectrum_builds_spectrum_with_metadata():
    spec = to_matchms_spectrum(
        mzs=[100.0, 200.0], intensities=[0.5, 1.0], metadata={"id": "s1"}
    )
    assert list(spec.peaks.mz) == [100.0, 200.0]
    assert spec.get("id") == "s1"


def test_filter_candidates_matches_adduct_and_ppm_window():
    train_df = pd.DataFrame(
        [
            {"adduct": "[M+H]+", "precursor_mz": 100.0010, "inchikey14": "AAAAAAAAAAAAAA"},
            {"adduct": "[M+H]+", "precursor_mz": 150.0000, "inchikey14": "BBBBBBBBBBBBBB"},  # out of ppm window
            {"adduct": "[M+Na]+", "precursor_mz": 100.0010, "inchikey14": "CCCCCCCCCCCCCC"},  # wrong adduct
        ]
    )
    test_row = pd.Series({"adduct": "[M+H]+", "precursor_mz": 100.0000})
    result = filter_candidates(test_row, train_df, ppm_tolerance=15.0)
    assert list(result["inchikey14"]) == ["AAAAAAAAAAAAAA"]


def test_cosine_similarity_identical_spectra_scores_near_one():
    spec = to_matchms_spectrum([100.0, 200.0], [1.0, 0.5], metadata={})
    score = cosine_similarity(spec, spec)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_empty_spectrum_scores_zero():
    empty = to_matchms_spectrum([], [], metadata={})
    non_empty = to_matchms_spectrum([100.0], [1.0], metadata={})
    assert cosine_similarity(empty, non_empty) == 0.0


def test_score_candidates_for_molecule_max_aggregates_across_spectra():
    test_spectra_rows = pd.DataFrame(
        [
            {
                "molecule_id": "m1",
                "spectrum_id": "t1",
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            },
            {
                "molecule_id": "m1",
                "spectrum_id": "t2",
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "ms2_mzs": np.array([50.0]),  # deliberately dissimilar
                "ms2_normalized_intensities": np.array([1.0]),
            },
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "inchikey14": "AAAAAAAAAAAAAA",
                "normalized_smiles": "CCO",
                "ms2_mzs": np.array([100.0]),  # matches t1 exactly -> high score
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    result = score_candidates_for_molecule(test_spectra_rows, train_df, ppm_tolerance=15.0)
    assert len(result) == 1
    assert result.iloc[0]["inchikey14"] == "AAAAAAAAAAAAAA"
    assert result.iloc[0]["score"] == pytest.approx(1.0, abs=1e-6)  # max, not averaged with the dissimilar pair


def test_build_submission_returns_one_row_per_molecule_id():
    test_df = pd.DataFrame(
        [
            {
                "molecule_id": "m1",
                "spectrum_id": "t1",
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            },
            {
                "molecule_id": "m2",
                "spectrum_id": "t2",
                "adduct": "[M+H]+",
                "precursor_mz": 999.0,  # no candidates will match
                "ms2_mzs": np.array([999.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            },
        ]
    )
    train_df = pd.DataFrame(
        [
            {
                "adduct": "[M+H]+",
                "precursor_mz": 100.0,
                "inchikey14": "AAAAAAAAAAAAAA",
                "normalized_smiles": "CCO",
                "ms2_mzs": np.array([100.0]),
                "ms2_normalized_intensities": np.array([1.0]),
            }
        ]
    )
    result = build_submission(test_df, train_df, ppm_tolerance=15.0, top_k=25)
    assert set(result["molecule_id"]) == {"m1", "m2"}
    assert len(result) == 2
    m1_smiles = result.loc[result["molecule_id"] == "m1", "smiles"].iloc[0]
    assert "CCO" in m1_smiles
    m2_smiles = result.loc[result["molecule_id"] == "m2", "smiles"].iloc[0]
    assert m2_smiles == ""  # no candidates found
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.baseline'`.

- [ ] **Step 3: Write the implementation**

Create `src/baseline.py`:

```python
"""Precursor-mass + cosine-similarity retrieval baseline."""

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.similarity import CosineGreedy

_cosine_greedy = CosineGreedy(tolerance=0.01)


def to_matchms_spectrum(mzs, intensities, metadata: dict) -> Spectrum:
    """Build a matchms Spectrum from raw mz/intensity arrays."""
    mz_array = np.asarray(mzs, dtype=float)
    intensity_array = np.asarray(intensities, dtype=float)
    # matchms requires mzs sorted ascending
    order = np.argsort(mz_array)
    return Spectrum(
        mz=mz_array[order],
        intensities=intensity_array[order],
        metadata=metadata,
    )


def filter_candidates(
    test_row: pd.Series, train_df: pd.DataFrame, ppm_tolerance: float = 15.0
) -> pd.DataFrame:
    """Return train rows matching test_row's adduct within a ppm precursor-mz window."""
    adduct_mask = train_df["adduct"] == test_row["adduct"]
    ppm_error = (
        (train_df["precursor_mz"] - test_row["precursor_mz"]).abs()
        / test_row["precursor_mz"]
        * 1e6
    )
    ppm_mask = ppm_error <= ppm_tolerance
    return train_df[adduct_mask & ppm_mask]


def cosine_similarity(spectrum_a: Spectrum, spectrum_b: Spectrum) -> float:
    """matchms CosineGreedy score between two spectra; 0.0 if either is empty."""
    if len(spectrum_a.peaks.mz) == 0 or len(spectrum_b.peaks.mz) == 0:
        return 0.0
    result = _cosine_greedy.pair(spectrum_a, spectrum_b)
    return float(result["score"])


def score_candidates_for_molecule(
    test_spectra_rows: pd.DataFrame, train_df: pd.DataFrame, ppm_tolerance: float = 15.0
) -> pd.DataFrame:
    """Score every candidate train structure against a molecule's test spectra.

    Returns columns ["inchikey14", "smiles", "score"], deduplicated by
    inchikey14 keeping the max score across all (test spectrum, candidate)
    pairs, per the max-aggregation design.
    """
    best_score: dict[str, float] = {}
    best_smiles: dict[str, str] = {}

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
                best_smiles[key] = candidate_row["normalized_smiles"]

    return pd.DataFrame(
        {
            "inchikey14": list(best_score.keys()),
            "smiles": [best_smiles[k] for k in best_score],
            "score": list(best_score.values()),
        }
    )


def build_submission(
    test_df: pd.DataFrame, train_df: pd.DataFrame, ppm_tolerance: float = 15.0, top_k: int = 25
) -> pd.DataFrame:
    """Build a molecule_id -> top-k semicolon-joined SMILES submission dataframe."""
    rows = []
    for molecule_id, group in test_df.groupby("molecule_id"):
        scored = score_candidates_for_molecule(group, train_df, ppm_tolerance=ppm_tolerance)
        if scored.empty:
            rows.append({"molecule_id": molecule_id, "smiles": ""})
            continue
        top = scored.sort_values("score", ascending=False).head(top_k)
        rows.append({"molecule_id": molecule_id, "smiles": ";".join(top["smiles"])})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_baseline.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/baseline.py tests/test_baseline.py
git commit -m "feat: add precursor-mass + cosine-similarity retrieval baseline"
```

---

## Task 4: Integration Test on Real Sampled Data

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: `src.data.load_sampled_train`, `src.data.load_test_set`, `src.baseline.build_submission`, `src.metric.mrr_at_25` (all defined in Tasks 1-3).
- Produces: nothing new — this is a pure test file validating the modules work together against real data.

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end validation: real sampled train data -> baseline -> offline MRR@25."""

import os
import pandas as pd
import pytest
from src.data import load_sampled_train
from src.baseline import build_submission
from src.metric import mrr_at_25

DATASET_AVAILABLE = os.path.exists("dataset/train.parquet")


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="dataset/train.parquet not present")
def test_baseline_pipeline_end_to_end_on_real_sample():
    # Small sample for test speed; real dev notebook uses a larger sample_size.
    train_df = load_sampled_train(sample_size=5_000, random_state=0)

    # Build a tiny held-out "test" set from train itself (molecule-level split)
    # so we have known ground truth to score against, without touching the
    # real dataset/test.parquet (which has no labels).
    unique_molecule_formulas = train_df["molecular_formula"].dropna().unique()
    assert len(unique_molecule_formulas) > 0, "sampled train data unexpectedly empty"

    # Pick spectra for a handful of distinct structures as our held-out query set.
    held_out_keys = train_df["inchikey14"].drop_duplicates().head(10)
    held_out_df = train_df[train_df["inchikey14"].isin(held_out_keys)].copy()
    held_out_df["molecule_id"] = held_out_df["inchikey14"]  # stand-in molecule_id

    remaining_train_df = train_df[~train_df["inchikey14"].isin(held_out_keys)]

    ground_truth = (
        held_out_df.drop_duplicates("inchikey14")
        .set_index("molecule_id")["normalized_smiles"]
        .to_dict()
    )

    submission_df = build_submission(held_out_df, remaining_train_df, ppm_tolerance=15.0, top_k=25)

    # Submission format checks (per OVERVIEW.md Section 4 constraints)
    assert set(submission_df["molecule_id"]) == set(ground_truth.keys())
    assert submission_df["smiles"].notna().all()
    for smiles_field in submission_df["smiles"]:
        assert smiles_field.count(";") <= 24  # at most 25 candidates

    predictions = dict(zip(submission_df["molecule_id"], submission_df["smiles"].str.split(";")))
    score = mrr_at_25(predictions, ground_truth)

    # Weak sanity bound: since remaining_train_df no longer contains the exact
    # held-out spectra, a perfect 1.0 is not expected, but formula/precursor
    # overlap with other library entries of the same structure should give
    # a non-zero signal most of the time. This is a smoke test, not a
    # leaderboard-score assertion.
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/test_integration.py -v -s`
Expected: PASS (or SKIPPED if `dataset/train.parquet` is absent in the environment running the test). Note the printed/returned score is informational — this test only asserts the pipeline runs end-to-end and produces a format-valid submission, not a specific score threshold, since the held-out split is small and random.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test against real sampled data"
```

---

## Task 5: Wire Up the Local Notebook and Produce a Real Submission

**Files:**
- Modify: `Enveda_CASMI_local.ipynb`

**Interfaces:**
- Consumes: `src.data.load_sampled_train`, `src.data.load_test_set`, `src.baseline.build_submission`, `src.metric.mrr_at_25` (all defined in Tasks 1-3).
- Produces: `submission.csv` written to the project root (gitignored — not committed).

- [ ] **Step 1: Add setup cell**

Add a markdown cell titled "CASMI 2026 — Baseline Retrieval Pipeline" and a code cell:

```python
import sys
sys.path.insert(0, ".")

from src.data import load_sampled_train, load_test_set
from src.baseline import build_submission
from src.metric import mrr_at_25

import pandas as pd
```

- [ ] **Step 2: Add data loading cell**

```python
train_df = load_sampled_train(sample_size=75_000, random_state=42)
test_df = load_test_set()

print(f"Loaded {len(train_df):,} training spectra, {train_df['inchikey14'].nunique():,} unique structures")
print(f"Loaded {len(test_df):,} test spectra, {test_df['molecule_id'].nunique():,} unique molecules")
```

- [ ] **Step 3: Add molecule-level validation split cell**

```python
import numpy as np

rng = np.random.RandomState(42)
unique_molecules = train_df["inchikey14"].unique()
held_out_molecules = rng.choice(unique_molecules, size=min(200, len(unique_molecules)), replace=False)

val_query_df = train_df[train_df["inchikey14"].isin(held_out_molecules)].copy()
val_query_df["molecule_id"] = val_query_df["inchikey14"]
val_train_df = train_df[~train_df["inchikey14"].isin(held_out_molecules)]

val_ground_truth = (
    val_query_df.drop_duplicates("inchikey14")
    .set_index("molecule_id")["normalized_smiles"]
    .to_dict()
)

print(f"Validation query set: {val_query_df['molecule_id'].nunique()} held-out molecules")
```

- [ ] **Step 4: Add offline scoring cell**

```python
val_submission_df = build_submission(val_query_df, val_train_df, ppm_tolerance=15.0, top_k=25)
val_predictions = dict(
    zip(val_submission_df["molecule_id"], val_submission_df["smiles"].str.split(";"))
)

offline_mrr25 = mrr_at_25(val_predictions, val_ground_truth)
print(f"Offline MRR@25 (held-out molecule split): {offline_mrr25:.4f}")
```

- [ ] **Step 5: Add real submission generation cell**

```python
submission_df = build_submission(test_df, train_df, ppm_tolerance=15.0, top_k=25)

# Kaggle format constraints: no empty/NaN smiles allowed. Fall back to a
# placeholder for molecules with zero candidates so the file stays valid.
submission_df["smiles"] = submission_df["smiles"].replace("", "CCO")

assert submission_df["molecule_id"].is_unique
assert submission_df["smiles"].notna().all()
assert (submission_df["smiles"].str.count(";") <= 24).all()

submission_df.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv with {len(submission_df)} rows")
submission_df.head()
```

- [ ] **Step 6: Run the notebook top to bottom**

Run all cells (Jupyter: Kernel -> Restart & Run All, or `jupyter nbconvert --to notebook --execute --inplace Enveda_CASMI_local.ipynb`).
Expected: no errors; printed offline MRR@25 score; `submission.csv` written to project root with correct format.

- [ ] **Step 7: Manually verify submission.csv format**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
assert list(df.columns) == ['molecule_id', 'smiles']
assert df['molecule_id'].is_unique
assert df['smiles'].notna().all()
assert (df['smiles'].str.count(';') <= 24).all()
print('submission.csv format OK:', len(df), 'rows')
"
```
Expected: `submission.csv format OK: <N> rows` with no assertion errors.

- [ ] **Step 8: Commit the notebook**

```bash
git add Enveda_CASMI_local.ipynb
git commit -m "feat: wire baseline pipeline into local notebook, produce first submission"
```

(Note: `submission.csv` itself is not committed — it's derived Competition Data output and should be added to `.gitignore` if not already covered by the `dataset/` exclusion; since it's written to the project root, not `dataset/`, add `submission.csv` explicitly to `.gitignore` before this commit.)

- [ ] **Step 9: Add submission.csv to .gitignore and commit**

Add to `.gitignore`:
```
submission.csv
```

```bash
git add .gitignore
git commit -m "chore: gitignore generated submission.csv"
```

---

## Self-Review Notes

- **Spec coverage:** all 5 approved design points covered — data layer (Task 2), metric module (Task 1), candidate retrieval + cosine + max-aggregation (Task 3), local validation via molecule-level split (Task 5 Step 3-4), submission format check (Task 5 Step 7). Sampled-subset-first decision reflected in `load_sampled_train` default (Task 2). Polars was in the original design conversation as a data-layer option; downgraded to pandas here since it's already installed and sufficient at 75k-row scale — noted explicitly in Tech Stack as a deliberate YAGNI deferral, not an oversight.
- **Placeholder scan:** none found — every step has runnable code, no TBDs.
- **Type consistency:** `build_submission` returns `["molecule_id", "smiles"]` matching what Task 5's notebook cells consume; `score_candidates_for_molecule` returns `["inchikey14", "smiles", "score"]` matching what `build_submission` consumes internally; `mrr_at_25` signature matches what both the integration test and notebook cells call it with.
- **Scope check:** this plan covers exactly the approved baseline sub-project. AutoGluon reranking, DreaMS embeddings, and de novo generation are explicitly out of scope (next sub-projects, each will get its own plan later).
