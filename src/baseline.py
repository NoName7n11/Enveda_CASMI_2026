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
    # Candidates are re-filtered per test spectrum (the ppm/adduct window
    # depends on the test row), but the same train_df row often reappears
    # across a molecule's multiple spectra. Cache each candidate's Spectrum
    # by its train_df index so it's built at most once per molecule instead
    # of once per (test spectrum, candidate) pair.
    candidate_spectrum_cache: dict[int, Spectrum] = {}

    for _, test_row in test_spectra_rows.iterrows():
        candidates = filter_candidates(test_row, train_df, ppm_tolerance=ppm_tolerance)
        if candidates.empty:
            continue
        test_spectrum = to_matchms_spectrum(
            test_row["ms2_mzs"], test_row["ms2_normalized_intensities"], metadata={}
        )
        for candidate_idx, candidate_row in candidates.iterrows():
            if candidate_idx not in candidate_spectrum_cache:
                candidate_spectrum_cache[candidate_idx] = to_matchms_spectrum(
                    candidate_row["ms2_mzs"],
                    candidate_row["ms2_normalized_intensities"],
                    metadata={},
                )
            candidate_spectrum = candidate_spectrum_cache[candidate_idx]
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
