"""Precursor-mass + cosine-similarity retrieval baseline.

Spectral cosine similarity is implemented directly in numpy rather than via
the `matchms` library. matchms's import chain (matchms -> exporting ->
scipy.sparse -> numpy.char -> numpy._core.strings -> numpy._core.umath)
proved unreliable on Kaggle's hosted notebook environment: across many
fresh-kernel, fresh-container attempts, plain `pip install matchms`
reproducibly broke numpy's own `_core.umath` module (ImportError:
cannot import name '_center'), independent of kernel state, CPU vs GPU
image, or numpy/scipy version pinning. Kaggle's own official tutorial
notebook for this competition (inversion/casmi-denovo-tutorial-notebook)
also avoids matchms, using rdkit fingerprints instead. Reimplementing the
one piece of matchms this project actually used -- CosineGreedy's
greedy tolerance-matched peak-pair cosine score -- removes that fragile
dependency entirely.
"""

import numpy as np
import pandas as pd

Spectrum = dict  # {"mz": np.ndarray (sorted ascending), "intensities": np.ndarray}


def to_matchms_spectrum(mzs, intensities, metadata: dict | None = None) -> Spectrum:
    """Build a Spectrum (sorted mz/intensity arrays) from raw mz/intensity data.

    `metadata` is accepted for call-site compatibility but unused -- only
    mz/intensity arrays are needed for the cosine similarity computed here.
    """
    mz_array = np.asarray(mzs, dtype=float)
    intensity_array = np.asarray(intensities, dtype=float)
    order = np.argsort(mz_array)
    return {"mz": mz_array[order], "intensities": intensity_array[order]}


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


def cosine_similarity(
    spectrum_a: Spectrum, spectrum_b: Spectrum, tolerance: float = 0.01
) -> float:
    """Greedy tolerance-matched cosine similarity between two spectra.

    Reimplements matchms.similarity.CosineGreedy's algorithm directly in
    numpy: for every peak pair within `tolerance` Da, compute an intensity
    product score; greedily accept the highest-scoring pairs first (each
    peak used at most once); normalize the summed matched-pair products by
    the full (unmatched-inclusive) norms of both spectra. Returns 0.0 if
    either spectrum has no peaks or no peaks match within tolerance.
    """
    mz_a, intensities_a = spectrum_a["mz"], spectrum_a["intensities"]
    mz_b, intensities_b = spectrum_b["mz"], spectrum_b["intensities"]
    if len(mz_a) == 0 or len(mz_b) == 0:
        return 0.0

    norm_a = np.linalg.norm(intensities_a)
    norm_b = np.linalg.norm(intensities_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # All candidate (i, j) peak pairs within tolerance, scored by intensity
    # product, sorted descending so the greedy pass accepts the best matches
    # first -- matching matchms's CosineGreedy pairing behavior.
    diffs = np.abs(mz_a[:, None] - mz_b[None, :])
    pair_i, pair_j = np.nonzero(diffs <= tolerance)
    if len(pair_i) == 0:
        return 0.0

    pair_scores = intensities_a[pair_i] * intensities_b[pair_j]
    order = np.argsort(-pair_scores)

    used_a = np.zeros(len(mz_a), dtype=bool)
    used_b = np.zeros(len(mz_b), dtype=bool)
    matched_sum = 0.0
    for k in order:
        i, j = pair_i[k], pair_j[k]
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = True
        used_b[j] = True
        matched_sum += pair_scores[k]

    return float(matched_sum / (norm_a * norm_b))


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
