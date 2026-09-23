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
