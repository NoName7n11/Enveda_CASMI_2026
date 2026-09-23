"""AutoGluon-based reranker: richer per-candidate features + trained reordering."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from src.baseline import Spectrum, filter_candidates, to_matchms_spectrum, cosine_similarity
from src.data import make_validation_split

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
    # Candidates are re-filtered per test spectrum, but the same train_df
    # row and the same SMILES often reappear across a molecule's multiple
    # spectra. Cache the built Spectrum (by train_df index) and the RDKit
    # descriptors (by SMILES) so each is computed at most once per molecule
    # instead of once per (test spectrum, candidate) pair.
    candidate_spectrum_cache: dict[int, Spectrum] = {}
    descriptor_cache: dict[str, dict] = {}

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
                ppm_error = (
                    abs(candidate_row["precursor_mz"] - test_row["precursor_mz"])
                    / test_row["precursor_mz"]
                    * 1e6
                )
                num_peaks = candidate_row.get("num_peaks")
                if num_peaks is None or pd.isna(num_peaks):
                    num_peaks = len(candidate_row["ms2_mzs"])
                smiles = candidate_row["normalized_smiles"]
                if smiles not in descriptor_cache:
                    descriptor_cache[smiles] = _rdkit_descriptors(smiles)
                best_row[key] = {
                    "inchikey14": key,
                    "smiles": smiles,
                    "cosine_score": score,
                    "ppm_error": ppm_error,
                    "num_peaks_candidate": num_peaks,
                    **descriptor_cache[smiles],
                }

    if not best_row:
        return pd.DataFrame(columns=_FEATURE_COLUMNS)

    return pd.DataFrame(list(best_row.values()), columns=_FEATURE_COLUMNS)


def build_training_examples(
    train_df: pd.DataFrame,
    n_held_out: int = 200,
    random_state: int = 42,
    ppm_tolerance: float = 15.0,
) -> pd.DataFrame:
    """Build a labeled training set for the reranker from held-out molecules.

    Uses make_validation_split to get held-out query molecules, extracts
    candidate features for each, and labels each candidate row
    is_correct=1 if its inchikey14 matches that molecule's molecule_id
    (which IS the held-out molecule's true inchikey14), else 0. This
    matches the competition's actual scoring criterion (InChIKey14
    equality, see src/metric.py::mrr_at_25) rather than exact SMILES
    string equality.
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
        # Label by InChIKey14 equality against molecule_id, not by exact
        # SMILES string match. molecule_id IS the held-out molecule's
        # inchikey14 (see src.data.make_validation_split, where
        # val_query_df["molecule_id"] = val_query_df["inchikey14"]), and the
        # competition's actual scoring metric (src/metric.py::mrr_at_25)
        # matches by tautomer-canonicalized InChIKey14, not exact SMILES
        # text (stereo/tautomer differences are not penalized per
        # OVERVIEW.md). Comparing against exact SMILES here would be
        # stricter than the real scoring criterion, incorrectly labeling a
        # candidate wrong when it's the same structure but a different
        # tautomer/stereo-variant SMILES string.
        features["is_correct"] = (features["inchikey14"] == molecule_id).astype(int)
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
        label=label_column, path=model_path, problem_type="binary", eval_metric="roc_auc"
    )
    predictor.fit(training_df[feature_columns + [label_column]], time_limit=time_limit)
    return predictor


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
