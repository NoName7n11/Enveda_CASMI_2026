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
