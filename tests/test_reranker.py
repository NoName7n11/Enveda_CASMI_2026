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
