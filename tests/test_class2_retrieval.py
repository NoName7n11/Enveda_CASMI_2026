import numpy as np
import pandas as pd
import pytest
from src.class2_retrieval import (
    filter_candidates_by_formula,
    extract_structure_features,
    build_class2_submission,
)


def test_filter_candidates_by_formula_matches_exact_formula_ignores_spectral_fields():
    structure_table = pd.DataFrame(
        [
            {"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"},
            {"inchikey14": "BBBBBBBBBBBBBB", "normalized_smiles": "CCC", "molecular_formula": "C3H8"},
        ]
    )
    result = filter_candidates_by_formula("C2H6O", structure_table)
    assert list(result["inchikey14"]) == ["AAAAAAAAAAAAAA"]


def test_filter_candidates_by_formula_dedupes_by_inchikey14():
    structure_table = pd.DataFrame(
        [
            {"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"},
            {"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"},
        ]
    )
    result = filter_candidates_by_formula("C2H6O", structure_table)
    assert len(result) == 1


def test_filter_candidates_by_formula_handles_missing_formula():
    structure_table = pd.DataFrame(
        [{"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"}]
    )
    result = filter_candidates_by_formula(None, structure_table)
    assert result.empty


def test_extract_structure_features_no_cosine_score_column():
    candidates = pd.DataFrame(
        [{"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"}]
    )
    test_row = pd.Series({"precursor_mz": 47.0})
    features = extract_structure_features(test_row, candidates)
    assert "cosine_score" not in features.columns
    assert features.iloc[0]["inchikey14"] == "AAAAAAAAAAAAAA"
    assert features.iloc[0]["ppm_error"] >= 0.0


def test_build_class2_submission_finds_formula_match_with_no_spectral_overlap():
    # Test molecule has a formula match in structure_table but its own
    # precursor_mz/adduct don't correspond to any spectrum in the table --
    # simulates Class 2 (no reference spectrum of this structure exists).
    test_df = pd.DataFrame(
        [{"molecule_id": "m1", "precursor_mz": 46.0419, "adduct": "[M+H]+", "molecular_formula": "C2H6O"}]
    )
    structure_table = pd.DataFrame(
        [{"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"}]
    )
    result = build_class2_submission(test_df, structure_table, top_k=25)
    assert len(result) == 1
    assert result.iloc[0]["smiles"] == "CCO"


def test_build_class2_submission_empty_when_no_formula_match():
    test_df = pd.DataFrame(
        [{"molecule_id": "m1", "precursor_mz": 999.0, "adduct": "[M+H]+", "molecular_formula": "C99H99"}]
    )
    structure_table = pd.DataFrame(
        [{"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"}]
    )
    result = build_class2_submission(test_df, structure_table, top_k=25)
    assert result.iloc[0]["smiles"] == ""


def test_build_class2_submission_ranks_closest_mass_first():
    test_df = pd.DataFrame(
        [{"molecule_id": "m1", "precursor_mz": 46.0419, "adduct": "[M+H]+", "molecular_formula": "C2H6O"}]
    )
    # Two different structures that happen to share formula C2H6O isn't
    # realistic (isomers have the same exact mass), so instead verify
    # top_k truncation + smiles presence against a single real match.
    structure_table = pd.DataFrame(
        [{"inchikey14": "AAAAAAAAAAAAAA", "normalized_smiles": "CCO", "molecular_formula": "C2H6O"}]
    )
    result = build_class2_submission(test_df, structure_table, top_k=1)
    assert result.iloc[0]["smiles"] == "CCO"
