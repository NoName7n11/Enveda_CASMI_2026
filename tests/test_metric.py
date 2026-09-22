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
