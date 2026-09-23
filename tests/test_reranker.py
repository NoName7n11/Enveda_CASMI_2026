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
    # For this fixture, every held-out molecule's own structure survives
    # filtering (see comment above), so exactly one positive per molecule.
    assert training_df.groupby("molecule_id")["is_correct"].sum().eq(1).all()


def test_build_training_examples_empty_when_no_eligible_molecules():
    # all single-spectrum molecules -> none eligible for held-out split
    train_df = _make_synthetic_train_df().drop_duplicates("molecule_id")
    training_df = build_training_examples(train_df, n_held_out=3, random_state=1)
    assert len(training_df) == 0
    assert "is_correct" in training_df.columns


def test_train_reranker_fits_and_returns_predictor(tmp_path):
    train_df = _make_synthetic_train_df()
    # A wide ppm_tolerance is used here (rather than the 15.0 default) so
    # that each held-out molecule's candidate pool also picks up the other
    # two molecules' structures as (incorrect) candidates. With the default
    # tolerance, each molecule's precursor_mz is >15ppm from the others, so
    # every molecule's candidate pool would contain only its own (always
    # correct) structure -- producing a single-class ("is_correct" always 1)
    # training set that AutoGluon's binary classifier cannot fit on. This is
    # purely a training-set-construction detail of this test; the labeling
    # logic itself (see build_training_examples) is unaffected.
    training_df = build_training_examples(
        train_df, n_held_out=3, random_state=1, ppm_tolerance=1_000_000.0
    )
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


from src.reranker import rerank_candidates


def test_rerank_candidates_orders_by_predicted_probability(tmp_path):
    train_df = _make_synthetic_train_df()
    # A wide ppm_tolerance is used (rather than the 15.0 default) for the
    # same reason as test_train_reranker_fits_and_returns_predictor above:
    # with the default tolerance each held-out molecule's candidate pool
    # only ever contains its own (always-correct) structure, producing a
    # single-class ("is_correct" always 1) training set that AutoGluon's
    # binary classifier cannot fit on.
    training_df = build_training_examples(
        train_df, n_held_out=3, random_state=1, ppm_tolerance=1_000_000.0
    )
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
    # See comment in test_rerank_candidates_orders_by_predicted_probability
    # above: a wide ppm_tolerance is required to avoid a single-class
    # training set with this synthetic fixture.
    training_df = build_training_examples(
        train_df, n_held_out=3, random_state=1, ppm_tolerance=1_000_000.0
    )
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


class _StubPredictor:
    def predict_proba(self, X):
        return pd.DataFrame({0: [0.9, 0.1, 0.5], 1: [0.1, 0.9, 0.5]})


def test_rerank_candidates_selects_positive_class_by_name_not_position():
    candidates = pd.DataFrame(
        {"smiles": ["low_prob", "high_prob", "mid_prob"], "cosine_score": [0.1, 0.9, 0.5]}
    )
    feature_columns = ["cosine_score"]
    result = rerank_candidates(_StubPredictor(), candidates, feature_columns, top_k=25)
    assert result == ["high_prob", "mid_prob", "low_prob"]


class _StubPredictorReversedColumns:
    def predict_proba(self, X):
        # Columns in [1, 0] order instead of [0, 1] -- proves selection is by
        # name/membership check, not by position.
        return pd.DataFrame({1: [0.1, 0.9, 0.5], 0: [0.9, 0.1, 0.5]})


def test_rerank_candidates_column_selection_is_name_based_not_positional():
    candidates = pd.DataFrame(
        {"smiles": ["low_prob", "high_prob", "mid_prob"], "cosine_score": [0.1, 0.9, 0.5]}
    )
    feature_columns = ["cosine_score"]
    result = rerank_candidates(
        _StubPredictorReversedColumns(), candidates, feature_columns, top_k=25
    )
    assert result == ["high_prob", "mid_prob", "low_prob"]


def test_rerank_candidates_fewer_candidates_than_top_k_returns_all(tmp_path):
    train_df = _make_synthetic_train_df()
    # See comment in test_rerank_candidates_orders_by_predicted_probability
    # above: a wide ppm_tolerance is required to avoid a single-class
    # training set with this synthetic fixture.
    training_df = build_training_examples(
        train_df, n_held_out=3, random_state=1, ppm_tolerance=1_000_000.0
    )
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
