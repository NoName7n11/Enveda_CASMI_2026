import pandas as pd
import pytest
from src.data import load_test_set, load_sampled_train, make_validation_split


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


def _make_fake_train_df_for_split(n_multi_spectra_molecules: int, n_single_spectra_molecules: int) -> pd.DataFrame:
    rows = []
    for m in range(n_multi_spectra_molecules):
        inchikey = f"MULTI{m:04d}XXXXXX"
        n_spectra = 2 + (m % 3)  # 2-4 spectra per molecule
        for s in range(n_spectra):
            rows.append(
                {
                    "inchikey14": inchikey,
                    "normalized_smiles": f"SMILES_{inchikey}",
                    "instrument_type": "timsTOF" if m % 2 == 0 else "Orbitrap",
                    "precursor_mz": 100.0 + m,
                    "adduct": "[M+H]+",
                    "spectrum_idx": s,
                }
            )
    for m in range(n_single_spectra_molecules):
        inchikey = f"SINGLE{m:04d}XXXXX"
        rows.append(
            {
                "inchikey14": inchikey,
                "normalized_smiles": f"SMILES_{inchikey}",
                "instrument_type": "timsTOF",
                "precursor_mz": 200.0 + m,
                "adduct": "[M+H]+",
                "spectrum_idx": 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fake_split_train_df():
    return _make_fake_train_df_for_split(n_multi_spectra_molecules=50, n_single_spectra_molecules=20)


def test_make_validation_split_reproducible_with_same_seed(fake_split_train_df):
    q_a, t_a, gt_a = make_validation_split(fake_split_train_df, n_held_out=10, random_state=42)
    q_b, t_b, gt_b = make_validation_split(fake_split_train_df, n_held_out=10, random_state=42)
    pd.testing.assert_frame_equal(q_a.reset_index(drop=True), q_b.reset_index(drop=True))
    pd.testing.assert_frame_equal(t_a.reset_index(drop=True), t_b.reset_index(drop=True))
    assert gt_a == gt_b


def test_make_validation_split_leaves_other_spectra_in_train_pool(fake_split_train_df):
    val_query_df, val_train_df, val_ground_truth = make_validation_split(
        fake_split_train_df, n_held_out=10, random_state=42
    )
    held_out_keys = set(val_query_df["inchikey14"].unique())
    assert len(held_out_keys) > 0
    for key in held_out_keys:
        # each held-out molecule must still have at least one spectrum left in val_train_df
        assert (val_train_df["inchikey14"] == key).sum() >= 1


def test_make_validation_split_never_selects_single_spectrum_molecules(fake_split_train_df):
    single_spectra_keys = {
        k for k, n in fake_split_train_df.groupby("inchikey14").size().items() if n < 2
    }
    val_query_df, _, _ = make_validation_split(fake_split_train_df, n_held_out=10, random_state=42)
    assert not set(val_query_df["inchikey14"].unique()) & single_spectra_keys


def test_make_validation_split_ground_truth_keys_match_query_molecule_ids(fake_split_train_df):
    val_query_df, _, val_ground_truth = make_validation_split(
        fake_split_train_df, n_held_out=10, random_state=42
    )
    assert set(val_ground_truth.keys()) == set(val_query_df["molecule_id"].unique())
