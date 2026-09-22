import pandas as pd
import pytest
from src.data import load_test_set, load_sampled_train


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
