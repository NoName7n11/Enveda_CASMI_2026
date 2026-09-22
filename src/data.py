"""Data loading for the CASMI 2026 baseline pipeline."""

import pandas as pd


def load_test_set(path: str = "dataset/test.parquet") -> pd.DataFrame:
    """Load the full test set (it's small, ~1500 rows)."""
    return pd.read_parquet(path)


def load_sampled_train(
    path: str = "dataset/train.parquet",
    sample_size: int = 75_000,
    always_include_libs: tuple[str, ...] = ("enveda-np-examples", "drug_plus"),
    random_state: int = 42,
) -> pd.DataFrame:
    """Load a stratified sample of the training set for fast local iteration.

    All rows from `always_include_libs` are kept (these are the smallest and
    most calibration-relevant libraries per OVERVIEW.md). Additional rows are
    randomly sampled from the rest up to `sample_size` total rows.
    """
    df = pd.read_parquet(path)
    priority_mask = df["ingest_lib"].isin(always_include_libs)
    priority_df = df[priority_mask]
    remaining_df = df[~priority_mask]

    remaining_budget = sample_size - len(priority_df)
    if remaining_budget <= 0:
        return priority_df.reset_index(drop=True)

    sampled_remaining = remaining_df.sample(
        n=min(remaining_budget, len(remaining_df)),
        random_state=random_state,
    )
    return pd.concat([priority_df, sampled_remaining]).reset_index(drop=True)
