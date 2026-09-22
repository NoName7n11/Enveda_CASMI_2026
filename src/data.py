"""Data loading for the CASMI 2026 baseline pipeline."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

    The full training parquet (~3GB on disk, ~2.5M rows with nested MS2 peak
    array columns) is read once as a pyarrow Table -- converting that whole
    table to pandas directly can exceed available memory on a typical dev
    machine, so only the small sampled row subset is ever materialized as a
    pandas DataFrame.
    """
    lib_column = pq.read_table(path, columns=["ingest_lib"]).column("ingest_lib").to_pandas()
    priority_mask = lib_column.isin(always_include_libs)
    priority_idx = lib_column.index[priority_mask]
    remaining_idx = lib_column.index[~priority_mask]

    remaining_budget = sample_size - len(priority_idx)
    if remaining_budget <= 0:
        selected_idx = priority_idx
    else:
        sampled_remaining_idx = (
            pd.Index(remaining_idx)
            .to_series()
            .sample(n=min(remaining_budget, len(remaining_idx)), random_state=random_state)
            .index
        )
        selected_idx = priority_idx.append(sampled_remaining_idx)

    full_table = pq.read_table(path)
    selected_table = full_table.take(pa.array(selected_idx))
    return selected_table.to_pandas().reset_index(drop=True)
