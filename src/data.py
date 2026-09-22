"""Data loading for the CASMI 2026 baseline pipeline."""

import numpy as np
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


def make_validation_split(
    train_df: pd.DataFrame, n_held_out: int = 200, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build an offline validation split that mirrors the real Class 1 scenario.

    Unlike a naive molecule-level split (holding out every spectrum of a
    structure), this holds out only SOME spectra per held-out molecule: for
    each eligible held-out `inchikey14`, at least one of its OTHER spectra
    stays in `val_train_df` so the correct structure remains retrievable from
    the candidate pool. A molecule with only a single spectrum total cannot be
    split this way and is never selected as held-out (it stays entirely in
    the training/candidate pool).

    The held-out molecule selection prefers rows with
    `instrument_type == "timsTOF"` (matching the real competition test set,
    which is entirely timsTOF per OVERVIEW.md); if timsTOF-eligible molecules
    are fewer than `n_held_out`, the remainder is filled in from non-timsTOF
    eligible molecules rather than erroring out, since this is only an
    offline proxy metric.

    Returns (val_query_df, val_train_df, val_ground_truth) where
    val_ground_truth maps molecule_id (the held-out spectra's inchikey14,
    standing in for a real molecule_id) -> normalized_smiles.
    """
    rng = np.random.RandomState(random_state)

    spectra_counts = train_df.groupby("inchikey14").size()
    eligible_keys = spectra_counts[spectra_counts >= 2].index

    eligible_df = train_df[train_df["inchikey14"].isin(eligible_keys)]
    if "instrument_type" in eligible_df.columns:
        timstof_keys = pd.Index(
            eligible_df.loc[eligible_df["instrument_type"] == "timsTOF", "inchikey14"].unique()
        )
    else:
        timstof_keys = pd.Index([], dtype=object)
    other_eligible_keys = pd.Index(eligible_keys).difference(timstof_keys)

    n_from_timstof = min(n_held_out, len(timstof_keys))
    selected_timstof = rng.choice(timstof_keys, size=n_from_timstof, replace=False)

    remaining_needed = n_held_out - n_from_timstof
    n_from_other = min(remaining_needed, len(other_eligible_keys))
    selected_other = rng.choice(other_eligible_keys, size=n_from_other, replace=False)

    held_out_molecules = np.concatenate([selected_timstof, selected_other])

    query_row_indices = []
    train_row_indices = []
    for key in held_out_molecules:
        group_idx = train_df.index[train_df["inchikey14"] == key]
        shuffled = rng.permutation(group_idx.to_numpy())
        n_query = max(1, len(shuffled) // 2)
        # Always leave at least one spectrum of this molecule in the pool.
        n_query = min(n_query, len(shuffled) - 1)
        query_row_indices.extend(shuffled[:n_query])
        train_row_indices.extend(shuffled[n_query:])

    held_out_set = set(held_out_molecules)
    non_held_out_idx = train_df.index[~train_df["inchikey14"].isin(held_out_set)]

    val_query_df = train_df.loc[query_row_indices].copy()
    val_query_df["molecule_id"] = val_query_df["inchikey14"]

    val_train_idx = pd.Index(train_row_indices).append(non_held_out_idx)
    val_train_df = train_df.loc[val_train_idx]

    val_ground_truth = (
        val_query_df.drop_duplicates("inchikey14")
        .set_index("molecule_id")["normalized_smiles"]
        .to_dict()
    )

    return val_query_df, val_train_df, val_ground_truth
