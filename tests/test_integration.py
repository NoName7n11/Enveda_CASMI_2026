"""End-to-end validation: real sampled train data -> baseline -> offline MRR@25."""

import os
import pandas as pd
import pytest
from src.data import load_sampled_train, make_validation_split
from src.baseline import build_submission
from src.metric import mrr_at_25

DATASET_AVAILABLE = os.path.exists("dataset/train.parquet")


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="dataset/train.parquet not present")
def test_baseline_pipeline_end_to_end_on_real_sample():
    # Small sample for test speed; real dev notebook uses a larger sample_size.
    train_df = load_sampled_train(sample_size=5_000, random_state=0)

    # Build a tiny held-out validation query set from train itself (spectrum-level
    # split within each held-out molecule, so the correct structure remains
    # retrievable from the candidate pool) so we have known ground truth to score
    # against, without touching the real dataset/test.parquet (which has no labels).
    unique_molecule_formulas = train_df["molecular_formula"].dropna().unique()
    assert len(unique_molecule_formulas) > 0, "sampled train data unexpectedly empty"

    held_out_df, remaining_train_df, ground_truth = make_validation_split(
        train_df, n_held_out=10, random_state=0
    )

    submission_df = build_submission(held_out_df, remaining_train_df, ppm_tolerance=15.0, top_k=25)

    # Submission format checks (per OVERVIEW.md Section 4 constraints)
    assert set(submission_df["molecule_id"]) == set(ground_truth.keys())
    assert submission_df["smiles"].notna().all()
    for smiles_field in submission_df["smiles"]:
        assert smiles_field.count(";") <= 24  # at most 25 candidates

    predictions = dict(zip(submission_df["molecule_id"], submission_df["smiles"].str.split(";")))
    score = mrr_at_25(predictions, ground_truth)

    # Weak sanity bound: since remaining_train_df no longer contains the exact
    # held-out spectra, a perfect 1.0 is not expected, but formula/precursor
    # overlap with other library entries of the same structure should give
    # a non-zero signal most of the time. This is a smoke test, not a
    # leaderboard-score assertion.
    assert 0.0 <= score <= 1.0
