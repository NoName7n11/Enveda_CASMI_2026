"""End-to-end validation: real sampled train data -> reranker training -> MRR@25 comparison."""

import os
import pandas as pd
import pytest
from src.data import load_sampled_train, make_validation_split
from src.baseline import build_submission
from src.metric import mrr_at_25
from src.reranker import (
    build_training_examples,
    train_reranker,
    extract_candidate_features,
    rerank_candidates,
)

DATASET_AVAILABLE = os.path.exists("dataset/train.parquet")

_FEATURE_COLUMNS = [
    "cosine_score", "ppm_error", "num_peaks_candidate",
    "mol_wt", "log_p", "num_rings", "num_rotatable_bonds", "num_hbd", "num_hba",
]


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="dataset/train.parquet not present")
def test_reranker_pipeline_trains_and_reranks_on_real_sample(tmp_path):
    train_df = load_sampled_train(sample_size=5_000, random_state=0)

    training_df = build_training_examples(train_df, n_held_out=20, random_state=1)
    if training_df.empty or training_df["is_correct"].nunique() < 2:
        pytest.skip(
            "Sampled data produced no usable training examples with both "
            "classes present — expected occasionally at small sample sizes, "
            "not a code defect (see Review Focus item 5)."
        )

    predictor = train_reranker(
        training_df, model_path=str(tmp_path / "integration_model"),
        feature_columns=_FEATURE_COLUMNS, time_limit=30,
    )

    # Build a fresh held-out split (different random_state) to compare
    # baseline vs reranked MRR@25 on the same query set.
    val_query_df, val_train_df, val_ground_truth = make_validation_split(
        train_df, n_held_out=20, random_state=2
    )
    if not val_ground_truth:
        pytest.skip("No eligible held-out molecules in this sample at this random_state.")

    baseline_predictions = {}
    reranked_predictions = {}
    for molecule_id, group in val_query_df.groupby("molecule_id"):
        features = extract_candidate_features(group, val_train_df, ppm_tolerance=15.0)
        if features.empty:
            baseline_predictions[molecule_id] = [""]
            reranked_predictions[molecule_id] = [""]
            continue
        baseline_order = features.sort_values("cosine_score", ascending=False)["smiles"].tolist()
        baseline_predictions[molecule_id] = baseline_order if baseline_order else [""]
        reranked_predictions[molecule_id] = rerank_candidates(
            predictor, features, _FEATURE_COLUMNS, top_k=25
        ) or [""]

    baseline_mrr = mrr_at_25(baseline_predictions, val_ground_truth)
    reranked_mrr = mrr_at_25(reranked_predictions, val_ground_truth)

    print(f"Baseline MRR@25: {baseline_mrr:.4f}, Reranked MRR@25: {reranked_mrr:.4f}")

    # Smoke assertions only (per plan scope — this is not a score threshold
    # test, sample is too small/random for that): both scores are valid
    # MRR values, and the pipeline ran end to end without crashing.
    assert 0.0 <= baseline_mrr <= 1.0
    assert 0.0 <= reranked_mrr <= 1.0
