"""Unit tests for threshold search and Macro F0.5 calibration."""

import numpy as np
import polars as pl
import pytest

from backend.src.eval.threshold_search import build_ground_truth_map, save_evaluation_report, search_optimal_threshold


def test_build_ground_truth_map():
    gt_df = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_ids": ["S2-1, S3-1", "", None],
    })
    gt_map = build_ground_truth_map(gt_df)
    assert gt_map["S1-1"] == {"S2-1", "S3-1"}
    assert gt_map["S1-2"] == set()
    assert gt_map["S1-3"] == set()


def test_search_optimal_threshold():
    # Ground truth: S1-1 matches S2-10, S1-2 matches S2-20, S1-3 is singleton
    gt_df = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_ids": ["S2-10", "S2-20", ""],
    })

    # Val scored: S1-1 has high prob on S2-10 (0.85) and low on S2-11 (0.4)
    # S1-2 has high prob on S2-20 (0.75)
    # S1-3 has weak noise candidate S2-30 (0.45)
    val_scored = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
        "candidate_entity_id": ["S2-10", "S2-11", "S2-20", "S2-30"],
        "pred_prob": [0.85, 0.40, 0.75, 0.45],
    })

    best_thresh, best_f05, sweep_df = search_optimal_threshold(
        val_scored_df=val_scored,
        ground_truth_df=gt_df,
        threshold_min=0.30,
        threshold_max=0.90,
        threshold_step=0.10,
    )

    # At threshold 0.50-0.70:
    # S1-1 predicts {S2-10} -> exact match (1.0)
    # S1-2 predicts {S2-20} -> exact match (1.0)
    # S1-3 predicts {} -> singleton correct (1.0)
    # Macro F0.5 should be 1.0!
    assert best_f05 == 1.0
    assert 0.50 <= best_thresh <= 0.75
    assert sweep_df.height > 0
