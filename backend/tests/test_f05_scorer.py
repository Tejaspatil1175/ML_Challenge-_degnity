"""Unit tests for exact Macro F0.5 computation and singleton handling."""

import pytest
from backend.src.eval.f05_scorer import compute_entity_f05, compute_macro_f05


def test_singleton_correct_prediction():
    """Singleton entity with no true matches, correctly predicted empty -> 1.0."""
    score = compute_entity_f05(true_set=set(), pred_set=set())
    assert score == 1.0


def test_singleton_false_merge_penalty():
    """Singleton entity with no true matches, falsely predicted match -> 0.0."""
    score = compute_entity_f05(true_set=set(), pred_set={"S2-123"})
    assert score == 0.0


def test_missed_match():
    """Entity with true matches, predicted empty -> 0.0."""
    score = compute_entity_f05(true_set={"S2-1", "S3-2"}, pred_set=set())
    assert score == 0.0


def test_perfect_match():
    """Entity with perfect prediction -> 1.0."""
    score = compute_entity_f05(true_set={"S2-1", "S3-2"}, pred_set={"S2-1", "S3-2"})
    assert score == 1.0


def test_partial_match_f05():
    """Entity with 1 TP and 1 FP out of 1 True:
    Precision = 1/2 = 0.5, Recall = 1/1 = 1.0
    F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = ~0.5556
    """
    score = compute_entity_f05(true_set={"S2-1"}, pred_set={"S2-1", "S2-99"})
    assert pytest.approx(score, rel=1e-3) == 0.555555


def test_macro_average():
    """Test macro average across multiple entities."""
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),  # singleton
    }
    preds = {
        "S1-1": {"S2-1"},
        "S1-2": set(),  # correct singleton
    }
    macro = compute_macro_f05(gt, preds)
    assert macro == 1.0
