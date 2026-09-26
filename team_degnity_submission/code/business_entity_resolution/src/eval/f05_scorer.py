"""Macro F0.5 evaluation metric and blocking recall evaluation suite.

Implements exact entity-level Macro F0.5 score computation and blocking recall metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import polars as pl

from backend.src.logger import get_logger

logger = get_logger(__name__)


def compute_entity_f05(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5,
) -> float:
    """Computes F0.5 score for a single reference entity.

    Rules:
        - If true_set is empty and pred_set is empty: score = 1.0 (correct non-match).
        - If true_set is empty and pred_set is non-empty: score = 0.0 (false merge penalty).
        - If true_set is non-empty and pred_set is empty: score = 0.0 (missed match).
        - If both non-empty: standard F_beta formula with beta=0.5 (beta^2 = 0.25).
    """
    n_true = len(true_set)
    n_pred = len(pred_set)

    # Singleton Cases
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    if n_pred == 0:
        return 0.0

    # Overlap
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0

    precision = tp / n_pred
    recall = tp / n_true

    beta_sq = beta * beta  # 0.25 for beta=0.5
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0

    f05 = ((1.0 + beta_sq) * precision * recall) / denom
    return f05


def compute_macro_f05(
    ground_truth_map: Dict[str, Set[str]],
    prediction_map: Dict[str, Set[str]],
    beta: float = 0.5,
) -> float:
    """Computes the Macro-averaged F0.5 across all reference S1 entities.

    Args:
        ground_truth_map: Mapping of source1_entity_id -> set of true matched IDs.
        prediction_map: Mapping of source1_entity_id -> set of predicted matched IDs.
        beta: F-score beta parameter (default 0.5).

    Returns:
        Macro F0.5 score in range [0.0, 1.0].
    """
    if not ground_truth_map:
        return 0.0

    total_score = 0.0
    for s1_id, true_set in ground_truth_map.items():
        pred_set = prediction_map.get(s1_id, set())
        total_score += compute_entity_f05(true_set, pred_set, beta=beta)

    return total_score / len(ground_truth_map)


def evaluate_blocking_recall(
    candidate_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
) -> Dict[str, float]:
    """Measures blocking candidate recall against true matches in ground truth with high-speed Polars join.

    Args:
        candidate_df: DataFrame with [source1_entity_id, candidate_entity_id].
        ground_truth_df: DataFrame with [source1_entity_id, matched_entity_ids].

    Returns:
        Dictionary containing recall, total true pairs, retrieved true pairs, and candidate density.
    """
    logger.info("Computing blocking recall against ground truth...")

    # Explode ground truth matched IDs into pairwise format using Polars
    gt_exploded = (
        ground_truth_df
        .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .rename({"matched_entity_ids": "candidate_entity_id"})
        .select(["source1_entity_id", "candidate_entity_id"])
    )
    total_true_pairs = gt_exploded.height

    if total_true_pairs == 0:
        return {
            "blocking_recall": 1.0,
            "total_true_pairs": 0,
            "retrieved_true_pairs": 0,
            "total_candidates": candidate_df.height,
            "avg_candidates_per_entity": 0.0,
        }

    # High-speed inner join in Polars to count retrieved true pairs
    retrieved_true = candidate_df.select(
        ["source1_entity_id", "candidate_entity_id"]
    ).join(
        gt_exploded, on=["source1_entity_id", "candidate_entity_id"], how="inner"
    ).height

    recall = retrieved_true / total_true_pairs

    unique_s1 = candidate_df.select(pl.col("source1_entity_id").n_unique()).item() if candidate_df.height > 0 else 1
    avg_cands = candidate_df.height / max(unique_s1, 1)

    stats = {
        "blocking_recall": round(recall, 4),
        "total_true_pairs": total_true_pairs,
        "retrieved_true_pairs": retrieved_true,
        "total_candidates": candidate_df.height,
        "avg_candidates_per_entity": round(avg_cands, 2),
    }

    logger.info(
        f"Blocking Recall: {stats['blocking_recall']*100:.2f}% "
        f"({retrieved_true:,}/{total_true_pairs:,} true pairs retrieved, "
        f"{candidate_df.height:,} total candidates, {stats['avg_candidates_per_entity']:.1f} avg cands/entity)"
    )
    return stats
