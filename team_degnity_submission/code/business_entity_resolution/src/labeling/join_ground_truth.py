"""Ground truth labeling and target creation module.

Joins candidate pairs with training ground truth annotations to construct binary classification targets (y in {0, 1}).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import polars as pl

from backend.src.config import cfg
from backend.src.data.loader import load_ground_truth
from backend.src.logger import get_logger

logger = get_logger(__name__)


def create_training_labels(
    candidate_features_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    output_labeled_path: Optional[Path] = None,
) -> pl.DataFrame:
    """Assigns binary labels (y=1 for true matches, y=0 for false candidates) to feature rows.

    Args:
        candidate_features_df: DataFrame containing candidate pairs and extracted features.
        ground_truth_df: Ground truth annotations [source1_entity_id, matched_entity_ids].
        output_labeled_path: Optional path to save labeled parquet table.

    Returns:
        polars.DataFrame containing all feature columns plus target column 'is_match' (Int32).
    """
    logger.info("Constructing binary matching labels from ground truth via vectorized join...")
    t0 = time.time()

    # Explode ground truth matched IDs into pairwise format with target flag
    gt_exploded = (
        ground_truth_df
        .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .rename({"matched_entity_ids": "candidate_entity_id"})
        .with_columns(pl.lit(1, dtype=pl.Int32).alias("is_match"))
        .select(["source1_entity_id", "candidate_entity_id", "is_match"])
    )

    # Vectorized left join in Polars: unmatched candidates automatically become 0
    labeled_df = (
        candidate_features_df
        .join(gt_exploded, on=["source1_entity_id", "candidate_entity_id"], how="left")
        .with_columns(pl.col("is_match").fill_null(0))
    )

    n_pos = labeled_df.filter(pl.col("is_match") == 1).height
    n_neg = labeled_df.height - n_pos
    pos_pct = (n_pos / max(labeled_df.height, 1)) * 100

    elapsed = time.time() - t0
    logger.info(
        f"Labeling complete in {elapsed:.2f}s: {labeled_df.height:,} total rows | "
        f"Positives (y=1): {n_pos:,} ({pos_pct:.2f}%), Negatives (y=0): {n_neg:,} ({100-pos_pct:.2f}%)"
    )

    if output_labeled_path:
        output_labeled_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.write_parquet(output_labeled_path, compression="zstd")
        logger.info(f"Saved labeled feature dataset to: {output_labeled_path}")

    return labeled_df
