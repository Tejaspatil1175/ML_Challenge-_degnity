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
    logger.info("Constructing binary matching labels from ground truth...")
    t0 = time.time()

    # Build fast lookup of true match pairs: Set[(s1_id, match_id)]
    true_pairs: Set[Tuple[str, str]] = set()
    for row in ground_truth_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        if m_str:
            for cand_id in m_str.split(","):
                cand_id = cand_id.strip()
                if cand_id:
                    true_pairs.add((s1_id, cand_id))

    logger.info(f"Loaded {len(true_pairs):,} true match pairs from ground truth.")

    # Vectorized label assignment
    s1_ids = candidate_features_df["source1_entity_id"].to_list()
    cand_ids = candidate_features_df["candidate_entity_id"].to_list()

    labels = [
        1 if (s1, cand) in true_pairs else 0
        for s1, cand in zip(s1_ids, cand_ids)
    ]

    labeled_df = candidate_features_df.with_columns(
        pl.Series("is_match", labels, dtype=pl.Int32)
    )

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_pct = (n_pos / max(len(labels), 1)) * 100

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
