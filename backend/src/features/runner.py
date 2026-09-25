"""Feature extraction stage orchestrator.

Coordinates pairwise feature computation and target label assignment for training and test candidate sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import polars as pl

from backend.src.config import cfg
from backend.src.data.loader import load_ground_truth
from backend.src.features.pair_features import extract_features_for_candidates
from backend.src.labeling.join_ground_truth import create_training_labels
from backend.src.logger import get_logger

logger = get_logger(__name__)


def run_features_stage(
    split: str = "train",
    sample_n: Optional[int] = None,
) -> pl.DataFrame:
    """Executes the complete feature extraction and labeling pipeline for a split.

    Args:
        split: 'train' or 'test'.
        sample_n: Optional limit on candidate pairs for fast debugging.

    Returns:
        polars.DataFrame containing computed feature matrix.
    """
    logger.info(f"Starting feature extraction stage for split='{split}' (sample_n={sample_n})...")

    proc_dir = Path(cfg.paths.processed_dir)
    cand_file = Path(cfg.paths.output_dir) / f"{split}_candidate_pairs.tsv"
    if not cand_file.exists():
        # Fallback to candidate_pairs_tsv if test
        cand_file = Path(cfg.paths.candidate_pairs_tsv)

    logger.info(f"Reading candidate pairs from: {cand_file}")
    cand_df = pl.read_csv(cand_file, separator="\t")
    if sample_n and sample_n > 0:
        cand_df = cand_df.head(sample_n)

    # Load normalized parquet tables
    s1_norm = pl.read_parquet(proc_dir / f"{split}_s1_norm.parquet")
    s2_norm = pl.read_parquet(proc_dir / f"{split}_s2_norm.parquet")
    s3_norm = pl.read_parquet(proc_dir / f"{split}_s3_norm.parquet")

    # Extract features
    features_df = extract_features_for_candidates(
        candidate_df=cand_df,
        s1_norm_df=s1_norm,
        s2_norm_df=s2_norm,
        s3_norm_df=s3_norm,
    )

    out_parquet = proc_dir / f"{split}_features.parquet"

    if split == "train" and cfg.paths.train_gt.exists():
        gt_df = load_ground_truth(cfg.paths.train_gt)
        labeled_df = create_training_labels(
            candidate_features_df=features_df,
            ground_truth_df=gt_df,
            output_labeled_path=out_parquet,
        )
        return labeled_df
    else:
        features_df.write_parquet(out_parquet, compression="zstd")
        logger.info(f"Saved feature dataset to: {out_parquet}")
        return features_df
