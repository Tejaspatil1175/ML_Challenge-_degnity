"""Full candidate blocking pipeline orchestrator.

Coordinates key-based and semantic blocking, measures recall on train sets, and writes candidate pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import polars as pl

from backend.src.blocking.keys import run_key_based_blocking
from backend.src.blocking.merge import merge_and_deduplicate_candidates, save_candidate_pairs_tsv
from backend.src.config import cfg
from backend.src.data.loader import load_ground_truth
from backend.src.eval.f05_scorer import evaluate_blocking_recall
from backend.src.logger import get_logger

logger = get_logger(__name__)


def run_blocking_stage(
    split: str = "train",
    sample_n: Optional[int] = None,
    output_candidate_path: Optional[Path] = None,
) -> Tuple[pl.DataFrame, Optional[Dict[str, float]]]:
    """Executes the candidate blocking workflow for either train or test split.

    Args:
        split: 'train' or 'test'.
        sample_n: Optional row limit for fast debugging.
        output_candidate_path: Optional destination path for candidate pairs TSV.

    Returns:
        Tuple of (merged_candidate_dataframe, optional_recall_stats_dict).
    """
    logger.info(f"Starting blocking stage for split='{split}' (sample_n={sample_n})...")
    
    proc_dir = Path(cfg.paths.processed_dir)
    s1_norm = proc_dir / f"{split}_s1_norm.parquet"
    s2_norm = proc_dir / f"{split}_s2_norm.parquet"
    s3_norm = proc_dir / f"{split}_s3_norm.parquet"

    # 1. Deterministic Key Blocking via DuckDB SQL
    key_cands = run_key_based_blocking(
        s1_parquet_path=s1_norm,
        s2_parquet_path=s2_norm,
        s3_parquet_path=s3_norm,
        max_cands_per_entity=cfg.blocking.get("max_candidates_per_entity", 50),
    )

    # 2. Merge & deduplicate
    final_cands = merge_and_deduplicate_candidates(key_cands)

    # 3. Save candidate pairs TSV
    out_tsv = output_candidate_path or (
        Path(cfg.paths.output_dir) / f"{split}_candidate_pairs.tsv" if split == "train" else Path(cfg.paths.candidate_pairs_tsv)
    )
    save_candidate_pairs_tsv(final_cands, output_path=out_tsv)

    # 4. Measure recall if train split
    recall_stats = None
    if split == "train" and cfg.paths.train_gt.exists():
        gt_df = load_ground_truth(cfg.paths.train_gt, sample_n=sample_n)
        recall_stats = evaluate_blocking_recall(final_cands, gt_df)

        # Write recall report
        rep_dir = Path(cfg.paths.reports_dir)
        rep_dir.mkdir(parents=True, exist_ok=True)
        report_file = rep_dir / "blocking_recall.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# Blocking Recall Evaluation Report\n\n")
            f.write(f"- **Blocking Recall:** {recall_stats['blocking_recall']*100:.2f}%\n")
            f.write(f"- **Retrieved True Pairs:** {recall_stats['retrieved_true_pairs']:,} / {recall_stats['total_true_pairs']:,}\n")
            f.write(f"- **Total Generated Candidates:** {recall_stats['total_candidates']:,}\n")
            f.write(f"- **Avg Candidates per S1 Entity:** {recall_stats['avg_candidates_per_entity']:.1f}\n")
        logger.info(f"Blocking recall report saved to: {report_file}")

    return final_cands, recall_stats
