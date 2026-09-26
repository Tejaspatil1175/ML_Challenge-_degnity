"""Threshold calibration and Macro F0.5 optimization engine.

Performs grid search across probability decision thresholds to maximize macro-averaged F0.5 score on validation data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import polars as pl

from backend.src.config import cfg
from backend.src.eval.f05_scorer import compute_macro_f05
from backend.src.logger import get_logger

logger = get_logger(__name__)


def build_ground_truth_map(
    ground_truth_df: pl.DataFrame,
    target_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Builds a fast lookup dictionary of source1_entity_id -> set(matched_entity_ids).

    Args:
        ground_truth_df: Ground truth annotations table.
        target_s1_ids: Optional subset of S1 entities to filter for.

    Returns:
        Dictionary mapping entity_id to set of true match IDs.
    """
    gt_map: Dict[str, Set[str]] = {}
    for row in ground_truth_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        if target_s1_ids is not None and s1_id not in target_s1_ids:
            continue
        m_str = row["matched_entity_ids"]
        if m_str and str(m_str).strip():
            gt_map[s1_id] = {m.strip() for m in str(m_str).split(",") if m.strip()}
        else:
            gt_map[s1_id] = set()
    return gt_map


def search_optimal_threshold(
    val_scored_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    threshold_min: float = 0.30,
    threshold_max: float = 0.90,
    threshold_step: float = 0.02,
    beta: float = 0.5,
) -> Tuple[float, float, pl.DataFrame]:
    """Sweeps decision thresholds to maximize validation Macro F0.5.

    Args:
        val_scored_df: DataFrame with ['source1_entity_id', 'candidate_entity_id', 'pred_prob'].
        ground_truth_df: Ground truth table with all validation reference entities.
        threshold_min: Minimum probability threshold.
        threshold_max: Maximum probability threshold.
        threshold_step: Step increment for threshold grid.
        beta: F-score beta parameter (default 0.5).

    Returns:
        Tuple of (best_threshold, best_f05_score, sweep_results_df).
    """
    logger.info("Starting threshold calibration sweep for Macro F0.5 optimization...")
    t0 = time.time()

    val_s1_ids = set(val_scored_df["source1_entity_id"].unique().to_list())
    gt_map = build_ground_truth_map(ground_truth_df, target_s1_ids=val_s1_ids)

    logger.info(f"Evaluating across {len(gt_map):,} validation S1 entities...")

    thresholds = np.arange(threshold_min, threshold_max + (threshold_step / 2.0), threshold_step)
    
    # Pre-extract numpy arrays for high-speed filtering
    s1_array = val_scored_df["source1_entity_id"].to_numpy()
    cand_array = val_scored_df["candidate_entity_id"].to_numpy()
    prob_array = val_scored_df["pred_prob"].to_numpy()

    sweep_records = []
    best_score = -1.0
    best_thresh = float(threshold_min)
    from tqdm import tqdm

    pbar = tqdm(thresholds, desc="Threshold Calibration Progress", unit="thresh")
    for thresh in pbar:
        thresh = round(float(thresh), 4)
        mask = prob_array >= thresh
        passing_s1 = s1_array[mask]
        passing_cand = cand_array[mask]

        # Build prediction map for active threshold
        pred_map: Dict[str, Set[str]] = {s1: set() for s1 in gt_map.keys()}
        for s1, cand in zip(passing_s1, passing_cand):
            pred_map[s1].add(cand)

        score = compute_macro_f05(gt_map, pred_map, beta=beta)
        
        # Calculate positive predictions count
        total_predicted_pairs = int(np.sum(mask))

        sweep_records.append({
            "threshold": thresh,
            "macro_f05": round(score, 5),
            "predicted_pairs": total_predicted_pairs,
        })

        if score > best_score:
            best_score = score
            best_thresh = thresh

        pbar.set_postfix({
            "thresh": f"{thresh:.2f}",
            "f05": f"{score:.4f}",
            "best_f05": f"{best_score:.4f} (@{best_thresh:.2f})"
        })

    sweep_df = pl.DataFrame(sweep_records)
    elapsed = time.time() - t0

    logger.info(
        f"Threshold sweep complete in {elapsed:.2f}s | "
        f"Optimal Threshold: {best_thresh:.3f} -> Peak Macro F0.5: {best_score:.5f}"
    )

    return best_thresh, best_score, sweep_df


def save_evaluation_report(
    best_threshold: float,
    best_f05: float,
    sweep_df: pl.DataFrame,
    report_path: Optional[Path] = None,
) -> Path:
    """Generates markdown evaluation report."""
    target_path = report_path or (Path(cfg.paths.reports_dir) / "eval_results.md")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation & Threshold Calibration Report\n\n")
        f.write(f"- **Optimal Decision Threshold (tau):** `{best_threshold:.3f}`\n")
        f.write(f"- **Validation Peak Macro F0.5:** `{best_f05:.5f}`\n")
        f.write(f"- **Metric Formulation:** Macro-averaged entity-level F_0.5 (Precision 2x weighted over Recall)\n\n")
        f.write("## Threshold Sweep Grid\n\n")
        f.write("| Threshold | Macro F0.5 | Predicted Pairs |\n")
        f.write("|---|---|---|\n")
        for row in sweep_df.iter_rows(named=True):
            star = " **(Best)**" if row["threshold"] == best_threshold else ""
            f.write(f"| {row['threshold']:.2f}{star} | {row['macro_f05']:.5f} | {row['predicted_pairs']:,} |\n")

    logger.info(f"Saved evaluation report to: {target_path}")
    return target_path
