"""Matching results TSV writer and end-to-end test inference runner.

Filters model candidate probabilities by optimal threshold and outputs compliant matching_results.tsv.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import polars as pl

from backend.src.config import cfg
from backend.src.data.loader import load_source
from backend.src.exceptions import ModelTrainingError, SubmissionValidationError
from backend.src.logger import get_logger
from backend.src.model.persist import load_model
from backend.src.model.train import get_feature_columns, predict_pair_probabilities
from backend.src.predict.candidate_writer import write_candidate_pairs_tsv

logger = get_logger(__name__)


def write_matching_results_tsv(
    scored_candidates_df: pl.DataFrame,
    all_s1_ids: List[str] | Set[str],
    threshold: float = 0.65,
    output_path: Optional[Path] = None,
) -> Path:
    """Formats and writes matching_results.tsv meeting exact leaderboard rules.

    Schema:
        source1_entity_id\tmatched_entity_ids

    Args:
        scored_candidates_df: DataFrame containing ['source1_entity_id', 'candidate_entity_id', 'pred_prob'].
        all_s1_ids: All required Source 1 entity IDs in test split.
        threshold: Decision threshold for positive classification (pred_prob >= threshold).
        output_path: Destination TSV file path.

    Returns:
        Path to written matching TSV file.
    """
    target = output_path or Path(cfg.paths.output_dir) / "matching_results.tsv"
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info(
        f"Filtering matching candidates with threshold >= {threshold:.3f} "
        f"for {len(all_s1_ids):,} S1 entities..."
    )

    match_map: Dict[str, List[str]] = defaultdict(list)
    seen_matches: Set[tuple[str, str]] = set()

    if scored_candidates_df.height > 0:
        # Filter candidate pairs exceeding decision threshold
        filtered = scored_candidates_df.filter(pl.col("pred_prob") >= threshold)
        
        for s1_id, cand_id in zip(filtered["source1_entity_id"], filtered["candidate_entity_id"]):
            if not cand_id or not str(cand_id).strip():
                continue
            cand_id = str(cand_id).strip()
            
            # Constraint check: Disallow self-matches
            if cand_id.startswith("S1-") or not cand_id.startswith(("S2-", "S3-")):
                logger.warning(f"Skipping invalid candidate ID '{cand_id}' for S1 '{s1_id}'")
                continue

            pair = (s1_id, cand_id)
            if pair not in seen_matches:
                seen_matches.add(pair)
                match_map[s1_id].append(cand_id)

    written_count = 0
    non_empty_count = 0

    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = match_map.get(s1_id, [])
            match_str = ",".join(matches)
            f.write(f"{s1_id}\t{match_str}\n")
            written_count += 1
            if match_str:
                non_empty_count += 1

    elapsed = time.time() - t0
    logger.info(
        f"Saved matching_results.tsv to {target} in {elapsed:.2f}s: "
        f"{written_count:,} total entities ({non_empty_count:,} with matches, {written_count - non_empty_count:,} singletons)."
    )
    return target


def run_prediction_pipeline(
    model_path: Optional[Path] = None,
    threshold: Optional[float] = None,
    sample_n: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Executes end-to-end test set inference: candidate pairs and final matches generation.

    Args:
        model_path: Optional path to serialized model. Defaults to latest_model.joblib.
        threshold: Optional classification threshold override.
        sample_n: Optional sample row limit.

    Returns:
        Tuple of (candidate_pairs_tsv_path, matching_results_tsv_path).
    """
    logger.info("Starting end-to-end test prediction pipeline...")
    t0 = time.time()

    # 1. Load test S1 reference entities
    test_s1_path = Path(cfg.paths.test_s1)
    if not test_s1_path.exists():
        raise SubmissionValidationError(f"Test Source 1 file not found at: {test_s1_path}")
    
    test_s1_df = load_source(test_s1_path, sample_n=sample_n)
    all_s1_ids = test_s1_df["entity_id"].to_list()
    logger.info(f"Loaded {len(all_s1_ids):,} test reference S1 entities.")

    # 2. Ensure test features exist
    proc_dir = Path(cfg.paths.processed_dir)
    test_feat_path = proc_dir / "test_features.parquet"
    if not test_feat_path.exists():
        logger.info(f"Test features not found at {test_feat_path}. Running test feature extraction...")
        from backend.src.features.runner import run_features_stage
        run_features_stage(split="test", sample_n=sample_n)

    test_feat_df = pl.read_parquet(test_feat_path)
    logger.info(f"Loaded test candidate feature matrix: {test_feat_df.height:,} rows.")

    # 3. Write candidate_pairs.tsv
    cand_path = write_candidate_pairs_tsv(test_feat_df, all_s1_ids=all_s1_ids)

    # 4. Load model and predict probabilities
    m_path = model_path or (Path(cfg.paths.models_dir) / "latest_model.joblib")
    model, metadata = load_model(m_path)
    feature_cols = metadata.get("metadata", {}).get("feature_names") or get_feature_columns(test_feat_df)

    probs = predict_pair_probabilities(model, test_feat_df, feature_cols=feature_cols)
    scored_test_df = test_feat_df.with_columns(pl.Series("pred_prob", probs))

    # 5. Determine threshold (use calibrated optimal if available, else default)
    dec_thresh = threshold
    if dec_thresh is None:
        rep_file = Path(cfg.paths.reports_dir) / "eval_results.md"
        if rep_file.exists():
            try:
                import re
                text = rep_file.read_text(encoding="utf-8")
                match = re.search(r"Optimal Threshold:\s*\*\*([0-9.]+)\*\*", text)
                if match:
                    dec_thresh = float(match.group(1))
                    logger.info(f"Loaded calibrated optimal threshold from {rep_file}: {dec_thresh:.3f}")
            except Exception:
                pass
    if dec_thresh is None:
        dec_thresh = cfg.evaluation.get("default_threshold", 0.65)
    logger.info(f"Applying decision threshold: {dec_thresh:.3f}")

    # 6. Write matching_results.tsv
    match_path = write_matching_results_tsv(
        scored_candidates_df=scored_test_df,
        all_s1_ids=all_s1_ids,
        threshold=dec_thresh,
    )

    elapsed = time.time() - t0
    logger.info(f"Test prediction pipeline completed successfully in {elapsed:.2f}s.")
    return cand_path, match_path
