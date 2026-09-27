"""
Score pre-computed chunk parquets using the correct model_v1.joblib (17 features, 500 estimators).
Run: python run_score_chunks.py
"""
from __future__ import annotations
import sys, time, re, gc, glob
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from backend.src.config import cfg
from backend.src.logger import get_logger
from backend.src.model.persist import load_model
from backend.src.model.train import predict_pair_probabilities
from backend.src.data.loader import load_source
import polars as pl

logger = get_logger("run_score_chunks")

# The correct model with 17 features trained on 28M pairs (ROC-AUC 0.9987)
CORRECT_MODEL = "models/model_v1.joblib"
# Optimal threshold from evaluation (val Macro F0.5 sweep)
DEFAULT_THRESH = 0.840

def main() -> int:
    t_total = time.time()
    proc_dir = Path(cfg.paths.processed_dir)
    output_dir = Path(cfg.paths.output_dir)

    # 1. Load test S1 IDs
    logger.info("Loading test S1 entities...")
    test_s1_df = load_source(Path(cfg.paths.test_s1))
    all_s1_ids = test_s1_df["entity_id"].to_list()
    logger.info(f"  -> {len(all_s1_ids):,} S1 entities")

    # 2. Load the CORRECT full model
    model_path = Path(CORRECT_MODEL)
    logger.info(f"Loading model from {model_path}...")
    model, metadata = load_model(model_path)
    feature_cols = metadata.get("metadata", {}).get("feature_names")
    logger.info(f"  -> Feature cols ({len(feature_cols)}): {feature_cols}")
    logger.info(f"  -> n_estimators: {model.n_estimators}")

    # 3. Threshold
    dec_thresh = DEFAULT_THRESH
    logger.info(f"  -> Using threshold: {dec_thresh:.3f}")

    # 4. Score each chunk, accumulate matches above threshold
    chunk_files = sorted(glob.glob(str(proc_dir / "_test_feat_chunk_*.parquet")))
    logger.info(f"Found {len(chunk_files)} chunk files to score.")

    match_map = defaultdict(list)
    cand_map = defaultdict(list)
    seen_match = set()
    seen_cand = set()
    total_above = 0

    for i, chunk_path in enumerate(chunk_files):
        t0 = time.time()
        chunk_df = pl.read_parquet(chunk_path)

        probs = predict_pair_probabilities(model, chunk_df, feature_cols=feature_cols)

        s1_col = chunk_df["source1_entity_id"].to_list()
        cand_col = chunk_df["candidate_entity_id"].to_list()

        chunk_above = 0
        for s1_id, cand_id, prob in zip(s1_col, cand_col, probs):
            pair = (s1_id, cand_id)
            if pair not in seen_cand:
                seen_cand.add(pair)
                cand_map[s1_id].append(cand_id)
            if prob >= dec_thresh and pair not in seen_match:
                seen_match.add(pair)
                match_map[s1_id].append(cand_id)
                chunk_above += 1

        total_above += chunk_above
        del chunk_df, s1_col, cand_col, probs
        gc.collect()

        logger.info(
            f"  Chunk {i+1}/{len(chunk_files)} scored in {time.time()-t0:.1f}s | "
            f"Chunk matches: {chunk_above:,} | Total matches: {total_above:,}"
        )

    logger.info(f"Scoring complete. Total matches: {total_above:,} | S1 entities with matches: {len(match_map):,}")

    # 5. Write candidate_pairs.tsv
    cand_path = output_dir / "candidate_pairs.tsv"
    logger.info(f"Writing candidate_pairs.tsv...")
    t0 = time.time()
    cand_count = 0
    with open(cand_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = cand_map.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")
            if cands: cand_count += 1
    logger.info(f"  -> {cand_count:,} entities with candidates in {time.time()-t0:.1f}s")

    # 6. Write matching_results.tsv
    match_path = output_dir / "matching_results.tsv"
    logger.info(f"Writing matching_results.tsv (threshold={dec_thresh:.3f})...")
    t0 = time.time()
    non_empty = 0
    with open(match_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = match_map.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(matches)}\n")
            if matches: non_empty += 1
    logger.info(f"  -> {non_empty:,} entities with matches in {time.time()-t0:.1f}s")

    # 7. Validate
    logger.info("Running official validator...")
    import subprocess
    result = subprocess.run([
        sys.executable,
        "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
        "--matching", str(match_path),
        "--candidate", str(cand_path),
        "--test-dir", "6ab10eb3b23ba_student_resource/student_resource/dataset/test",
    ], capture_output=True, text=True, encoding="utf-8")
    print(result.stdout)
    if result.returncode == 0:
        logger.info("PASS - Submission files valid and ready to upload!")
    else:
        logger.error(f"FAIL: {result.stderr}")
        return 1

    logger.info(f"Total pipeline time: {time.time()-t_total:.1f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
