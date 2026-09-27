"""High-recall end-to-end streaming feature extraction, LightGBM scoring, and output generation.

Streams raw candidate pairs from data_cache/processed/test_s1_norm_raw_candidates.parquet in 500,000-pair chunks.
Computes 17 pairwise similarity features and predicts probabilities via model_v1.joblib.
Writes validated candidate_pairs.tsv and matching_results.tsv.
"""

from __future__ import annotations

import gc
import glob
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from backend.src.config import cfg
from backend.src.data.loader import load_source
from backend.src.logger import get_logger
from backend.src.model.persist import load_model
from backend.src.model.train import predict_pair_probabilities

logger = get_logger("run_high_recall_predict")

MODEL_PATH = "models/model_v1.joblib"
# High-precision threshold calibrated for Macro F0.5
DECISION_THRESH = 0.880
CHUNK_SIZE = 500_000


def build_lookup_table(parquet_path: Path) -> Dict[str, Tuple[str, str, str, str, str]]:
    """Loads normalized fields into a fast in-memory dictionary."""
    logger.info(f"Loading lookup from {parquet_path.name}...")
    df = pl.read_parquet(
        parquet_path,
        columns=["entity_id", "clean_name", "clean_address", "clean_country", "clean_zipcode", "clean_city"],
    )
    d = {}
    for row in df.iter_rows():
        d[row[0]] = (row[1] or "", row[2] or "", row[3] or "", row[4] or "", row[5] or "")
    logger.info(f"  -> {len(d):,} entities in {parquet_path.name}")
    return d


def extract_features_vectorized(
    s1_ids: List[str],
    cand_ids: List[str],
    s1_dict: Dict[str, Tuple[str, str, str, str, str]],
    cand_dict: Dict[str, Tuple[str, str, str, str, str]],
) -> pl.DataFrame:
    """Computes the exact 17 feature columns matching model_v1.joblib training schema."""
    n = len(s1_ids)

    # Pre-allocate feature arrays
    name_ratio = np.empty(n, dtype=np.float32)
    name_tok_sort = np.empty(n, dtype=np.float32)
    name_tok_set = np.empty(n, dtype=np.float32)
    name_partial = np.empty(n, dtype=np.float32)
    name_jw = np.empty(n, dtype=np.float32)
    name_lev = np.empty(n, dtype=np.float32)

    addr_ratio = np.empty(n, dtype=np.float32)
    addr_tok_sort = np.empty(n, dtype=np.float32)
    addr_tok_set = np.empty(n, dtype=np.float32)

    name_jaccard = np.empty(n, dtype=np.float32)
    addr_jaccard = np.empty(n, dtype=np.float32)
    name_char3 = np.empty(n, dtype=np.float32)

    country_match = np.empty(n, dtype=np.float32)
    zip_match = np.empty(n, dtype=np.float32)
    city_match = np.empty(n, dtype=np.float32)

    name_len_diff = np.empty(n, dtype=np.float32)
    addr_len_diff = np.empty(n, dtype=np.float32)

    for i in range(n):
        s1_id = s1_ids[i]
        c_id = cand_ids[i]

        s1_n, s1_a, s1_c, s1_z, s1_ci = s1_dict.get(s1_id, ("", "", "", "", ""))
        cn_n, cn_a, cn_c, cn_z, cn_ci = cand_dict.get(c_id, ("", "", "", "", ""))

        # 1. Name string metrics
        name_ratio[i] = fuzz.ratio(s1_n, cn_n) / 100.0
        name_tok_sort[i] = fuzz.token_sort_ratio(s1_n, cn_n) / 100.0
        name_tok_set[i] = fuzz.token_set_ratio(s1_n, cn_n) / 100.0
        name_partial[i] = fuzz.partial_ratio(s1_n, cn_n) / 100.0
        name_jw[i] = float(distance.JaroWinkler.similarity(s1_n, cn_n))
        name_lev[i] = float(distance.Levenshtein.normalized_similarity(s1_n, cn_n))

        # 2. Address string metrics
        addr_ratio[i] = fuzz.ratio(s1_a, cn_a) / 100.0
        addr_tok_sort[i] = fuzz.token_sort_ratio(s1_a, cn_a) / 100.0
        addr_tok_set[i] = fuzz.token_set_ratio(s1_a, cn_a) / 100.0

        # 3. Token & N-Gram overlap
        w1_n = set(s1_n.split())
        w2_n = set(cn_n.split())
        un_n = len(w1_n.union(w2_n))
        name_jaccard[i] = (len(w1_n.intersection(w2_n)) / un_n) if un_n > 0 else 0.0

        w1_a = set(s1_a.split())
        w2_a = set(cn_a.split())
        un_a = len(w1_a.union(w2_a))
        addr_jaccard[i] = (len(w1_a.intersection(w2_a)) / un_a) if un_a > 0 else 0.0

        # 3-gram name jaccard
        if len(s1_n) < 3 or len(cn_n) < 3:
            name_char3[i] = 1.0 if s1_n == cn_n and s1_n else 0.0
        else:
            g1 = {s1_n[j:j+3] for j in range(len(s1_n) - 2)}
            g2 = {cn_n[j:j+3] for j in range(len(cn_n) - 2)}
            ug = len(g1.union(g2))
            name_char3[i] = (len(g1.intersection(g2)) / ug) if ug > 0 else 0.0

        # 4. Structured matches
        country_match[i] = 1.0 if s1_c and s1_c == cn_c else (0.5 if not s1_c or not cn_c else 0.0)
        zip_match[i] = 1.0 if s1_z and s1_z == cn_z else (0.5 if not s1_z or not cn_z else 0.0)
        city_match[i] = 1.0 if s1_ci and s1_ci == cn_ci else (0.5 if not s1_ci or not cn_ci else 0.0)

        # 5. Length differences
        name_len_diff[i] = abs(len(s1_n) - len(cn_n)) / max(len(s1_n), len(cn_n), 1)
        addr_len_diff[i] = abs(len(s1_a) - len(cn_a)) / max(len(s1_a), len(cn_a), 1)

    return pl.DataFrame({
        "name_ratio": name_ratio,
        "name_tok_sort": name_tok_sort,
        "name_tok_set": name_tok_set,
        "name_partial": name_partial,
        "name_jw": name_jw,
        "name_lev": name_lev,
        "addr_ratio": addr_ratio,
        "addr_tok_sort": addr_tok_sort,
        "addr_tok_set": addr_tok_set,
        "name_jaccard": name_jaccard,
        "addr_jaccard": addr_jaccard,
        "name_char3": name_char3,
        "country_match": country_match,
        "zip_match": zip_match,
        "city_match": city_match,
        "name_len_diff": name_len_diff,
        "addr_len_diff": addr_len_diff,
    })


def main() -> int:
    t_start = time.time()
    proc_dir = Path(cfg.paths.processed_dir)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test S1 list
    logger.info("Loading test S1 entities...")
    test_s1_df = load_source(Path(cfg.paths.test_s1))
    all_s1_ids = test_s1_df["entity_id"].to_list()
    logger.info(f"  -> Total test S1 entities: {len(all_s1_ids):,}")

    # 2. Load model
    logger.info(f"Loading LightGBM model from {MODEL_PATH}...")
    model, metadata = load_model(Path(MODEL_PATH))
    feature_cols = metadata.get("metadata", {}).get("feature_names")
    logger.info(f"  -> Feature columns ({len(feature_cols)}): {feature_cols}")

    # 3. Load lookups
    s1_dict = build_lookup_table(proc_dir / "test_s1_norm.parquet")
    s2_dict = build_lookup_table(proc_dir / "test_s2_norm.parquet")
    s3_dict = build_lookup_table(proc_dir / "test_s3_norm.parquet")
    cand_dict = {**s2_dict, **s3_dict}
    del s2_dict, s3_dict
    gc.collect()

    # 4. Stream raw candidates in chunks
    cand_parquet = proc_dir / "test_s1_norm_raw_candidates.parquet"
    if not cand_parquet.exists():
        logger.error(f"Candidate parquet not found: {cand_parquet}")
        return 1

    total_pairs = pl.scan_parquet(cand_parquet).select(pl.len()).collect().item()
    logger.info(f"Streaming {total_pairs:,} candidate pairs from {cand_parquet.name} in chunks of {CHUNK_SIZE:,}...")

    match_map = defaultdict(list)
    cand_map = defaultdict(list)
    seen_cand = set()
    total_matches = 0

    num_chunks = (total_pairs + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Read parquet in slices
    for chunk_idx in range(num_chunks):
        t0 = time.time()
        offset = chunk_idx * CHUNK_SIZE
        chunk_cands = pl.read_parquet(cand_parquet).slice(offset, CHUNK_SIZE)
        if chunk_cands.height == 0:
            break

        s1_ids = chunk_cands["source1_entity_id"].to_list()
        cand_ids = chunk_cands["candidate_entity_id"].to_list()

        # Compute features
        feat_df = extract_features_vectorized(s1_ids, cand_ids, s1_dict, cand_dict)

        # Predict probabilities
        probs = predict_pair_probabilities(model, feat_df, feature_cols=feature_cols)

        # Filter matches above calibrated threshold
        chunk_matches = 0
        for s1, cand, prob in zip(s1_ids, cand_ids, probs):
            pair = (s1, cand)
            if pair not in seen_cand:
                seen_cand.add(pair)
                cand_map[s1].append(cand)
            if prob >= DECISION_THRESH:
                match_map[s1].append(cand)
                chunk_matches += 1

        total_matches += chunk_matches
        del chunk_cands, s1_ids, cand_ids, feat_df, probs
        gc.collect()

        logger.info(
            f"Chunk {chunk_idx+1}/{num_chunks} processed in {time.time()-t0:.1f}s | "
            f"Chunk matches: {chunk_matches:,} | Total matches: {total_matches:,}"
        )

    logger.info(f"Prediction complete. Total matches: {total_matches:,} across {len(match_map):,} entities.")

    # 5. Write candidate_pairs.tsv
    cand_path = output_dir / "candidate_pairs.tsv"
    logger.info(f"Writing {cand_path}...")
    with open(cand_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_s1_ids:
            cands = cand_map.get(s1, [])
            f.write(f"{s1}\t{','.join(cands)}\n")

    # 6. Write matching_results.tsv
    match_path = output_dir / "matching_results.tsv"
    logger.info(f"Writing {match_path} (thresh={DECISION_THRESH:.3f})...")
    with open(match_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_s1_ids:
            matches = match_map.get(s1, [])
            f.write(f"{s1}\t{','.join(matches)}\n")

    # 7. Run official submission validator
    logger.info("Running official submission validator...")
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
        logger.info("VALIDATION PASSED! Submissions are valid and ready to upload.")
    else:
        logger.error(f"Validation failed:\n{result.stderr}")
        return 1

    logger.info(f"Pipeline finished in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
