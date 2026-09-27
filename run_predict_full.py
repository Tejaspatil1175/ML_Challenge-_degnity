"""
Memory-efficient full test prediction - streams 26M pairs in chunks to avoid OOM.
Run from workspace root: python run_predict_full.py
"""
from __future__ import annotations
import sys, time, re, gc
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from backend.src.config import cfg
from backend.src.logger import get_logger
from backend.src.model.persist import load_model
from backend.src.model.train import get_feature_columns, predict_pair_probabilities
from backend.src.features.pair_features import _process_pair_chunk
from backend.src.predict.candidate_writer import write_candidate_pairs_tsv
from backend.src.predict.matching_writer import write_matching_results_tsv
from backend.src.predict.validator import validate_submission_files
from backend.src.data.loader import load_source
import polars as pl
import numpy as np

logger = get_logger("run_predict_full")

CHUNK_SIZE = 500_000   # pairs per chunk — ~1.5 GB peak RAM
FEAT_COLS = [
    "source1_entity_id","candidate_entity_id",
    "name_ratio","name_tok_sort","name_tok_set","name_partial","name_jw","name_lev",
    "addr_ratio","addr_tok_sort","addr_tok_set",
    "name_jaccard","addr_jaccard","name_char3",
    "country_match","zip_match","city_match",
    "name_len_diff","addr_len_diff",
]

def build_lookup(norm_df, cols):
    d = {}
    for row in norm_df.select(cols).iter_rows():
        d[row[0]] = (row[1] or "", row[2] or "", row[3] or "", row[4] or "", row[5] or "")
    return d

def main() -> int:
    t_total = time.time()
    proc_dir = Path(cfg.paths.processed_dir)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test S1 IDs
    logger.info("Loading test S1 entities...")
    test_s1_df = load_source(Path(cfg.paths.test_s1))
    all_s1_ids = test_s1_df["entity_id"].to_list()
    logger.info(f"  -> {len(all_s1_ids):,} S1 entities")

    # 2. Load normalized lookups into memory (these are small enough)
    logger.info("Loading normalized parquet lookups...")
    norm_cols = ["entity_id","clean_name","clean_address","clean_country","clean_zipcode","clean_city"]
    s1_dict = build_lookup(pl.read_parquet(proc_dir / "test_s1_norm.parquet"), norm_cols)
    s2_dict = build_lookup(pl.read_parquet(proc_dir / "test_s2_norm.parquet"), norm_cols)
    s3_dict = build_lookup(pl.read_parquet(proc_dir / "test_s3_norm.parquet"), norm_cols)
    cand_dict = {**s2_dict, **s3_dict}
    del s2_dict, s3_dict
    gc.collect()
    logger.info(f"  -> S1 lookup: {len(s1_dict):,} | Cand lookup: {len(cand_dict):,}")

    # 3. Check if test_features.parquet already fully computed
    test_feat_path = proc_dir / "test_features.parquet"
    feat_done = False
    if test_feat_path.exists():
        try:
            n = pl.read_parquet(test_feat_path, columns=["source1_entity_id"])["source1_entity_id"].n_unique()
            if n >= 800_000:
                feat_done = True
                logger.info(f"  -> Reusing cached test_features.parquet ({n:,} unique S1).")
        except Exception:
            pass

    if not feat_done:
        cand_tsv = output_dir / "test_candidate_pairs.tsv"
        logger.info(f"Streaming feature extraction from {cand_tsv} in {CHUNK_SIZE:,}-pair chunks...")
        t0 = time.time()

        # Count total rows first for progress
        with open(cand_tsv, encoding="utf-8") as f:
            total_rows = sum(1 for _ in f) - 1  # minus header
        num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        logger.info(f"  -> {total_rows:,} pairs, {num_chunks} chunks")

        parquet_chunks = []
        chunk_idx = 0

        reader = pl.read_csv_batched(cand_tsv, separator="\t", batch_size=CHUNK_SIZE)
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch = batches[0]

            # Handle aggregated format (source1_entity_id | candidate_entity_ids)
            if "candidate_entity_ids" in batch.columns and "candidate_entity_id" not in batch.columns:
                batch = (
                    batch
                    .filter(pl.col("candidate_entity_ids").is_not_null() & (pl.col("candidate_entity_ids") != ""))
                    .with_columns(pl.col("candidate_entity_ids").str.split(","))
                    .explode("candidate_entity_ids")
                    .rename({"candidate_entity_ids": "candidate_entity_id"})
                )

            s1_ids = batch["source1_entity_id"].to_list()
            cand_ids = batch["candidate_entity_id"].to_list()

            chunk_feat = _process_pair_chunk(s1_ids, cand_ids, s1_dict, cand_dict)

            # Save each chunk as a temporary parquet
            chunk_path = proc_dir / f"_test_feat_chunk_{chunk_idx:04d}.parquet"
            chunk_feat.write_parquet(chunk_path, compression="zstd")
            parquet_chunks.append(chunk_path)

            elapsed = time.time() - t0
            done = min((chunk_idx + 1) * CHUNK_SIZE, total_rows)
            rate = done / max(elapsed, 0.001)
            eta = (total_rows - done) / max(rate, 1)
            logger.info(
                f"  Chunk {chunk_idx+1}/{num_chunks}: {done:,}/{total_rows:,} pairs "
                f"({rate:,.0f}/s, ETA {eta/60:.1f}m)"
            )
            chunk_idx += 1
            del batch, chunk_feat, s1_ids, cand_ids
            gc.collect()

        # Merge all chunk parquets into final test_features.parquet
        logger.info(f"Merging {len(parquet_chunks)} chunk files into test_features.parquet...")
        merged = pl.scan_parquet([str(p) for p in parquet_chunks]).collect()
        merged.write_parquet(test_feat_path, compression="zstd")
        logger.info(f"  -> Saved test_features.parquet: {merged.shape} in {time.time()-t0:.1f}s")

        # Cleanup chunk files
        for p in parquet_chunks:
            try: p.unlink()
            except: pass

        test_feat_df = merged
        del merged
        gc.collect()
    else:
        logger.info("Loading test_features.parquet...")
        test_feat_df = pl.read_parquet(test_feat_path)
        logger.info(f"  -> {test_feat_df.shape}")

    # 4. Write candidate_pairs.tsv
    logger.info("Writing candidate_pairs.tsv...")
    cand_path = write_candidate_pairs_tsv(test_feat_df, all_s1_ids=all_s1_ids)

    # 5. Load model & score
    model_path = Path(cfg.paths.models_dir) / "latest_model.joblib"
    logger.info(f"Loading model from {model_path}...")
    model, metadata = load_model(model_path)
    feature_cols = metadata.get("metadata", {}).get("feature_names") or get_feature_columns(test_feat_df)
    logger.info(f"  -> Features: {feature_cols}")

    logger.info(f"Scoring {test_feat_df.height:,} pairs...")
    t0 = time.time()
    probs = predict_pair_probabilities(model, test_feat_df, feature_cols=feature_cols)
    scored_df = test_feat_df.with_columns(pl.Series("pred_prob", probs))
    logger.info(f"  -> Scoring done in {time.time()-t0:.1f}s")

    # 6. Threshold from eval report or default
    dec_thresh = None
    rep_file = Path(cfg.paths.reports_dir) / "eval_results.md"
    if rep_file.exists():
        try:
            text = rep_file.read_text(encoding="utf-8")
            m = re.search(r"Optimal[^`\n]*`([0-9.]+)`", text)
            if m:
                dec_thresh = float(m.group(1))
                logger.info(f"  -> Threshold from eval report: {dec_thresh:.3f}")
        except Exception:
            pass
    if dec_thresh is None:
        dec_thresh = float(cfg.evaluation.get("default_threshold", 0.65))
        logger.info(f"  -> Using default threshold: {dec_thresh:.3f}")

    # 7. Write matching_results.tsv
    logger.info(f"Writing matching_results.tsv (threshold={dec_thresh:.3f})...")
    match_path = write_matching_results_tsv(
        scored_candidates_df=scored_df,
        all_s1_ids=all_s1_ids,
        threshold=dec_thresh,
    )

    # 8. Validate
    logger.info("Validating submission files...")
    is_valid, issues = validate_submission_files(matching_tsv_path=match_path, candidate_tsv_path=cand_path)
    if is_valid:
        logger.info("  -> PASS")
    else:
        logger.error(f"  -> FAIL: {issues}")
        return 1

    above_thresh = int((scored_df["pred_prob"] >= dec_thresh).sum())
    logger.info(f"\nDone in {time.time()-t_total:.1f}s | Pairs above threshold: {above_thresh:,}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
