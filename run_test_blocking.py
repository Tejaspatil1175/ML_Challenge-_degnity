"""High-recall blocking execution on test set.

Runs 7 complementary passes across names, word-order permutations, and addresses.
Exports deduplicated candidate pairs to data_cache/processed/test_s1_norm_raw_candidates.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from backend.src.config import cfg
from backend.src.logger import get_logger
from backend.src.blocking.keys import run_key_based_blocking

logger = get_logger("run_test_blocking")


def main() -> int:
    t0 = time.time()
    proc_dir = Path(cfg.paths.processed_dir)

    s1_norm = proc_dir / "test_s1_norm.parquet"
    s2_norm = proc_dir / "test_s2_norm.parquet"
    s3_norm = proc_dir / "test_s3_norm.parquet"

    logger.info("Starting high-recall blocking on test set...")
    cand_df = run_key_based_blocking(
        s1_parquet_path=s1_norm,
        s2_parquet_path=s2_norm,
        s3_parquet_path=s3_norm,
        max_cands_per_entity=30,
    )

    n_pairs = cand_df.height
    n_unique_s1 = cand_df["source1_entity_id"].n_unique()
    avg_per_s1 = n_pairs / max(n_unique_s1, 1)

    logger.info("=" * 60)
    logger.info(f"BLOCKING COMPLETE in {time.time()-t0:.1f}s")
    logger.info(f"Total candidate pairs: {n_pairs:,}")
    logger.info(f"Unique S1 entities with candidates: {n_unique_s1:,} / 1,732,544 ({n_unique_s1/1732544*100:.2f}%)")
    logger.info(f"Average candidates per S1: {avg_per_s1:.1f}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
