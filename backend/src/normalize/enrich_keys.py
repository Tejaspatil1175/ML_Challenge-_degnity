"""Enriches normalized parquet files with high-recall blocking keys.

Adds:
- token_sort_key: Sorted top-2 business name tokens (>= 3 chars) excluding legal stopwords.
- addr_num: First sequence of digits in address (street / plot / building number).
- addr_tok_key: Sorted top-2 address tokens (>= 5 chars) excluding common road/street stopwords.
"""

from __future__ import annotations

import gc
import re
import sys
import time
from pathlib import Path
from typing import List

WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import polars as pl
from tqdm import tqdm

from backend.src.config import cfg
from backend.src.logger import get_logger

logger = get_logger(__name__)

_LEGAL_STOPWORDS = re.compile(
    r"\b(private|limited|incorporated|corp|corporation|llc|inc|ltd|co|pvt|services|enterprises|enterprise|solutions|holdings|group)\b",
    re.IGNORECASE,
)
_ADDR_STOPWORDS = re.compile(
    r"\b(street|avenue|road|floor|apartment|near|behind|opposite|block|phase|nagar|colony|building|house|plot|lane|cross|main|sector)\b",
    re.IGNORECASE,
)
_DIGIT_PATTERN = re.compile(r"([0-9]+[a-z0-9\-\/]*)")


def compute_enrichment_columns(
    clean_names: List[str],
    clean_addrs: List[str],
    batch_size: int = 250000,
) -> tuple[List[str], List[str], List[str]]:
    """Computes token_sort_key, addr_num, and addr_tok_key efficiently."""
    n_rows = len(clean_names)
    tok_keys: List[str] = [""] * n_rows
    addr_nums: List[str] = [""] * n_rows
    addr_toks: List[str] = [""] * n_rows

    for i in range(n_rows):
        n = clean_names[i]
        a = clean_addrs[i]

        # 1. Token Sort Key
        if n:
            n_cl = _LEGAL_STOPWORDS.sub("", n)
            toks = sorted([w for w in n_cl.split() if len(w) >= 3])
            tok_keys[i] = "_".join(toks[:2]) if len(toks) >= 2 else (toks[0] if toks else "")

        # 2. Address Number & Address Significant Tokens
        if a:
            m = _DIGIT_PATTERN.search(a)
            if m:
                addr_nums[i] = m.group(1)

            a_cl = _ADDR_STOPWORDS.sub("", a)
            atok = sorted([w for w in a_cl.split() if len(w) >= 5])
            addr_toks[i] = "_".join(atok[:2]) if len(atok) >= 2 else (atok[0] if atok else "")

    return tok_keys, addr_nums, addr_toks


def enrich_parquet(parquet_path: Path) -> None:
    """Reads a normalized parquet file, appends enrichment columns, and writes back."""
    if not parquet_path.exists():
        logger.warning(f"File not found: {parquet_path}")
        return

    logger.info(f"Checking {parquet_path.name}...")
    schema = pl.read_parquet_schema(parquet_path)
    if "token_sort_key" in schema and "addr_num" in schema and "addr_tok_key" in schema:
        logger.info(f"  -> {parquet_path.name} already enriched. Skipping.")
        return

    t0 = time.time()
    logger.info(f"Loading {parquet_path.name}...")
    df = pl.read_parquet(parquet_path)
    n_rows = df.height
    logger.info(f"  -> {n_rows:,} rows loaded.")

    c_names = df["clean_name"].to_list()
    c_addrs = df["clean_address"].to_list()

    logger.info(f"Computing enrichment keys for {n_rows:,} rows...")
    tok_keys, addr_nums, addr_toks = compute_enrichment_columns(c_names, c_addrs)

    df_enriched = df.with_columns([
        pl.Series("token_sort_key", tok_keys, dtype=pl.Utf8),
        pl.Series("addr_num", addr_nums, dtype=pl.Utf8),
        pl.Series("addr_tok_key", addr_toks, dtype=pl.Utf8),
    ])

    del c_names, c_addrs, tok_keys, addr_nums, addr_toks
    gc.collect()

    temp_path = parquet_path.with_suffix(".tmp.parquet")
    logger.info(f"Writing enriched parquet to {temp_path.name}...")
    df_enriched.write_parquet(temp_path, compression="zstd")

    del df, df_enriched
    gc.collect()

    temp_path.replace(parquet_path)
    logger.info(f"Successfully updated {parquet_path.name} in {time.time()-t0:.1f}s.")


def main() -> int:
    proc_dir = Path(cfg.paths.processed_dir)
    targets = [
        proc_dir / "test_s1_norm.parquet",
        proc_dir / "test_s2_norm.parquet",
        proc_dir / "test_s3_norm.parquet",
    ]
    for target in targets:
        enrich_parquet(target)
    logger.info("Enrichment complete for all targets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
