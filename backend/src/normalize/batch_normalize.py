"""Batch text normalization and parquet caching engine.

Processes large multi-million row datasets and writes optimized Parquet shards to data_cache/processed/.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

from backend.src.config import cfg
from backend.src.data.loader import load_source
from backend.src.data.schema_checks import validate_source_schema
from backend.src.logger import get_logger
from backend.src.normalize.address import extract_address_parts, normalize_address
from backend.src.normalize.name import normalize_name

logger = get_logger(__name__)


def normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Applies canonical text normalization and structured extraction to a source DataFrame.

    Adds columns:
        - clean_name: Normalized, unaccented, expanded name.
        - clean_address: Normalized address.
        - clean_country: Standardized uppercase country ISO code.
        - name_prefix_4: First 4 characters of clean_name for blocking.
        - clean_zipcode: Extracted postal / PIN code.
        - clean_city: Extracted city name.

    Args:
        df: Raw source Polars DataFrame.

    Returns:
        Polars DataFrame with added normalized columns.
    """
    logger.debug(f"Normalizing DataFrame with {df.height:,} rows...")
    t0 = time.time()

    # 1. Normalize name and address via map_elements
    clean_names = df["business_name"].map_elements(normalize_name, return_dtype=pl.Utf8)
    clean_addrs = df["business_address"].map_elements(normalize_address, return_dtype=pl.Utf8)
    clean_countries = (
        df["country"]
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )

    # 2. Extract prefix for fast blocking keys using native slice
    name_prefixes = clean_names.str.slice(0, 4).str.strip_chars()

    # 3. Extract postal codes & cities with progress tracking
    raw_addrs_list = df["business_address"].to_list()
    countries_list = df["country"].to_list()

    zipcodes: List[str] = []
    cities: List[str] = []

    from tqdm import tqdm
    for addr, cntry in tqdm(
        zip(raw_addrs_list, countries_list),
        total=len(raw_addrs_list),
        desc="  -> Parsing address parts",
        unit="rows",
        leave=False,
    ):
        parts = extract_address_parts(addr, country=cntry)
        zipcodes.append(parts.get("zipcode") or "")
        cities.append(parts.get("city") or "")

    # Combine into enriched DataFrame
    df_norm = df.with_columns([
        clean_names.alias("clean_name"),
        clean_addrs.alias("clean_address"),
        clean_countries.alias("clean_country"),
        name_prefixes.alias("name_prefix_4"),
        pl.Series("clean_zipcode", zipcodes, dtype=pl.Utf8),
        pl.Series("clean_city", cities, dtype=pl.Utf8),
    ])

    elapsed = time.time() - t0
    logger.info(f"  [OK] Normalization complete in {elapsed:.2f}s ({df.height / max(elapsed, 0.001):,.0f} rows/s).")
    return df_norm


def run_normalization_pipeline(
    sample_n: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Runs normalization across all 6 raw source files and saves cached Parquet files.

    Args:
        sample_n: If set, normalizes a sample for fast debugging.
        output_dir: Target directory to store parquet files (defaults to data_cache/processed).

    Returns:
        Dictionary mapping source dataset key to cached Parquet path.
    """
    from tqdm import tqdm

    target_dir = output_dir or Path(cfg.paths.processed_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("train_s1", cfg.paths.train_s1, "S1", "train_s1_norm.parquet"),
        ("train_s2", cfg.paths.train_s2, "S2", "train_s2_norm.parquet"),
        ("train_s3", cfg.paths.train_s3, "S3", "train_s3_norm.parquet"),
        ("test_s1", cfg.paths.test_s1, "S1", "test_s1_norm.parquet"),
        ("test_s2", cfg.paths.test_s2, "S2", "test_s2_norm.parquet"),
        ("test_s3", cfg.paths.test_s3, "S3", "test_s3_norm.parquet"),
    ]

    saved_paths: Dict[str, Path] = {}
    total_sources = len(sources)

    with tqdm(total=total_sources, desc="Normalization Progress", unit="file") as pbar:
        for idx, (key, path, prefix, out_name) in enumerate(sources, start=1):
            out_path = target_dir / out_name
            pbar.set_postfix_str(f"[{idx}/{total_sources}] {key}")
            logger.info(f"[{idx}/{total_sources}] Processing {key} ({path.name}) -> {out_name}...")
            
            df = load_source(path, sample_n=sample_n)
            validate_source_schema(df, expected_source=prefix)
            
            df_norm = normalize_dataframe(df)
            df_norm.write_parquet(out_path, compression="zstd")
            
            saved_paths[key] = out_path
            pbar.update(1)
            logger.info(f"Saved normalized parquet: {out_path} ({df_norm.height:,} rows)")

    logger.info("Batch normalization completed for all sources.")
    return saved_paths
