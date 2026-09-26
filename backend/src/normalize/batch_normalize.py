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


def normalize_dataframe(df: pl.DataFrame, dataset_name: str = "") -> pl.DataFrame:
    """Applies canonical text normalization and structured extraction in a fast single pass.

    Adds columns:
        - clean_name: Normalized, unaccented, expanded name.
        - clean_address: Normalized address.
        - clean_country: Standardized uppercase country ISO code.
        - name_prefix_4: First 4 characters of clean_name for blocking.
        - clean_zipcode: Extracted postal / PIN code.
        - clean_city: Extracted city name.

    Args:
        df: Raw source Polars DataFrame.
        dataset_name: Optional name for progress display.

    Returns:
        Polars DataFrame with added normalized columns.
    """
    from tqdm import tqdm

    n_rows = df.height
    logger.info(f"  -> Starting high-speed normalization on {n_rows:,} rows...")
    t0 = time.time()

    raw_names = df["business_name"].to_list()
    raw_addrs = df["business_address"].to_list()
    raw_countries = df["country"].to_list()

    clean_names: List[str] = []
    clean_addrs: List[str] = []
    clean_countries: List[str] = []
    name_prefixes: List[str] = []
    zipcodes: List[str] = []
    cities: List[str] = []

    pbar_desc = f"  [{dataset_name}] Normalizing" if dataset_name else "  Normalizing rows"
    for name, addr, cntry in tqdm(
        zip(raw_names, raw_addrs, raw_countries),
        total=n_rows,
        desc=pbar_desc,
        unit="rows",
        leave=False,
    ):
        c_name = normalize_name(name)
        c_addr = normalize_address(addr)
        c_cntry = (str(cntry).strip().upper()) if cntry is not None else ""

        clean_names.append(c_name)
        clean_addrs.append(c_addr)
        clean_countries.append(c_cntry)
        name_prefixes.append(c_name[:4].strip())

        parts = extract_address_parts(addr, country=c_cntry, clean_address=c_addr)
        zipcodes.append(parts.get("zipcode") or "")
        cities.append(parts.get("city") or "")

    # Construct output Polars DataFrame in one batch
    df_norm = df.with_columns([
        pl.Series("clean_name", clean_names, dtype=pl.Utf8),
        pl.Series("clean_address", clean_addrs, dtype=pl.Utf8),
        pl.Series("clean_country", clean_countries, dtype=pl.Utf8),
        pl.Series("name_prefix_4", name_prefixes, dtype=pl.Utf8),
        pl.Series("clean_zipcode", zipcodes, dtype=pl.Utf8),
        pl.Series("clean_city", cities, dtype=pl.Utf8),
    ])

    elapsed = time.time() - t0
    rate = n_rows / max(elapsed, 0.001)
    logger.info(f"  [OK] Finished {dataset_name} ({n_rows:,} rows) in {elapsed:.2f}s ({rate:,.0f} rows/s).")
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
            
            df_norm = normalize_dataframe(df, dataset_name=key)
            df_norm.write_parquet(out_path, compression="zstd")
            
            saved_paths[key] = out_path
            pbar.update(1)
            logger.info(f"Saved normalized parquet: {out_path} ({df_norm.height:,} rows)")

    logger.info("Batch normalization completed for all sources.")
    return saved_paths
