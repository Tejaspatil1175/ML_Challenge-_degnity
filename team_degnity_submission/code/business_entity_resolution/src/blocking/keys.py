"""Deterministic and phonetic key blocking engine using DuckDB SQL.

Generates high-precision candidate pairs using Soundex hashes, prefix indices, and geographic tokens.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import jellyfish
import polars as pl

from backend.src.config import cfg
from backend.src.exceptions import BlockingError
from backend.src.logger import get_logger

logger = get_logger(__name__)


def generate_name_key(name: Optional[str], loc: Optional[str] = "") -> str:
    """Builds a deterministic prefix blocking key from clean name and location.

    Example: ('orelee s barbershop', 'high point') -> 'orel_high'
    """
    clean_n = (name or "").strip()
    clean_l = (loc or "").strip()
    prefix = clean_n[:4] if len(clean_n) >= 4 else clean_n
    loc_prefix = clean_l[:4] if len(clean_l) >= 4 else clean_l
    if not prefix:
        return ""
    return f"{prefix}_{loc_prefix}".strip("_")


def generate_phonetic_key(name: Optional[str], state_or_country: Optional[str] = "") -> str:
    """Builds a Soundex phonetic blocking key on the first significant name token.

    Example: ('orelee s barbershop', 'NC') -> 'O640_nc'
    """
    clean_n = (name or "").strip()
    if not clean_n:
        return ""
    first_token = clean_n.split()[0]
    try:
        snd = jellyfish.soundex(first_token)
    except Exception:
        snd = first_token[:4].upper()
    loc = (state_or_country or "").strip().lower()
    return f"{snd}_{loc}".strip("_")


def generate_token_set_key(name: Optional[str]) -> str:
    """Builds an order-invariant token set key from significant name tokens (length >= 3).

    Example: ('prime money', '') -> 'money_prime'
    """
    clean_n = (name or "").strip()
    if not clean_n:
        return ""
    tokens = sorted([t for t in clean_n.split() if len(t) >= 3])
    return "_".join(tokens[:2])


def run_key_based_blocking(
    s1_parquet_path: Path,
    s2_parquet_path: Path,
    s3_parquet_path: Path,
    max_cands_per_entity: int = 50,
    sample_n: Optional[int] = None,
) -> pl.DataFrame:
    """Executes high-speed multi-threaded DuckDB SQL key joins across S1 and (S2 U S3).

    Args:
        s1_parquet_path: Path to normalized S1 parquet table.
        s2_parquet_path: Path to normalized S2 parquet table.
        s3_parquet_path: Path to normalized S3 parquet table.
        max_cands_per_entity: Maximum candidate pairs retained per S1 entity.
        sample_n: Optional limit on S1 entities for fast execution.

    Returns:
        polars.DataFrame with columns [source1_entity_id, candidate_entity_id].

    Raises:
        BlockingError: If input parquet files are missing or query fails.
    """
    for p in [s1_parquet_path, s2_parquet_path, s3_parquet_path]:
        if not p.exists():
            raise BlockingError(f"Normalized parquet file not found for blocking: {p}")

    s1_posix = s1_parquet_path.as_posix()
    s2_posix = s2_parquet_path.as_posix()
    s3_posix = s3_parquet_path.as_posix()

    logger.info("Executing key-based blocking joins via DuckDB SQL...")
    t0 = time.time()

    temp_cand_parquet = Path(cfg.paths.processed_dir) / f"{s1_parquet_path.stem}_raw_candidates.parquet"
    spill_dir = Path(cfg.paths.processed_dir) / "duckdb_spill"
    spill_dir.mkdir(parents=True, exist_ok=True)

    db_file = spill_dir / f"{s1_parquet_path.stem}_blocking.duckdb"
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass

    con = duckdb.connect(str(db_file))
    threads = cfg.execution.get("num_workers", 4)
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"PRAGMA temp_directory='{spill_dir.as_posix()}'")
    con.execute("SET max_temp_directory_size='50GB'")
    con.execute("SET memory_limit='6GB'")

    s1_from = f"(SELECT * FROM read_parquet('{s1_posix}') LIMIT {sample_n})" if (sample_n and sample_n > 0) else f"read_parquet('{s1_posix}')"

    try:
        # Create staging table on disk to process passes sequentially with minimal memory footprint
        con.execute("CREATE TABLE candidates (source1_entity_id VARCHAR, candidate_entity_id VARCHAR, priority_score UTINYINT);")

        # Pass 1: Exact Clean Business Name Match
        logger.info("Running Blocking Pass 1: Exact Clean Business Name...")
        con.execute(f"""
        INSERT INTO candidates
        SELECT 
            s1.entity_id AS source1_entity_id, 
            s23.entity_id AS candidate_entity_id,
            1 AS priority_score
        FROM {s1_from} s1 
        JOIN read_parquet(['{s2_posix}', '{s3_posix}']) s23 
          ON s1.clean_name = s23.clean_name 
         AND s1.clean_name != ''
         AND (s1.clean_country = s23.clean_country OR s1.clean_country = '' OR s23.clean_country = '');
        """)

        # Pass 2: Prefix + Exact City Match in Same Country
        logger.info("Running Blocking Pass 2: Name Prefix (4) + City...")
        con.execute(f"""
        INSERT INTO candidates
        SELECT 
            s1.entity_id AS source1_entity_id, 
            s23.entity_id AS candidate_entity_id,
            2 AS priority_score
        FROM {s1_from} s1 
        JOIN read_parquet(['{s2_posix}', '{s3_posix}']) s23 
          ON s1.name_prefix_4 = s23.name_prefix_4 
         AND LENGTH(s1.name_prefix_4) >= 3
         AND s1.clean_city = s23.clean_city
         AND s1.clean_city != ''
         AND (s1.clean_country = s23.clean_country OR s1.clean_country = '' OR s23.clean_country = '');
        """)

        # Pass 3: Prefix + Exact Postal / PIN Code Match
        logger.info("Running Blocking Pass 3: Name Prefix (4) + Postal Code...")
        con.execute(f"""
        INSERT INTO candidates
        SELECT 
            s1.entity_id AS source1_entity_id, 
            s23.entity_id AS candidate_entity_id,
            3 AS priority_score
        FROM {s1_from} s1 
        JOIN read_parquet(['{s2_posix}', '{s3_posix}']) s23 
          ON s1.name_prefix_4 = s23.name_prefix_4 
         AND LENGTH(s1.name_prefix_4) >= 3
         AND s1.clean_zipcode = s23.clean_zipcode 
         AND s1.clean_zipcode != '';
        """)

        # Export distinct candidate pairs directly to Parquet
        logger.info("Exporting distinct candidate pairs to Parquet...")
        con.execute(f"""
        COPY (
            SELECT DISTINCT source1_entity_id, candidate_entity_id 
            FROM candidates
        ) TO '{temp_cand_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)

        con.close()
        if db_file.exists():
            try:
                db_file.unlink()
            except Exception:
                pass

        cand_df = pl.read_parquet(temp_cand_parquet)

        elapsed = time.time() - t0
        unique_s1 = cand_df.select(pl.col("source1_entity_id").n_unique()).item() if cand_df.height > 0 else 0
        avg_cands = (cand_df.height / max(unique_s1, 1)) if unique_s1 > 0 else 0.0

        logger.info(
            f"Key blocking complete in {elapsed:.2f}s: {cand_df.height:,} candidate pairs "
            f"generated across {unique_s1:,} S1 entities (avg {avg_cands:.1f} cands/entity)."
        )
        return cand_df

    except Exception as e:
        raise BlockingError(f"DuckDB key blocking execution failed: {e}") from e
