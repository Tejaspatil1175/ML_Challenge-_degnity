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
) -> pl.DataFrame:
    """Executes high-speed multi-threaded DuckDB SQL key joins across S1 and (S2 U S3).

    Args:
        s1_parquet_path: Path to normalized S1 parquet table.
        s2_parquet_path: Path to normalized S2 parquet table.
        s3_parquet_path: Path to normalized S3 parquet table.
        max_cands_per_entity: Maximum candidate pairs retained per S1 entity.

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

    con = duckdb.connect(database=":memory:")
    threads = cfg.execution.get("num_workers", 4)
    con.execute(f"PRAGMA threads={threads}")

    try:
        # Join S1 against S2 and S3 union on exact prefix/zip/country or exact name
        query = f"""
        WITH s23 AS (
            SELECT entity_id, clean_name, clean_address, clean_country, clean_zipcode, clean_city, name_prefix_4
            FROM read_parquet(['{s2_posix}', '{s3_posix}'])
        ),
        raw_candidates AS (
            SELECT 
                s1.entity_id AS source1_entity_id, 
                s23.entity_id AS candidate_entity_id,
                CASE 
                    WHEN s1.clean_name = s23.clean_name THEN 1
                    WHEN s1.clean_zipcode = s23.clean_zipcode AND s1.clean_zipcode != '' THEN 2
                    WHEN s1.clean_city = s23.clean_city AND s1.clean_city != '' THEN 3
                    ELSE 4
                END AS priority_score
            FROM read_parquet('{s1_posix}') s1
            JOIN s23 
              ON (s1.clean_country = s23.clean_country OR s1.clean_country = '' OR s23.clean_country = '')
             AND (
                 (s1.name_prefix_4 = s23.name_prefix_4 AND LENGTH(s1.name_prefix_4) >= 3)
                 OR (s1.clean_name = s23.clean_name AND s1.clean_name != '')
                 OR (s1.clean_zipcode = s23.clean_zipcode AND s1.clean_zipcode != '' AND s1.name_prefix_4 = s23.name_prefix_4)
             )
        ),
        ranked_candidates AS (
            SELECT 
                source1_entity_id, 
                candidate_entity_id,
                ROW_NUMBER() OVER(PARTITION BY source1_entity_id ORDER BY priority_score) as rank
            FROM raw_candidates
        )
        SELECT source1_entity_id, candidate_entity_id
        FROM ranked_candidates
        WHERE rank <= {max_cands_per_entity}
        """

        arrow_table = con.execute(query).arrow()
        cand_df = pl.from_arrow(arrow_table)
        con.close()

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
