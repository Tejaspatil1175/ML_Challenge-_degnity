"""High-performance TSV data loader using DuckDB and Polars.

Provides zero-copy streaming ingestion and sampled loading of source datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import duckdb
import polars as pl

from backend.src.config import cfg
from backend.src.exceptions import DataLoadError
from backend.src.logger import get_logger

logger = get_logger(__name__)


def load_source(
    file_path: Union[str, Path],
    sample_n: Optional[int] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Loads a source TSV file (S1, S2, or S3) into a Polars DataFrame.

    Columns expected: entity_id, business_name, business_address, country.

    Args:
        file_path: Absolute or relative path to the TSV file.
        sample_n: If specified, returns a sampled subset of rows.
        seed: Random seed for deterministic sampling.

    Returns:
        polars.DataFrame containing the four canonical columns.

    Raises:
        DataLoadError: If the file cannot be found, parsed, or read.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataLoadError(f"Source data file does not exist: {path}")

    # Standard POSIX path representation for DuckDB SQL compatibility
    posix_path = path.as_posix()
    logger.debug(f"Loading source TSV via DuckDB: {posix_path} (sample_n={sample_n})")

    try:
        con = duckdb.connect(database=":memory:")
        
        # Configure DuckDB for multi-threaded TSV parsing
        threads = cfg.execution.get("num_workers", 4)
        con.execute(f"PRAGMA threads={threads}")

        query = f"""
            SELECT 
                CAST(entity_id AS VARCHAR) AS entity_id,
                COALESCE(CAST(business_name AS VARCHAR), '') AS business_name,
                COALESCE(CAST(business_address AS VARCHAR), '') AS business_address,
                COALESCE(CAST(country AS VARCHAR), '') AS country
            FROM read_csv(
                '{posix_path}',
                delim='\\t',
                header=True,
                all_varchar=True,
                quote='',
                escape='',
                null_padding=True,
                ignore_errors=False
            )
        """

        if sample_n is not None and sample_n > 0:
            # Deterministic reservoir sample via DuckDB
            query += f" USING SAMPLE {sample_n} ROWS (reservoir, {seed})"

        arrow_table = con.execute(query).arrow()
        df = pl.from_arrow(arrow_table)
        con.close()

        logger.info(
            f"Successfully loaded '{path.name}': {df.height:,} rows, {df.width} columns."
        )
        return df

    except Exception as e:
        raise DataLoadError(f"Failed to load TSV source file '{path}': {e}") from e


def load_ground_truth(
    file_path: Union[str, Path],
    sample_n: Optional[int] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Loads train_ground_truth.tsv into a Polars DataFrame.

    Columns expected: source1_entity_id, matched_entity_ids.

    Args:
        file_path: Path to train_ground_truth.tsv.
        sample_n: If specified, returns a sampled subset.
        seed: Random seed for sampling.

    Returns:
        polars.DataFrame with columns [source1_entity_id, matched_entity_ids].
    """
    path = Path(file_path)
    if not path.exists():
        raise DataLoadError(f"Ground truth file does not exist: {path}")

    posix_path = path.as_posix()
    logger.debug(f"Loading ground truth TSV: {posix_path} (sample_n={sample_n})")

    try:
        con = duckdb.connect(database=":memory:")
        query = f"""
            SELECT 
                CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
                COALESCE(CAST(matched_entity_ids AS VARCHAR), '') AS matched_entity_ids
            FROM read_csv(
                '{posix_path}',
                delim='\\t',
                header=True,
                all_varchar=True,
                quote='',
                escape='',
                null_padding=True,
                ignore_errors=False
            )
        """
        if sample_n is not None and sample_n > 0:
            query += f" USING SAMPLE {sample_n} ROWS (reservoir, {seed})"

        arrow_table = con.execute(query).arrow()
        df = pl.from_arrow(arrow_table)
        con.close()

        logger.info(
            f"Successfully loaded ground truth '{path.name}': {df.height:,} records."
        )
        return df

    except Exception as e:
        raise DataLoadError(f"Failed to load ground truth file '{path}': {e}") from e
