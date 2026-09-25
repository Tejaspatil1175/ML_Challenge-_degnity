"""Schema verification and validation checks for raw dataset inputs.

Ensures column names, types, entity ID prefixes, and data integrity adhere to invariants.
"""

from __future__ import annotations

from typing import List, Optional, Set
import polars as pl

from backend.src.exceptions import SchemaError
from backend.src.logger import get_logger

logger = get_logger(__name__)

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
VALID_PREFIXES = {"S1": "S1-", "S2": "S2-", "S3": "S3-"}


def validate_source_schema(
    df: pl.DataFrame,
    expected_source: Optional[str] = None,
    allow_empty: bool = False,
) -> bool:
    """Validates the schema and structure of a source DataFrame (S1, S2, or S3).

    Args:
        df: Polars DataFrame to validate.
        expected_source: One of 'S1', 'S2', 'S3' to enforce entity ID prefix checks.
        allow_empty: If False, raises SchemaError on empty DataFrames.

    Returns:
        True if all validations pass.

    Raises:
        SchemaError: If any column is missing, row is fully null, or prefix is invalid.
    """
    if df is None:
        raise SchemaError("Source DataFrame cannot be None.")

    if df.height == 0 and not allow_empty:
        raise SchemaError("Source DataFrame is unexpectedly empty (0 rows).")

    # 1. Column presence check
    missing_cols = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SchemaError(
            f"Missing required columns in source DataFrame: {missing_cols}. "
            f"Found columns: {df.columns}"
        )

    # 2. Entity ID prefix check (if expected_source specified)
    if expected_source in VALID_PREFIXES and df.height > 0:
        expected_prefix = VALID_PREFIXES[expected_source]
        # Check first 10,000 IDs for prefix adherence
        sample_ids = df.select("entity_id").head(10000)["entity_id"].to_list()
        invalid_ids = [eid for eid in sample_ids if not str(eid).startswith(expected_prefix)]
        if invalid_ids:
            raise SchemaError(
                f"Source {expected_source} contains invalid entity_id prefixes. "
                f"Expected prefix '{expected_prefix}', found invalid IDs: {invalid_ids[:5]}"
            )

    # 3. Entity ID uniqueness check (S1 should be 100% unique reference set)
    if expected_source == "S1" and df.height > 0:
        n_unique = df.select(pl.col("entity_id").n_unique()).item()
        if n_unique != df.height:
            raise SchemaError(
                f"Source 1 contains duplicate entity_id entries: {df.height - n_unique} duplicates detected."
            )

    # 4. Check for completely null / blank rows
    if df.height > 0:
        blank_rows = df.filter(
            (pl.col("entity_id") == "") &
            (pl.col("business_name") == "") &
            (pl.col("business_address") == "")
        ).height
        if blank_rows > 0:
            raise SchemaError(f"Found {blank_rows} completely empty/blank rows in DataFrame.")

    logger.debug(f"Source schema validation passed for {expected_source or 'generic source'}.")
    return True


def validate_ground_truth_schema(df: pl.DataFrame) -> bool:
    """Validates the schema of train_ground_truth.tsv DataFrame.

    Args:
        df: Polars DataFrame to validate.

    Returns:
        True if validation passes.

    Raises:
        SchemaError: If ground truth schema or ID format is invalid.
    """
    if df is None or df.height == 0:
        raise SchemaError("Ground truth DataFrame is empty or None.")

    missing_cols = [col for col in REQUIRED_GROUND_TRUTH_COLUMNS if col not in df.columns]
    if missing_cols:
        raise SchemaError(f"Missing required ground truth columns: {missing_cols}")

    # Check S1 prefix in source1_entity_id
    sample_ids = df.select("source1_entity_id").head(5000)["source1_entity_id"].to_list()
    invalid_s1 = [eid for eid in sample_ids if not str(eid).startswith("S1-")]
    if invalid_s1:
        raise SchemaError(f"Ground truth source1_entity_id contains invalid prefixes: {invalid_s1[:5]}")

    logger.debug("Ground truth schema validation passed.")
    return True
