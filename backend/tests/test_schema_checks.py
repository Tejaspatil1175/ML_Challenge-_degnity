"""Unit tests for schema checks and input validation."""

import pytest
import polars as pl

from backend.src.data.schema_checks import validate_source_schema, validate_ground_truth_schema
from backend.src.exceptions import SchemaError


def test_valid_source_schema():
    """Test that a well-formed S1 DataFrame passes validation."""
    df = pl.DataFrame({
        "entity_id": ["S1-100", "S1-101"],
        "business_name": ["Acme Corp", "Beta LLC"],
        "business_address": ["123 Main St", "456 Market St"],
        "country": ["US", "US"]
    })
    assert validate_source_schema(df, expected_source="S1") is True


def test_missing_column_raises():
    """Test that missing required column raises SchemaError."""
    df = pl.DataFrame({
        "entity_id": ["S1-100"],
        "business_name": ["Acme Corp"],
        # missing business_address and country
    })
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_source_schema(df, expected_source="S1")


def test_invalid_prefix_raises():
    """Test that mismatched entity ID prefix raises SchemaError."""
    df = pl.DataFrame({
        "entity_id": ["S2-999"],  # Invalid for S1
        "business_name": ["Acme Corp"],
        "business_address": ["123 Main St"],
        "country": ["US"]
    })
    with pytest.raises(SchemaError, match="invalid entity_id prefixes"):
        validate_source_schema(df, expected_source="S1")


def test_empty_dataframe_raises():
    """Test that 0-row DataFrame raises SchemaError when allow_empty=False."""
    df = pl.DataFrame({
        "entity_id": pl.Series(dtype=pl.Utf8),
        "business_name": pl.Series(dtype=pl.Utf8),
        "business_address": pl.Series(dtype=pl.Utf8),
        "country": pl.Series(dtype=pl.Utf8)
    })
    with pytest.raises(SchemaError, match="unexpectedly empty"):
        validate_source_schema(df, expected_source="S1", allow_empty=False)


def test_valid_ground_truth_schema():
    """Test ground truth schema validation."""
    gt_df = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-10,S3-20", ""]
    })
    assert validate_ground_truth_schema(gt_df) is True
