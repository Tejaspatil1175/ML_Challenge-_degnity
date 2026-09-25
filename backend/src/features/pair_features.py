"""Pairwise feature engineering engine.

Extracts string distances, token overlap, phonetic metrics, and structured agreement features for candidate pairs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import duckdb
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz

from backend.src.config import cfg
from backend.src.exceptions import FeatureExtractionError
from backend.src.logger import get_logger

logger = get_logger(__name__)


def compute_word_jaccard(s1: str, s2: str) -> float:
    """Computes word-level Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    union = len(set1.union(set2))
    return len(set1.intersection(set2)) / union if union > 0 else 0.0


def compute_char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity."""
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 and s1 else 0.0
    set1 = {s1[i : i + n] for i in range(len(s1) - n + 1)}
    set2 = {s2[i : i + n] for i in range(len(s2) - n + 1)}
    union = len(set1.union(set2))
    return len(set1.intersection(set2)) / union if union > 0 else 0.0


def extract_pairwise_feature_vector(
    s1_name: str,
    cand_name: str,
    s1_addr: str,
    cand_addr: str,
    s1_country: str,
    cand_country: str,
    s1_zip: str,
    cand_zip: str,
    s1_city: str,
    cand_city: str,
) -> Dict[str, float]:
    """Calculates all similarity features for a single candidate pair."""
    s1_n = s1_name or ""
    cand_n = cand_name or ""
    s1_a = s1_addr or ""
    cand_a = cand_addr or ""

    # 1. String metrics (Name)
    name_ratio = fuzz.ratio(s1_n, cand_n) / 100.0
    name_tok_sort = fuzz.token_sort_ratio(s1_n, cand_n) / 100.0
    name_tok_set = fuzz.token_set_ratio(s1_n, cand_n) / 100.0
    name_partial = fuzz.partial_ratio(s1_n, cand_n) / 100.0
    name_jw = float(distance.JaroWinkler.similarity(s1_n, cand_n))
    name_lev = float(distance.Levenshtein.normalized_similarity(s1_n, cand_n))

    # 2. String metrics (Address)
    addr_ratio = fuzz.ratio(s1_a, cand_a) / 100.0
    addr_tok_sort = fuzz.token_sort_ratio(s1_a, cand_a) / 100.0
    addr_tok_set = fuzz.token_set_ratio(s1_a, cand_a) / 100.0

    # 3. N-Gram & Token overlap
    name_jaccard = compute_word_jaccard(s1_n, cand_n)
    addr_jaccard = compute_word_jaccard(s1_a, cand_a)
    name_char3 = compute_char_ngram_jaccard(s1_n, cand_n, 3)

    # 4. Structured Geographic and Location Agreement
    country_match = 1.0 if s1_country and s1_country == cand_country else (0.5 if not s1_country or not cand_country else 0.0)
    zip_match = 1.0 if s1_zip and s1_zip == cand_zip else (0.5 if not s1_zip or not cand_zip else 0.0)
    city_match = 1.0 if s1_city and s1_city == cand_city else (0.5 if not s1_city or not cand_city else 0.0)

    # 5. Length Ratios
    len_max_n = max(len(s1_n), len(cand_n), 1)
    name_len_diff = abs(len(s1_n) - len(cand_n)) / len_max_n

    len_max_a = max(len(s1_a), len(cand_a), 1)
    addr_len_diff = abs(len(s1_a) - len(cand_a)) / len_max_a

    return {
        "name_ratio": round(name_ratio, 4),
        "name_tok_sort": round(name_tok_sort, 4),
        "name_tok_set": round(name_tok_set, 4),
        "name_partial": round(name_partial, 4),
        "name_jw": round(name_jw, 4),
        "name_lev": round(name_lev, 4),
        "addr_ratio": round(addr_ratio, 4),
        "addr_tok_sort": round(addr_tok_sort, 4),
        "addr_tok_set": round(addr_tok_set, 4),
        "name_jaccard": round(name_jaccard, 4),
        "addr_jaccard": round(addr_jaccard, 4),
        "name_char3": round(name_char3, 4),
        "country_match": country_match,
        "zip_match": zip_match,
        "city_match": city_match,
        "name_len_diff": round(name_len_diff, 4),
        "addr_len_diff": round(addr_len_diff, 4),
    }


def extract_features_for_candidates(
    candidate_df: pl.DataFrame,
    s1_norm_df: pl.DataFrame,
    s2_norm_df: pl.DataFrame,
    s3_norm_df: pl.DataFrame,
    batch_size: int = 50000,
) -> pl.DataFrame:
    """Extracts pairwise feature matrix across all candidate pairs.

    Args:
        candidate_df: DataFrame with columns [source1_entity_id, candidate_entity_id].
        s1_norm_df: Normalized Source 1 DataFrame.
        s2_norm_df: Normalized Source 2 DataFrame.
        s3_norm_df: Normalized Source 3 DataFrame.
        batch_size: Processing batch size.

    Returns:
        polars.DataFrame containing candidate IDs and all numeric feature columns.

    Raises:
        FeatureExtractionError: If feature extraction produces NaN or fails.
    """
    logger.info(f"Extracting pairwise feature matrix for {candidate_df.height:,} candidate pairs...")
    t0 = time.time()

    if candidate_df.height == 0:
        raise FeatureExtractionError("Cannot extract features for empty candidate DataFrame.")

    # Combine S2 and S3 for fast lookup
    target_combined = pl.concat([s2_norm_df, s3_norm_df]).unique(subset=["entity_id"])

    # Prepare lookup dictionaries for instant access
    s1_dict = {
        row["entity_id"]: row
        for row in s1_norm_df.select([
            "entity_id", "clean_name", "clean_address", "clean_country", "clean_zipcode", "clean_city"
        ]).iter_rows(named=True)
    }

    cand_dict = {
        row["entity_id"]: row
        for row in target_combined.select([
            "entity_id", "clean_name", "clean_address", "clean_country", "clean_zipcode", "clean_city"
        ]).iter_rows(named=True)
    }

    feature_rows = []
    cand_pairs = list(zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_id"]))

    for s1_id, cand_id in cand_pairs:
        s1_rec = s1_dict.get(s1_id, {})
        cand_rec = cand_dict.get(cand_id, {})

        feats = extract_pairwise_feature_vector(
            s1_name=s1_rec.get("clean_name", ""),
            cand_name=cand_rec.get("clean_name", ""),
            s1_addr=s1_rec.get("clean_address", ""),
            cand_addr=cand_rec.get("clean_address", ""),
            s1_country=s1_rec.get("clean_country", ""),
            cand_country=cand_rec.get("clean_country", ""),
            s1_zip=s1_rec.get("clean_zipcode", ""),
            cand_zip=cand_rec.get("clean_zipcode", ""),
            s1_city=s1_rec.get("clean_city", ""),
            cand_city=cand_rec.get("clean_city", ""),
        )
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = cand_id
        feature_rows.append(feats)

    features_df = pl.DataFrame(feature_rows)

    # Check for NaNs or nulls
    null_counts = features_df.null_count().sum_horizontal().item()
    if null_counts > 0:
        raise FeatureExtractionError(f"Found {null_counts} null values in feature matrix.")

    elapsed = time.time() - t0
    rate = features_df.height / max(elapsed, 0.001)
    logger.info(
        f"Feature extraction complete in {elapsed:.2f}s ({rate:,.0f} pairs/sec). "
        f"Shape: {features_df.shape}"
    )

    return features_df
