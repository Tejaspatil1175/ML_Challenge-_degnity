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


def _process_pair_chunk(
    s1_ids: List[str],
    cand_ids: List[str],
    s1_dict: Dict[str, Tuple[str, str, str, str, str]],
    cand_dict: Dict[str, Tuple[str, str, str, str, str]],
) -> pl.DataFrame:
    """Worker function to compute pairwise feature vectors for a chunk using preallocated numpy arrays."""
    n = len(s1_ids)
    
    arr_name_ratio = np.empty(n, dtype=np.float32)
    arr_name_tok_sort = np.empty(n, dtype=np.float32)
    arr_name_tok_set = np.empty(n, dtype=np.float32)
    arr_name_partial = np.empty(n, dtype=np.float32)
    arr_name_jw = np.empty(n, dtype=np.float32)
    arr_name_lev = np.empty(n, dtype=np.float32)
    arr_addr_ratio = np.empty(n, dtype=np.float32)
    arr_addr_tok_sort = np.empty(n, dtype=np.float32)
    arr_addr_tok_set = np.empty(n, dtype=np.float32)
    arr_name_jaccard = np.empty(n, dtype=np.float32)
    arr_addr_jaccard = np.empty(n, dtype=np.float32)
    arr_name_char3 = np.empty(n, dtype=np.float32)
    arr_country_match = np.empty(n, dtype=np.float32)
    arr_zip_match = np.empty(n, dtype=np.float32)
    arr_city_match = np.empty(n, dtype=np.float32)
    arr_name_len_diff = np.empty(n, dtype=np.float32)
    arr_addr_len_diff = np.empty(n, dtype=np.float32)

    empty_tuple = ("", "", "", "", "")

    for i in range(n):
        s1_id = s1_ids[i]
        cand_id = cand_ids[i]

        s1_n, s1_a, s1_c, s1_z, s1_ct = s1_dict.get(s1_id, empty_tuple)
        cand_n, cand_a, cand_c, cand_z, cand_ct = cand_dict.get(cand_id, empty_tuple)

        # 1. String metrics (Name)
        arr_name_ratio[i] = round(fuzz.ratio(s1_n, cand_n) / 100.0, 4)
        arr_name_tok_sort[i] = round(fuzz.token_sort_ratio(s1_n, cand_n) / 100.0, 4)
        arr_name_tok_set[i] = round(fuzz.token_set_ratio(s1_n, cand_n) / 100.0, 4)
        arr_name_partial[i] = round(fuzz.partial_ratio(s1_n, cand_n) / 100.0, 4)
        arr_name_jw[i] = round(float(distance.JaroWinkler.similarity(s1_n, cand_n)), 4)
        arr_name_lev[i] = round(float(distance.Levenshtein.normalized_similarity(s1_n, cand_n)), 4)

        # 2. String metrics (Address)
        arr_addr_ratio[i] = round(fuzz.ratio(s1_a, cand_a) / 100.0, 4)
        arr_addr_tok_sort[i] = round(fuzz.token_sort_ratio(s1_a, cand_a) / 100.0, 4)
        arr_addr_tok_set[i] = round(fuzz.token_set_ratio(s1_a, cand_a) / 100.0, 4)

        # 3. N-Gram & Token overlap
        arr_name_jaccard[i] = round(compute_word_jaccard(s1_n, cand_n), 4)
        arr_addr_jaccard[i] = round(compute_word_jaccard(s1_a, cand_a), 4)
        arr_name_char3[i] = round(compute_char_ngram_jaccard(s1_n, cand_n, 3), 4)

        # 4. Structured Geographic and Location Agreement
        arr_country_match[i] = 1.0 if s1_c and s1_c == cand_c else (0.5 if not s1_c or not cand_c else 0.0)
        arr_zip_match[i] = 1.0 if s1_z and s1_z == cand_z else (0.5 if not s1_z or not cand_z else 0.0)
        arr_city_match[i] = 1.0 if s1_ct and s1_ct == cand_ct else (0.5 if not s1_ct or not cand_ct else 0.0)

        # 5. Length Ratios
        len_max_n = max(len(s1_n), len(cand_n), 1)
        arr_name_len_diff[i] = round(abs(len(s1_n) - len(cand_n)) / len_max_n, 4)

        len_max_a = max(len(s1_a), len(cand_a), 1)
        arr_addr_len_diff[i] = round(abs(len(s1_a) - len(cand_a)) / len_max_a, 4)

    return pl.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "name_ratio": arr_name_ratio,
        "name_tok_sort": arr_name_tok_sort,
        "name_tok_set": arr_name_tok_set,
        "name_partial": arr_name_partial,
        "name_jw": arr_name_jw,
        "name_lev": arr_name_lev,
        "addr_ratio": arr_addr_ratio,
        "addr_tok_sort": arr_addr_tok_sort,
        "addr_tok_set": arr_addr_tok_set,
        "name_jaccard": arr_name_jaccard,
        "addr_jaccard": arr_addr_jaccard,
        "name_char3": arr_name_char3,
        "country_match": arr_country_match,
        "zip_match": arr_zip_match,
        "city_match": arr_city_match,
        "name_len_diff": arr_name_len_diff,
        "addr_len_diff": arr_addr_len_diff,
    })


def extract_features_for_candidates(
    candidate_df: pl.DataFrame,
    s1_norm_df: pl.DataFrame,
    s2_norm_df: pl.DataFrame,
    s3_norm_df: pl.DataFrame,
    batch_size: int = 100000,
) -> pl.DataFrame:
    """Extracts pairwise feature matrix across all candidate pairs in memory-efficient batches.

    Args:
        candidate_df: DataFrame with columns [source1_entity_id, candidate_entity_id].
        s1_norm_df: Normalized Source 1 DataFrame.
        s2_norm_df: Normalized Source 2 DataFrame.
        s3_norm_df: Normalized Source 3 DataFrame.
        batch_size: Processing batch size (default 100,000 for minimal memory footprint).

    Returns:
        polars.DataFrame containing candidate IDs and all numeric feature columns.

    Raises:
        FeatureExtractionError: If feature extraction produces NaN or fails.
    """
    total_candidates = candidate_df.height
    logger.info(f"Extracting pairwise feature matrix for {total_candidates:,} candidate pairs...")
    t0 = time.time()

    if total_candidates == 0:
        raise FeatureExtractionError("Cannot extract features for empty candidate DataFrame.")

    # Filter normalized tables to only entities present in the candidate pairs to minimize memory
    needed_s1 = set(candidate_df["source1_entity_id"])
    needed_cand = set(candidate_df["candidate_entity_id"])

    cols = ["entity_id", "clean_name", "clean_address", "clean_country", "clean_zipcode", "clean_city"]

    s1_filtered = s1_norm_df.filter(pl.col("entity_id").is_in(needed_s1)).select(cols)
    target_filtered = (
        pl.concat([s2_norm_df, s3_norm_df])
        .filter(pl.col("entity_id").is_in(needed_cand))
        .unique(subset=["entity_id"])
        .select(cols)
    )

    s1_dict: Dict[str, Tuple[str, str, str, str, str]] = {
        row[0]: (row[1] or "", row[2] or "", row[3] or "", row[4] or "", row[5] or "")
        for row in s1_filtered.iter_rows()
    }
    del s1_filtered

    cand_dict: Dict[str, Tuple[str, str, str, str, str]] = {
        row[0]: (row[1] or "", row[2] or "", row[3] or "", row[4] or "", row[5] or "")
        for row in target_filtered.iter_rows()
    }
    del target_filtered, needed_s1, needed_cand

    from tqdm import tqdm
    import gc

    s1_col = candidate_df["source1_entity_id"].to_list()
    cand_col = candidate_df["candidate_entity_id"].to_list()

    chunks_df: List[pl.DataFrame] = []
    num_batches = (total_candidates + batch_size - 1) // batch_size

    with tqdm(
        total=total_candidates,
        desc="Feature Extraction",
        unit="pairs",
        mininterval=3.0,
        leave=True,
    ) as pbar:
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, total_candidates)

            chunk_s1 = s1_col[start_idx:end_idx]
            chunk_cand = cand_col[start_idx:end_idx]

            chunk_df = _process_pair_chunk(chunk_s1, chunk_cand, s1_dict, cand_dict)
            chunks_df.append(chunk_df)

            pbar.update(end_idx - start_idx)

    # Free memory
    del s1_col, cand_col, s1_dict, cand_dict
    gc.collect()

    logger.info("Concatenating extracted chunk DataFrames...")
    features_df = pl.concat(chunks_df)
    del chunks_df
    gc.collect()

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
