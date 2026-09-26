"""Candidate pool merging, deduplication, and persistence module.

Combines rule-based and semantic candidate pairs and writes candidate_pairs.tsv according to challenge specifications.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional
import polars as pl

from backend.src.config import cfg
from backend.src.logger import get_logger

logger = get_logger(__name__)


def merge_and_deduplicate_candidates(
    key_candidates: pl.DataFrame,
    dense_candidates: Optional[pl.DataFrame] = None,
    max_cands_per_entity: int = 50,
) -> pl.DataFrame:
    """Combines candidate DataFrames, deduplicates, and caps candidates per S1 entity.

    Args:
        key_candidates: DataFrame with [source1_entity_id, candidate_entity_id].
        dense_candidates: Optional DataFrame from dense semantic search.
        max_cands_per_entity: Maximum number of candidate pairs retained per reference entity.

    Returns:
        polars.DataFrame with unique [source1_entity_id, candidate_entity_id] pairs.
    """
    logger.info("Merging candidate pools...")
    t0 = time.time()

    dfs = [key_candidates.select(["source1_entity_id", "candidate_entity_id"])]
    if dense_candidates is not None and dense_candidates.height > 0:
        dfs.append(dense_candidates.select(["source1_entity_id", "candidate_entity_id"]))

    combined = pl.concat(dfs).unique(subset=["source1_entity_id", "candidate_entity_id"])

    # Rank and cap per entity
    ranked = combined.with_columns(
        pl.int_range(0, pl.len()).over("source1_entity_id").alias("rank")
    ).filter(pl.col("rank") < max_cands_per_entity).select(["source1_entity_id", "candidate_entity_id"])

    elapsed = time.time() - t0
    unique_s1 = ranked.select(pl.col("source1_entity_id").n_unique()).item() if ranked.height > 0 else 0
    avg_c = ranked.height / max(unique_s1, 1)

    logger.info(
        f"Merged & capped candidates in {elapsed:.2f}s: {ranked.height:,} total pairs "
        f"across {unique_s1:,} S1 entities ({avg_c:.1f} avg cands/entity)."
    )
    return ranked


def save_candidate_pairs_tsv(
    candidate_df: pl.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Writes candidate pairs to tab-separated TSV file with official schema.

    Schema: source1_entity_id\tcandidate_entity_id
    """
    target = output_path or Path(cfg.paths.candidate_pairs_tsv)
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing candidate pairs to: {target}...")
    candidate_df.select(["source1_entity_id", "candidate_entity_id"]).write_csv(
        target,
        separator="\t",
        include_header=True,
    )

    logger.info(f"Successfully saved {candidate_df.height:,} candidate pairs to TSV.")
    return target
