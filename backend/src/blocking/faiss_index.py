"""FAISS vector indexing and dense top-K candidate retrieval.

Constructs indexed nearest-neighbor vector stores and performs batched similarity searches.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
import polars as pl

from backend.src.config import cfg
from backend.src.logger import get_logger

logger = get_logger(__name__)


def build_faiss_index(
    embeddings: np.ndarray,
    output_index_path: Optional[Path] = None,
) -> faiss.Index:
    """Builds a FAISS index with inner-product (cosine) metric on normalized embeddings.

    Args:
        embeddings: Float32 array of shape (N, dim), unit normalized.
        output_index_path: Optional path to write serialized FAISS index.

    Returns:
        faiss.IndexFlatIP instance.
    """
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP (N={embeddings.shape[0]:,}, dim={dim})...")
    
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    elapsed = time.time() - t0
    
    logger.info(f"FAISS index created in {elapsed:.2f}s with {index.ntotal:,} vectors.")

    if output_index_path:
        output_index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_index_path))
        logger.info(f"Saved FAISS index to: {output_index_path}")

    return index


def query_dense_candidates(
    s1_embeddings: np.ndarray,
    s1_ids: List[str],
    target_index: faiss.Index,
    target_ids: List[str],
    top_k: int = 10,
    min_similarity: float = 0.35,
) -> pl.DataFrame:
    """Queries top-K nearest neighbors from target FAISS index for each S1 vector.

    Args:
        s1_embeddings: Query embeddings for Source 1 records.
        s1_ids: List of Source 1 entity_id strings.
        target_index: FAISS index built over target (S2/S3) vectors.
        target_ids: List of target entity_id strings matching the index row order.
        top_k: Number of nearest neighbors to retrieve per S1 record.
        min_similarity: Minimum cosine similarity threshold to keep candidate.

    Returns:
        polars.DataFrame with columns [source1_entity_id, candidate_entity_id, dense_score].
    """
    logger.info(f"Querying Top-{top_k} dense candidates for {len(s1_ids):,} S1 entities...")
    t0 = time.time()

    distances, indices = target_index.search(s1_embeddings, top_k)
    elapsed = time.time() - t0

    logger.info(f"Dense vector search completed in {elapsed:.2f}s.")

    s1_col: List[str] = []
    cand_col: List[str] = []
    score_col: List[float] = []

    target_id_arr = np.array(target_ids)
    num_queries = len(s1_ids)

    for i in range(num_queries):
        s1_id = s1_ids[i]
        for k in range(top_k):
            idx = indices[i, k]
            score = float(distances[i, k])
            if idx >= 0 and idx < len(target_ids) and score >= min_similarity:
                cand_id = target_ids[idx]
                s1_col.append(s1_id)
                cand_col.append(cand_id)
                score_col.append(round(score, 4))

    cand_df = pl.DataFrame({
        "source1_entity_id": s1_col,
        "candidate_entity_id": cand_col,
        "dense_score": score_col,
    })

    logger.info(f"Retrieved {cand_df.height:,} semantic candidate pairs above score {min_similarity}.")
    return cand_df
