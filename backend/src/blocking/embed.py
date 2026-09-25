"""Embedding generation and vector encoding module.

Provides batch encoding for normalized entity texts using SentenceTransformers with TF-IDF fallback.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import polars as pl
from tqdm import tqdm

from backend.src.config import cfg
from backend.src.logger import get_logger

logger = get_logger(__name__)

_MODEL_INSTANCE = None
_FALLBACK_VECTORIZER = None


def get_embedding_model(model_name: Optional[str] = None):
    """Loads and caches the SentenceTransformer embedding model, or falls back to TF-IDF if offline."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            name = model_name or cfg.blocking.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"Loading dense embedding model: {name}")
            _MODEL_INSTANCE = SentenceTransformer(name)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using offline TF-IDF dense projection.")
            _MODEL_INSTANCE = "fallback_tfidf"
    return _MODEL_INSTANCE


def build_composite_text(clean_name: str, clean_address: str, clean_country: str) -> str:
    """Builds unified text representation for semantic embedding."""
    name = (clean_name or "").strip()
    addr = (clean_address or "").strip()
    cntry = (clean_country or "").strip()
    return f"{name} | {addr} | {cntry}".strip(" |")


def batch_embed_texts(
    texts: List[str],
    batch_size: int = 1024,
    show_progress: bool = False,
) -> np.ndarray:
    """Encodes a list of text strings into L2-normalized dense embeddings.

    Args:
        texts: List of string representations to encode.
        batch_size: Batch size for model inference.
        show_progress: Whether to display a progress bar.

    Returns:
        np.ndarray of shape (len(texts), dim) with float32 dtype and unit norm.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    if model != "fallback_tfidf":
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.warning(f"SentenceTransformer encoding failed ({e}). Falling back to TF-IDF.")

    # High-speed offline fallback: Character & word TF-IDF with L2 normalization
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    global _FALLBACK_VECTORIZER
    if _FALLBACK_VECTORIZER is None:
        _FALLBACK_VECTORIZER = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=384,
            sublinear_tf=True,
        )
        tfidf_mat = _FALLBACK_VECTORIZER.fit_transform(texts)
    else:
        tfidf_mat = _FALLBACK_VECTORIZER.transform(texts)

    dense = tfidf_mat.toarray().astype(np.float32)
    # Pad or slice to exactly 384 dims
    if dense.shape[1] < 384:
        pad = np.zeros((dense.shape[0], 384 - dense.shape[1]), dtype=np.float32)
        dense = np.hstack([dense, pad])
    elif dense.shape[1] > 384:
        dense = dense[:, :384]

    return normalize(dense, norm="l2", axis=1).astype(np.float32)


def encode_parquet_table(
    parquet_path: Path,
    output_npy_path: Optional[Path] = None,
    batch_size: int = 1024,
    resume: bool = True,
) -> np.ndarray:
    """Encodes a normalized source parquet table and caches normalized embeddings to disk.

    Args:
        parquet_path: Path to normalized parquet file.
        output_npy_path: Destination path for .npy embedding cache.
        batch_size: Batch size for encoding.
        resume: If True and target .npy exists, loads directly from disk.

    Returns:
        np.ndarray with unit-normalized vectors.
    """
    if output_npy_path and output_npy_path.exists() and resume:
        logger.info(f"Loading cached embeddings from: {output_npy_path}")
        return np.load(output_npy_path)

    logger.info(f"Reading {parquet_path.name} for embedding generation...")
    df = pl.read_parquet(parquet_path)

    texts = [
        build_composite_text(n, a, c)
        for n, a, c in zip(df["clean_name"], df["clean_address"], df["clean_country"])
    ]

    t0 = time.time()
    embeddings = batch_embed_texts(texts, batch_size=batch_size)
    elapsed = time.time() - t0

    logger.info(
        f"Encoded {len(texts):,} texts in {elapsed:.2f}s "
        f"({len(texts) / max(elapsed, 0.001):,.0f} texts/sec). Shape: {embeddings.shape}"
    )

    if output_npy_path:
        output_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy_path, embeddings)
        logger.info(f"Saved embedding vectors to: {output_npy_path}")

    return embeddings
