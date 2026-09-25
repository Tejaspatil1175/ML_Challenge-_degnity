"""Embedding generation and vector encoding module.

Provides batch encoding for normalized entity texts using SentenceTransformers / MiniLM.
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


def get_embedding_model(model_name: Optional[str] = None):
    """Loads and caches the SentenceTransformer embedding model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer
        name = model_name or cfg.blocking.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading dense embedding model: {name}")
        _MODEL_INSTANCE = SentenceTransformer(name)
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
    show_progress: bool = True,
) -> np.ndarray:
    """Encodes a list of text strings into L2-normalized dense embeddings.

    Args:
        texts: List of string representations to encode.
        batch_size: Batch size for model inference.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        np.ndarray of shape (len(texts), 384) with float32 dtype and unit norm.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


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

    # Build composite texts
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
