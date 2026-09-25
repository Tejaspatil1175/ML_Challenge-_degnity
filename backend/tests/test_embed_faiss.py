"""Unit tests for dense embedding generation and vector search indexing."""

import numpy as np
from backend.src.blocking.embed import batch_embed_texts, build_composite_text
from backend.src.blocking.faiss_index import build_faiss_index, query_dense_candidates


def test_build_composite_text():
    """Test text representation construction."""
    res = build_composite_text("acme corp", "123 main st", "US")
    assert res == "acme corp | 123 main st | US"


def test_dense_embedding_generation():
    """Test embedding vectors are generated with valid dimensions and L2 normalization."""
    texts = ["walmart supercenter", "target store", "best buy"]
    vecs = batch_embed_texts(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 3
    assert vecs.shape[1] == 384
    # Check L2 unit norm
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        assert np.isclose(n, 1.0, atol=1e-3)


def test_vector_index_search():
    """Test vector index construction and candidate querying."""
    target_texts = ["apple store", "microsoft store", "google headquarters"]
    target_vecs = batch_embed_texts(target_texts)
    index = build_faiss_index(target_vecs)

    query_vec = batch_embed_texts(["apple store inc"])
    cands = query_dense_candidates(
        s1_embeddings=query_vec,
        s1_ids=["S1-1"],
        target_index=index,
        target_ids=["S2-1", "S2-2", "S2-3"],
        top_k=2,
        min_similarity=0.10,
    )
    assert cands.height >= 1
    assert "source1_entity_id" in cands.columns
    assert "candidate_entity_id" in cands.columns
    assert "dense_score" in cands.columns
