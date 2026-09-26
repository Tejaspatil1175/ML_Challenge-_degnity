"""Unit tests for model training and persistence."""

import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from backend.src.model.persist import load_model, save_model
from backend.src.model.train import get_feature_columns, predict_pair_probabilities, train_matching_model


@pytest.fixture
def dummy_labeled_features():
    np.random.seed(42)
    n_rows = 100
    s1_ids = [f"S1-{i % 20:03d}" for i in range(n_rows)]
    cand_ids = [f"S2-{i:03d}" for i in range(n_rows)]
    
    # Create simple informative features
    name_sim = np.random.uniform(0, 1, n_rows)
    addr_sim = np.random.uniform(0, 1, n_rows)
    is_match = (name_sim > 0.6) & (addr_sim > 0.4)
    labels = is_match.astype(int)

    data = {
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "name_ratio": name_sim,
        "addr_ratio": addr_sim,
        "is_match": labels,
    }
    return pl.DataFrame(data)


def test_train_matching_model_and_persist(dummy_labeled_features, tmp_path):
    model, feature_names, val_split, val_probs = train_matching_model(
        dummy_labeled_features,
        val_ratio=0.3,
        model_version="test_v1",
        custom_params={"n_estimators": 20, "learning_rate": 0.1},
    )

    assert model is not None
    assert feature_names == ["name_ratio", "addr_ratio"]
    assert len(val_probs) == val_split.height

    # Test persistence
    save_path = tmp_path / "model.joblib"
    save_model(model, save_path, metadata={"features": feature_names})
    assert save_path.exists()
    assert save_path.with_suffix(".json").exists()

    # Test loading and identical predictions
    loaded_model, metadata = load_model(save_path)
    assert metadata["metadata"]["features"] == feature_names

    orig_probs = predict_pair_probabilities(model, dummy_labeled_features, feature_names)
    loaded_probs = predict_pair_probabilities(loaded_model, dummy_labeled_features, feature_names)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5)
