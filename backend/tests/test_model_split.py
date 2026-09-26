"""Unit tests for leakage-safe entity splitting."""

import polars as pl
import pytest

from backend.src.model.split import split_train_val, get_group_kfold_splits


def test_split_train_val_no_leakage():
    # 5 entities with multiple candidate rows each
    data = {
        "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3", "S1-3", "S1-3", "S1-4", "S1-5"],
        "candidate_entity_id": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S2-6", "S2-7", "S2-8"],
        "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "is_match": [1, 0, 1, 0, 1, 0, 0, 1],
    }
    df = pl.DataFrame(data)

    train_df, val_df = split_train_val(df, val_ratio=0.4, random_seed=42)

    train_s1 = set(train_df["source1_entity_id"].to_list())
    val_s1 = set(val_df["source1_entity_id"].to_list())

    # Ensure zero overlap
    overlap = train_s1.intersection(val_s1)
    assert len(overlap) == 0, f"Leakage detected between train and val: {overlap}"

    # Ensure total rows sum up
    assert train_df.height + val_df.height == df.height
    assert len(train_s1) + len(val_s1) == 5


def test_group_kfold_splits():
    data = {
        "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
        "candidate_entity_id": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S2-6"],
        "is_match": [1, 0, 1, 0, 0, 1],
    }
    df = pl.DataFrame(data)

    splits = get_group_kfold_splits(df, n_splits=3)
    assert len(splits) == 3

    for fold_idx, (train_f, val_f) in enumerate(splits):
        train_s1 = set(train_f["source1_entity_id"].to_list())
        val_s1 = set(val_f["source1_entity_id"].to_list())
        overlap = train_s1.intersection(val_s1)
        assert len(overlap) == 0, f"Fold {fold_idx} has leakage: {overlap}"
        assert train_f.height + val_f.height == df.height
