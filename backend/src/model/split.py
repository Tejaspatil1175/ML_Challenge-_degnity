"""Leakage-safe entity-level dataset splitting.

Ensures candidate pairs for any given source1_entity_id are grouped exclusively
into either train or validation sets to avoid data leakage.
"""

from __future__ import annotations

from typing import Generator, List, Tuple
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from backend.src.config import cfg
from backend.src.logger import get_logger

logger = get_logger(__name__)


def split_train_val(
    df: pl.DataFrame,
    val_ratio: float = 0.20,
    random_seed: int = 42,
    group_col: str = "source1_entity_id",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Splits a candidate/feature DataFrame into train and validation sets by entity group.

    Args:
        df: Polars DataFrame containing candidate pairs and features.
        val_ratio: Fraction of unique entities reserved for validation (e.g. 0.20).
        random_seed: Random state seed.
        group_col: Column name identifying reference entity groups.

    Returns:
        Tuple of (train_df, val_df) with 0 entity overlap.
    """
    if df.height == 0:
        return df, df

    unique_groups = df.select(group_col).unique().to_series().to_numpy()
    n_groups = len(unique_groups)

    if n_groups <= 1:
        logger.warning(f"Only {n_groups} group found. Cannot perform multi-group split.")
        return df, df.head(0)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_seed)
    
    # We can split directly on the unique groups array or on the full dataframe
    rng = np.random.default_rng(random_seed)
    shuffled_groups = rng.permutation(unique_groups)
    val_size = max(1, int(n_groups * val_ratio))
    val_groups_set = set(shuffled_groups[:val_size])
    train_groups_set = set(shuffled_groups[val_size:])

    train_df = df.filter(pl.col(group_col).is_in(list(train_groups_set)))
    val_df = df.filter(pl.col(group_col).is_in(list(val_groups_set)))

    logger.info(
        f"Group split complete: {n_groups:,} unique entities -> "
        f"Train: {train_df.height:,} rows ({len(train_groups_set):,} entities), "
        f"Val: {val_df.height:,} rows ({len(val_groups_set):,} entities)"
    )
    return train_df, val_df


def get_group_kfold_splits(
    df: pl.DataFrame,
    n_splits: int = 5,
    group_col: str = "source1_entity_id",
) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
    """Generates GroupKFold cross-validation splits by entity group.

    Args:
        df: Polars DataFrame with feature rows and group column.
        n_splits: Number of CV folds.
        group_col: Column name of grouping key.

    Returns:
        List of (train_fold_df, val_fold_df) pairs.
    """
    if df.height == 0:
        return []

    groups = df[group_col].to_numpy()
    n_unique = len(np.unique(groups))
    actual_splits = min(n_splits, n_unique)

    gkf = GroupKFold(n_splits=actual_splits)
    splits = []
    
    indices = np.arange(df.height)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups), start=1):
        train_fold = df[train_idx.tolist()]
        val_fold = df[val_idx.tolist()]
        splits.append((train_fold, val_fold))
        logger.debug(
            f"Fold {fold_idx}/{actual_splits}: Train={train_fold.height:,} rows, Val={val_fold.height:,} rows"
        )

    logger.info(f"Generated {len(splits)} GroupKFold splits across {n_unique:,} entities.")
    return splits
