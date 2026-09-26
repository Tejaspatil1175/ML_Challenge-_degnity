"""LightGBM entity matching classifier training pipeline.

Trains gradient boosting classifier on candidate pair features with class imbalance compensation and feature importance logging.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from backend.src.config import cfg
from backend.src.logger import get_logger
from backend.src.model.persist import save_model
from backend.src.model.split import split_train_val

logger = get_logger(__name__)

EXCLUDE_FEATURE_COLS = {
    "source1_entity_id",
    "candidate_entity_id",
    "is_match",
    "rank",
}


def get_feature_columns(df: pl.DataFrame) -> List[str]:
    """Extracts numeric feature column names excluding IDs and targets."""
    return [col for col in df.columns if col not in EXCLUDE_FEATURE_COLS]


def train_matching_model(
    train_features_df: pl.DataFrame,
    val_ratio: float = 0.20,
    model_version: str = "v1",
    custom_params: Optional[Dict[str, Any]] = None,
) -> Tuple[lgb.LGBMClassifier, List[str], pl.DataFrame, np.ndarray]:
    """Trains a LightGBM classifier on entity candidate features.

    Args:
        train_features_df: Labeled DataFrame containing candidate features and 'is_match' target.
        val_ratio: Proportion of unique entities reserved for validation.
        model_version: Version identifier string for saved artifact.
        custom_params: Optional override parameters for LightGBM.

    Returns:
        Tuple of (trained_model, feature_names, val_df, val_probabilities).
    """
    logger.info("Starting LightGBM entity matching training pipeline...")
    t0 = time.time()

    # Split train/val by entity group
    train_split, val_split = split_train_val(
        train_features_df,
        val_ratio=val_ratio,
        random_seed=cfg.execution.get("random_seed", 42),
    )

    feature_cols = get_feature_columns(train_features_df)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    X_train = train_split.select(feature_cols).to_numpy()
    y_train = train_split["is_match"].to_numpy()

    X_val = val_split.select(feature_cols).to_numpy() if val_split.height > 0 else X_train
    y_val = val_split["is_match"].to_numpy() if val_split.height > 0 else y_train

    # Compute positive weight for class imbalance
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    logger.info(f"Class counts - Positives: {n_pos:,}, Negatives: {n_neg:,} | scale_pos_weight: {scale_pos_weight:.2f}")

    # Set up LightGBM hyperparameters
    lgbm_config = cfg.training.get("lgbm_params", {}).copy()
    if custom_params:
        lgbm_config.update(custom_params)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": lgbm_config.get("learning_rate", 0.05),
        "num_leaves": lgbm_config.get("num_leaves", 63),
        "n_estimators": lgbm_config.get("n_estimators", 400),
        "min_child_samples": lgbm_config.get("min_data_in_leaf", 50),
        "colsample_bytree": lgbm_config.get("feature_fraction", 0.8),
        "subsample": 0.8,
        "scale_pos_weight": min(scale_pos_weight, 50.0),  # cap to avoid over-prediction
        "random_state": cfg.execution.get("random_seed", 42),
        "n_jobs": lgbm_config.get("n_jobs", -1),
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    
    # Train model
    logger.info("Fitting LightGBM model...")
    eval_set = [(X_val, y_val)] if val_split.height > 0 else None
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if val_split.height > 0 else []

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        callbacks=callbacks,
    )

    # Predict validation probabilities
    val_probs = model.predict_proba(X_val)[:, 1]

    if val_split.height > 0 and len(np.unique(y_val)) > 1:
        val_auc = roc_auc_score(y_val, val_probs)
        logger.info(f"Validation ROC-AUC: {val_auc:.4f}")

    # Feature Importance logging
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("Top Feature Importances (split count):")
    for idx in sorted_idx[:10]:
        logger.info(f"  - {feature_cols[idx]}: {importances[idx]}")

    # Save model artifacts
    models_dir = Path(cfg.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = models_dir / f"model_{model_version}.joblib"
    latest_path = models_dir / "latest_model.joblib"

    metadata = {
        "version": model_version,
        "feature_names": feature_cols,
        "hyperparameters": params,
        "train_rows": train_split.height,
        "val_rows": val_split.height,
        "scale_pos_weight": scale_pos_weight,
    }

    save_model(model, versioned_path, metadata=metadata)
    save_model(model, latest_path, metadata=metadata)

    elapsed = time.time() - t0
    logger.info(f"Model training and serialization completed in {elapsed:.2f}s.")
    return model, feature_cols, val_split, val_probs


def predict_pair_probabilities(
    model: lgb.LGBMClassifier,
    feature_df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Computes match probabilities for candidate pairs.

    Args:
        model: Trained LightGBM classifier.
        feature_df: DataFrame with candidate feature columns.
        feature_cols: Optional list of expected feature column names.

    Returns:
        1D numpy array of probabilities (scores in [0.0, 1.0]).
    """
    cols = feature_cols or get_feature_columns(feature_df)
    X = feature_df.select(cols).to_numpy()
    probs = model.predict_proba(X)[:, 1]
    return probs
