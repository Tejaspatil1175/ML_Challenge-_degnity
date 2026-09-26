"""Model persistence and artifact serialization module.

Provides versioned saving and loading of trained LightGBM models with sidecar metadata.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib

from backend.src.config import cfg
from backend.src.exceptions import ModelTrainingError
from backend.src.logger import get_logger

logger = get_logger(__name__)


def save_model(
    model: Any,
    filepath: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves a trained model artifact and its associated metadata.

    Args:
        model: Trained classifier (e.g., LightGBM LGBMClassifier or Booster).
        filepath: Destination path for model binary (e.g. .joblib or .txt).
        metadata: Optional dictionary with feature names, hyperparameters, etc.

    Returns:
        Path to the saved model file.
    """
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model artifact to: {target_path}")
    
    # Save binary via joblib
    joblib.dump(model, target_path)

    # Save sidecar metadata JSON
    meta_path = target_path.with_suffix(".json")
    meta_dict = {
        "model_file": target_path.name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_class": model.__class__.__name__,
        "metadata": metadata or {},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    logger.info(f"Model and metadata successfully saved ({target_path.stat().st_size:,} bytes).")
    return target_path


def load_model(filepath: Path | str) -> Tuple[Any, Dict[str, Any]]:
    """Loads a serialized model and its metadata from disk.

    Args:
        filepath: Path to saved model file.

    Returns:
        Tuple of (loaded_model, metadata_dict).

    Raises:
        MLPipelineError: If file not found or corrupted.
    """
    target_path = Path(filepath)
    if not target_path.exists():
        raise ModelTrainingError(f"Model file not found at: {target_path}")

    try:
        model = joblib.load(target_path)
    except Exception as e:
        raise ModelTrainingError(f"Failed to load model from {target_path}: {e}") from e

    meta_path = target_path.with_suffix(".json")
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata from {meta_path}: {e}")

    logger.info(f"Loaded model ({model.__class__.__name__}) from {target_path}")
    return model, metadata
