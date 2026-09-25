"""Configuration management module for Business Entity Resolution.

Loads parameters from settings.yaml and provides strongly-typed path and hyperparameter access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


def find_workspace_root() -> Path:
    """Finds workspace root directory containing .git or backend folder."""
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        if (curr / ".git").exists() or (curr / "backend").exists():
            return curr
        if curr.parent == curr:
            break
        curr = curr.parent
    return Path(__file__).resolve().parent.parent.parent


WORKSPACE_ROOT = find_workspace_root()
BACKEND_ROOT = WORKSPACE_ROOT / "backend"
CONFIG_FILE_PATH = BACKEND_ROOT / "config" / "settings.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        if yaml is not None:
            return yaml.safe_load(f) or {}
        
        # Simple fallback parser if yaml module is not yet installed
        lines = f.readlines()
        data: Dict[str, Any] = {}
        curr_key = None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not v:
                    curr_key = k
                    data[curr_key] = {}
                else:
                    if curr_key and line.startswith("  "):
                        data[curr_key][k] = v
                    else:
                        data[k] = v
        return data


@dataclass
class PathConfig:
    workspace_root: Path = WORKSPACE_ROOT
    backend_root: Path = BACKEND_ROOT
    raw_dataset_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset")
    train_s1: Path = field(init=False)
    train_s2: Path = field(init=False)
    train_s3: Path = field(init=False)
    train_gt: Path = field(init=False)
    test_s1: Path = field(init=False)
    test_s2: Path = field(init=False)
    test_s3: Path = field(init=False)
    
    processed_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "data_cache" / "processed")
    embeddings_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "data_cache" / "embeddings")
    faiss_index_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "data_cache" / "faiss")
    models_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "models")
    output_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "output")
    reports_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "reports")
    candidate_pairs_tsv: Path = field(init=False)
    matching_results_tsv: Path = field(init=False)

    def __post_init__(self):
        # Auto-discover raw dataset directory if default path doesn't exist
        if not self.raw_dataset_dir.exists():
            candidates = [
                self.workspace_root / "dataset",
                self.workspace_root / "data",
                self.workspace_root / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset",
            ]
            for cand in candidates:
                if cand.exists():
                    self.raw_dataset_dir = cand
                    break

        self.train_s1 = self.raw_dataset_dir / "train" / "train_source1.tsv"
        self.train_s2 = self.raw_dataset_dir / "train" / "train_source2.tsv"
        self.train_s3 = self.raw_dataset_dir / "train" / "train_source3.tsv"
        self.train_gt = self.raw_dataset_dir / "train" / "train_ground_truth.tsv"
        self.test_s1 = self.raw_dataset_dir / "test" / "test_source1.tsv"
        self.test_s2 = self.raw_dataset_dir / "test" / "test_source2.tsv"
        self.test_s3 = self.raw_dataset_dir / "test" / "test_source3.tsv"
        
        self.candidate_pairs_tsv = self.output_dir / "candidate_pairs.tsv"
        self.matching_results_tsv = self.output_dir / "matching_results.tsv"

    def ensure_directories(self) -> None:
        """Creates necessary cache, model, output and report directories."""
        for d in [self.processed_dir, self.embeddings_dir, self.faiss_index_dir, 
                 self.models_dir, self.output_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)


class Config:
    """Singleton Configuration instance."""
    _instance: Optional[Config] = None

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_FILE_PATH
        self.raw_dict = _load_yaml(self.config_path) if self.config_path.exists() else {}
        self.paths = PathConfig()
        self.execution = self.raw_dict.get("execution", {})
        self.normalization = self.raw_dict.get("normalization", {})
        self.blocking = self.raw_dict.get("blocking", {})
        self.features = self.raw_dict.get("features", {})
        self.training = self.raw_dict.get("training", {})
        self.evaluation = self.raw_dict.get("evaluation", {})
        self.logging = self.raw_dict.get("logging", {})

    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> Config:
        if cls._instance is None or config_path is not None:
            cls._instance = Config(config_path)
        return cls._instance


# Global configuration handle
cfg = Config.get_instance()
