"""Central logging utility for Business Entity Resolution.

Provides structured, leveled, timestamped console and file loggers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from backend.src.config import cfg


def get_logger(name: str, log_file: Optional[Path] = None, level: Optional[str] = None) -> logging.Logger:
    """Returns a configured logger instance with formatted console and optional file handlers."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_level_str = (level or cfg.logging.get("level", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    # Standard format: 2026-09-25 21:50:00 [INFO] (src.module) Message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    
    # Optional File Handler
    if cfg.logging.get("log_to_file", True) or log_file is not None:
        target_file = log_file or Path(cfg.paths.reports_dir) / "pipeline.log"
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(target_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not initialize file logger at {target_file}: {e}")
            
    return logger
