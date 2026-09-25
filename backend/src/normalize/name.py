"""Business entity name normalization engine.

Cleans, transliterates, and canonicalizes commercial entity names using regex rules and suffix dictionaries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import unidecode
import yaml

from backend.src.config import cfg
from backend.src.logger import get_logger

logger = get_logger(__name__)

# Compile regex substitution rules once
_DEFAULT_SUFFIX_PATH = Path(__file__).parent / "suffix_dict.yaml"


def _load_suffix_patterns(path: Optional[Path] = None) -> List[Tuple[re.Pattern, str]]:
    """Loads and compiles regex replacement rules from YAML dictionary."""
    dict_path = path or _DEFAULT_SUFFIX_PATH
    patterns = []
    if dict_path.exists():
        with open(dict_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            replacements: Dict[str, str] = data.get("replacements", {})
            for pat_str, repl in replacements.items():
                compiled = re.compile(pat_str, re.IGNORECASE)
                patterns.append((compiled, repl))
    return patterns


_COMPILED_PATTERNS = _load_suffix_patterns()
_PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_REGEX = re.compile(r"\s+")


def normalize_name(name: Optional[str]) -> str:
    """Normalizes a business name to a canonical comparable format.

    Steps:
        1. Transliterate Unicode glyphs to ASCII via unidecode.
        2. Lowercase.
        3. Expand ampersands and legal suffixes (e.g. Corp -> corporation, Pvt -> private).
        4. Strip punctuation and non-alphanumeric symbols.
        5. Normalize whitespace.

    Args:
        name: Raw business name string.

    Returns:
        Canonical normalized name string.
    """
    if not name or not isinstance(name, str):
        return ""

    # 1. Transliteration (handles accents, non-ASCII chars)
    text = unidecode.unidecode(name)

    # 2. Lowercase
    text = text.lower()

    # 3. Apply compiled suffix and abbreviation expansion rules
    for pattern, repl in _COMPILED_PATTERNS:
        text = pattern.sub(f" {repl} ", text)

    # 4. Remove punctuation
    text = _PUNCT_REGEX.sub(" ", text)

    # 5. Normalize whitespace
    text = _MULTI_SPACE_REGEX.sub(" ", text).strip()

    return text
