"""Address normalization and component extraction engine.

Handles international street abbreviation expansions, postal code parsing, and address decomposition.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import unidecode

from backend.src.logger import get_logger

logger = get_logger(__name__)

# Common address token expansions
_ADDRESS_EXPANSIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\brd\.?\b", re.I), "road"),
    (re.compile(r"\bst\.?\b", re.I), "street"),
    (re.compile(r"\bave\.?\b", re.I), "avenue"),
    (re.compile(r"\bav\.?\b", re.I), "avenue"),
    (re.compile(r"\bblvd\.?\b", re.I), "boulevard"),
    (re.compile(r"\bdr\.?\b", re.I), "drive"),
    (re.compile(r"\bln\.?\b", re.I), "lane"),
    (re.compile(r"\bct\.?\b", re.I), "court"),
    (re.compile(r"\bpl\.?\b", re.I), "place"),
    (re.compile(r"\bpkwy\.?\b", re.I), "parkway"),
    (re.compile(r"\bhwy\.?\b", re.I), "highway"),
    (re.compile(r"\bste\.?\b", re.I), "suite"),
    (re.compile(r"\bapt\.?\b", re.I), "apartment"),
    (re.compile(r"\bfl\.?\b", re.I), "floor"),
    (re.compile(r"\bbldg\.?\b", re.I), "building"),
    (re.compile(r"\bp\.?o\.?\s*box\b", re.I), "pobox"),
    (re.compile(r"\bdept\.?\b", re.I), "department"),
    (re.compile(r"\bno\.?\s*(\d+)", re.I), r"\1"),
    (re.compile(r"\b#\s*(\d+)", re.I), r"\1"),
]

_PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_REGEX = re.compile(r"\s+")

# Postal / PIN code regex extractors
_US_ZIP_REGEX = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_IN_PIN_REGEX = re.compile(r"\b\d{6}\b")
_FR_POSTAL_REGEX = re.compile(r"\b\d{5}\b")
_GENERIC_POSTAL_REGEX = re.compile(r"\b[A-Z0-9]{3,10}\b", re.I)
_STREET_NUM_REGEX = re.compile(r"^\s*(\d+[A-Za-z]?)\b")


def normalize_address(address: Optional[str]) -> str:
    """Normalizes raw address text into a standardized string representation.

    Args:
        address: Raw unstructured address string.

    Returns:
        Cleaned, lowercased, and expanded address string.
    """
    if not address or not isinstance(address, str):
        return ""

    # 1. Transliterate Unicode glyphs
    text = unidecode.unidecode(address)

    # 2. Lowercase
    text = text.lower()

    # 3. Apply abbreviation expansions
    for pattern, repl in _ADDRESS_EXPANSIONS:
        text = pattern.sub(f" {repl} ", text)

    # 4. Remove punctuation
    text = _PUNCT_REGEX.sub(" ", text)

    # 5. Clean whitespace
    text = _MULTI_SPACE_REGEX.sub(" ", text).strip()

    return text


def extract_address_parts(
    address: Optional[str],
    country: Optional[str] = None,
    clean_address: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Extracts structured address components (street number, city, state, postal code).

    Designed to be robust, fail-safe, and high performance: eliminates redundant regex passes.

    Args:
        address: Raw unstructured address string.
        country: Optional country ISO code or name.
        clean_address: Optional pre-normalized address string to avoid redundant normalization.

    Returns:
        Dictionary with keys: 'street_number', 'street_name', 'city', 'state', 'zipcode'.
    """
    result: Dict[str, Optional[str]] = {
        "street_number": None,
        "street_name": None,
        "city": None,
        "state": None,
        "zipcode": None,
    }

    if not address or not isinstance(address, str):
        return result

    c_addr = clean_address if clean_address is not None else normalize_address(address)
    if not c_addr:
        return result

    # 1. Extract street number if starting with digits
    num_match = _STREET_NUM_REGEX.search(c_addr)
    if num_match:
        result["street_number"] = num_match.group(1)

    # 2. Country-specific postal code extraction
    country_upper = (country or "").strip().upper()
    if country_upper in ("US", "USA", "UNITED STATES"):
        zip_match = _US_ZIP_REGEX.search(address)
        if zip_match:
            result["zipcode"] = zip_match.group(0)[:5]
    elif country_upper in ("IN", "INDIA"):
        pin_match = _IN_PIN_REGEX.search(address)
        if pin_match:
            result["zipcode"] = pin_match.group(0)
    elif country_upper in ("FR", "FRANCE"):
        fr_match = _FR_POSTAL_REGEX.search(address)
        if fr_match:
            result["zipcode"] = fr_match.group(0)
    else:
        # Generic fallback: search for 5-6 digit sequences or postal tokens
        pin_match = _IN_PIN_REGEX.search(address) or _US_ZIP_REGEX.search(address)
        if pin_match:
            result["zipcode"] = pin_match.group(0)

    # 3. Fast city/state decomposition using commas from raw address without redundant regex calls
    parts = [p.strip().lower() for p in address.split(",") if p.strip()]
    if len(parts) >= 3:
        result["street_name"] = parts[0]
        result["city"] = parts[1]
        result["state"] = parts[2]
    elif len(parts) == 2:
        result["street_name"] = parts[0]
        result["city"] = parts[1]

    return result
