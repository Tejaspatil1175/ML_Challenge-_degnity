"""Unit tests for key-based blocking functions."""

from backend.src.blocking.keys import (
    generate_name_key,
    generate_phonetic_key,
    generate_token_set_key,
)


def test_name_prefix_key_generation():
    """Test prefix key generation for matching name variants."""
    k1 = generate_name_key("walmart supercenter", "bentonville")
    k2 = generate_name_key("walmart inc", "bentonville")
    assert k1.startswith("walm")
    assert k2.startswith("walm")
    assert k1 == k2


def test_phonetic_soundex_key_generation():
    """Test Soundex phonetic key for slight spelling/homophone variations."""
    # Smith vs Smyth
    k1 = generate_phonetic_key("smith enterprises", "US")
    k2 = generate_phonetic_key("smyth enterprises", "US")
    assert k1 == k2
    assert k1 == "S530_us"


def test_token_set_key_order_invariance():
    """Test token set key produces identical result regardless of word order."""
    k1 = generate_token_set_key("prime money")
    k2 = generate_token_set_key("money prime")
    assert k1 == k2
    assert k1 == "money_prime"


def test_empty_string_keys():
    """Test graceful handling of empty or None inputs."""
    assert generate_name_key("", "") == ""
    assert generate_phonetic_key(None) == ""
    assert generate_token_set_key("") == ""
