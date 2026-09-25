"""Unit tests for pairwise feature extraction."""

import math
from backend.src.features.pair_features import (
    compute_word_jaccard,
    compute_char_ngram_jaccard,
    extract_pairwise_feature_vector,
)


def test_similar_pair_feature_vector():
    """Test feature extraction on a known near-duplicate pair."""
    feats = extract_pairwise_feature_vector(
        s1_name="walmart supercenter",
        cand_name="walmart supercenter inc",
        s1_addr="123 main street bentonville ar 72712",
        cand_addr="123 main st suite 100 bentonville ar 72712",
        s1_country="US",
        cand_country="US",
        s1_zip="72712",
        cand_zip="72712",
        s1_city="bentonville",
        cand_city="bentonville",
    )
    assert feats["name_ratio"] > 0.80
    assert feats["name_jw"] > 0.90
    assert feats["country_match"] == 1.0
    assert feats["zip_match"] == 1.0
    assert feats["city_match"] == 1.0
    # Verify no NaNs
    for k, v in feats.items():
        assert not math.isnan(v), f"Feature {k} is NaN"


def test_dissimilar_pair_feature_vector():
    """Test feature extraction on completely unrelated entities."""
    feats = extract_pairwise_feature_vector(
        s1_name="sharma medical store",
        cand_name="johnсон bakery llc",
        s1_addr="plot 42 mg road bengaluru",
        cand_addr="789 oak avenue chicago il",
        s1_country="India",
        cand_country="US",
        s1_zip="560001",
        cand_zip="60601",
        s1_city="bengaluru",
        cand_city="chicago",
    )
    assert feats["name_ratio"] < 0.35
    assert feats["country_match"] == 0.0
    assert feats["zip_match"] == 0.0


def test_jaccard_metrics():
    """Test token and n-gram Jaccard calculations."""
    j_word = compute_word_jaccard("prime money limited", "prime money private limited")
    assert j_word == 0.75  # 3 intersection out of 4 union

    j_char = compute_char_ngram_jaccard("apple", "apple", 3)
    assert j_char == 1.0
