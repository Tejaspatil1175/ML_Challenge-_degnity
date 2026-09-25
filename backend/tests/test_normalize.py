"""Unit tests for text normalization and address extraction."""

import pytest

from backend.src.normalize.name import normalize_name
from backend.src.normalize.address import normalize_address, extract_address_parts


@pytest.mark.parametrize(
    "raw_name, expected",
    [
        ("ABC Corp.", "abc corporation"),
        ("M/S Sharma Pvt. Ltd.", "sharma private limited"),
        ("Tech & Co.", "tech and company"),
        ("Global Industries Inc", "global industries incorporated"),
        ("Alpha LLC #101", "alpha llc 101"),
        ("Café Enterprises Ltd.", "cafe enterprises limited"),
        ("B+ Retail Inc", "b and retail incorporated"),
        ("Orelee's Barbershop", "orelee s barbershop"),
        ("Precision Engineering GMBH", "precision engineering gmbh"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_name(raw_name, expected):
    """Test 8+ before/after test cases for business name canonicalization."""
    assert normalize_name(raw_name) == expected


@pytest.mark.parametrize(
    "raw_addr, expected",
    [
        ("1795 Westchester Dr., High Point, NC", "1795 westchester drive high point nc"),
        ("Plot No. 42, M.G. Rd, Bengaluru 560001", "plot 42 m g road bengaluru 560001"),
        ("123 Main St, Ste 400, New York, NY 10001", "123 main street suite 400 new york ny 10001"),
        ("8 Rue de la Paix, Paris, 75002", "8 rue de la paix paris 75002"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_address(raw_addr, expected):
    """Test address normalization with abbreviation expansion."""
    assert normalize_address(raw_addr) == expected


def test_extract_address_parts_us():
    """Test address component extraction for US."""
    parts = extract_address_parts("1795 Westchester Drive, High Point, NC 27262", country="US")
    assert parts["street_number"] == "1795"
    assert parts["city"] == "high point"
    assert parts["zipcode"] == "27262"


def test_extract_address_parts_india():
    """Test address component extraction for India."""
    parts = extract_address_parts("42 MG Road, Bengaluru, Karnataka 560001", country="India")
    assert parts["street_number"] == "42"
    assert parts["zipcode"] == "560001"


def test_open_set_country_france():
    """Test that open-set country (e.g. France) does not fail or crash."""
    parts = extract_address_parts("10 Avenue des Champs-Elysees, Paris, 75008", country="France")
    assert parts["street_number"] == "10"
    assert parts["zipcode"] == "75008"
    assert parts["city"] == "paris"


def test_open_set_country_unknown():
    """Test arbitrary novel country handles gracefully without crashing."""
    parts = extract_address_parts("100 Queen Street, Auckland, 1010", country="New Zealand")
    assert parts["street_number"] == "100"
    assert isinstance(parts, dict)
