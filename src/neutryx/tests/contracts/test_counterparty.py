"""Tests for counterparty models."""
from __future__ import annotations

import pytest

from neutryx.contracts.counterparty import (
    Counterparty,
    CounterpartyCredit,
    CreditRating,
    EntityType,
)
from neutryx.market.credit.hazard import HazardRateCurve


def test_counterparty_credit_default_lgd():
    """Test CounterpartyCredit with default LGD."""
    credit = CounterpartyCredit()
    assert credit.lgd == 0.6
    assert credit.get_lgd() == 0.6
    assert credit.get_recovery_rate() == 0.4


def test_counterparty_credit_with_recovery_rate():
    """Test CounterpartyCredit with explicit recovery rate."""
    credit = CounterpartyCredit(recovery_rate=0.5)
    assert credit.get_recovery_rate() == 0.5
    assert credit.get_lgd() == 0.5


def test_counterparty_credit_with_rating():
    """Test CounterpartyCredit with credit rating."""
    credit = CounterpartyCredit(
        rating=CreditRating.BBB,
        lgd=0.55,
        credit_spread_bps=120.0,
    )
    assert credit.rating == CreditRating.BBB
    assert credit.get_lgd() == 0.55
    assert credit.credit_spread_bps == 120.0


def test_counterparty_credit_with_hazard_curve():
    """Test CounterpartyCredit with hazard rate curve."""
    hazard_curve = HazardRateCurve(
        maturities=[1.0, 2.0, 5.0],
        intensities=[0.01, 0.015, 0.02],
    )
    credit = CounterpartyCredit(
        rating=CreditRating.A,
        hazard_curve=hazard_curve,
    )
    assert credit.hazard_curve is not None
    assert len(credit.hazard_curve.maturities) == 3


def test_counterparty_basic():
    """Test basic Counterparty creation."""
    cp = Counterparty(
        id="CP001",
        name="Bank ABC",
        entity_type=EntityType.FINANCIAL,
    )
    assert cp.id == "CP001"
    assert cp.name == "Bank ABC"
    assert cp.entity_type == EntityType.FINANCIAL
    assert not cp.is_bank
    assert not cp.is_clearinghouse


def test_counterparty_with_lei():
    """Test Counterparty with LEI."""
    cp = Counterparty(
        id="CP002",
        name="Corporate XYZ",
        entity_type=EntityType.CORPORATE,
        lei="549300ABCDEFGHIJK123",
        jurisdiction="US",
    )
    assert cp.lei == "549300ABCDEFGHIJK123"
    assert cp.jurisdiction == "US"


def test_counterparty_with_credit():
    """Test Counterparty with credit attributes."""
    credit = CounterpartyCredit(
        rating=CreditRating.AA,
        lgd=0.5,
        credit_spread_bps=50.0,
    )
    cp = Counterparty(
        id="CP003",
        name="Bank DEF",
        entity_type=EntityType.FINANCIAL,
        credit=credit,
        is_bank=True,
    )
    assert cp.credit is not None
    assert cp.credit.rating == CreditRating.AA
    assert cp.is_bank
    assert cp.get_lgd() == 0.5


def test_counterparty_has_credit_curve():
    """Test has_credit_curve method."""
    # Without credit
    cp1 = Counterparty(
        id="CP004",
        name="Test",
        entity_type=EntityType.CORPORATE,
    )
    assert not cp1.has_credit_curve()

    # With credit but no curve
    cp2 = Counterparty(
        id="CP005",
        name="Test 2",
        entity_type=EntityType.CORPORATE,
        credit=CounterpartyCredit(),
    )
    assert not cp2.has_credit_curve()

    # With credit and curve
    hazard_curve = HazardRateCurve(
        maturities=[1.0, 5.0],
        intensities=[0.01, 0.02],
    )
    cp3 = Counterparty(
        id="CP006",
        name="Test 3",
        entity_type=EntityType.CORPORATE,
        credit=CounterpartyCredit(hazard_curve=hazard_curve),
    )
    assert cp3.has_credit_curve()


def test_counterparty_get_lgd_no_credit():
    """Test get_lgd with no credit info."""
    cp = Counterparty(
        id="CP007",
        name="Test",
        entity_type=EntityType.CORPORATE,
    )
    # Should return default 0.6
    assert cp.get_lgd() == 0.6


def test_counterparty_clearinghouse():
    """Test clearinghouse counterparty."""
    cp = Counterparty(
        id="CCP001",
        name="LCH Clearnet",
        entity_type=EntityType.FINANCIAL,
        is_clearinghouse=True,
    )
    assert cp.is_clearinghouse


def test_counterparty_repr():
    """Test string representation."""
    cp = Counterparty(
        id="CP008",
        name="Test Bank",
        entity_type=EntityType.FINANCIAL,
        credit=CounterpartyCredit(rating=CreditRating.A_PLUS),
    )
    repr_str = repr(cp)
    assert "CP008" in repr_str
    assert "Test Bank" in repr_str
    assert "A+" in repr_str


def test_counterparty_lei_validation():
    """Test LEI length validation."""
    # Valid 20-character LEI
    cp = Counterparty(
        id="CP009",
        name="Test",
        entity_type=EntityType.CORPORATE,
        lei="12345678901234567890",
    )
    assert len(cp.lei) == 20

    # Invalid LEI length should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        Counterparty(
            id="CP010",
            name="Test",
            entity_type=EntityType.CORPORATE,
            lei="short",
        )


def test_credit_rating_enum():
    """Test CreditRating enum values."""
    assert CreditRating.AAA.value == "AAA"
    assert CreditRating.BBB_MINUS.value == "BBB-"
    assert CreditRating.D.value == "D"
    assert CreditRating.NR.value == "NR"


def test_entity_type_enum():
    """Test EntityType enum values."""
    assert EntityType.CORPORATE.value == "Corporate"
    assert EntityType.FINANCIAL.value == "Financial"
    assert EntityType.SOVEREIGN.value == "Sovereign"
    assert EntityType.SPV.value == "SPV"
