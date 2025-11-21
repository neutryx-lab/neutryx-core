"""Tests for CSA contract models."""
from __future__ import annotations

import pytest

from neutryx.portfolio.contracts.csa import (
    CSA,
    CollateralTerms,
    CollateralType,
    DisputeResolution,
    EligibleCollateral,
    ThresholdTerms,
    ValuationFrequency,
)


def test_threshold_terms_defaults():
    """Test ThresholdTerms with default values."""
    terms = ThresholdTerms()
    assert terms.threshold_party_a == 0.0
    assert terms.threshold_party_b == 0.0
    assert terms.mta_party_a == 0.0
    assert terms.mta_party_b == 0.0
    assert terms.rounding == 100_000.0


def test_threshold_terms_bilateral():
    """Test ThresholdTerms with bilateral values."""
    terms = ThresholdTerms(
        threshold_party_a=1_000_000.0,
        threshold_party_b=5_000_000.0,
        mta_party_a=100_000.0,
        mta_party_b=250_000.0,
        independent_amount_party_a=500_000.0,
        independent_amount_party_b=0.0,
        rounding=50_000.0,
    )
    assert terms.get_threshold("A") == 1_000_000.0
    assert terms.get_threshold("B") == 5_000_000.0
    assert terms.get_mta("A") == 100_000.0
    assert terms.get_mta("B") == 250_000.0
    assert terms.get_independent_amount("A") == 500_000.0
    assert terms.get_independent_amount("B") == 0.0


def test_threshold_terms_invalid_party():
    """Test ThresholdTerms with invalid party label."""
    terms = ThresholdTerms()
    with pytest.raises(ValueError, match="Party must be"):
        terms.get_threshold("C")
    with pytest.raises(ValueError, match="Party must be"):
        terms.get_mta("X")


def test_eligible_collateral_cash():
    """Test EligibleCollateral for cash."""
    collateral = EligibleCollateral(
        collateral_type=CollateralType.CASH,
        currency="USD",
        haircut=0.0,
    )
    assert collateral.collateral_type == CollateralType.CASH
    assert collateral.currency == "USD"
    assert collateral.haircut == 0.0


def test_eligible_collateral_government_bond():
    """Test EligibleCollateral for government bonds."""
    collateral = EligibleCollateral(
        collateral_type=CollateralType.GOVERNMENT_BOND,
        currency="USD",
        haircut=0.02,  # 2% haircut
        rating_threshold="AA-",
        maturity_max_years=10.0,
        concentration_limit=0.5,  # Max 50%
    )
    assert collateral.haircut == 0.02
    assert collateral.rating_threshold == "AA-"
    assert collateral.maturity_max_years == 10.0
    assert collateral.concentration_limit == 0.5


def test_eligible_collateral_apply_haircut():
    """Test haircut application."""
    collateral = EligibleCollateral(
        collateral_type=CollateralType.CORPORATE_BOND,
        haircut=0.05,  # 5% haircut
    )
    market_value = 1_000_000.0
    collateral_value = collateral.apply_haircut(market_value)
    assert collateral_value == 950_000.0  # 1M * (1 - 0.05)


def test_collateral_terms_basic():
    """Test CollateralTerms basic setup."""
    terms = CollateralTerms(
        base_currency="USD",
        valuation_frequency=ValuationFrequency.DAILY,
        eligible_collateral=[
            EligibleCollateral(
                collateral_type=CollateralType.CASH,
                currency="USD",
                haircut=0.0,
            ),
        ],
    )
    assert terms.base_currency == "USD"
    assert terms.valuation_frequency == ValuationFrequency.DAILY
    assert len(terms.eligible_collateral) == 1


def test_collateral_terms_is_eligible():
    """Test is_eligible method."""
    terms = CollateralTerms(
        base_currency="EUR",
        eligible_collateral=[
            EligibleCollateral(
                collateral_type=CollateralType.CASH,
                currency="EUR",
                haircut=0.0,
            ),
            EligibleCollateral(
                collateral_type=CollateralType.GOVERNMENT_BOND,
                haircut=0.02,
            ),
        ],
    )
    assert terms.is_eligible(CollateralType.CASH)
    assert terms.is_eligible(CollateralType.GOVERNMENT_BOND)
    assert not terms.is_eligible(CollateralType.EQUITY)


def test_collateral_terms_get_collateral_spec():
    """Test get_collateral_spec method."""
    cash_spec = EligibleCollateral(
        collateral_type=CollateralType.CASH,
        currency="USD",
        haircut=0.0,
    )
    terms = CollateralTerms(
        base_currency="USD",
        eligible_collateral=[cash_spec],
    )
    spec = terms.get_collateral_spec(CollateralType.CASH)
    assert spec is not None
    assert spec.currency == "USD"

    spec_none = terms.get_collateral_spec(CollateralType.EQUITY)
    assert spec_none is None


def test_csa_basic():
    """Test basic CSA creation."""
    csa = CSA(
        id="CSA001",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        collateral_terms=CollateralTerms(
            base_currency="USD",
            eligible_collateral=[
                EligibleCollateral(
                    collateral_type=CollateralType.CASH,
                    currency="USD",
                    haircut=0.0,
                ),
            ],
        ),
    )
    assert csa.id == "CSA001"
    assert csa.party_a_id == "CP001"
    assert csa.party_b_id == "CP002"
    assert csa.variation_margin_required
    assert not csa.initial_margin_required


def test_csa_is_party():
    """Test is_party method."""
    csa = CSA(
        id="CSA002",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        collateral_terms=CollateralTerms(base_currency="USD"),
    )
    assert csa.is_party("CP001")
    assert csa.is_party("CP002")
    assert not csa.is_party("CP003")


def test_csa_get_other_party():
    """Test get_other_party method."""
    csa = CSA(
        id="CSA003",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        collateral_terms=CollateralTerms(base_currency="USD"),
    )
    assert csa.get_other_party("CP001") == "CP002"
    assert csa.get_other_party("CP002") == "CP001"

    with pytest.raises(ValueError, match="not a party"):
        csa.get_other_party("CP003")


def test_csa_get_party_label():
    """Test get_party_label method."""
    csa = CSA(
        id="CSA004",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        collateral_terms=CollateralTerms(base_currency="USD"),
    )
    assert csa.get_party_label("CP001") == "A"
    assert csa.get_party_label("CP002") == "B"

    with pytest.raises(ValueError, match="not a party"):
        csa.get_party_label("CP003")


def test_csa_calculate_collateral_requirement():
    """Test calculate_collateral_requirement method."""
    csa = CSA(
        id="CSA005",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        threshold_terms=ThresholdTerms(
            threshold_party_a=1_000_000.0,
            threshold_party_b=500_000.0,
            independent_amount_party_a=100_000.0,
        ),
        collateral_terms=CollateralTerms(base_currency="USD"),
    )

    # Exposure of 2M for Party A (exceeds 1M threshold)
    requirement = csa.calculate_collateral_requirement(
        exposure=2_000_000.0,
        posted_by="CP001",
    )
    # (2M - 1M threshold) + 100K IA = 1.1M
    assert requirement == 1_100_000.0

    # Exposure below threshold
    requirement_low = csa.calculate_collateral_requirement(
        exposure=500_000.0,
        posted_by="CP001",
    )
    # Below threshold, only IA required
    assert requirement_low == 100_000.0


def test_csa_apply_rounding():
    """Test apply_rounding method."""
    csa = CSA(
        id="CSA006",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        threshold_terms=ThresholdTerms(rounding=100_000.0),
        collateral_terms=CollateralTerms(base_currency="USD"),
    )

    # Should round to nearest 100K
    assert csa.apply_rounding(1_234_567.0) == 1_200_000.0
    assert csa.apply_rounding(1_567_890.0) == 1_600_000.0
    assert csa.apply_rounding(50_000.0) == 0.0  # Rounds down to 0
    assert csa.apply_rounding(75_000.0) == 100_000.0  # Rounds up to 100K


def test_csa_apply_mta():
    """Test apply_mta method."""
    csa = CSA(
        id="CSA007",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        threshold_terms=ThresholdTerms(mta_party_a=250_000.0),
        collateral_terms=CollateralTerms(base_currency="USD"),
    )

    # Transfer above MTA
    transfer = csa.apply_mta(
        current_collateral=1_000_000.0,
        required_collateral=1_500_000.0,
        posted_by="CP001",
    )
    assert transfer == 500_000.0

    # Transfer below MTA (should be 0)
    transfer_small = csa.apply_mta(
        current_collateral=1_000_000.0,
        required_collateral=1_100_000.0,
        posted_by="CP001",
    )
    assert transfer_small == 0.0


def test_csa_with_initial_margin():
    """Test CSA with initial margin requirement."""
    csa = CSA(
        id="CSA008",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        collateral_terms=CollateralTerms(base_currency="USD"),
        initial_margin_required=True,
        variation_margin_required=True,
    )
    assert csa.initial_margin_required
    assert csa.variation_margin_required


def test_csa_segregation_and_rehypothecation():
    """Test CSA segregation and rehypothecation settings."""
    csa = CSA(
        id="CSA009",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        collateral_terms=CollateralTerms(base_currency="USD"),
        segregation_required=True,
        rehypothecation_allowed=False,
    )
    assert csa.segregation_required
    assert not csa.rehypothecation_allowed


def test_csa_repr():
    """Test string representation."""
    csa = CSA(
        id="CSA010",
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date="2024-01-01",
        collateral_terms=CollateralTerms(base_currency="USD"),
        initial_margin_required=True,
    )
    repr_str = repr(csa)
    assert "CSA010" in repr_str
    assert "CP001" in repr_str
    assert "CP002" in repr_str
    assert "IM=True" in repr_str


def test_valuation_frequency_enum():
    """Test ValuationFrequency enum."""
    assert ValuationFrequency.DAILY.value == "Daily"
    assert ValuationFrequency.WEEKLY.value == "Weekly"
    assert ValuationFrequency.MONTHLY.value == "Monthly"


def test_collateral_type_enum():
    """Test CollateralType enum."""
    assert CollateralType.CASH.value == "Cash"
    assert CollateralType.GOVERNMENT_BOND.value == "GovernmentBond"
    assert CollateralType.EQUITY.value == "Equity"
