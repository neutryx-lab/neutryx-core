"""Tests for Convention Profile System"""

import pytest
from neutryx.market.convention_profiles import (
    ConventionProfile,
    ConventionProfileRegistry,
    ProductTypeConvention,
    LegConvention,
    get_convention_profile,
    get_convention_registry,
)
from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import ACT_360, ACT_365, THIRTY_360
from neutryx.core.dates.business_day import MODIFIED_FOLLOWING


class TestConventionProfile:
    """Test ConventionProfile class"""

    def test_profile_creation(self):
        """Test creating a convention profile"""
        profile = ConventionProfile(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            calendars=["USD"],
            spot_lag=2,
        )

        assert profile.currency == "USD"
        assert profile.product_type == ProductTypeConvention.INTEREST_RATE_SWAP
        assert profile.get_profile_id() == "USD_IRS"

    def test_profile_to_dict(self):
        """Test converting profile to dictionary"""
        profile = ConventionProfile(
            currency="EUR",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            calendars=["TARGET"],
            spot_lag=2,
            description="EUR OIS",
        )

        profile_dict = profile.to_dict()
        assert profile_dict["currency"] == "EUR"
        assert profile_dict["product_type"] == "OIS"
        assert profile_dict["spot_lag"] == 2
        assert "fixed_leg" in profile_dict


class TestConventionProfileRegistry:
    """Test ConventionProfileRegistry class"""

    def test_registry_initialization(self):
        """Test that registry initializes with standard profiles"""
        registry = ConventionProfileRegistry()

        # Check that major currencies are registered
        assert registry.has_profile("USD", ProductTypeConvention.INTEREST_RATE_SWAP)
        assert registry.has_profile("EUR", ProductTypeConvention.INTEREST_RATE_SWAP)
        assert registry.has_profile("GBP", ProductTypeConvention.INTEREST_RATE_SWAP)
        assert registry.has_profile("JPY", ProductTypeConvention.INTEREST_RATE_SWAP)

    def test_get_usd_irs_profile(self):
        """Test retrieving USD IRS profile"""
        registry = ConventionProfileRegistry()
        profile = registry.get_profile("USD", ProductTypeConvention.INTEREST_RATE_SWAP)

        assert profile is not None
        assert profile.currency == "USD"
        assert profile.fixed_leg.frequency == Frequency.SEMI_ANNUAL
        assert str(profile.fixed_leg.day_count) == "Thirty360"
        assert profile.floating_leg.frequency == Frequency.QUARTERLY
        assert str(profile.floating_leg.day_count) == "Actual360"

    def test_get_eur_ois_profile(self):
        """Test retrieving EUR OIS profile"""
        registry = ConventionProfileRegistry()
        profile = registry.get_profile("EUR", ProductTypeConvention.OVERNIGHT_INDEX_SWAP)

        assert profile is not None
        assert profile.currency == "EUR"
        assert profile.fixed_leg.frequency == Frequency.ANNUAL
        assert str(profile.fixed_leg.day_count) == "Actual360"
        assert profile.floating_leg.rate_index.name == "ESTR"

    def test_get_gbp_irs_profile(self):
        """Test retrieving GBP IRS profile"""
        registry = ConventionProfileRegistry()
        profile = registry.get_profile("GBP", ProductTypeConvention.INTEREST_RATE_SWAP)

        assert profile is not None
        assert profile.currency == "GBP"
        assert profile.spot_lag == 0  # GBP typically T+0
        assert str(profile.fixed_leg.day_count) == "Actual365Fixed"

    def test_list_currencies(self):
        """Test listing all currencies"""
        registry = ConventionProfileRegistry()
        currencies = registry.list_currencies()

        assert "USD" in currencies
        assert "EUR" in currencies
        assert "GBP" in currencies
        assert "JPY" in currencies
        assert "CHF" in currencies

    def test_list_product_types(self):
        """Test listing product types"""
        registry = ConventionProfileRegistry()
        product_types = registry.list_product_types("USD")

        assert ProductTypeConvention.INTEREST_RATE_SWAP in product_types
        assert ProductTypeConvention.OVERNIGHT_INDEX_SWAP in product_types
        assert ProductTypeConvention.FORWARD_RATE_AGREEMENT in product_types

    def test_register_custom_profile(self):
        """Test registering a custom profile"""
        registry = ConventionProfileRegistry()

        custom_profile = ConventionProfile(
            currency="CAD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            calendars=["CAD"],
            spot_lag=2,
            description="CAD IRS",
        )

        registry.register_profile(custom_profile)

        # Retrieve and verify
        retrieved = registry.get_profile("CAD", ProductTypeConvention.INTEREST_RATE_SWAP)
        assert retrieved is not None
        assert retrieved.currency == "CAD"

    def test_profile_not_found(self):
        """Test retrieving non-existent profile"""
        registry = ConventionProfileRegistry()
        profile = registry.get_profile("ZZZ", ProductTypeConvention.INTEREST_RATE_SWAP)

        assert profile is None


class TestGlobalRegistry:
    """Test global registry singleton"""

    def test_get_global_registry(self):
        """Test getting global registry"""
        registry1 = get_convention_registry()
        registry2 = get_convention_registry()

        # Should be the same instance (singleton)
        assert registry1 is registry2

    def test_convenience_function(self):
        """Test convenience function for getting profiles"""
        profile = get_convention_profile("USD", ProductTypeConvention.INTEREST_RATE_SWAP)

        assert profile is not None
        assert profile.currency == "USD"
        assert profile.product_type == ProductTypeConvention.INTEREST_RATE_SWAP


class TestMultiCurrencyConventions:
    """Test conventions for multiple currencies"""

    @pytest.mark.parametrize("currency,fixed_freq,float_freq", [
        ("USD", Frequency.SEMI_ANNUAL, Frequency.QUARTERLY),
        ("EUR", Frequency.ANNUAL, Frequency.SEMI_ANNUAL),
        ("GBP", Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL),
        ("JPY", Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL),
    ])
    def test_irs_conventions(self, currency, fixed_freq, float_freq):
        """Test IRS conventions for major currencies"""
        profile = get_convention_profile(currency, ProductTypeConvention.INTEREST_RATE_SWAP)

        assert profile is not None
        assert profile.fixed_leg.frequency == fixed_freq
        assert profile.floating_leg.frequency == float_freq

    @pytest.mark.parametrize("currency,day_count_str", [
        ("USD", "Actual360"),
        ("EUR", "Actual360"),
        ("GBP", "Actual365Fixed"),
        ("JPY", "Actual365Fixed"),
    ])
    def test_ois_day_count(self, currency, day_count_str):
        """Test OIS day count conventions"""
        profile = get_convention_profile(currency, ProductTypeConvention.OVERNIGHT_INDEX_SWAP)

        assert profile is not None
        assert str(profile.fixed_leg.day_count) == day_count_str
        assert str(profile.floating_leg.day_count) == day_count_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
