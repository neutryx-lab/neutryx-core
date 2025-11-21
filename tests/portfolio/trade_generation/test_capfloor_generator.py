"""Tests for Cap/Floor Generator"""

import pytest
from datetime import date

from neutryx.portfolio.trade_generation.generators.capfloor import (
    CapFloorGenerator,
    generate_cap_trade,
    generate_floor_trade,
)
from neutryx.portfolio.contracts.trade import ProductType
from neutryx.products.linear_rates.caps_floors import InterestRateCapFloorCollar, CapFloorType
from neutryx.core.dates.schedule import Frequency


class TestCapFloorGenerator:
    """Test Cap/Floor Generator"""

    def test_capfloor_generator_creation(self):
        """Test creating cap/floor generator"""
        generator = CapFloorGenerator()
        assert generator is not None
        assert generator.factory is not None

    def test_frequency_to_payments_per_year(self):
        """Test converting Frequency to payment frequency"""
        generator = CapFloorGenerator()

        assert generator._frequency_to_payments_per_year(Frequency.ANNUAL) == 1
        assert generator._frequency_to_payments_per_year(Frequency.SEMI_ANNUAL) == 2
        assert generator._frequency_to_payments_per_year(Frequency.QUARTERLY) == 4
        assert generator._frequency_to_payments_per_year(Frequency.MONTHLY) == 12

    def test_generate_usd_cap(self):
        """Test generating USD interest rate cap"""
        generator = CapFloorGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,  # 5% cap
            counterparty_id="CP-001",
            is_cap=True,
            volatility=0.20,
        )

        # Verify trade
        assert result.trade is not None
        assert result.trade.currency == "USD"
        assert result.trade.notional == 100_000_000
        assert result.trade.product_type == ProductType.SWAPTION
        assert result.trade.convention_profile_id == "USD_CAP"
        assert result.trade.generated_from_convention is True

        # Verify product
        assert isinstance(result.product, InterestRateCapFloorCollar)
        assert result.product.notional == 100_000_000
        assert result.product.strike == 0.05
        assert result.product.cap_floor_type == CapFloorType.CAP
        assert result.product.T == 5.0
        assert result.product.volatility == 0.20
        assert result.product.payment_frequency == 4  # Quarterly

        # Verify product details
        assert result.trade.product_details["product_subtype"] == "CAP"
        assert result.trade.product_details["strike"] == 0.05
        assert result.trade.product_details["volatility"] == 0.20

    def test_generate_usd_floor(self):
        """Test generating USD interest rate floor"""
        generator = CapFloorGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.02,  # 2% floor
            counterparty_id="CP-001",
            is_cap=False,
            volatility=0.15,
        )

        # Verify trade
        assert result.trade.currency == "USD"
        assert result.trade.convention_profile_id == "USD_FLOOR"

        # Verify product
        assert result.product.strike == 0.02
        assert result.product.cap_floor_type == CapFloorType.FLOOR
        assert result.product.volatility == 0.15

        # Verify product details
        assert result.trade.product_details["product_subtype"] == "FLOOR"

    def test_generate_cap_method(self):
        """Test convenience generate_cap method"""
        generator = CapFloorGenerator()

        result = generator.generate_cap(
            currency="EUR",
            trade_date=date(2024, 1, 15),
            maturity_years=10.0,
            notional=200_000_000,
            strike=0.04,
            counterparty_id="CP-002",
            volatility=0.25,
        )

        assert result.product.cap_floor_type == CapFloorType.CAP
        assert result.product.strike == 0.04
        assert result.trade.currency == "EUR"

    def test_generate_floor_method(self):
        """Test convenience generate_floor method"""
        generator = CapFloorGenerator()

        result = generator.generate_floor(
            currency="GBP",
            trade_date=date(2024, 1, 15),
            maturity_years=7.0,
            notional=150_000_000,
            strike=0.03,
            counterparty_id="CP-003",
            volatility=0.18,
        )

        assert result.product.cap_floor_type == CapFloorType.FLOOR
        assert result.product.strike == 0.03
        assert result.trade.currency == "GBP"

    def test_convenience_function_cap(self):
        """Test convenience function for cap generation"""
        trade, product, result = generate_cap_trade(
            "USD",
            date(2024, 1, 15),
            5.0,
            100_000_000,
            0.05,
            "CP-001",
            volatility=0.20,
        )

        assert trade is not None
        assert product is not None
        assert result is not None
        assert trade.id is not None
        assert product.cap_floor_type == CapFloorType.CAP
        assert product.T == 5.0

    def test_convenience_function_floor(self):
        """Test convenience function for floor generation"""
        trade, product, result = generate_floor_trade(
            "EUR",
            date(2024, 1, 15),
            5.0,
            100_000_000,
            0.02,
            "CP-001",
            volatility=0.15,
        )

        assert trade is not None
        assert product is not None
        assert result is not None
        assert product.cap_floor_type == CapFloorType.FLOOR
        assert product.strike == 0.02

    def test_capfloor_with_metadata(self):
        """Test generating cap/floor with full metadata"""
        generator = CapFloorGenerator()

        result = generator.generate_cap(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-004",
            volatility=0.20,
            trade_number="CAP-2024-001",
            book_id="BOOK-004",
            desk_id="DESK-RATES",
            trader_id="TRADER-999",
        )

        assert result.trade.trade_number == "CAP-2024-001"
        assert result.trade.book_id == "BOOK-004"
        assert result.trade.desk_id == "DESK-RATES"
        assert result.trade.trader_id == "TRADER-999"

    @pytest.mark.parametrize("currency,expected_profile", [
        ("USD", "USD_CAP"),
        ("EUR", "EUR_CAP"),
        ("GBP", "GBP_CAP"),
    ])
    def test_multiple_currencies_cap(self, currency, expected_profile):
        """Test generating caps in multiple currencies"""
        generator = CapFloorGenerator()

        result = generator.generate_cap(
            currency=currency,
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        assert result.trade.currency == currency
        assert result.trade.convention_profile_id == expected_profile

    @pytest.mark.parametrize("currency,expected_profile", [
        ("USD", "USD_FLOOR"),
        ("EUR", "EUR_FLOOR"),
        ("GBP", "GBP_FLOOR"),
    ])
    def test_multiple_currencies_floor(self, currency, expected_profile):
        """Test generating floors in multiple currencies"""
        generator = CapFloorGenerator()

        result = generator.generate_floor(
            currency=currency,
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.02,
            counterparty_id="CP-001",
        )

        assert result.trade.currency == currency
        assert result.trade.convention_profile_id == expected_profile

    def test_capfloor_dates_calculation(self):
        """Test that dates are calculated correctly"""
        generator = CapFloorGenerator()

        trade_date = date(2024, 1, 15)
        result = generator.generate_cap(
            currency="USD",
            trade_date=trade_date,
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        # USD has spot_lag=2
        from datetime import timedelta
        expected_effective = trade_date + timedelta(days=2)
        assert result.trade.effective_date == expected_effective

        # Maturity should be approximately 5 years from effective
        days_diff = (result.trade.maturity_date - expected_effective).days
        assert 1825 <= days_diff <= 1826  # 5 years * 365 days

    def test_capfloor_different_volatilities(self):
        """Test caps/floors with different volatility levels"""
        generator = CapFloorGenerator()

        # Low volatility
        result_low = generator.generate_cap(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
            volatility=0.10,  # 10%
        )

        # High volatility
        result_high = generator.generate_cap(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
            volatility=0.30,  # 30%
        )

        assert result_low.product.volatility == 0.10
        assert result_high.product.volatility == 0.30
        assert result_low.trade.product_details["volatility"] == 0.10
        assert result_high.trade.product_details["volatility"] == 0.30

    def test_capfloor_different_strikes(self):
        """Test caps/floors with different strike levels"""
        generator = CapFloorGenerator()

        # Low strike cap (cheaper premium)
        result_high_strike = generator.generate_cap(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.06,  # 6%
            counterparty_id="CP-001",
        )

        # High strike floor (more expensive premium)
        result_high_floor = generator.generate_floor(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.03,  # 3%
            counterparty_id="CP-001",
        )

        assert result_high_strike.product.strike == 0.06
        assert result_high_floor.product.strike == 0.03

    def test_capfloor_validation_result(self):
        """Test that validation result is created"""
        generator = CapFloorGenerator()

        result = generator.generate_cap(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        assert result.validation_result is not None
        assert result.convention_profile is not None
        assert result.convention_profile.currency == "USD"

    def test_invalid_currency_cap_raises_error(self):
        """Test that invalid currency raises error for cap"""
        generator = CapFloorGenerator()

        with pytest.raises(ValueError, match="No Cap convention profile found"):
            generator.generate_cap(
                currency="XXX",  # Invalid currency
                trade_date=date(2024, 1, 15),
                maturity_years=5.0,
                notional=100_000_000,
                strike=0.05,
                counterparty_id="CP-001",
            )

    def test_invalid_currency_floor_raises_error(self):
        """Test that invalid currency raises error for floor"""
        generator = CapFloorGenerator()

        with pytest.raises(ValueError, match="No Floor convention profile found"):
            generator.generate_floor(
                currency="XXX",  # Invalid currency
                trade_date=date(2024, 1, 15),
                maturity_years=5.0,
                notional=100_000_000,
                strike=0.02,
                counterparty_id="CP-001",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
