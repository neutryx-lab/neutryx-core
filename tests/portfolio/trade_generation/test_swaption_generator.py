"""Tests for Swaption Generator"""

import pytest
from datetime import date

from neutryx.portfolio.trade_generation.generators.swaption import (
    SwaptionGenerator,
    generate_swaption_trade,
)
from neutryx.portfolio.contracts.trade import ProductType
from neutryx.products.swaptions import EuropeanSwaption, SwaptionType
from neutryx.core.dates.schedule import Frequency


class TestSwaptionGenerator:
    """Test Swaption Generator"""

    def test_swaption_generator_creation(self):
        """Test creating swaption generator"""
        generator = SwaptionGenerator()
        assert generator is not None
        assert generator.factory is not None

    def test_frequency_to_payments_per_year(self):
        """Test converting Frequency to payment frequency"""
        generator = SwaptionGenerator()

        assert generator._frequency_to_payments_per_year(Frequency.ANNUAL) == 1
        assert generator._frequency_to_payments_per_year(Frequency.SEMI_ANNUAL) == 2
        assert generator._frequency_to_payments_per_year(Frequency.QUARTERLY) == 4
        assert generator._frequency_to_payments_per_year(Frequency.MONTHLY) == 12

    def test_calculate_annuity(self):
        """Test annuity calculation"""
        generator = SwaptionGenerator()

        # Test 5Y swap, semi-annual payments
        annuity = generator._calculate_annuity(
            swap_maturity_years=5.0,
            payment_frequency=2,
            discount_rate=0.05,
        )

        # Annuity should be positive and reasonable
        assert annuity > 0
        assert 4.0 < annuity < 5.0  # Approximate expected range

    def test_generate_usd_payer_swaption(self):
        """Test generating USD payer swaption (1Y into 5Y)"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,  # 1Y option
            swap_maturity_years=5.0,    # Into 5Y swap (1x5 swaption)
            notional=100_000_000,
            strike=0.05,  # 5% swap rate
            counterparty_id="CP-001",
            is_payer=True,
            volatility=0.20,
        )

        # Verify trade
        assert result.trade is not None
        assert result.trade.currency == "USD"
        assert result.trade.notional == 100_000_000
        assert result.trade.product_type == ProductType.SWAPTION
        assert result.trade.convention_profile_id == "USD_SWAPTION"
        assert result.trade.generated_from_convention is True

        # Verify product
        assert isinstance(result.product, EuropeanSwaption)
        assert result.product.notional == 100_000_000
        assert result.product.strike == 0.05
        assert result.product.swaption_type == SwaptionType.PAYER
        assert result.product.T == 1.0  # Option maturity
        assert result.product.annuity > 0

        # Verify product details
        assert result.trade.product_details["product_subtype"] == "SWAPTION"
        assert result.trade.product_details["swaption_type"] == "PAYER"
        assert result.trade.product_details["option_maturity_years"] == 1.0
        assert result.trade.product_details["swap_maturity_years"] == 5.0
        assert result.trade.product_details["strike"] == 0.05
        assert result.trade.product_details["volatility"] == 0.20

    def test_generate_usd_receiver_swaption(self):
        """Test generating USD receiver swaption"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=2.0,
            swap_maturity_years=10.0,  # 2x10 swaption
            notional=200_000_000,
            strike=0.045,  # 4.5% swap rate
            counterparty_id="CP-002",
            is_payer=False,  # Receiver swaption
            volatility=0.18,
        )

        # Verify product
        assert result.product.swaption_type == SwaptionType.RECEIVER
        assert result.product.T == 2.0
        assert result.product.strike == 0.045

        # Verify product details
        assert result.trade.product_details["swaption_type"] == "RECEIVER"

    def test_generate_eur_swaption(self):
        """Test generating EUR swaption"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="EUR",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=150_000_000,
            strike=0.035,  # 3.5% swap rate
            counterparty_id="CP-003",
            is_payer=True,
        )

        assert result.trade.currency == "EUR"
        assert result.trade.convention_profile_id == "EUR_SWAPTION"
        assert result.product.strike == 0.035

    def test_generate_gbp_swaption(self):
        """Test generating GBP swaption"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="GBP",
            trade_date=date(2024, 1, 15),
            option_maturity_years=0.5,  # 6M option
            swap_maturity_years=3.0,    # 6Mx3Y swaption
            notional=100_000_000,
            strike=0.055,
            counterparty_id="CP-004",
            is_payer=False,
        )

        assert result.trade.currency == "GBP"
        assert result.trade.convention_profile_id == "GBP_SWAPTION"
        assert result.product.T == 0.5

    def test_convenience_function_swaption(self):
        """Test convenience function for swaption generation"""
        trade, product, result = generate_swaption_trade(
            "USD",
            date(2024, 1, 15),
            1.0,  # 1Y option
            5.0,  # Into 5Y swap
            100_000_000,
            0.05,  # 5% strike
            "CP-001",
            is_payer=True,
        )

        assert trade is not None
        assert product is not None
        assert result is not None
        assert trade.id is not None
        assert product.T == 1.0
        assert product.strike == 0.05

    def test_swaption_with_metadata(self):
        """Test generating swaption with full metadata"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-005",
            trade_number="SWAPTION-2024-001",
            book_id="BOOK-005",
            desk_id="DESK-RATES",
            trader_id="TRADER-111",
        )

        assert result.trade.trade_number == "SWAPTION-2024-001"
        assert result.trade.book_id == "BOOK-005"
        assert result.trade.desk_id == "DESK-RATES"
        assert result.trade.trader_id == "TRADER-111"

    @pytest.mark.parametrize("currency,expected_profile", [
        ("USD", "USD_SWAPTION"),
        ("EUR", "EUR_SWAPTION"),
        ("GBP", "GBP_SWAPTION"),
    ])
    def test_multiple_currencies(self, currency, expected_profile):
        """Test generating swaptions in multiple currencies"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency=currency,
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        assert result.trade.currency == currency
        assert result.trade.convention_profile_id == expected_profile

    @pytest.mark.parametrize("option_years,swap_years", [
        (0.5, 5.0),  # 6M into 5Y
        (1.0, 5.0),  # 1Y into 5Y
        (2.0, 10.0),  # 2Y into 10Y
        (1.0, 10.0),  # 1Y into 10Y
    ])
    def test_multiple_maturities(self, option_years, swap_years):
        """Test generating swaptions with various maturity combinations"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=option_years,
            swap_maturity_years=swap_years,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        assert result.product.T == option_years
        assert result.trade.product_details["swap_maturity_years"] == swap_years

    def test_swaption_dates_calculation(self):
        """Test that dates are calculated correctly"""
        generator = SwaptionGenerator()

        trade_date = date(2024, 1, 15)
        result = generator.generate(
            currency="USD",
            trade_date=trade_date,
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        # USD has spot_lag=2
        from datetime import timedelta
        expected_effective = trade_date + timedelta(days=2)
        assert result.trade.effective_date == expected_effective

        # Option expiry should be 1 year from effective
        option_expiry_days = (result.trade.product_details["option_expiry_date"])
        # Parse and verify it's ~1 year from effective
        from datetime import date as dt
        option_expiry = dt.fromisoformat(option_expiry_days)
        days_to_expiry = (option_expiry - expected_effective).days
        assert 365 <= days_to_expiry <= 366

        # Final maturity should be option expiry + 5Y
        days_to_maturity = (result.trade.maturity_date - option_expiry).days
        assert 1825 <= days_to_maturity <= 1826  # 5 years * 365 days

    def test_swaption_different_volatilities(self):
        """Test swaptions with different volatility levels"""
        generator = SwaptionGenerator()

        # Low volatility
        result_low = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
            volatility=0.10,  # 10%
        )

        # High volatility
        result_high = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
            volatility=0.30,  # 30%
        )

        assert result_low.trade.product_details["volatility"] == 0.10
        assert result_high.trade.product_details["volatility"] == 0.30

    def test_swaption_different_strikes(self):
        """Test swaptions with different strike levels"""
        generator = SwaptionGenerator()

        # Low strike
        result_low = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.03,  # 3%
            counterparty_id="CP-001",
        )

        # High strike
        result_high = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.07,  # 7%
            counterparty_id="CP-001",
        )

        assert result_low.product.strike == 0.03
        assert result_high.product.strike == 0.07

    def test_swaption_annuity_is_positive(self):
        """Test that calculated annuity is always positive"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        assert result.product.annuity > 0
        assert result.trade.product_details["annuity"] > 0

    def test_swaption_validation_result(self):
        """Test that validation result is created"""
        generator = SwaptionGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            option_maturity_years=1.0,
            swap_maturity_years=5.0,
            notional=100_000_000,
            strike=0.05,
            counterparty_id="CP-001",
        )

        assert result.validation_result is not None
        assert result.convention_profile is not None
        assert result.convention_profile.currency == "USD"

    def test_invalid_currency_raises_error(self):
        """Test that invalid currency raises appropriate error"""
        generator = SwaptionGenerator()

        with pytest.raises(ValueError, match="No Swaption convention profile found"):
            generator.generate(
                currency="XXX",  # Invalid currency
                trade_date=date(2024, 1, 15),
                option_maturity_years=1.0,
                swap_maturity_years=5.0,
                notional=100_000_000,
                strike=0.05,
                counterparty_id="CP-001",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
