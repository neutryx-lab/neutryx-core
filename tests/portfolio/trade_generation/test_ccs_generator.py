"""Tests for Cross-Currency Swap (CCS) Generator"""

import pytest
from datetime import date

from neutryx.portfolio.trade_generation.generators.ccs import (
    CCSGenerator,
    generate_ccs_trade,
)
from neutryx.portfolio.contracts.trade import ProductType
from neutryx.products.linear_rates.swaps import CrossCurrencySwap
from neutryx.core.dates.schedule import Frequency


class TestCCSGenerator:
    """Test Cross-Currency Swap Generator"""

    def test_ccs_generator_creation(self):
        """Test creating CCS generator"""
        generator = CCSGenerator()
        assert generator is not None
        assert generator.factory is not None

    def test_frequency_to_payments_per_year(self):
        """Test converting Frequency to payment frequency"""
        generator = CCSGenerator()

        assert generator._frequency_to_payments_per_year(Frequency.ANNUAL) == 1
        assert generator._frequency_to_payments_per_year(Frequency.SEMI_ANNUAL) == 2
        assert generator._frequency_to_payments_per_year(Frequency.QUARTERLY) == 4
        assert generator._frequency_to_payments_per_year(Frequency.MONTHLY) == 12

    def test_generate_usdeur_ccs_with_fx_reset(self):
        """Test generating USD/EUR CCS with FX reset"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair="USDEUR",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional_domestic=100_000_000,  # 100M USD
            notional_foreign=90_000_000,    # 90M EUR
            domestic_rate=0.04,  # 4% USD
            foreign_rate=0.03,   # 3% EUR
            fx_spot=1.11,  # USD/EUR = 1.11
            counterparty_id="CP-001",
            fx_reset=True,
        )

        # Verify trade
        assert result.trade is not None
        assert result.trade.currency == "USD"
        assert result.trade.notional == 100_000_000
        assert result.trade.product_type == ProductType.INTEREST_RATE_SWAP
        assert result.trade.convention_profile_id == "USDEUR_CCS"
        assert result.trade.generated_from_convention is True

        # Verify product
        assert isinstance(result.product, CrossCurrencySwap)
        assert result.product.notional_domestic == 100_000_000
        assert result.product.notional_foreign == 90_000_000
        assert result.product.domestic_rate == 0.04
        assert result.product.foreign_rate == 0.03
        assert result.product.fx_spot == 1.11
        assert result.product.fx_reset is True
        assert result.product.T == 5.0

        # Verify product details
        assert result.trade.product_details["product_subtype"] == "CROSS_CURRENCY_SWAP"
        assert result.trade.product_details["currency_pair"] == "USDEUR"
        assert result.trade.product_details["domestic_currency"] == "USD"
        assert result.trade.product_details["foreign_currency"] == "EUR"

    def test_generate_usdeur_ccs_without_fx_reset(self):
        """Test generating USD/EUR CCS without FX reset"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair="USDEUR",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional_domestic=100_000_000,
            notional_foreign=90_000_000,
            domestic_rate=0.04,
            foreign_rate=0.03,
            fx_spot=1.11,
            counterparty_id="CP-001",
            fx_reset=False,  # No FX reset
        )

        assert result.product.fx_reset is False
        assert result.trade.product_details["fx_reset"] is False

    def test_generate_usdjpy_ccs(self):
        """Test generating USD/JPY CCS"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair="USDJPY",
            trade_date=date(2024, 1, 15),
            maturity_years=10.0,
            notional_domestic=100_000_000,  # 100M USD
            notional_foreign=11_000_000_000,  # 11B JPY
            domestic_rate=0.045,  # 4.5% USD
            foreign_rate=0.005,   # 0.5% JPY
            fx_spot=110.0,  # USD/JPY = 110
            counterparty_id="CP-002",
        )

        assert result.trade.currency == "USD"
        assert result.trade.convention_profile_id == "USDJPY_CCS"
        assert result.product.notional_domestic == 100_000_000
        assert result.product.notional_foreign == 11_000_000_000
        assert result.product.fx_spot == 110.0
        assert result.product.T == 10.0
        assert result.trade.product_details["domestic_currency"] == "USD"
        assert result.trade.product_details["foreign_currency"] == "JPY"

    def test_generate_eurgbp_ccs(self):
        """Test generating EUR/GBP CCS"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair="EURGBP",
            trade_date=date(2024, 1, 15),
            maturity_years=7.0,
            notional_domestic=80_000_000,  # 80M EUR
            notional_foreign=70_000_000,   # 70M GBP
            domestic_rate=0.03,  # 3% EUR
            foreign_rate=0.045,  # 4.5% GBP
            fx_spot=1.14,  # EUR/GBP = 1.14
            counterparty_id="CP-003",
        )

        assert result.trade.currency == "EUR"
        assert result.trade.convention_profile_id == "EURGBP_CCS"
        assert result.product.T == 7.0
        assert result.trade.product_details["domestic_currency"] == "EUR"
        assert result.trade.product_details["foreign_currency"] == "GBP"

    def test_convenience_function_ccs(self):
        """Test convenience function for CCS generation"""
        trade, product, result = generate_ccs_trade(
            "USDEUR",
            date(2024, 1, 15),
            5.0,
            100_000_000,
            90_000_000,
            0.04,
            0.03,
            1.11,
            "CP-001",
        )

        assert trade is not None
        assert product is not None
        assert result is not None
        assert trade.id is not None
        assert product.T == 5.0
        assert product.fx_spot == 1.11

    def test_ccs_with_metadata(self):
        """Test generating CCS with full metadata"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair="USDEUR",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional_domestic=100_000_000,
            notional_foreign=90_000_000,
            domestic_rate=0.04,
            foreign_rate=0.03,
            fx_spot=1.11,
            counterparty_id="CP-004",
            trade_number="CCS-2024-001",
            book_id="BOOK-003",
            desk_id="DESK-FX",
            trader_id="TRADER-789",
        )

        assert result.trade.trade_number == "CCS-2024-001"
        assert result.trade.book_id == "BOOK-003"
        assert result.trade.desk_id == "DESK-FX"
        assert result.trade.trader_id == "TRADER-789"

    @pytest.mark.parametrize("currency_pair,domestic_ccy,foreign_ccy", [
        ("USDEUR", "USD", "EUR"),
        ("USDJPY", "USD", "JPY"),
        ("EURGBP", "EUR", "GBP"),
    ])
    def test_multiple_currency_pairs(self, currency_pair, domestic_ccy, foreign_ccy):
        """Test generating CCS with various currency pairs"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair=currency_pair,
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional_domestic=100_000_000,
            notional_foreign=90_000_000,
            domestic_rate=0.04,
            foreign_rate=0.03,
            fx_spot=1.11,
            counterparty_id="CP-001",
        )

        assert result.trade.currency == domestic_ccy
        assert result.trade.product_details["domestic_currency"] == domestic_ccy
        assert result.trade.product_details["foreign_currency"] == foreign_ccy

    def test_ccs_dates_calculation(self):
        """Test that dates are calculated correctly"""
        generator = CCSGenerator()

        trade_date = date(2024, 1, 15)
        result = generator.generate(
            currency_pair="USDEUR",
            trade_date=trade_date,
            maturity_years=5.0,
            notional_domestic=100_000_000,
            notional_foreign=90_000_000,
            domestic_rate=0.04,
            foreign_rate=0.03,
            fx_spot=1.11,
            counterparty_id="CP-001",
        )

        # USDEUR has spot_lag=2
        from datetime import timedelta
        expected_effective = trade_date + timedelta(days=2)
        assert result.trade.effective_date == expected_effective

        # Maturity should be approximately 5 years from effective
        days_diff = (result.trade.maturity_date - expected_effective).days
        assert 1825 <= days_diff <= 1826  # 5 years * 365 days

    def test_ccs_payment_frequency(self):
        """Test payment frequency uses higher frequency leg"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair="USDEUR",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional_domestic=100_000_000,
            notional_foreign=90_000_000,
            domestic_rate=0.04,
            foreign_rate=0.03,
            fx_spot=1.11,
            counterparty_id="CP-001",
        )

        # USD leg: semi-annual (2), EUR leg: annual (1) -> should use 2
        assert result.product.payment_frequency == 2

    def test_ccs_validation_result(self):
        """Test that validation result is created"""
        generator = CCSGenerator()

        result = generator.generate(
            currency_pair="USDEUR",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional_domestic=100_000_000,
            notional_foreign=90_000_000,
            domestic_rate=0.04,
            foreign_rate=0.03,
            fx_spot=1.11,
            counterparty_id="CP-001",
        )

        assert result.validation_result is not None
        assert result.convention_profile is not None
        assert result.convention_profile.currency == "USDEUR"

    def test_invalid_currency_pair_raises_error(self):
        """Test that invalid currency pair raises appropriate error"""
        generator = CCSGenerator()

        with pytest.raises(ValueError, match="No Cross-Currency Swap convention profile found"):
            generator.generate(
                currency_pair="XXXYYY",  # Invalid currency pair
                trade_date=date(2024, 1, 15),
                maturity_years=5.0,
                notional_domestic=100_000_000,
                notional_foreign=90_000_000,
                domestic_rate=0.04,
                foreign_rate=0.03,
                fx_spot=1.11,
                counterparty_id="CP-001",
            )

    def test_invalid_currency_pair_format_raises_error(self):
        """Test that invalid currency pair format raises error"""
        generator = CCSGenerator()

        # Mock profile exists but format is invalid
        with pytest.raises(ValueError, match="Invalid currency pair format"):
            generator.generate(
                currency_pair="USD",  # Too short
                trade_date=date(2024, 1, 15),
                maturity_years=5.0,
                notional_domestic=100_000_000,
                notional_foreign=90_000_000,
                domestic_rate=0.04,
                foreign_rate=0.03,
                fx_spot=1.11,
                counterparty_id="CP-001",
            )

    def test_ccs_different_rates(self):
        """Test CCS with different interest rate scenarios"""
        generator = CCSGenerator()

        # High USD rate, low EUR rate
        result = generator.generate(
            currency_pair="USDEUR",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional_domestic=100_000_000,
            notional_foreign=90_000_000,
            domestic_rate=0.055,  # 5.5% USD
            foreign_rate=0.015,   # 1.5% EUR
            fx_spot=1.11,
            counterparty_id="CP-001",
        )

        assert result.product.domestic_rate == 0.055
        assert result.product.foreign_rate == 0.015
        assert result.trade.product_details["domestic_rate"] == 0.055
        assert result.trade.product_details["foreign_rate"] == 0.015


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
