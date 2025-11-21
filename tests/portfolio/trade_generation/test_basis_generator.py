"""Tests for Basis Swap Generator"""

import pytest
from datetime import date

from neutryx.portfolio.trade_generation.generators.basis import (
    BasisSwapGenerator,
    generate_basis_swap_trade,
)
from neutryx.portfolio.contracts.trade import ProductType
from neutryx.products.linear_rates.swaps import BasisSwap, Tenor
from neutryx.core.dates.schedule import Frequency


class TestBasisSwapGenerator:
    """Test Basis Swap Generator"""

    def test_basis_swap_generator_creation(self):
        """Test creating basis swap generator"""
        generator = BasisSwapGenerator()
        assert generator is not None
        assert generator.factory is not None

    def test_frequency_to_tenor_mapping(self):
        """Test mapping Frequency to Tenor"""
        generator = BasisSwapGenerator()

        assert generator._frequency_to_tenor(Frequency.MONTHLY) == Tenor.ONE_MONTH
        assert generator._frequency_to_tenor(Frequency.QUARTERLY) == Tenor.THREE_MONTH
        assert generator._frequency_to_tenor(Frequency.SEMI_ANNUAL) == Tenor.SIX_MONTH
        assert generator._frequency_to_tenor(Frequency.ANNUAL) == Tenor.TWELVE_MONTH

        # Invalid frequency
        with pytest.raises(ValueError):
            generator._frequency_to_tenor(Frequency.ZERO)

    def test_tenor_to_payment_frequency(self):
        """Test converting Tenor to payment frequency"""
        generator = BasisSwapGenerator()

        assert generator._tenor_to_payment_frequency(Tenor.ONE_MONTH) == 12
        assert generator._tenor_to_payment_frequency(Tenor.THREE_MONTH) == 4
        assert generator._tenor_to_payment_frequency(Tenor.SIX_MONTH) == 2
        assert generator._tenor_to_payment_frequency(Tenor.TWELVE_MONTH) == 1

    def test_generate_usd_basis_swap_default_tenors(self):
        """Test generating USD basis swap with default convention tenors"""
        generator = BasisSwapGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=50_000_000,
            basis_spread=0.0025,  # 25 basis points
            counterparty_id="CP-001",
        )

        # Verify trade
        assert result.trade is not None
        assert result.trade.currency == "USD"
        assert result.trade.notional == 50_000_000
        assert result.trade.product_type == ProductType.INTEREST_RATE_SWAP
        assert result.trade.convention_profile_id == "USD_BASIS"
        assert result.trade.generated_from_convention is True

        # Verify product
        assert isinstance(result.product, BasisSwap)
        assert result.product.notional == 50_000_000
        assert result.product.T == 5.0
        assert result.product.basis_spread == 0.0025
        # USD default: 3M vs 1M SOFR
        assert result.product.tenor_1 == Tenor.THREE_MONTH
        assert result.product.tenor_2 == Tenor.ONE_MONTH

        # Verify product details
        assert result.trade.product_details["product_subtype"] == "BASIS_SWAP"
        assert result.trade.product_details["basis_spread"] == 0.0025
        assert result.trade.product_details["maturity_years"] == 5.0

    def test_generate_eur_basis_swap_default_tenors(self):
        """Test generating EUR basis swap with default convention tenors"""
        generator = BasisSwapGenerator()

        result = generator.generate(
            currency="EUR",
            trade_date=date(2024, 1, 15),
            maturity_years=10.0,
            notional=100_000_000,
            basis_spread=0.0015,  # 15 basis points
            counterparty_id="CP-002",
        )

        assert result.trade.currency == "EUR"
        assert result.trade.convention_profile_id == "EUR_BASIS"
        # EUR default: 3M vs 6M EURIBOR
        assert result.product.tenor_1 == Tenor.THREE_MONTH
        assert result.product.tenor_2 == Tenor.SIX_MONTH
        assert result.product.T == 10.0
        assert result.product.basis_spread == 0.0015

    def test_generate_gbp_basis_swap_default_tenors(self):
        """Test generating GBP basis swap with default convention tenors"""
        generator = BasisSwapGenerator()

        result = generator.generate(
            currency="GBP",
            trade_date=date(2024, 1, 15),
            maturity_years=7.0,
            notional=75_000_000,
            basis_spread=0.002,  # 20 basis points
            counterparty_id="CP-003",
        )

        assert result.trade.currency == "GBP"
        assert result.trade.convention_profile_id == "GBP_BASIS"
        # GBP default: 3M vs 6M SONIA
        assert result.product.tenor_1 == Tenor.THREE_MONTH
        assert result.product.tenor_2 == Tenor.SIX_MONTH
        assert result.product.T == 7.0

    def test_generate_basis_swap_custom_tenors(self):
        """Test generating basis swap with custom tenor overrides"""
        generator = BasisSwapGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=50_000_000,
            basis_spread=0.003,
            counterparty_id="CP-001",
            tenor_1=Tenor.SIX_MONTH,  # Override to 6M
            tenor_2=Tenor.THREE_MONTH,  # Override to 3M
        )

        # Verify custom tenors are used
        assert result.product.tenor_1 == Tenor.SIX_MONTH
        assert result.product.tenor_2 == Tenor.THREE_MONTH
        assert result.trade.product_details["tenor_1"] == "6M"
        assert result.trade.product_details["tenor_2"] == "3M"

    def test_convenience_function_basis_swap(self):
        """Test convenience function for basis swap generation"""
        trade, product, result = generate_basis_swap_trade(
            "USD",
            date(2024, 1, 15),
            5.0,  # 5 years
            50_000_000,
            0.0025,  # 25bp spread
            "CP-001",
        )

        assert trade is not None
        assert product is not None
        assert result is not None
        assert trade.id is not None
        assert product.T == 5.0
        assert product.basis_spread == 0.0025

    def test_basis_swap_with_metadata(self):
        """Test generating basis swap with full metadata"""
        generator = BasisSwapGenerator()

        result = generator.generate(
            currency="EUR",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=80_000_000,
            basis_spread=0.002,
            counterparty_id="CP-004",
            trade_number="BASIS-2024-001",
            book_id="BOOK-002",
            desk_id="DESK-RATES",
            trader_id="TRADER-456",
        )

        assert result.trade.trade_number == "BASIS-2024-001"
        assert result.trade.book_id == "BOOK-002"
        assert result.trade.desk_id == "DESK-RATES"
        assert result.trade.trader_id == "TRADER-456"

    @pytest.mark.parametrize("currency,tenor_1,tenor_2", [
        ("USD", Tenor.ONE_MONTH, Tenor.THREE_MONTH),
        ("USD", Tenor.THREE_MONTH, Tenor.SIX_MONTH),
        ("EUR", Tenor.THREE_MONTH, Tenor.SIX_MONTH),
        ("GBP", Tenor.ONE_MONTH, Tenor.THREE_MONTH),
    ])
    def test_multiple_tenor_combinations(self, currency, tenor_1, tenor_2):
        """Test generating basis swaps with various tenor combinations"""
        generator = BasisSwapGenerator()

        result = generator.generate(
            currency=currency,
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=50_000_000,
            basis_spread=0.0025,
            counterparty_id="CP-001",
            tenor_1=tenor_1,
            tenor_2=tenor_2,
        )

        assert result.product.tenor_1 == tenor_1
        assert result.product.tenor_2 == tenor_2
        assert result.trade.currency == currency

    def test_basis_swap_dates_calculation(self):
        """Test that dates are calculated correctly"""
        generator = BasisSwapGenerator()

        trade_date = date(2024, 1, 15)
        result = generator.generate(
            currency="USD",
            trade_date=trade_date,
            maturity_years=5.0,
            notional=50_000_000,
            basis_spread=0.0025,
            counterparty_id="CP-001",
        )

        # USD has spot_lag=2
        from datetime import timedelta
        expected_effective = trade_date + timedelta(days=2)
        assert result.trade.effective_date == expected_effective

        # Maturity should be approximately 5 years from effective
        days_diff = (result.trade.maturity_date - expected_effective).days
        assert 1825 <= days_diff <= 1826  # 5 years * 365 days

    def test_basis_swap_payment_frequency(self):
        """Test payment frequency calculation uses higher frequency leg"""
        generator = BasisSwapGenerator()

        # Test with 1M vs 3M (should use 1M frequency = 12 payments/year)
        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=50_000_000,
            basis_spread=0.0025,
            counterparty_id="CP-001",
            tenor_1=Tenor.ONE_MONTH,
            tenor_2=Tenor.THREE_MONTH,
        )
        assert result.product.payment_frequency == 12  # Monthly

        # Test with 3M vs 6M (should use 3M frequency = 4 payments/year)
        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=50_000_000,
            basis_spread=0.0025,
            counterparty_id="CP-001",
            tenor_1=Tenor.THREE_MONTH,
            tenor_2=Tenor.SIX_MONTH,
        )
        assert result.product.payment_frequency == 4  # Quarterly

    def test_basis_swap_validation_result(self):
        """Test that validation result is created"""
        generator = BasisSwapGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            maturity_years=5.0,
            notional=50_000_000,
            basis_spread=0.0025,
            counterparty_id="CP-001",
        )

        assert result.validation_result is not None
        assert result.convention_profile is not None
        assert result.convention_profile.currency == "USD"

    def test_invalid_currency_raises_error(self):
        """Test that invalid currency raises appropriate error"""
        generator = BasisSwapGenerator()

        with pytest.raises(ValueError, match="No Basis Swap convention profile found"):
            generator.generate(
                currency="XXX",  # Invalid currency
                trade_date=date(2024, 1, 15),
                maturity_years=5.0,
                notional=50_000_000,
                basis_spread=0.0025,
                counterparty_id="CP-001",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
