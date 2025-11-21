"""Tests for FRA Generator"""

import pytest
from datetime import date

from neutryx.portfolio.trade_generation.generators.fra import (
    FRAGenerator,
    generate_fra_trade,
)
from neutryx.portfolio.contracts.trade import ProductType
from neutryx.products.linear_rates.fra import ForwardRateAgreement


class TestFRAGenerator:
    """Test FRA Generator"""

    def test_fra_generator_creation(self):
        """Test creating FRA generator"""
        generator = FRAGenerator()
        assert generator is not None
        assert generator.factory is not None

    def test_parse_fra_tenor(self):
        """Test parsing FRA tenor notation"""
        generator = FRAGenerator()

        # Valid tenors
        assert generator._parse_fra_tenor("3x6") == (3, 6)
        assert generator._parse_fra_tenor("6x12") == (6, 12)
        assert generator._parse_fra_tenor("3X9") == (3, 9)  # Case insensitive

        # Invalid tenors
        with pytest.raises(ValueError):
            generator._parse_fra_tenor("3")  # Missing 'x'

        with pytest.raises(ValueError):
            generator._parse_fra_tenor("3x3")  # Maturity <= settlement

        with pytest.raises(ValueError):
            generator._parse_fra_tenor("abc")  # Invalid format

    def test_generate_usd_fra_3x6(self):
        """Test generating USD FRA 3x6"""
        generator = FRAGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            fra_tenor="3x6",
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
            is_payer=True,
        )

        # Verify trade
        assert result.trade is not None
        assert result.trade.currency == "USD"
        assert result.trade.notional == 10_000_000
        assert result.trade.product_type == ProductType.FORWARD
        assert result.trade.convention_profile_id == "USD_FRA"

        # Verify product
        assert isinstance(result.product, ForwardRateAgreement)
        assert result.product.notional == 10_000_000
        assert result.product.fixed_rate == 0.045
        assert result.product.is_payer == True
        assert abs(result.product.T - 0.25) < 0.01  # ~3 months
        assert abs(result.product.period_length - 0.25) < 0.01  # ~3 month period

        # Verify product details
        assert result.trade.product_details["fra_tenor"] == "3x6"
        assert result.trade.product_details["settlement_months"] == 3
        assert result.trade.product_details["period_months"] == 3

    def test_generate_eur_fra_6x12(self):
        """Test generating EUR FRA 6x12"""
        generator = FRAGenerator()

        result = generator.generate(
            currency="EUR",
            trade_date=date(2024, 1, 15),
            fra_tenor="6x12",
            notional=5_000_000,
            fixed_rate=0.035,
            counterparty_id="CP-002",
            is_payer=False,  # Receiver (short FRA)
        )

        assert result.trade.currency == "EUR"
        assert result.product.is_payer == False
        assert abs(result.product.T - 0.5) < 0.01  # ~6 months
        assert abs(result.product.period_length - 0.5) < 0.01  # ~6 month period

    def test_convenience_function_fra(self):
        """Test convenience function for FRA generation"""
        trade, product, result = generate_fra_trade(
            "USD",
            date(2024, 1, 15),
            "3x9",  # 3mo to settlement, 6mo period
            10_000_000,
            0.045,
            "CP-001",
        )

        assert trade is not None
        assert product is not None
        assert result is not None
        assert trade.id is not None
        assert abs(product.T - 0.25) < 0.01
        assert abs(product.period_length - 0.5) < 0.01  # 6 month period

    def test_fra_with_metadata(self):
        """Test generating FRA with full metadata"""
        generator = FRAGenerator()

        result = generator.generate(
            currency="GBP",
            trade_date=date(2024, 1, 15),
            fra_tenor="3x6",
            notional=8_000_000,
            fixed_rate=0.051,
            counterparty_id="CP-003",
            trade_number="FRA-2024-001",
            book_id="BOOK-001",
            desk_id="DESK-RATES",
            trader_id="TRADER-123",
        )

        assert result.trade.trade_number == "FRA-2024-001"
        assert result.trade.book_id == "BOOK-001"
        assert result.trade.desk_id == "DESK-RATES"
        assert result.trade.trader_id == "TRADER-123"

    @pytest.mark.parametrize("currency,fra_tenor,settlement_mo,period_mo", [
        ("USD", "3x6", 3, 3),
        ("USD", "6x12", 6, 6),
        ("EUR", "3x9", 3, 6),
        ("GBP", "1x4", 1, 3),
    ])
    def test_multiple_fra_tenors(self, currency, fra_tenor, settlement_mo, period_mo):
        """Test generating FRAs with various tenors"""
        generator = FRAGenerator()

        result = generator.generate(
            currency=currency,
            trade_date=date(2024, 1, 15),
            fra_tenor=fra_tenor,
            notional=10_000_000,
            fixed_rate=0.04,
            counterparty_id="CP-001",
        )

        assert result.trade.product_details["settlement_months"] == settlement_mo
        assert result.trade.product_details["period_months"] == period_mo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
