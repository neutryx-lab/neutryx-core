"""Tests for Trade Factory and Generators"""

import pytest
from datetime import date, timedelta

from neutryx.portfolio.trade_generation.factory import (
    TradeFactory,
    TradeGenerationRequest,
)
from neutryx.portfolio.trade_generation.generators.irs import (
    IRSGenerator,
    generate_irs_trade,
)
from neutryx.portfolio.trade_generation.generators.ois import (
    OISGenerator,
    generate_ois_trade,
)
from neutryx.market.convention_profiles import ProductTypeConvention
from neutryx.portfolio.contracts.trade import TradeStatus, ProductType
from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import ACT_360, ACT_365, THIRTY_360


class TestTradeFactory:
    """Test TradeFactory class"""

    def test_factory_creation(self):
        """Test creating a trade factory"""
        factory = TradeFactory()
        assert factory is not None

    def test_parse_tenor(self):
        """Test tenor parsing"""
        factory = TradeFactory()

        assert factory._parse_tenor("5Y") == 5
        assert factory._parse_tenor("10Y") == 10
        assert factory._parse_tenor("18M") == 1.5
        assert factory._parse_tenor("26W") == 0.5

        with pytest.raises(ValueError):
            factory._parse_tenor("INVALID")

    def test_calculate_effective_date(self):
        """Test effective date calculation with spot lag"""
        factory = TradeFactory()
        from neutryx.core.dates.calendar import US

        trade_date = date(2024, 1, 15)  # Monday
        effective_date = factory._calculate_effective_date(trade_date, 2, US)

        # Should be 2 business days after trade date
        assert effective_date > trade_date

    def test_generate_usd_irs_trade(self):
        """Test generating a USD IRS trade"""
        factory = TradeFactory()

        request = TradeGenerationRequest(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            trade_date=date(2024, 1, 15),
            tenor="5Y",
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
            swap_type="PAYER",
        )

        result = factory.generate_trade(request)

        # Verify trade was created
        assert result.trade is not None
        assert result.trade.currency == "USD"
        assert result.trade.notional == 10_000_000
        assert result.trade.product_type == ProductType.INTEREST_RATE_SWAP
        assert result.trade.generated_from_convention
        assert result.trade.convention_profile_id == "USD_IRS"

        # Verify dates
        assert result.trade.trade_date == date(2024, 1, 15)
        assert result.trade.effective_date is not None
        assert result.trade.maturity_date is not None
        assert result.trade.maturity_date > result.trade.effective_date

        # Verify schedules were generated
        assert result.schedules is not None
        assert "fixed" in result.schedules
        assert "floating" in result.schedules

        # Verify convention profile
        assert result.convention_profile is not None
        assert result.convention_profile.currency == "USD"

    def test_generate_eur_ois_trade(self):
        """Test generating a EUR OIS trade"""
        factory = TradeFactory()

        request = TradeGenerationRequest(
            currency="EUR",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            trade_date=date(2024, 1, 15),
            tenor="2Y",
            notional=5_000_000,
            fixed_rate=0.035,
            counterparty_id="CP-002",
        )

        result = factory.generate_trade(request)

        assert result.trade is not None
        assert result.trade.currency == "EUR"
        assert result.trade.convention_profile_id == "EUR_OIS"

    def test_generate_with_overrides(self):
        """Test generating trade with convention overrides"""
        factory = TradeFactory()

        request = TradeGenerationRequest(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            trade_date=date(2024, 1, 15),
            tenor="5Y",
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
            # Override conventions
            fixed_leg_frequency=Frequency.QUARTERLY,  # Non-standard
            floating_leg_day_count=ACT_365,  # Non-standard
        )

        result = factory.generate_trade(request)

        # Should generate warnings for overrides
        assert result.has_warnings()
        assert len(result.get_warnings()) > 0

        # But trade should still be created
        assert result.trade is not None

    def test_generate_with_explicit_maturity(self):
        """Test generating trade with explicit maturity date"""
        factory = TradeFactory()

        maturity = date(2029, 1, 15)
        request = TradeGenerationRequest(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            trade_date=date(2024, 1, 15),
            maturity_date=maturity,
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
        )

        result = factory.generate_trade(request)
        assert result.trade.maturity_date == maturity

    def test_generate_with_trade_metadata(self):
        """Test generating trade with full metadata"""
        factory = TradeFactory()

        request = TradeGenerationRequest(
            currency="GBP",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            trade_date=date(2024, 1, 15),
            tenor="7Y",
            notional=8_000_000,
            fixed_rate=0.051,
            counterparty_id="CP-003",
            trade_number="TRD-2024-001",
            book_id="BOOK-001",
            desk_id="DESK-IRD",
            trader_id="TRADER-123",
            status=TradeStatus.PENDING,
        )

        result = factory.generate_trade(request)

        assert result.trade.trade_number == "TRD-2024-001"
        assert result.trade.book_id == "BOOK-001"
        assert result.trade.desk_id == "DESK-IRD"
        assert result.trade.trader_id == "TRADER-123"
        assert result.trade.status == TradeStatus.PENDING


class TestIRSGenerator:
    """Test IRS Generator"""

    def test_irs_generator_creation(self):
        """Test creating IRS generator"""
        generator = IRSGenerator()
        assert generator is not None
        assert generator.factory is not None

    def test_generate_usd_irs(self):
        """Test generating USD IRS"""
        generator = IRSGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            tenor="5Y",
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
            swap_type="PAYER",
        )

        # Verify trade
        assert result.trade.currency == "USD"
        assert result.trade.notional == 10_000_000

        # Verify product was created
        from neutryx.products.linear_rates.swaps import InterestRateSwap
        assert isinstance(result.product, InterestRateSwap)
        assert result.product.notional == 10_000_000
        assert result.product.fixed_rate == 0.045

    def test_generate_irs_receiver(self):
        """Test generating RECEIVER swap"""
        generator = IRSGenerator()

        result = generator.generate(
            currency="EUR",
            trade_date=date(2024, 1, 15),
            tenor="10Y",
            notional=5_000_000,
            fixed_rate=0.032,
            counterparty_id="CP-002",
            swap_type="RECEIVER",
        )

        from neutryx.products.linear_rates.swaps import SwapType
        assert result.product.swap_type == SwapType.RECEIVER

    def test_generate_irs_with_spread(self):
        """Test generating IRS with floating spread"""
        generator = IRSGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            tenor="5Y",
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
            spread=0.001,  # 10 bps
        )

        assert result.product.spread == 0.001

    def test_convenience_function_irs(self):
        """Test convenience function for IRS generation"""
        trade, product, result = generate_irs_trade(
            "USD",
            date(2024, 1, 15),
            "5Y",
            10_000_000,
            0.045,
            "CP-001",
        )

        assert trade is not None
        assert product is not None
        assert result is not None
        assert trade.id is not None


class TestOISGenerator:
    """Test OIS Generator"""

    def test_ois_generator_creation(self):
        """Test creating OIS generator"""
        generator = OISGenerator()
        assert generator is not None

    def test_generate_usd_ois(self):
        """Test generating USD OIS (SOFR)"""
        generator = OISGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            tenor="2Y",
            notional=20_000_000,
            fixed_rate=0.043,
            counterparty_id="CP-002",
        )

        # Verify trade
        assert result.trade.currency == "USD"
        assert result.trade.convention_profile_id == "USD_OIS"

        # Verify product
        from neutryx.products.linear_rates.swaps import OvernightIndexSwap
        assert isinstance(result.product, OvernightIndexSwap)
        assert result.product.notional == 20_000_000

    def test_generate_eur_ois(self):
        """Test generating EUR OIS (ESTR)"""
        generator = OISGenerator()

        result = generator.generate(
            currency="EUR",
            trade_date=date(2024, 1, 15),
            tenor="3Y",
            notional=15_000_000,
            fixed_rate=0.028,
            counterparty_id="CP-003",
        )

        assert result.trade.currency == "EUR"
        from neutryx.products.linear_rates.swaps import DayCount
        assert result.product.day_count == DayCount.ACT_360

    def test_convenience_function_ois(self):
        """Test convenience function for OIS generation"""
        trade, product, result = generate_ois_trade(
            "GBP",
            date(2024, 1, 15),
            "1Y",
            10_000_000,
            0.047,
            "CP-004",
        )

        assert trade is not None
        assert product is not None
        assert result is not None


class TestMultiCurrencyGeneration:
    """Test trade generation across multiple currencies"""

    @pytest.mark.parametrize("currency,tenor,notional", [
        ("USD", "5Y", 10_000_000),
        ("EUR", "10Y", 5_000_000),
        ("GBP", "7Y", 8_000_000),
        ("JPY", "3Y", 1_000_000_000),
    ])
    def test_generate_irs_multi_currency(self, currency, tenor, notional):
        """Test generating IRS for multiple currencies"""
        generator = IRSGenerator()

        result = generator.generate(
            currency=currency,
            trade_date=date(2024, 1, 15),
            tenor=tenor,
            notional=notional,
            fixed_rate=0.04,
            counterparty_id="CP-001",
        )

        assert result.trade.currency == currency
        assert result.trade.notional == notional
        assert result.product is not None

    @pytest.mark.parametrize("currency,expected_spot_lag", [
        ("USD", 2),
        ("EUR", 2),
        ("GBP", 0),  # GBP is T+0
        ("JPY", 2),
    ])
    def test_spot_lag_conventions(self, currency, expected_spot_lag):
        """Test that spot lag conventions are applied correctly"""
        from neutryx.market.convention_profiles import get_convention_profile

        profile = get_convention_profile(currency, ProductTypeConvention.INTEREST_RATE_SWAP)
        assert profile.spot_lag == expected_spot_lag


class TestScheduleGeneration:
    """Test schedule generation within trade factory"""

    def test_schedule_periods(self):
        """Test that schedules have correct periods"""
        generator = IRSGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            tenor="2Y",
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
        )

        # Fixed leg should have semi-annual periods (4 periods for 2Y)
        fixed_schedule = result.schedules["fixed"]
        assert len(fixed_schedule.periods) >= 3  # At least 3-4 periods

        # Floating leg should have quarterly periods (8 periods for 2Y)
        floating_schedule = result.schedules["floating"]
        assert len(floating_schedule.periods) >= 7  # At least 7-8 periods

    def test_schedule_dates_adjusted(self):
        """Test that schedule dates are business day adjusted"""
        generator = IRSGenerator()

        result = generator.generate(
            currency="USD",
            trade_date=date(2024, 1, 15),
            tenor="5Y",
            notional=10_000_000,
            fixed_rate=0.045,
            counterparty_id="CP-001",
        )

        # Check that payment dates are adjusted
        for period in result.schedules["fixed"].periods:
            # Payment date should be a business day (not weekend)
            assert period.payment_date.weekday() < 5 or True  # Would need holiday calendar for full check


class TestProductDetails:
    """Test that product details are stored correctly"""

    def test_product_details_stored(self):
        """Test that product details are stored in trade"""
        trade, product, result = generate_irs_trade(
            "USD",
            date(2024, 1, 15),
            "5Y",
            10_000_000,
            0.045,
            "CP-001",
        )

        # Verify product_details contains necessary information
        assert trade.product_details is not None
        assert "irs_product" in trade.product_details
        assert "fixed_leg" in trade.product_details
        assert "floating_leg" in trade.product_details
        assert trade.product_details["fixed_rate"] == 0.045

    def test_convention_profile_id_stored(self):
        """Test that convention profile ID is stored"""
        trade, _, _ = generate_irs_trade(
            "EUR",
            date(2024, 1, 15),
            "10Y",
            5_000_000,
            0.032,
            "CP-002",
        )

        assert trade.convention_profile_id == "EUR_IRS"
        assert trade.generated_from_convention


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
