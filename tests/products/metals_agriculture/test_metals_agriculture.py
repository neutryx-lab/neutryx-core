"""Comprehensive tests for metals and agriculture derivatives."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.metals_agriculture.precious_metals import (
    GoldOption,
    PlatinumOption,
    PreciousMetalFuture,
    PreciousMetalType,
    SilverOption,
    black_scholes_commodity,
)
from neutryx.products.metals_agriculture.base_metals import (
    AluminumFuture,
    BaseMetalOption,
    BaseMetalType,
    CopperFuture,
)
from neutryx.products.metals_agriculture.agriculture import (
    AgriculturalOption,
    AgriculturalType,
    CornFuture,
    SoybeanFuture,
    WheatFuture,
)


class TestPreciousMetals:
    """Tests for precious metals derivatives."""

    def test_precious_metal_future(self):
        """Test precious metal futures."""
        gold_future = PreciousMetalFuture(
            T=1.0,
            forward_price=1800.0,
            metal_type=PreciousMetalType.GOLD,
            notional=100.0,  # 100 oz
        )

        # Spot at $1850
        payoff = gold_future.payoff_terminal(1850.0)

        # Payoff = 100 oz × ($1850 - $1800) = $5,000
        assert payoff == 5000.0, "Gold future payoff should match formula"

    def test_fair_forward_price_gold(self):
        """Test fair forward price for gold."""
        gold_future = PreciousMetalFuture(
            T=1.0,
            forward_price=1800.0,
            metal_type=PreciousMetalType.GOLD,
            storage_cost=0.005,
            lease_rate=0.001,
        )

        fair_price = gold_future.fair_forward_price(spot=1800.0, risk_free_rate=0.03)

        # With r=3%, storage=0.5%, lease=0.1%, carry = 3.4%
        # Fair forward ≈ 1800 × exp(0.034) ≈ 1862
        assert 1850 < fair_price < 1870, "Fair forward should reflect carry costs"

    def test_gold_option_pricing(self):
        """Test gold option pricing."""
        gold_call = GoldOption(
            T=1.0,
            strike=1800.0,
            is_call=True,
            volatility=0.15,
        )

        price = gold_call.price(spot=1850.0)

        # ITM gold call should have value
        assert price > 5000.0, "ITM gold call should exceed intrinsic"

    def test_silver_option_pricing(self):
        """Test silver option pricing."""
        silver_put = SilverOption(
            T=1.0,
            strike=25.0,
            is_call=False,
            volatility=0.25,
        )

        price = silver_put.price(spot=22.0)

        # ITM silver put should have value
        assert price > 15000.0, "ITM silver put should have value"

    def test_platinum_option(self):
        """Test platinum option."""
        platinum_call = PlatinumOption(
            T=0.5,
            strike=1000.0,
            is_call=True,
            volatility=0.20,
        )

        price = platinum_call.price(spot=1050.0)

        assert price > 0, "ITM platinum call should have positive value"

    def test_silver_higher_vol_than_gold(self):
        """Test that silver has higher volatility than gold."""
        gold = GoldOption(T=1.0, strike=1800.0)
        silver = SilverOption(T=1.0, strike=25.0)

        assert silver.volatility > gold.volatility, "Silver should have higher vol than gold"


class TestBlackScholesCommodity:
    """Test Black-Scholes commodity pricing."""

    def test_bs_commodity_call_itm(self):
        """Test commodity call ITM."""
        price = black_scholes_commodity(
            S=110.0,
            K=100.0,
            T=1.0,
            r=0.03,
            q=0.02,  # Convenience yield
            sigma=0.20,
            is_call=True,
        )

        assert price > 10.0, "ITM call should have value > intrinsic"

    def test_bs_commodity_put_itm(self):
        """Test commodity put ITM."""
        price = black_scholes_commodity(
            S=90.0,
            K=100.0,
            T=1.0,
            r=0.03,
            q=0.02,
            sigma=0.20,
            is_call=False,
        )

        assert price > 10.0, "ITM put should have value > intrinsic"

    def test_bs_commodity_atm(self):
        """Test ATM commodity option."""
        call_price = black_scholes_commodity(
            S=100.0,
            K=100.0,
            T=1.0,
            r=0.03,
            q=0.02,
            sigma=0.20,
            is_call=True,
        )

        put_price = black_scholes_commodity(
            S=100.0,
            K=100.0,
            T=1.0,
            r=0.03,
            q=0.02,
            sigma=0.20,
            is_call=False,
        )

        # ATM options should have similar values
        assert abs(call_price - put_price) < 2.0, "ATM call and put should be close"


class TestBaseMetals:
    """Tests for base metals derivatives."""

    def test_copper_future(self):
        """Test copper futures."""
        copper = CopperFuture(
            T=1.0,
            forward_price=4.0,  # $/lb
            notional=25_000.0,  # 25,000 lbs
        )

        payoff = copper.payoff_terminal(4.2)

        # 25,000 lbs × $0.20/lb = $5,000
        assert abs(payoff - 5000.0) < 0.01, "Copper future payoff should match"

    def test_aluminum_future(self):
        """Test aluminum futures."""
        aluminum = AluminumFuture(
            T=1.0,
            forward_price=2500.0,  # $/tonne
            notional=25_000.0,  # kg
        )

        payoff = aluminum.payoff_terminal(2550.0)

        assert payoff > 0, "Aluminum should profit when spot > forward"

    def test_base_metal_option_copper(self):
        """Test base metal option for copper."""
        copper_call = BaseMetalOption(
            T=1.0,
            strike=4.0,
            metal_type=BaseMetalType.COPPER,
            is_call=True,
            volatility=0.22,
        )

        price = copper_call.price(spot=4.2)

        assert price > 0, "ITM copper call should have positive value"

    def test_base_metal_option_aluminum(self):
        """Test base metal option for aluminum."""
        aluminum_put = BaseMetalOption(
            T=1.0,
            strike=2600.0,
            metal_type=BaseMetalType.ALUMINUM,
            is_call=False,
            volatility=0.20,
        )

        price = aluminum_put.price(spot=2500.0)

        assert price > 0, "ITM aluminum put should have positive value"


class TestAgriculture:
    """Tests for agricultural derivatives."""

    def test_corn_future(self):
        """Test corn futures."""
        corn = CornFuture(
            T=1.0,
            forward_price=5.0,  # $/bushel
            notional=5_000.0,  # 5,000 bushels
        )

        payoff = corn.payoff_terminal(5.5)

        # 5,000 bushels × $0.50/bushel = $2,500
        assert payoff == 2500.0, "Corn future payoff should match"

    def test_wheat_future(self):
        """Test wheat futures."""
        wheat = WheatFuture(
            T=1.0,
            forward_price=6.0,
            notional=5_000.0,
        )

        payoff = wheat.payoff_terminal(5.5)

        assert payoff < 0, "Should lose when spot < forward"

    def test_soybean_future(self):
        """Test soybean futures."""
        soybeans = SoybeanFuture(
            T=1.0,
            forward_price=12.0,
            notional=5_000.0,
        )

        payoff = soybeans.payoff_terminal(12.5)

        assert payoff == 2500.0, "Soybean payoff should match"

    def test_agricultural_option_corn_call(self):
        """Test corn call option."""
        corn_call = AgriculturalOption(
            T=0.5,
            strike=5.0,
            commodity_type=AgriculturalType.CORN,
            is_call=True,
            volatility=0.30,
        )

        price = corn_call.price(spot=5.5)

        assert price > 0, "ITM corn call should have positive value"

    def test_agricultural_option_wheat_put(self):
        """Test wheat put option."""
        wheat_put = AgriculturalOption(
            T=0.5,
            strike=6.5,
            commodity_type=AgriculturalType.WHEAT,
            is_call=False,
            volatility=0.28,
        )

        price = wheat_put.price(spot=6.0)

        assert price > 0, "ITM wheat put should have positive value"

    def test_agricultural_option_soybeans(self):
        """Test soybean option."""
        soybean_call = AgriculturalOption(
            T=1.0,
            strike=12.0,
            commodity_type=AgriculturalType.SOYBEANS,
            is_call=True,
            volatility=0.25,
        )

        price = soybean_call.price(spot=13.0)

        assert price > 5000.0, "ITM soybean call should exceed intrinsic"

    def test_ag_has_higher_vol_than_metals(self):
        """Test that agricultural commodities have higher volatility."""
        corn_option = AgriculturalOption(T=1.0, strike=5.0, commodity_type=AgriculturalType.CORN)
        gold_option = GoldOption(T=1.0, strike=1800.0)

        assert (
            corn_option.volatility > gold_option.volatility
        ), "Ag should have higher vol than gold"


class TestCommodityStorage:
    """Test storage cost impacts."""

    def test_higher_storage_for_grains(self):
        """Test that grains have higher storage costs."""
        corn = CornFuture(T=1.0, forward_price=5.0)
        gold = PreciousMetalFuture(T=1.0, forward_price=1800.0)

        assert (
            corn.storage_cost > gold.storage_cost
        ), "Grains should have higher storage than gold"

    def test_convenience_yield_impact(self):
        """Test convenience yield affects forward prices."""
        base_corn = CornFuture(
            T=1.0,
            forward_price=5.0,
            convenience_yield=0.04,
        )

        forward_with_cy = base_corn.fair_forward_price(spot=5.0, risk_free_rate=0.03)

        # Net carry = r(3%) + storage(6%) - convenience(4%) = 5%
        # Forward ≈ 5.0 × exp(0.05) ≈ 5.26
        assert 5.20 < forward_with_cy < 5.30, "Forward should reflect 5% net carry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
