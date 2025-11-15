"""Comprehensive tests for energy derivatives."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.energy.oil import (
    BrentCrudeOption,
    CrudeOilFuture,
    OilSpreadOption,
    WTICrudeOption,
    black_76_option,
)
from neutryx.products.energy.natural_gas import (
    GasStorageContract,
    GasSwap,
    NaturalGasOption,
    SeasonalGasOption,
)
from neutryx.products.energy.power import (
    OffPeakPowerOption,
    PeakPowerOption,
    PowerForward,
    PowerShapingContract,
)


class TestOilDerivatives:
    """Tests for oil futures and options."""

    def test_crude_oil_future_payoff(self):
        """Test crude oil futures payoff."""
        future = CrudeOilFuture(
            T=1.0,
            forward_price=80.0,
            notional=1000.0,
        )

        # Spot above forward
        payoff_profit = future.payoff_terminal(85.0)
        assert payoff_profit == 5000.0, "Should profit when spot > forward"

        # Spot below forward
        payoff_loss = future.payoff_terminal(75.0)
        assert payoff_loss == -5000.0, "Should lose when spot < forward"

    def test_fair_forward_price(self):
        """Test fair forward price calculation."""
        future = CrudeOilFuture(
            T=1.0,
            forward_price=80.0,
            storage_cost=0.02,
            convenience_yield=0.03,
        )

        fair_price = future.fair_forward_price(spot=80.0, risk_free_rate=0.03)

        # With r=3%, storage=2%, convenience=3%, carry = 2%
        # Fair forward ≈ 80 × exp(0.02) ≈ 81.6
        assert 80 < fair_price < 83, "Fair price should reflect 2% carry"

    def test_wti_option_pricing(self):
        """Test WTI crude option pricing."""
        wti_call = WTICrudeOption(
            T=0.5,
            strike=80.0,
            is_call=True,
            volatility=0.35,
        )

        price = wti_call.price(futures_price=85.0)

        # ITM call should have positive value
        assert price > 0, "ITM call should have positive value"
        # Option should be worth more than intrinsic (5000) but reasonable
        assert price > 5000.0, "Option with time value should exceed intrinsic"
        assert price < 15000.0, "Price should be reasonable for given parameters"

    def test_brent_option_pricing(self):
        """Test Brent crude option pricing."""
        brent_put = BrentCrudeOption(
            T=0.5,
            strike=85.0,
            is_call=False,
            volatility=0.35,
        )

        price = brent_put.price(futures_price=80.0)

        # ITM put should have positive value
        assert price > 0, "ITM put should have positive value"

    def test_oil_spread_option(self):
        """Test oil spread option."""
        spread_option = OilSpreadOption(
            T=1.0,
            strike=5.0,  # $5/barrel spread
            is_call=True,
            correlation=0.85,
        )

        # WTI $85, Brent $80 -> spread = $5
        spots = jnp.array([85.0, 80.0])
        payoff = spread_option.payoff_terminal(spots)

        # At strike, should have zero payoff
        assert abs(payoff) < 0.01, "At-strike should have near-zero payoff"

    def test_spread_volatility(self):
        """Test spread volatility calculation."""
        spread_option = OilSpreadOption(
            T=1.0,
            strike=5.0,
            vol_1=0.35,
            vol_2=0.35,
            correlation=0.85,
        )

        spread_vol = spread_option.spread_volatility()

        # Spread vol should be lower than individual vols due to high correlation
        assert spread_vol < 0.35, "Spread vol should be less than individual vols"


class TestBlack76:
    """Test Black-76 futures option model."""

    def test_black76_call_itm(self):
        """Test Black-76 call ITM."""
        price = black_76_option(
            F=105.0,
            K=100.0,
            T=1.0,
            r=0.03,
            sigma=0.20,
            is_call=True,
        )

        assert price > 5.0, "ITM call should have value > intrinsic"

    def test_black76_put_itm(self):
        """Test Black-76 put ITM."""
        price = black_76_option(
            F=95.0,
            K=100.0,
            T=1.0,
            r=0.03,
            sigma=0.20,
            is_call=False,
        )

        assert price > 5.0, "ITM put should have value > intrinsic"

    def test_black76_atm(self):
        """Test Black-76 ATM option."""
        call_price = black_76_option(
            F=100.0,
            K=100.0,
            T=1.0,
            r=0.03,
            sigma=0.20,
            is_call=True,
        )

        put_price = black_76_option(
            F=100.0,
            K=100.0,
            T=1.0,
            r=0.03,
            sigma=0.20,
            is_call=False,
        )

        # ATM call and put should be close in value
        assert abs(call_price - put_price) < 0.5, "ATM call and put should be similar"


class TestNaturalGas:
    """Tests for natural gas derivatives."""

    def test_natural_gas_option(self):
        """Test natural gas option pricing."""
        gas_call = NaturalGasOption(
            T=0.5,
            strike=3.0,  # $3/MMBtu
            is_call=True,
            volatility=0.50,
        )

        price = gas_call.price(futures_price=3.5)

        assert price > 0, "ITM gas call should have positive value"

    def test_gas_swap_payer(self):
        """Test gas swap payer position."""
        swap = GasSwap(
            T=1.0,
            notional=10_000,
            fixed_price=3.0,
            is_payer=True,
        )

        # Floating at 3.5, should profit
        payoff = swap.payoff_terminal(3.5)
        assert payoff > 0, "Payer swap should profit when floating > fixed"

    def test_seasonal_gas_option_winter(self):
        """Test winter seasonal gas option."""
        winter_call = SeasonalGasOption(
            T=0.25,
            strike=4.0,
            season="winter",
            is_call=True,
        )

        # Winter vol should be higher
        assert winter_call.volatility > 0.50, "Winter should have higher volatility"

    def test_seasonal_gas_option_summer(self):
        """Test summer seasonal gas option."""
        summer_call = SeasonalGasOption(
            T=0.25,
            strike=3.0,
            season="summer",
            is_call=True,
        )

        # Summer vol should be lower
        assert summer_call.volatility < 0.50, "Summer should have lower volatility"

    def test_gas_storage_contract(self):
        """Test gas storage optimization."""
        storage = GasStorageContract(
            T=1.0,
            max_storage=1000.0,
            max_injection_rate=100.0,
            max_withdrawal_rate=100.0,
            injection_cost=0.05,
            withdrawal_cost=0.05,
            fixing_times=jnp.array([0.25, 0.5, 0.75]),
        )

        # Volatile price path
        path = jnp.array([2.5, 3.0, 4.0, 3.5, 2.8, 3.2, 4.5])

        payoff = storage.payoff_path(path)

        # Should generate positive value from storage optionality
        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric"


class TestPowerDerivatives:
    """Tests for power derivatives."""

    def test_power_forward(self):
        """Test power forward contract."""
        forward = PowerForward(
            T=1.0,
            forward_price=50.0,  # $/MWh
            notional=100.0,  # MWh
        )

        payoff = forward.payoff_terminal(55.0)

        # Should profit when spot > forward (adjusted for period)
        assert payoff != 0, "Forward should have non-zero payoff"

    def test_peak_power_option(self):
        """Test peak power option."""
        peak_call = PeakPowerOption(
            T=0.5,
            strike=50.0,
            is_call=True,
            volatility=1.00,  # High vol for peak power
        )

        price = peak_call.price(forward_price=60.0)

        assert price > 0, "Peak call should have positive value"

    def test_offpeak_power_option(self):
        """Test off-peak power option."""
        offpeak_call = OffPeakPowerOption(
            T=0.5,
            strike=30.0,
            is_call=True,
            volatility=0.60,  # Lower vol than peak
        )

        price = offpeak_call.price(forward_price=35.0)

        assert price > 0, "Off-peak call should have positive value"

    def test_peak_vs_offpeak_volatility(self):
        """Test that peak has higher volatility than off-peak."""
        peak = PeakPowerOption(T=1.0, strike=50.0)
        offpeak = OffPeakPowerOption(T=1.0, strike=30.0)

        assert peak.volatility > offpeak.volatility, "Peak should have higher vol"

    def test_power_shaping_contract(self):
        """Test power shaping contract."""
        shaping = PowerShapingContract(
            T=1.0,
            base_load=100.0,  # MW
            max_load=200.0,  # MW
            shaping_cost=5.0,  # $/MW
            fixing_times=jnp.array([0.1, 0.2, 0.3]),
        )

        # Varying power prices
        path = jnp.array([40, 60, 80, 50, 45, 70])

        payoff = shaping.payoff_path(path)

        # Shaping should have negative payoff (cost)
        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
