"""Tests for quanto products."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.fx_complex.quanto import (
    QuantoDrift,
    QuantoForward,
    QuantoOption,
    QuantoSwap,
    quanto_drift_adjustment,
    quanto_option_delta,
)


class TestQuantoOption:
    """Test cases for Quanto Options."""

    def test_quanto_call_pricing(self):
        """Test quanto call option pricing."""
        quanto_call = QuantoOption(
            T=1.0,
            strike=100.0,
            is_call=True,
            domestic_rate=0.03,
            foreign_rate=0.02,
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.5,
            fixed_fx_rate=1.0,
        )

        price = quanto_call.price(spot=105.0)

        assert price > 0, "Quanto call should have positive value"
        assert price < 105.0, "Price should be less than spot for ATM option"

    def test_quanto_put_pricing(self):
        """Test quanto put option pricing."""
        quanto_put = QuantoOption(
            T=1.0,
            strike=100.0,
            is_call=False,
            domestic_rate=0.03,
            foreign_rate=0.02,
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.5,
        )

        price = quanto_put.price(spot=95.0)

        assert price > 0, "Quanto put should have positive value"

    def test_quanto_adjustment_impact(self):
        """Test that correlation impacts quanto pricing."""
        # Positive correlation
        quanto_pos_corr = QuantoOption(
            T=1.0,
            strike=100.0,
            is_call=True,
            domestic_rate=0.03,
            foreign_rate=0.02,
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.8,
        )

        # Negative correlation
        quanto_neg_corr = QuantoOption(
            T=1.0,
            strike=100.0,
            is_call=True,
            domestic_rate=0.03,
            foreign_rate=0.02,
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=-0.8,
        )

        price_pos = quanto_pos_corr.price(spot=100.0)
        price_neg = quanto_neg_corr.price(spot=100.0)

        # Prices should differ due to correlation
        assert abs(price_pos - price_neg) > 0.01, "Correlation should affect price"

    def test_quanto_payoff_terminal(self):
        """Test quanto option payoff calculation."""
        quanto = QuantoOption(
            T=1.0,
            strike=100.0,
            is_call=True,
            fixed_fx_rate=1.5,
        )

        # ITM payoff
        payoff_itm = quanto.payoff_terminal(110.0)
        assert payoff_itm == 1.5 * 10.0, "Should be (spot-strike) * fx_rate"

        # OTM payoff
        payoff_otm = quanto.payoff_terminal(95.0)
        assert payoff_otm == 0.0, "OTM option should have zero payoff"


class TestQuantoForward:
    """Test cases for Quanto Forwards."""

    def test_quanto_forward_payoff(self):
        """Test quanto forward payoff."""
        quanto_fwd = QuantoForward(
            T=1.0,
            forward_price=100.0,
            fixed_fx_rate=1.2,
            is_long=True,
        )

        # Spot above forward
        payoff_profit = quanto_fwd.payoff_terminal(105.0)
        assert payoff_profit > 0, "Long forward should profit when spot > forward"
        assert payoff_profit == 1.2 * 5.0, "Payoff should be (spot-fwd) * fx_rate"

        # Spot below forward
        payoff_loss = quanto_fwd.payoff_terminal(95.0)
        assert payoff_loss < 0, "Long forward should lose when spot < forward"

    def test_fair_forward_price_quanto_adjustment(self):
        """Test fair forward price with quanto adjustment."""
        quanto_fwd = QuantoForward(
            T=1.0,
            forward_price=100.0,
            domestic_rate=0.03,
            foreign_rate=0.02,
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.5,
        )

        fair_fwd = quanto_fwd.fair_forward_price(spot=100.0)

        # Fair forward should be above spot due to rate differential and quanto adjustment
        assert fair_fwd > 100.0, "Fair forward should exceed spot with positive rates"

    def test_quanto_adjustment_direction(self):
        """Test that correlation affects forward price direction."""
        params = {
            "T": 1.0,
            "forward_price": 100.0,
            "domestic_rate": 0.03,
            "foreign_rate": 0.02,
            "asset_vol": 0.20,
            "fx_vol": 0.10,
        }

        # Positive correlation
        fwd_pos = QuantoForward(**params, correlation=0.8)
        fair_pos = fwd_pos.fair_forward_price(100.0)

        # Negative correlation
        fwd_neg = QuantoForward(**params, correlation=-0.8)
        fair_neg = fwd_neg.fair_forward_price(100.0)

        # Positive correlation should increase forward price
        assert fair_pos > fair_neg, "Positive correlation increases forward price"


class TestQuantoSwap:
    """Test cases for Quanto Swaps."""

    def test_quanto_swap_payer(self):
        """Test payer quanto swap."""
        quanto_swap = QuantoSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.04,
            payment_frequency=4,
            is_payer=True,
        )

        pv = quanto_swap.payoff_terminal(0.0)

        assert isinstance(pv, (float, jnp.ndarray)), "Should return numeric value"

    def test_quanto_swap_receiver(self):
        """Test receiver quanto swap."""
        quanto_swap = QuantoSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.04,
            payment_frequency=4,
            is_payer=False,
        )

        pv = quanto_swap.payoff_terminal(0.0)

        assert isinstance(pv, (float, jnp.ndarray)), "Should return numeric value"

    def test_quanto_swap_direction(self):
        """Test that payer and receiver swaps have opposite signs."""
        base_params = {
            "T": 5.0,
            "notional": 1_000_000,
            "fixed_rate": 0.04,
            "domestic_rate": 0.03,
            "foreign_rate": 0.05,
        }

        payer = QuantoSwap(**base_params, is_payer=True)
        receiver = QuantoSwap(**base_params, is_payer=False)

        pv_payer = payer.payoff_terminal(0.0)
        pv_receiver = receiver.payoff_terminal(0.0)

        # Should have opposite signs (approximately)
        assert pv_payer * pv_receiver <= 0, "Payer and receiver should have opposite signs"


class TestQuantoUtilities:
    """Test utility functions for quanto products."""

    def test_quanto_drift_adjustment_calculation(self):
        """Test quanto drift adjustment formula."""
        adjustment = quanto_drift_adjustment(
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.5,
        )

        # Expected: 0.5 * 0.20 * 0.10 = 0.01
        expected = 0.5 * 0.20 * 0.10

        assert abs(adjustment - expected) < 1e-6, "Drift adjustment should match formula"

    def test_quanto_drift_zero_correlation(self):
        """Test zero drift adjustment with zero correlation."""
        adjustment = quanto_drift_adjustment(
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.0,
        )

        assert adjustment == 0.0, "Zero correlation should give zero adjustment"

    def test_quanto_option_delta(self):
        """Test quanto option delta calculation."""
        delta_call = quanto_option_delta(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            domestic_rate=0.03,
            foreign_rate=0.02,
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.5,
            is_call=True,
        )

        # ATM call delta should be around 0.5 (adjusted for quanto)
        assert 0.3 < delta_call < 0.7, "ATM call delta should be around 0.5"

        # Put delta should be negative
        delta_put = quanto_option_delta(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            domestic_rate=0.03,
            foreign_rate=0.02,
            asset_vol=0.20,
            fx_vol=0.10,
            correlation=0.5,
            is_call=False,
        )

        assert delta_put < 0, "Put delta should be negative"


class TestQuantoDriftHelper:
    """Test QuantoDrift helper class."""

    def test_quanto_drift_calculate(self):
        """Test QuantoDrift.calculate_adjustment method."""
        adjustment = QuantoDrift.calculate_adjustment(
            asset_vol=0.20,
            fx_vol=0.15,
            correlation=-0.6,
        )

        expected = -0.6 * 0.20 * 0.15

        assert abs(adjustment - expected) < 1e-6, "Should match formula"

    def test_quanto_vs_vanilla_spread(self):
        """Test quanto vs vanilla spread calculation."""
        spread = QuantoDrift.quanto_vs_vanilla_spread(
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            rates=(0.03, 0.02),
            vols=(0.20, 0.10),
            correlation=0.5,
        )

        # Spread should be non-zero when correlation is non-zero
        assert spread != 0.0, "Quanto spread should be non-zero with correlation"

    def test_implied_correlation_bounds(self):
        """Test that implied correlation stays in valid range."""
        implied_corr = QuantoDrift.implied_correlation(
            market_quanto_price=10.5,
            theoretical_price_no_quanto=10.0,
            asset_vol=0.20,
            fx_vol=0.10,
            sensitivity=1.0,
        )

        # Should be clipped to [-1, 1]
        assert -1.0 <= implied_corr <= 1.0, "Correlation should be in valid range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
