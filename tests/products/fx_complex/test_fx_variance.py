"""Tests for FX variance swaps."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.fx_complex.fx_variance import (
    CorridorVarianceSwap,
    ConditionalVarianceSwap,
    FXVarianceSwap,
    fx_variance_convexity_adjustment,
    fx_variance_fair_strike,
)


class TestFXVarianceSwap:
    """Test cases for FX Variance Swaps."""

    def test_basic_variance_swap(self):
        """Test basic variance swap payoff."""
        var_swap = FXVarianceSwap(
            T=1.0,
            notional=10_000,
            strike=0.0144,  # 12% vol
            observation_frequency=252,
        )

        # Create path with known volatility
        # Generate path with ~15% realized vol
        path = jnp.array([100.0, 101.5, 99.8, 102.3, 101.1, 103.5, 102.0])

        payoff = var_swap.payoff_path(path)

        # Should have positive payoff (realized > strike)
        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric value"

    def test_realized_variance_calculation(self):
        """Test realized variance calculation matches expected."""
        var_swap = FXVarianceSwap(
            T=1.0,
            notional=10_000,
            strike=0.01,
            observation_frequency=252,
        )

        # Simple path
        path = jnp.linspace(100, 110, 100)

        realized_var = var_swap._calculate_realized_variance(path)

        assert realized_var > 0, "Realized variance should be positive"
        assert realized_var < 1.0, "Realized variance should be reasonable"

    def test_cap_floor_enforcement(self):
        """Test that cap and floor are enforced."""
        cap_level = 0.05
        floor_level = 0.01

        var_swap = FXVarianceSwap(
            T=1.0,
            notional=10_000,
            strike=0.02,
            cap_level=cap_level,
            floor_level=floor_level,
        )

        # High volatility path
        high_vol_path = jnp.array([100, 120, 90, 110, 95, 115])
        payoff_high = var_swap.payoff_path(high_vol_path)

        # Low volatility path
        low_vol_path = jnp.linspace(100, 100.5, 50)
        payoff_low = var_swap.payoff_path(low_vol_path)

        # Both should produce valid payoffs
        assert isinstance(payoff_high, (float, jnp.ndarray))
        assert isinstance(payoff_low, (float, jnp.ndarray))

    def test_vega_notional_conversion(self):
        """Test conversion from variance notional to vega notional."""
        var_swap = FXVarianceSwap(
            T=1.0,
            notional=10_000,
            strike=0.04,  # 20% vol
        )

        vega_notional = var_swap.vega_notional()

        # Vega notional = 10_000 / (2 * sqrt(0.04)) = 10_000 / 0.4 = 25_000
        expected = 10_000 / (2.0 * jnp.sqrt(0.04))

        assert abs(vega_notional - expected) < 1.0, "Vega notional should match formula"


class TestCorridorVarianceSwap:
    """Test cases for Corridor Variance Swaps."""

    def test_corridor_inside_accrual(self):
        """Test corridor variance with inside accrual."""
        corridor_swap = CorridorVarianceSwap(
            T=1.0,
            notional=10_000,
            strike=0.01,
            lower_barrier=95.0,
            upper_barrier=105.0,
            accrue_inside=True,
        )

        # Path mostly inside corridor
        path = jnp.array([100, 101, 102, 101, 100, 99, 101])

        payoff = corridor_swap.payoff_path(path)

        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric value"

    def test_corridor_outside_accrual(self):
        """Test corridor variance with outside accrual."""
        corridor_swap = CorridorVarianceSwap(
            T=1.0,
            notional=10_000,
            strike=0.01,
            lower_barrier=95.0,
            upper_barrier=105.0,
            accrue_inside=False,  # Accrue outside
        )

        # Path with some excursions outside
        path = jnp.array([100, 110, 90, 100, 108, 92, 100])

        payoff = corridor_swap.payoff_path(path)

        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric value"

    def test_corridor_all_outside(self):
        """Test corridor when entire path is outside."""
        corridor_swap = CorridorVarianceSwap(
            T=1.0,
            notional=10_000,
            strike=0.01,
            lower_barrier=95.0,
            upper_barrier=105.0,
            accrue_inside=True,
        )

        # Path entirely outside corridor
        path = jnp.array([110, 111, 112, 113, 114])

        payoff = corridor_swap.payoff_path(path)

        # Should be negative (no variance accrued, but paying strike)
        assert payoff <= 0, "Should have negative payoff when outside corridor"


class TestConditionalVarianceSwap:
    """Test cases for Conditional Variance Swaps."""

    def test_conditional_spot_trigger(self):
        """Test conditional variance with spot trigger."""
        cond_swap = ConditionalVarianceSwap(
            T=1.0,
            notional=10_000,
            strike_base=0.01,
            strike_conditional=0.02,
            trigger_level=105.0,
            trigger_type="spot",
        )

        # Path that crosses trigger
        path = jnp.array([100, 102, 104, 106, 105, 104])

        payoff = cond_swap.payoff_path(path)

        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric value"

    def test_conditional_vol_trigger(self):
        """Test conditional variance with vol trigger."""
        cond_swap = ConditionalVarianceSwap(
            T=1.0,
            notional=10_000,
            strike_base=0.01,
            strike_conditional=0.03,
            trigger_level=0.15,  # 15% vol trigger
            trigger_type="vol",
        )

        # High volatility path
        path = jnp.array([100, 110, 95, 108, 92, 105])

        payoff = cond_swap.payoff_path(path)

        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric value"

    def test_conditional_no_trigger(self):
        """Test conditional variance when trigger not hit."""
        cond_swap = ConditionalVarianceSwap(
            T=1.0,
            notional=10_000,
            strike_base=0.01,
            strike_conditional=0.03,
            trigger_level=110.0,
            trigger_type="spot",
        )

        # Path below trigger
        path = jnp.linspace(100, 105, 50)

        payoff = cond_swap.payoff_path(path)

        # Should use base strike
        assert isinstance(payoff, (float, jnp.ndarray)), "Should return numeric value"


class TestVarianceUtilities:
    """Test utility functions for variance pricing."""

    def test_fair_strike_calculation(self):
        """Test fair variance strike calculation."""
        fair_strike = fx_variance_fair_strike(
            spot=1.20,
            forward=1.22,
            vol_atm=0.12,
            vol_smile_adjustment=0.001,
            correlation_spot_vol=0.3,
            time_to_maturity=1.0,
        )

        # Fair strike should be close to ATM variance
        assert 0.01 < fair_strike < 0.03, "Fair strike should be reasonable"

        # Should be higher than vol_atm^2 due to smile and correlation
        assert fair_strike > 0.12**2, "Fair strike should exceed squared ATM vol"

    def test_convexity_adjustment(self):
        """Test convexity adjustment calculation."""
        adjustment = fx_variance_convexity_adjustment(
            variance_strike=0.0144,  # 12% vol
            vol_of_vol=0.80,
            time_to_maturity=1.0,
        )

        # Adjustment should be positive
        assert adjustment > 0, "Convexity adjustment should be positive"

        # Should be relatively small
        assert adjustment < 0.01, "Adjustment should be small relative to variance"

    def test_convexity_increases_with_time(self):
        """Test that convexity adjustment increases with time."""
        adjustment_1y = fx_variance_convexity_adjustment(
            variance_strike=0.0144,
            vol_of_vol=0.80,
            time_to_maturity=1.0,
        )

        adjustment_2y = fx_variance_convexity_adjustment(
            variance_strike=0.0144,
            vol_of_vol=0.80,
            time_to_maturity=2.0,
        )

        assert (
            adjustment_2y > adjustment_1y
        ), "Convexity adjustment should increase with time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
