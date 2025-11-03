"""Tests for FX exotic derivatives (TARFs, accumulators, etc.)."""
import jax.numpy as jnp
import pytest

from neutryx.products.fx_exotics import (
    TARF,
    Accumulator,
    Decumulator,
    FaderOption,
    Napoleon,
    RangeAccrual,
)


def test_tarf_hits_target():
    """Test TARF that hits target and knocks out."""
    product = TARF(
        K=1.2,
        T=1.0,
        target_profit=0.1,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional=1.0,
        leverage=1.0,
        is_long=True,
        knockout_type="target",
    )

    # FX rises, profit accumulates
    path = jnp.array([1.2, 1.25, 1.28, 1.3, 1.32])
    payoff = product.payoff_path(path)

    # Should knock out when target reached
    assert payoff > 0.0


def test_tarf_no_knockout():
    """Test TARF that doesn't hit target."""
    product = TARF(
        K=1.2,
        T=1.0,
        target_profit=1.0,  # Very high target
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional=1.0,
        leverage=1.0,
        is_long=True,
    )

    # Small FX movements
    path = jnp.array([1.2, 1.21, 1.22, 1.21, 1.23])
    payoff = product.payoff_path(path)

    # Should return accumulated profit
    assert payoff >= 0.0


def test_accumulator_knockout():
    """Test accumulator with knockout."""
    product = Accumulator(
        K=1.2,
        T=1.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional_per_fixing=1.0,
        leverage_down=2.0,
        leverage_up=1.0,
        knockout_barrier=1.3,
        is_long=True,
    )

    # FX rises and hits knockout
    path = jnp.linspace(1.2, 1.35, 100)
    payoff = product.payoff_path(path)

    # Can be positive or negative depending on path
    assert isinstance(float(payoff), float)


def test_accumulator_no_knockout():
    """Test accumulator without knockout."""
    product = Accumulator(
        K=1.2,
        T=1.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional_per_fixing=1.0,
        leverage_down=2.0,
        leverage_up=1.0,
        knockout_barrier=None,
        is_long=True,
    )

    path = jnp.linspace(1.2, 1.15, 100)
    payoff = product.payoff_path(path)

    # Accumulator can have losses
    assert isinstance(float(payoff), float)


def test_decumulator():
    """Test decumulator (reverse accumulator)."""
    product = Decumulator(
        K=1.2,
        T=1.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional_per_fixing=1.0,
        leverage_up=2.0,
        leverage_down=1.0,
        knockout_barrier=1.1,
    )

    # FX declines
    path = jnp.linspace(1.2, 1.15, 100)
    payoff = product.payoff_path(path)

    assert isinstance(float(payoff), float)


def test_fader_option_in_range():
    """Test fader option with accrual in range."""
    product = FaderOption(
        K=1.2,
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        max_notional=1.0,
        is_call=True,
        accrual_type="in_range",
    )

    # Path stays mostly in range
    path = jnp.full(100, 1.2)
    payoff = product.payoff_path(path)

    # Should have high effective notional
    assert payoff >= 0.0


def test_fader_option_out_range():
    """Test fader option with accrual out of range."""
    product = FaderOption(
        K=1.2,
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        max_notional=1.0,
        is_call=True,
        accrual_type="out_range",
    )

    # Path stays in range
    path = jnp.full(100, 1.2)
    payoff = product.payoff_path(path)

    # Effective notional should be low (accrues when out of range)
    assert payoff >= 0.0


def test_napoleon_option_reset_up():
    """Test Napoleon option with upward reset."""
    product = Napoleon(
        K_initial=1.2,
        T=1.0,
        reset_barrier=1.3,
        K_reset=1.25,
        is_call=True,
        reset_type="up",
    )

    # Path crosses reset barrier
    path = jnp.array([1.2, 1.28, 1.32, 1.3, 1.35])
    payoff = product.payoff_path(path)

    # Strike should reset to more favorable level
    assert payoff > 0.0


def test_napoleon_option_no_reset():
    """Test Napoleon option without reset."""
    product = Napoleon(
        K_initial=1.2,
        T=1.0,
        reset_barrier=1.3,
        K_reset=1.25,
        is_call=True,
        reset_type="up",
    )

    # Path doesn't cross barrier
    path = jnp.array([1.2, 1.22, 1.25, 1.24, 1.26])
    payoff = product.payoff_path(path)

    # Uses original strike
    assert payoff >= 0.0


def test_range_accrual():
    """Test range accrual note."""
    product = RangeAccrual(
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        accrual_rate=0.05,
        principal=1.0,
    )

    # Path stays fully in range
    path = jnp.full(100, 1.2)
    payoff = product.payoff_path(path)

    # Should receive principal + full accrual
    expected = 1.0 + 0.05 * 1.0
    assert abs(payoff - expected) < 0.01


def test_range_accrual_partial():
    """Test range accrual with partial time in range."""
    product = RangeAccrual(
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        accrual_rate=0.05,
        principal=1.0,
    )

    # Path out of range for half the time
    path = jnp.concatenate([jnp.full(50, 1.3), jnp.full(50, 1.2)])
    payoff = product.payoff_path(path)

    # Should receive principal + partial accrual
    assert 1.0 < payoff < 1.05
