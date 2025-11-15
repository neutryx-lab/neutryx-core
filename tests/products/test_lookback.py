"""Tests for lookback option payoffs."""

import jax.numpy as jnp
import pytest

from neutryx.products.lookback import (
    LookbackFloatStrikeCall,
    LookbackFloatStrikePut,
    LookbackFixedStrikeCall,
    LookbackFixedStrikePut,
    LookbackPartialFixedStrikeCall,
    LookbackPartialFixedStrikePut,
)


def test_lookback_float_strike_call_uptrend():
    """Test floating strike lookback call with upward trend."""
    option = LookbackFloatStrikeCall(T=1.0)
    # Path: min = 90, terminal = 120
    path = jnp.array([100.0, 95.0, 90.0, 100.0, 110.0, 120.0])

    payoff = option.payoff_path(path)

    # Payoff = ST - min(St) = 120 - 90 = 30
    expected_payoff = 120.0 - 90.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_float_strike_call_downtrend():
    """Test floating strike lookback call with downward trend."""
    option = LookbackFloatStrikeCall(T=1.0)
    # Path: min = 80, terminal = 85
    path = jnp.array([100.0, 95.0, 90.0, 85.0, 80.0, 85.0])

    payoff = option.payoff_path(path)

    # Payoff = ST - min(St) = 85 - 80 = 5
    expected_payoff = 85.0 - 80.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_float_strike_put_downtrend():
    """Test floating strike lookback put with downward trend."""
    option = LookbackFloatStrikePut(T=1.0)
    # Path: max = 120, terminal = 90
    path = jnp.array([100.0, 110.0, 120.0, 110.0, 95.0, 90.0])

    payoff = option.payoff_path(path)

    # Payoff = max(St) - ST = 120 - 90 = 30
    expected_payoff = 120.0 - 90.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_float_strike_put_uptrend():
    """Test floating strike lookback put with upward trend."""
    option = LookbackFloatStrikePut(T=1.0)
    # Path: max = 120, terminal = 115
    path = jnp.array([100.0, 105.0, 110.0, 115.0, 120.0, 115.0])

    payoff = option.payoff_path(path)

    # Payoff = max(St) - ST = 120 - 115 = 5
    expected_payoff = 120.0 - 115.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_fixed_strike_call_itm():
    """Test fixed strike lookback call in-the-money."""
    option = LookbackFixedStrikeCall(K=100.0, T=1.0)
    # Path: max = 130
    path = jnp.array([100.0, 110.0, 120.0, 130.0, 125.0, 120.0])

    payoff = option.payoff_path(path)

    # Payoff = max(max(St) - K, 0) = max(130 - 100, 0) = 30
    expected_payoff = 130.0 - 100.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_fixed_strike_call_otm():
    """Test fixed strike lookback call out-of-the-money."""
    option = LookbackFixedStrikeCall(K=150.0, T=1.0)
    # Path: max = 130
    path = jnp.array([100.0, 110.0, 120.0, 130.0, 125.0, 120.0])

    payoff = option.payoff_path(path)

    # Payoff = max(130 - 150, 0) = 0
    assert payoff == 0.0


def test_lookback_fixed_strike_put_itm():
    """Test fixed strike lookback put in-the-money."""
    option = LookbackFixedStrikePut(K=100.0, T=1.0)
    # Path: min = 80
    path = jnp.array([100.0, 95.0, 90.0, 85.0, 80.0, 85.0])

    payoff = option.payoff_path(path)

    # Payoff = max(K - min(St), 0) = max(100 - 80, 0) = 20
    expected_payoff = 100.0 - 80.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_fixed_strike_put_otm():
    """Test fixed strike lookback put out-of-the-money."""
    option = LookbackFixedStrikePut(K=70.0, T=1.0)
    # Path: min = 80
    path = jnp.array([100.0, 95.0, 90.0, 85.0, 80.0, 85.0])

    payoff = option.payoff_path(path)

    # Payoff = max(70 - 80, 0) = 0
    assert payoff == 0.0


def test_lookback_partial_fixed_call_full_observation():
    """Test partial lookback call with full observation period."""
    option = LookbackPartialFixedStrikeCall(K=100.0, T=1.0, observation_start=0.0)
    # Same as regular fixed strike with observation_start=0
    path = jnp.array([100.0, 110.0, 120.0, 130.0, 125.0, 120.0])

    payoff = option.payoff_path(path)

    # Payoff = max(130 - 100, 0) = 30
    expected_payoff = 130.0 - 100.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_partial_fixed_call_partial_observation():
    """Test partial lookback call with partial observation period."""
    option = LookbackPartialFixedStrikeCall(K=100.0, T=1.0, observation_start=0.5)
    # Observation starts at 50% of path
    # Path: [100, 110, 120, | 130, 125, 120]
    #                        ^ observation starts here (idx=3)
    path = jnp.array([100.0, 110.0, 120.0, 130.0, 125.0, 120.0])

    payoff = option.payoff_path(path)

    # Max from idx 3 onwards: max(130, 125, 120) = 130
    # Payoff = max(130 - 100, 0) = 30
    expected_payoff = 130.0 - 100.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_partial_fixed_put_partial_observation():
    """Test partial lookback put with partial observation period."""
    option = LookbackPartialFixedStrikePut(K=100.0, T=1.0, observation_start=0.5)
    # Observation starts at 50% of path
    # Path: [100, 95, 90, | 85, 80, 85]
    #                      ^ observation starts here
    path = jnp.array([100.0, 95.0, 90.0, 85.0, 80.0, 85.0])

    payoff = option.payoff_path(path)

    # Min from idx 3 onwards: min(85, 80, 85) = 80
    # Payoff = max(100 - 80, 0) = 20
    expected_payoff = 100.0 - 80.0
    assert jnp.isclose(payoff, expected_payoff)


def test_lookback_always_positive_payoff():
    """Test that floating strike lookback options always have non-negative payoff."""
    # Floating strike call
    call_option = LookbackFloatStrikeCall(T=1.0)
    # Any path should give non-negative payoff
    path = jnp.array([100.0, 95.0, 90.0, 85.0, 80.0, 75.0])
    call_payoff = call_option.payoff_path(path)
    assert call_payoff >= 0.0

    # Floating strike put
    put_option = LookbackFloatStrikePut(T=1.0)
    put_payoff = put_option.payoff_path(path)
    assert put_payoff >= 0.0


def test_lookback_constant_path():
    """Test lookback options with constant path."""
    # Floating strike call with constant path
    call_option = LookbackFloatStrikeCall(T=1.0)
    path = jnp.array([100.0, 100.0, 100.0, 100.0, 100.0])
    call_payoff = call_option.payoff_path(path)
    # ST - min = 100 - 100 = 0
    assert call_payoff == 0.0

    # Floating strike put with constant path
    put_option = LookbackFloatStrikePut(T=1.0)
    put_payoff = put_option.payoff_path(path)
    # max - ST = 100 - 100 = 0
    assert put_payoff == 0.0


def test_lookback_requires_path():
    """Test that lookback options require full path."""
    option = LookbackFloatStrikeCall(T=1.0)
    assert option.requires_path is True
    assert option.supports_pde is False


def test_lookback_single_point_path():
    """Test lookback option with single point path (edge case)."""
    option = LookbackFloatStrikeCall(T=1.0)
    path = jnp.array([110.0])

    payoff = option.payoff_path(path)

    # ST - min = 110 - 110 = 0
    assert payoff == 0.0


def test_lookback_fixed_vs_float_relationship():
    """Test relationship between fixed and floating strike lookbacks."""
    path = jnp.array([100.0, 110.0, 120.0, 130.0, 125.0, 120.0])

    # Floating strike call payoff
    float_call = LookbackFloatStrikeCall(T=1.0)
    float_payoff = float_call.payoff_path(path)
    # = ST - min = 120 - 100 = 20

    # Fixed strike call with K = min(path)
    fixed_call = LookbackFixedStrikeCall(K=100.0, T=1.0)
    fixed_payoff = fixed_call.payoff_path(path)
    # = max - K = 130 - 100 = 30

    # Fixed strike should be higher (includes the max, not just terminal)
    assert fixed_payoff >= float_payoff
