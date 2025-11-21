"""Tests for Asian option payoffs."""

import jax.numpy as jnp
import pytest

from neutryx.products.asian import (
    AsianArithmetic,
    AsianGeometric,
    AsianArithmeticFloatingStrike,
    AsianGeometricFloatingStrike,
)


def test_asian_arithmetic_call_itm():
    """Test arithmetic Asian call in-the-money."""
    option = AsianArithmetic(K=100.0, T=1.0, is_call=True)
    # Path with average > strike
    path = jnp.array([100.0, 105.0, 110.0, 115.0, 120.0])

    payoff = option.payoff_path(path)

    # Average = 110, Strike = 100, Payoff = 10
    avg = path.mean()
    expected_payoff = avg - 100.0
    assert jnp.isclose(payoff, expected_payoff)


def test_asian_arithmetic_call_otm():
    """Test arithmetic Asian call out-of-the-money."""
    option = AsianArithmetic(K=100.0, T=1.0, is_call=True)
    # Path with average < strike
    path = jnp.array([90.0, 92.0, 88.0, 85.0, 95.0])

    payoff = option.payoff_path(path)

    # Average = 90, Strike = 100, Payoff = 0
    assert payoff == 0.0


def test_asian_arithmetic_put_itm():
    """Test arithmetic Asian put in-the-money."""
    option = AsianArithmetic(K=100.0, T=1.0, is_call=False)
    # Path with average < strike
    path = jnp.array([90.0, 92.0, 88.0, 85.0, 95.0])

    payoff = option.payoff_path(path)

    # Average = 90, Strike = 100, Payoff = 10
    avg = path.mean()
    expected_payoff = 100.0 - avg
    assert jnp.isclose(payoff, expected_payoff)


def test_asian_arithmetic_put_otm():
    """Test arithmetic Asian put out-of-the-money."""
    option = AsianArithmetic(K=100.0, T=1.0, is_call=False)
    # Path with average > strike
    path = jnp.array([100.0, 105.0, 110.0, 115.0, 120.0])

    payoff = option.payoff_path(path)

    # Average = 110, Strike = 100, Payoff = 0
    assert payoff == 0.0


def test_asian_geometric_call_itm():
    """Test geometric Asian call in-the-money."""
    option = AsianGeometric(K=100.0, T=1.0, is_call=True)
    # Path with geometric average > strike
    path = jnp.array([100.0, 105.0, 110.0, 115.0, 120.0])

    payoff = option.payoff_path(path)

    # Geometric mean = exp(mean(log(prices)))
    geometric_avg = jnp.exp(jnp.log(path).mean())
    expected_payoff = geometric_avg - 100.0
    assert jnp.isclose(payoff, expected_payoff, rtol=1e-5)


def test_asian_geometric_call_otm():
    """Test geometric Asian call out-of-the-money."""
    option = AsianGeometric(K=110.0, T=1.0, is_call=True)
    # Path with geometric average < strike
    path = jnp.array([90.0, 92.0, 88.0, 85.0, 95.0])

    payoff = option.payoff_path(path)

    # Geometric average < 110, Payoff = 0
    assert payoff == 0.0


def test_asian_geometric_put_itm():
    """Test geometric Asian put in-the-money."""
    option = AsianGeometric(K=100.0, T=1.0, is_call=False)
    # Path with geometric average < strike
    path = jnp.array([90.0, 92.0, 88.0, 85.0, 95.0])

    payoff = option.payoff_path(path)

    # Geometric mean
    geometric_avg = jnp.exp(jnp.log(path).mean())
    expected_payoff = 100.0 - geometric_avg
    assert jnp.isclose(payoff, expected_payoff, rtol=1e-5)


def test_asian_geometric_less_than_arithmetic():
    """Test that geometric average <= arithmetic average."""
    path = jnp.array([100.0, 105.0, 110.0, 115.0, 120.0])

    arith_option = AsianArithmetic(K=100.0, T=1.0, is_call=True)
    geom_option = AsianGeometric(K=100.0, T=1.0, is_call=True)

    arith_payoff = arith_option.payoff_path(path)
    geom_payoff = geom_option.payoff_path(path)

    # Geometric average payoff should be <= arithmetic average payoff
    assert geom_payoff <= arith_payoff


def test_asian_arithmetic_floating_strike_call():
    """Test arithmetic floating strike call."""
    option = AsianArithmeticFloatingStrike(T=1.0, is_call=True)
    # Path where terminal > average
    path = jnp.array([100.0, 102.0, 98.0, 105.0, 120.0])

    payoff = option.payoff_path(path)

    # Terminal = 120, Average = 105, Payoff = 15
    avg = path.mean()
    terminal = path[-1]
    expected_payoff = terminal - avg
    assert jnp.isclose(payoff, expected_payoff)


def test_asian_arithmetic_floating_strike_put():
    """Test arithmetic floating strike put."""
    option = AsianArithmeticFloatingStrike(T=1.0, is_call=False)
    # Path where terminal < average
    path = jnp.array([100.0, 110.0, 115.0, 112.0, 90.0])

    payoff = option.payoff_path(path)

    # Terminal = 90, Average = 105.4, Payoff = 15.4
    avg = path.mean()
    terminal = path[-1]
    expected_payoff = avg - terminal
    assert jnp.isclose(payoff, expected_payoff)


def test_asian_geometric_floating_strike_call():
    """Test geometric floating strike call."""
    option = AsianGeometricFloatingStrike(T=1.0, is_call=True)
    # Path where terminal > geometric average
    path = jnp.array([100.0, 102.0, 98.0, 105.0, 120.0])

    payoff = option.payoff_path(path)

    # Terminal = 120, Geometric Average < 120, Payoff > 0
    geometric_avg = jnp.exp(jnp.log(path).mean())
    terminal = path[-1]
    expected_payoff = terminal - geometric_avg
    assert jnp.isclose(payoff, expected_payoff, rtol=1e-5)


def test_asian_geometric_floating_strike_put():
    """Test geometric floating strike put."""
    option = AsianGeometricFloatingStrike(T=1.0, is_call=False)
    # Path where terminal < geometric average
    path = jnp.array([100.0, 110.0, 115.0, 112.0, 90.0])

    payoff = option.payoff_path(path)

    # Terminal = 90, Geometric Average > 90, Payoff > 0
    geometric_avg = jnp.exp(jnp.log(path).mean())
    terminal = path[-1]
    expected_payoff = geometric_avg - terminal
    assert jnp.isclose(payoff, expected_payoff, rtol=1e-5)


def test_asian_requires_path():
    """Test that Asian options require full path."""
    option = AsianArithmetic(K=100.0, T=1.0, is_call=True)
    assert option.requires_path is True
    assert option.supports_pde is False


def test_asian_single_point_path():
    """Test Asian option with single point path (edge case)."""
    option = AsianArithmetic(K=100.0, T=1.0, is_call=True)
    path = jnp.array([110.0])

    payoff = option.payoff_path(path)

    # Average = 110, Payoff = 10
    assert jnp.isclose(payoff, 10.0)


def test_asian_constant_path():
    """Test Asian option with constant path."""
    option = AsianArithmetic(K=100.0, T=1.0, is_call=True)
    path = jnp.array([105.0, 105.0, 105.0, 105.0, 105.0])

    payoff = option.payoff_path(path)

    # Average = 105, Payoff = 5
    assert jnp.isclose(payoff, 5.0)

    # For geometric, should be the same
    geom_option = AsianGeometric(K=100.0, T=1.0, is_call=True)
    geom_payoff = geom_option.payoff_path(path)

    # For constant path, arithmetic = geometric
    assert jnp.isclose(payoff, geom_payoff)
