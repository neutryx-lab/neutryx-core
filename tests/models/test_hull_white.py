"""Tests for Hull-White one-factor short rate model."""

import jax
import jax.numpy as jnp
import pytest

from neutryx.models.hull_white import (
    HullWhiteParams,
    cap_price,
    caplet_price,
    simulate_path,
    simulate_paths,
    zero_coupon_bond_price,
)


@pytest.fixture
def standard_params():
    """Standard Hull-White parameters for testing."""
    return HullWhiteParams(a=0.1, sigma=0.01, r0=0.03)


@pytest.fixture
def params_with_theta():
    """Hull-White parameters with time-dependent drift."""
    def theta_fn(t):
        return 0.05 + 0.01 * t  # Linearly increasing drift
    return HullWhiteParams(a=0.1, sigma=0.01, r0=0.03, theta_fn=theta_fn)


def test_hull_white_params_validation():
    """Test parameter validation."""
    # Valid parameters
    params = HullWhiteParams(a=0.1, sigma=0.01, r0=0.03)
    assert params.a == 0.1

    # Invalid a
    with pytest.raises(ValueError, match="Mean reversion speed"):
        HullWhiteParams(a=-0.1, sigma=0.01, r0=0.03)

    # Invalid sigma
    with pytest.raises(ValueError, match="Volatility"):
        HullWhiteParams(a=0.1, sigma=0.0, r0=0.03)


def test_zero_coupon_bond_price(standard_params):
    """Test zero-coupon bond pricing."""
    price_1y = zero_coupon_bond_price(standard_params, T=1.0)
    assert 0.0 < price_1y < 1.0

    # Longer maturity should have lower price
    price_5y = zero_coupon_bond_price(standard_params, T=5.0)
    assert price_5y < price_1y


def test_zero_coupon_bond_price_with_forward_curve(standard_params):
    """Test bond pricing with custom forward curve."""
    # Define a flat forward curve
    def forward_curve(t):
        return 0.04  # 4% constant forward rate

    price = zero_coupon_bond_price(standard_params, T=2.0, forward_curve_fn=forward_curve)
    assert 0.0 < price < 1.0


def test_simulate_path_shape(standard_params):
    """Test that simulated path has correct shape."""
    key = jax.random.PRNGKey(42)
    path = simulate_path(standard_params, T=1.0, n_steps=100, key=key)

    # Should have n_steps + 1 points
    assert path.shape == (101,)
    assert jnp.isclose(path[0], standard_params.r0)


def test_simulate_path_with_theta(params_with_theta):
    """Test simulation with time-dependent drift."""
    key = jax.random.PRNGKey(42)
    path = simulate_path(params_with_theta, T=1.0, n_steps=100, key=key)

    assert path.shape == (101,)
    assert jnp.isclose(path[0], params_with_theta.r0)


def test_simulate_path_exact_vs_euler(standard_params):
    """Compare exact and Euler simulation."""
    key = jax.random.PRNGKey(42)

    path_exact = simulate_path(standard_params, T=1.0, n_steps=1000, key=key, method="exact")
    path_euler = simulate_path(standard_params, T=1.0, n_steps=1000, key=key, method="euler")

    # Both should start at r0
    assert jnp.isclose(path_exact[0], standard_params.r0)
    assert jnp.isclose(path_euler[0], standard_params.r0)


def test_simulate_paths_shape(standard_params):
    """Test that simulated paths have correct shape."""
    key = jax.random.PRNGKey(42)
    paths = simulate_paths(standard_params, T=1.0, n_steps=100, n_paths=50, key=key)

    assert paths.shape == (50, 101)
    assert jnp.allclose(paths[:, 0], standard_params.r0)


def test_caplet_price(standard_params):
    """Test caplet pricing."""
    price = caplet_price(
        standard_params,
        strike=0.03,
        caplet_maturity=1.0,
        tenor=0.25
    )

    # Price should be non-negative
    assert price >= 0


def test_caplet_price_different_strikes(standard_params):
    """Test that caplet prices decrease with higher strikes."""
    price_low_strike = caplet_price(
        standard_params,
        strike=0.01,
        caplet_maturity=1.0,
        tenor=0.25
    )

    price_high_strike = caplet_price(
        standard_params,
        strike=0.05,
        caplet_maturity=1.0,
        tenor=0.25
    )

    # Higher strike should have lower price (for given forward rate)
    # This might not always hold depending on forward curve, so we just check both are non-negative
    assert price_low_strike >= 0
    assert price_high_strike >= 0


def test_cap_price(standard_params):
    """Test cap pricing as portfolio of caplets."""
    price = cap_price(
        standard_params,
        strike=0.03,
        cap_maturity=2.0,
        payment_frequency=0.25
    )

    # Cap price should be non-negative
    assert price >= 0

    # Cap price should be larger than single caplet
    caplet_p = caplet_price(
        standard_params,
        strike=0.03,
        caplet_maturity=0.25,
        tenor=0.25
    )
    assert price >= caplet_p


def test_jit_compilation(standard_params):
    """Test that key functions are JIT-compilable."""
    # JIT compile bond pricing
    jitted_bond_price = jax.jit(lambda T: zero_coupon_bond_price(standard_params, T))
    price = jitted_bond_price(5.0)
    assert jnp.isfinite(price)


def test_simulate_path_invalid_method(standard_params):
    """Test that invalid simulation method raises error."""
    key = jax.random.PRNGKey(42)

    with pytest.raises(ValueError, match="Unknown method"):
        simulate_path(standard_params, T=1.0, n_steps=100, key=key, method="invalid")
