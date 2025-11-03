"""Tests for Cox-Ingersoll-Ross (CIR) short rate model."""

import jax
import jax.numpy as jnp
import pytest

from neutryx.models.cir import (
    CIRParams,
    bond_option_price,
    simulate_path,
    simulate_paths,
    yield_curve,
    zero_coupon_bond_price,
)


@pytest.fixture
def standard_params():
    """Standard CIR parameters for testing (Feller condition satisfied)."""
    return CIRParams(a=0.5, b=0.05, sigma=0.1, r0=0.03)


@pytest.fixture
def params_no_feller():
    """CIR parameters where Feller condition is NOT satisfied."""
    return CIRParams(a=0.1, b=0.01, sigma=0.2, r0=0.03)


def test_cir_params_validation():
    """Test parameter validation."""
    # Valid parameters
    params = CIRParams(a=0.5, b=0.05, sigma=0.1, r0=0.03)
    assert params.a == 0.5

    # Invalid a (negative)
    with pytest.raises(ValueError, match="Mean reversion speed"):
        CIRParams(a=-0.1, b=0.05, sigma=0.1, r0=0.03)

    # Invalid b (zero)
    with pytest.raises(ValueError, match="Long-term mean"):
        CIRParams(a=0.5, b=0.0, sigma=0.1, r0=0.03)

    # Invalid sigma (negative)
    with pytest.raises(ValueError, match="Volatility"):
        CIRParams(a=0.5, b=0.05, sigma=-0.1, r0=0.03)

    # Invalid r0 (negative)
    with pytest.raises(ValueError, match="Initial rate"):
        CIRParams(a=0.5, b=0.05, sigma=0.1, r0=-0.01)


def test_feller_condition(standard_params, params_no_feller):
    """Test Feller condition check."""
    # Standard params satisfy Feller: 2*0.5*0.05 = 0.05 >= 0.1² = 0.01
    assert standard_params.feller_condition_satisfied()

    # params_no_feller don't: 2*0.1*0.01 = 0.002 < 0.2² = 0.04
    assert not params_no_feller.feller_condition_satisfied()


def test_zero_coupon_bond_price(standard_params):
    """Test zero-coupon bond pricing."""
    price_1y = zero_coupon_bond_price(standard_params, T=1.0)
    assert 0.0 < price_1y < 1.0

    # Longer maturity should have lower price
    price_5y = zero_coupon_bond_price(standard_params, T=5.0)
    assert price_5y < price_1y

    # Very short maturity should be close to 1
    price_short = zero_coupon_bond_price(standard_params, T=0.01)
    assert price_short > 0.99


def test_yield_curve(standard_params):
    """Test yield curve calculation."""
    maturities = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    yields = yield_curve(standard_params, maturities)

    # All yields should be positive
    assert jnp.all(yields > 0)

    # Yields should be in reasonable range
    assert jnp.all(yields < 0.2)

    # Check shape
    assert yields.shape == maturities.shape


def test_simulate_path_euler_shape(standard_params):
    """Test that simulated path has correct shape (Euler method)."""
    key = jax.random.PRNGKey(42)
    path = simulate_path(standard_params, T=1.0, n_steps=100, key=key, method="euler")

    # Should have n_steps + 1 points
    assert path.shape == (101,)

    # First value should be r0
    assert jnp.isclose(path[0], standard_params.r0)

    # All rates should be positive
    assert jnp.all(path > 0)


def test_simulate_path_milstein_shape(standard_params):
    """Test that simulated path has correct shape (Milstein method)."""
    key = jax.random.PRNGKey(42)
    path = simulate_path(standard_params, T=1.0, n_steps=100, key=key, method="milstein")

    assert path.shape == (101,)
    assert jnp.isclose(path[0], standard_params.r0)
    assert jnp.all(path > 0)


def test_simulate_path_positivity(standard_params):
    """Test that simulated rates remain positive."""
    key = jax.random.PRNGKey(42)

    for method in ["euler", "milstein"]:
        path = simulate_path(standard_params, T=1.0, n_steps=100, key=key, method=method)
        # All rates must be positive
        assert jnp.all(path > 0), f"Negative rates in {method} method"


def test_simulate_paths_shape(standard_params):
    """Test that simulated paths have correct shape."""
    key = jax.random.PRNGKey(42)
    paths = simulate_paths(standard_params, T=1.0, n_steps=100, n_paths=50, key=key)

    # Should have shape [n_paths, n_steps + 1]
    assert paths.shape == (50, 101)

    # All paths should start at r0
    assert jnp.allclose(paths[:, 0], standard_params.r0)

    # All rates should be positive
    assert jnp.all(paths > 0)


def test_simulate_paths_mean_reversion(standard_params):
    """Test that simulated paths exhibit mean reversion."""
    # Start above long-term mean
    params = CIRParams(a=0.5, b=0.05, sigma=0.05, r0=0.10)
    key = jax.random.PRNGKey(42)

    paths = simulate_paths(params, T=10.0, n_steps=1000, n_paths=1000, key=key)

    # After sufficient time, mean should converge to b
    final_mean = jnp.mean(paths[:, -1])
    # Allow some tolerance due to stochastic nature
    assert jnp.abs(final_mean - params.b) < 0.02


def test_bond_option_price(standard_params):
    """Test bond option pricing."""
    price_call = bond_option_price(
        standard_params,
        strike=0.95,
        option_maturity=1.0,
        bond_maturity=2.0,
        is_call=True
    )

    # Price should be non-negative
    assert price_call >= 0

    price_put = bond_option_price(
        standard_params,
        strike=0.95,
        option_maturity=1.0,
        bond_maturity=2.0,
        is_call=False
    )

    assert price_put >= 0


def test_bond_option_maturity_validation(standard_params):
    """Test that bond maturity must be greater than option maturity."""
    with pytest.raises(ValueError, match="must be greater than"):
        bond_option_price(
            standard_params,
            strike=0.95,
            option_maturity=2.0,
            bond_maturity=1.0,  # Invalid
            is_call=True
        )


def test_simulate_path_invalid_method(standard_params):
    """Test that invalid simulation method raises error."""
    key = jax.random.PRNGKey(42)

    with pytest.raises(ValueError, match="Unknown method"):
        simulate_path(standard_params, T=1.0, n_steps=100, key=key, method="invalid")


def test_jit_compilation(standard_params):
    """Test that key functions are JIT-compilable."""
    # JIT compile bond pricing
    jitted_bond_price = jax.jit(lambda T: zero_coupon_bond_price(standard_params, T))
    price = jitted_bond_price(5.0)
    assert jnp.isfinite(price)

    # JIT compile yield curve
    maturities = jnp.array([1.0, 2.0, 5.0])
    jitted_yields = jax.jit(lambda m: yield_curve(standard_params, m))
    yields = jitted_yields(maturities)
    assert jnp.all(jnp.isfinite(yields))
