"""Tests for Vasicek short rate model."""

import jax
import jax.numpy as jnp
import pytest

from neutryx.models.vasicek import (
    VasicekParams,
    bond_option_price,
    simulate_path,
    simulate_paths,
    yield_curve,
    zero_coupon_bond_price,
)


@pytest.fixture
def standard_params():
    """Standard Vasicek parameters for testing."""
    return VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)


def test_vasicek_params_validation():
    """Test parameter validation."""
    # Valid parameters
    params = VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)
    assert params.a == 0.1
    assert params.b == 0.05

    # Invalid a (negative)
    with pytest.raises(ValueError, match="Mean reversion speed"):
        VasicekParams(a=-0.1, b=0.05, sigma=0.01, r0=0.03)

    # Invalid sigma (zero)
    with pytest.raises(ValueError, match="Volatility"):
        VasicekParams(a=0.1, b=0.05, sigma=0.0, r0=0.03)


def test_zero_coupon_bond_price(standard_params):
    """Test zero-coupon bond pricing."""
    # Bond price should be between 0 and 1
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
    assert jnp.all(yields < 0.2)  # Less than 20%

    # Check shape
    assert yields.shape == maturities.shape


def test_yield_curve_monotonicity(standard_params):
    """Test that yield curve is monotonic when appropriate."""
    maturities = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yields = yield_curve(standard_params, maturities)

    # For standard params with r0 < b, yield curve should be increasing
    if standard_params.r0 < standard_params.b:
        diffs = jnp.diff(yields)
        # Allow small violations due to numerical precision
        assert jnp.all(diffs > -1e-10)


def test_simulate_path_shape():
    """Test that simulated path has correct shape."""
    params = VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)
    key = jax.random.PRNGKey(42)

    path = simulate_path(params, T=1.0, n_steps=100, key=key)

    # Should have n_steps + 1 points (including initial value)
    assert path.shape == (101,)

    # First value should be r0
    assert jnp.isclose(path[0], params.r0)


def test_simulate_path_exact_vs_euler():
    """Compare exact and Euler simulation methods."""
    params = VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)
    key = jax.random.PRNGKey(42)

    path_exact = simulate_path(params, T=1.0, n_steps=1000, key=key, method="exact")
    path_euler = simulate_path(params, T=1.0, n_steps=1000, key=key, method="euler")

    # Both should start at r0
    assert jnp.isclose(path_exact[0], params.r0)
    assert jnp.isclose(path_euler[0], params.r0)

    # Final values should be somewhat close (but not identical due to discretization)
    # This is a weak test - just checking they're in the same ballpark
    assert jnp.abs(path_exact[-1] - path_euler[-1]) < 0.05


def test_simulate_paths_shape():
    """Test that simulated paths have correct shape."""
    params = VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)
    key = jax.random.PRNGKey(42)

    paths = simulate_paths(params, T=1.0, n_steps=100, n_paths=50, key=key)

    # Should have shape [n_paths, n_steps + 1]
    assert paths.shape == (50, 101)

    # All paths should start at r0
    assert jnp.allclose(paths[:, 0], params.r0)


def test_simulate_paths_mean_reversion():
    """Test that simulated paths exhibit mean reversion."""
    params = VasicekParams(a=0.5, b=0.05, sigma=0.01, r0=0.10)  # Start above long-term mean
    key = jax.random.PRNGKey(42)

    paths = simulate_paths(params, T=10.0, n_steps=1000, n_paths=1000, key=key)

    # After sufficient time, mean should converge to b
    final_mean = jnp.mean(paths[:, -1])
    assert jnp.abs(final_mean - params.b) < 0.01  # Within 1%


def test_bond_option_price_call(standard_params):
    """Test bond option pricing for call option."""
    price = bond_option_price(
        standard_params,
        strike=0.95,
        option_maturity=1.0,
        bond_maturity=2.0,
        is_call=True
    )

    # Price should be positive and less than the bond price
    assert price > 0
    bond_price = zero_coupon_bond_price(standard_params, T=2.0)
    assert price < bond_price


def test_bond_option_price_put(standard_params):
    """Test bond option pricing for put option."""
    price = bond_option_price(
        standard_params,
        strike=0.95,
        option_maturity=1.0,
        bond_maturity=2.0,
        is_call=False
    )

    # Price should be non-negative
    assert price >= 0


def test_bond_option_put_call_parity(standard_params):
    """Test put-call parity for bond options."""
    strike = 0.95
    T_opt = 1.0
    T_bond = 2.0

    call_price = bond_option_price(standard_params, strike, T_opt, T_bond, is_call=True)
    put_price = bond_option_price(standard_params, strike, T_opt, T_bond, is_call=False)

    # Put-call parity: C - P = P(0, T_bond) - K * P(0, T_opt)
    P_Tbond = zero_coupon_bond_price(standard_params, T_bond)
    P_Topt = zero_coupon_bond_price(standard_params, T_opt)

    lhs = call_price - put_price
    rhs = P_Tbond - strike * P_Topt

    assert jnp.isclose(lhs, rhs, atol=1e-6)


def test_bond_option_maturity_validation(standard_params):
    """Test that bond maturity must be greater than option maturity."""
    with pytest.raises(ValueError, match="must be greater than"):
        bond_option_price(
            standard_params,
            strike=0.95,
            option_maturity=2.0,
            bond_maturity=1.0,  # Invalid: bond matures before option
            is_call=True
        )


def test_simulate_path_invalid_method():
    """Test that invalid simulation method raises error."""
    params = VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)
    key = jax.random.PRNGKey(42)

    with pytest.raises(ValueError, match="Unknown method"):
        simulate_path(params, T=1.0, n_steps=100, key=key, method="invalid")


def test_jit_compilation():
    """Test that key functions are JIT-compilable."""
    params = VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)

    # JIT compile bond pricing
    jitted_bond_price = jax.jit(lambda T: zero_coupon_bond_price(params, T))
    price = jitted_bond_price(5.0)
    assert jnp.isfinite(price)

    # JIT compile yield curve
    maturities = jnp.array([1.0, 2.0, 5.0])
    jitted_yields = jax.jit(lambda m: yield_curve(params, m))
    yields = jitted_yields(maturities)
    assert jnp.all(jnp.isfinite(yields))
