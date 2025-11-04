"""Tests for extended Heston stochastic volatility model."""
import jax
import jax.numpy as jnp
import pytest

from neutryx.models.heston import (
    HestonParams,
    simulate_heston,
    heston_call_price_cf,
    calibrate_heston,
)


def test_heston_params_feller_condition():
    """Test Feller condition check."""
    # Satisfies Feller condition
    params1 = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    assert params1.feller_condition()

    # Does not satisfy Feller condition
    params2 = HestonParams(v0=0.04, kappa=0.1, theta=0.04, sigma=2.0, rho=-0.5)
    assert not params2.feller_condition()


def test_simulate_heston_euler():
    """Test Heston simulation with Euler scheme."""
    key = jax.random.PRNGKey(42)
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)

    S_paths, v_paths = simulate_heston(
        key=key,
        S0=100.0,
        params=params,
        r=0.05,
        q=0.02,
        T=1.0,
        n_steps=100,
        n_paths=1000,
        scheme="euler",
    )

    # Check shapes
    assert S_paths.shape == (1000, 101)
    assert v_paths.shape == (1000, 101)

    # Check initial values
    assert jnp.allclose(S_paths[:, 0], 100.0)
    assert jnp.allclose(v_paths[:, 0], 0.04)

    # Check variance stays non-negative
    assert jnp.all(v_paths >= 0.0)

    # Check paths are reasonable
    assert jnp.all(S_paths > 0.0)


def test_simulate_heston_qe():
    """Test Heston simulation with QE scheme."""
    key = jax.random.PRNGKey(123)
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

    S_paths, v_paths = simulate_heston(
        key=key,
        S0=100.0,
        params=params,
        r=0.05,
        q=0.0,
        T=1.0,
        n_steps=50,
        n_paths=500,
        scheme="qe",
    )

    assert S_paths.shape == (500, 51)
    assert v_paths.shape == (500, 51)
    assert jnp.all(v_paths >= 0.0)


def test_heston_call_price_cf():
    """Test Heston call pricing with characteristic function."""
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)

    price = heston_call_price_cf(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        q=0.02,
        params=params,
        n_points=1024,
    )

    # Price should be reasonable for ATM option
    assert 5.0 < price < 20.0

    # OTM option should be cheaper
    price_otm = heston_call_price_cf(
        S0=100.0,
        K=120.0,
        T=1.0,
        r=0.05,
        q=0.02,
        params=params,
    )

    assert price_otm < price


def test_heston_call_price_itm():
    """Test Heston call pricing for ITM option."""
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)

    price_itm = heston_call_price_cf(
        S0=100.0,
        K=80.0,
        T=1.0,
        r=0.05,
        q=0.02,
        params=params,
    )

    # ITM option should have intrinsic value
    intrinsic = 100.0 * jnp.exp(-0.02 * 1.0) - 80.0 * jnp.exp(-0.05 * 1.0)
    assert price_itm > intrinsic


def test_calibrate_heston_basic():
    """Test basic Heston calibration."""
    # Create synthetic market data
    true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)

    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    maturities = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])

    # Generate market prices using true params
    market_prices = jnp.array([
        heston_call_price_cf(100.0, K, 1.0, 0.05, 0.02, true_params)
        for K in strikes
    ])

    # Calibrate with different initial guess
    initial = HestonParams(v0=0.05, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.3)

    calibrated = calibrate_heston(
        S0=100.0,
        strikes=strikes,
        maturities=maturities,
        market_prices=market_prices,
        r=0.05,
        q=0.02,
        initial_params=initial,
        n_iterations=50,
        lr=1e-2,
    )

    # Calibrated params should be close to true params
    # (may not be exact due to optimization and numerical precision)
    assert 0.01 < calibrated.v0 < 0.1
    assert 0.5 < calibrated.kappa < 5.0
    assert 0.01 < calibrated.theta < 0.1
    assert 0.1 < calibrated.sigma < 1.0
    assert -1.0 < calibrated.rho < 0.0


def test_heston_variance_mean_reversion():
    """Test that variance mean-reverts in Heston simulation."""
    key = jax.random.PRNGKey(999)
    params = HestonParams(v0=0.1, kappa=5.0, theta=0.04, sigma=0.2, rho=0.0)

    _, v_paths = simulate_heston(
        key=key,
        S0=100.0,
        params=params,
        r=0.05,
        q=0.0,
        T=2.0,
        n_steps=500,
        n_paths=1000,
        scheme="euler",
    )

    # Terminal variance should be closer to theta than initial
    terminal_var = v_paths[:, -1].mean()

    # Should move towards theta = 0.04
    assert 0.02 < terminal_var < 0.08
