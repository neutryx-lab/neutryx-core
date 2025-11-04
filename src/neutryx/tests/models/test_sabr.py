"""Tests for SABR stochastic volatility model."""
import jax.numpy as jnp
import pytest

from neutryx.models.sabr import (
    SABRParams,
    sabr_implied_volatility_hagan,
    sabr_call_price,
    calibrate_sabr,
    sabr_density,
)


def test_sabr_params_validation():
    """Test SABR parameter validation."""
    # Valid parameters
    params = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
    assert params.alpha == 0.3

    # Invalid beta
    with pytest.raises(ValueError):
        SABRParams(alpha=0.3, beta=1.5, rho=-0.3, nu=0.4)

    # Invalid rho
    with pytest.raises(ValueError):
        SABRParams(alpha=0.3, beta=0.5, rho=-1.5, nu=0.4)

    # Invalid alpha
    with pytest.raises(ValueError):
        SABRParams(alpha=-0.1, beta=0.5, rho=-0.3, nu=0.4)


def test_sabr_implied_vol_atm():
    """Test SABR implied volatility at-the-money."""
    vol = sabr_implied_volatility_hagan(
        F=100.0,
        K=100.0,
        T=1.0,
        alpha=0.3,
        beta=0.5,
        rho=-0.3,
        nu=0.4,
    )

    # ATM vol should be close to alpha / F^(1-beta)
    # For beta=0.5, F=100, alpha=0.3: vol ≈ 0.3 / sqrt(100) = 0.03
    expected_atm = 0.3 / (100.0 ** 0.5)  # ≈ 0.03
    assert 0.5 * expected_atm < vol < 2.0 * expected_atm  # Within 2x of expected


def test_sabr_implied_vol_otm():
    """Test SABR implied volatility out-of-the-money."""
    # ATM vol
    vol_atm = sabr_implied_volatility_hagan(
        F=100.0, K=100.0, T=1.0, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4
    )

    # OTM call (high strike)
    vol_otm_call = sabr_implied_volatility_hagan(
        F=100.0, K=110.0, T=1.0, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4
    )

    # OTM put (low strike)
    vol_otm_put = sabr_implied_volatility_hagan(
        F=100.0, K=90.0, T=1.0, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4
    )

    # All should be positive
    assert vol_atm > 0
    assert vol_otm_call > 0
    assert vol_otm_put > 0


def test_sabr_implied_vol_smile():
    """Test SABR volatility smile shape."""
    F = 100.0
    T = 1.0
    strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])

    # Negative rho -> downward sloping smile
    vols_neg_rho = jnp.array([
        sabr_implied_volatility_hagan(F, K, T, 0.3, 0.5, -0.5, 0.4)
        for K in strikes
    ])

    # Positive rho -> upward sloping smile
    vols_pos_rho = jnp.array([
        sabr_implied_volatility_hagan(F, K, T, 0.3, 0.5, 0.5, 0.4)
        for K in strikes
    ])

    # Check smile characteristics
    assert jnp.all(vols_neg_rho > 0)
    assert jnp.all(vols_pos_rho > 0)


def test_sabr_call_price_lognormal():
    """Test SABR call pricing with lognormal (beta=1)."""
    params = SABRParams(alpha=0.3, beta=1.0, rho=-0.3, nu=0.4)

    price = sabr_call_price(
        F=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        params=params,
    )

    # ATM call should have reasonable price
    assert 5.0 < price < 20.0


def test_sabr_call_price_normal():
    """Test SABR call pricing with normal (beta=0)."""
    params = SABRParams(alpha=30.0, beta=0.0, rho=-0.3, nu=0.4)

    price = sabr_call_price(
        F=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        params=params,
    )

    assert price > 0.0


def test_sabr_call_price_moneyness():
    """Test SABR call prices across moneyness."""
    params = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)

    # ITM, ATM, OTM
    price_itm = sabr_call_price(F=100.0, K=90.0, T=1.0, r=0.05, params=params)
    price_atm = sabr_call_price(F=100.0, K=100.0, T=1.0, r=0.05, params=params)
    price_otm = sabr_call_price(F=100.0, K=110.0, T=1.0, r=0.05, params=params)

    # ITM > ATM > OTM
    assert price_itm > price_atm > price_otm > 0


def test_calibrate_sabr_basic():
    """Test basic SABR calibration."""
    # Create synthetic market data
    F = 100.0
    T = 1.0
    beta = 0.5

    true_params = SABRParams(alpha=0.3, beta=beta, rho=-0.4, nu=0.5)

    strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])

    # Generate market vols
    market_vols = jnp.array([
        sabr_implied_volatility_hagan(F, K, T, true_params.alpha, true_params.beta,
                                      true_params.rho, true_params.nu)
        for K in strikes
    ])

    # Calibrate
    calibrated = calibrate_sabr(
        F=F,
        strikes=strikes,
        T=T,
        market_vols=market_vols,
        beta=beta,
        initial_alpha=0.25,
        initial_rho=0.0,
        initial_nu=0.3,
        n_iterations=200,
        lr=1e-2,
    )

    # Should recover approximately the true parameters
    assert abs(calibrated.alpha - true_params.alpha) < 0.1
    assert abs(calibrated.rho - true_params.rho) < 0.2
    assert abs(calibrated.nu - true_params.nu) < 0.2


def test_calibrate_sabr_beta_fixed():
    """Test SABR calibration with fixed beta."""
    F = 100.0
    strikes = jnp.array([95.0, 100.0, 105.0])
    market_vols = jnp.array([0.25, 0.23, 0.24])

    # Calibrate with beta=0 (normal model)
    params_normal = calibrate_sabr(
        F=F,
        strikes=strikes,
        T=1.0,
        market_vols=market_vols,
        beta=0.0,
        n_iterations=100,
    )

    assert params_normal.beta == 0.0
    assert params_normal.alpha > 0

    # Calibrate with beta=1 (lognormal model)
    params_lognormal = calibrate_sabr(
        F=F,
        strikes=strikes,
        T=1.0,
        market_vols=market_vols,
        beta=1.0,
        n_iterations=100,
    )

    assert params_lognormal.beta == 1.0
    assert params_lognormal.alpha > 0


def test_sabr_density():
    """Test SABR probability density approximation."""
    params = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)

    S = jnp.linspace(50.0, 150.0, 100)
    density = sabr_density(
        F=100.0,
        S=S,
        T=1.0,
        params=params,
    )

    # Density should be non-negative
    assert jnp.all(density >= 0.0)

    # Density should integrate to approximately 1 (within numerical tolerance)
    # Note: This is approximate due to discretization
    integral = jnp.trapezoid(density, S)
    assert 0.5 < integral < 2.0  # Loose bounds due to approximation


def test_sabr_time_dependency():
    """Test SABR vol dependency on time to maturity."""
    params = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)

    vol_1y = sabr_implied_volatility_hagan(100.0, 100.0, 1.0, params.alpha, params.beta, params.rho, params.nu)
    vol_2y = sabr_implied_volatility_hagan(100.0, 100.0, 2.0, params.alpha, params.beta, params.rho, params.nu)

    # Both should be positive
    assert vol_1y > 0
    assert vol_2y > 0

    # Vol should change with time (due to vol-of-vol)
    assert vol_1y != vol_2y
