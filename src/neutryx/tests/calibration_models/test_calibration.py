"""Tests for calibration controllers on synthetic data."""

import jax.numpy as jnp

from neutryx.calibration.heston import (
    HestonCalibrationController,
    generate_heston_market_data,
)
from neutryx.calibration.sabr import (
    SABRCalibrationController,
    generate_sabr_market_data,
)
from neutryx.calibration.slv import (
    SLVCalibrationController,
    generate_slv_market_data,
)
from neutryx.calibration.models.heston import HestonParams
from neutryx.calibration.models.sabr import SABRParams


def test_sabr_controller_recovers_parameters():
    forward = 100.0
    strike_levels = jnp.array([85.0, 95.0, 105.0, 115.0])
    maturity_levels = jnp.array([0.5, 1.0])
    strikes, maturities = jnp.meshgrid(strike_levels, maturity_levels, indexing="ij")
    strikes = strikes.ravel()
    maturities = maturities.ravel()
    true_params = SABRParams(alpha=0.25, beta=0.6, rho=-0.35, nu=0.4)

    market_data = generate_sabr_market_data(forward, strikes, maturities, true_params)

    controller = SABRCalibrationController(max_steps=600, tol=1e-9)
    result = controller.calibrate(market_data)

    assert result.converged or result.loss_history[-1] < 1e-8

    calibrated = result.params
    assert jnp.isclose(calibrated["alpha"], true_params.alpha, rtol=0.12)
    assert jnp.isclose(calibrated["beta"], true_params.beta, rtol=0.08)
    assert jnp.isclose(calibrated["rho"], true_params.rho, rtol=0.15)
    assert jnp.isclose(calibrated["nu"], true_params.nu, rtol=0.12)


def test_heston_controller_recovers_surface():
    spot = 100.0
    rate = 0.03
    dividend = 0.0
    strike_levels = jnp.array([90.0, 100.0, 110.0])
    maturity_levels = jnp.array([0.5, 1.0, 1.5])
    strikes, maturities = jnp.meshgrid(strike_levels, maturity_levels, indexing="ij")
    strikes = strikes.ravel()
    maturities = maturities.ravel()

    true_params = HestonParams(v0=0.05, kappa=1.8, theta=0.04, sigma=0.5, rho=-0.6)
    market_data = generate_heston_market_data(
        spot,
        strikes,
        maturities,
        true_params,
        rate=rate,
        dividend=dividend,
    )

    controller = HestonCalibrationController(max_steps=250, tol=5e-7)
    result = controller.calibrate(market_data)

    assert result.loss_history[-1] < 5e-3

    calibrated = result.params
    assert calibrated["v0"] > 0
    assert calibrated["theta"] > 0
    assert jnp.isclose(calibrated["rho"], true_params.rho, atol=0.2)
    assert jnp.isclose(calibrated["sigma"], true_params.sigma, rtol=0.35)

    # Feller condition penalty should keep violation tiny
    feller_violation = calibrated["sigma"] ** 2 - 2 * calibrated["kappa"] * calibrated["theta"]
    assert feller_violation < 5e-3


def test_slv_controller_recovers_synthetic_surface():
    forward = 100.0
    strike_levels = jnp.array([85.0, 95.0, 105.0, 115.0])
    maturity_levels = jnp.array([0.5, 1.0, 1.5])
    strikes, maturities = jnp.meshgrid(strike_levels, maturity_levels, indexing="ij")
    strikes = strikes.ravel()
    maturities = maturities.ravel()

    true_params = {
        "base_vol": 0.22,
        "local_slope": -0.25,
        "local_curvature": 0.45,
        "mixing": 0.4,
        "time_decay": 0.08,
    }

    market_data = generate_slv_market_data(forward, strikes, maturities, true_params)

    controller = SLVCalibrationController(max_steps=350, tol=1e-9)
    result = controller.calibrate(market_data)

    assert result.converged or result.loss_history[-1] < 1e-10

    calibrated = result.params
    for key, value in true_params.items():
        assert jnp.isclose(calibrated[key], value, rtol=0.2)
