import jax.numpy as jnp
import pytest

from neutryx.market.vol_surface import (
    SmileModel,
    VolatilitySurface,
)


def _build_surface(forward, strikes, vols, smile_model):
    return VolatilitySurface(
        forward=forward,
        tenors=jnp.array([0.5]),
        strikes=[jnp.asarray(strikes)],
        implied_vols=[jnp.asarray(vols)],
        smile_model=smile_model,
    )


def test_polynomial_calibration_recovers_coefficients():
    forward = 100.0
    strikes = jnp.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    log_m = jnp.log(strikes / forward)
    coefficients = jnp.array([0.22, -0.05, 0.12])
    vols = coefficients[0] + coefficients[1] * log_m + coefficients[2] * log_m ** 2

    surface = _build_surface(forward, strikes, vols, SmileModel.POLYNOMIAL)
    params = surface.calibrate_smile(0.5)

    assert jnp.allclose(params.coefficients, coefficients, atol=1e-4)


def test_vanna_volga_calibration_matches_synthetic_surface():
    forward = 100.0
    strikes = jnp.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    log_m = jnp.log(strikes / forward)

    atm = 0.2
    rr = -0.03
    bf = 0.15
    kappa = 0.25
    vols = atm + rr * jnp.tanh(log_m / kappa) + 0.5 * bf * (log_m ** 2)

    surface = _build_surface(forward, strikes, vols, SmileModel.VANNA_VOLGA)
    params = surface.calibrate_smile(0.5)

    assert pytest.approx(atm, rel=1e-3) == params.atm_vol
    assert pytest.approx(rr, rel=1e-2) == params.risk_reversal
    assert pytest.approx(bf, rel=1e-2) == params.butterfly


def test_polynomial_calibration_fails_with_insufficient_data():
    forward = 100.0
    strikes = jnp.array([90.0, 110.0])
    vols = jnp.array([0.2, 0.25])

    surface = _build_surface(forward, strikes, vols, SmileModel.POLYNOMIAL)

    with pytest.raises(ValueError):
        surface.calibrate_smile(0.5)


def test_vanna_volga_requires_minimum_points():
    forward = 100.0
    strikes = jnp.array([90.0, 110.0])
    vols = jnp.array([0.2, 0.22])

    surface = _build_surface(forward, strikes, vols, SmileModel.VANNA_VOLGA)

    with pytest.raises(ValueError):
        surface.calibrate_smile(0.5)
