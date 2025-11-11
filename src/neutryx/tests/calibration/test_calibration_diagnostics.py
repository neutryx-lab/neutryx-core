"""Tests for calibration diagnostics utilities."""

import jax
import jax.numpy as jnp

from neutryx.calibration.diagnostics import (
    CalibrationDiagnostics,
    build_residual_plot_data,
    generate_calibration_diagnostics,
)
from neutryx.models.heston import HestonParams, heston_call_price_semi_analytical
from neutryx.models.heston import calibrate_heston


def _heston_surface(params, *, strike, maturity, S0, r, q):
    pricing_fn = jax.vmap(lambda K, T: heston_call_price_semi_analytical(S0, K, T, r, q, params))
    return pricing_fn(strike, maturity)


def test_residual_plot_dataframe_roundtrip():
    strikes = jnp.array([95.0, 100.0, 105.0])
    maturities = jnp.array([1.0, 1.0, 1.0])
    residuals = jnp.array([0.1, -0.05, 0.02])

    plot_data = build_residual_plot_data({"strike": strikes, "maturity": maturities}, residuals)
    df = plot_data.to_dataframe()

    assert set(df.columns) == {"strike", "maturity", "residual"}
    assert df.shape[0] == strikes.size
    assert jnp.isclose(df["residual"].to_numpy(), residuals).all()


def test_generate_calibration_diagnostics_heston():
    S0, r, q = 100.0, 0.01, 0.0
    strikes = jnp.array([95.0, 100.0, 105.0])
    maturities = jnp.array([0.5, 1.0, 1.5])

    true_params = HestonParams(v0=0.05, kappa=1.6, theta=0.04, sigma=0.35, rho=-0.5)
    market_prices = _heston_surface(true_params, strike=strikes, maturity=maturities, S0=S0, r=r, q=q)

    # Reduced iterations for CI performance (was 25)
    calibrated = calibrate_heston(
        S0,
        strikes,
        maturities,
        market_prices,
        r=r,
        q=q,
        n_iterations=10,  # Reduced from 25 for faster CI
        lr=8e-3,  # Slightly increased learning rate to compensate
    )

    diagnostics = generate_calibration_diagnostics(
        lambda params, **coords: _heston_surface(params, S0=S0, r=r, q=q, **coords),
        calibrated,
        market_prices,
        coordinates={"strike": strikes, "maturity": maturities},
        parameter_names=["v0", "kappa", "theta", "sigma", "rho"],
    )

    assert isinstance(diagnostics, CalibrationDiagnostics)
    assert diagnostics.mse >= 0.0
    assert diagnostics.mae >= 0.0
    assert diagnostics.max_abs_error >= 0.0
    assert diagnostics.residual_plot is not None
    assert diagnostics.residual_plot.to_dataframe().shape[0] == strikes.size
    assert diagnostics.identifiability is not None
    assert diagnostics.identifiability.correlation_matrix.shape == (5, 5)
    assert diagnostics.identifiability.condition_number > 0
    assert set(diagnostics.identifiability.parameter_std.keys()) == {
        "v0",
        "kappa",
        "theta",
        "sigma",
        "rho",
    }
