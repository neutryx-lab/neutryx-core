"""Regression-style tests for the unified interest-rate interface."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutryx.models.g2pp import G2PPParams, zero_coupon_bond_price as g2pp_price
from neutryx.models.interest_rate import (
    G2PPInterestRateModel,
    InterestRateModel,
    QuasiGaussianInterestRateModel,
)
from neutryx.models.quasi_gaussian import (
    QuasiGaussianParams,
    zero_coupon_bond_price as qg_price,
)


G2PP_PARAMS = G2PPParams(
    a=0.12,
    b=0.35,
    sigma_x=0.01,
    sigma_y=0.017,
    rho=-0.6,
    r0=0.021,
)


def _constant_fn(value: float):
    return lambda _t: value


QG_PARAMS = QuasiGaussianParams(
    alpha_fn=lambda t: 0.08 + 0.005 * t,
    beta_fn=_constant_fn(0.18),
    sigma_x_fn=_constant_fn(0.012),
    sigma_y_fn=_constant_fn(0.017),
    forward_curve_fn=lambda t: 0.018 + 0.0008 * t,
    rho=-0.45,
    r0=0.019,
)


@pytest.mark.parametrize(
    "model, baseline_fn, maturities",
    [
        (
            G2PPInterestRateModel(G2PP_PARAMS),
            lambda T: g2pp_price(G2PP_PARAMS, float(T)),
            jnp.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]),
        ),
        (
            QuasiGaussianInterestRateModel(QG_PARAMS),
            lambda T: qg_price(QG_PARAMS, float(T)),
            jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0]),
        ),
    ],
)
def test_zero_coupon_matches_reference(model: InterestRateModel, baseline_fn, maturities):
    """The wrapper should exactly reproduce the reference pricing functions."""

    model_prices = jnp.array([model.zero_coupon_bond_price(float(T)) for T in maturities])
    baseline_prices = jnp.array([baseline_fn(T) for T in maturities])
    assert jnp.allclose(model_prices, baseline_prices, atol=1e-10)


def test_g2pp_conditional_moments_agree_with_closed_form():
    model = G2PPInterestRateModel(G2PP_PARAMS)
    state = (0.015, -0.003)
    horizon = 2.5
    mean, variance = model.conditional_moments(t=0.0, horizon=horizon, state=state)

    a, b = G2PP_PARAMS.a, G2PP_PARAMS.b
    sigma_x, sigma_y = G2PP_PARAMS.sigma_x, G2PP_PARAMS.sigma_y
    rho = G2PP_PARAMS.rho

    expected_mean = state[0] * jnp.exp(-a * horizon) + state[1] * jnp.exp(-b * horizon)
    expected_var = (
        (sigma_x**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * horizon))
        + (sigma_y**2 / (2.0 * b)) * (1.0 - jnp.exp(-2.0 * b * horizon))
        + 2.0 * rho * sigma_x * sigma_y / (a + b) * (1.0 - jnp.exp(-(a + b) * horizon))
    )

    assert pytest.approx(float(expected_mean), rel=1e-12) == float(mean)
    assert pytest.approx(float(expected_var), rel=1e-12) == float(variance)


@pytest.mark.parametrize(
    "model",
    [
        G2PPInterestRateModel(G2PP_PARAMS),
        QuasiGaussianInterestRateModel(QG_PARAMS),
    ],
)
def test_simulation_shapes(model: InterestRateModel):
    key = jax.random.PRNGKey(0)
    r_paths, factors = model.simulate_paths(horizon=0.25, steps=4, paths=3, key=key)
    assert r_paths.shape == (3, 5)
    assert factors.shape == (3, 5, 2)
