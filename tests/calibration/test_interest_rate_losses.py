"""Regression tests for calibration helpers in the interest-rate toolkit."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutryx.calibration import (
    g2pp_parameter_specs,
    g2pp_zero_curve_loss,
    quasi_gaussian_parameter_specs,
    quasi_gaussian_zero_curve_loss,
)
from neutryx.models.g2pp import G2PPParams, zero_coupon_bond_price as g2pp_price
from neutryx.models.quasi_gaussian import (
    QuasiGaussianParams,
    zero_coupon_bond_price as qg_price,
)


G2PP_MARKET_MATS = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
G2PP_MARKET_DISCOUNTS = jnp.array(
    [
        0.9986324310302734,
        0.9948046803474426,
        0.9812051057815552,
        0.961615264415741,
        0.9112087488174438,
        0.8531278371810913,
        0.7624638080596924,
    ]
)


def _constant_fn(value: float):
    return lambda _t: value


QG_MARKET_MATS = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
QG_MARKET_DISCOUNTS = jnp.array(
    [
        0.9954845905303955,
        0.9909364581108093,
        0.9817320108413696,
        0.9628446698188782,
        0.9432772397994995,
        0.9021447896957397,
    ]
)


@pytest.mark.parametrize(
    "loss_fn, params, maturities, market_discounts",
    [
        (
            g2pp_zero_curve_loss,
            G2PPParams(a=0.12, b=0.35, sigma_x=0.01, sigma_y=0.017, rho=-0.6, r0=0.021),
            G2PP_MARKET_MATS,
            G2PP_MARKET_DISCOUNTS,
        ),
        (
            quasi_gaussian_zero_curve_loss,
            QuasiGaussianParams(
                alpha_fn=lambda t: 0.08 + 0.005 * t,
                beta_fn=_constant_fn(0.18),
                sigma_x_fn=_constant_fn(0.012),
                sigma_y_fn=_constant_fn(0.017),
                forward_curve_fn=lambda t: 0.018 + 0.0008 * t,
                rho=-0.45,
                r0=0.019,
            ),
            QG_MARKET_MATS,
            QG_MARKET_DISCOUNTS,
        ),
    ],
)
def test_zero_curve_losses_return_zero_for_perfect_fit(
    loss_fn, params, maturities, market_discounts
):
    pricing_fn = g2pp_price if isinstance(params, G2PPParams) else qg_price
    predicted = jnp.array([pricing_fn(params, float(T)) for T in maturities])
    loss = loss_fn(predicted, market_discounts, market_data={"maturities": maturities})
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "loss_fn, params, maturities, market_discounts",
    [
        (
            g2pp_zero_curve_loss,
            G2PPParams(a=0.12, b=0.35, sigma_x=0.01, sigma_y=0.017, rho=-0.6, r0=0.021),
            G2PP_MARKET_MATS,
            G2PP_MARKET_DISCOUNTS,
        ),
        (
            quasi_gaussian_zero_curve_loss,
            QuasiGaussianParams(
                alpha_fn=lambda t: 0.08 + 0.005 * t,
                beta_fn=_constant_fn(0.18),
                sigma_x_fn=_constant_fn(0.012),
                sigma_y_fn=_constant_fn(0.017),
                forward_curve_fn=lambda t: 0.018 + 0.0008 * t,
                rho=-0.45,
                r0=0.019,
            ),
            QG_MARKET_MATS,
            QG_MARKET_DISCOUNTS,
        ),
    ],
)
def test_zero_curve_losses_are_sensitive_to_shift(loss_fn, params, maturities, market_discounts):
    pricing_fn = g2pp_price if isinstance(params, G2PPParams) else qg_price
    predicted = jnp.array([pricing_fn(params, float(T)) for T in maturities])
    shocked = predicted * (1.0 + 0.002)
    loss = loss_fn(shocked, market_discounts, market_data={"maturities": maturities})
    assert float(loss) > 0


def test_g2pp_parameter_specs_apply_constraints():
    specs = g2pp_parameter_specs()
    unconstrained = {name: jnp.array(0.0) for name in specs}
    constrained = {name: specs[name].transform.apply(value) for name, value in unconstrained.items()}
    assert all(constrained[name] > 0 for name in ["a", "b", "sigma_x", "sigma_y"])
    assert jnp.all(jnp.abs(constrained["rho"]) < 1.0)


def test_quasi_gaussian_parameter_specs_custom_initialisation():
    custom = {"alpha": 0.2, "rho": 0.1}
    specs = quasi_gaussian_parameter_specs(custom)
    assert specs["alpha"].init == 0.2
    assert specs["rho"].init == 0.1
