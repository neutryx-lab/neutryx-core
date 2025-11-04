"""Tests for credit risk modules (CDS, hazard rate, structural models)."""
import jax.numpy as jnp
from jax.scipy.stats import norm

from neutryx.market.credit import (
    BlackCoxModel,
    HazardRateCurve,
    MertonModel,
    calibrate_black_cox_barrier,
    calibrate_merton_from_equity,
    calibrate_piecewise_hazard,
    cds_par_spread,
    cds_pv,
)


def _equity_vol_from_merton(model: MertonModel) -> float:
    d1, _ = model._d1d2()
    equity_value = model.equity_value()
    return model.asset_vol * model.asset_value * norm.cdf(d1) / equity_value


def test_cds_hazard_calibration_matches_input_spreads():
    payment_times = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])
    discount_factors = jnp.exp(-0.02 * payment_times)
    hazard = jnp.array([0.015, 0.018, 0.022, 0.026, 0.03])
    # Construct market par spreads for each maturity bucket
    market_spreads = []
    for idx in range(payment_times.shape[0]):
        sub_curve = HazardRateCurve(payment_times[: idx + 1], hazard[: idx + 1])
        spread = cds_par_spread(
            payment_times[: idx + 1],
            discount_factors[: idx + 1],
            sub_curve,
            recovery=0.4,
        )
        market_spreads.append(float(spread))
    market_spreads = jnp.array(market_spreads)

    calibrated = calibrate_piecewise_hazard(
        payment_times,
        discount_factors,
        market_spreads,
        recovery=0.4,
    )

    assert jnp.allclose(calibrated.intensities, hazard, atol=1e-4)

    par_spread_full = cds_par_spread(
        payment_times,
        discount_factors,
        calibrated,
        recovery=0.4,
    )
    pv = cds_pv(
        float(par_spread_full),
        payment_times,
        discount_factors,
        calibrated,
        recovery=0.4,
    )
    assert jnp.isclose(pv, 0.0, atol=1e-6)


def test_merton_calibration_recovers_asset_inputs():
    true_model = MertonModel(
        asset_value=150.0,
        debt=100.0,
        maturity=3.0,
        asset_vol=0.25,
        rate=0.03,
    )
    equity_value = float(true_model.equity_value())
    equity_vol = float(_equity_vol_from_merton(true_model))

    calibrated = calibrate_merton_from_equity(
        debt=100.0,
        maturity=3.0,
        rate=0.03,
        equity_value=equity_value,
        equity_vol=equity_vol,
    )

    assert jnp.isclose(calibrated.asset_value, true_model.asset_value, rtol=1e-4)
    assert jnp.isclose(calibrated.asset_vol, true_model.asset_vol, rtol=1e-4)
    assert 0.0 < calibrated.default_probability() < 1.0


def test_black_cox_barrier_calibration_hits_target_default():
    model = BlackCoxModel(
        asset_value=120.0,
        debt=100.0,
        barrier=85.0,
        maturity=2.0,
        asset_vol=0.2,
        rate=0.02,
    )
    default_prob = float(model.default_probability())
    assert 0.0 < default_prob < 1.0

    calibrated = calibrate_black_cox_barrier(
        asset_value=120.0,
        debt=100.0,
        maturity=2.0,
        asset_vol=0.2,
        rate=0.02,
        target_default_prob=default_prob,
    )

    assert jnp.isclose(calibrated.default_probability(), default_prob, atol=1e-6)
    assert calibrated.credit_spread() >= 0.0
    survival = calibrated.survival_probability()
    assert 0.0 <= survival <= 1.0
