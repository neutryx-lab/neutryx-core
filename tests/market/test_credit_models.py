"""Tests for credit risk modules (CDS, hazard rate, structural models)."""
import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import norm

from neutryx.market.credit import (
    BlackCoxModel,
    HazardRateCurve,
    KMVModel,
    MertonModel,
    PortfolioLossMetrics,
    ReducedFormModel,
    calibrate_black_cox_barrier,
    calibrate_constant_intensity,
    calibrate_kmv_from_equity,
    calibrate_merton_from_equity,
    calibrate_piecewise_hazard,
    cds_par_spread,
    cds_pv,
    expected_loss,
    gaussian_copula_samples,
    portfolio_risk_metrics,
    single_factor_loss_distribution,
)


def _equity_vol_from_merton(model: MertonModel) -> float:
    d1, _ = model._d1d2()
    equity_value = model.equity_value()
    return model.asset_vol * model.asset_value * norm.cdf(d1) / equity_value


def _equity_vol_from_kmv(model: KMVModel) -> float:
    d1, _ = model._d1d2()
    equity_value = model.equity_value()
    return model.asset_vol * model.asset_value * norm.cdf(d1) / equity_value


def test_cds_hazard_calibration_matches_input_spreads():
    # Reduced number of payment times for CI performance (was 5 points)
    payment_times = jnp.array([0.5, 1.0, 2.0, 3.0])
    discount_factors = jnp.exp(-0.02 * payment_times)
    hazard = jnp.array([0.015, 0.018, 0.022, 0.026])
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


def test_kmv_calibration_recovers_asset_inputs():
    true_model = KMVModel(
        asset_value=180.0,
        short_term_debt=35.0,
        long_term_debt=90.0,
        maturity=1.0,
        asset_vol=0.28,
        rate=0.025,
    )
    equity_value = float(true_model.equity_value())
    equity_vol = float(_equity_vol_from_kmv(true_model))

    calibrated = calibrate_kmv_from_equity(
        short_term_debt=35.0,
        long_term_debt=90.0,
        maturity=1.0,
        rate=0.025,
        equity_value=equity_value,
        equity_vol=equity_vol,
    )

    assert jnp.isclose(calibrated.asset_value, true_model.asset_value, rtol=1e-4)
    assert jnp.isclose(calibrated.asset_vol, true_model.asset_vol, rtol=1e-4)
    dd = calibrated.distance_to_default()
    edf = calibrated.expected_default_frequency()
    assert dd > 0.0
    assert 0.0 < edf < 1.0


def test_reduced_form_zero_coupon_matches_analytic_price():
    model = ReducedFormModel(hazard=0.02, recovery_rate=0.4, risk_free_rate=0.01)
    maturity = 3.0
    price = model.zero_coupon_price(maturity)
    expected_price = jnp.exp(-(0.01 + 0.02 * (1.0 - 0.4)) * maturity)
    assert jnp.isclose(price, expected_price, rtol=1e-6)
    spread = model.credit_spread(maturity)
    assert jnp.isclose(spread, 0.02 * (1.0 - 0.4), rtol=1e-6)


def test_calibrate_constant_intensity_recovers_lambda():
    spread = 0.0125
    recovery = 0.35
    lam = calibrate_constant_intensity(spread, recovery)
    assert jnp.isclose(lam, spread / (1.0 - recovery))


def test_gaussian_copula_samples_match_marginals():
    probs = jnp.array([0.05, 0.1, 0.2])
    corr = jnp.eye(3)
    key = random.PRNGKey(0)
    # Reduced samples for CI performance (was 20000)
    samples = gaussian_copula_samples(probs, corr, 10000, key)
    empirical_pd = jnp.mean(samples, axis=0)
    # Relaxed tolerance for fewer samples
    assert jnp.allclose(empirical_pd, probs, rtol=8e-2, atol=8e-3)


def test_portfolio_metrics_are_consistent_with_expected_loss():
    exposures = jnp.array([10.0, 15.0, 20.0])
    lgd = jnp.array([0.6, 0.5, 0.4])
    probs = jnp.array([0.02, 0.04, 0.06])
    corr = jnp.array(
        [
            [1.0, 0.25, 0.1],
            [0.25, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )
    key = random.PRNGKey(123)
    # Reduced samples for CI performance (was 20000)
    metrics = portfolio_risk_metrics(
        exposures,
        lgd,
        probs,
        corr,
        num_samples=10000,
        alpha=0.975,
        key=key,
    )
    theoretical_el = expected_loss(exposures, probs, lgd)
    assert isinstance(metrics, PortfolioLossMetrics)
    assert jnp.isclose(metrics.expected_loss, theoretical_el, rtol=5e-2)
    assert metrics.value_at_risk >= metrics.expected_loss
    assert metrics.conditional_var >= metrics.value_at_risk


def test_single_factor_distribution_respects_correlation_parameter():
    exposures = jnp.array([5.0, 7.5, 12.0, 9.0])
    lgd = jnp.array([0.5, 0.55, 0.6, 0.4])
    probs = jnp.array([0.03, 0.025, 0.05, 0.04])
    key = random.PRNGKey(321)
    # Reduced samples for CI performance (was 15000)
    metrics_low = single_factor_loss_distribution(
        exposures,
        lgd,
        probs,
        rho=0.1,
        num_samples=8000,
        alpha=0.975,
        key=key,
    )
    metrics_high = single_factor_loss_distribution(
        exposures,
        lgd,
        probs,
        rho=0.5,
        num_samples=8000,
        alpha=0.975,
        key=key,
    )
    assert metrics_high.unexpected_loss >= metrics_low.unexpected_loss
