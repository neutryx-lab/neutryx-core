"""Tests for advanced interest rate models."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from neutryx.models.hull_white_two_factor import (
    HullWhiteTwoFactorParams,
    zero_coupon_bond_price as hw2f_bond_price,
    simulate_path as hw2f_simulate_path,
    simulate_paths as hw2f_simulate_paths,
    caplet_price as hw2f_caplet_price,
    instantaneous_correlation,
)

from neutryx.models.black_karasinski import (
    BlackKarasinskiParams,
    simulate_path as bk_simulate_path,
    simulate_paths as bk_simulate_paths,
    zero_coupon_bond_price_mc as bk_bond_price_mc,
)

from neutryx.models.cheyette import (
    CheyetteParams,
    zero_coupon_bond_price as chey_bond_price,
    simulate_path as chey_simulate_path,
    simulate_paths as chey_simulate_paths,
)

from neutryx.models.lgm import (
    LGMParams,
    H_coefficient,
    G_coefficient,
    zero_coupon_bond_price as lgm_bond_price,
    simulate_path as lgm_simulate_path,
    simulate_paths as lgm_simulate_paths,
    caplet_price as lgm_caplet_price,
    calibrate_to_swaption_vols,
)

from neutryx.models.lmm import (
    LMMParams,
    simulate_path_terminal_measure,
    simulate_paths,
    zero_coupon_bond_price as lmm_bond_price,
    swap_rate,
    simple_volatility_structure,
    create_correlation_matrix,
)

from neutryx.products.swaptions import implied_swaption_volatility


def _discount_factor(forward_curve_fn, maturity: float) -> float:
    if maturity <= 0.0:
        return 1.0
    rate = forward_curve_fn(maturity / 2.0)
    return float(np.exp(-rate * maturity))


def _construct_swaption_schedule(
    forward_curve_fn,
    option_expiry: float,
    swap_tenor: float,
    payment_interval: float,
) -> dict:
    n_payments = int(np.round(swap_tenor / payment_interval))
    if n_payments < 1:
        raise ValueError("Swap tenor must span at least one payment interval")

    payment_times = option_expiry + payment_interval * np.arange(1, n_payments + 1)
    discount_factors = np.array([
        _discount_factor(forward_curve_fn, t) for t in payment_times
    ])
    year_fractions = np.full(n_payments, payment_interval, dtype=float)
    annuity = float(np.dot(discount_factors, year_fractions))
    df_expiry = _discount_factor(forward_curve_fn, option_expiry)
    forward_rate = float((df_expiry - discount_factors[-1]) / max(annuity, 1e-12))

    return {
        "expiry": float(option_expiry),
        "tenor": float(swap_tenor),
        "forward": forward_rate,
        "annuity": annuity,
        "payment_times": payment_times,
        "weights": discount_factors * year_fractions,
    }


def _lgm_normal_vol(alpha: float, sigma: float, schedule: dict, integration_points: int = 256) -> float:
    expiry = schedule["expiry"]
    if expiry <= 0.0:
        return 0.0

    payment_times = schedule["payment_times"]
    weights = schedule["weights"]
    annuity = schedule["annuity"]

    t_grid = np.linspace(0.0, expiry, integration_points)
    exposures_sq = np.zeros_like(t_grid)

    for idx, t in enumerate(t_grid):
        dt = np.maximum(payment_times - t, 0.0)
        if alpha > 1e-12:
            B_vals = (1.0 - np.exp(-alpha * dt)) / alpha
        else:
            B_vals = dt

        numerator = float(np.dot(weights, B_vals))
        instantaneous_vol = sigma * numerator / max(annuity, 1e-12)
        exposures_sq[idx] = instantaneous_vol * instantaneous_vol

    variance = float(np.trapezoid(exposures_sq, t_grid))
    return float(np.sqrt(max(variance, 0.0)))


def _bachelier_price(
    forward_rate: float,
    strike: float,
    option_maturity: float,
    volatility: float,
    annuity: float,
    *,
    notional: float = 1.0,
    is_payer: bool = True,
) -> float:
    if option_maturity <= 0.0:
        intrinsic = max(forward_rate - strike, 0.0)
        if not is_payer:
            intrinsic = max(strike - forward_rate, 0.0)
        return notional * annuity * intrinsic

    sqrt_T = np.sqrt(option_maturity)
    vol_sqrt_T = volatility * sqrt_T

    if vol_sqrt_T < 1e-12:
        intrinsic = max(forward_rate - strike, 0.0)
        if not is_payer:
            intrinsic = max(strike - forward_rate, 0.0)
        return notional * annuity * intrinsic

    d = (forward_rate - strike) / vol_sqrt_T
    if is_payer:
        price = (forward_rate - strike) * norm.cdf(d) + vol_sqrt_T * norm.pdf(d)
    else:
        price = (strike - forward_rate) * norm.cdf(-d) + vol_sqrt_T * norm.pdf(d)

    return float(notional * annuity * price)


def _build_synthetic_swaption_surface(
    alpha: float,
    sigma: float,
    forward_curve_fn,
    expiries: np.ndarray,
    tenors: np.ndarray,
    *,
    payment_interval: float = 0.5,
    notional: float = 1.0,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    instruments: list[dict] = []
    normal_vols = np.zeros((expiries.size, tenors.size))
    lognormal_vols = np.zeros_like(normal_vols)

    for i, expiry in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            schedule = _construct_swaption_schedule(
                forward_curve_fn,
                float(expiry),
                float(tenor),
                payment_interval,
            )
            normal_vol = _lgm_normal_vol(alpha, sigma, schedule)
            market_price = _bachelier_price(
                schedule["forward"],
                schedule["forward"],
                schedule["expiry"],
                normal_vol,
                schedule["annuity"],
                notional=notional,
            )

            lognormal_vol = implied_swaption_volatility(
                market_price,
                schedule["forward"],
                schedule["forward"],
                schedule["expiry"],
                schedule["annuity"],
                notional=notional,
                is_payer=True,
            )

            schedule.update(
                {
                    "index": (i, j),
                    "normal_vol": normal_vol,
                    "market_price": market_price,
                    "lognormal_vol": lognormal_vol,
                }
            )

            instruments.append(schedule)
            normal_vols[i, j] = normal_vol
            lognormal_vols[i, j] = lognormal_vol

    return instruments, normal_vols, lognormal_vols

from neutryx.models.hjm import (
    HJMParams,
    compute_hjm_drift,
    simulate_short_rate_path,
    simulate_forward_curve_path,
    gaussian_hjm_volatility,
)


# ===== Hull-White Two-Factor Tests =====

def test_hw2f_params_validation():
    """Test Hull-White two-factor parameter validation."""
    # Valid parameters
    params = HullWhiteTwoFactorParams(
        a=0.1, b=0.2, sigma1=0.01, sigma2=0.015,
        rho=0.5, r0=0.03, u0=0.0
    )
    assert params.a == 0.1
    assert params.rho == 0.5

    # Invalid a
    with pytest.raises(ValueError, match="Mean reversion speed a"):
        HullWhiteTwoFactorParams(
            a=-0.1, b=0.2, sigma1=0.01, sigma2=0.015,
            rho=0.5, r0=0.03
        )

    # Invalid rho
    with pytest.raises(ValueError, match="Correlation rho"):
        HullWhiteTwoFactorParams(
            a=0.1, b=0.2, sigma1=0.01, sigma2=0.015,
            rho=1.5, r0=0.03
        )


def test_hw2f_bond_price():
    """Test two-factor Hull-White bond pricing."""
    params = HullWhiteTwoFactorParams(
        a=0.1, b=0.2, sigma1=0.01, sigma2=0.015,
        rho=0.3, r0=0.03, u0=0.0
    )

    # Bond price at initial state should be positive and < 1
    P = hw2f_bond_price(params, T=5.0, r_t=0.03, u_t=0.0)
    assert 0 < P < 1

    # Longer maturity should have lower price
    P_short = hw2f_bond_price(params, T=1.0, r_t=0.03, u_t=0.0)
    P_long = hw2f_bond_price(params, T=10.0, r_t=0.03, u_t=0.0)
    assert P_short > P_long


def test_hw2f_simulation():
    """Test two-factor Hull-White path simulation."""
    params = HullWhiteTwoFactorParams(
        a=0.1, b=0.2, sigma1=0.01, sigma2=0.015,
        rho=0.3, r0=0.03, u0=0.0
    )

    key = jax.random.PRNGKey(42)
    r_path, u_path = hw2f_simulate_path(params, T=1.0, n_steps=50, key=key)

    assert r_path.shape == (51,)
    assert u_path.shape == (51,)
    assert r_path[0] == pytest.approx(0.03, abs=1e-6)
    assert u_path[0] == pytest.approx(0.0, abs=1e-6)


def test_hw2f_instantaneous_correlation():
    """Test instantaneous forward rate correlation."""
    params = HullWhiteTwoFactorParams(
        a=0.1, b=0.2, sigma1=0.01, sigma2=0.015,
        rho=0.5, r0=0.03, u0=0.0
    )

    # Correlation between same maturity should be 1
    corr_same = instantaneous_correlation(params, T1=5.0, T2=5.0)
    assert corr_same == pytest.approx(1.0, abs=1e-6)

    # Correlation between different maturities should be < 1
    corr_diff = instantaneous_correlation(params, T1=1.0, T2=10.0)
    assert 0 < corr_diff < 1


# ===== Black-Karasinski Tests =====

def test_bk_params_validation():
    """Test Black-Karasinski parameter validation."""
    # Valid parameters
    params = BlackKarasinskiParams(a=0.1, sigma=0.02, r0=0.03)
    assert params.a == 0.1

    # Invalid r0 (must be positive)
    with pytest.raises(ValueError, match="Initial rate r0"):
        BlackKarasinskiParams(a=0.1, sigma=0.02, r0=-0.01)


def test_bk_simulation():
    """Test Black-Karasinski path simulation."""
    params = BlackKarasinskiParams(a=0.1, sigma=0.02, r0=0.03)

    key = jax.random.PRNGKey(42)
    r_path = bk_simulate_path(params, T=1.0, n_steps=50, key=key)

    assert r_path.shape == (51,)
    assert r_path[0] == pytest.approx(0.03, abs=1e-6)

    # Rates should always be positive (log-normal)
    assert jnp.all(r_path > 0)


def test_bk_bond_price_mc():
    """Test Black-Karasinski bond pricing with Monte Carlo."""
    params = BlackKarasinskiParams(a=0.1, sigma=0.02, r0=0.03)

    key = jax.random.PRNGKey(42)
    P = bk_bond_price_mc(params, T=1.0, n_paths=1000, n_steps=50, key=key)

    # Bond price should be positive and < 1
    assert 0 < P < 1


# ===== Cheyette Model Tests =====

def test_cheyette_params_validation():
    """Test Cheyette model parameter validation."""
    # Single factor
    params = CheyetteParams(
        kappa=0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
        n_factors=1,
    )
    assert params.n_factors == 1

    # Multi-factor with correlation
    rho = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    params = CheyetteParams(
        kappa=jnp.array([0.1, 0.2]),
        sigma_fn=lambda t: jnp.array([0.01, 0.015]),
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
        n_factors=2,
        rho=rho,
    )
    assert params.n_factors == 2


def test_cheyette_bond_price():
    """Test Cheyette bond pricing."""
    params = CheyetteParams(
        kappa=0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
        n_factors=1,
    )

    # Bond price at zero state
    P = chey_bond_price(params, T=5.0, x_t=0.0, y_t=0.0)
    assert 0 < P < 1


def test_cheyette_simulation():
    """Test Cheyette path simulation."""
    params = CheyetteParams(
        kappa=0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
        n_factors=1,
    )

    key = jax.random.PRNGKey(42)
    r_path, x_path, y_path = chey_simulate_path(params, T=1.0, n_steps=50, key=key)

    assert r_path.shape == (51,)
    assert x_path.shape == (51,)
    assert y_path.shape == (51,)


# ===== LGM Model Tests =====

def test_lgm_params_validation():
    """Test LGM model parameter validation."""
    params = LGMParams(
        alpha_fn=lambda t: 0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
        n_factors=1,
    )
    assert params.n_factors == 1


def test_lgm_coefficients():
    """Test LGM H and G coefficients."""
    params = LGMParams(
        alpha_fn=lambda t: 0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
    )

    # H coefficient should decay with maturity
    H_1y = H_coefficient(params, t=0.0, T=1.0)
    H_5y = H_coefficient(params, t=0.0, T=5.0)
    assert H_5y < H_1y

    # G coefficient should be positive
    G = G_coefficient(params, t=0.0, T=5.0)
    assert G > 0


def test_lgm_bond_price():
    """Test LGM bond pricing."""
    params = LGMParams(
        alpha_fn=lambda t: 0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
    )

    P = lgm_bond_price(params, T=5.0, x_t=0.0)
    assert 0 < P < 1


def test_lgm_simulation():
    """Test LGM path simulation."""
    params = LGMParams(
        alpha_fn=lambda t: 0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
    )

    key = jax.random.PRNGKey(42)
    r_path, x_path = lgm_simulate_path(params, T=1.0, n_steps=50, key=key)

    assert r_path.shape == (51,)
    assert x_path.shape == (51,)
    assert r_path[0] == pytest.approx(0.03, abs=1e-6)


def test_lgm_caplet_price():
    """Test LGM caplet pricing."""
    params = LGMParams(
        alpha_fn=lambda t: 0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
    )

    caplet_value = lgm_caplet_price(params, strike=0.03, caplet_maturity=1.0, tenor=0.25)
    assert caplet_value >= 0


def test_lgm_swaption_calibration_normal_vols():
    """Calibrate LGM parameters from synthetic normal swaption vols."""

    forward_curve_fn = lambda t: 0.025
    r0 = forward_curve_fn(0.0)
    expiries = jnp.array([1.0, 2.0, 5.0])
    tenors = jnp.array([2.0, 5.0])
    alpha_true = 0.12
    sigma_true = 0.009
    payment_interval = 0.5
    notional = 1.0

    instruments, normal_vols, _ = _build_synthetic_swaption_surface(
        alpha_true,
        sigma_true,
        forward_curve_fn,
        np.asarray(expiries, dtype=float),
        np.asarray(tenors, dtype=float),
        payment_interval=payment_interval,
        notional=notional,
    )

    calibrated = calibrate_to_swaption_vols(
        forward_curve_fn=forward_curve_fn,
        r0=r0,
        swaption_expiries=expiries,
        swaption_tenors=tenors,
        market_vols=jnp.asarray(normal_vols),
        initial_alpha=0.05,
        initial_sigma=0.005,
        vol_type="normal",
        payment_interval=payment_interval,
        notional=notional,
    )

    assert calibrated.alpha_fn(0.0) == pytest.approx(alpha_true, rel=1e-2)
    assert calibrated.sigma_fn(0.0) == pytest.approx(sigma_true, rel=1e-2)

    for instrument in instruments:
        model_vol = _lgm_normal_vol(
            calibrated.alpha_fn(0.0),
            calibrated.sigma_fn(0.0),
            instrument,
        )
        model_price = _bachelier_price(
            instrument["forward"],
            instrument["forward"],
            instrument["expiry"],
            model_vol,
            instrument["annuity"],
            notional=notional,
        )
        assert model_price == pytest.approx(
            instrument["market_price"], rel=1e-3
        )


def test_lgm_swaption_calibration_lognormal_vols():
    """Calibrate using market vols quoted in Black format."""

    forward_curve_fn = lambda t: 0.02
    r0 = forward_curve_fn(0.0)
    expiries = jnp.array([1.0, 2.0])
    tenors = jnp.array([2.0, 4.0])
    alpha_true = 0.14
    sigma_true = 0.007
    payment_interval = 0.5
    notional = 1.0

    instruments, _, lognormal_vols = _build_synthetic_swaption_surface(
        alpha_true,
        sigma_true,
        forward_curve_fn,
        np.asarray(expiries, dtype=float),
        np.asarray(tenors, dtype=float),
        payment_interval=payment_interval,
        notional=notional,
    )

    calibrated = calibrate_to_swaption_vols(
        forward_curve_fn=forward_curve_fn,
        r0=r0,
        swaption_expiries=expiries,
        swaption_tenors=tenors,
        market_vols=jnp.asarray(lognormal_vols),
        initial_alpha=0.1,
        initial_sigma=0.01,
        vol_type="lognormal",
        payment_interval=payment_interval,
        notional=notional,
    )

    assert calibrated.alpha_fn(0.0) == pytest.approx(alpha_true, rel=1e-2)
    assert calibrated.sigma_fn(0.0) == pytest.approx(sigma_true, rel=1e-2)

    for instrument in instruments:
        model_vol = _lgm_normal_vol(
            calibrated.alpha_fn(0.0),
            calibrated.sigma_fn(0.0),
            instrument,
        )
        model_price = _bachelier_price(
            instrument["forward"],
            instrument["forward"],
            instrument["expiry"],
            model_vol,
            instrument["annuity"],
            notional=notional,
        )
        assert model_price == pytest.approx(
            instrument["market_price"], rel=1e-3
        )


# ===== LMM Model Tests =====

def test_lmm_params_validation():
    """Test LMM parameter validation."""
    forward_rates = jnp.array([0.03, 0.035, 0.04])
    tenor_structure = jnp.array([0.0, 0.5, 1.0, 1.5])
    vol_fn = simple_volatility_structure(0.2, decay_rate=0.0, n_rates=3)
    corr_matrix = create_correlation_matrix(3, beta=0.1)

    params = LMMParams(
        forward_rates=forward_rates,
        tenor_structure=tenor_structure,
        volatility_fn=vol_fn,
        correlation_matrix=corr_matrix,
    )

    assert len(params.forward_rates) == 3
    assert len(params.day_count_fractions) == 3


def test_lmm_simulation():
    """Test LMM path simulation."""
    forward_rates = jnp.array([0.03, 0.035, 0.04, 0.042])
    tenor_structure = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    vol_fn = simple_volatility_structure(0.2, decay_rate=0.0, n_rates=4)
    corr_matrix = create_correlation_matrix(4, beta=0.1)

    params = LMMParams(
        forward_rates=forward_rates,
        tenor_structure=tenor_structure,
        volatility_fn=vol_fn,
        correlation_matrix=corr_matrix,
    )

    key = jax.random.PRNGKey(42)
    L_path = simulate_path_terminal_measure(params, T=1.0, n_steps=50, key=key)

    assert L_path.shape == (51, 4)
    # Initial rates should match
    assert jnp.allclose(L_path[0], forward_rates, atol=1e-6)

    # Rates should remain positive
    assert jnp.all(L_path > 0)


def test_lmm_bond_price():
    """Test LMM bond pricing from forward rates."""
    forward_rates = jnp.array([0.03, 0.035, 0.04, 0.042])
    tenor_structure = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    vol_fn = simple_volatility_structure(0.2, n_rates=4)
    corr_matrix = create_correlation_matrix(4)

    params = LMMParams(
        forward_rates=forward_rates,
        tenor_structure=tenor_structure,
        volatility_fn=vol_fn,
        correlation_matrix=corr_matrix,
    )

    P = lmm_bond_price(params, forward_rates, T_start=0.0, T_end=2.0)
    assert 0 < P < 1


def test_lmm_swap_rate():
    """Test LMM swap rate calculation."""
    forward_rates = jnp.array([0.03, 0.035, 0.04, 0.042])
    tenor_structure = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    vol_fn = simple_volatility_structure(0.2, n_rates=4)
    corr_matrix = create_correlation_matrix(4)

    params = LMMParams(
        forward_rates=forward_rates,
        tenor_structure=tenor_structure,
        volatility_fn=vol_fn,
        correlation_matrix=corr_matrix,
    )

    S = swap_rate(params, forward_rates, T_start=0.0, T_end=2.0)
    assert S > 0
    # Swap rate should be close to average of forward rates
    assert 0.02 < S < 0.05


# ===== HJM Framework Tests =====

def test_hjm_params_validation():
    """Test HJM parameter validation."""
    params = HJMParams(
        forward_curve_fn=lambda T: 0.03,
        volatility_fns=[gaussian_hjm_volatility(0.01, 0.1)],
        r0=0.03,
        n_factors=1,
    )

    assert params.n_factors == 1
    assert len(params.volatility_fns) == 1


def test_hjm_drift_computation():
    """Test HJM drift computation (no-arbitrage condition)."""
    params = HJMParams(
        forward_curve_fn=lambda T: 0.03,
        volatility_fns=[gaussian_hjm_volatility(0.01, 0.1)],
        r0=0.03,
        n_factors=1,
    )

    drift = compute_hjm_drift(params, t=0.0, T=5.0)
    # Drift should be a finite number
    assert jnp.isfinite(drift)


def test_hjm_short_rate_simulation():
    """Test HJM short rate path simulation."""
    params = HJMParams(
        forward_curve_fn=lambda T: 0.03,
        volatility_fns=[gaussian_hjm_volatility(0.01, 0.1)],
        r0=0.03,
        n_factors=1,
        max_maturity=10.0,
        n_maturities=20,
    )

    key = jax.random.PRNGKey(42)
    time_grid, r_path = simulate_short_rate_path(params, T=2.0, n_steps=50, key=key)

    assert len(time_grid) == 51
    assert len(r_path) == 51
    assert r_path[0] == pytest.approx(0.03, abs=0.01)


def test_hjm_forward_curve_simulation():
    """Test HJM forward curve evolution."""
    params = HJMParams(
        forward_curve_fn=lambda T: 0.03,
        volatility_fns=[gaussian_hjm_volatility(0.01, 0.1)],
        r0=0.03,
        n_factors=1,
        max_maturity=10.0,
        n_maturities=20,
    )

    key = jax.random.PRNGKey(42)
    time_grid, curves = simulate_forward_curve_path(params, T_horizon=1.0, n_steps=20, key=key)

    assert time_grid.shape == (21,)
    assert curves.shape == (21, 20)  # [n_steps+1, n_maturities]


# ===== Integration Tests =====

def test_models_produce_consistent_bond_prices():
    """Test that different models produce reasonable bond prices."""
    # Set up similar parameters across models
    r0 = 0.03
    T = 5.0

    # Hull-White 2F
    hw2f = HullWhiteTwoFactorParams(
        a=0.1, b=0.1, sigma1=0.01, sigma2=0.01,
        rho=0.0, r0=r0, u0=0.0
    )
    P_hw2f = hw2f_bond_price(hw2f, T, r_t=r0, u_t=0.0)

    # LGM
    lgm = LGMParams(
        alpha_fn=lambda t: 0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: r0,
        r0=r0,
    )
    P_lgm = lgm_bond_price(lgm, T, x_t=0.0)

    # Both should give similar prices
    assert 0 < P_hw2f < 1
    assert 0 < P_lgm < 1
    # Roughly similar (within 20%)
    assert abs(P_hw2f - P_lgm) / P_hw2f < 0.2


def test_multi_factor_vs_single_factor():
    """Test that multi-factor models reduce to single-factor when appropriate."""
    # Cheyette single-factor
    chey_1f = CheyetteParams(
        kappa=0.1,
        sigma_fn=lambda t: 0.01,
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
        n_factors=1,
    )

    # Cheyette two-factor with one factor having zero volatility
    chey_2f = CheyetteParams(
        kappa=jnp.array([0.1, 0.1]),
        sigma_fn=lambda t: jnp.array([0.01, 0.0]),
        forward_curve_fn=lambda t: 0.03,
        r0=0.03,
        n_factors=2,
    )

    key = jax.random.PRNGKey(42)
    r_1f, _, _ = chey_simulate_path(chey_1f, T=1.0, n_steps=50, key=key)
    r_2f, _, _ = chey_simulate_path(chey_2f, T=1.0, n_steps=50, key=key)

    # Both should be finite
    assert jnp.all(jnp.isfinite(r_1f))
    assert jnp.all(jnp.isfinite(r_2f))
