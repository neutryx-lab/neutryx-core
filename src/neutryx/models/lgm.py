"""Linear Gaussian Markov (LGM) model implementation with JAX.

The Linear Gaussian Markov (LGM) model is a modern reformulation of the
Hull-White model that explicitly parameterizes the model in terms of
market observables (forward rates and volatilities).

The model describes a single state variable x(t):

    dx(t) = -α(t) * x(t) dt + σ(t) dW(t)

where:
    - x(t): state variable (Gaussian, mean-reverting to zero)
    - α(t): time-dependent mean reversion speed
    - σ(t): time-dependent volatility
    - W(t): standard Brownian motion

The short rate is:
    r(t) = f^M(0, t) + H(t, T) * x(t)

where:
    - f^M(0, t): market instantaneous forward rate at time 0 for time t
    - H(t, T): bond coefficient function

Key features:
1. Exact fit to initial forward curve (by construction)
2. Calibration to market swaption volatilities
3. Affine bond prices (analytical formulas)
4. Markovian (single state variable for single-factor)
5. Gaussian rates (can be negative, unlike log-normal models)

The LGM model is essentially equivalent to Hull-White but with a different
parameterization that makes calibration more intuitive and stable.

For multi-factor LGM:
    dx_i(t) = -α_i(t) * x_i(t) dt + σ_i(t) dW_i(t)
    r(t) = f^M(0, t) + Σᵢ H_i(t, T) * x_i(t)

References
----------
Hagan, P., & Lesniewski, A. (2008). "LIBOR market model with SABR style
stochastic volatility." Working paper.

Andersen, L., & Piterbarg, V. (2010). "Interest Rate Modeling."
Atlantic Financial Press. (Chapter on LGM)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.optimize import least_squares
from scipy.stats import norm

from neutryx.products.swaptions import black_swaption_price


@dataclass
class LGMParams:
    """Parameters for the Linear Gaussian Markov model.

    Attributes
    ----------
    alpha_fn : Callable
        Time-dependent mean reversion function α(t). Returns scalar or array.
    sigma_fn : Callable
        Time-dependent volatility function σ(t). Returns scalar or array.
    forward_curve_fn : Callable
        Market forward curve f^M(0, t)
    r0 : float
        Initial short rate
    n_factors : int, optional
        Number of factors (default: 1)
    rho : Optional[Array]
        Correlation matrix between factors (n_factors × n_factors).
        If None, assumes independent factors.
    """
    alpha_fn: Callable[[float], float | jnp.ndarray]
    sigma_fn: Callable[[float], float | jnp.ndarray]
    forward_curve_fn: Callable[[float], float]
    r0: float
    n_factors: int = 1
    rho: Optional[jnp.ndarray] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.n_factors < 1:
            raise ValueError(f"n_factors must be >= 1, got {self.n_factors}")

        if self.n_factors > 1:
            if self.rho is None:
                # Independent factors
                self.rho = jnp.eye(self.n_factors)
            else:
                self.rho = jnp.array(self.rho)
                if self.rho.shape != (self.n_factors, self.n_factors):
                    raise ValueError(
                        f"Correlation matrix shape {self.rho.shape} must be "
                        f"({self.n_factors}, {self.n_factors})"
                    )
        else:
            self.rho = jnp.array([[1.0]])


def H_coefficient(
    params: LGMParams,
    t: float,
    T: float,
) -> float | jnp.ndarray:
    """Compute the H coefficient H(t, T) for bond pricing.

    The H function satisfies:
        ∂H/∂t = -α(t) * H(t, T)
        H(T, T) = 0

    Analytical solution:
        H(t, T) = exp(-∫ₜᵀ α(s) ds)

    For piecewise constant α, this becomes:
        H(t, T) = exp(-α * (T - t))

    Parameters
    ----------
    params : LGMParams
        LGM model parameters
    t : float
        Current time
    T : float
        Maturity time

    Returns
    -------
    float or Array
        H coefficient(s). Scalar for single-factor, array for multi-factor.
    """
    # Simplified: assume piecewise constant α
    # For full implementation, numerically integrate α(s) from t to T

    if params.n_factors == 1:
        # Average α over [t, T]
        mid_t = (t + T) / 2.0
        alpha_avg = params.alpha_fn(mid_t)
        H = jnp.exp(-alpha_avg * (T - t))
    else:
        # Multi-factor
        mid_t = (t + T) / 2.0
        alphas = params.alpha_fn(mid_t)
        if jnp.isscalar(alphas):
            alphas = jnp.ones(params.n_factors) * alphas
        else:
            alphas = jnp.atleast_1d(alphas)
        H = jnp.exp(-alphas * (T - t))

    return H


def G_coefficient(
    params: LGMParams,
    t: float,
    T: float,
) -> float | jnp.ndarray:
    """Compute the G coefficient G(t, T) for variance.

    The G function is:
        G(t, T) = ∫ₜᵀ σ²(s) * H²(s, T) ds

    This captures the variance contribution of the state variable.

    Parameters
    ----------
    params : LGMParams
        LGM model parameters
    t : float
        Current time
    T : float
        Maturity time

    Returns
    -------
    float or Array
        G coefficient(s)
    """
    # Simplified: assume piecewise constant α and σ
    # For exact computation, numerically integrate

    if params.n_factors == 1:
        mid_t = (t + T) / 2.0
        alpha_avg = params.alpha_fn(mid_t)
        sigma_avg = params.sigma_fn(mid_t)

        # G ≈ σ² / (2α) * (1 - exp(-2α(T-t)))
        G = (sigma_avg * sigma_avg / (2.0 * alpha_avg)) * \
            (1.0 - jnp.exp(-2.0 * alpha_avg * (T - t)))
    else:
        mid_t = (t + T) / 2.0
        alphas = jnp.atleast_1d(params.alpha_fn(mid_t))
        sigmas = jnp.atleast_1d(params.sigma_fn(mid_t))

        if alphas.size == 1:
            alphas = jnp.ones(params.n_factors) * alphas[0]
        if sigmas.size == 1:
            sigmas = jnp.ones(params.n_factors) * sigmas[0]

        G = (sigmas * sigmas / (2.0 * alphas)) * \
            (1.0 - jnp.exp(-2.0 * alphas * (T - t)))

    return G


def zero_coupon_bond_price(
    params: LGMParams,
    T: float,
    x_t: float | jnp.ndarray,
    t: float = 0.0,
) -> float:
    """Calculate zero-coupon bond price under LGM model.

    The bond price has the affine form:
        P(t, T) = P^M(t, T) * exp(-H(t, T) · x(t) - 0.5 * G(t, T))

    where:
        - P^M(t, T): market bond price from forward curve
        - H(t, T): bond coefficient
        - G(t, T): variance term
        - x(t): state variable(s)

    Parameters
    ----------
    params : LGMParams
        LGM model parameters
    T : float
        Bond maturity time
    x_t : float or Array
        Current state variable(s). Scalar for single-factor, array for multi-factor.
    t : float, optional
        Current time (default: 0)

    Returns
    -------
    float
        Zero-coupon bond price
    """
    # Market bond price from forward curve
    # P^M(t, T) = exp(-∫ₜᵀ f^M(0, s) ds)
    tau = T - t
    avg_fwd = params.forward_curve_fn((t + T) / 2.0)
    P_market = jnp.exp(-avg_fwd * tau)

    # H and G coefficients
    H_t_T = H_coefficient(params, t, T)
    G_t_T = G_coefficient(params, t, T)

    if params.n_factors == 1:
        # Single factor
        adjustment = jnp.exp(-H_t_T * x_t - 0.5 * G_t_T)
    else:
        # Multi-factor
        x_t = jnp.atleast_1d(x_t)
        H_t_T = jnp.atleast_1d(H_t_T)
        G_t_T = jnp.atleast_1d(G_t_T)

        x_term = jnp.dot(H_t_T, x_t)
        G_term = 0.5 * jnp.sum(G_t_T)

        adjustment = jnp.exp(-x_term - G_term)

    bond_price = P_market * adjustment

    return float(bond_price)


def simulate_path(
    params: LGMParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate a single path of the LGM process.

    Parameters
    ----------
    params : LGMParams
        LGM model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    tuple[Array, Array]
        Tuple of (r_path, x_path) where:
        - r_path: short rate path of shape [n_steps + 1]
        - x_path: state variable path(s) of shape [n_steps + 1] or [n_steps + 1, n_factors]

    Notes
    -----
    The state variable follows:
        x(t+dt) = x(t) * exp(-α(t) dt) + σ(t) * sqrt((1 - exp(-2α(t) dt)) / (2α(t))) * Z
        r(t) = f^M(0, t) + x(t)  (for single-factor)
    """
    dt = T / n_steps

    if params.n_factors == 1:
        # Single factor
        x0 = 0.0

        # Generate random normals
        Z = jax.random.normal(key, shape=(n_steps,))

        def step_fn(carry, inputs):
            x_t, t = carry
            z = inputs

            # Get time-dependent parameters
            alpha_t = params.alpha_fn(t)
            sigma_t = params.sigma_fn(t)

            # Update state variable (exact for OU process)
            exp_neg_alpha_dt = jnp.exp(-alpha_t * dt)
            vol_scaling = jnp.sqrt((1.0 - jnp.exp(-2.0 * alpha_t * dt)) / (2.0 * alpha_t))

            x_next = x_t * exp_neg_alpha_dt + sigma_t * vol_scaling * z

            # Short rate
            r_t = params.forward_curve_fn(t) + x_t

            t_next = t + dt

            return (x_next, t_next), (r_t, x_next)

        _, (r_path, x_path) = lax.scan(step_fn, (x0, 0.0), Z)

        # Prepend initial values
        r0 = params.r0
        r_path_full = jnp.concatenate([jnp.array([r0]), r_path])
        x_path_full = jnp.concatenate([jnp.array([x0]), x_path])

    else:
        # Multi-factor case
        x0 = jnp.zeros(params.n_factors)

        # Generate correlated random normals
        Z_indep = jax.random.normal(key, shape=(n_steps, params.n_factors))
        L = jnp.linalg.cholesky(params.rho)
        Z_corr = Z_indep @ L.T

        def step_fn(carry, inputs):
            x_t, t = carry
            z = inputs  # Shape: [n_factors]

            # Get time-dependent parameters
            alphas_t = params.alpha_fn(t)
            sigmas_t = params.sigma_fn(t)

            # Ensure arrays
            if jnp.isscalar(alphas_t):
                alphas_t = jnp.ones(params.n_factors) * alphas_t
            else:
                alphas_t = jnp.atleast_1d(alphas_t)

            if jnp.isscalar(sigmas_t):
                sigmas_t = jnp.ones(params.n_factors) * sigmas_t
            else:
                sigmas_t = jnp.atleast_1d(sigmas_t)

            # Update state variables
            exp_neg_alpha_dt = jnp.exp(-alphas_t * dt)
            vol_scaling = jnp.sqrt((1.0 - jnp.exp(-2.0 * alphas_t * dt)) / (2.0 * alphas_t))

            x_next = x_t * exp_neg_alpha_dt + sigmas_t * vol_scaling * z

            # Short rate is forward rate plus sum of factors
            r_t = params.forward_curve_fn(t) + jnp.sum(x_t)

            t_next = t + dt

            return (x_next, t_next), (r_t, x_next)

        _, (r_path, x_path) = lax.scan(step_fn, (x0, 0.0), Z_corr)

        # Prepend initial values
        r0 = params.r0
        r_path_full = jnp.concatenate([jnp.array([r0]), r_path])
        x_path_full = jnp.concatenate([x0[None, :], x_path], axis=0)

    return r_path_full, x_path_full


def simulate_paths(
    params: LGMParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate multiple paths of the LGM process.

    Parameters
    ----------
    params : LGMParams
        LGM model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths to simulate
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    tuple[Array, Array]
        Tuple of (r_paths, x_paths) with appropriate shapes
    """
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k)

    r_paths, x_paths = jax.vmap(sim_single_path)(keys)
    return r_paths, x_paths


def caplet_price(
    params: LGMParams,
    strike: float,
    caplet_maturity: float,
    tenor: float,
) -> float:
    """Price a caplet under the LGM model using analytical formula.

    Similar to Hull-White, LGM admits analytical caplet pricing formulas.

    Parameters
    ----------
    params : LGMParams
        LGM model parameters
    strike : float
        Strike rate
    caplet_maturity : float
        Time to caplet maturity (reset date)
    tenor : float
        Accrual period (e.g., 0.25 for 3M LIBOR)

    Returns
    -------
    float
        Caplet price

    Notes
    -----
    Caplet = (1 + K*τ) * Put_on_bond(P(T, T+τ), K_bond)
    where K_bond = 1 / (1 + K*τ)
    """
    T = caplet_maturity
    tau = tenor

    # Bond equivalent strike
    K_bond = 1.0 / (1.0 + strike * tau)

    # Bond prices at current time
    x_0 = 0.0 if params.n_factors == 1 else jnp.zeros(params.n_factors)
    P_T = zero_coupon_bond_price(params, T, x_0)
    P_T_tau = zero_coupon_bond_price(params, T + tau, x_0)

    # H coefficient for the caplet period
    H_T_Ttau = H_coefficient(params, T, T + tau)

    # Volatility of bond price ratio at time T
    G_0_T = G_coefficient(params, 0.0, T)

    if params.n_factors == 1:
        sigma_P = float(jnp.sqrt(G_0_T)) * H_T_Ttau
    else:
        # Simplified for multi-factor
        G_0_T = jnp.atleast_1d(G_0_T)
        H_T_Ttau = jnp.atleast_1d(H_T_Ttau)
        sigma_P = float(jnp.sqrt(jnp.sum(G_0_T * H_T_Ttau * H_T_Ttau)))

    # Black-Scholes formula for put on bond
    if sigma_P > 1e-10:
        d1 = (jnp.log(P_T_tau / (K_bond * P_T)) + 0.5 * sigma_P * sigma_P) / sigma_P
        d2 = d1 - sigma_P

        # Put option value
        put_value = K_bond * P_T * norm.cdf(float(-d2)) - P_T_tau * norm.cdf(float(-d1))
    else:
        # Intrinsic value
        put_value = jnp.maximum(K_bond * P_T - P_T_tau, 0.0)

    # Convert put on bond to caplet
    caplet_value = (1.0 + strike * tau) * put_value

    return float(caplet_value)


def _discount_factor_from_forward(
    forward_curve_fn: Callable[[float], float], maturity: float
) -> float:
    """Approximate discount factor using the provided forward curve."""

    if maturity <= 0.0:
        return 1.0

    avg_forward = forward_curve_fn(maturity / 2.0)
    return float(np.exp(-avg_forward * maturity))


def _construct_swaption_instrument(
    forward_curve_fn: Callable[[float], float],
    option_expiry: float,
    swap_tenor: float,
    payment_interval: float,
) -> dict[str, float | np.ndarray]:
    """Pre-compute schedule quantities used during calibration."""

    if payment_interval <= 0.0:
        raise ValueError("payment_interval must be positive")

    if option_expiry <= 0.0:
        raise ValueError("Swaption expiry must be positive")

    n_payments = int(np.round(swap_tenor / payment_interval))
    if n_payments < 1:
        raise ValueError("Swap tenor must be at least one payment interval")

    payment_times = option_expiry + payment_interval * np.arange(1, n_payments + 1)
    discount_factors = np.array(
        [_discount_factor_from_forward(forward_curve_fn, t) for t in payment_times]
    )

    year_fractions = np.full(n_payments, payment_interval, dtype=float)
    annuity = float(np.dot(discount_factors, year_fractions))

    df_expiry = _discount_factor_from_forward(forward_curve_fn, option_expiry)
    forward_rate = float((df_expiry - discount_factors[-1]) / max(annuity, 1e-12))

    return {
        "option_expiry": option_expiry,
        "swap_tenor": swap_tenor,
        "forward_rate": forward_rate,
        "strike": forward_rate,  # Assume ATM surface for calibration
        "annuity": annuity,
        "discount_factors": discount_factors,
        "year_fractions": year_fractions,
        "payment_times": payment_times,
        "weights": discount_factors * year_fractions,
    }


def _bachelier_swaption_price(
    forward_swap_rate: float,
    strike: float,
    option_maturity: float,
    volatility: float,
    annuity: float,
    notional: float,
    is_payer: bool = True,
) -> float:
    """Price a swaption using Bachelier (normal) model."""

    if option_maturity <= 0.0:
        intrinsic = max(forward_swap_rate - strike, 0.0)
        if not is_payer:
            intrinsic = max(strike - forward_swap_rate, 0.0)
        return notional * annuity * intrinsic

    sqrt_T = float(np.sqrt(option_maturity))
    vol_sqrt_T = float(volatility * sqrt_T)

    if vol_sqrt_T < 1e-12:
        intrinsic = max(forward_swap_rate - strike, 0.0)
        if not is_payer:
            intrinsic = max(strike - forward_swap_rate, 0.0)
        return notional * annuity * intrinsic

    d = (forward_swap_rate - strike) / vol_sqrt_T
    pdf = norm.pdf(d)
    if is_payer:
        price = (forward_swap_rate - strike) * norm.cdf(d) + vol_sqrt_T * pdf
    else:
        price = (strike - forward_swap_rate) * norm.cdf(-d) + vol_sqrt_T * pdf

    return float(notional * annuity * price)


def _swaption_price_from_vol(
    instrument: dict[str, float | np.ndarray],
    volatility: float,
    vol_type: str,
    notional: float,
) -> float:
    """Convert a market volatility to a swaption price."""

    forward_rate = float(instrument["forward_rate"])
    strike = float(instrument["strike"])
    option_expiry = float(instrument["option_expiry"])
    annuity = float(instrument["annuity"])

    if vol_type == "normal":
        return _bachelier_swaption_price(
            forward_rate,
            strike,
            option_expiry,
            float(volatility),
            annuity,
            notional,
            True,
        )

    if vol_type == "lognormal":
        return float(
            black_swaption_price(
                forward_rate,
                strike,
                option_expiry,
                float(volatility),
                annuity,
                notional,
                is_payer=True,
            )
        )

    raise ValueError("vol_type must be either 'normal' or 'lognormal'")


def _compute_swaption_normal_variance(
    alpha: float,
    sigma: float,
    option_expiry: float,
    payment_times: np.ndarray,
    weights: np.ndarray,
    annuity: float,
    integration_points: int = 128,
) -> float:
    """Compute the variance of the forward swap rate under the LGM model."""

    if option_expiry <= 0.0:
        return 0.0

    t_grid = np.linspace(0.0, option_expiry, integration_points)
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
    return max(variance, 0.0)


def calibrate_to_swaption_vols(
    forward_curve_fn: Callable[[float], float],
    r0: float,
    swaption_expiries: jnp.ndarray,
    swaption_tenors: jnp.ndarray,
    market_vols: jnp.ndarray,
    initial_alpha: float = 0.1,
    initial_sigma: float = 0.01,
    *,
    vol_type: str = "normal",
    payment_interval: float = 0.5,
    notional: float = 1.0,
    integration_points: int = 128,
) -> LGMParams:
    """Calibrate LGM model to market swaption volatilities.

    This routine assumes a single-factor LGM model with piecewise
    constant parameters and calibrates constant α and σ values by
    minimizing pricing errors against an ATM swaption surface.

    Parameters
    ----------
    forward_curve_fn : Callable
        Market forward curve
    r0 : float
        Initial short rate
    swaption_expiries : Array
        Swaption expiry times
    swaption_tenors : Array
        Underlying swap tenors
    market_vols : Array
        Market swaption volatilities (normal or log-normal)
    initial_alpha : float, optional
        Initial guess for mean reversion (default: 0.1)
    initial_sigma : float, optional
        Initial guess for volatility (default: 0.01)
    vol_type : {"normal", "lognormal"}, optional
        Type of the provided market volatilities (default: "normal").
    payment_interval : float, optional
        Year fraction between swap payments (default: 0.5 for semi-annual).
    notional : float, optional
        Notional used for pricing in the objective (default: 1.0).
    integration_points : int, optional
        Number of discretization points used when integrating the swap
        rate variance (default: 128).

    Returns
    -------
    LGMParams
        Calibrated LGM parameters

    Notes
    -----
    Full calibration involves:
    1. Choose α(t) and σ(t) parameterization (e.g., piecewise constant)
    2. Minimize error between model and market swaption prices
    3. Iterate until convergence

    This simplified version uses constant parameters.
    """

    vol_type = vol_type.lower()

    expiries = np.asarray(jnp.atleast_1d(swaption_expiries), dtype=float)
    tenors = np.asarray(jnp.atleast_1d(swaption_tenors), dtype=float)
    market_vols_arr = np.asarray(market_vols, dtype=float)

    if expiries.ndim != 1 or tenors.ndim != 1:
        raise ValueError("swaption_expiries and swaption_tenors must be 1-D arrays")

    if market_vols_arr.shape != (expiries.size, tenors.size):
        raise ValueError(
            "market_vols must be of shape (n_expiries, n_tenors)"
        )

    instruments: list[dict[str, float | np.ndarray]] = []
    market_prices: list[float] = []

    for i, expiry in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            instrument = _construct_swaption_instrument(
                forward_curve_fn, float(expiry), float(tenor), payment_interval
            )
            market_vol = float(market_vols_arr[i, j])
            market_price = _swaption_price_from_vol(
                instrument, market_vol, vol_type, notional
            )
            instrument["market_vol"] = market_vol
            instrument["market_price"] = market_price
            instruments.append(instrument)
            market_prices.append(market_price)

    market_prices_arr = np.asarray(market_prices, dtype=float)

    def objective(x: np.ndarray) -> np.ndarray:
        alpha, sigma = float(x[0]), float(x[1])

        if alpha <= 0.0 or sigma <= 0.0:
            return market_prices_arr * 10.0

        residuals = []
        for instrument in instruments:
            variance = _compute_swaption_normal_variance(
                alpha,
                sigma,
                float(instrument["option_expiry"]),
                instrument["payment_times"],
                instrument["weights"],
                float(instrument["annuity"]),
                integration_points=integration_points,
            )
            normal_vol = float(np.sqrt(max(variance, 0.0)))
            model_price = _bachelier_swaption_price(
                float(instrument["forward_rate"]),
                float(instrument["strike"]),
                float(instrument["option_expiry"]),
                normal_vol,
                float(instrument["annuity"]),
                notional,
                True,
            )
            residuals.append(model_price - float(instrument["market_price"]))

        return np.asarray(residuals, dtype=float)

    x0 = np.array([initial_alpha, initial_sigma], dtype=float)
    lower_bounds = np.array([1e-6, 1e-6], dtype=float)
    upper_bounds = np.array([5.0, 1.0], dtype=float)

    result = least_squares(objective, x0=x0, bounds=(lower_bounds, upper_bounds))

    if not result.success:
        raise RuntimeError(
            f"Swaption calibration failed to converge: {result.message}"
        )

    calibrated_alpha = float(result.x[0])
    calibrated_sigma = float(result.x[1])

    def alpha_fn(_: float) -> float:
        return calibrated_alpha

    def sigma_fn(_: float) -> float:
        return calibrated_sigma

    return LGMParams(
        alpha_fn=alpha_fn,
        sigma_fn=sigma_fn,
        forward_curve_fn=forward_curve_fn,
        r0=r0,
        n_factors=1,
    )


__all__ = [
    "LGMParams",
    "H_coefficient",
    "G_coefficient",
    "zero_coupon_bond_price",
    "simulate_path",
    "simulate_paths",
    "caplet_price",
    "calibrate_to_swaption_vols",
]
