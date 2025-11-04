"""SABR (Stochastic Alpha Beta Rho) volatility model.

The SABR model is widely used in interest rate and FX markets for modeling
the volatility smile. It provides analytical approximations for implied
volatility and supports full calibration to market data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import jax
import jax.numpy as jnp
import optax

from neutryx.core.utils.precision import apply_loss_scaling, get_loss_scale, undo_loss_scaling
from neutryx.infrastructure.tracking import (
    BaseTracker,
    TrackingConfig,
    calibration_metric_template,
    calibration_param_template,
    resolve_tracker,
)

Array = jnp.ndarray


@dataclass
class SABRParams:
    """SABR model parameters.

    The SABR model dynamics:
        dF = alpha * F^beta * dW_1
        d(alpha) = nu * alpha * dW_2
        corr(dW_1, dW_2) = rho

    Attributes
    ----------
    alpha : float
        Initial volatility (ATM volatility parameter)
    beta : float
        CEV exponent (0 = normal, 0.5 = CIR, 1 = lognormal)
    rho : float
        Correlation between forward and volatility (-1 to 1)
    nu : float
        Volatility of volatility (vol-of-vol)
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        """Validate parameters (only when not inside JAX transformation)."""
        try:
            # Only validate concrete values, not tracers
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            if not (0 <= self.beta <= 1):
                raise ValueError(f"beta must be in [0, 1], got {self.beta}")
            if not (-1 <= self.rho <= 1):
                raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
            if self.nu < 0:
                raise ValueError(f"nu must be non-negative, got {self.nu}")
        except (jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError):
            # Skip validation inside JAX transformations
            pass


def sabr_implied_volatility_hagan(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Compute SABR implied volatility using Hagan's approximation.

    This is the original Hagan et al. (2002) approximation formula.

    Parameters
    ----------
    F : float
        Forward price
    K : float
        Strike price
    T : float
        Time to expiry
    alpha : float
        Initial volatility
    beta : float
        CEV exponent
    rho : float
        Correlation
    nu : float
        Vol-of-vol

    Returns
    -------
    float
        Implied volatility
    """
    F = jnp.asarray(F)
    K = jnp.asarray(K)

    # Handle ATM case
    eps = 1e-7
    is_atm = jnp.abs(F - K) < eps

    # Pre-compute powers
    FK_mid = (F + K) / 2.0
    F_beta = jnp.power(F, 1.0 - beta)
    F_2beta = jnp.power(F, 2.0 * (1.0 - beta))

    # First term
    log_FK = jnp.log(F / K)
    FK_beta = jnp.power(F * K, (1.0 - beta) / 2.0)

    term1_numerator = alpha
    term1_denominator = FK_beta * (
        1.0
        + ((1.0 - beta) ** 2 / 24.0) * (log_FK ** 2)
        + ((1.0 - beta) ** 4 / 1920.0) * (log_FK ** 4)
    )

    term1 = term1_numerator / term1_denominator

    # Second term (z / x(z))
    z = (nu / alpha) * FK_beta * jnp.log(F / K)
    x_z = jnp.log((jnp.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))

    # Avoid division by zero
    x_z = jnp.where(jnp.abs(z) < eps, 1.0, x_z)
    z = jnp.where(jnp.abs(z) < eps, eps, z)

    term2 = z / x_z

    # Third term (time-dependent correction)
    term3 = (
        1.0
        + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / F_2beta)
            + (rho * beta * nu * alpha / (4.0 * F_beta))
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
        )
        * T
    )

    # ATM implied vol
    atm_vol = (alpha / F_beta) * term3

    # Full formula
    sigma_impl = term1 * term2 * term3

    # Return ATM vol when F ≈ K
    result = jnp.where(is_atm, atm_vol, sigma_impl)

    return jnp.maximum(result, 1e-8)


def hagan_implied_vol(F: float, K: float, T: float, params: SABRParams) -> float:
    """Compute SABR implied volatility using Hagan's formula (convenience wrapper).

    This is a convenience function that wraps sabr_implied_volatility_hagan
    with a SABRParams object.

    Parameters
    ----------
    F : float
        Forward price
    K : float
        Strike price
    T : float
        Time to maturity
    params : SABRParams
        SABR parameters

    Returns
    -------
    float
        Implied volatility (annualized)
    """
    return sabr_implied_volatility_hagan(
        F, K, T, params.alpha, params.beta, params.rho, params.nu
    )


def sabr_call_price(
    F: float,
    K: float,
    T: float,
    r: float,
    params: SABRParams,
) -> float:
    """Compute call option price under SABR using normal Black formula.

    Parameters
    ----------
    F : float
        Forward price
    K : float
        Strike
    T : float
        Time to expiry
    r : float
        Discount rate
    params : SABRParams
        SABR parameters

    Returns
    -------
    float
        Call option price
    """
    from jax.scipy.stats import norm

    # Get SABR implied vol
    sigma = sabr_implied_volatility_hagan(
        F, K, T, params.alpha, params.beta, params.rho, params.nu
    )

    # Black's formula (can use normal or lognormal depending on beta)
    if params.beta == 1.0:
        # Lognormal Black
        d1 = (jnp.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)
        price = jnp.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif params.beta == 0.0:
        # Normal Black (Bachelier)
        d = (F - K) / (sigma * jnp.sqrt(T))
        price = jnp.exp(-r * T) * (
            (F - K) * norm.cdf(d) + sigma * jnp.sqrt(T) * jnp.exp(-0.5 * d ** 2) / jnp.sqrt(2.0 * jnp.pi)
        )
    else:
        # General case: use lognormal approximation
        d1 = (jnp.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)
        price = jnp.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    return price


def calibrate_sabr(
    F: float,
    strikes: Array,
    T: float | Array,
    market_vols: Array,
    beta: float = 0.5,
    initial_alpha: float | None = None,
    initial_rho: float = 0.0,
    initial_nu: float = 0.3,
    n_iterations: int = 500,
    lr: float = 1e-2,
    use_loss_scaling: bool = False,
    tracker: Optional[BaseTracker] = None,
    tracking_config: Optional[TrackingConfig | Mapping[str, Any]] = None,
    log_every: Optional[int] = None,
) -> SABRParams:
    """Calibrate SABR parameters to market implied volatilities.

    Parameters
    ----------
    F : float
        Forward price
    strikes : Array
        Strike prices
    T : float | Array
        Time to expiry (single value or array matching strikes)
    market_vols : Array
        Market implied volatilities
    beta : float
        Fixed beta parameter (typically 0.5 or 1.0)
    initial_alpha : float | None
        Initial alpha guess (if None, use ATM vol)
    initial_rho : float
        Initial rho guess
    initial_nu : float
        Initial nu guess
    n_iterations : int
        Number of optimization iterations
    lr : float
        Learning rate
    use_loss_scaling : bool
        Use mixed precision loss scaling (default: False)
    tracker : BaseTracker | None
        Tracking backend for logging calibration progress
    tracking_config : TrackingConfig | Mapping | None
        Configuration for tracking
    log_every : int | None
        Log metrics every N iterations (overrides tracking_config.log_every)

    Returns
    -------
    SABRParams
        Calibrated SABR parameters
    """
    strikes = jnp.asarray(strikes)
    market_vols = jnp.asarray(market_vols)

    # Handle scalar or array T
    if isinstance(T, (int, float)):
        T_val = float(T)
        maturities = jnp.full_like(strikes, T_val)
    else:
        maturities = jnp.asarray(T)

    # Initial guess for alpha (ATM vol if not provided)
    if initial_alpha is None:
        atm_idx = jnp.argmin(jnp.abs(strikes - F))
        initial_alpha = float(market_vols[atm_idx])

    # Parameters to optimize (alpha, rho, nu)
    # Use transformations to enforce constraints
    params = {
        "alpha_raw": jnp.log(jnp.array(initial_alpha)),  # log(alpha) for positivity
        "rho_raw": jnp.arctanh(jnp.array(initial_rho * 0.99)),  # arctanh for [-1, 1]
        "nu_raw": jnp.log(jnp.array(initial_nu)),  # log(nu) for positivity
    }

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def transform_params(params_raw):
        """Transform unconstrained params to SABR params."""
        alpha = jnp.exp(params_raw["alpha_raw"])
        rho = jnp.tanh(params_raw["rho_raw"])
        nu = jnp.exp(params_raw["nu_raw"])
        return alpha, rho, nu

    def loss_fn(params_raw):
        """Loss function: MSE between model and market vols."""
        alpha, rho, nu = transform_params(params_raw)

        # Compute model implied vols
        model_vols = jax.vmap(
            lambda K, T_i: sabr_implied_volatility_hagan(F, K, T_i, alpha, beta, rho, nu)
        )(strikes, maturities)

        # Mean squared error
        return jnp.mean((model_vols - market_vols) ** 2)

    def scaled_loss_fn(params_raw, loss_scale):
        """Loss with scaling for mixed precision."""
        return apply_loss_scaling(loss_fn(params_raw), loss_scale=loss_scale)

    # Setup tracking
    tracking_cfg: Optional[TrackingConfig]
    if isinstance(tracking_config, TrackingConfig):
        tracking_cfg = tracking_config
    elif tracking_config is not None:
        tracking_cfg = TrackingConfig(**dict(tracking_config))
    else:
        tracking_cfg = None

    log_cadence = log_every or (tracking_cfg.log_every if tracking_cfg else 25)

    # Optimization loop with tracking
    with resolve_tracker(tracker, tracking_cfg) as active_tracker:
        # Log initial parameters
        initial_dict = {
            "alpha": float(initial_alpha),
            "beta": beta,
            "rho": initial_rho,
            "nu": initial_nu,
        }
        active_tracker.log_params(
            calibration_param_template(initial_dict, prefix="sabr")
        )

        for i in range(n_iterations):
            if use_loss_scaling:
                loss_scale = get_loss_scale()
                loss_val_scaled, grads_scaled = jax.value_and_grad(
                    lambda p: scaled_loss_fn(p, loss_scale)
                )(params)
                grads = undo_loss_scaling(grads_scaled, loss_scale=loss_scale)
                loss_val = undo_loss_scaling(loss_val_scaled, loss_scale=loss_scale)
            else:
                loss_val, grads = jax.value_and_grad(loss_fn)(params)

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Log metrics
            if i % log_cadence == 0 or i == n_iterations - 1:
                param_dict = {
                    "alpha_raw": float(params["alpha_raw"]),
                    "rho_raw": float(params["rho_raw"]),
                    "nu_raw": float(params["nu_raw"]),
                }
                active_tracker.log_metrics(
                    calibration_metric_template(float(loss_val), param_dict, prefix="sabr"),
                    step=i,
                )

    # Extract final parameters
    alpha_final, rho_final, nu_final = transform_params(params)

    return SABRParams(
        alpha=float(alpha_final),
        beta=beta,
        rho=float(rho_final),
        nu=float(nu_final),
    )


def sabr_density(
    F: float,
    S: Array,
    T: float,
    params: SABRParams,
) -> Array:
    """Approximate probability density under SABR.

    Uses the relationship: density = d²C/dK² where C is call price.

    Parameters
    ----------
    F : float
        Forward price
    S : Array
        Spot prices for density evaluation
    T : float
        Time to expiry
    params : SABRParams
        SABR parameters

    Returns
    -------
    Array
        Probability density values
    """
    # Use finite difference to compute second derivative
    dK = 0.01

    def call_price(K):
        return sabr_call_price(F, K, T, 0.0, params)

    # Second derivative via central difference
    density = jax.vmap(
        lambda s: (call_price(s + dK) - 2 * call_price(s) + call_price(s - dK)) / (dK ** 2)
    )(S)

    return jnp.maximum(density, 0.0)


__all__ = [
    "SABRParams",
    "sabr_implied_volatility_hagan",
    "hagan_implied_vol",
    "sabr_call_price",
    "calibrate_sabr",
    "sabr_density",
]
