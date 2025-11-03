"""SABR (Stochastic Alpha Beta Rho) volatility model.

The SABR model is widely used in interest rate and FX markets for modeling
the volatility smile. It provides analytical approximations for implied
volatility and supports full calibration to market data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import optax

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

    def __post_init__(self):
        """Validate parameters."""
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.nu < 0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")


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

    # ATM formula
    FK_mid = (F + K) / 2.0
    FK_mid_beta = jnp.power(FK_mid, beta)

    z = (nu / alpha) * FK_mid_beta * jnp.log(F / K)
    x_z = jnp.log((jnp.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))

    # Avoid division by zero
    x_z = jnp.where(jnp.abs(z) < eps, 1.0, x_z)
    z = jnp.where(jnp.abs(z) < eps, eps, z)

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
    term2 = z / x_z

    # Third term (time-dependent correction)
    FK_2beta = jnp.power(FK_mid, 2.0 * beta)
    term3 = (
        1.0
        + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / FK_2beta)
            + (rho * beta * nu * alpha / (4.0 * FK_mid_beta))
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
        )
        * T
    )

    # ATM implied vol
    atm_vol = (alpha / jnp.power(FK_mid, 1.0 - beta)) * term3

    # Full formula
    sigma_impl = term1 * term2 * term3

    # Return ATM vol when F ≈ K
    result = jnp.where(is_atm, atm_vol, sigma_impl)

    return jnp.maximum(result, 1e-8)


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
    T: float,
    market_vols: Array,
    beta: float = 0.5,
    initial_alpha: float | None = None,
    initial_rho: float = 0.0,
    initial_nu: float = 0.3,
    n_iterations: int = 500,
    lr: float = 1e-2,
) -> SABRParams:
    """Calibrate SABR parameters to market implied volatilities.

    Parameters
    ----------
    F : float
        Forward price
    strikes : Array
        Strike prices
    T : float
        Time to expiry
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

    Returns
    -------
    SABRParams
        Calibrated SABR parameters
    """
    strikes = jnp.asarray(strikes)
    market_vols = jnp.asarray(market_vols)

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
            lambda K: sabr_implied_volatility_hagan(F, K, T, alpha, beta, rho, nu)
        )(strikes)

        # Mean squared error
        return jnp.mean((model_vols - market_vols) ** 2)

    # Optimization loop
    for _ in range(n_iterations):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

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
    "sabr_call_price",
    "calibrate_sabr",
    "sabr_density",
]
