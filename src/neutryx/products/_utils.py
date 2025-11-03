"""Internal utility functions for product pricing.

This module contains common mathematical operations and helper functions
used across multiple product types to reduce code duplication.
"""

from __future__ import annotations

import jax.numpy as jnp


def ensure_array(x: jnp.ndarray | float) -> jnp.ndarray:
    """Convert input to JAX array.

    Parameters
    ----------
    x : array-like or scalar
        Input to convert

    Returns
    -------
    jnp.ndarray
        JAX array
    """
    return jnp.asarray(x)


def vanilla_payoff(
    spot: jnp.ndarray | float,
    strike: float,
    is_call: bool = True
) -> jnp.ndarray:
    """Compute vanilla option payoff.

    Parameters
    ----------
    spot : array-like or scalar
        Spot price(s)
    strike : float
        Strike price
    is_call : bool, default=True
        True for call, False for put

    Returns
    -------
    jnp.ndarray
        Option payoff, max(intrinsic, 0)
    """
    spot = ensure_array(spot)
    intrinsic = spot - strike if is_call else strike - spot
    return jnp.maximum(intrinsic, 0.0)


def compute_d1_d2_fx(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute d1 and d2 for FX option pricing (Garman-Kohlhagen).

    Parameters
    ----------
    S : float
        Spot FX rate
    K : float
        Strike price
    T : float
        Time to maturity
    r_d : float
        Domestic risk-free rate
    r_f : float
        Foreign risk-free rate
    sigma : float
        Volatility

    Returns
    -------
    d1 : jnp.ndarray
        d1 parameter
    d2 : jnp.ndarray
        d2 parameter
    """
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def compute_d1_d2_lognormal(
    forward: float,
    strike: float,
    T: float,
    volatility: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute d1 and d2 for log-normal Black model.

    Parameters
    ----------
    forward : float
        Forward price/rate
    strike : float
        Strike price/rate
    T : float
        Time to maturity
    volatility : float
        Volatility

    Returns
    -------
    d1 : jnp.ndarray
        d1 parameter
    d2 : jnp.ndarray
        d2 parameter
    """
    sqrt_T = jnp.sqrt(T)
    log_moneyness = jnp.log(forward / strike)
    d1 = (log_moneyness + 0.5 * volatility**2 * T) / (volatility * sqrt_T)
    d2 = d1 - volatility * sqrt_T
    return d1, d2


def compute_coupon_payment(
    coupon_rate: float,
    face_value: float,
    frequency: int
) -> float:
    """Compute periodic coupon payment.

    Parameters
    ----------
    coupon_rate : float
        Annual coupon rate
    face_value : float
        Face value of bond
    frequency : int
        Payment frequency per year

    Returns
    -------
    float
        Coupon payment per period
    """
    return (coupon_rate * face_value) / frequency


def compute_discount_factors(
    n_periods: int,
    rate: float,
    method: str = "power",
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """Compute discount factors for multiple periods.

    Parameters
    ----------
    n_periods : int
        Number of periods
    rate : float
        Discount rate per period
    method : str, default="power"
        Discounting method: "power" uses (1+r)^-t, "exp" uses exp(-r*t)
    dtype : jnp.dtype, default=jnp.float32
        Data type for output

    Returns
    -------
    jnp.ndarray
        Array of discount factors for periods 1, 2, ..., n_periods
    """
    times = jnp.arange(1, n_periods + 1, dtype=dtype)
    if method == "power":
        return jnp.power(1.0 + rate, -times)
    elif method == "exp":
        return jnp.exp(-rate * times)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'power' or 'exp'.")


def extract_terminal(path: jnp.ndarray) -> jnp.ndarray:
    """Extract terminal value(s) from price path(s).

    Parameters
    ----------
    path : jnp.ndarray
        Price path(s), shape (..., n_steps) or (n_steps,)

    Returns
    -------
    jnp.ndarray
        Terminal value(s), shape (...,) or scalar
    """
    return path[..., -1]


def check_barrier_hit_up(
    path: jnp.ndarray,
    barrier: float
) -> jnp.ndarray:
    """Check if path hits upper barrier.

    Parameters
    ----------
    path : jnp.ndarray
        Price path
    barrier : float
        Upper barrier level

    Returns
    -------
    jnp.ndarray
        Boolean indicating if barrier was hit
    """
    return path.max() >= barrier


def check_barrier_hit_down(
    path: jnp.ndarray,
    barrier: float
) -> jnp.ndarray:
    """Check if path hits lower barrier.

    Parameters
    ----------
    path : jnp.ndarray
        Price path
    barrier : float
        Lower barrier level

    Returns
    -------
    jnp.ndarray
        Boolean indicating if barrier was hit
    """
    return path.min() <= barrier


def monte_carlo_price(
    payoffs: jnp.ndarray,
    r: float,
    T: float
) -> float:
    """Compute Monte Carlo option price from payoffs.

    Parameters
    ----------
    payoffs : jnp.ndarray
        Payoffs for each path
    r : float
        Risk-free rate
    T : float
        Time to maturity

    Returns
    -------
    float
        Discounted average payoff
    """
    discount = jnp.exp(-r * T)
    return float((discount * payoffs).mean())
