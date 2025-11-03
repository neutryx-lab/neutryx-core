"""Hull-White one-factor short rate model implementation with JAX.

The Hull-White model is a one-factor model where the short rate follows:

    dr(t) = [θ(t) - a * r(t)]dt + σ dW(t)

where:
    - a: mean reversion speed (constant)
    - σ: volatility (constant)
    - θ(t): time-dependent drift that allows fitting to initial term structure
    - r(t): short rate at time t
    - W(t): standard Brownian motion

The Hull-White model is an extension of the Vasicek model with time-dependent
mean reversion level, allowing it to fit the current term structure of interest rates.

References
----------
Hull, J., & White, A. (1990). "Pricing interest-rate-derivative securities."
The Review of Financial Studies, 3(4), 573-592.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class HullWhiteParams:
    """Parameters for the Hull-White one-factor model.

    Attributes
    ----------
    a : float
        Mean reversion speed (must be positive)
    sigma : float
        Volatility (must be positive)
    r0 : float
        Initial short rate
    theta_fn : Optional[Callable]
        Time-dependent drift function θ(t). If None, uses constant drift.
    """
    a: float
    sigma: float
    r0: float
    theta_fn: Optional[Callable[[float], float]] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.a <= 0:
            raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")


def zero_coupon_bond_price(
    params: HullWhiteParams,
    T: float,
    forward_curve_fn: Optional[Callable[[float], float]] = None
) -> float:
    """Calculate zero-coupon bond price under Hull-White model.

    When fitted to an initial forward curve f(0, t), the bond price is:

        P(0, T) = A(0, T) * exp(-B(0, T) * r(0))

    where:
        B(0, T) = (1 - exp(-aT)) / a
        A(0, T) = P_market(0, T) / exp(-B(0, T) * f(0, 0)) * exp(V(0, T))
        V(0, T) = (σ² / (2a²)) * (B(0, T) - T) - (σ² / (4a)) * B(0, T)²

    Parameters
    ----------
    params : HullWhiteParams
        Hull-White model parameters
    T : float
        Time to maturity
    forward_curve_fn : Optional[Callable]
        Market forward curve f(0, t). If None, uses flat curve at r0.

    Returns
    -------
    float
        Zero-coupon bond price
    """
    a, sigma, r0 = params.a, params.sigma, params.r0

    # B(T) = (1 - exp(-aT)) / a
    B_T = (1.0 - jnp.exp(-a * T)) / a

    # V(T) term for volatility adjustment
    V_T = (sigma * sigma / (2.0 * a * a)) * (B_T - T) - \
          (sigma * sigma / (4.0 * a)) * B_T * B_T

    if forward_curve_fn is None:
        # Simple case: flat forward curve at r0
        # Reduces to Vasicek-like formula
        mean_level = r0
        A_T = jnp.exp((B_T - T) * mean_level + V_T)
    else:
        # Fit to market forward curve
        # This requires numerical integration of forward curve
        # For simplicity, we'll use flat approximation here
        # Full implementation would integrate f(0, s) from 0 to T
        f0 = forward_curve_fn(0.0)
        A_T = jnp.exp((B_T - T) * f0 + V_T)

    # P(0, T) = A(T) * exp(-B(T) * r0)
    return A_T * jnp.exp(-B_T * r0)


def simulate_path(
    params: HullWhiteParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    """Simulate a single path of the Hull-White short rate process.

    Parameters
    ----------
    params : HullWhiteParams
        Hull-White model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key
    method : str, optional
        Simulation method: "exact" (default) or "euler"

    Returns
    -------
    Array
        Short rate path of shape [n_steps + 1]

    Notes
    -----
    When θ(t) is constant, the exact simulation formula is:

        r(t + dt) = r(t) * exp(-a * dt) + α(t) * (1 - exp(-a * dt))
                    + σ * sqrt((1 - exp(-2 * a * dt)) / (2 * a)) * Z

    where α(t) = θ(t) / a
    """
    a, sigma, r0 = params.a, params.sigma, params.r0
    theta_fn = params.theta_fn
    dt = T / n_steps

    # Generate random normals
    Z = jax.random.normal(key, shape=(n_steps,))

    if method == "exact":
        exp_neg_a_dt = jnp.exp(-a * dt)
        one_minus_exp = 1.0 - exp_neg_a_dt
        vol_term = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * dt)) / (2.0 * a))

        def step_fn(carry, inputs):
            r_t, t = carry
            z = inputs

            # Compute θ(t) at current time
            if theta_fn is not None:
                theta_t = theta_fn(t)
            else:
                theta_t = 0.0

            # Mean reversion level at time t
            alpha_t = theta_t / a if a != 0 else theta_t

            # Exact update
            mean = r_t * exp_neg_a_dt + alpha_t * one_minus_exp
            r_next = mean + vol_term * z
            t_next = t + dt

            return (r_next, t_next), r_next

        _, r_path = lax.scan(step_fn, (r0, 0.0), Z)

    elif method == "euler":
        sqrt_dt = jnp.sqrt(dt)

        def step_fn(carry, inputs):
            r_t, t = carry
            z = inputs

            # Compute θ(t) at current time
            if theta_fn is not None:
                theta_t = theta_fn(t)
            else:
                theta_t = 0.0

            # Euler-Maruyama discretization
            dr = (theta_t - a * r_t) * dt + sigma * sqrt_dt * z
            r_next = r_t + dr
            t_next = t + dt

            return (r_next, t_next), r_next

        _, r_path = lax.scan(step_fn, (r0, 0.0), Z)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'euler'.")

    # Prepend initial rate
    return jnp.concatenate([jnp.array([r0]), r_path])


def simulate_paths(
    params: HullWhiteParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    """Simulate multiple paths of the Hull-White short rate process.

    Parameters
    ----------
    params : HullWhiteParams
        Hull-White model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths to simulate
    key : jax.random.KeyArray
        JAX random key
    method : str, optional
        Simulation method: "exact" (default) or "euler"

    Returns
    -------
    Array
        Short rate paths of shape [n_paths, n_steps + 1]
    """
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)


def caplet_price(
    params: HullWhiteParams,
    strike: float,
    caplet_maturity: float,
    tenor: float,
    forward_curve_fn: Optional[Callable[[float], float]] = None
) -> float:
    """Price a caplet under the Hull-White model.

    A caplet pays max(L(T) - K, 0) at time T + τ, where:
    - L(T): LIBOR rate at time T for tenor τ
    - K: strike rate
    - T: caplet maturity (reset date)
    - τ: tenor (accrual period)

    Parameters
    ----------
    params : HullWhiteParams
        Hull-White model parameters
    strike : float
        Strike rate
    caplet_maturity : float
        Time to caplet maturity (reset date)
    tenor : float
        Accrual period (e.g., 0.25 for 3M LIBOR)
    forward_curve_fn : Optional[Callable]
        Market forward curve. If None, uses flat curve.

    Returns
    -------
    float
        Caplet price

    Notes
    -----
    The caplet can be expressed as a put option on a zero-coupon bond:
        Caplet = (1 + K*τ) * Put(K_bond, T, T+τ)
    where K_bond = 1 / (1 + K*τ)
    """
    from scipy.stats import norm

    a, sigma = params.a, params.sigma
    T = caplet_maturity
    tau = tenor

    # Bond equivalent strike
    K_bond = 1.0 / (1.0 + strike * tau)

    # Bond prices
    P_T = zero_coupon_bond_price(params, T, forward_curve_fn)
    P_T_tau = zero_coupon_bond_price(params, T + tau, forward_curve_fn)

    # B(τ) for remaining period
    B_tau = (1.0 - jnp.exp(-a * tau)) / a

    # Volatility of bond price ratio
    sigma_P = sigma * B_tau * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T)) / (2.0 * a))

    # Black-Scholes type formula for put on bond
    if sigma_P > 1e-10:
        d1 = (jnp.log(P_T_tau / (K_bond * P_T)) + 0.5 * sigma_P * sigma_P) / sigma_P
        d2 = d1 - sigma_P

        # Put option value
        put_value = K_bond * P_T * norm.cdf(float(-d2)) - P_T_tau * norm.cdf(float(-d1))
    else:
        # Intrinsic value when vol is zero
        put_value = jnp.maximum(K_bond * P_T - P_T_tau, 0.0)

    # Convert put on bond to caplet
    caplet = (1.0 + strike * tau) * put_value

    return float(caplet)


def cap_price(
    params: HullWhiteParams,
    strike: float,
    cap_maturity: float,
    payment_frequency: float = 0.25,
    forward_curve_fn: Optional[Callable[[float], float]] = None
) -> float:
    """Price a cap as a portfolio of caplets.

    A cap is a series of caplets with different maturities.

    Parameters
    ----------
    params : HullWhiteParams
        Hull-White model parameters
    strike : float
        Strike rate
    cap_maturity : float
        Total maturity of the cap
    payment_frequency : float, optional
        Payment frequency in years (default: 0.25 for quarterly)
    forward_curve_fn : Optional[Callable]
        Market forward curve

    Returns
    -------
    float
        Cap price
    """
    # Number of caplets
    n_caplets = int(cap_maturity / payment_frequency)

    total_price = 0.0
    for i in range(n_caplets):
        caplet_mat = (i + 1) * payment_frequency
        caplet_p = caplet_price(
            params,
            strike,
            caplet_mat,
            payment_frequency,
            forward_curve_fn
        )
        total_price += caplet_p

    return total_price


__all__ = [
    "HullWhiteParams",
    "zero_coupon_bond_price",
    "simulate_path",
    "simulate_paths",
    "caplet_price",
    "cap_price",
]
