"""Vasicek short rate model implementation with JAX.

The Vasicek model describes the evolution of interest rates as a mean-reverting
Ornstein-Uhlenbeck process:

    dr(t) = a(b - r(t))dt + σ dW(t)

where:
    - a: mean reversion speed
    - b: long-term mean level
    - σ: volatility
    - r(t): short rate at time t
    - W(t): standard Brownian motion

References
----------
Vasicek, O. (1977). "An equilibrium characterization of the term structure."
Journal of Financial Economics, 5(2), 177-188.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class VasicekParams:
    """Parameters for the Vasicek short rate model.

    Attributes
    ----------
    a : float
        Mean reversion speed (must be positive)
    b : float
        Long-term mean level (the level to which rates revert)
    sigma : float
        Volatility (must be positive)
    r0 : float
        Initial short rate
    """
    a: float
    b: float
    sigma: float
    r0: float

    def __post_init__(self):
        """Validate parameters."""
        if self.a <= 0:
            raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")


def zero_coupon_bond_price(params: VasicekParams, T: float) -> float:
    """Calculate zero-coupon bond price under Vasicek model.

    Analytical formula for the price of a zero-coupon bond maturing at time T:

        P(0, T) = A(T) * exp(-B(T) * r(0))

    where:
        B(T) = (1 - exp(-aT)) / a
        A(T) = exp((B(T) - T)(a²b - σ²/2) / a² - σ²B(T)² / (4a))

    Parameters
    ----------
    params : VasicekParams
        Vasicek model parameters
    T : float
        Time to maturity

    Returns
    -------
    float
        Zero-coupon bond price

    Examples
    --------
    >>> params = VasicekParams(a=0.1, b=0.05, sigma=0.01, r0=0.03)
    >>> price = zero_coupon_bond_price(params, T=5.0)
    >>> print(f"Bond price: {price:.6f}")
    """
    a, b, sigma, r0 = params.a, params.b, params.sigma, params.r0

    # B(T) = (1 - exp(-aT)) / a
    B_T = (1.0 - jnp.exp(-a * T)) / a

    # A(T) = exp((B(T) - T)(a²b - σ²/2) / a² - σ²B(T)² / (4a))
    term1 = (B_T - T) * (a * a * b - sigma * sigma / 2.0) / (a * a)
    term2 = -sigma * sigma * B_T * B_T / (4.0 * a)
    A_T = jnp.exp(term1 + term2)

    # P(0, T) = A(T) * exp(-B(T) * r0)
    return A_T * jnp.exp(-B_T * r0)


def yield_curve(params: VasicekParams, maturities: jnp.ndarray) -> jnp.ndarray:
    """Calculate zero-coupon yield curve under Vasicek model.

    The yield y(T) is defined as:
        y(T) = -log(P(0, T)) / T

    Parameters
    ----------
    params : VasicekParams
        Vasicek model parameters
    maturities : Array
        Array of maturities (in years)

    Returns
    -------
    Array
        Zero-coupon yields for each maturity
    """
    def yield_at_maturity(T):
        P_T = zero_coupon_bond_price(params, T)
        return -jnp.log(P_T) / T

    return jax.vmap(yield_at_maturity)(maturities)


def simulate_path(
    params: VasicekParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    """Simulate a single path of the Vasicek short rate process.

    Parameters
    ----------
    params : VasicekParams
        Vasicek model parameters
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
    The exact simulation uses the analytical solution of the Ornstein-Uhlenbeck SDE:

        r(t + dt) = r(t) * exp(-a * dt) + b * (1 - exp(-a * dt))
                    + σ * sqrt((1 - exp(-2 * a * dt)) / (2 * a)) * Z

    where Z ~ N(0, 1)
    """
    a, b, sigma, r0 = params.a, params.b, params.sigma, params.r0
    dt = T / n_steps

    # Generate random normals
    Z = jax.random.normal(key, shape=(n_steps,))

    if method == "exact":
        # Exact simulation using analytical solution
        exp_neg_a_dt = jnp.exp(-a * dt)

        # Mean and std of r(t+dt) | r(t)
        def step_fn(r_t, z):
            mean = r_t * exp_neg_a_dt + b * (1.0 - exp_neg_a_dt)
            std = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * dt)) / (2.0 * a))
            r_next = mean + std * z
            return r_next, r_next

        _, r_path = lax.scan(step_fn, r0, Z)

    elif method == "euler":
        # Euler-Maruyama discretization
        sqrt_dt = jnp.sqrt(dt)

        def step_fn(r_t, z):
            dr = a * (b - r_t) * dt + sigma * sqrt_dt * z
            r_next = r_t + dr
            return r_next, r_next

        _, r_path = lax.scan(step_fn, r0, Z)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'euler'.")

    # Prepend initial rate
    return jnp.concatenate([jnp.array([r0]), r_path])


def simulate_paths(
    params: VasicekParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    """Simulate multiple paths of the Vasicek short rate process.

    Parameters
    ----------
    params : VasicekParams
        Vasicek model parameters
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


def bond_option_price(
    params: VasicekParams,
    strike: float,
    option_maturity: float,
    bond_maturity: float,
    is_call: bool = True
) -> float:
    """Price a European option on a zero-coupon bond under Vasicek model.

    Analytical formula for a call option on a zero-coupon bond:

        C = P(0, T_opt) * Φ(h) - K * P(0, T_bond) * Φ(h - σ_P)

    where:
        - T_opt: option maturity
        - T_bond: bond maturity (T_bond > T_opt)
        - K: strike price
        - σ_P: volatility of bond price

    Parameters
    ----------
    params : VasicekParams
        Vasicek model parameters
    strike : float
        Strike price
    option_maturity : float
        Time to option expiration
    bond_maturity : float
        Time to bond maturity (must be > option_maturity)
    is_call : bool, optional
        If True, price a call option. If False, price a put option.

    Returns
    -------
    float
        Option price

    References
    ----------
    Jamshidian, F. (1989). "An exact bond option formula."
    The Journal of Finance, 44(1), 205-209.
    """
    from scipy.stats import norm

    if bond_maturity <= option_maturity:
        raise ValueError(
            f"Bond maturity ({bond_maturity}) must be greater than "
            f"option maturity ({option_maturity})"
        )

    a, b, sigma, r0 = params.a, params.b, params.sigma, params.r0

    T_opt = option_maturity
    T_bond = bond_maturity

    # Bond prices at option maturity and bond maturity
    P_0_Topt = zero_coupon_bond_price(params, T_opt)
    P_0_Tbond = zero_coupon_bond_price(params, T_bond)

    # B(T) for remaining time from option maturity to bond maturity
    tau = T_bond - T_opt
    B_tau = (1.0 - jnp.exp(-a * tau)) / a

    # Volatility of bond price
    sigma_P = sigma * B_tau * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T_opt)) / (2.0 * a))

    # h parameter
    h = (jnp.log(P_0_Tbond / (strike * P_0_Topt))) / sigma_P + sigma_P / 2.0

    if is_call:
        # Call option
        price = P_0_Tbond * norm.cdf(float(h)) - strike * P_0_Topt * norm.cdf(float(h - sigma_P))
    else:
        # Put option (put-call parity)
        price = strike * P_0_Topt * norm.cdf(float(-h + sigma_P)) - P_0_Tbond * norm.cdf(float(-h))

    return float(price)


__all__ = [
    "VasicekParams",
    "zero_coupon_bond_price",
    "yield_curve",
    "simulate_path",
    "simulate_paths",
    "bond_option_price",
]
