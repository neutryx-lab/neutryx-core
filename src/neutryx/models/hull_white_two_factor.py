"""Hull-White two-factor short rate model implementation with JAX.

The Hull-White two-factor model extends the one-factor model by introducing
a second stochastic factor, providing richer dynamics for the term structure:

    dr(t) = [θ(t) + u(t) - a * r(t)]dt + σ₁ dW₁(t)
    du(t) = -b * u(t)dt + σ₂ dW₂(t)

where:
    - r(t): short rate at time t
    - u(t): second factor (stochastic mean level)
    - a, b: mean reversion speeds (a, b > 0)
    - σ₁, σ₂: volatilities
    - θ(t): time-dependent drift
    - W₁(t), W₂(t): correlated Brownian motions with correlation ρ
    - dW₁ dW₂ = ρ dt

The two-factor model provides:
1. Better fit to volatility term structure
2. Richer correlation structure between different rates
3. More realistic decorrelation of forward rates

References
----------
Hull, J., & White, A. (1994). "Numerical procedures for implementing term
structure models II: Two-factor models." Journal of Derivatives, 2(2), 37-48.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class HullWhiteTwoFactorParams:
    """Parameters for the Hull-White two-factor model.

    Attributes
    ----------
    a : float
        Mean reversion speed for short rate (must be positive)
    b : float
        Mean reversion speed for second factor (must be positive)
    sigma1 : float
        Volatility of short rate factor (must be positive)
    sigma2 : float
        Volatility of second factor (must be positive)
    rho : float
        Correlation between the two Brownian motions (-1 ≤ ρ ≤ 1)
    r0 : float
        Initial short rate
    u0 : float
        Initial second factor (typically 0)
    theta_fn : Optional[Callable]
        Time-dependent drift function θ(t). If None, uses constant drift.
    """
    a: float
    b: float
    sigma1: float
    sigma2: float
    rho: float
    r0: float
    u0: float = 0.0
    theta_fn: Optional[Callable[[float], float]] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.a <= 0:
            raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
        if self.b <= 0:
            raise ValueError(f"Mean reversion speed b must be positive, got {self.b}")
        if self.sigma1 <= 0:
            raise ValueError(f"Volatility sigma1 must be positive, got {self.sigma1}")
        if self.sigma2 <= 0:
            raise ValueError(f"Volatility sigma2 must be positive, got {self.sigma2}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"Correlation rho must be in [-1, 1], got {self.rho}")


def zero_coupon_bond_price(
    params: HullWhiteTwoFactorParams,
    T: float,
    r_t: float,
    u_t: float,
    t: float = 0.0,
) -> float:
    """Calculate zero-coupon bond price under two-factor Hull-White model.

    The bond price P(t, T) has the affine form:
        P(t, T) = A(t, T) * exp(-B(t, T) * r(t) - C(t, T) * u(t))

    where B, C, and A satisfy ODEs that can be solved analytically.

    Parameters
    ----------
    params : HullWhiteTwoFactorParams
        Two-factor Hull-White model parameters
    T : float
        Bond maturity time
    r_t : float
        Current short rate
    u_t : float
        Current second factor value
    t : float, optional
        Current time (default: 0)

    Returns
    -------
    float
        Zero-coupon bond price
    """
    a, b, sigma1, sigma2, rho = params.a, params.b, params.sigma1, params.sigma2, params.rho
    tau = T - t

    # B(τ) = (1 - exp(-aτ)) / a
    B_tau = (1.0 - jnp.exp(-a * tau)) / a

    # C(τ) = (1 - exp(-bτ)) / b
    C_tau = (1.0 - jnp.exp(-b * tau)) / b

    # Volatility adjustment term V(t, T)
    # V = (σ₁² / (2a²)) * (B - τ) - (σ₁² / (4a)) * B²
    #     + (σ₂² / (2b²)) * (C - τ) - (σ₂² / (4b)) * C²
    #     + (ρ σ₁ σ₂ / (ab)) * (BC - (1-exp(-aτ))(1-exp(-bτ))/(ab))

    V_r = (sigma1 * sigma1 / (2.0 * a * a)) * (B_tau - tau) - \
          (sigma1 * sigma1 / (4.0 * a)) * B_tau * B_tau

    V_u = (sigma2 * sigma2 / (2.0 * b * b)) * (C_tau - tau) - \
          (sigma2 * sigma2 / (4.0 * b)) * C_tau * C_tau

    # Cross-term for correlation
    exp_a_tau = jnp.exp(-a * tau)
    exp_b_tau = jnp.exp(-b * tau)
    cross_term = B_tau * C_tau - (1.0 - exp_a_tau) * (1.0 - exp_b_tau) / (a * b)
    V_cross = (rho * sigma1 * sigma2 / (a * b)) * cross_term

    V_total = V_r + V_u + V_cross

    # A(t, T) includes theta term (simplified for constant theta)
    if params.theta_fn is not None:
        # Would need to integrate theta from t to T
        # For now, use simplified version
        theta_t = params.theta_fn(t)
        A_tau = jnp.exp((B_tau - tau) * theta_t / a + V_total)
    else:
        A_tau = jnp.exp(V_total)

    # P(t, T) = A(τ) * exp(-B(τ) * r(t) - C(τ) * u(t))
    return A_tau * jnp.exp(-B_tau * r_t - C_tau * u_t)


def simulate_path(
    params: HullWhiteTwoFactorParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate a single path of the two-factor Hull-White process.

    Parameters
    ----------
    params : HullWhiteTwoFactorParams
        Two-factor Hull-White model parameters
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
    tuple[Array, Array]
        Tuple of (r_path, u_path) where:
        - r_path: short rate path of shape [n_steps + 1]
        - u_path: second factor path of shape [n_steps + 1]

    Notes
    -----
    For exact simulation, both r and u follow Ornstein-Uhlenbeck processes
    that can be sampled exactly using their conditional distributions.
    """
    a, b, sigma1, sigma2, rho = params.a, params.b, params.sigma1, params.sigma2, params.rho
    r0, u0 = params.r0, params.u0
    theta_fn = params.theta_fn
    dt = T / n_steps

    # Generate correlated random normals
    key1, key2 = jax.random.split(key)
    Z1 = jax.random.normal(key1, shape=(n_steps,))
    Z2_indep = jax.random.normal(key2, shape=(n_steps,))

    # Correlate Z2 with Z1: Z2 = ρ * Z1 + sqrt(1 - ρ²) * Z2_indep
    Z2 = rho * Z1 + jnp.sqrt(1.0 - rho * rho) * Z2_indep

    if method == "exact":
        # Exact simulation using conditional distributions
        exp_neg_a_dt = jnp.exp(-a * dt)
        exp_neg_b_dt = jnp.exp(-b * dt)

        one_minus_exp_a = 1.0 - exp_neg_a_dt
        one_minus_exp_b = 1.0 - exp_neg_b_dt

        vol1_term = sigma1 * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * dt)) / (2.0 * a))
        vol2_term = sigma2 * jnp.sqrt((1.0 - jnp.exp(-2.0 * b * dt)) / (2.0 * b))

        def step_fn(carry, inputs):
            r_t, u_t, t = carry
            z1, z2 = inputs

            # Update u(t) first (doesn't depend on r)
            u_next = u_t * exp_neg_b_dt + vol2_term * z2

            # Update r(t) (depends on u)
            if theta_fn is not None:
                theta_t = theta_fn(t)
            else:
                theta_t = 0.0

            # Mean reversion with u(t) contribution
            mean_r = r_t * exp_neg_a_dt + (theta_t / a + u_t / b) * one_minus_exp_a
            r_next = mean_r + vol1_term * z1

            t_next = t + dt

            return (r_next, u_next, t_next), (r_next, u_next)

        _, (r_path, u_path) = lax.scan(step_fn, (r0, u0, 0.0), (Z1, Z2))

    elif method == "euler":
        # Euler-Maruyama discretization
        sqrt_dt = jnp.sqrt(dt)

        def step_fn(carry, inputs):
            r_t, u_t, t = carry
            z1, z2 = inputs

            if theta_fn is not None:
                theta_t = theta_fn(t)
            else:
                theta_t = 0.0

            # du = -b * u * dt + σ₂ * dW₂
            du = -b * u_t * dt + sigma2 * sqrt_dt * z2
            u_next = u_t + du

            # dr = [θ(t) + u(t) - a * r(t)] * dt + σ₁ * dW₁
            dr = (theta_t + u_t - a * r_t) * dt + sigma1 * sqrt_dt * z1
            r_next = r_t + dr

            t_next = t + dt

            return (r_next, u_next, t_next), (r_next, u_next)

        _, (r_path, u_path) = lax.scan(step_fn, (r0, u0, 0.0), (Z1, Z2))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact' or 'euler'.")

    # Prepend initial values
    r_path_full = jnp.concatenate([jnp.array([r0]), r_path])
    u_path_full = jnp.concatenate([jnp.array([u0]), u_path])

    return r_path_full, u_path_full


def simulate_paths(
    params: HullWhiteTwoFactorParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate multiple paths of the two-factor Hull-White process.

    Parameters
    ----------
    params : HullWhiteTwoFactorParams
        Two-factor Hull-White model parameters
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
    tuple[Array, Array]
        Tuple of (r_paths, u_paths) where:
        - r_paths: short rate paths of shape [n_paths, n_steps + 1]
        - u_paths: second factor paths of shape [n_paths, n_steps + 1]
    """
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    r_paths, u_paths = jax.vmap(sim_single_path)(keys)
    return r_paths, u_paths


def caplet_price(
    params: HullWhiteTwoFactorParams,
    strike: float,
    caplet_maturity: float,
    tenor: float,
) -> float:
    """Price a caplet under the two-factor Hull-White model.

    This uses Monte Carlo simulation since the two-factor model doesn't
    have a closed-form caplet formula as clean as the one-factor model.

    Parameters
    ----------
    params : HullWhiteTwoFactorParams
        Two-factor Hull-White model parameters
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
    For production use, consider implementing the semi-analytical formula
    based on the two-factor model's bond option pricing formula.
    """
    # Simplified Monte Carlo implementation
    # For a full implementation, use the analytical bond option formula
    # Similar to the one-factor case but with two factors

    # Bond prices at current time
    P_T = zero_coupon_bond_price(params, caplet_maturity, params.r0, params.u0)
    P_T_tau = zero_coupon_bond_price(
        params, caplet_maturity + tenor, params.r0, params.u0
    )

    # Forward rate
    fwd_rate = (P_T / P_T_tau - 1.0) / tenor

    # Simplified Black approximation for now
    # Full implementation would use two-factor bond option formula
    from scipy.stats import norm

    # Approximate volatility (simplified)
    a, b, sigma1, sigma2 = params.a, params.b, params.sigma1, params.sigma2
    tau_opt = caplet_maturity

    # Combined volatility effect
    vol_approx = jnp.sqrt(
        sigma1 * sigma1 * (1.0 - jnp.exp(-2.0 * a * tau_opt)) / (2.0 * a) +
        sigma2 * sigma2 * (1.0 - jnp.exp(-2.0 * b * tau_opt)) / (2.0 * b)
    ) / tau_opt

    # Black formula
    if vol_approx > 1e-10:
        d1 = (jnp.log(fwd_rate / strike) + 0.5 * vol_approx * vol_approx * tau_opt) / \
             (vol_approx * jnp.sqrt(tau_opt))
        d2 = d1 - vol_approx * jnp.sqrt(tau_opt)

        caplet_value = P_T_tau * tenor * (
            fwd_rate * norm.cdf(float(d1)) - strike * norm.cdf(float(d2))
        )
    else:
        caplet_value = P_T_tau * tenor * jnp.maximum(fwd_rate - strike, 0.0)

    return float(caplet_value)


def instantaneous_correlation(
    params: HullWhiteTwoFactorParams,
    T1: float,
    T2: float,
    t: float = 0.0
) -> float:
    """Calculate instantaneous correlation between two forward rates.

    One of the key advantages of the two-factor model is that it can
    produce realistic decorrelation between forward rates of different
    maturities.

    Parameters
    ----------
    params : HullWhiteTwoFactorParams
        Two-factor Hull-White model parameters
    T1 : float
        First maturity
    T2 : float
        Second maturity
    t : float, optional
        Current time (default: 0)

    Returns
    -------
    float
        Instantaneous correlation between forward rates f(t, T1) and f(t, T2)

    Notes
    -----
    The correlation structure is richer than in one-factor models,
    with correlation typically decreasing as |T1 - T2| increases.
    """
    a, b, sigma1, sigma2, rho = params.a, params.b, params.sigma1, params.sigma2, params.rho

    tau1 = T1 - t
    tau2 = T2 - t

    # Volatility of forward rate f(t, T1)
    B1 = (1.0 - jnp.exp(-a * tau1)) / a
    C1 = (1.0 - jnp.exp(-b * tau1)) / b
    var_f1 = sigma1 * sigma1 * B1 * B1 + sigma2 * sigma2 * C1 * C1 + \
             2.0 * rho * sigma1 * sigma2 * B1 * C1

    # Volatility of forward rate f(t, T2)
    B2 = (1.0 - jnp.exp(-a * tau2)) / a
    C2 = (1.0 - jnp.exp(-b * tau2)) / b
    var_f2 = sigma1 * sigma1 * B2 * B2 + sigma2 * sigma2 * C2 * C2 + \
             2.0 * rho * sigma1 * sigma2 * B2 * C2

    # Covariance
    cov_f1_f2 = sigma1 * sigma1 * B1 * B2 + sigma2 * sigma2 * C1 * C2 + \
                rho * sigma1 * sigma2 * (B1 * C2 + B2 * C1)

    # Correlation
    correlation = cov_f1_f2 / jnp.sqrt(var_f1 * var_f2)

    return float(correlation)


__all__ = [
    "HullWhiteTwoFactorParams",
    "zero_coupon_bond_price",
    "simulate_path",
    "simulate_paths",
    "caplet_price",
    "instantaneous_correlation",
]
