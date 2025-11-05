"""Black-Karasinski short rate model implementation with JAX.

The Black-Karasinski model is a log-normal short rate model where the
logarithm of the short rate follows a mean-reverting process:

    d ln(r(t)) = [θ(t) - a * ln(r(t))]dt + σ dW(t)

Equivalently, for the short rate itself:
    dr(t) / r(t) = [θ(t) + σ²/2 - a * ln(r(t))]dt + σ dW(t)

where:
    - r(t): short rate at time t (always positive)
    - a: mean reversion speed
    - σ: volatility of log(r)
    - θ(t): time-dependent drift
    - W(t): standard Brownian motion

Key features:
1. Rates are always positive (log-normal specification)
2. Volatility proportional to rate level
3. Can fit initial term structure via time-dependent θ(t)
4. Generally requires numerical methods (trinomial tree, Monte Carlo)

The Black-Karasinski model is popular in practice because:
- It ensures positive rates naturally
- It has more realistic volatility structure than normal models
- It can match market cap/swaption volatilities

References
----------
Black, F., & Karasinski, P. (1991). "Bond and option pricing when short rates
are lognormal." Financial Analysts Journal, 47(4), 52-59.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class BlackKarasinskiParams:
    """Parameters for the Black-Karasinski short rate model.

    Attributes
    ----------
    a : float
        Mean reversion speed (must be positive)
    sigma : float
        Volatility of log(r) (must be positive)
    r0 : float
        Initial short rate (must be positive)
    theta_fn : Optional[Callable]
        Time-dependent drift function θ(t). If None, uses constant drift
        that keeps E[ln(r)] constant.
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
        if self.r0 <= 0:
            raise ValueError(f"Initial rate r0 must be positive, got {self.r0}")


def simulate_path(
    params: BlackKarasinskiParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
    method: str = "euler"
) -> jnp.ndarray:
    """Simulate a single path of the Black-Karasinski short rate process.

    Parameters
    ----------
    params : BlackKarasinskiParams
        Black-Karasinski model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key
    method : str, optional
        Simulation method: "euler" (default) or "milstein"

    Returns
    -------
    Array
        Short rate path of shape [n_steps + 1]

    Notes
    -----
    We simulate ln(r) as an Ornstein-Uhlenbeck process, then exponentiate
    to get r. This ensures r remains positive.

    For the Euler scheme on ln(r):
        ln(r(t+dt)) = ln(r(t)) + [θ(t) - a * ln(r(t))] * dt + σ * sqrt(dt) * Z

    The Milstein correction is zero for this additive noise case.
    """
    a, sigma, r0 = params.a, params.sigma, params.r0
    theta_fn = params.theta_fn
    dt = T / n_steps

    # Initial log rate
    ln_r0 = jnp.log(r0)

    # Generate random normals
    Z = jax.random.normal(key, shape=(n_steps,))

    if method == "euler":
        sqrt_dt = jnp.sqrt(dt)

        def step_fn(carry, inputs):
            ln_r_t, t = carry
            z = inputs

            # Compute θ(t) at current time
            if theta_fn is not None:
                theta_t = theta_fn(t)
            else:
                # Default: keep E[ln(r)] constant at ln(r0)
                theta_t = a * jnp.log(r0)

            # Euler step for ln(r)
            d_ln_r = (theta_t - a * ln_r_t) * dt + sigma * sqrt_dt * z
            ln_r_next = ln_r_t + d_ln_r

            t_next = t + dt

            return (ln_r_next, t_next), ln_r_next

        _, ln_r_path = lax.scan(step_fn, (ln_r0, 0.0), Z)

    elif method == "milstein":
        # For ln(r), Milstein = Euler since diffusion coefficient is constant
        # But we include it for completeness
        sqrt_dt = jnp.sqrt(dt)

        def step_fn(carry, inputs):
            ln_r_t, t = carry
            z = inputs

            if theta_fn is not None:
                theta_t = theta_fn(t)
            else:
                theta_t = a * jnp.log(r0)

            # Milstein correction is zero for additive noise
            d_ln_r = (theta_t - a * ln_r_t) * dt + sigma * sqrt_dt * z
            ln_r_next = ln_r_t + d_ln_r

            t_next = t + dt

            return (ln_r_next, t_next), ln_r_next

        _, ln_r_path = lax.scan(step_fn, (ln_r0, 0.0), Z)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'euler' or 'milstein'.")

    # Prepend initial log rate
    ln_r_path_full = jnp.concatenate([jnp.array([ln_r0]), ln_r_path])

    # Exponentiate to get actual rates (always positive)
    r_path = jnp.exp(ln_r_path_full)

    return r_path


def simulate_paths(
    params: BlackKarasinskiParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "euler"
) -> jnp.ndarray:
    """Simulate multiple paths of the Black-Karasinski short rate process.

    Parameters
    ----------
    params : BlackKarasinskiParams
        Black-Karasinski model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths to simulate
    key : jax.random.KeyArray
        JAX random key
    method : str, optional
        Simulation method: "euler" (default) or "milstein"

    Returns
    -------
    Array
        Short rate paths of shape [n_paths, n_steps + 1]
    """
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)


def zero_coupon_bond_price_mc(
    params: BlackKarasinskiParams,
    T: float,
    n_paths: int = 10000,
    n_steps: int = 100,
    key: jax.random.KeyArray = None,
) -> float:
    """Calculate zero-coupon bond price using Monte Carlo simulation.

    The Black-Karasinski model does not have a closed-form bond pricing formula,
    so we use Monte Carlo simulation:

        P(0, T) = E[exp(-∫₀ᵀ r(s) ds)]

    Parameters
    ----------
    params : BlackKarasinskiParams
        Black-Karasinski model parameters
    T : float
        Time to maturity
    n_paths : int, optional
        Number of Monte Carlo paths (default: 10000)
    n_steps : int, optional
        Number of time steps (default: 100)
    key : jax.random.KeyArray, optional
        JAX random key. If None, creates a new key.

    Returns
    -------
    float
        Zero-coupon bond price estimate

    Notes
    -----
    For production pricing, consider using a trinomial tree instead of
    Monte Carlo for better efficiency and accuracy.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Simulate rate paths
    rate_paths = simulate_paths(params, T, n_steps, n_paths, key)

    # Compute integral of rates using trapezoidal rule
    dt = T / n_steps
    # Average adjacent rates for trapezoidal integration
    rate_avg = 0.5 * (rate_paths[:, :-1] + rate_paths[:, 1:])
    integral = jnp.sum(rate_avg, axis=1) * dt

    # Discount factors
    discount_factors = jnp.exp(-integral)

    # Bond price is expectation of discount factor
    bond_price = jnp.mean(discount_factors)

    return float(bond_price)


def calibrate_theta_to_yield_curve(
    params: BlackKarasinskiParams,
    maturities: jnp.ndarray,
    market_zero_rates: jnp.ndarray,
) -> Callable[[float], float]:
    """Calibrate time-dependent θ(t) to match market yield curve.

    This is a simplified calibration that uses piecewise constant θ(t).
    For production use, implement iterative calibration with trinomial tree.

    Parameters
    ----------
    params : BlackKarasinskiParams
        Black-Karasinski model parameters (a and σ are fixed)
    maturities : Array
        Market maturities
    market_zero_rates : Array
        Market zero rates (continuously compounded)

    Returns
    -------
    Callable
        Function θ(t) that interpolates calibrated drift

    Notes
    -----
    Full calibration requires:
    1. Build trinomial tree
    2. Iteratively solve for θ at each time step
    3. Match market discount curve exactly

    This simplified version provides a starting point.
    """
    # Simplified: fit a polynomial to the market forward rate structure
    # θ(t) ≈ a * ln(f(0,t)) where f(0,t) is instantaneous forward rate

    # Compute forward rates from zero rates
    # f(0, t) = r(t) + t * dr/dt
    forward_rates = jnp.gradient(market_zero_rates * maturities) / jnp.gradient(maturities)

    # Create interpolation function for θ(t)
    def theta_fn(t: float) -> float:
        # Linear interpolation of a * ln(forward_rate)
        idx = jnp.searchsorted(maturities, t)
        if idx == 0:
            fwd = forward_rates[0]
        elif idx >= len(forward_rates):
            fwd = forward_rates[-1]
        else:
            # Linear interpolation
            t1, t2 = maturities[idx - 1], maturities[idx]
            f1, f2 = forward_rates[idx - 1], forward_rates[idx]
            weight = (t - t1) / (t2 - t1)
            fwd = f1 + weight * (f2 - f1)

        # θ(t) ≈ a * ln(forward_rate) for mean-reversion to forward rate
        return float(params.a * jnp.log(jnp.maximum(fwd, 1e-6)))

    return theta_fn


def caplet_price_mc(
    params: BlackKarasinskiParams,
    strike: float,
    caplet_maturity: float,
    tenor: float,
    n_paths: int = 50000,
    key: jax.random.KeyArray = None,
) -> float:
    """Price a caplet using Monte Carlo simulation.

    Parameters
    ----------
    params : BlackKarasinskiParams
        Black-Karasinski model parameters
    strike : float
        Strike rate
    caplet_maturity : float
        Time to caplet maturity (reset date)
    tenor : float
        Accrual period (e.g., 0.25 for 3M LIBOR)
    n_paths : int, optional
        Number of Monte Carlo paths (default: 50000)
    key : jax.random.KeyArray, optional
        JAX random key

    Returns
    -------
    float
        Caplet price

    Notes
    -----
    Caplet payoff: max(L(T) - K, 0) * τ
    where L(T) is the LIBOR rate observed at time T.

    For efficiency in production, use trinomial tree pricing.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    T = caplet_maturity
    n_steps = max(50, int(T * 50))  # At least 50 steps per year

    # Split keys for two simulations
    key1, key2 = jax.random.split(key)

    # Simulate paths to T
    rate_paths_T = simulate_paths(params, T, n_steps, n_paths, key1)

    # Continue simulations from T to T+τ to get bond prices
    # For simplicity, simulate full path to T+τ
    rate_paths_full = simulate_paths(params, T + tenor, n_steps + 10, n_paths, key2)

    dt = (T + tenor) / (n_steps + 10)

    # Compute LIBOR rate at time T
    # L(T, T+τ) = (P(T, T) - P(T, T+τ)) / (τ * P(T, T+τ))
    #          = (1 - P(T, T+τ)) / (τ * P(T, T+τ))

    # For each path, compute bond price from T to T+τ
    idx_T = n_steps
    # Remaining rates from T to T+τ
    remaining_rates = rate_paths_full[:, idx_T:]

    # Integrate from T to T+τ
    rate_avg = 0.5 * (remaining_rates[:, :-1] + remaining_rates[:, 1:])
    integral_T_to_Ttau = jnp.sum(rate_avg[:, :10], axis=1) * dt

    # Bond price P(T, T+τ) for each path
    P_T_Ttau = jnp.exp(-integral_T_to_Ttau)

    # LIBOR rate
    L_T = (1.0 - P_T_Ttau) / (tenor * P_T_Ttau)

    # Caplet payoff at time T+τ
    caplet_payoff = jnp.maximum(L_T - strike, 0.0) * tenor

    # Discount back to time 0
    # Need to discount by exp(-∫₀^(T+τ) r(s) ds)
    full_integral = jnp.sum(0.5 * (rate_paths_full[:, :-1] + rate_paths_full[:, 1:]), axis=1) * dt
    discount_to_0 = jnp.exp(-full_integral)

    # Discounted payoff
    discounted_payoff = caplet_payoff * discount_to_0

    # Price is expectation
    caplet_value = jnp.mean(discounted_payoff)

    return float(caplet_value)


__all__ = [
    "BlackKarasinskiParams",
    "simulate_path",
    "simulate_paths",
    "zero_coupon_bond_price_mc",
    "calibrate_theta_to_yield_curve",
    "caplet_price_mc",
]
