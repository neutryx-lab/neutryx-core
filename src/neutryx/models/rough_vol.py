"""Rough volatility models driven by fractional Brownian motion.

This module implements a simple Rough Bergomi Monte Carlo sampler based on the
fractional Brownian motion (fBm) representation with Hurst parameter ``H``.
The implementation follows a direct Cholesky sampling scheme for the fBm
increments allowing time-inhomogeneous forward variance curves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from ..core.engine import MCConfig, MCPaths, resolve_schedule, time_grid

Array = jnp.ndarray
Schedule = float | Sequence[float] | Array | Callable[[Array], Array | float]

__all__ = [
    "RoughBergomiParams",
    "RoughBergomiPaths",
    "simulate_rough_bergomi",
    "calibrate_forward_variance",
    "price_european_call_mc",
]


@dataclass
class RoughBergomiParams:
    """Parameters for the Rough Bergomi model.

    Attributes
    ----------
    hurst
        Hurst parameter controlling the roughness of the volatility process.
    eta
        Volatility of volatility scaling parameter.
    rho
        Correlation between the asset Brownian motion and the volatility driver.
    forward_variance
        Forward variance curve, supplied either as a scalar, array-like or a
        callable accepting the time grid and returning variance levels.
    """

    hurst: float
    eta: float
    rho: float
    forward_variance: Schedule = 0.04


@dataclass
class RoughBergomiPaths(MCPaths):
    """Container extending :class:`~neutryx.core.engine.MCPaths` with rough data."""

    variance: Array | None = None
    fbm: Array | None = None


def _fractional_increment_covariance(
    hurst: float, steps: int, *, dtype: jnp.dtype, dt: float
) -> Array:
    """Covariance matrix for fractional Brownian motion increments."""

    if not (0.0 < hurst < 1.0):
        raise ValueError("hurst must lie in (0, 1).")

    idx = jnp.arange(steps, dtype=dtype)
    diff = jnp.abs(idx[:, None] - idx[None, :])
    two_h = 2.0 * hurst
    base = 0.5 * (
        jnp.power(diff + 1.0, two_h)
        + jnp.power(jnp.abs(diff - 1.0), two_h)
        - 2.0 * jnp.power(diff, two_h)
    )
    cov = base * (dt ** two_h)
    jitter = 1e-10 * jnp.eye(steps, dtype=dtype)
    return cov + jitter


def _sample_fractional_noise(
    z1: Array, cov_cholesky: Array, *, dtype: jnp.dtype
) -> Array:
    """Transform standard normals into fractional Brownian increments."""

    return jnp.matmul(z1, cov_cholesky.T).astype(dtype)


def simulate_rough_bergomi(
    key: jax.random.KeyArray,
    S0: float,
    T: float,
    r: float,
    q: float,
    cfg: MCConfig,
    params: RoughBergomiParams,
    *,
    return_full: bool = False,
    store_log: bool = False,
) -> Array | RoughBergomiPaths:
    """Simulate Rough Bergomi asset paths via Monte Carlo.

    The algorithm samples fractional Brownian motion increments using a
    Cholesky factorisation of the covariance matrix. The same Gaussian draws
    drive both the volatility and price Brownian motions, preserving the
    instantaneous correlation ``rho``.
    """

    timeline = time_grid(float(T), cfg.steps, dtype=cfg.dtype)
    dt = T / cfg.steps
    key_z1, key_z2 = jax.random.split(key)
    base_paths = cfg.base_paths
    z1 = jax.random.normal(key_z1, (base_paths, cfg.steps), dtype=cfg.dtype)
    z2 = jax.random.normal(key_z2, (base_paths, cfg.steps), dtype=cfg.dtype)

    if cfg.antithetic:
        z1 = jnp.concatenate([z1, -z1], axis=0)
        z2 = jnp.concatenate([z2, -z2], axis=0)

    cov = _fractional_increment_covariance(
        params.hurst, cfg.steps, dtype=cfg.dtype, dt=dt
    )
    chol = jnp.linalg.cholesky(cov)
    fbm_increments = _sample_fractional_noise(z1, chol, dtype=cfg.dtype)
    fbm_levels = jnp.concatenate(
        [jnp.zeros((fbm_increments.shape[0], 1), dtype=cfg.dtype), jnp.cumsum(fbm_increments, axis=1)],
        axis=1,
    )

    xi_curve = resolve_schedule(params.forward_variance, timeline, dtype=cfg.dtype)
    xi_curve = jnp.clip(xi_curve, a_min=1e-12)

    t_pow = timeline[1:] ** (2.0 * params.hurst)
    log_variance = params.eta * fbm_levels[:, 1:] - 0.5 * (params.eta ** 2) * t_pow
    variance_levels = jnp.exp(log_variance + jnp.log(xi_curve[1:]))
    variance_paths = jnp.concatenate(
        [jnp.full((variance_levels.shape[0], 1), xi_curve[0], dtype=cfg.dtype), variance_levels],
        axis=1,
    )

    sqrt_variance = jnp.sqrt(jnp.maximum(variance_paths[:, :-1], 1e-12))
    dW1 = jnp.sqrt(dt) * z1
    rho = jnp.clip(params.rho, -0.999, 0.999)
    dW_s = rho * dW1 + jnp.sqrt(1.0 - rho * rho) * jnp.sqrt(dt) * z2

    drifts = (r - q) - 0.5 * variance_paths[:, :-1]
    log_increments = drifts * dt + sqrt_variance * dW_s
    log_S0 = jnp.log(S0)
    cumulative = jnp.cumsum(log_increments, axis=1)
    log_paths = jnp.concatenate(
        [jnp.full((log_increments.shape[0], 1), log_S0, dtype=cfg.dtype), log_S0 + cumulative],
        axis=1,
    )
    price_paths = jnp.exp(log_paths)

    if not return_full:
        return price_paths

    log_values = log_paths if store_log else None
    metadata = {
        "model": "rough_bergomi",
        "hurst": params.hurst,
        "eta": params.eta,
        "rho": params.rho,
        "paths": price_paths.shape[0],
        "steps": cfg.steps,
    }
    return RoughBergomiPaths(
        values=price_paths,
        times=timeline,
        log_values=log_values,
        metadata=metadata,
        variance=variance_paths,
        fbm=fbm_levels,
    )


def calibrate_forward_variance(times: Array, variance_paths: Array) -> Array:
    """Estimate the forward variance curve from simulated variance paths."""

    if variance_paths.ndim != 2:
        raise ValueError("variance_paths must be a 2D array [paths, time].")
    if variance_paths.shape[1] != times.shape[0]:
        raise ValueError("times length must match variance path length.")
    return variance_paths.mean(axis=0)


def price_european_call_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    cfg: MCConfig,
    params: RoughBergomiParams,
) -> Array:
    """Price European call under rough Bergomi via Monte Carlo.

    Parameters
    ----------
    key : JAX PRNG key
    S0 : Initial stock price
    K : Strike price
    T : Time to maturity
    r : Risk-free rate
    q : Dividend yield
    cfg : Monte Carlo configuration
    params : Rough Bergomi parameters

    Returns
    -------
    price : Call option price
    """
    from ..core.engine import present_value

    paths = simulate_rough_bergomi(
        key, S0, T, r, q, cfg, params, return_full=False
    )
    ST = paths[:, -1]
    payoffs = jnp.maximum(ST - K, 0.0)
    return present_value(payoffs, jnp.array(T), r)
