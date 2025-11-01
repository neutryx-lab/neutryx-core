"""Heston model characteristic function implementation.

This module provides the characteristic function for the Heston (1993) stochastic volatility model.
"""
import jax.numpy as jnp

def characteristic_function(u, t, v0, kappa, theta, sigma, rho, r, q=0.0):
    """Compute the Heston characteristic function under risk-neutral measure.

    Implements the characteristic function from Heston (1993) for stochastic volatility.
    The model dynamics are:
        dS/S = (r-q)dt + sqrt(v)dW_S
        dv = kappa(theta - v)dt + sigma*sqrt(v)dW_v
        corr(dW_S, dW_v) = rho

    Args:
        u: Complex argument for characteristic function
        t: Time to maturity
        v0: Initial variance
        kappa: Mean reversion speed of variance
        theta: Long-term mean variance
        sigma: Volatility of variance (vol-of-vol)
        rho: Correlation between asset and variance Brownian motions
        r: Risk-free interest rate
        q: Dividend yield (default: 0.0)

    Returns:
        Complex value of characteristic function phi(u) at given parameters
    """
    iu = 1j * u
    a = kappa * theta
    b = kappa - rho * sigma * iu
    d = jnp.sqrt(b*b + (sigma**2) * (iu + u*u))
    g = (b - d) / (b + d)
    exp_dt = jnp.exp(-d * t)
    C = iu * (r - q) * t + (a / (sigma**2)) * ((b - d) * t - 2.0 * jnp.log((1 - g * exp_dt) / (1 - g)))
    D = ((b - d) / (sigma**2)) * ((1 - exp_dt) / (1 - g * exp_dt))
    return jnp.exp(C + D * v0)
