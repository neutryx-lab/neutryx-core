"""Variance Gamma (VG) process for option pricing.

The Variance Gamma model is a pure jump Lévy process with infinite activity.
It can be constructed as a Brownian motion with drift evaluated at a random time
given by a Gamma process.

References:
    Madan, D. B., Carr, P. P., & Chang, E. C. (1998).
    The variance gamma process and option pricing.
    European Finance Review, 2(1), 79-105.
"""
import jax
import jax.numpy as jnp

from neutryx.core.engine import Array, MCConfig
from neutryx.core.utils.math import compute_option_payoff, discount_payoff


def simulate_variance_gamma(
    key: Array,
    S0: float,
    mu: float,
    theta: float,
    sigma: float,
    nu: float,
    T: float,
    cfg: MCConfig,
) -> Array:
    """Simulate Variance Gamma process paths.

    The VG process is defined as:
    X(t) = theta * G(t) + sigma * W(G(t))

    where G(t) is a Gamma process with mean t and variance nu * t,
    and W is a standard Brownian motion.

    Parameters
    ----------
    key : Array
        PRNG key
    S0 : float
        Initial asset price
    mu : float
        Drift (r - q)
    theta : float
        Drift of the underlying Brownian motion
    sigma : float
        Volatility of the underlying Brownian motion
    nu : float
        Variance rate of the Gamma time change (controls kurtosis)
    T : float
        Time to maturity
    cfg : MCConfig
        Monte Carlo configuration

    Returns
    -------
    Array
        Simulated paths of shape [paths, steps+1]
    """
    dtype = cfg.dtype
    dt = T / cfg.steps

    # Gamma process parameters
    # G(t) ~ Gamma(shape = t/nu, scale = nu)
    shape = dt / nu
    scale = nu

    # Split keys
    key_gamma, key_norm = jax.random.split(key)

    # Generate Gamma time changes
    gamma_increments = jax.random.gamma(
        key_gamma, a=shape, shape=(cfg.base_paths, cfg.steps)
    ) * scale

    # Generate Brownian motion increments
    normals = jax.random.normal(key_norm, (cfg.base_paths, cfg.steps), dtype=dtype)

    # VG increments: X(dt) = theta * G(dt) + sigma * W(G(dt))
    # W(G(dt)) ~ N(0, G(dt)), so we scale normals by sqrt(gamma_increments)
    vg_increments = theta * gamma_increments + sigma * jnp.sqrt(gamma_increments) * normals

    # Adjust for drift compensation
    # E[exp(X(t)) - 1] = exp(theta*t + 0.5*sigma^2*t) * (1/(1 - theta*nu - 0.5*sigma^2*nu))^(t/nu) - 1
    omega = (1.0 / nu) * jnp.log(1.0 - theta * nu - 0.5 * sigma**2 * nu)
    drift = (mu + omega) * dt

    increments = drift + vg_increments

    if cfg.antithetic:
        anti_increments = drift + theta * gamma_increments - sigma * jnp.sqrt(gamma_increments) * normals
        increments = jnp.concatenate([increments, anti_increments], axis=0)

    log_S0 = jnp.log(jnp.asarray(S0, dtype=dtype))
    total_paths = increments.shape[0]
    cum_returns = jnp.cumsum(increments, axis=1)
    log_paths = jnp.concatenate(
        [jnp.full((total_paths, 1), log_S0, dtype=dtype), log_S0 + cum_returns],
        axis=1,
    )

    return jnp.exp(log_paths)


def price_vanilla_vg_mc(
    key: Array,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    theta: float,
    sigma: float,
    nu: float,
    cfg: MCConfig,
    kind: str = "call",
) -> float:
    """Price vanilla European option under Variance Gamma model using Monte Carlo.

    Parameters
    ----------
    key : Array
        PRNG key
    S0 : float
        Initial asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    theta : float
        Drift of the underlying Brownian motion
    sigma : float
        Volatility of the underlying Brownian motion
    nu : float
        Variance rate (controls kurtosis and jump intensity)
    cfg : MCConfig
        Monte Carlo configuration
    kind : str
        "call" or "put"

    Returns
    -------
    float
        Option price
    """
    mu = r - q
    paths = simulate_variance_gamma(key, S0, mu, theta, sigma, nu, T, cfg)
    ST = paths[:, -1]

    payoffs = compute_option_payoff(ST, K, kind)
    discounted = discount_payoff(payoffs, r, T)
    return float(discounted.mean())


def vg_characteristic_function(u, t, theta, sigma, nu):
    """Characteristic function of the Variance Gamma process.

    Useful for Fourier-based pricing methods (FFT, COS method).

    Parameters
    ----------
    u : complex or Array
        Frequency parameter
    t : float
        Time
    theta : float
        Drift parameter
    sigma : float
        Volatility parameter
    nu : float
        Variance rate

    Returns
    -------
    complex or Array
        Characteristic function value
    """
    # phi(u) = (1 - i*u*theta*nu + 0.5*sigma^2*nu*u^2)^(-t/nu)
    return (1.0 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u**2) ** (-t / nu)


__all__ = [
    "simulate_variance_gamma",
    "price_vanilla_vg_mc",
    "vg_characteristic_function",
]
