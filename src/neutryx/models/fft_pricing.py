"""FFT-based option pricing methods.

This module implements Fast Fourier Transform (FFT) based pricing for European options,
particularly useful for models with known characteristic functions.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
import jax


def carr_madan_fft(
    S0: float,
    K: float,
    T: float,
    r: float,
    char_func: callable,
    N: int = 4096,
    alpha: float = 1.5,
    eta: float = 0.25,
) -> float:
    """Price European call option using Carr-Madan FFT method.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    char_func : callable
        Characteristic function phi(u) of log(S_T)
    N : int
        Number of FFT points (power of 2)
    alpha : float
        Damping parameter (should be > 1)
    eta : float
        Grid spacing in frequency domain

    Returns
    -------
    float
        European call option price
    """
    # Grid in frequency domain
    lambda_val = 2 * jnp.pi / (N * eta)
    b = N * lambda_val / 2

    # Strike grid
    ku = -b + lambda_val * jnp.arange(N)
    K_grid = jnp.exp(ku)

    # FFT integration
    v = jnp.arange(N) * eta

    # Modified characteristic function for call option
    def psi(v_val):
        u = v_val - (alpha + 1) * 1j
        numerator = jnp.exp(-r * T) * char_func(u)
        denominator = alpha**2 + alpha - v_val**2 + 1j * (2 * alpha + 1) * v_val
        return numerator / denominator

    # Compute FFT input
    x = jnp.exp(1j * v * b) * psi(v) * eta
    # Simpson's rule weights
    w = jnp.ones(N)
    w = w.at[::2].set(2.0)
    w = w.at[1::2].set(4.0)
    w = w.at[0].set(1.0)
    w = w.at[-1].set(1.0)
    w = w * (1.0 / 3.0)

    x = x * w

    # Apply FFT
    fft_result = jnp.fft.fft(x)

    # Extract call prices
    call_prices = (jnp.exp(-alpha * ku) / jnp.pi) * jnp.real(fft_result)

    # Interpolate to get price at strike K
    price = jnp.interp(K, K_grid, call_prices)

    return float(price)


def lewis_fft(
    S0: float,
    K: float,
    T: float,
    r: float,
    char_func: callable,
    N: int = 4096,
    dk: float = 0.01,
) -> float:
    """Price European option using Lewis (2001) FFT method.

    This method is more stable than Carr-Madan for some parameter ranges.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    char_func : callable
        Characteristic function phi(u) of log(S_T)
    N : int
        Number of FFT points
    dk : float
        Strike spacing in log-space

    Returns
    -------
    float
        European call option price
    """
    # Log-strike grid
    k_grid = dk * jnp.arange(-N / 2, N / 2)

    # Frequency grid
    dv = 2 * jnp.pi / (N * dk)
    v = dv * jnp.arange(N)

    # Centered around log(K/S0)
    log_K_S0 = jnp.log(K / S0)

    # Modified characteristic function
    def integrand(v_val):
        u = v_val - 0.5j
        phi = char_func(u)
        return jnp.exp(-1j * v_val * log_K_S0) * phi / (v_val**2 + 0.25)

    # Compute FFT input
    x = integrand(v) * dv

    # Apply FFT
    fft_result = jnp.fft.fft(x)

    # Extract option prices
    prices = S0 - jnp.sqrt(S0 * K) * jnp.exp(-r * T) * jnp.real(fft_result) / jnp.pi

    # Interpolate to get price at the target strike
    k_target = jnp.log(K / S0)
    price = jnp.interp(k_target, k_grid, prices)

    return float(price)


def heston_char_func(u: complex, S0: float, T: float, r: float, q: float,
                     v0: float, kappa: float, theta: float, sigma_v: float, rho: float) -> complex:
    """Characteristic function for Heston stochastic volatility model.

    Parameters
    ----------
    u : complex
        Frequency parameter
    S0 : float
        Initial spot price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-term variance
    sigma_v : float
        Volatility of variance
    rho : float
        Correlation between spot and variance

    Returns
    -------
    complex
        Characteristic function value
    """
    # Heston parameters
    d = jnp.sqrt((rho * sigma_v * u * 1j - kappa)**2 + sigma_v**2 * (u * 1j + u**2))
    g = (kappa - rho * sigma_v * u * 1j - d) / (kappa - rho * sigma_v * u * 1j + d)

    # Characteristic function components
    C = (r - q) * u * 1j * T + (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * u * 1j - d) * T - 2 * jnp.log((1 - g * jnp.exp(-d * T)) / (1 - g))
    )

    D = ((kappa - rho * sigma_v * u * 1j - d) / sigma_v**2) * (
        (1 - jnp.exp(-d * T)) / (1 - g * jnp.exp(-d * T))
    )

    phi = jnp.exp(C + D * v0 + 1j * u * jnp.log(S0))

    return phi


def price_european_fft_heston(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    is_call: bool = True,
) -> float:
    """Price European option under Heston model using FFT.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-term variance
    sigma_v : float
        Volatility of variance
    rho : float
        Correlation between spot and variance
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Option price
    """
    # Create characteristic function
    def char_func(u):
        return heston_char_func(u, S0, T, r, q, v0, kappa, theta, sigma_v, rho)

    # Price using Carr-Madan FFT
    call_price = carr_madan_fft(S0, K, T, r, char_func)

    if is_call:
        return call_price
    else:
        # Put-call parity
        put_price = call_price - S0 * jnp.exp(-q * T) + K * jnp.exp(-r * T)
        return float(put_price)


def bs_char_func(u: complex, S0: float, T: float, r: float, q: float, sigma: float) -> complex:
    """Characteristic function for Black-Scholes model.

    Parameters
    ----------
    u : complex
        Frequency parameter
    S0 : float
        Initial spot price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility

    Returns
    -------
    complex
        Characteristic function value
    """
    log_S = jnp.log(S0) + (r - q - 0.5 * sigma**2) * T
    var = sigma**2 * T
    phi = jnp.exp(1j * u * log_S - 0.5 * u**2 * var)
    return phi


def price_european_fft_bs(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    is_call: bool = True,
) -> float:
    """Price European option under Black-Scholes using FFT.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    q : float
        Dividend yield
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Option price
    """
    def char_func(u):
        return bs_char_func(u, S0, T, r, q, sigma)

    call_price = carr_madan_fft(S0, K, T, r, char_func)

    if is_call:
        return call_price
    else:
        put_price = call_price - S0 * jnp.exp(-q * T) + K * jnp.exp(-r * T)
        return float(put_price)


__all__ = [
    "carr_madan_fft",
    "lewis_fft",
    "heston_char_func",
    "bs_char_func",
    "price_european_fft_heston",
    "price_european_fft_bs",
]
