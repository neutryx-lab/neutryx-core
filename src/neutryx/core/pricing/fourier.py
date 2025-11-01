"""Fourier-based option pricing algorithms.

This module implements Carr-Madan FFT pricing and the COS method for
European options under models that expose their characteristic function.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp

Array = jnp.ndarray


class CharacteristicFunctionModel(ABC):
    """Interface for models defined by their characteristic function.

    Models inheriting from this class must provide the characteristic function
    of the log-price process under the risk-neutral measure. The characteristic
    function is defined as ``E[e^{iu \\log S_T}]``.
    """

    spot: float
    rate: float
    dividend: float

    def __init__(self, spot: float, rate: float, dividend: float = 0.0) -> None:
        self.spot = float(spot)
        self.rate = float(rate)
        self.dividend = float(dividend)

    @abstractmethod
    def characteristic_function(self, u: Array, maturity: float) -> Array:
        """Evaluate the characteristic function ``phi(u)`` at maturity."""

    def cumulants(self, maturity: float) -> tuple[float, float, float, float]:
        """Return the first four cumulants of ``log S_T``.

        Sub-classes can override this method to provide analytic cumulants. By
        default the cumulants are approximated from numerical derivatives of the
        characteristic function.
        """

        return _cumulants_from_characteristic(self.characteristic_function, maturity)


@dataclass
class BlackScholesCharacteristicModel(CharacteristicFunctionModel):
    """Black-Scholes model expressed via its characteristic function."""

    spot: float
    rate: float
    volatility: float
    dividend: float = 0.0

    def __post_init__(self) -> None:
        super().__init__(self.spot, self.rate, self.dividend)

    def characteristic_function(self, u: Array, maturity: float) -> Array:  # type: ignore[override]
        u = jnp.asarray(u, dtype=jnp.complex64)
        sigma2_t = (self.volatility**2) * maturity
        drift = jnp.log(self.spot) + (self.rate - self.dividend - 0.5 * self.volatility**2) * maturity
        return jnp.exp(1j * u * drift - 0.5 * sigma2_t * u * u)

    def cumulants(self, maturity: float) -> tuple[float, float, float, float]:  # type: ignore[override]
        mean = jnp.log(self.spot) + (self.rate - self.dividend - 0.5 * self.volatility**2) * maturity
        variance = (self.volatility**2) * maturity
        return float(mean), float(variance), 0.0, 0.0


def carr_madan_fft(
    model: CharacteristicFunctionModel,
    maturity: float,
    strikes: Sequence[float],
    *,
    alpha: float = 1.5,
    grid_size: int = 4096,
    eta: float = 0.25,
) -> Array:
    """Price European calls using the Carr-Madan FFT algorithm."""

    if grid_size & 1:
        raise ValueError("grid_size must be even for the Carr-Madan FFT method")

    strikes_arr = jnp.asarray(strikes)
    log_strikes = jnp.log(strikes_arr)

    j = jnp.arange(grid_size)
    u = j * eta

    cf_values = model.characteristic_function(u - 1j * (alpha + 1.0), maturity)
    denominator = alpha**2 + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
    psi = jnp.exp(-model.rate * maturity) * cf_values / denominator

    weights = _simpson_weights(grid_size)
    lambda_ = 2.0 * jnp.pi / (grid_size * eta)
    b = 0.5 * grid_size * lambda_

    fft_input = psi * weights * (eta / 3.0) * jnp.exp(1j * u * b)
    fft_values = jnp.fft.fft(fft_input)

    k_grid = -b + lambda_ * j
    call_prices = jnp.exp(-alpha * k_grid) / jnp.pi * fft_values.real

    return jnp.interp(log_strikes, k_grid, call_prices).real


def cos_method(
    model: CharacteristicFunctionModel,
    maturity: float,
    strikes: Sequence[float],
    *,
    expansion_terms: int = 256,
    truncation: float = 10.0,
    option: str = "call",
) -> Array:
    """Price European options using the COS method."""

    if expansion_terms <= 0:
        raise ValueError("expansion_terms must be positive")

    strikes_arr = jnp.asarray(strikes)
    log_k = jnp.log(strikes_arr)

    c1, c2, _, c4 = model.cumulants(maturity)
    c2 = float(max(c2, 1e-12))
    range_term = jnp.sqrt(c2 + jnp.sqrt(jnp.maximum(c4, 0.0)))
    a = float(c1 - truncation * range_term)
    b = float(c1 + truncation * range_term)

    k = jnp.arange(expansion_terms, dtype=jnp.float32)
    u = k * jnp.pi / (b - a)
    phi = model.characteristic_function(u, maturity) * jnp.exp(-1j * u * a)
    Fk = (2.0 / (b - a)) * jnp.real(phi)
    Fk = Fk.at[0].set(Fk[0] * 0.5)

    results = []
    for strike, log_strike in zip(strikes_arr, log_k):
        if option == "call":
            lower = float(jnp.clip(log_strike, a, b))
            upper = b
            chi = _chi(k, a, b, lower, upper)
            psi = _psi(k, a, b, lower, upper)
            payoff_coeff = chi - strike * psi
        elif option == "put":
            lower = a
            upper = float(jnp.clip(log_strike, a, b))
            chi = _chi(k, a, b, lower, upper)
            psi = _psi(k, a, b, lower, upper)
            payoff_coeff = strike * psi - chi
        else:
            raise ValueError("option must be 'call' or 'put'")

        price = jnp.exp(-model.rate * maturity) * jnp.sum(Fk * payoff_coeff)
        results.append(float(price.real))

    return jnp.asarray(results)


def _simpson_weights(n: int) -> Array:
    j = jnp.arange(n)
    weights = jnp.where((j == 0) | (j == n - 1), 1.0, jnp.where(j % 2 == 1, 4.0, 2.0))
    return weights


def _chi(k: Array, a: float, b: float, c: float, d: float) -> Array:
    if d <= c:
        return jnp.zeros(k.shape, dtype=jnp.float32)

    theta = k * jnp.pi / (b - a)
    exp_d = jnp.exp(d)
    exp_c = jnp.exp(c)
    term_d = jnp.cos(theta * (d - a)) + theta * jnp.sin(theta * (d - a))
    term_c = jnp.cos(theta * (c - a)) + theta * jnp.sin(theta * (c - a))
    denom = 1.0 + theta * theta
    return (exp_d * term_d - exp_c * term_c) / denom


def _psi(k: Array, a: float, b: float, c: float, d: float) -> Array:
    if d <= c:
        return jnp.zeros(k.shape, dtype=jnp.float32)

    theta = k * jnp.pi / (b - a)
    diff = jnp.sin(theta * (d - a)) - jnp.sin(theta * (c - a))
    result = jnp.where(k == 0, d - c, diff / theta)
    return result


def _cumulants_from_characteristic(cf, maturity: float, h: float = 1e-4) -> tuple[float, float, float, float]:
    offsets = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * h
    values = cf(offsets, maturity)

    d1 = (values[0] - 8.0 * values[1] + 8.0 * values[3] - values[4]) / (12.0 * h)
    d2 = (-values[0] + 16.0 * values[1] - 30.0 * values[2] + 16.0 * values[3] - values[4]) / (12.0 * h**2)
    d3 = (values[0] - 2.0 * values[1] + 2.0 * values[3] - values[4]) / (2.0 * h**3)
    d4 = (values[0] - 4.0 * values[1] + 6.0 * values[2] - 4.0 * values[3] + values[4]) / (h**4)

    e1 = d1 / (1j)
    e2 = -d2
    e3 = 1j * d3
    e4 = d4

    mean = float(jnp.real(e1))
    second = float(jnp.real(e2))
    third = float(jnp.real(e3))
    fourth = float(jnp.real(e4))

    variance = max(second - mean * mean, 0.0)
    central_third = third - 3.0 * mean * second + 2.0 * mean**3
    central_fourth = fourth - 4.0 * mean * third + 6.0 * mean * mean * second - 3.0 * mean**4
    cumulant_fourth = central_fourth - 3.0 * variance * variance

    return mean, variance, float(central_third), float(cumulant_fourth)
