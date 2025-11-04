"""Volatility surface utilities including SABR interpolation and smile modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp

ArrayLike = jnp.ndarray | float


@dataclass
class VolSmile:
    """Simple volatility smile with linear interpolation.

    Attributes:
        K: Strike prices
        iv: Implied volatilities
        T: Time to maturity
    """
    K: jnp.ndarray
    iv: jnp.ndarray
    T: float

    def interp(self, k):
        """Interpolate implied volatility at strike k using linear interpolation."""
        return jnp.interp(k, self.K, self.iv)


@dataclass
class SABRParameters:
    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_implied_vol(
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    params: SABRParameters,
    epsilon: float = 1e-07,
) -> ArrayLike:
    f = jnp.asarray(forward)
    k = jnp.asarray(strike)
    t = jnp.asarray(maturity)

    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu

    zero_maturity = jnp.where(t <= 0.0, 0.0, 1.0)

    def _atm_vol(fwd, mat):
        fk = fwd ** (1.0 - beta)
        factor = alpha / fk
        term = (
            ((1.0 - beta) ** 2 / 24.0) * (alpha**2 / (fwd ** (2.0 - 2.0 * beta)))
            + 0.25 * rho * beta * nu * alpha / (fwd ** (1.0 - beta))
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        return factor * (1.0 + term * mat)

    def _general_vol(fwd, strike, mat):
        fk_beta = (fwd * strike) ** ((1.0 - beta) / 2.0)
        log_fk = jnp.log(fwd / strike)
        z = (nu / alpha) * fk_beta * log_fk
        x_z = jnp.log((jnp.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

        z_safe = jnp.where(jnp.abs(z) < epsilon, 1.0, z / x_z)
        correction1 = (
            1.0
            + ((1.0 - beta) ** 2 / 24.0) * (log_fk**2)
            + ((1.0 - beta) ** 4 / 1920.0) * (log_fk**4)
        )
        term = (
            ((1.0 - beta) ** 2 / 24.0) * (alpha**2 / (fk_beta**2))
            + (rho * beta * nu * alpha) / (4.0 * fk_beta)
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        return (alpha / (fk_beta * correction1)) * z_safe * (1.0 + term * mat)

    atm_mask = jnp.abs(f - k) < epsilon
    atm_vol = _atm_vol(f, jnp.maximum(t, epsilon))
    general_vol = _general_vol(f, k, jnp.maximum(t, epsilon))

    vol = jnp.where(atm_mask, atm_vol, general_vol)
    return vol * zero_maturity


@dataclass
class SABRSurface:
    """
    SABR volatility surface with time-dependent parameters.

    Implements the VolatilitySurface protocol using SABR model interpolation.

    Attributes:
        expiries: Expiry times (in years)
        forwards: Forward prices at each expiry
        params: SABR parameters at each expiry
    """

    expiries: Sequence[float]
    forwards: Sequence[float]
    params: Sequence[SABRParameters]

    def implied_vol(self, expiry: float, strike: ArrayLike) -> ArrayLike:
        """
        Compute implied volatility using SABR model.

        Args:
            expiry: Time to expiry
            strike: Strike price(s)

        Returns:
            Implied volatility
        """
        exp_array = jnp.asarray(self.expiries)
        fwd_array = jnp.asarray(self.forwards)
        alpha_array = jnp.array([p.alpha for p in self.params])
        beta_array = jnp.array([p.beta for p in self.params])
        rho_array = jnp.array([p.rho for p in self.params])
        nu_array = jnp.array([p.nu for p in self.params])

        fwd = jnp.interp(expiry, exp_array, fwd_array)
        alpha = jnp.interp(expiry, exp_array, alpha_array)
        beta = jnp.interp(expiry, exp_array, beta_array)
        rho = jnp.interp(expiry, exp_array, rho_array)
        nu = jnp.interp(expiry, exp_array, nu_array)

        params = SABRParameters(float(alpha), float(beta), float(rho), float(nu))
        return sabr_implied_vol(fwd, strike, expiry, params)

    def value(self, expiry: float, strike: ArrayLike) -> ArrayLike:
        """Alias for implied_vol to implement Surface protocol."""
        return self.implied_vol(expiry, strike)

    def __call__(self, expiry: float, strike: ArrayLike) -> ArrayLike:
        """Alias for implied_vol for convenient syntax."""
        return self.implied_vol(expiry, strike)


@dataclass
class ImpliedVolSurface:
    """
    Grid-based implied volatility surface.

    Implements the VolatilitySurface protocol using bilinear interpolation
    on a regular grid of (expiry, strike) points.

    Attributes:
        expiries: Expiry times (in years)
        strikes: Strike prices
        vols: Implied volatility grid (expiries × strikes)
    """

    expiries: Sequence[float]
    strikes: Sequence[float]
    vols: Sequence[Sequence[float]]

    def implied_vol(self, expiry: ArrayLike, strike: ArrayLike) -> ArrayLike:
        """
        Compute implied volatility via bilinear interpolation.

        Args:
            expiry: Time(s) to expiry
            strike: Strike price(s)

        Returns:
            Implied volatility
        """
        expiry_array = jnp.asarray(self.expiries)
        strike_array = jnp.asarray(self.strikes)
        vol_grid = jnp.asarray(self.vols)

        def _interp_single(exp_val, strike_val):
            strike_slice = jnp.array(
                [jnp.interp(strike_val, strike_array, row) for row in vol_grid]
            )
            return jnp.interp(exp_val, expiry_array, strike_slice)

        vectorised = jnp.vectorize(_interp_single)
        return vectorised(expiry, strike)

    def value(self, expiry: ArrayLike, strike: ArrayLike) -> ArrayLike:
        """Alias for implied_vol to implement Surface protocol."""
        return self.implied_vol(expiry, strike)

    def __call__(self, expiry: ArrayLike, strike: ArrayLike) -> ArrayLike:
        """Alias for implied_vol for convenient syntax."""
        return self.implied_vol(expiry, strike)
