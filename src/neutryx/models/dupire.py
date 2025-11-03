"""Dupire local volatility model.

The Dupire (1994) local volatility model calibrates to a full volatility surface
and produces arbitrage-free option prices. The local volatility function is derived
from market option prices using Dupire's formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

from .sde import SDE

Array = jnp.ndarray


@dataclass
class DupireParams:
    """Parameters for Dupire local volatility model.

    The local volatility function sigma_L(S, t) is typically represented
    as a surface calibrated to market prices.

    Attributes
    ----------
    local_vol_surface : Callable[[float, float], float]
        Function sigma_L(S, t) returning local vol at spot S and time t
    S_grid : Array | None
        Spot grid for interpolation
    T_grid : Array | None
        Time grid for interpolation
    vol_values : Array | None
        Local volatility values on grid (S_grid, T_grid)
    """

    local_vol_surface: Callable[[float, float], float] | None = None
    S_grid: Array | None = None
    T_grid: Array | None = None
    vol_values: Array | None = None

    def local_vol(self, S: float, t: float) -> float:
        """Evaluate local volatility at (S, t).

        Parameters
        ----------
        S : float
            Spot price
        t : float
            Time

        Returns
        -------
        float
            Local volatility sigma_L(S, t)
        """
        if self.local_vol_surface is not None:
            return self.local_vol_surface(S, t)
        elif self.vol_values is not None and self.S_grid is not None and self.T_grid is not None:
            # Bilinear interpolation
            return jnp.interp(
                t,
                self.T_grid,
                jnp.interp(S, self.S_grid, self.vol_values, axis=0),
            )
        else:
            raise ValueError("No local volatility function or grid specified")


@dataclass
class DupireSDE(SDE):
    """Dupire local volatility SDE.

    dS = mu * S * dt + sigma_L(S, t) * S * dW

    Attributes
    ----------
    mu : float
        Drift (risk-free rate - dividend yield)
    params : DupireParams
        Local volatility parameters
    """

    mu: float
    params: DupireParams

    def drift(self, t: float, S: float) -> float:
        """Drift term."""
        return self.mu * S

    def diffusion(self, t: float, S: float) -> float:
        """Diffusion term with local volatility."""
        return self.params.local_vol(S, t) * S


def dupire_formula(
    K: Array,
    T: Array,
    call_prices: Array,
    r: float,
    q: float = 0.0,
    dK: float = 1.0,
    dT: float = 0.01,
) -> Array:
    """Compute local volatility from call option prices using Dupire's formula.

    Dupire's formula:
        sigma_L^2(K, T) = (dC/dT + (r-q)K*dC/dK + q*C) / (0.5 * K^2 * d2C/dK2)

    Parameters
    ----------
    K : Array
        Strike prices (must be on regular grid)
    T : Array
        Maturities (must be on regular grid)
    call_prices : Array
        Call prices C(K, T), shape (n_strikes, n_maturities)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    dK : float
        Strike grid spacing
    dT : float
        Time grid spacing

    Returns
    -------
    Array
        Local variance sigma_L^2(K, T)
    """
    # Ensure arrays
    K = jnp.asarray(K)
    T = jnp.asarray(T)
    call_prices = jnp.asarray(call_prices)

    # Compute derivatives using finite differences
    # dC/dT
    dC_dT = jnp.gradient(call_prices, dT, axis=1)

    # dC/dK (first derivative)
    dC_dK = jnp.gradient(call_prices, dK, axis=0)

    # d2C/dK2 (second derivative)
    d2C_dK2 = jnp.gradient(dC_dK, dK, axis=0)

    # Expand K for broadcasting
    K_expanded = K[:, jnp.newaxis]

    # Dupire numerator
    numerator = dC_dT + (r - q) * K_expanded * dC_dK + q * call_prices

    # Dupire denominator
    denominator = 0.5 * K_expanded ** 2 * d2C_dK2

    # Local variance (prevent division by zero)
    local_var = jnp.where(
        jnp.abs(denominator) > 1e-10,
        numerator / denominator,
        0.0,
    )

    # Ensure non-negative and reasonable bounds
    local_var = jnp.clip(local_var, 0.0, 10.0)

    return local_var


def calibrate_local_vol_surface(
    strikes: Array,
    maturities: Array,
    market_call_prices: Array,
    S0: float,
    r: float,
    q: float = 0.0,
) -> DupireParams:
    """Calibrate local volatility surface from market option prices.

    Parameters
    ----------
    strikes : Array
        Strike prices (regularly spaced)
    maturities : Array
        Maturities (regularly spaced)
    market_call_prices : Array
        Market call prices, shape (n_strikes, n_maturities)
    S0 : float
        Current spot price
    r : float
        Risk-free rate
    q : float
        Dividend yield

    Returns
    -------
    DupireParams
        Calibrated local volatility parameters
    """
    strikes = jnp.asarray(strikes)
    maturities = jnp.asarray(maturities)
    market_call_prices = jnp.asarray(market_call_prices)

    # Compute grid spacings
    dK = float(jnp.mean(jnp.diff(strikes)))
    dT = float(jnp.mean(jnp.diff(maturities)))

    # Compute local variance using Dupire's formula
    local_var = dupire_formula(strikes, maturities, market_call_prices, r, q, dK, dT)

    # Local volatility = sqrt(local variance)
    local_vol = jnp.sqrt(jnp.maximum(local_var, 0.0))

    # Create interpolation function
    # Note: We use strikes as S_grid (not perfectly accurate but practical)
    return DupireParams(
        S_grid=strikes,
        T_grid=maturities,
        vol_values=local_vol,
    )


def local_vol_from_implied_vol(
    K: Array,
    T: Array,
    implied_vols: Array,
    S0: float,
    r: float,
    q: float = 0.0,
) -> Array:
    """Compute local volatility from implied volatility surface.

    This uses the relationship between local vol and implied vol via
    Dupire's formula applied to Black-Scholes prices.

    Parameters
    ----------
    K : Array
        Strikes
    T : Array
        Maturities
    implied_vols : Array
        Implied volatilities, shape (n_strikes, n_maturities)
    S0 : float
        Spot price
    r : float
        Risk-free rate
    q : float
        Dividend yield

    Returns
    -------
    Array
        Local volatility surface
    """
    from neutryx.models.bs import price as bs_price

    K = jnp.asarray(K)
    T = jnp.asarray(T)
    implied_vols = jnp.asarray(implied_vols)

    # Compute call prices from implied vols
    K_grid, T_grid = jnp.meshgrid(K, T, indexing="ij")

    def compute_price(k, t, sigma):
        return bs_price(S0, k, t, r, q, sigma, kind="call")

    # Vectorize over all (K, T) pairs
    call_prices = jax.vmap(jax.vmap(compute_price))(
        K_grid, T_grid, implied_vols
    )

    # Apply Dupire formula
    dK = float(jnp.mean(jnp.diff(K)))
    dT = float(jnp.mean(jnp.diff(T)))
    local_var = dupire_formula(K, T, call_prices, r, q, dK, dT)

    return jnp.sqrt(jnp.maximum(local_var, 0.0))


__all__ = [
    "DupireParams",
    "DupireSDE",
    "dupire_formula",
    "calibrate_local_vol_surface",
    "local_vol_from_implied_vol",
]
