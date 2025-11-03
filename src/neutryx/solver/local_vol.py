"""Dupire local volatility surface construction from implied vols."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..models import bs as bs_model

Array = jnp.ndarray


@dataclass
class LocalVolSurface:
    strikes: Array
    maturities: Array
    local_vol: Array
    call_prices: Array
    implied_vol: Array

    def value(self, T: float, K: float) -> float:
        """Bilinear interpolation of local volatility at (T, K)."""
        T_arr = jnp.asarray(self.maturities)
        K_arr = jnp.asarray(self.strikes)

        t_idx = jnp.searchsorted(T_arr, T, side="left")
        t_idx = jnp.clip(t_idx, 1, len(T_arr) - 1)
        k_idx = jnp.searchsorted(K_arr, K, side="left")
        k_idx = jnp.clip(k_idx, 1, len(K_arr) - 1)

        t0, t1 = T_arr[t_idx - 1], T_arr[t_idx]
        k0, k1 = K_arr[k_idx - 1], K_arr[k_idx]
        wt = jnp.where(t1 == t0, 0.0, (T - t0) / (t1 - t0))
        wk = jnp.where(k1 == k0, 0.0, (K - k0) / (k1 - k0))

        lv = self.local_vol
        v00 = lv[t_idx - 1, k_idx - 1]
        v01 = lv[t_idx - 1, k_idx]
        v10 = lv[t_idx, k_idx - 1]
        v11 = lv[t_idx, k_idx]

        return float(
            (1 - wt) * ((1 - wk) * v00 + wk * v01)
            + wt * ((1 - wk) * v10 + wk * v11)
        )


def call_price_surface_from_iv(
    S0: float,
    strikes: Array,
    maturities: Array,
    implied_vol: Array,
    *,
    r: float = 0.0,
    q: float = 0.0,
) -> Array:
    """Map implied volatility surface to call prices via Black-Scholes."""

    strikes = jnp.asarray(strikes)
    maturities = jnp.asarray(maturities)
    vol_surface = jnp.asarray(implied_vol)

    def price_row(T, vol_row):
        return jax.vmap(lambda K, sigma: bs_model.price(S0, K, T, r, q, sigma, kind="call"))(strikes, vol_row)

    return jax.vmap(price_row)(maturities, vol_surface)


def dupire_local_volatility_surface(
    S0: float,
    strikes: Array,
    maturities: Array,
    implied_vol: Array,
    *,
    r: float = 0.0,
    q: float = 0.0,
    smoothing: float = 1e-6,
    min_vol: float = 1e-4,
    max_vol: float = 5.0,
) -> LocalVolSurface:
    """Compute Dupire local volatility surface using finite differences."""

    strikes = jnp.asarray(strikes)
    maturities = jnp.asarray(maturities)
    implied_vol = jnp.asarray(implied_vol)

    call_prices = call_price_surface_from_iv(S0, strikes, maturities, implied_vol, r=r, q=q)

    if maturities.size > 1:
        diff_t = jnp.diff(maturities)
        dt = float(diff_t[0])
        if not jnp.allclose(diff_t, diff_t[0]):
            raise ValueError("Non-uniform maturities not supported for Dupire solver.")
    else:
        dt = 1.0

    if strikes.size > 1:
        diff_k = jnp.diff(strikes)
        dk = float(diff_k[0])
        if not jnp.allclose(diff_k, diff_k[0]):
            raise ValueError("Non-uniform strikes not supported for Dupire solver.")
    else:
        dk = 1.0

    dC_dT = jnp.gradient(call_prices, dt, axis=0)
    dC_dK = jnp.gradient(call_prices, dk, axis=1)
    d2C_dK2 = jnp.gradient(dC_dK, dk, axis=1)

    numerator = dC_dT + (r - q) * strikes[None, :] * dC_dK + q * call_prices
    denom = 0.5 * strikes[None, :] ** 2 * d2C_dK2
    denom = jnp.where(jnp.abs(denom) < smoothing, jnp.sign(denom) * smoothing, denom)

    local_var = numerator / denom
    local_var = jnp.clip(local_var, min_vol ** 2, max_vol ** 2)
    local_vol = jnp.sqrt(local_var)

    return LocalVolSurface(
        strikes=strikes,
        maturities=maturities,
        local_vol=local_vol,
        call_prices=call_prices,
        implied_vol=implied_vol,
    )


__all__ = [
    "LocalVolSurface",
    "call_price_surface_from_iv",
    "dupire_local_volatility_surface",
]

