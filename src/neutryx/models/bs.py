from typing import Dict, Literal, Tuple

import jax.numpy as jnp
from jax.scipy.stats import norm

from neutryx.core.autodiff import value_grad_hvp


def _d1d2(S, K, T, r, q, sigma):
    vol = jnp.maximum(sigma, 1e-12)
    sqrtT = jnp.sqrt(jnp.maximum(T, 1e-12))
    d1 = (jnp.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    return d1, d2

def price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: Literal["call","put"]="call") -> float:
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if kind == "call":
        return jnp.exp(-q*T)*S*norm.cdf(d1) - jnp.exp(-r*T)*K*norm.cdf(d2)
    else:
        return jnp.exp(-r*T)*K*norm.cdf(-d2) - jnp.exp(-q*T)*S*norm.cdf(-d1)

def greeks(S, K, T, r, q, sigma) -> Tuple[float,float,float,float]:
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    nd1 = jnp.exp(-0.5*d1*d1) / jnp.sqrt(2.0*jnp.pi)
    delta = jnp.exp(-q*T) * norm.cdf(d1)
    gamma = jnp.exp(-q*T) * nd1 / (S * sigma * jnp.sqrt(T) + 1e-12)
    vega  = jnp.exp(-q*T) * S * nd1 * jnp.sqrt(T)
    theta = - (jnp.exp(-q*T) * S * nd1 * sigma)/(2*jnp.sqrt(T)) - r*K*jnp.exp(-r*T)*norm.cdf(d2) + q*S*jnp.exp(-q*T)*norm.cdf(d1)
    return delta, gamma, vega, theta


def second_order_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: Literal["call", "put"] = "call") -> Dict[str, float]:
    """Compute second-order Greeks using algorithmic differentiation."""

    params = jnp.array([S, sigma], dtype=jnp.float64)

    def price_with_params(x):
        spot, vol = x
        return price(spot, K, T, r, q, vol, kind=kind)

    value, grad, hvp = value_grad_hvp(price_with_params)(params)
    column_s = hvp(jnp.array([1.0, 0.0], dtype=params.dtype))
    column_sigma = hvp(jnp.array([0.0, 1.0], dtype=params.dtype))

    return {
        "price": float(value),
        "delta": float(grad[0]),
        "vega": float(grad[1]),
        "gamma": float(column_s[0]),
        "vanna": float(column_s[1]),
        "vomma": float(column_sigma[1]),
    }

def implied_vol(S, K, T, r, q, price_target, kind="call", tol=1e-8, max_iter=100):
    # Bisection on [1e-6, 5.0]
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        p = price(S,K,T,r,q,mid,kind)
        lo, hi = (lo, mid) if p > price_target else (mid, hi)
        if jnp.abs(p - price_target) < tol:
            return mid
    return 0.5*(lo+hi)
