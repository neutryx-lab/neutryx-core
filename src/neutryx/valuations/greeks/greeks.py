"""Greeks calculation using analytical, finite difference, and pathwise methods.

This module provides utilities for computing option sensitivities (Greeks) through
multiple methods:
- Analytical formulas (Black-Scholes)
- Finite difference (bump-and-revalue)
- Automatic differentiation (pathwise)
"""

import inspect

import jax
import jax.numpy as jnp

from neutryx.models import bs as bs_model


def bs_analytic_greeks(S, K, T, r, q, sigma):
    """Compute Black-Scholes analytical Greeks.

    Returns
    -------
    tuple
        (delta, gamma, vega, theta, rho) for the given parameters
    """
    return bs_model.greeks(S, K, T, r, q, sigma)


def bs_second_order_greeks(S, K, T, r, q, sigma, kind="call"):
    """Expose second-order Greeks computed via autodiff for Black-Scholes.

    Parameters
    ----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    kind : str
        "call" or "put"

    Returns
    -------
    dict
        Dictionary containing second-order Greeks (gamma, vanna, volga, etc.)
    """
    return bs_model.second_order_greeks(S, K, T, r, q, sigma, kind=kind)


_SPOT_KEYWORD_HINTS = ("spot", "underlying", "asset")


def _call_pricer_with_spot(pricer, spot, kwargs):
    """Call ``pricer`` ensuring the spot argument is passed correctly."""
    signature = inspect.signature(pricer)
    candidate_name = None
    candidate_param = None

    for param in signature.parameters.values():
        if param.name == "self":
            continue
        lower_name = param.name.lower()
        if lower_name in {"s", "s0"} or any(hint in lower_name for hint in _SPOT_KEYWORD_HINTS):
            candidate_name = param.name
            candidate_param = param
            break

    if candidate_name and candidate_param.kind is not inspect.Parameter.POSITIONAL_ONLY:
        call_kwargs = dict(kwargs)
        call_kwargs[candidate_name] = spot
        try:
            return pricer(**call_kwargs)
        except TypeError:
            # Fall back to positional invocation if keyword passing fails
            pass

    cleaned_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key.lower() not in {"s", "s0"}
    }
    return pricer(spot, **cleaned_kwargs)


def mc_delta_bump(pricer, S, bump=1e-4, **kwargs):
    """Compute delta using finite difference (bump and revalue).

    Parameters
    ----------
    pricer : callable
        Pricing function that accepts S and other parameters
    S : float
        Current asset price
    bump : float
        Relative bump size for finite difference
    **kwargs
        Additional parameters passed to pricer

    Returns
    -------
    float
        Delta estimate via finite difference
    """
    base = _call_pricer_with_spot(pricer, S, kwargs)
    up = _call_pricer_with_spot(pricer, S * (1 + bump), kwargs)
    return (up - base) / (S * bump)


def mc_vega_bump(pricer, sigma, bump=0.01, **kwargs):
    """Compute vega using finite difference (bump and revalue).

    Parameters
    ----------
    pricer : callable
        Pricing function that accepts sigma and other parameters
    sigma : float
        Current volatility
    bump : float
        Absolute bump size for volatility (e.g., 0.01 = 1%)
    **kwargs
        Additional parameters passed to pricer

    Returns
    -------
    float
        Vega estimate (price change per 1% vol change)
    """
    base = pricer(sigma=sigma, **kwargs)
    up = pricer(sigma=sigma + bump, **kwargs)
    return (up - base) / bump


def mc_theta_bump(pricer, T, dt=1.0/365.0, **kwargs):
    """Compute theta using finite difference (bump and revalue).

    Parameters
    ----------
    pricer : callable
        Pricing function that accepts T and other parameters
    T : float
        Current time to maturity
    dt : float
        Time step for finite difference (default: 1 day)
    **kwargs
        Additional parameters passed to pricer

    Returns
    -------
    float
        Theta estimate (price change per day)
    """
    if T <= dt:
        # At expiry, theta is undefined
        return 0.0

    base = pricer(T=T, **kwargs)
    down = pricer(T=T - dt, **kwargs)
    return (down - base) / dt


def mc_rho_bump(pricer, r, bump=0.0001, **kwargs):
    """Compute rho using finite difference (bump and revalue).

    Parameters
    ----------
    pricer : callable
        Pricing function that accepts r and other parameters
    r : float
        Current risk-free rate
    bump : float
        Absolute bump size for rate (e.g., 0.0001 = 1bp)
    **kwargs
        Additional parameters passed to pricer

    Returns
    -------
    float
        Rho estimate (price change per 1bp rate change)
    """
    base = pricer(r=r, **kwargs)
    up = pricer(r=r + bump, **kwargs)
    return (up - base) / bump


def mc_gamma_bump(pricer, S, bump=1e-4, **kwargs):
    """Compute gamma using second-order finite difference.

    Parameters
    ----------
    pricer : callable
        Pricing function that accepts S and other parameters
    S : float
        Current asset price
    bump : float
        Relative bump size for finite difference
    **kwargs
        Additional parameters passed to pricer

    Returns
    -------
    float
        Gamma estimate via second-order finite difference
    """
    base = _call_pricer_with_spot(pricer, S, kwargs)
    up = _call_pricer_with_spot(pricer, S * (1 + bump), kwargs)
    down = _call_pricer_with_spot(pricer, S * (1 - bump), kwargs)
    return (up - 2 * base + down) / (S * bump) ** 2


def mc_all_greeks_bump(pricer, S, K, T, r, q, sigma,
                       delta_bump=1e-4, vega_bump=0.01,
                       theta_dt=1.0/365.0, rho_bump=0.0001, **kwargs):
    """Compute all first-order Greeks using finite difference.

    Parameters
    ----------
    pricer : callable
        Pricing function that accepts all parameters
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    delta_bump : float
        Relative bump for delta
    vega_bump : float
        Absolute bump for vega
    theta_dt : float
        Time step for theta
    rho_bump : float
        Absolute bump for rho
    **kwargs
        Additional parameters passed to pricer

    Returns
    -------
    dict
        Dictionary containing all Greeks
    """
    params = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, **kwargs}

    delta = mc_delta_bump(pricer, S=S, bump=delta_bump, K=K, T=T, r=r, q=q, sigma=sigma, **kwargs)
    vega = mc_vega_bump(pricer, sigma=sigma, bump=vega_bump, S=S, K=K, T=T, r=r, q=q, **kwargs)
    theta = mc_theta_bump(pricer, T=T, dt=theta_dt, S=S, K=K, r=r, q=q, sigma=sigma, **kwargs)
    rho = mc_rho_bump(pricer, r=r, bump=rho_bump, S=S, K=K, T=T, q=q, sigma=sigma, **kwargs)
    gamma = mc_gamma_bump(pricer, S=S, bump=delta_bump, K=K, T=T, r=r, q=q, sigma=sigma, **kwargs)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def mc_greeks_autodiff(pricer_fn, S, K, T, r, q, sigma, **kwargs):
    """Compute Greeks using automatic differentiation.

    This uses JAX's autodiff to compute exact gradients of the pricing function.
    More accurate than finite difference but requires JAX-compatible pricer.

    Parameters
    ----------
    pricer_fn : callable
        JAX-compatible pricing function
    S, K, T, r, q, sigma : float
        Standard option parameters
    **kwargs
        Additional parameters

    Returns
    -------
    dict
        Dictionary containing price and Greeks
    """
    def price_wrapper(params):
        return pricer_fn(S=params[0], K=K, T=T, r=params[1],
                        q=q, sigma=params[2], **kwargs)

    params = jnp.array([S, r, sigma])

    # Compute price and gradients
    price = price_wrapper(params)
    grads = jax.grad(price_wrapper)(params)

    return {
        "price": float(price),
        "delta": float(grads[0]),
        "rho": float(grads[1]),
        "vega": float(grads[2]),
    }
