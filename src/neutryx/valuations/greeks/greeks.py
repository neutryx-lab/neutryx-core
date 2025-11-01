import jax, jax.numpy as jnp

from neutryx.models import bs as bs_model

def bs_analytic_greeks(S, K, T, r, q, sigma):
    return bs_model.greeks(S, K, T, r, q, sigma)


def bs_second_order_greeks(S, K, T, r, q, sigma, kind="call"):
    """Expose second-order Greeks computed via autodiff for Black-Scholes."""

    return bs_model.second_order_greeks(S, K, T, r, q, sigma, kind=kind)

def mc_delta_bump(pricer, bump=1e-4, *args, **kwargs):
    def f(S):
        return pricer(S=S, *args, **kwargs)
    base = f(kwargs["S"])
    up = f(kwargs["S"]*(1+bump))
    return (up - base) / (kwargs["S"]*bump + 1e-12)
