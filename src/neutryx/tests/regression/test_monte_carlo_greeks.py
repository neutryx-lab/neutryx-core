import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from neutryx.core.engine import MCConfig
from neutryx.models import bs as bs_model
from neutryx.monte_carlo import (
    AMCInputs,
    american_put_lsm,
    asian_arithmetic_call,
    european_call,
    european_put,
    pathwise_price_and_greeks,
)


def _analytic_call_metrics(S, K, T, r, q, sigma):
    d1, d2 = bs_model._d1d2(S, K, T, r, q, sigma)
    disc_q = jnp.exp(-q * T)
    disc_r = jnp.exp(-r * T)
    price = bs_model.price(S, K, T, r, q, sigma, kind="call")
    delta = disc_q * norm.cdf(d1)
    vega = disc_q * S * norm.pdf(d1) * jnp.sqrt(T)
    return price, delta, vega


def _analytic_put_metrics(S, K, T, r, q, sigma):
    call_price, call_delta, vega = _analytic_call_metrics(S, K, T, r, q, sigma)
    disc_r = jnp.exp(-r * T)
    disc_q = jnp.exp(-q * T)
    price = call_price - S * disc_q + K * disc_r
    delta = call_delta - disc_q
    return price, delta, vega


def test_pathwise_european_matches_black_scholes():
    key = jax.random.PRNGKey(123)
    params = AMCInputs(S0=100.0, r=0.01, q=0.0, sigma=0.2, T=1.0)
    cfg = MCConfig(steps=96, paths=40_000, antithetic=True)

    payoffs = [european_call(100.0), european_put(100.0)]
    results = pathwise_price_and_greeks(key, params, cfg, payoffs)

    call_price, call_delta, call_vega = _analytic_call_metrics(100.0, 100.0, 1.0, 0.01, 0.0, 0.2)
    put_price, put_delta, put_vega = _analytic_put_metrics(100.0, 100.0, 1.0, 0.01, 0.0, 0.2)

    call = results["european_call_100"]
    put = results["european_put_100"]

    assert jnp.isclose(call.price, call_price, rtol=5e-3, atol=5e-2)
    assert jnp.isclose(call.delta, call_delta, rtol=5e-2, atol=2e-2)
    assert jnp.isclose(call.vega, call_vega, rtol=5e-2, atol=3e-2)

    assert jnp.isclose(put.price, put_price, rtol=5e-3, atol=5e-2)
    assert jnp.isclose(put.delta, put_delta, rtol=5e-2, atol=2e-2)
    assert jnp.isclose(put.vega, put_vega, rtol=5e-2, atol=3e-2)


def test_pathwise_multipayoff_regression():
    key = jax.random.PRNGKey(321)
    params = AMCInputs(S0=95.0, r=0.015, q=0.0, sigma=0.25, T=1.5)
    cfg = MCConfig(steps=72, paths=20_000, antithetic=True)

    payoffs = [
        european_call(90.0),
        asian_arithmetic_call(92.5),
        american_put_lsm(90.0),
    ]

    results = pathwise_price_and_greeks(key, params, cfg, payoffs)

    baseline = {
        "european_call_90": {
            "price": 14.841634,
            "delta": 0.6554522,
            "vega": 42.027626,
        },
        "asian_call_92.5": {
            "price": 8.342668,
            "delta": 0.6087635,
            "vega": 24.883976,
        },
        "american_put_90": {
            "price": 8.043872,
            "delta": -0.3614939,
            "vega": 42.964798,
        },
    }

    for name, expected in baseline.items():
        result = results[name]
        assert jnp.isclose(result.price, expected["price"], rtol=3e-2, atol=5e-2)
        assert jnp.isclose(result.delta, expected["delta"], rtol=6e-2, atol=5e-2)
        assert jnp.isclose(result.vega, expected["vega"], rtol=8e-2, atol=8e-2)

