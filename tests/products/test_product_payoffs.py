"""Unit tests for product payoff implementations."""
import jax
import jax.numpy as jnp

from neutryx.products.american import american_put_lsm
from neutryx.products.asian import AsianArithmetic
from neutryx.products.barrier import UpAndOutCall
from neutryx.products.lookback import LookbackFloatStrikeCall
from neutryx.products.vanilla import European


def test_european_payoff_call_and_put():
    option_call = European(K=100.0, T=1.0, is_call=True)
    option_put = European(K=100.0, T=1.0, is_call=False)

    spots = jnp.array([90.0, 100.0, 110.0])
    call_payoffs = jax.vmap(option_call.payoff)(spots)
    put_payoffs = jax.vmap(option_put.payoff)(spots)

    assert jnp.all(call_payoffs == jnp.array([0.0, 0.0, 10.0]))
    assert jnp.all(put_payoffs == jnp.array([10.0, 0.0, 0.0]))


def test_asian_arithmetic_payoff_call_and_put():
    path = jnp.array([100.0, 110.0, 120.0, 130.0])

    call = AsianArithmetic(K=115.0, T=1.0, is_call=True)
    put = AsianArithmetic(K=115.0, T=1.0, is_call=False)

    avg_price = path.mean()
    assert jnp.isclose(call.payoff(path), max(avg_price - 115.0, 0.0))
    assert jnp.isclose(put.payoff(path), max(115.0 - avg_price, 0.0))


def test_up_and_out_vectorized_payoff():
    option = UpAndOutCall(K=100.0, T=1.0, B=120.0)
    paths = jnp.array([
        [100.0, 110.0, 118.0, 115.0],  # survives
        [100.0, 121.0, 119.0, 130.0],  # knocked out
    ])

    payoffs = jax.vmap(option.payoff)(paths)
    assert jnp.isclose(payoffs[0], 15.0)
    assert jnp.isclose(payoffs[1], 0.0)


def test_lookback_payoff_matches_formula():
    option = LookbackFloatStrikeCall(T=1.0)
    path = jnp.array([100.0, 95.0, 105.0, 90.0, 110.0])

    expected = path[-1] - path.min()
    assert jnp.isclose(option.payoff(path), expected)


def test_american_put_lsm_monotonicity():
    decreasing_paths = jnp.array([
        [100.0, 85.0, 70.0],
        [100.0, 80.0, 60.0],
    ])
    increasing_paths = jnp.array([
        [100.0, 120.0, 140.0],
        [100.0, 130.0, 150.0],
    ])

    dt = 0.5
    price_decreasing = american_put_lsm(decreasing_paths, 100.0, 0.01, dt)
    price_increasing = american_put_lsm(increasing_paths, 100.0, 0.01, dt)

    assert price_decreasing > 0.0
    assert jnp.isclose(price_increasing, 0.0)
    assert price_decreasing > price_increasing
