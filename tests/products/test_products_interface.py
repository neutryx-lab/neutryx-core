import jax
import jax.numpy as jnp
import pytest

from neutryx.core.engine import MCConfig, present_value, simulate_gbm
from neutryx.products import (
    PAYOFF_CATALOGUE,
    AsianArithmetic,
    European,
    LookbackFloatStrikeCall,
    UpAndOutCall,
)


def test_catalogue_contains_expected_entries():
    expected = {
        "european",
        "asian_arithmetic",
        "up_and_out_call",
        "lookback_float_strike_call",
    }
    assert expected <= set(PAYOFF_CATALOGUE.keys())


def test_european_terminal_and_path_payoffs_align():
    option = European(K=100.0, T=1.0, is_call=True)
    path = jnp.array([100.0, 102.0, 105.0])
    assert option.supports_pde

    terminal_payoff = option.payoff_terminal(path[-1])
    path_payoff = option.payoff_path(path)
    assert jnp.isclose(terminal_payoff, path_payoff)

    grid = jnp.array([90.0, 100.0, 110.0])
    expected = jnp.maximum(grid - option.K, 0.0)
    computed = option.terminal_payoffs(grid)
    assert jnp.allclose(expected, computed)


def test_asian_requires_path_and_vectorises():
    option = AsianArithmetic(K=100.0, T=1.0, is_call=True)
    assert option.requires_path
    with pytest.raises(NotImplementedError):
        option.payoff_terminal(jnp.array(100.0))

    paths = jnp.array(
        [
            [100.0, 102.0, 104.0],
            [100.0, 99.0, 98.0],
        ]
    )
    expected = jnp.maximum(paths.mean(axis=1) - option.K, 0.0)
    computed = option.path_payoffs(paths)
    assert jnp.allclose(expected, computed)


def test_barrier_knock_out_behaviour():
    option = UpAndOutCall(K=100.0, T=1.0, B=120.0)
    hit_path = jnp.array([100.0, 121.0, 115.0])
    no_hit_path = jnp.array([100.0, 110.0, 115.0])

    assert option.payoff_path(hit_path) == 0.0
    expected_no_hit = jnp.maximum(no_hit_path[-1] - option.K, 0.0)
    assert jnp.isclose(option.payoff_path(no_hit_path), expected_no_hit)


def test_lookback_payoff_matches_definition():
    option = LookbackFloatStrikeCall(T=1.0)
    path = jnp.array([100.0, 90.0, 120.0])
    assert jnp.isclose(option.payoff_path(path), path[-1] - path.min())


def test_products_integrate_with_mc_engine():
    key = jax.random.PRNGKey(0)
    cfg = MCConfig(steps=16, paths=2000)
    paths = simulate_gbm(key, S0=100.0, mu=0.0, sigma=0.2, T=1.0, cfg=cfg)

    option = European(K=100.0, T=1.0)
    payoffs = option.path_payoffs(paths)
    price = present_value(payoffs, option.T, r=0.01)
    assert jnp.isfinite(price)
