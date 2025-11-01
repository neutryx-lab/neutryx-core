import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from neutryx.core.engine import MCConfig, price_vanilla_mc, simulate_gbm
from neutryx.monte_carlo.qmc import (
    MLMCLevel,
    MLMCOrchestrator,
    SobolGenerator,
    price_european_qmc,
)
from neutryx.products.asian import AsianArithmetic
from neutryx.products.vanilla import European


def black_scholes_call(S0, K, T, r, q, sigma):
    sqrtT = jnp.sqrt(T)
    d1 = (jnp.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


def test_sobol_normal_reproducible():
    gen = SobolGenerator(seed=123, scramble=True)
    draws = gen.normal(32, dim=8)
    draws_again = gen.normal(32, dim=8)
    assert draws.shape == (32, 8)
    assert jnp.allclose(draws, draws_again)
    means = draws.mean(axis=0)
    assert jnp.allclose(means, jnp.zeros(8), atol=1e-6)


def test_quasi_monte_carlo_pricing_beats_mc():
    option = European(K=100.0, T=1.0, is_call=True)
    params = dict(S0=100.0, r=0.05, q=0.0, sigma=0.2)
    analytic = float(black_scholes_call(params["S0"], option.K, option.T, params["r"], params["q"], params["sigma"]))

    cfg = MCConfig(steps=32, paths=4096)
    mc_price = price_vanilla_mc(
        jax.random.PRNGKey(0),
        S0=params["S0"],
        K=option.K,
        T=option.T,
        r=params["r"],
        q=params["q"],
        sigma=params["sigma"],
        cfg=cfg,
        is_call=option.is_call,
    )
    qmc_price = price_european_qmc(option, paths=4096, steps=32, generator=SobolGenerator(seed=0), **params)

    mc_error = abs(float(mc_price) - analytic)
    qmc_error = abs(float(qmc_price) - analytic)
    assert qmc_error < mc_error


def test_mlmc_reduces_bias_and_variance():
    option = AsianArithmetic(K=95.0, T=1.0, is_call=True)
    params = dict(S0=100.0, r=0.03, q=0.0, sigma=0.2)

    def discounted_payoff(paths):
        payoffs = jax.vmap(option.payoff)(paths)
        return jnp.exp(-params["r"] * option.T) * payoffs

    levels = [
        MLMCLevel(steps=32, paths=2048),
        MLMCLevel(steps=64, paths=1024),
        MLMCLevel(steps=128, paths=512),
        MLMCLevel(steps=256, paths=256),
    ]

    orchestrator = MLMCOrchestrator(levels, generator=SobolGenerator(seed=5))
    result = orchestrator.run(
        discounted_payoff,
        S0=params["S0"],
        mu=params["r"] - params["q"],
        sigma=params["sigma"],
        T=option.T,
    )

    reference_gen = SobolGenerator(seed=42)
    ref_steps = 512
    ref_paths = 16384
    ref_normals = reference_gen.normal(ref_paths, ref_steps, center=False)
    ref_cfg = MCConfig(steps=ref_steps, paths=ref_paths)
    ref_paths_sim = simulate_gbm(
        jax.random.PRNGKey(0),
        params["S0"],
        params["r"] - params["q"],
        params["sigma"],
        option.T,
        ref_cfg,
        normal_draws=ref_normals,
    )
    ref_value = float(jnp.mean(discounted_payoff(ref_paths_sim)))

    assert abs(float(result.price) - ref_value) < 0.02
    assert result.total_paths == sum(level.paths for level in levels)
    assert result.level_variances[-1] < result.level_variances[0]
