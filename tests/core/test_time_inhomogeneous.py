"""Tests for time-inhomogeneous coefficient support in Monte Carlo engine."""
import jax
import jax.numpy as jnp

from neutryx.core.engine import MCConfig, price_vanilla_mc, simulate_gbm, time_grid


def test_simulate_gbm_time_dependent_coefficients():
    """Deterministic schedule with zero volatility should match analytic path."""

    cfg = MCConfig(steps=4, paths=1, dtype="float32")
    T = 1.0
    S0 = 100.0

    def mu_schedule(t):
        return 0.1 + 0.05 * t

    sigma = 0.0
    key = jax.random.PRNGKey(0)
    paths = simulate_gbm(key, S0, mu_schedule, sigma, T, cfg, return_full=True)

    timeline = time_grid(T, cfg.steps, dtype="float32")
    midpoints = 0.5 * (timeline[:-1] + timeline[1:])
    expected_ST = S0 * jnp.exp(jnp.sum(mu_schedule(midpoints) * (T / cfg.steps)))

    assert jnp.allclose(paths.values[0, -1], expected_ST, rtol=1e-6, atol=1e-6)


def test_price_vanilla_mc_time_dependent_rates():
    """Pricing with time-dependent rates should respect deterministic evolution."""

    cfg = MCConfig(steps=5, paths=32, dtype="float32")
    T = 1.0
    S0, K = 100.0, 90.0

    def r_schedule(t):
        return 0.02 + 0.01 * t

    def q_schedule(t):
        return 0.0 * t

    sigma = 0.0
    key = jax.random.PRNGKey(123)

    price = price_vanilla_mc(key, S0, K, T, r_schedule, q_schedule, sigma, cfg)

    timeline = time_grid(T, cfg.steps, dtype="float32")
    midpoints = 0.5 * (timeline[:-1] + timeline[1:])
    drift = r_schedule(midpoints) - q_schedule(midpoints)
    ST = S0 * jnp.exp(jnp.sum(drift * (T / cfg.steps)))
    discount = jnp.exp(-jnp.sum(r_schedule(midpoints) * (T / cfg.steps)))
    expected_price = discount * max(ST - K, 0.0)

    assert jnp.allclose(price, expected_price, rtol=1e-6, atol=1e-6)
