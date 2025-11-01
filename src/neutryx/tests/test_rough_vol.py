"""Tests for rough volatility Monte Carlo implementation."""
import jax
import jax.numpy as jnp

from neutryx.core.engine import MCConfig
from neutryx.models import (
    RoughBergomiParams,
    calibrate_forward_variance,
    price_european_call_mc,
    simulate_rough_bergomi,
)
from neutryx.models.bs import price as bs_price


def test_simulate_rough_bergomi_reduces_to_bs():
    """With zero vol-of-vol the model should match Black-Scholes pricing."""

    key = jax.random.PRNGKey(0)
    params = RoughBergomiParams(hurst=0.1, eta=0.0, rho=0.0, forward_variance=0.04)
    cfg = MCConfig(steps=8, paths=8192, dtype=jnp.float32)
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0

    price_rb = price_european_call_mc(key, S0, K, T, r, q, cfg, params)
    price_bs = bs_price(S0, K, T, r, q, jnp.sqrt(0.04))

    assert jnp.isfinite(price_rb)
    assert jnp.abs(price_rb - price_bs) / price_bs < 0.05


def test_forward_variance_calibration_matches_curve():
    """Mean variance over paths should recover the imposed forward curve."""

    key = jax.random.PRNGKey(42)
    cfg = MCConfig(steps=12, paths=4096, dtype=jnp.float32)
    times = jnp.linspace(0.0, 1.0, cfg.steps + 1)
    xi_curve = 0.04 + 0.02 * times

    params = RoughBergomiParams(
        hurst=0.2,
        eta=0.0,
        rho=-0.4,
        forward_variance=lambda t: 0.04 + 0.02 * t,
    )

    paths = simulate_rough_bergomi(key, 100.0, 1.0, 0.01, 0.0, cfg, params, return_full=True)
    estimated_curve = calibrate_forward_variance(paths.times, paths.variance)

    error = jnp.max(jnp.abs(estimated_curve - xi_curve))
    assert error < 5e-3


def test_simulate_rough_bergomi_shapes():
    """Returned paths should have expected shapes and metadata."""

    key = jax.random.PRNGKey(7)
    cfg = MCConfig(steps=6, paths=128, dtype=jnp.float32)
    params = RoughBergomiParams(hurst=0.3, eta=1.0, rho=-0.5, forward_variance=0.05)

    paths = simulate_rough_bergomi(key, 90.0, 0.5, 0.0, 0.0, cfg, params, return_full=True, store_log=True)

    assert paths.values.shape == (cfg.paths, cfg.steps + 1)
    assert paths.variance.shape == (cfg.paths, cfg.steps + 1)
    assert paths.fbm.shape == (cfg.paths, cfg.steps + 1)
    assert paths.log_values is not None
    assert paths.metadata["model"] == "rough_bergomi"
