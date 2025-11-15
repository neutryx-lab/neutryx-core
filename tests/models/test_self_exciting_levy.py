"""Unit tests for the self-exciting Lévy simulation utilities."""

import jax
import jax.numpy as jnp
import pytest

from neutryx.core.engine import MCConfig
from neutryx.models.equity_models import TimeChangedLevyParams, simulate_time_changed_levy
from neutryx.models.jump_clustering import SelfExcitingLevyParams, simulate_self_exciting_levy


@pytest.mark.parametrize(
    "levy_type, base_params",
    [
        ("VG", {"theta": -0.14, "sigma": 0.2, "nu": 0.2}),
        ("NIG", {"alpha": 15.0, "beta": -5.0, "delta": 0.5, "mu": 0.0}),
        ("CGMY", {"C": 1.0, "G": 10.0, "M": 10.0, "Y": 0.5}),
    ],
)
@pytest.mark.unit
def test_self_exciting_reduces_to_base_process(levy_type: str, base_params: dict) -> None:
    """With zero self-excitation the model should match the base Lévy dynamics."""

    key = jax.random.PRNGKey(0)
    cfg = MCConfig(steps=64, paths=1500, dtype=jnp.float32)

    S0, T, r, q = 100.0, 1.0, 0.05, 0.0
    params = SelfExcitingLevyParams(
        base_levy_params=base_params,
        levy_type=levy_type,
        lambda0=1.0,
        alpha=0.0,
        beta=2.0,
    )

    paths = simulate_self_exciting_levy(key, S0, T, r, q, params, cfg)
    tc_params = TimeChangedLevyParams(levy_process=levy_type, levy_params=base_params)
    base_paths = simulate_time_changed_levy(jax.random.PRNGKey(1), S0, T, r, q, tc_params, cfg)

    assert paths.shape == (cfg.paths, cfg.steps + 1)

    terminal_prices = jnp.asarray(paths[:, -1])
    expected_terminal = S0 * jnp.exp((r - q) * T)
    assert float(terminal_prices.mean()) == pytest.approx(float(expected_terminal), rel=0.05)

    log_returns_self = jnp.log(paths[:, -1] / S0)
    log_returns_base = jnp.log(base_paths[:, -1] / S0)

    assert float(log_returns_self.mean()) == pytest.approx(float(log_returns_base.mean()), abs=0.05)
    assert float(log_returns_self.var()) == pytest.approx(float(log_returns_base.var()), rel=0.3)


@pytest.mark.unit
def test_self_excitation_increases_jump_volatility() -> None:
    """Positive self-excitation should amplify jump variability relative to the base model."""

    key = jax.random.PRNGKey(42)
    cfg = MCConfig(steps=64, paths=1800, dtype=jnp.float32)

    S0, T, r, q = 100.0, 1.0, 0.05, 0.0
    base_params = {"theta": -0.14, "sigma": 0.2, "nu": 0.2, "jump_threshold": 0.015}

    tc_params = TimeChangedLevyParams(levy_process="VG", levy_params=base_params)
    base_paths = simulate_time_changed_levy(jax.random.PRNGKey(4), S0, T, r, q, tc_params, cfg)
    base_log_returns = jnp.log(base_paths[:, -1] / S0)

    params = SelfExcitingLevyParams(
        base_levy_params=base_params,
        levy_type="VG",
        lambda0=1.0,
        alpha=1.2,
        beta=2.0,
    )
    clustered_paths = simulate_self_exciting_levy(key, S0, T, r, q, params, cfg)
    clustered_log_returns = jnp.log(clustered_paths[:, -1] / S0)

    assert float(clustered_log_returns.var()) > float(base_log_returns.var())
