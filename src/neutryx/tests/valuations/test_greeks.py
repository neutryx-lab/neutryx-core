import jax
import jax.numpy as jnp

from neutryx.core.engine import MCConfig, price_vanilla_mc
from neutryx.valuations.greeks.greeks import mc_delta_bump


def test_mc_delta_bump_runs():
    key = jax.random.PRNGKey(0)
    cfg = MCConfig(steps=8, paths=256)

    delta = mc_delta_bump(
        price_vanilla_mc,
        S=100.0,
        bump=1e-4,
        key=key,
        K=100.0,
        T=1.0,
        r=0.01,
        q=0.0,
        sigma=0.2,
        cfg=cfg,
        is_call=True,
    )

    assert jnp.isfinite(delta).item()
