import jax

from neutryx.core.engine import MCConfig, simulate_gbm
from neutryx.valuations.exposure import epe


def test_epe_nonnegative():
    key = jax.random.PRNGKey(0)
    cfg = MCConfig(steps=8, paths=1000)
    paths = simulate_gbm(key, 100.0, 0.01, 0.2, 1.0, cfg)
    val = float(epe(paths, K=100.0, is_call=True))
    assert val >= 0.0
