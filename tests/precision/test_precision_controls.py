import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax import tree_util
from jax.scipy.special import logsumexp as jax_logsumexp

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from neutryx.core.engine import MCConfig, simulate_gbm, time_grid
from neutryx.core.utils import (
    SUPPORTED_DTYPES,
    apply_loss_scaling,
    canonicalize_dtype,
    get_precision_state,
    logsumexp,
    precision_scope,
    set_global_precision,
    undo_loss_scaling,
)


@pytest.fixture(autouse=True)
def _reset_precision_state():
    original = get_precision_state()
    yield
    set_global_precision(
        compute_dtype=original.compute_dtype,
        loss_scale=original.loss_scale,
    )


def test_canonicalize_dtype_supports_strings():
    assert canonicalize_dtype("float32") == jnp.float32
    assert canonicalize_dtype("bfloat16") == jnp.bfloat16
    with pytest.raises(ValueError):
        canonicalize_dtype("float16")


def test_precision_scope_updates_and_restores():
    baseline = get_precision_state()
    with precision_scope(compute_dtype="bfloat16", loss_scale=128.0) as scoped:
        assert scoped.compute_dtype == jnp.bfloat16
        assert math.isclose(scoped.loss_scale, 128.0)
        inner = get_precision_state()
        assert inner.compute_dtype == jnp.bfloat16
        assert math.isclose(inner.loss_scale, 128.0)
    restored = get_precision_state()
    assert restored.compute_dtype == baseline.compute_dtype
    assert math.isclose(restored.loss_scale, baseline.loss_scale)


def test_loss_scaling_round_trip():
    grads = {"a": jnp.ones((3,)), "b": None, "c": 2.5}
    scale = 256.0
    scaled = apply_loss_scaling(grads, loss_scale=scale)
    restored = undo_loss_scaling(scaled, loss_scale=scale)
    leaves_original = [leaf for leaf in tree_util.tree_leaves(grads) if leaf is not None]
    leaves_restored = [leaf for leaf in tree_util.tree_leaves(restored) if leaf is not None]
    for original_leaf, restored_leaf in zip(leaves_original, leaves_restored):
        assert jnp.allclose(restored_leaf, original_leaf)


def test_logsumexp_matches_reference():
    data = jnp.array([[1e2, -1e3, 3.0], [5.0, 7.0, -2.0]], dtype=jnp.float32)
    ours = logsumexp(data, axis=1)
    reference = jax_logsumexp(data, axis=1)
    assert jnp.allclose(ours, reference)


def test_mcconfig_string_dtype_and_simulation():
    cfg = MCConfig(steps=4, paths=4, dtype="bfloat16")
    assert cfg.dtype == jnp.bfloat16
    key = jax.random.PRNGKey(0)
    paths = simulate_gbm(key, 1.0, 0.01, 0.2, 1.0, cfg)
    assert paths.dtype == jnp.bfloat16
    timeline = time_grid(1.0, cfg.steps, dtype=cfg.dtype)
    assert timeline.dtype == jnp.bfloat16


def test_supported_dtypes_listing():
    assert set(SUPPORTED_DTYPES.keys()) == {"float32", "bfloat16"}
