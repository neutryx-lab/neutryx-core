import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.append(path_str)

import jax
import jax.numpy as jnp
import pytest

if not hasattr(jax.random, "KeyArray"):
    try:
        from jax import Array as _JaxArray  # type: ignore
    except ImportError:
        _JaxArray = type(jax.random.PRNGKey(0))
    jax.random.KeyArray = _JaxArray  # type: ignore[attr-defined]

from neutryx.core.engine import MCConfig, simulate_gbm_resumable
from neutryx.models import CheckpointManager, ModelWorkflow


def test_model_workflow_checkpoint_resume(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "ckpts")
    workflow = ModelWorkflow(name="demo", total_steps=3, checkpoint_manager=manager)

    def step(step_idx: int, state):
        state = dict(state)
        state["value"] = state.get("value", 0) + 1
        trigger = state.get("trigger", True)
        if step_idx == 1 and trigger:
            state["trigger"] = False
            state["_interrupt"] = True
        return state

    intermediate = workflow.run(step)
    assert intermediate["value"] == 2

    # Resume run; the trigger flag persists as False so the second pass completes.
    final = workflow.run(step)
    assert final["value"] == 3
    assert not manager.meta_path.exists()


def test_simulate_gbm_resumable_matches_direct() -> None:
    key = jax.random.PRNGKey(0)
    cfg = MCConfig(steps=8, paths=8)
    resumable = simulate_gbm_resumable(key, 100.0, 0.05, 0.2, 1.0, cfg, chunk_size=4)
    repeat = simulate_gbm_resumable(key, 100.0, 0.05, 0.2, 1.0, cfg, chunk_size=4)
    assert jnp.allclose(resumable, repeat)


def test_simulate_gbm_resumable_resume(tmp_path: Path) -> None:
    key = jax.random.PRNGKey(42)
    cfg = MCConfig(steps=4, paths=8)
    manager = CheckpointManager(tmp_path / "resume")
    expected = simulate_gbm_resumable(key, 100.0, 0.05, 0.2, 1.0, cfg, chunk_size=4)

    with pytest.raises(RuntimeError):
        simulate_gbm_resumable(
            key,
            100.0,
            0.05,
            0.2,
            1.0,
            cfg,
            chunk_size=4,
            checkpoint_manager=manager,
            max_chunks=1,
        )

    # Resume should pick up the stored chunk and finish the computation.
    result = simulate_gbm_resumable(
        key,
        100.0,
        0.05,
        0.2,
        1.0,
        cfg,
        chunk_size=4,
        checkpoint_manager=manager,
    )
    assert jnp.allclose(result, expected)
    assert not any(manager.directory.glob("*.pkl"))
