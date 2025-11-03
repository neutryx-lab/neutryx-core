from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.profiling import KernelProfiler


def test_kernel_profiler_collects_events(tmp_path: Path) -> None:
    logdir = tmp_path / "trace"
    with KernelProfiler(logdir) as profiler:
        x = jnp.ones((16, 16))
        y = x @ x
        _ = y.block_until_ready()

    events = profiler.events
    assert isinstance(events, list)
    assert all(evt.duration_ns > 0 for evt in events)

    df = profiler.to_pandas()
    assert not df.empty
    assert {"name", "duration_ns", "duration_ms"}.issubset(df.columns)

    summary = profiler.summary(top_k=5)
    assert not summary.empty
    assert "total_duration_ns" in summary.columns
