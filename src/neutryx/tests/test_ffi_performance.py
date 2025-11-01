from __future__ import annotations

import math

from neutryx.ffi.performance import BenchmarkResult, compare, summarise, time_callable


def _slow_square(value: float) -> float:
    return math.sqrt(value) ** 2


def _fast_square(value: float) -> float:
    return value * value


def test_time_callable_basic() -> None:
    result = time_callable(_fast_square, iterations=3, warmup=1, value=2.0)
    assert isinstance(result, BenchmarkResult)
    assert result.iterations == 3
    assert len(result.durations) == 3
    assert result.average >= 0.0


def test_compare_and_summarise() -> None:
    results = compare({"slow": _slow_square, "fast": _fast_square}, iterations=2, warmup=0, value=5.0)
    assert len(results) == 2
    summary = summarise(results)
    assert "slow_avg_ms" in summary
    assert "fast_avg_ms" in summary
    assert summary["slow_avg_ms"] >= 0.0
