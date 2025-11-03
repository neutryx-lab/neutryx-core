"""Utilities for comparing native and Python implementations."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, List, Mapping, MutableMapping, Sequence

CallableLike = Callable[..., object]

__all__ = [
    "BenchmarkResult",
    "time_callable",
    "compare",
    "summarise",
]


@dataclass
class BenchmarkResult:
    """Container for a single benchmark measurement."""

    label: str
    durations: Sequence[float]
    iterations: int

    @property
    def average(self) -> float:
        return statistics.mean(self.durations)

    @property
    def median(self) -> float:
        return statistics.median(self.durations)

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.durations)

    @property
    def throughput(self) -> float:
        return self.iterations / self.average if self.average else float("inf")

    def as_dict(self) -> Mapping[str, float | str]:
        return {
            "label": self.label,
            "avg_ms": self.average * 1_000,
            "median_ms": self.median * 1_000,
            "stdev_ms": self.stdev * 1_000,
            "iterations": float(self.iterations),
        }


def time_callable(func: CallableLike, *, iterations: int = 1, warmup: int = 0, **kwargs) -> BenchmarkResult:
    """Time ``func`` executed ``iterations`` times."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    durations: List[float] = []

    for _ in range(warmup):
        func(**kwargs)

    for _ in range(iterations):
        start = time.perf_counter()
        func(**kwargs)
        durations.append(time.perf_counter() - start)

    return BenchmarkResult(label=getattr(func, "__name__", "callable"), durations=durations, iterations=iterations)


def compare(callables: Mapping[str, CallableLike], *, iterations: int = 5, warmup: int = 1, **kwargs) -> Sequence[BenchmarkResult]:
    """Benchmark multiple callables using a shared configuration."""

    results: List[BenchmarkResult] = []
    for label, func in callables.items():
        result = time_callable(func, iterations=iterations, warmup=warmup, **kwargs)
        results.append(replace(result, label=label))
    return results


def summarise(results: Iterable[BenchmarkResult]) -> MutableMapping[str, float | str]:
    """Generate a compact summary for reporting in CI logs."""

    ordered = list(results)
    if not ordered:
        return {}

    baseline = ordered[0]
    summary: MutableMapping[str, float | str] = {}
    summary[f"{baseline.label}_avg_ms"] = baseline.average * 1_000
    for result in ordered[1:]:
        speedup = baseline.average / result.average if result.average else float("inf")
        summary[f"{result.label}_avg_ms"] = result.average * 1_000
        summary[f"{result.label}_speedup_vs_{baseline.label}"] = speedup
    return summary
