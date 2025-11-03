"""Reporting helpers for benchmark runs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class BenchmarkResult:
    """Container for a completed benchmark measurement."""

    workload: str
    target: str
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    extras: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "workload": self.workload,
            "target": self.target,
            "metrics": self.metrics,
            "config": self.config,
        }
        payload.update(self.extras)
        return payload


def save_results(results: Iterable[BenchmarkResult], path: Path) -> None:
    """Persist benchmark results to a JSON file."""

    serialisable = [result.to_dict() for result in results]
    path.write_text(json.dumps(serialisable, indent=2, sort_keys=True))


def summarise_results(results: Iterable[BenchmarkResult]) -> str:
    """Create a simple aligned text table summarising the metrics."""

    rows: List[Dict[str, Any]] = []
    for result in results:
        metrics = result.metrics
        rows.append(
            {
                "workload": result.workload,
                "target": result.target,
                "exec_time": metrics.get("execution_time_s", float("nan")),
                "compile_time": metrics.get("compile_time_s", float("nan")),
                "throughput": metrics.get("updates_per_second")
                or metrics.get("paths_per_second")
                or metrics.get("path_time_product"),
            }
        )

    if not rows:
        return "No benchmark results to summarise."

    header = "{:<20} {:<15} {:>12} {:>14} {:>15}".format(
        "Workload", "Target", "Exec (s)", "Compile (s)", "Throughput"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            "{:<20} {:<15} {:>12.6f} {:>14.6f} {:>15.2f}".format(
                row["workload"],
                row["target"],
                row["exec_time"],
                row["compile_time"],
                row["throughput"] or float("nan"),
            )
        )
    return "\n".join(lines)
