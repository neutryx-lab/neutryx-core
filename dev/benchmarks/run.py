"""Command line interface for benchmarking Neutryx workloads."""
from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import jax  # noqa: E402

if not hasattr(jax.random, "KeyArray"):  # pragma: no cover - compatibility shim
    jax.random.KeyArray = jax.Array  # type: ignore[attr-defined]

if __package__ is None or __package__ == "":  # pragma: no cover - CLI execution path
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    sys.path.append(str(repo_root / "src"))

    from dev.benchmarks.hardware import HardwareTarget, parse_targets
    from dev.benchmarks.monte_carlo import (
        MonteCarloBenchmarkConfig,
        run_monte_carlo_benchmark,
    )
    from dev.benchmarks.pde import PDEBenchmarkConfig, run_pde_benchmark
    from dev.benchmarks.reporting import (
        BenchmarkResult,
        save_results,
        summarise_results,
    )
else:  # pragma: no cover - package import path
    from .hardware import HardwareTarget, parse_targets
    from .monte_carlo import MonteCarloBenchmarkConfig, run_monte_carlo_benchmark
    from .pde import PDEBenchmarkConfig, run_pde_benchmark
    from .reporting import BenchmarkResult, save_results, summarise_results

WORKLOAD_RUNNERS: Dict[str, Callable[[dict, HardwareTarget], dict]] = {
    "pde": lambda cfg, target: run_pde_benchmark(PDEBenchmarkConfig(**cfg), target),
    "monte-carlo": lambda cfg, target: run_monte_carlo_benchmark(
        MonteCarloBenchmarkConfig(**cfg), target
    ),
}

DEFAULT_WORKLOADS = tuple(WORKLOAD_RUNNERS.keys())


def parse_kv_pairs(pairs: Iterable[str]) -> dict:
    parsed: Dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid parameter '{pair}'. Expected key=value format.")
        key, value = pair.split("=", maxsplit=1)
        parsed[key] = eval_literal(value)
    return parsed


def eval_literal(value: str) -> object:
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=list(DEFAULT_WORKLOADS),
        choices=list(WORKLOAD_RUNNERS.keys()),
        help="Workloads to benchmark (default: all).",
    )
    parser.add_argument(
        "--device",
        dest="devices",
        action="append",
        default=[],
        help=(
            "Hardware targets in name[:backend] format. Repeat to run benchmarks across "
            "multiple devices. Defaults to the active JAX backend."
        ),
    )
    parser.add_argument(
        "--pde-param",
        dest="pde_params",
        action="append",
        default=[],
        help="Override PDE configuration as key=value pairs.",
    )
    parser.add_argument(
        "--mc-param",
        dest="mc_params",
        action="append",
        default=[],
        help="Override Monte Carlo configuration as key=value pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store raw benchmark results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output (useful for scripted runs).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        pde_params = parse_kv_pairs(args.pde_params)
        mc_params = parse_kv_pairs(args.mc_params)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    targets = parse_targets(args.devices)
    workloads = args.workloads

    results: List[BenchmarkResult] = []

    for target in targets:
        for workload in workloads:
            runner = WORKLOAD_RUNNERS[workload]
            cfg = pde_params if workload == "pde" else mc_params
            measurement = runner(cfg, target)
            result = BenchmarkResult(
                workload=measurement["workload"],
                target=target.describe(),
                metrics=measurement["metrics"],
                config=measurement["config"],
                extras={k: v for k, v in measurement.items() if k not in {"workload", "metrics", "config"}},
            )
            results.append(result)

    if args.output:
        save_results(results, args.output)

    if not args.quiet:
        print(summarise_results(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
