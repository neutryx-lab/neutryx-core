"""Monte-Carlo vs analytic benchmark used by the CI benchmark suite."""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp

from neutryx.core.engine import MCConfig, present_value, simulate_gbm
from neutryx.models.bs import price as bs_price
from tools.profiling import KernelProfiler


def run_benchmark(paths: int, profile_dir: str | None = None) -> KernelProfiler | None:
    key = jax.random.PRNGKey(0)
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.01, 0.0, 0.2
    cfg = MCConfig(steps=252, paths=paths)

    profiler: KernelProfiler | None = None
    context = (
        KernelProfiler(profile_dir, include_python_events=False)
        if profile_dir is not None
        else nullcontext()
    )

    with context as maybe_profiler:
        if isinstance(maybe_profiler, KernelProfiler):
            profiler = maybe_profiler
        start = time.time()
        paths_ = simulate_gbm(key, S, r - q, sigma, T, cfg)
        payoff = jnp.maximum(paths_[:, -1] - K, 0.0)
        pv = present_value(payoff, jnp.array(T), r)
        duration = time.time() - start
        analytic = bs_price(S, K, T, r, q, sigma)

    print(
        f"paths={paths:7d} price={float(pv):.4f} bs={float(analytic):.4f} time={duration:.3f}s"
    )
    return profiler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=os.environ.get("NEUTRYX_PROFILE_DIR"),
        help="Optional directory where kernel traces should be stored.",
    )
    parser.add_argument(
        "--paths",
        type=int,
        nargs="*",
        default=[10_000, 50_000, 200_000],
        help="Number of Monte-Carlo paths to benchmark (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile_dir: str | None = None
    if args.profile_dir:
        profile_dir = str(Path(args.profile_dir).expanduser())

    for paths in args.paths:
        profiler = run_benchmark(paths, profile_dir=profile_dir)
        if profiler is not None and profiler.events:
            try:
                summary = profiler.summary(top_k=10)
            except ImportError:
                summary = None
            if summary is not None:
                print("Top kernels (by cumulative duration):")
                print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
