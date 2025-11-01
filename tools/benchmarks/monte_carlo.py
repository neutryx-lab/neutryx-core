"""Monte Carlo benchmarking workload built on the core simulation engine."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from neutryx.core.engine import MCConfig, simulate_gbm

from .hardware import HardwareTarget


@dataclass
class MonteCarloBenchmarkConfig:
    """Configuration for the Monte Carlo workload."""

    paths: int = 262_144
    steps: int = 256
    maturity: float = 1.0
    mu: float = 0.05
    sigma: float = 0.2
    spot: float = 100.0
    dtype: jnp.dtype = jnp.float32
    antithetic: bool = True
    seed: int = 0

    def validate(self) -> None:
        if self.paths <= 0:
            raise ValueError("paths must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


@dataclass
class MonteCarloBenchmarkMetrics:
    compile_time_s: float
    execution_time_s: float
    paths_per_second: float
    path_time_product: float
    backend: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compile_time_s": self.compile_time_s,
            "execution_time_s": self.execution_time_s,
            "paths_per_second": self.paths_per_second,
            "path_time_product": self.path_time_product,
            "backend": self.backend,
        }


def run_monte_carlo_benchmark(
    config: MonteCarloBenchmarkConfig, target: HardwareTarget
) -> Dict[str, Any]:
    """Execute the Monte Carlo workload and return metrics."""

    config.validate()
    backend = target.resolve_backend()

    mc_cfg = MCConfig(
        steps=config.steps,
        paths=config.paths,
        dtype=config.dtype,
        antithetic=config.antithetic,
    )

    def kernel(key: jax.Array) -> jax.Array:
        return simulate_gbm(
            key,
            config.spot,
            config.mu,
            config.sigma,
            config.maturity,
            mc_cfg,
            return_full=False,
        )

    compiled = jax.jit(kernel, backend=backend)
    key = jax.random.PRNGKey(config.seed)

    start = time.perf_counter()
    warm = compiled(key)
    warm.block_until_ready()
    compile_time = time.perf_counter() - start

    # refresh key to avoid identical randomness in timed run
    key = jax.random.split(key, 1)[0]

    start = time.perf_counter()
    paths = compiled(key)
    paths.block_until_ready()
    execution_time = time.perf_counter() - start

    total_path_steps = config.paths * config.steps
    paths_per_second = config.paths / execution_time if execution_time > 0 else float("inf")
    throughput = total_path_steps / execution_time if execution_time > 0 else float("inf")

    metrics = MonteCarloBenchmarkMetrics(
        compile_time_s=compile_time,
        execution_time_s=execution_time,
        paths_per_second=paths_per_second,
        path_time_product=throughput,
        backend=backend,
    )

    return {
        "workload": "monte_carlo_gbm",
        "config": asdict(config),
        "metrics": metrics.to_dict(),
        "result_snapshot": float(paths.mean()),
    }
