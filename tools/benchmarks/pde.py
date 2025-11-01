"""PDE benchmarking workload using a heat equation integrator."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from .hardware import HardwareTarget


@dataclass
class PDEBenchmarkConfig:
    """Configuration for the heat equation workload."""

    grid_size: int = 2048
    time_steps: int = 2_000
    alpha: float = 0.25
    total_time: float = 1.0

    def validate(self) -> None:
        if self.grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if self.time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.total_time <= 0.0:
            raise ValueError("total_time must be positive")


@dataclass
class PDEBenchmarkMetrics:
    compile_time_s: float
    execution_time_s: float
    updates_per_second: float
    backend: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compile_time_s": self.compile_time_s,
            "execution_time_s": self.execution_time_s,
            "updates_per_second": self.updates_per_second,
            "backend": self.backend,
        }


def _heat_equation_kernel(config: PDEBenchmarkConfig) -> tuple[Callable[[jax.Array], jax.Array], float]:
    dx = 1.0 / (config.grid_size - 1)
    dt = config.total_time / config.time_steps
    coeff = min(config.alpha * dt / (dx * dx), 0.45)

    def step(u: jax.Array) -> jax.Array:
        inner = u[1:-1]
        return u.at[1:-1].set(inner + coeff * (u[2:] - 2.0 * inner + u[:-2]))

    def integrate(u0: jax.Array) -> jax.Array:
        def body(_, state):
            return step(state)

        return jax.lax.fori_loop(0, config.time_steps, body, u0)

    return integrate, coeff


def run_pde_benchmark(config: PDEBenchmarkConfig, target: HardwareTarget) -> Dict[str, Any]:
    """Execute the PDE workload and return metrics."""

    config.validate()
    backend = target.resolve_backend()

    x = jnp.linspace(0.0, 1.0, config.grid_size, dtype=jnp.float32)
    u0 = jnp.sin(jnp.pi * x)
    kernel, coeff = _heat_equation_kernel(config)
    compiled = jax.jit(kernel, backend=backend)

    start = time.perf_counter()
    warm = compiled(u0)
    warm.block_until_ready()
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    result = compiled(u0)
    result.block_until_ready()
    execution_time = time.perf_counter() - start

    updates = (config.grid_size - 2) * config.time_steps
    updates_per_second = updates / execution_time if execution_time > 0 else float("inf")

    metrics = PDEBenchmarkMetrics(
        compile_time_s=compile_time,
        execution_time_s=execution_time,
        updates_per_second=updates_per_second,
        backend=backend,
    )

    return {
        "workload": "pde_heat_equation",
        "config": asdict(config),
        "metrics": {**metrics.to_dict(), "stability_coeff": coeff},
        "result_snapshot": float(result.mean()),
    }
