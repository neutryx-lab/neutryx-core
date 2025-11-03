"""Core Monte Carlo simulation engine.

This module provides the foundational Monte Carlo simulation infrastructure including:
- GBM (Geometric Brownian Motion) path simulation
- Time grid generation
- Discounting utilities
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, Sequence

import jax
import jax.numpy as jnp
from jax import device_get

if TYPE_CHECKING:
    from neutryx.models.workflows import CheckpointManager

from neutryx.core.utils.parallel import ParallelConfig, ParallelExecutable, compile_parallel
from neutryx.core.utils.precision import canonicalize_dtype, get_compute_dtype

Array = jnp.ndarray

Schedule = float | Sequence[float] | Array | Callable[[Array], Array | float]


__all__ = [
    "Array",
    "MCConfig",
    "MCPaths",
    "discount_factor",
    "mc_expectation",
    "present_value",
    "price_vanilla_jump_diffusion_mc",
    "price_vanilla_mc",
    "resolve_schedule",
    "simulate_gbm",
    "simulate_gbm_resumable",
    "simulate_jump_diffusion",
    "time_grid",
]


class PayoffFn(Protocol):
    """Callable signature for path-wise payoffs."""

    def __call__(self, path: Array) -> Array:
        ...


@dataclass
class MCConfig:
    """Configuration for Monte Carlo simulations."""

    steps: int
    paths: int
    dtype: Any = None
    antithetic: bool = False

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("MCConfig.steps must be > 0.")
        if self.paths <= 0:
            raise ValueError("MCConfig.paths must be > 0.")
        if self.antithetic and self.paths % 2 != 0:
            raise ValueError("Antithetic sampling requires an even number of paths.")
        self.dtype = canonicalize_dtype(self.dtype)

    @property
    def base_paths(self) -> int:
        """Number of independent paths generated before optional antithetic pairing."""
        return self.paths // 2 if self.antithetic else self.paths


@dataclass
class MCPaths:
    """Container for Monte Carlo paths and associated metadata."""

    values: Array
    times: Array
    log_values: Optional[Array] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def terminal(self) -> Array:
        """Return terminal values for each path."""
        return self.values[:, -1]

    def mean_path(self) -> Array:
        """Return the average path across all simulations."""
        return self.values.mean(axis=0)


def time_grid(T: float, steps: int, *, dtype: Any = None) -> Array:
    """Generate an equally spaced time grid from 0 to T inclusive."""
    if steps <= 0:
        raise ValueError("steps must be > 0.")
    comp_dtype = canonicalize_dtype(dtype)
    return jnp.linspace(0.0, T, steps + 1, dtype=comp_dtype)


def resolve_schedule(value: Schedule, eval_times: Array, *, dtype: jnp.dtype) -> Array:
    """Evaluate (possibly time-dependent) coefficients on a time grid.

    Parameters
    ----------
    value
        Scalar, array-like or callable schedule describing the coefficient.
        Callables must accept an array of evaluation times and return an
        array broadcastable to the same shape.
    eval_times
        Points in time where the schedule should be evaluated.
    dtype
        Numerical dtype for the resulting array.

    Returns
    -------
    Array
        Array of shape ``eval_times.shape`` containing the coefficient values.
    """

    if callable(value):
        evaluated = value(eval_times)
    else:
        evaluated = value

    arr = jnp.asarray(evaluated, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full(eval_times.shape, arr.astype(dtype), dtype=dtype)
    if arr.shape != eval_times.shape:
        raise ValueError(
            "Schedule evaluation must produce an array matching the "
            f"evaluation grid shape. Expected {eval_times.shape}, got {arr.shape}."
        )
    return arr.astype(dtype)


def _prepare_increments(
    mu: Schedule, sigma: Schedule, T: float, cfg: MCConfig, *, timeline: Array
) -> tuple[Array, Array, Array, Array]:
    dt = T / cfg.steps
    midpoints = 0.5 * (timeline[:-1] + timeline[1:])
    mu_values = resolve_schedule(mu, midpoints, dtype=cfg.dtype)
    sigma_values = resolve_schedule(sigma, midpoints, dtype=cfg.dtype)
    drift = (mu_values - 0.5 * sigma_values * sigma_values) * dt
    vol = sigma_values * jnp.sqrt(dt)
    return drift, vol, mu_values, sigma_values


@lru_cache(maxsize=None)
def _get_gbm_executor(
    steps: int,
    base_paths: int,
    dtype_name: str,
    antithetic: bool,
) -> ParallelExecutable:
    dtype = jnp.dtype(dtype_name)
    config = ParallelConfig(axis_name="paths")

    def _impl(
        key: jax.random.KeyArray,
        drift: jax.Array,
        vol: jax.Array,
        log_S0: jax.Array,
    ) -> jax.Array:
        base_keys = jax.random.split(key, base_paths)
        normals = jax.vmap(lambda k: jax.random.normal(k, (steps,), dtype=dtype))(base_keys)
        increments = drift + vol * normals
        if antithetic:
            increments = jnp.concatenate([increments, drift + vol * (-normals)], axis=0)
        total_paths = increments.shape[0]
        log_prefix = jnp.full((total_paths, 1), log_S0, dtype=dtype)
        cum_returns = jnp.cumsum(increments, axis=1)
        return jnp.concatenate([log_prefix, log_S0 + cum_returns], axis=1)

    example_args = (
        jax.random.PRNGKey(0),
        jnp.asarray(0.0, dtype),
        jnp.asarray(0.0, dtype),
        jnp.asarray(0.0, dtype),
    )
    return compile_parallel(_impl, config, example_args=example_args)


def simulate_gbm(
    key: jax.random.KeyArray,
    S0: float,
    mu: Schedule,
    sigma: Schedule,
    T: float,
    cfg: MCConfig,
    *,
    normal_draws: Array | None = None,
    return_full: bool = False,
    store_log: bool = False,
) -> Array | MCPaths:
    """Simulate Geometric Brownian Motion paths using Euler discretisation.

    Parameters
    ----------
    key
        PRNG key used for sampling.
    S0
        Initial asset level.
    mu
        Drift of the process (already net of dividend yield).
    sigma
        Volatility of the process.
    T
        Maturity / horizon.
    cfg
        Monte Carlo configuration.
    normal_draws
        Optional standard normal draws of shape ``(paths, steps)``. When
        provided, sampling from ``key`` is skipped and the supplied draws are
        used directly (antithetic sampling is disabled in this mode).
    return_full
        If True, return an ``MCPaths`` container with times/log paths metadata.
        Otherwise, only the simulated paths (``Array``) are returned.
    store_log
        Whether to retain log paths inside ``MCPaths`` when ``return_full`` is True.

    Returns
    -------
    Array or MCPaths
        Paths shaped ``[paths, steps + 1]`` or container with metadata.
    """
    S0 = jnp.asarray(S0, dtype=cfg.dtype)
    timeline = time_grid(float(T), cfg.steps, dtype=cfg.dtype)
    drift, vol, mu_values, sigma_values = _prepare_increments(
        mu, sigma, T, cfg, timeline=timeline
    )
    log_S0 = jnp.log(S0)

    executor = _get_gbm_executor(
        cfg.steps, cfg.base_paths, jnp.dtype(cfg.dtype).name, cfg.antithetic
    )
    log_paths = executor(key, drift, vol, log_S0)
    paths = jnp.exp(log_paths)
    if not return_full:
        return paths

    log_values = log_paths if store_log else None
    metadata = {
        "model": "gbm",
        "mu": mu_values,
        "sigma": sigma_values,
        "antithetic": cfg.antithetic,
        "paths": cfg.paths,
        "steps": cfg.steps,
    }
    return MCPaths(values=paths, times=timeline, log_values=log_values, metadata=metadata)


def simulate_jump_diffusion(
    key: jax.random.KeyArray,
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_jump: float,
    sigma_jump: float,
    T: float,
    cfg: MCConfig,
    *,
    return_full: bool = False,
    store_log: bool = False,
) -> Array | MCPaths:
    """Simulate Merton jump-diffusion paths via Euler discretisation."""

    dtype = cfg.dtype
    S0 = jnp.asarray(S0, dtype=dtype)
    mu = jnp.asarray(mu, dtype=dtype)
    sigma = jnp.asarray(sigma, dtype=dtype)
    lam = jnp.asarray(lam, dtype=dtype)
    mu_jump = jnp.asarray(mu_jump, dtype=dtype)
    sigma_jump = jnp.asarray(sigma_jump, dtype=dtype)

    dt = T / cfg.steps
    sqrt_dt = jnp.sqrt(dt)
    kappa = jnp.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1.0
    drift = (mu - 0.5 * sigma ** 2 - lam * kappa) * dt
    vol = sigma * sqrt_dt
    lt = lam * dt

    key_norm, key_pois, key_jump = jax.random.split(key, 3)
    normals = jax.random.normal(key_norm, (cfg.base_paths, cfg.steps), dtype=dtype)
    counts = jax.random.poisson(key_pois, lam=lt, shape=(cfg.base_paths, cfg.steps))
    counts = counts.astype(dtype)
    jump_normals = jax.random.normal(key_jump, (cfg.base_paths, cfg.steps), dtype=dtype)

    jump_mean = counts * mu_jump
    jump_std = sigma_jump * jnp.sqrt(jnp.maximum(counts, 0.0))
    increments = drift + vol * normals + jump_mean + jump_std * jump_normals

    if cfg.antithetic:
        anti_normals = -normals
        anti_jumps = jump_mean + jump_std * (-jump_normals)
        increments = jnp.concatenate([increments, drift + vol * anti_normals + anti_jumps], axis=0)

    log_S0 = jnp.log(S0)
    total_paths = increments.shape[0]
    cum_returns = jnp.cumsum(increments, axis=1)
    log_paths = jnp.concatenate(
        [jnp.full((total_paths, 1), log_S0, dtype=dtype), log_S0 + cum_returns],
        axis=1,
    )
    paths = jnp.exp(log_paths)
    timeline = time_grid(float(T), cfg.steps, dtype=dtype)

    if not return_full:
        return paths

    log_values = log_paths if store_log else None
    total_paths = paths.shape[0]
    metadata = {
        "model": "merton_jump_diffusion",
        "mu": float(mu),
        "sigma": float(sigma),
        "lam": float(lam),
        "mu_jump": float(mu_jump),
        "sigma_jump": float(sigma_jump),
        "antithetic": cfg.antithetic,
        "paths": total_paths,
        "steps": cfg.steps,
    }
    return MCPaths(values=paths, times=timeline, log_values=log_values, metadata=metadata)


def discount_factor(r: float, t: float | Array, *, dtype: Any = None) -> Array:
    """Return discount factor(s) e^{-r t}."""
    comp_dtype = canonicalize_dtype(dtype) if dtype is not None else None
    if comp_dtype is None:
        t_arr = jnp.asarray(t)
        comp_dtype = t_arr.dtype if hasattr(t_arr, "dtype") else get_compute_dtype()
    t_arr = jnp.asarray(t, dtype=comp_dtype)
    r_cast = jnp.asarray(r, dtype=comp_dtype)
    return jnp.exp(-r_cast * t_arr)


def present_value(payoffs: Array, times: Array | float, r: float, *, axis: Optional[int] = 0) -> Array:
    """Discount path-wise payoffs at their respective times and average."""
    payoffs_arr = jnp.asarray(payoffs)
    disc = jnp.asarray(discount_factor(r, times, dtype=payoffs_arr.dtype), dtype=payoffs_arr.dtype)
    discounted = payoffs_arr * disc
    return jnp.mean(discounted, axis=axis)


def mc_expectation(paths: Array | MCPaths, payoff_fn: PayoffFn) -> Array:
    """Compute Monte Carlo expectation of a payoff over provided paths."""
    values = paths.values if isinstance(paths, MCPaths) else paths
    payoffs = jax.vmap(payoff_fn)(values)
    return payoffs.mean(axis=0)


def price_vanilla_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: Schedule,
    q: Schedule,
    sigma: Schedule,
    cfg: MCConfig,
    *,
    is_call: bool = True,
) -> Array:
    """Monte Carlo price for a vanilla European option using GBM dynamics."""
    timeline = time_grid(float(T), cfg.steps, dtype=cfg.dtype)
    midpoints = 0.5 * (timeline[:-1] + timeline[1:])
    r_values = resolve_schedule(r, midpoints, dtype=cfg.dtype)
    q_values = resolve_schedule(q, midpoints, dtype=cfg.dtype)
    mu_values = r_values - q_values

    paths = simulate_gbm(key, S0, mu_values, sigma, T, cfg)
    ST = paths[:, -1]
    intrinsic = jnp.maximum(ST - K, 0.0) if is_call else jnp.maximum(K - ST, 0.0)
    return present_value(intrinsic, jnp.asarray(T, dtype=paths.dtype), r)


def simulate_gbm_resumable(
    key: jax.random.KeyArray,
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    cfg: MCConfig,
    *,
    chunk_size: int = 1024,
    checkpoint_manager: CheckpointManager | None = None,
    max_chunks: int | None = None,
    return_full: bool = False,
    store_log: bool = False,
) -> Array | MCPaths:
    """Resumable variant of :func:`simulate_gbm` with checkpointing support.

    The simulation is split across ``chunk_size`` batches. After each batch the
    intermediate state is optionally persisted to ``checkpoint_manager`` so that
    a subsequent invocation can resume from the most recent completed chunk.

    Parameters
    ----------
    chunk_size
        Maximum number of paths simulated per chunk.
    checkpoint_manager
        Optional manager responsible for persisting checkpoints. When omitted
        the simulation behaves like a plain in-memory chunked execution.
    max_chunks
        Optional limit used mainly for smoke tests: after processing
        ``max_chunks`` chunks the routine returns early and raises a
        ``RuntimeError`` signalling an interrupted run. Any completed chunks
        remain available for resumption.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if cfg.antithetic and chunk_size % 2 != 0:
        raise ValueError("chunk_size must be even when antithetic sampling is enabled")

    total_paths = cfg.paths
    total_chunks = math.ceil(total_paths / chunk_size)
    if total_chunks == 1 and checkpoint_manager is None and max_chunks is None:
        return simulate_gbm(
            key,
            S0,
            mu,
            sigma,
            T,
            cfg,
            return_full=return_full,
            store_log=store_log,
        )

    # Lazy import to avoid circular dependency
    from neutryx.models.workflows import ModelWorkflow

    persist_chunks = checkpoint_manager is not None
    chunk_keys = jax.random.split(key, total_chunks)
    workflow = ModelWorkflow(
        name="simulate_gbm",
        total_steps=total_chunks,
        checkpoint_manager=checkpoint_manager,
    )

    def _normalise_chunk(payload: Array | MCPaths) -> Dict[str, Any]:
        if isinstance(payload, MCPaths):
            result: Dict[str, Any] = {
                "values": device_get(payload.values),
                "times": device_get(payload.times),
                "metadata": dict(payload.metadata),
            }
            log_values = payload.log_values
            result["log_values"] = device_get(log_values) if log_values is not None else None
            return result
        return {"values": device_get(payload)}

    def _write_chunk(step: int, data: Dict[str, Any]) -> str:
        assert checkpoint_manager is not None  # for typing
        path = checkpoint_manager.chunk_path("simulate_gbm", step)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(data, fh)
        tmp.replace(path)
        return path.name

    def _load_chunk(filename: str) -> Dict[str, Any]:
        assert checkpoint_manager is not None
        path = checkpoint_manager.directory / filename
        with path.open("rb") as fh:
            return pickle.load(fh)

    def _assemble(chunks: Sequence[Dict[str, Any]]) -> Array | MCPaths:
        if not chunks:
            raise RuntimeError("No chunks available to assemble")
        if return_full:
            values = jnp.concatenate([jnp.asarray(chunk["values"]) for chunk in chunks], axis=0)
            times = jnp.asarray(chunks[0]["times"])
            log_values = chunks[0].get("log_values")
            if log_values is not None:
                log_concat = jnp.concatenate(
                    [jnp.asarray(chunk["log_values"]) for chunk in chunks], axis=0
                )
            else:
                log_concat = None
            metadata = dict(chunks[0]["metadata"])
            metadata.update({"paths": int(values.shape[0]), "chunks": len(chunks)})
            return MCPaths(values=values, times=times, log_values=log_concat, metadata=metadata)
        return jnp.concatenate([jnp.asarray(chunk["values"]) for chunk in chunks], axis=0)

    def _step(step: int, state: Dict[str, Any]) -> Dict[str, Any]:
        produced = int(state.get("produced", 0))
        if produced >= total_paths:
            state["_interrupt"] = True
            return state

        remaining = total_paths - produced
        current_chunk = min(chunk_size, remaining)
        chunk_cfg = replace(cfg, paths=current_chunk)
        chunk_key = chunk_keys[step]
        payload = simulate_gbm(
            chunk_key,
            S0,
            mu,
            sigma,
            T,
            chunk_cfg,
            return_full=return_full,
            store_log=store_log,
        )
        normalised = _normalise_chunk(payload)

        if persist_chunks:
            files = list(state.get("chunk_files", []))
            filename = _write_chunk(step, normalised)
            if len(files) == step:
                files.append(filename)
            elif len(files) > step:
                files[step] = filename
            else:
                files.extend([""] * (step - len(files)))
                files.append(filename)
            state["chunk_files"] = files
        else:
            chunks = list(state.get("chunks", []))
            if len(chunks) == step:
                chunks.append(normalised)
            elif len(chunks) > step:
                chunks[step] = normalised
            else:
                chunks.extend([{}] * (step - len(chunks)))
                chunks.append(normalised)
            state["chunks"] = chunks

        produced += current_chunk
        state["produced"] = produced
        processed = int(state.get("processed_chunks", 0)) + 1
        state["processed_chunks"] = processed

        if max_chunks is not None and processed >= max_chunks:
            state["_interrupt"] = True
        return state

    state = workflow.run(_step)

    produced = int(state.get("produced", 0))
    if produced < total_paths:
        raise RuntimeError(
            "simulate_gbm_resumable interrupted before completion."
        )

    if persist_chunks:
        chunk_files = state.get("chunk_files", [])
        chunks = [_load_chunk(filename) for filename in chunk_files]
        result = _assemble(chunks)
        checkpoint_manager.cleanup_chunks()
        checkpoint_manager.mark_complete()
        return result

    chunks = state.get("chunks", [])
    return _assemble(chunks)


def price_vanilla_jump_diffusion_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lam: float,
    mu_jump: float,
    sigma_jump: float,
    cfg: MCConfig,
    kind: str = "call",
) -> Array:
    """Price vanilla option under Merton jump-diffusion via Monte Carlo.
    
    Parameters
    ----------
    key : JAX PRNG key
    S0 : Initial stock price
    K : Strike price
    T : Time to maturity
    r : Risk-free rate
    q : Dividend yield
    sigma : Diffusion volatility
    lam : Jump intensity (jumps per year)
    mu_jump : Mean of log-jump size
    sigma_jump : Std dev of log-jump size
    cfg : Monte Carlo configuration
    kind : "call" or "put"
    
    Returns
    -------
    price : Option price
    """
    mu = r - q
    paths = simulate_jump_diffusion(
        key, S0, mu, sigma, lam, mu_jump, sigma_jump, T, cfg, return_full=False
    )
    ST = paths[:, -1]
    
    if kind == "call":
        payoffs = jnp.maximum(ST - K, 0.0)
    elif kind == "put":
        payoffs = jnp.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"Unknown option kind: {kind}")
    
    return present_value(payoffs, jnp.array(T), r)
