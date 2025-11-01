from __future__ import annotations

from typing import Any, Iterable

import jax.numpy as jnp

from neutryx.core.utils.precision import canonicalize_dtype

__all__ = [
    "uniform",
    "log_uniform",
    "time_grid",
    "refine_grid",
    "merge_grids",
]


def uniform(n: int, a: float = 0.0, b: float = 1.0, dtype: Any = None) -> jnp.ndarray:
    """Return a uniform grid of ``n`` points between ``a`` and ``b`` inclusive."""
    if n < 2:
        raise ValueError("n must be >= 2 for a uniform grid.")
    comp_dtype = canonicalize_dtype(dtype)
    return jnp.linspace(a, b, n, dtype=comp_dtype)


def log_uniform(
    n: int,
    a: float = 1e-4,
    b: float = 1.0,
    dtype: Any = None,
) -> jnp.ndarray:
    """Return a log-uniform (geometric) grid between ``a`` and ``b`` (both > 0)."""
    if n < 2:
        raise ValueError("n must be >= 2 for a log-uniform grid.")
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive for a log-uniform grid.")
    comp_dtype = canonicalize_dtype(dtype)
    return jnp.geomspace(a, b, num=n, dtype=comp_dtype)


def time_grid(
    T: float,
    steps: int,
    *,
    start: float = 0.0,
    include_start: bool = True,
    dtype: Any = None,
) -> jnp.ndarray:
    """Create a time grid for a maturity ``T`` using ``steps`` intervals."""
    if steps <= 0:
        raise ValueError("steps must be > 0 for a time grid.")
    end = start + T
    comp_dtype = canonicalize_dtype(dtype)
    if include_start:
        return jnp.linspace(start, end, steps + 1, dtype=comp_dtype)
    return jnp.linspace(start + (T / steps), end, steps, dtype=comp_dtype)


def refine_grid(
    base: jnp.ndarray,
    extra_points: Iterable[float],
    *,
    unique: bool = True,
    dtype: Any = None,
) -> jnp.ndarray:
    """Merge a base grid with additional points and (optionally) uniquify/sort."""
    comp_dtype = canonicalize_dtype(dtype)
    base_arr = jnp.asarray(base, dtype=comp_dtype).ravel()
    extra_arr = jnp.asarray(extra_points, dtype=comp_dtype).ravel()
    combined = jnp.concatenate([base_arr, extra_arr])
    combined = jnp.sort(combined)
    if unique:
        combined = jnp.unique(combined)
    return combined


def merge_grids(*grids: Iterable[float], dtype: Any = None) -> jnp.ndarray:
    """Merge multiple grids into a single sorted unique array."""
    arrays = []
    comp_dtype = canonicalize_dtype(dtype)
    for grid in grids:
        arr = jnp.asarray(grid, dtype=comp_dtype).ravel()
        if arr.size == 0:
            continue
        arrays.append(arr)
    if not arrays:
        return jnp.asarray([], dtype=comp_dtype)
    stacked = jnp.concatenate(arrays)
    return jnp.unique(jnp.sort(stacked))
