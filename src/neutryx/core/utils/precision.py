"""Precision management utilities for mixed-precision numerical kernels.

This module centralises support for selecting compute dtypes and controlling
loss scaling across the codebase.  Only ``bfloat16`` and ``float32`` are
currently supported to guarantee numerically stable operations on both CPU and
accelerator backends.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator

import jax
import jax.numpy as jnp

__all__ = [
    "PrecisionState",
    "SUPPORTED_DTYPES",
    "apply_loss_scaling",
    "canonicalize_dtype",
    "cast_like",
    "get_compute_dtype",
    "get_loss_scale",
    "get_precision_state",
    "precision_scope",
    "set_global_precision",
    "undo_loss_scaling",
]

SUPPORTED_DTYPES: Dict[str, jnp.dtype] = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}


@dataclass(frozen=True)
class PrecisionState:
    """Container describing the global precision configuration."""

    compute_dtype: jnp.dtype = jnp.float32
    loss_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.loss_scale <= 0.0:
            raise ValueError("loss_scale must be strictly positive")
        object.__setattr__(self, "compute_dtype", canonicalize_dtype(self.compute_dtype))


def canonicalize_dtype(dtype: Any | None) -> jnp.dtype:
    """Return a supported ``jax.numpy`` dtype.

    Parameters
    ----------
    dtype:
        ``None`` uses the currently configured global compute dtype.  ``str``
        inputs (case insensitive) and ``numpy``/``jax`` dtype objects are also
        accepted.  Only ``bfloat16`` and ``float32`` are supported.
    """

    if dtype is None:
        return _GLOBAL_STATE.compute_dtype

    if isinstance(dtype, str):
        key = dtype.lower()
    else:
        try:
            key = jnp.dtype(dtype).name
        except TypeError as exc:  # pragma: no cover - defensive fallback
            raise TypeError(f"Unsupported dtype specification: {dtype!r}") from exc

    if key not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Allowed values: {tuple(SUPPORTED_DTYPES)}"
        )
    return SUPPORTED_DTYPES[key]


def get_compute_dtype(dtype: Any | None = None) -> jnp.dtype:
    """Return the dtype to use for computations."""

    return canonicalize_dtype(dtype)


def get_loss_scale() -> float:
    """Return the globally configured loss scaling factor."""

    return _GLOBAL_STATE.loss_scale


_GLOBAL_STATE = PrecisionState()


def get_precision_state() -> PrecisionState:
    """Return a copy of the current global precision state."""

    return PrecisionState(
        compute_dtype=_GLOBAL_STATE.compute_dtype,
        loss_scale=_GLOBAL_STATE.loss_scale,
    )


def set_global_precision(*, compute_dtype: Any | None = None, loss_scale: float | None = None) -> PrecisionState:
    """Update the global precision configuration."""

    global _GLOBAL_STATE
    state = _GLOBAL_STATE
    if compute_dtype is not None:
        state = replace(state, compute_dtype=canonicalize_dtype(compute_dtype))
    if loss_scale is not None:
        if loss_scale <= 0.0:
            raise ValueError("loss_scale must be strictly positive")
        state = replace(state, loss_scale=float(loss_scale))
    _GLOBAL_STATE = state
    return state


@contextmanager
def precision_scope(*, compute_dtype: Any | None = None, loss_scale: float | None = None) -> Iterator[PrecisionState]:
    """Temporarily override the global precision configuration."""

    previous = get_precision_state()
    try:
        yield set_global_precision(compute_dtype=compute_dtype, loss_scale=loss_scale)
    finally:
        set_global_precision(
            compute_dtype=previous.compute_dtype, loss_scale=previous.loss_scale
        )


def apply_loss_scaling(value: Any, *, loss_scale: float | None = None) -> Any:
    """Multiply ``value`` (scalar or pytree) by the configured loss scale."""

    scale = float(loss_scale if loss_scale is not None else get_loss_scale())
    return jax.tree_util.tree_map(
        lambda v: None if v is None else v * scale,
        value,
    )


def undo_loss_scaling(value: Any, *, loss_scale: float | None = None) -> Any:
    """Divide ``value`` (scalar or pytree) by the configured loss scale."""

    scale = float(loss_scale if loss_scale is not None else get_loss_scale())
    return jax.tree_util.tree_map(
        lambda v: None if v is None else v / scale,
        value,
    )


def cast_like(x: Any, dtype: Any | None = None) -> jnp.ndarray:
    """Convert ``x`` to an array of the requested compute dtype."""

    return jnp.asarray(x, dtype=get_compute_dtype(dtype))
