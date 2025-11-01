"""Mathematical helper routines used across Neutryx."""

from __future__ import annotations

from typing import Any, Iterable

import jax.numpy as jnp

from .precision import canonicalize_dtype, get_compute_dtype

SQRT2PI = jnp.sqrt(2.0 * jnp.pi)


def normal_pdf(x: Any) -> jnp.ndarray:
    """Standard normal probability density function."""

    dtype = get_compute_dtype()
    x_arr = jnp.asarray(x, dtype=dtype)
    return jnp.exp(-0.5 * x_arr * x_arr) / jnp.asarray(SQRT2PI, dtype=dtype)


def logsumexp(
    values: Any,
    *,
    axis: int | Iterable[int] | None = None,
    keepdims: bool = False,
    dtype: Any | None = None,
) -> jnp.ndarray:
    """Numerically stable log-sum-exp implementation."""

    comp_dtype = canonicalize_dtype(dtype)
    arr = jnp.asarray(values, dtype=comp_dtype)

    if axis is None:
        max_val = jnp.max(arr)
        shifted = jnp.exp(arr - max_val)
        result = jnp.log(jnp.sum(shifted)) + max_val
        if keepdims:
            return jnp.broadcast_to(result, arr.shape)
        return result

    max_val = jnp.max(arr, axis=axis, keepdims=True)
    shifted = jnp.exp(arr - max_val)
    sum_shifted = jnp.sum(shifted, axis=axis, keepdims=True)
    lse = jnp.log(sum_shifted) + max_val
    if not keepdims:
        lse = jnp.squeeze(lse, axis=axis)
    return lse
