"""Parameter transforms enforcing calibration constraints."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .base import ParameterTransform

Array = jnp.ndarray


def identity() -> ParameterTransform:
    return ParameterTransform(lambda x: x, lambda y: y)


def positive(eps: float = 1e-6) -> ParameterTransform:
    """Ensure strictly positive parameters via softplus."""

    def forward(x: Array) -> Array:
        return jax.nn.softplus(x) + eps

    def inverse(y: Array) -> Array:
        safe = jnp.maximum(y - eps, 1e-12)
        return jnp.log(jnp.expm1(safe))

    return ParameterTransform(forward, inverse)


def bounded(low: float, high: float) -> ParameterTransform:
    """Map unconstrained values to (low, high)."""

    width = high - low

    def forward(x: Array) -> Array:
        return low + width * jax.nn.sigmoid(x)

    def inverse(y: Array) -> Array:
        clipped = jnp.clip((y - low) / width, 1e-6, 1 - 1e-6)
        return jnp.log(clipped) - jnp.log1p(-clipped)

    return ParameterTransform(forward, inverse)


def symmetric(bound: float) -> ParameterTransform:
    """Symmetric bound mapping to (-bound, bound)."""

    def forward(x: Array) -> Array:
        return bound * jnp.tanh(x)

    def inverse(y: Array) -> Array:
        clipped = jnp.clip(y / bound, -0.999999, 0.999999)
        return jnp.arctanh(clipped)

    return ParameterTransform(forward, inverse)


def positive_with_upper(eps: float, upper: float) -> ParameterTransform:
    """Ensure (eps, upper) bounds using sigmoid."""

    def forward(x: Array) -> Array:
        return eps + (upper - eps) * jax.nn.sigmoid(x)

    def inverse(y: Array) -> Array:
        clipped = jnp.clip((y - eps) / (upper - eps), 1e-6, 1 - 1e-6)
        return jnp.log(clipped) - jnp.log1p(-clipped)

    return ParameterTransform(forward, inverse)
