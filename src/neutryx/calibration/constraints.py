"""Parameter transforms enforcing calibration constraints."""
from __future__ import annotations

from typing import Mapping, MutableMapping, Optional

import jax
import jax.numpy as jnp

from .base import ParameterSpec, ParameterTransform

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


def g2pp_parameter_specs(
    initial: Optional[Mapping[str, float]] = None,
) -> MutableMapping[str, ParameterSpec]:
    """Default parameter specification for G2++ calibration routines."""

    defaults = {
        "a": 0.1,
        "b": 0.3,
        "sigma_x": 0.01,
        "sigma_y": 0.015,
        "rho": -0.5,
    }
    transforms = {
        "a": positive(),
        "b": positive(),
        "sigma_x": positive(),
        "sigma_y": positive(),
        "rho": symmetric(0.999),
    }
    if initial is not None:
        defaults.update(initial)
    return {
        name: ParameterSpec(defaults[name], transforms[name]) for name in defaults
    }


def quasi_gaussian_parameter_specs(
    initial: Optional[Mapping[str, float]] = None,
) -> MutableMapping[str, ParameterSpec]:
    """Parameter specification for Quasi-Gaussian calibration."""

    defaults = {
        "alpha": 0.1,
        "beta": 0.25,
        "sigma_x": 0.01,
        "sigma_y": 0.012,
        "rho": -0.4,
    }
    transforms = {
        "alpha": positive(),
        "beta": positive(),
        "sigma_x": positive(),
        "sigma_y": positive(),
        "rho": symmetric(0.999),
    }
    if initial is not None:
        defaults.update(initial)
    return {
        name: ParameterSpec(defaults[name], transforms[name]) for name in defaults
    }
