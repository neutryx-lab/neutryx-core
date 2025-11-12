"""Autodiff-compatible loss functions for calibration."""
from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp

Array = jnp.ndarray


def mean_squared_error(
    predicted: Array,
    target: Array,
    *,
    weights: Optional[Array] = None,
    **_: Any,
) -> Array:
    diff = predicted - target
    if weights is not None:
        diff = jnp.sqrt(weights) * diff
    return jnp.mean(diff * diff)


def relative_mse(
    predicted: Array,
    target: Array,
    *,
    weights: Optional[Array] = None,
    eps: float = 1e-8,
    **_: Any,
) -> Array:
    scaled_target = jnp.where(jnp.abs(target) < eps, eps, target)
    diff = (predicted - target) / scaled_target
    if weights is not None:
        diff = jnp.sqrt(weights) * diff
    return jnp.mean(diff * diff)


def huber_loss(
    predicted: Array,
    target: Array,
    *,
    delta: float = 1e-1,
    weights: Optional[Array] = None,
    **_: Any,
) -> Array:
    diff = predicted - target
    abs_diff = jnp.abs(diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    if weights is not None:
        loss = weights * loss
    return jnp.mean(loss)


def g2pp_zero_curve_loss(
    predicted: Array,
    target: Array,
    *,
    weights: Optional[Array] = None,
    params: Optional[Any] = None,
    market_data: Optional[Any] = None,
    **_: Any,
) -> Array:
    """Loss emphasizing the long end of the term-structure for G2++ fits."""

    maturities = None
    if market_data is not None:
        maturities = market_data.get("maturities")
    if maturities is not None:
        maturity_weights = jnp.asarray(maturities)
        maturity_weights = maturity_weights / jnp.max(maturity_weights)
        weights = maturity_weights if weights is None else weights * maturity_weights
    return mean_squared_error(predicted, target, weights=weights)


def quasi_gaussian_zero_curve_loss(
    predicted: Array,
    target: Array,
    *,
    weights: Optional[Array] = None,
    params: Optional[Any] = None,
    market_data: Optional[Any] = None,
    **_: Any,
) -> Array:
    """Relative error style loss for Quasi-Gaussian discount curve fits."""

    if market_data is not None and market_data.get("use_relative", True):
        return relative_mse(predicted, target, weights=weights)
    return mean_squared_error(predicted, target, weights=weights)
