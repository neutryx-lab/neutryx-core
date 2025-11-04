"""Portfolio credit risk analytics built on copula-based simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import jax.random as random

from .correlation import (
    SingleFactorGaussianCopula,
    gaussian_copula_samples,
)

Array = jnp.ndarray


def _as_array(values: Sequence[float] | Array) -> Array:
    arr = jnp.asarray(values)
    if arr.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    return arr


@dataclass
class PortfolioLossMetrics:
    expected_loss: Array
    unexpected_loss: Array
    value_at_risk: Array
    conditional_var: Array


def expected_loss(exposures: Sequence[float], default_probs: Sequence[float], lgd: Sequence[float]) -> Array:
    exp_arr = _as_array(exposures)
    prob_arr = _as_array(default_probs)
    lgd_arr = _as_array(lgd)
    if not (exp_arr.shape == prob_arr.shape == lgd_arr.shape):
        raise ValueError("Exposure, probability and LGD arrays must have identical shape.")
    return jnp.sum(exp_arr * prob_arr * lgd_arr)


def simulate_portfolio_losses(
    exposures: Sequence[float],
    lgd: Sequence[float],
    default_probs: Sequence[float],
    correlation_matrix: Sequence[Sequence[float]],
    num_samples: int,
    key: Array,
) -> Array:
    exp_arr = _as_array(exposures)
    lgd_arr = _as_array(lgd)
    prob_arr = _as_array(default_probs)
    defaults = gaussian_copula_samples(prob_arr, correlation_matrix, num_samples, key)
    losses = exp_arr * lgd_arr
    return defaults.astype(jnp.float32) @ losses


def portfolio_risk_metrics(
    exposures: Sequence[float],
    lgd: Sequence[float],
    default_probs: Sequence[float],
    correlation_matrix: Sequence[Sequence[float]],
    num_samples: int,
    alpha: float,
    key: Array | None = None,
) -> PortfolioLossMetrics:
    if key is None:
        key = random.PRNGKey(0)
    loss_samples = simulate_portfolio_losses(
        exposures,
        lgd,
        default_probs,
        correlation_matrix,
        num_samples,
        key,
    )
    exp_loss = jnp.mean(loss_samples)
    unexpected = jnp.std(loss_samples)
    var_level = jnp.quantile(loss_samples, alpha)
    tail_losses = loss_samples[loss_samples >= var_level]
    cvar = jnp.mean(tail_losses) if tail_losses.size > 0 else var_level
    return PortfolioLossMetrics(exp_loss, unexpected, var_level, cvar)


def single_factor_loss_distribution(
    exposures: Sequence[float],
    lgd: Sequence[float],
    default_probs: Sequence[float],
    rho: float,
    num_samples: int,
    alpha: float,
    key: Array | None = None,
) -> PortfolioLossMetrics:
    if key is None:
        key = random.PRNGKey(0)
    exp_arr = _as_array(exposures)
    lgd_arr = _as_array(lgd)
    prob_arr = _as_array(default_probs)
    if not (exp_arr.shape == lgd_arr.shape == prob_arr.shape):
        raise ValueError("Exposure, LGD, and probability vectors must align.")
    copula = SingleFactorGaussianCopula(rho)
    defaults = copula.simulate(prob_arr, num_samples, key)
    losses = exp_arr * lgd_arr
    loss_samples = defaults.astype(jnp.float32) @ losses
    exp_loss = jnp.mean(loss_samples)
    unexpected = jnp.std(loss_samples)
    var_level = jnp.quantile(loss_samples, alpha)
    tail_losses = loss_samples[loss_samples >= var_level]
    cvar = jnp.mean(tail_losses) if tail_losses.size > 0 else var_level
    return PortfolioLossMetrics(exp_loss, unexpected, var_level, cvar)
