"""Default correlation modelling helpers for credit portfolios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import norm, t as student_t

Array = jnp.ndarray


def _ensure_matrix(matrix: Sequence[Sequence[float]] | Array) -> Array:
    arr = jnp.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    return arr


def gaussian_copula_samples(
    default_probabilities: Sequence[float] | Array,
    correlation_matrix: Sequence[Sequence[float]] | Array,
    num_samples: int,
    key: Array,
) -> Array:
    """Simulate correlated default indicators using a Gaussian copula."""

    probs = jnp.asarray(default_probabilities)
    if probs.ndim != 1:
        raise ValueError("Default probabilities must be a 1D array.")
    if jnp.any((probs <= 0.0) | (probs >= 1.0)):
        raise ValueError("Default probabilities must lie strictly between 0 and 1.")

    corr = _ensure_matrix(correlation_matrix)
    if corr.shape[0] != probs.shape[0]:
        raise ValueError("Correlation matrix dimension mismatch.")

    cov = corr
    normals = random.multivariate_normal(key, jnp.zeros(probs.shape[0]), cov, (num_samples,))
    uniforms = norm.cdf(normals)
    return uniforms < probs


def t_copula_samples(
    default_probabilities: Sequence[float] | Array,
    correlation_matrix: Sequence[Sequence[float]] | Array,
    degrees_of_freedom: float,
    num_samples: int,
    key: Array,
) -> Array:
    """Simulate correlated defaults using a Student t-copula."""

    if degrees_of_freedom <= 2:
        raise ValueError("Degrees of freedom must exceed 2 for finite variance.")
    probs = jnp.asarray(default_probabilities)
    if probs.ndim != 1:
        raise ValueError("Default probabilities must be a 1D array.")
    corr = _ensure_matrix(correlation_matrix)
    if corr.shape[0] != probs.shape[0]:
        raise ValueError("Correlation matrix dimension mismatch.")
    z_key, chi_key = random.split(key)
    gaussian = random.multivariate_normal(z_key, jnp.zeros(probs.shape[0]), corr, (num_samples,))
    chi2 = random.chisquare(chi_key, degrees_of_freedom, (num_samples, 1))
    scaled = gaussian / jnp.sqrt(chi2 / degrees_of_freedom)
    uniforms = student_t.cdf(scaled, degrees_of_freedom)
    return uniforms < probs


@dataclass
class SingleFactorGaussianCopula:
    """One-factor Gaussian copula commonly used for CDO analytics."""

    rho: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.rho < 1.0):
            raise ValueError("Asset correlation rho must be in [0, 1).")

    def simulate(self, default_probabilities: Sequence[float] | Array, num_samples: int, key: Array) -> Array:
        probs = jnp.asarray(default_probabilities)
        if probs.ndim != 1:
            raise ValueError("Default probabilities must be a 1D array.")
        if jnp.any((probs <= 0.0) | (probs >= 1.0)):
            raise ValueError("Default probabilities must lie strictly between 0 and 1.")
        systemic_key, idiosyncratic_key = random.split(key)
        systemic = random.normal(systemic_key, (num_samples, 1))
        idiosyncratic = random.normal(idiosyncratic_key, (num_samples, probs.shape[0]))
        latent = jnp.sqrt(self.rho) * systemic + jnp.sqrt(1.0 - self.rho) * idiosyncratic
        thresholds = norm.ppf(probs)
        return latent < thresholds
