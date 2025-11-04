"""Correlation matrices infrastructure for multi-asset derivatives.

This module provides tools for managing, calibrating, and ensuring
consistency of correlation matrices used in multi-asset pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.linalg import cholesky, sqrtm


class CorrelationMethod(Enum):
    """Method for correlation matrix construction."""
    CONSTANT = "constant"  # Constant correlation
    HISTORICAL = "historical"  # Historical estimation
    IMPLIED = "implied"  # Implied from market prices
    FACTOR = "factor"  # Factor model
    SHRINKAGE = "shrinkage"  # Ledoit-Wolf shrinkage


@dataclass
class CorrelationMatrix:
    """Correlation matrix with validation and decomposition.

    Attributes:
        matrix: Correlation matrix [n_assets, n_assets]
        asset_names: Names of assets
        is_validated: Whether matrix has been validated
    """

    matrix: Array
    asset_names: Optional[List[str]] = None
    is_validated: bool = False

    def __post_init__(self):
        """Initialize and validate correlation matrix."""
        self.matrix = jnp.asarray(self.matrix)

        if self.matrix.ndim != 2:
            raise ValueError("Correlation matrix must be 2D")

        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")

        if self.asset_names is not None:
            if len(self.asset_names) != self.matrix.shape[0]:
                raise ValueError("Number of asset names must match matrix dimension")

        if not self.is_validated:
            self.validate()

    def validate(self) -> List[str]:
        """Validate correlation matrix properties.

        Returns:
            List of validation warnings/errors
        """
        warnings = []
        n = self.matrix.shape[0]

        # Check diagonal is 1
        diag = jnp.diag(self.matrix)
        if not jnp.allclose(diag, 1.0, atol=1e-6):
            warnings.append("Diagonal elements are not all 1.0")

        # Check symmetry
        if not jnp.allclose(self.matrix, self.matrix.T, atol=1e-6):
            warnings.append("Matrix is not symmetric")

        # Check values in [-1, 1]
        if jnp.any(self.matrix < -1.0) or jnp.any(self.matrix > 1.0):
            warnings.append("Correlation values outside [-1, 1]")

        # Check positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(self.matrix)
        min_eigenvalue = float(jnp.min(eigenvalues))

        if min_eigenvalue < -1e-6:
            warnings.append(f"Matrix is not positive semi-definite (min eigenvalue: {min_eigenvalue:.6f})")

        object.__setattr__(self, "is_validated", True)

        return warnings

    def make_positive_definite(self, method: str = "eigenvalue") -> CorrelationMatrix:
        """Ensure matrix is positive definite.

        Args:
            method: Method to use ("eigenvalue", "nearestPD", "shrinkage")

        Returns:
            Positive definite correlation matrix
        """
        if method == "eigenvalue":
            # Eigenvalue floor method
            eigenvalues, eigenvectors = jnp.linalg.eigh(self.matrix)
            eigenvalues = jnp.maximum(eigenvalues, 1e-8)

            # Reconstruct matrix
            fixed_matrix = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T

            # Rescale to correlation matrix
            d = jnp.sqrt(jnp.diag(fixed_matrix))
            corr_matrix = fixed_matrix / jnp.outer(d, d)

            # Ensure diagonal is exactly 1
            corr_matrix = corr_matrix - jnp.diag(jnp.diag(corr_matrix)) + jnp.eye(self.matrix.shape[0])

            return CorrelationMatrix(corr_matrix, self.asset_names, is_validated=False)

        elif method == "shrinkage":
            # Shrinkage towards identity
            n = self.matrix.shape[0]
            identity = jnp.eye(n)
            alpha = 0.1  # Shrinkage intensity

            shrunk = (1 - alpha) * self.matrix + alpha * identity

            return CorrelationMatrix(shrunk, self.asset_names, is_validated=False)

        else:
            raise ValueError(f"Unknown method: {method}")

    def cholesky_decomposition(self) -> Array:
        """Compute Cholesky decomposition L such that LL^T = Corr.

        Returns:
            Lower triangular Cholesky factor

        Raises:
            ValueError: If matrix is not positive definite
        """
        try:
            # Use NumPy for Cholesky as it has better error handling
            L = np.linalg.cholesky(np.array(self.matrix))
            return jnp.asarray(L)
        except np.linalg.LinAlgError:
            # Matrix is not positive definite, fix it first
            fixed = self.make_positive_definite()
            return fixed.cholesky_decomposition()

    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets.

        Args:
            asset1: First asset name
            asset2: Second asset name

        Returns:
            Correlation coefficient
        """
        if self.asset_names is None:
            raise ValueError("Asset names not provided")

        idx1 = self.asset_names.index(asset1)
        idx2 = self.asset_names.index(asset2)

        return float(self.matrix[idx1, idx2])

    def set_correlation(self, asset1: str, asset2: str, value: float) -> CorrelationMatrix:
        """Set correlation between two assets.

        Args:
            asset1: First asset name
            asset2: Second asset name
            value: Correlation value

        Returns:
            New correlation matrix with updated value
        """
        if self.asset_names is None:
            raise ValueError("Asset names not provided")

        idx1 = self.asset_names.index(asset1)
        idx2 = self.asset_names.index(asset2)

        new_matrix = self.matrix.at[idx1, idx2].set(value)
        new_matrix = new_matrix.at[idx2, idx1].set(value)

        return CorrelationMatrix(new_matrix, self.asset_names, is_validated=False)

    def to_covariance(self, volatilities: Array) -> Array:
        """Convert correlation matrix to covariance matrix.

        Args:
            volatilities: Array of volatilities [n_assets]

        Returns:
            Covariance matrix
        """
        vol_matrix = jnp.outer(volatilities, volatilities)
        return self.matrix * vol_matrix


@dataclass
class CorrelationCalibrator:
    """Calibrate correlation matrices from market data.

    Supports multiple calibration methods:
    - Historical estimation from time series
    - Implied calibration from option prices
    - Factor models
    - Shrinkage estimators
    """

    method: CorrelationMethod = CorrelationMethod.HISTORICAL

    def calibrate_historical(
        self, returns: Array, window: Optional[int] = None
    ) -> CorrelationMatrix:
        """Calibrate correlation from historical returns.

        Args:
            returns: Return time series [n_observations, n_assets]
            window: Rolling window size (None = full sample)

        Returns:
            Calibrated correlation matrix
        """
        if window is not None:
            # Use only last 'window' observations
            returns = returns[-window:, :]

        # Compute sample correlation
        corr_matrix = jnp.corrcoef(returns, rowvar=False)

        return CorrelationMatrix(corr_matrix)

    def calibrate_with_shrinkage(
        self, returns: Array, shrinkage_intensity: Optional[float] = None
    ) -> CorrelationMatrix:
        """Calibrate correlation using Ledoit-Wolf shrinkage.

        Shrinkage towards identity matrix helps stabilize estimates
        when sample size is small relative to number of assets.

        Args:
            returns: Return time series [n_observations, n_assets]
            shrinkage_intensity: Shrinkage parameter (None = auto-estimate)

        Returns:
            Shrinkage correlation matrix
        """
        n_obs, n_assets = returns.shape

        # Sample correlation
        sample_corr = jnp.corrcoef(returns, rowvar=False)

        # Target: identity matrix
        target = jnp.eye(n_assets)

        if shrinkage_intensity is None:
            # Ledoit-Wolf optimal shrinkage intensity
            # Simplified formula
            shrinkage_intensity = min(1.0, (n_assets / n_obs) * 0.5)

        # Shrinkage estimator
        corr_matrix = (1 - shrinkage_intensity) * sample_corr + shrinkage_intensity * target

        return CorrelationMatrix(corr_matrix)

    def calibrate_factor_model(
        self, returns: Array, n_factors: int = 3
    ) -> Tuple[CorrelationMatrix, Array, Array]:
        """Calibrate factor model correlation structure.

        Uses PCA to extract common factors:
        R_i = Σ β_ij F_j + ε_i

        Args:
            returns: Return time series [n_observations, n_assets]
            n_factors: Number of factors to extract

        Returns:
            Tuple of (correlation_matrix, factor_loadings, factor_returns)
        """
        n_obs, n_assets = returns.shape

        # Standardize returns
        mean = jnp.mean(returns, axis=0)
        std = jnp.std(returns, axis=0) + 1e-8
        standardized = (returns - mean) / std

        # PCA: eigenvalue decomposition of correlation matrix
        corr = jnp.corrcoef(standardized, rowvar=False)
        eigenvalues, eigenvectors = jnp.linalg.eigh(corr)

        # Sort eigenvalues in descending order
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Extract top factors
        factor_loadings = eigenvectors[:, :n_factors] * jnp.sqrt(eigenvalues[:n_factors])

        # Reconstruct correlation from factors
        factor_corr = factor_loadings @ factor_loadings.T

        # Add diagonal to ensure unit diagonal
        residual_var = jnp.maximum(1.0 - jnp.sum(factor_loadings**2, axis=1), 0.0)
        reconstructed_corr = factor_corr + jnp.diag(residual_var)

        # Normalize to correlation
        d = jnp.sqrt(jnp.diag(reconstructed_corr))
        corr_matrix = reconstructed_corr / jnp.outer(d, d)

        # Factor returns (projections)
        factor_returns = standardized @ factor_loadings

        return CorrelationMatrix(corr_matrix), factor_loadings, factor_returns


@dataclass
class CorrelationScenario:
    """Correlation scenario for stress testing and risk analysis.

    Defines a stressed correlation scenario (e.g., crisis correlations).
    """

    name: str
    base_correlation: CorrelationMatrix
    stress_adjustments: Dict[Tuple[str, str], float]

    def apply(self) -> CorrelationMatrix:
        """Apply stress adjustments to base correlation.

        Returns:
            Stressed correlation matrix
        """
        stressed = self.base_correlation.matrix

        for (asset1, asset2), adjustment in self.stress_adjustments.items():
            if self.base_correlation.asset_names is None:
                raise ValueError("Asset names required for stress scenarios")

            idx1 = self.base_correlation.asset_names.index(asset1)
            idx2 = self.base_correlation.asset_names.index(asset2)

            # Apply additive adjustment
            new_corr = float(stressed[idx1, idx2]) + adjustment
            new_corr = jnp.clip(new_corr, -0.99, 0.99)

            stressed = stressed.at[idx1, idx2].set(new_corr)
            stressed = stressed.at[idx2, idx1].set(new_corr)

        # Ensure positive definite
        return CorrelationMatrix(
            stressed, self.base_correlation.asset_names, is_validated=False
        ).make_positive_definite()


def generate_correlated_normals(
    key: jax.random.KeyArray, correlation: CorrelationMatrix, n_samples: int
) -> Array:
    """Generate correlated normal random variables.

    Args:
        key: JAX random key
        correlation: Correlation matrix
        n_samples: Number of samples to generate

    Returns:
        Correlated normal samples [n_samples, n_assets]
    """
    n_assets = correlation.matrix.shape[0]

    # Generate independent normals
    independent = jax.random.normal(key, (n_samples, n_assets))

    # Apply Cholesky decomposition to induce correlation
    L = correlation.cholesky_decomposition()
    correlated = independent @ L.T

    return correlated


def correlation_term_structure(
    base_correlation: float, tenor_adjustment: Callable[[float], float], tenors: Array
) -> Array:
    """Model correlation term structure.

    Many correlation models assume correlation varies with tenor:
    ρ(T) = ρ_∞ + (ρ_0 - ρ_∞) × exp(-λT)

    Args:
        base_correlation: Long-term correlation ρ_∞
        tenor_adjustment: Function mapping tenor to correlation adjustment
        tenors: Array of tenors

    Returns:
        Array of correlations for each tenor
    """
    return jnp.array([base_correlation * tenor_adjustment(t) for t in tenors])


def equicorrelation_matrix(n_assets: int, rho: float) -> CorrelationMatrix:
    """Create equicorrelation matrix (constant off-diagonal correlation).

    Args:
        n_assets: Number of assets
        rho: Off-diagonal correlation

    Returns:
        Equicorrelation matrix
    """
    matrix = jnp.ones((n_assets, n_assets)) * rho + jnp.eye(n_assets) * (1 - rho)
    return CorrelationMatrix(matrix)


def block_correlation_matrix(
    block_sizes: List[int], intra_block_corr: List[float], inter_block_corr: float
) -> CorrelationMatrix:
    """Create block correlation matrix.

    Useful for sector-based correlation structures:
    - High correlation within sectors
    - Lower correlation between sectors

    Args:
        block_sizes: Size of each block (sector)
        intra_block_corr: Correlation within each block
        inter_block_corr: Correlation between blocks

    Returns:
        Block correlation matrix
    """
    n_total = sum(block_sizes)
    matrix = jnp.ones((n_total, n_total)) * inter_block_corr

    offset = 0
    for size, rho in zip(block_sizes, intra_block_corr):
        # Set intra-block correlation
        matrix = matrix.at[offset : offset + size, offset : offset + size].set(rho)
        offset += size

    # Set diagonal to 1
    matrix = matrix - jnp.diag(jnp.diag(matrix)) + jnp.eye(n_total)

    return CorrelationMatrix(matrix)
