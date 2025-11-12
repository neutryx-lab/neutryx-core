"""Covariance estimation utilities for portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


EstimatorMethod = Literal["sample", "ledoit_wolf"]
ShrinkageTarget = Literal["identity", "diagonal"]


@dataclass
class CovarianceEstimator:
    """Estimate covariance matrices from historical returns.

    Parameters
    ----------
    method:
        Estimation method. ``"sample"`` returns the empirical covariance matrix
        while ``"ledoit_wolf"`` applies the Ledoit-Wolf shrinkage procedure.
    shrinkage_target:
        Target structure for shrinkage based estimators.
    min_variance:
        Floor applied to the diagonal elements to ensure positive definiteness.
    """

    method: EstimatorMethod = "sample"
    shrinkage_target: ShrinkageTarget = "identity"
    min_variance: float = 1e-8

    def estimate(self, returns: np.ndarray) -> np.ndarray:
        """Estimate the covariance matrix from returns.

        Parameters
        ----------
        returns:
            Two dimensional array with shape ``(n_samples, n_assets)`` containing
            arithmetic returns.
        """

        if returns.ndim != 2:
            raise ValueError("Returns must be a 2-dimensional array of shape (n_samples, n_assets).")

        demeaned = returns - np.mean(returns, axis=0, keepdims=True)
        n_samples = demeaned.shape[0]
        if n_samples < 2:
            raise ValueError("At least two observations are required to estimate covariance.")

        sample_cov = (demeaned.T @ demeaned) / (n_samples - 1)
        sample_cov = self._stabilize(sample_cov)

        if self.method == "sample":
            return sample_cov
        if self.method == "ledoit_wolf":
            return self._ledoit_wolf(demeaned, sample_cov)
        raise ValueError(f"Unsupported estimation method: {self.method}")

    # ------------------------------------------------------------------
    def _ledoit_wolf(self, demeaned: np.ndarray, sample_cov: np.ndarray) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to the sample covariance matrix.

        The implementation follows the derivation from Ledoit & Wolf (2004) and
        supports shrinkage towards either the identity matrix or a diagonal
        target composed of the sample variances.
        """

        n_samples, n_assets = demeaned.shape
        if n_assets == 1:
            return sample_cov

        if self.shrinkage_target == "identity":
            mu = np.trace(sample_cov) / n_assets
            target = np.eye(n_assets) * mu
        elif self.shrinkage_target == "diagonal":
            target = np.diag(np.diag(sample_cov))
        else:
            raise ValueError(f"Unsupported shrinkage target: {self.shrinkage_target}")

        # Compute phi, the sum of squared deviations of sample covariances.
        x = demeaned
        phi_matrix = (x ** 2).T @ (x ** 2) / (n_samples - 1) - 2 * (
            (x.T @ x) * sample_cov
        ) / (n_samples - 1) + sample_cov ** 2
        phi = float(phi_matrix.sum())

        gamma = np.linalg.norm(sample_cov - target, "fro") ** 2
        if gamma == 0:
            shrinkage = 0.0
        else:
            kappa = phi / gamma
            shrinkage = max(0.0, min(1.0, kappa / n_samples))

        shrunk = shrinkage * target + (1.0 - shrinkage) * sample_cov
        return self._stabilize(shrunk)

    def _stabilize(self, covariance: np.ndarray) -> np.ndarray:
        diagonal = np.clip(np.diag(covariance), a_min=self.min_variance, a_max=None)
        stabilized = covariance.copy()
        np.fill_diagonal(stabilized, diagonal)
        return stabilized

    def with_params(
        self,
        *,
        method: Optional[EstimatorMethod] = None,
        shrinkage_target: Optional[ShrinkageTarget] = None,
        min_variance: Optional[float] = None,
    ) -> "CovarianceEstimator":
        """Return a new estimator with updated parameters."""

        return CovarianceEstimator(
            method=method or self.method,
            shrinkage_target=shrinkage_target or self.shrinkage_target,
            min_variance=min_variance or self.min_variance,
        )
