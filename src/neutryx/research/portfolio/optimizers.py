"""Classical portfolio optimizers including Black-Litterman."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .estimators import CovarianceEstimator
from .views import ViewCollection


@dataclass
class MinimumVarianceOptimizer:
    """Compute the global minimum variance portfolio."""

    allow_short: bool = False

    def optimize(self, covariance: ArrayLike) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=float)
        n_assets = covariance.shape[0]
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")

        def objective(weights: np.ndarray) -> float:
            return float(weights.T @ covariance @ weights)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = None if self.allow_short else tuple((0.0, 1.0) for _ in range(n_assets))

        initial = np.full(n_assets, 1.0 / n_assets)
        result = minimize(objective, initial, bounds=bounds, constraints=constraints)
        if not result.success:
            raise RuntimeError(f"Minimum variance optimization failed: {result.message}")
        return result.x


@dataclass
class MaximumSharpeRatioOptimizer:
    """Maximize the portfolio Sharpe ratio."""

    risk_free_rate: float = 0.0
    allow_short: bool = False

    def optimize(self, expected_returns: ArrayLike, covariance: ArrayLike) -> np.ndarray:
        expected_returns = np.asarray(expected_returns, dtype=float)
        covariance = np.asarray(covariance, dtype=float)

        n_assets = expected_returns.shape[0]
        if covariance.shape != (n_assets, n_assets):
            raise ValueError("Covariance shape must be (n_assets, n_assets).")

        excess_returns = expected_returns - self.risk_free_rate

        def objective(weights: np.ndarray) -> float:
            variance = float(weights.T @ covariance @ weights)
            if variance <= 0:
                return np.inf
            portfolio_return = float(weights @ excess_returns)
            sharpe = portfolio_return / np.sqrt(variance)
            return -sharpe

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = None if self.allow_short else tuple((0.0, 1.0) for _ in range(n_assets))
        initial = np.full(n_assets, 1.0 / n_assets)
        result = minimize(objective, initial, bounds=bounds, constraints=constraints)
        if not result.success:
            raise RuntimeError(f"Maximum Sharpe optimization failed: {result.message}")
        return result.x


@dataclass
class BlackLittermanPosterior:
    """Container for the posterior distribution from the Black-Litterman model."""

    mean: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray


@dataclass
class BlackLittermanModel:
    """Black-Litterman portfolio model."""

    asset_names: list[str]
    market_weights: ArrayLike
    risk_aversion: float = 2.5
    tau: float = 0.05
    covariance_estimator: CovarianceEstimator = field(default_factory=CovarianceEstimator)

    def __post_init__(self) -> None:
        self.market_weights = np.asarray(self.market_weights, dtype=float)
        if not np.isclose(self.market_weights.sum(), 1.0):
            self.market_weights = self.market_weights / np.sum(self.market_weights)
        if len(self.market_weights) != len(self.asset_names):
            raise ValueError("Market weights must match number of assets.")
        self._covariance: Optional[np.ndarray] = None
        self._prior_mean: Optional[np.ndarray] = None

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            raise ValueError("Covariance has not been estimated. Call 'fit' with return data.")
        return self._covariance

    @property
    def prior_mean(self) -> np.ndarray:
        if self._prior_mean is None:
            raise ValueError("Model has not been fitted. Call 'fit' with return data.")
        return self._prior_mean

    def fit(self, returns: np.ndarray) -> None:
        """Estimate the prior distribution from historical returns."""

        covariance = self.covariance_estimator.estimate(returns)
        self._covariance = covariance
        implied = self.risk_aversion * covariance @ self.market_weights
        self._prior_mean = implied

    def posterior(self, views: Optional[ViewCollection] = None) -> BlackLittermanPosterior:
        """Compute the posterior mean, covariance and optimal weights."""

        if self._covariance is None or self._prior_mean is None:
            raise ValueError("Model must be fitted before computing the posterior.")

        covariance = self._covariance
        prior_mean = self._prior_mean

        if views is None:
            posterior_mean = prior_mean
            posterior_covariance = covariance + self.tau * covariance
        else:
            P, Q, Omega = views.to_matrices(covariance, self.tau)
            if P.size == 0:
                posterior_mean = prior_mean
                posterior_covariance = covariance + self.tau * covariance
            else:
                tau_cov_inv = np.linalg.inv(self.tau * covariance)
                middle = np.linalg.inv(tau_cov_inv + P.T @ np.linalg.inv(Omega) @ P)
                posterior_mean = middle @ (tau_cov_inv @ prior_mean + P.T @ np.linalg.inv(Omega) @ Q)
                posterior_covariance = covariance + middle

        weights = np.linalg.solve(self.risk_aversion * covariance, posterior_mean)
        weights /= np.sum(weights)
        return BlackLittermanPosterior(mean=posterior_mean, covariance=posterior_covariance, weights=weights)
