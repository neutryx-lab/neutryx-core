"""Utilities for handling Black-Litterman view inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass
class PortfolioView:
    """Represents a single investor view.

    The view can either be specified directly with a weight vector matching the
    asset universe, or with a mapping ``{asset: weight}``.
    """

    weights: Sequence[float] | Mapping[str, float]
    expected_return: float
    confidence: float = 0.5
    label: str | None = None

    def resolve(self, asset_names: Sequence[str]) -> np.ndarray:
        """Return a weight vector aligned with ``asset_names``."""

        if isinstance(self.weights, Mapping):
            weight_vector = np.zeros(len(asset_names))
            asset_index: Dict[str, int] = {name: idx for idx, name in enumerate(asset_names)}
            for key, value in self.weights.items():
                if key not in asset_index:
                    raise KeyError(f"Unknown asset '{key}' in view '{self.label or ''}'.")
                weight_vector[asset_index[key]] = value
            return weight_vector

        weight_array = np.asarray(self.weights, dtype=float)
        if weight_array.shape[0] != len(asset_names):
            raise ValueError(
                "Weight vector length does not match number of assets in the universe."
            )
        return weight_array

    @classmethod
    def relative(
        cls,
        asset_long: str,
        asset_short: str,
        expected_outperformance: float,
        *,
        confidence: float = 0.5,
        label: str | None = None,
    ) -> "PortfolioView":
        """Create a relative view between two assets."""

        weights = {asset_long: 1.0, asset_short: -1.0}
        return cls(weights=weights, expected_return=expected_outperformance, confidence=confidence, label=label)


@dataclass
class ViewCollection:
    """Container for multiple :class:`PortfolioView` objects."""

    asset_names: Sequence[str]
    views: List[PortfolioView] = field(default_factory=list)

    def add(self, view: PortfolioView) -> None:
        self.views.append(view)

    def extend(self, views: Iterable[PortfolioView]) -> None:
        for view in views:
            self.add(view)

    def to_matrices(self, covariance: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (P, Q, Omega) matrices for the Black-Litterman model."""

        if not self.views:
            n_assets = len(self.asset_names)
            return (
                np.zeros((0, n_assets)),
                np.zeros(0),
                np.zeros((0, 0)),
            )

        tau_cov = tau * covariance
        rows: List[np.ndarray] = []
        q_values: List[float] = []
        variances: List[float] = []
        for view in self.views:
            weights = view.resolve(self.asset_names)
            rows.append(weights)
            q_values.append(view.expected_return)
            view_var = float(weights @ tau_cov @ weights.T)
            if view_var <= 0:
                view_var = 1e-8
            view_var /= max(view.confidence, 1e-6)
            variances.append(view_var)

        P = np.vstack(rows)
        Q = np.asarray(q_values)
        Omega = np.diag(variances)
        return P, Q, Omega

    def summary(self) -> List[Dict[str, float]]:
        """Return a structured representation of the views."""

        summaries: List[Dict[str, float]] = []
        for view in self.views:
            weights = view.resolve(self.asset_names)
            entry = {
                "label": view.label or "",
                "expected_return": view.expected_return,
                "confidence": view.confidence,
            }
            for name, weight in zip(self.asset_names, weights):
                if weight != 0:
                    entry[f"weight_{name}"] = weight
            summaries.append(entry)
        return summaries
