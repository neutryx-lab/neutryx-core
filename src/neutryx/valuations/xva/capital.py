"""Capital metric utilities for the XVA engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp

from neutryx.valuations import cva as cva_module

Array = jnp.ndarray


@dataclass
class CapitalCalculator:
    """Compute regulatory and economic capital metrics from exposure profiles."""

    discount_curve: Array
    default_probabilities: Array
    lgd: float = 0.6
    funding_spread: float = 0.0
    hurdle_rate: float = 0.1

    def __post_init__(self) -> None:
        self.discount_curve = jnp.asarray(self.discount_curve)
        self.default_probabilities = jnp.asarray(self.default_probabilities)
        if self.discount_curve.shape != self.default_probabilities.shape:
            raise ValueError("Discount curve and default probabilities must have identical shapes.")

    def cva(self, epe: Array) -> float:
        """Compute credit valuation adjustment from the expected positive exposure profile."""

        if epe.shape != self.discount_curve.shape:
            raise ValueError("EPE profile must align with discount curve length.")
        return float(cva_module.cva(epe, self.discount_curve, self.default_probabilities, self.lgd))

    def pfe(self, pathwise: Array, quantile: float = 0.95) -> Array:
        """Potential future exposure at the provided quantile."""

        if not 0.0 < quantile < 1.0:
            raise ValueError("Quantile must lie in (0, 1).")
        positive = jnp.maximum(pathwise, 0.0)
        return jnp.quantile(positive, quantile, axis=0)

    def expected_shortfall(self, pathwise: Array, alpha: float = 0.975) -> Array:
        """Expected shortfall of the positive exposure distribution."""

        if not 0.0 < alpha < 1.0:
            raise ValueError("Alpha must lie in (0, 1).")
        positive = jnp.maximum(pathwise, 0.0)
        sorted_paths = jnp.sort(positive, axis=0)
        n_paths = sorted_paths.shape[0]
        threshold = int(jnp.ceil(alpha * n_paths))
        if threshold >= n_paths:
            return jnp.zeros(sorted_paths.shape[1], dtype=sorted_paths.dtype)
        tail = sorted_paths[threshold:]
        return tail.mean(axis=0)

    def kva(self, expected_shortfall_profile: Array) -> float:
        """Compute a simplified capital valuation adjustment."""

        if expected_shortfall_profile.shape != self.discount_curve.shape:
            raise ValueError("Expected shortfall profile must align with discount curve length.")
        discounted = expected_shortfall_profile * self.discount_curve
        return float((discounted * self.hurdle_rate).sum())

    def funding_cost(self, epe: Array) -> float:
        """Approximate funding cost from the exposure profile."""

        if epe.shape != self.discount_curve.shape:
            raise ValueError("EPE profile must align with discount curve length.")
        return float((self.discount_curve * epe * self.funding_spread).sum())

    def compute_all(
        self,
        *,
        epe: Array,
        pathwise: Array,
        quantile: float = 0.95,
        alpha: float = 0.975,
    ) -> Mapping[str, float | Array]:
        """Convenience helper returning the full capital metric stack."""

        pfe_profile = self.pfe(pathwise, quantile=quantile)
        es_profile = self.expected_shortfall(pathwise, alpha=alpha)
        return {
            "cva": self.cva(epe),
            "funding_cost": self.funding_cost(epe),
            "pfe": pfe_profile,
            "expected_shortfall": es_profile,
            "kva": self.kva(es_profile),
        }
