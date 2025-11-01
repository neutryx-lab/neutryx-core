"""Aggregation layer orchestrating exposure and capital metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp

from .capital import CapitalCalculator
from .exposure import ExposureCube

Array = jnp.ndarray


@dataclass
class AggregationEngine:
    """Combine exposure simulations with capital calculations."""

    capital_calculator: CapitalCalculator

    def summarize(
        self,
        cube: ExposureCube,
        *,
        quantile: float = 0.95,
        alpha: float = 0.975,
    ) -> Mapping[str, float | Array]:
        """Return consolidated XVA metrics for the provided exposure cube."""

        epe = cube.aggregate_expected_positive()
        ene = cube.aggregate_expected_negative()
        net = cube.aggregate_net_exposure()
        pathwise = cube.aggregate_pathwise()
        capital_metrics = self.capital_calculator.compute_all(
            epe=epe, pathwise=pathwise, quantile=quantile, alpha=alpha
        )
        summary: dict[str, float | Array] = {
            "times": cube.times,
            "epe": epe,
            "ene": ene,
            "net_exposure": net,
        }
        summary.update(capital_metrics)
        return summary
