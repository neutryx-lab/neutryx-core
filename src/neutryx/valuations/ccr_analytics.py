"""Enhanced Counterparty Credit Risk (CCR) Analytics.

This module provides comprehensive CCR analytics building on existing XVA infrastructure:
- Advanced exposure metrics (EE, PFE, EPE, Effective EPE for Basel)
- Multi-netting set aggregation
- Collateral optimization
- Enhanced wrong-way risk analytics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from neutryx.valuations.xva.exposure import ExposureResult, ExposureCube


class ExposureMetric(Enum):
    """Types of exposure metrics."""

    EE = "expected_exposure"  # Expected Exposure
    PFE = "potential_future_exposure"  # Potential Future Exposure
    EPE = "expected_positive_exposure"  # Expected Positive Exposure
    ENE = "expected_negative_exposure"  # Expected Negative Exposure
    EEPE = "effective_epe"  # Effective EPE for Basel
    MAXPFE = "max_pfe"  # Maximum PFE


@dataclass
class ExposureProfile:
    """Comprehensive exposure profile for a counterparty or netting set.

    Attributes:
        times: Time grid for exposure profile
        expected_exposure: Expected Exposure (EE) at each time point
        pfe_95: 95th percentile Potential Future Exposure
        pfe_97_5: 97.5th percentile PFE (Basel standard)
        pfe_99: 99th percentile PFE
        expected_positive_exposure: Expected Positive Exposure (EPE)
        expected_negative_exposure: Expected Negative Exposure (ENE)
        effective_epe: Effective EPE for Basel capital calculation
        max_pfe: Maximum PFE over the profile
        pathwise_exposures: Full path distribution [n_paths, n_times]
    """

    times: Array
    expected_exposure: Array
    pfe_95: Array
    pfe_97_5: Array
    pfe_99: Array
    expected_positive_exposure: Array
    expected_negative_exposure: Array
    effective_epe: float
    max_pfe: float
    pathwise_exposures: Optional[Array] = None

    @classmethod
    def from_paths(
        cls,
        times: Array,
        exposure_paths: Array,
        include_pathwise: bool = False,
    ) -> "ExposureProfile":
        """Compute all exposure metrics from simulated paths.

        Args:
            times: Time grid
            exposure_paths: Exposure paths [n_paths, n_times]
            include_pathwise: Whether to store full pathwise data

        Returns:
            ExposureProfile with all metrics computed
        """
        # Expected Exposure (mean across paths)
        ee = jnp.mean(exposure_paths, axis=0)

        # PFE at different quantiles
        pfe_95 = jnp.percentile(exposure_paths, 95.0, axis=0)
        pfe_97_5 = jnp.percentile(exposure_paths, 97.5, axis=0)
        pfe_99 = jnp.percentile(exposure_paths, 99.0, axis=0)

        # EPE and ENE
        positive_exposures = jnp.maximum(exposure_paths, 0.0)
        negative_exposures = jnp.maximum(-exposure_paths, 0.0)

        epe_profile = jnp.mean(positive_exposures, axis=0)
        ene_profile = jnp.mean(negative_exposures, axis=0)

        # Effective EPE (time-weighted average of EPE, non-decreasing)
        # Per Basel: Effective EPE = max(EPE(t0), average(EPE(0, t)))
        epe_cumulative = jnp.cumsum(epe_profile) / jnp.arange(1, len(epe_profile) + 1)
        effective_epe_profile = jnp.maximum.accumulate(epe_cumulative)
        effective_epe = float(effective_epe_profile[-1])

        # Max PFE
        max_pfe = float(jnp.max(pfe_97_5))

        return cls(
            times=times,
            expected_exposure=ee,
            pfe_95=pfe_95,
            pfe_97_5=pfe_97_5,
            pfe_99=pfe_99,
            expected_positive_exposure=epe_profile,
            expected_negative_exposure=ene_profile,
            effective_epe=effective_epe,
            max_pfe=max_pfe,
            pathwise_exposures=exposure_paths if include_pathwise else None,
        )

    def time_averaged_epe(self) -> float:
        """Compute time-averaged EPE."""
        # Trapezoidal integration
        dt = jnp.diff(self.times, prepend=0.0)
        return float(jnp.sum(self.expected_positive_exposure * dt))

    def peak_exposure(self, metric: str = "pfe_97_5") -> Tuple[float, float]:
        """Find peak exposure and the time it occurs.

        Args:
            metric: Which metric to use ('pfe_95', 'pfe_97_5', 'pfe_99', 'ee')

        Returns:
            Tuple of (peak_value, time_of_peak)
        """
        profile = getattr(self, metric)
        peak_idx = int(jnp.argmax(profile))
        peak_value = float(profile[peak_idx])
        peak_time = float(self.times[peak_idx])

        return peak_value, peak_time


@dataclass
class NettingSetExposure:
    """Exposure for a single netting set.

    Attributes:
        netting_set_id: Identifier for the netting set
        counterparty_id: Counterparty identifier
        profile: Exposure profile
        trade_ids: List of trade IDs in this netting set
        collateralized: Whether collateral is applied
        csa_threshold: CSA threshold if collateralized
    """

    netting_set_id: str
    counterparty_id: str
    profile: ExposureProfile
    trade_ids: List[str] = field(default_factory=list)
    collateralized: bool = False
    csa_threshold: float = 0.0


@dataclass
class MultiNettingSetAggregator:
    """Aggregate exposures across multiple netting sets.

    Handles:
    - Counterparty-level aggregation
    - Portfolio-level aggregation
    - Correlation between netting sets
    - Diversification benefits
    """

    netting_sets: List[NettingSetExposure]
    correlation_matrix: Optional[Array] = None

    def aggregate_by_counterparty(self) -> Dict[str, ExposureProfile]:
        """Aggregate exposure by counterparty.

        Assumes perfect correlation between netting sets of the same counterparty.

        Returns:
            Dictionary mapping counterparty_id to aggregated ExposureProfile
        """
        # Group by counterparty
        counterparty_groups: Dict[str, List[NettingSetExposure]] = {}

        for ns in self.netting_sets:
            if ns.counterparty_id not in counterparty_groups:
                counterparty_groups[ns.counterparty_id] = []
            counterparty_groups[ns.counterparty_id].append(ns)

        # Aggregate each counterparty
        aggregated = {}

        for cp_id, netting_sets_list in counterparty_groups.items():
            # Simple sum across netting sets (perfect correlation assumption)
            # More sophisticated: use correlation matrix

            # Check all have same time grid
            times = netting_sets_list[0].profile.times
            assert all(jnp.array_equal(ns.profile.times, times) for ns in netting_sets_list)

            # Sum exposures
            if all(ns.profile.pathwise_exposures is not None for ns in netting_sets_list):
                # We have pathwise data - sum paths
                total_paths = sum(ns.profile.pathwise_exposures for ns in netting_sets_list)
                aggregated[cp_id] = ExposureProfile.from_paths(
                    times=times,
                    exposure_paths=total_paths,
                    include_pathwise=True,
                )
            else:
                # No pathwise data - sum metrics directly (conservative)
                total_ee = sum(ns.profile.expected_exposure for ns in netting_sets_list)
                total_pfe_95 = sum(ns.profile.pfe_95 for ns in netting_sets_list)
                total_pfe_97_5 = sum(ns.profile.pfe_97_5 for ns in netting_sets_list)
                total_pfe_99 = sum(ns.profile.pfe_99 for ns in netting_sets_list)
                total_epe = sum(ns.profile.expected_positive_exposure for ns in netting_sets_list)
                total_ene = sum(ns.profile.expected_negative_exposure for ns in netting_sets_list)

                effective_epe = float(jnp.max(jnp.cumsum(total_epe) / jnp.arange(1, len(total_epe) + 1)))
                max_pfe = float(jnp.max(total_pfe_97_5))

                aggregated[cp_id] = ExposureProfile(
                    times=times,
                    expected_exposure=total_ee,
                    pfe_95=total_pfe_95,
                    pfe_97_5=total_pfe_97_5,
                    pfe_99=total_pfe_99,
                    expected_positive_exposure=total_epe,
                    expected_negative_exposure=total_ene,
                    effective_epe=effective_epe,
                    max_pfe=max_pfe,
                )

        return aggregated

    def aggregate_portfolio(self, correlation: float = 0.5) -> ExposureProfile:
        """Aggregate to portfolio level with correlation adjustment.

        Args:
            correlation: Average correlation between counterparties (0 to 1)

        Returns:
            Portfolio-level ExposureProfile

        Notes:
            Diversification benefit = sqrt(1 + (N-1) * correlation) / sqrt(N)
            where N is number of counterparties
        """
        n_counterparties = len(set(ns.counterparty_id for ns in self.netting_sets))

        # Aggregate by counterparty first
        cp_exposures = self.aggregate_by_counterparty()

        # Get time grid (all should be same)
        times = self.netting_sets[0].profile.times

        # Sum EPE across counterparties
        total_epe = sum(prof.expected_positive_exposure for prof in cp_exposures.values())

        # Apply diversification benefit
        diversification_factor = jnp.sqrt(1 + (n_counterparties - 1) * correlation) / jnp.sqrt(n_counterparties)

        diversified_epe = total_epe * diversification_factor

        # For PFE, use sum of max PFEs (conservative)
        pfe_sum = sum(prof.pfe_97_5 for prof in cp_exposures.values())

        # Simple aggregation for other metrics
        ee_sum = sum(prof.expected_exposure for prof in cp_exposures.values())

        effective_epe = float(jnp.max(jnp.cumsum(diversified_epe) / jnp.arange(1, len(diversified_epe) + 1)))
        max_pfe = float(jnp.max(pfe_sum))

        return ExposureProfile(
            times=times,
            expected_exposure=ee_sum,
            pfe_95=pfe_sum,  # Conservative
            pfe_97_5=pfe_sum,
            pfe_99=pfe_sum,
            expected_positive_exposure=diversified_epe,
            expected_negative_exposure=jnp.zeros_like(times),  # Simplified
            effective_epe=effective_epe,
            max_pfe=max_pfe,
        )


@dataclass
class CollateralOptimizer:
    """Optimize collateral terms to minimize XVA costs.

    Finds optimal CSA threshold and independent amount to balance:
    - CVA reduction (lower threshold = less exposure)
    - Collateral posting costs (lower threshold = more collateral)
    - Operational costs
    """

    exposure_profile: ExposureProfile
    lgd: float = 0.6
    hazard_rate: float = 0.01
    discount_rate: float = 0.05
    collateral_cost: float = 0.001  # Cost per unit of collateral posted
    operational_cost_fn: Optional[Callable[[float], float]] = None

    def compute_cva(self, threshold: float, independent_amount: float = 0.0) -> float:
        """Compute CVA for given collateral terms.

        Args:
            threshold: CSA threshold
            independent_amount: Independent amount (always at risk)

        Returns:
            CVA value
        """
        # Apply collateral adjustment to EPE
        collateralized_epe = jnp.maximum(
            self.exposure_profile.expected_positive_exposure - threshold,
            0.0
        ) + independent_amount

        # Compute CVA
        dt = jnp.diff(self.exposure_profile.times, prepend=0.0)
        df = jnp.exp(-self.discount_rate * self.exposure_profile.times)

        # Marginal default probability
        survival = jnp.exp(-self.hazard_rate * self.exposure_profile.times)
        dpd = jnp.concatenate([jnp.array([0.0]), -jnp.diff(survival)])

        cva = float(jnp.sum(df * collateralized_epe * dpd * self.lgd))

        return cva

    def compute_collateral_cost(self, threshold: float) -> float:
        """Estimate cost of posting collateral.

        Args:
            threshold: CSA threshold

        Returns:
            Present value of collateral costs
        """
        # Expected collateral posted = max(0, EPE - threshold)
        collateral_posted = jnp.maximum(
            self.exposure_profile.expected_positive_exposure - threshold,
            0.0
        )

        # Cost of collateral
        dt = jnp.diff(self.exposure_profile.times, prepend=0.0)
        df = jnp.exp(-self.discount_rate * self.exposure_profile.times)

        cost = float(jnp.sum(df * collateral_posted * self.collateral_cost * dt))

        return cost

    def total_cost(self, threshold: float, independent_amount: float = 0.0) -> float:
        """Total cost including CVA, collateral costs, and operational costs.

        Args:
            threshold: CSA threshold
            independent_amount: Independent amount

        Returns:
            Total cost
        """
        cva = self.compute_cva(threshold, independent_amount)
        collateral_cost = self.compute_collateral_cost(threshold)

        operational_cost = 0.0
        if self.operational_cost_fn is not None:
            operational_cost = self.operational_cost_fn(threshold)

        return cva + collateral_cost + operational_cost

    def optimize_threshold(
        self,
        threshold_range: Tuple[float, float] = (0.0, 1_000_000.0),
        n_points: int = 100,
    ) -> Tuple[float, float]:
        """Find optimal threshold using grid search.

        Args:
            threshold_range: (min_threshold, max_threshold)
            n_points: Number of points to evaluate

        Returns:
            Tuple of (optimal_threshold, minimal_cost)
        """
        thresholds = jnp.linspace(threshold_range[0], threshold_range[1], n_points)

        costs = jnp.array([self.total_cost(float(t)) for t in thresholds])

        optimal_idx = int(jnp.argmin(costs))
        optimal_threshold = float(thresholds[optimal_idx])
        minimal_cost = float(costs[optimal_idx])

        return optimal_threshold, minimal_cost


@dataclass
class WrongWayRiskAnalytics:
    """Enhanced wrong-way risk analytics.

    Extends basic WWR modeling with:
    - WWR impact quantification
    - Specific vs general WWR decomposition
    - WWR-adjusted CVA
    """

    base_exposure_profile: ExposureProfile
    wwr_correlation: float = 0.0
    wwr_volatility: float = 0.0

    def compute_wwr_multiplier(self, correlation: float, volatility: float = 0.3) -> float:
        """Compute WWR multiplier for CVA adjustment.

        Args:
            correlation: Correlation between exposure and credit quality
            volatility: Volatility of exposure

        Returns:
            WWR multiplier (> 1.0 for wrong-way risk, < 1.0 for right-way risk)

        Notes:
            Simplified Gaussian copula approximation:
            WWR_multiplier ≈ 1 + 0.5 * correlation * volatility
        """
        return 1.0 + 0.5 * correlation * volatility

    def wwr_adjusted_epe(self, correlation: Optional[float] = None) -> Array:
        """Compute WWR-adjusted EPE profile.

        Args:
            correlation: WWR correlation (uses self.wwr_correlation if None)

        Returns:
            Adjusted EPE profile
        """
        corr = correlation if correlation is not None else self.wwr_correlation

        # Estimate exposure volatility from profile
        if self.base_exposure_profile.pathwise_exposures is not None:
            vol = float(jnp.std(self.base_exposure_profile.pathwise_exposures))
        else:
            # Approximate from EPE profile
            vol = float(jnp.std(self.base_exposure_profile.expected_positive_exposure))

        multiplier = self.compute_wwr_multiplier(corr, vol)

        return self.base_exposure_profile.expected_positive_exposure * multiplier

    def decompose_wwr_impact(self) -> Dict[str, float]:
        """Decompose WWR impact into components.

        Returns:
            Dictionary with WWR impact breakdown
        """
        base_effective_epe = self.base_exposure_profile.effective_epe

        # Compute WWR-adjusted EPE
        wwr_epe = self.wwr_adjusted_epe()

        # Effective EPE with WWR
        wwr_effective_epe = float(
            jnp.max(jnp.cumsum(wwr_epe) / jnp.arange(1, len(wwr_epe) + 1))
        )

        # WWR impact
        wwr_impact = wwr_effective_epe - base_effective_epe
        wwr_impact_pct = (wwr_impact / base_effective_epe) * 100

        return {
            "base_effective_epe": base_effective_epe,
            "wwr_adjusted_effective_epe": wwr_effective_epe,
            "wwr_impact": wwr_impact,
            "wwr_impact_percent": wwr_impact_pct,
            "wwr_correlation": self.wwr_correlation,
        }


__all__ = [
    "ExposureMetric",
    "ExposureProfile",
    "NettingSetExposure",
    "MultiNettingSetAggregator",
    "CollateralOptimizer",
    "WrongWayRiskAnalytics",
]
