"""Standardised Approach for Counterparty Credit Risk (SA-CCR)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Tuple

import jax.numpy as jnp


class AssetClass(str, Enum):
    """Supported SA-CCR asset classes."""

    INTEREST_RATE = "IR"
    FX = "FX"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"


DEFAULT_SUPERVISORY_FACTORS: Dict[AssetClass, float] = {
    AssetClass.INTEREST_RATE: 0.005,
    AssetClass.FX: 0.04,
    AssetClass.CREDIT: 0.1,
    AssetClass.EQUITY: 0.32,
    AssetClass.COMMODITY: 0.18,
}

DEFAULT_CORRELATIONS: Dict[AssetClass, float] = {
    AssetClass.INTEREST_RATE: 0.7,
    AssetClass.FX: 0.6,
    AssetClass.CREDIT: 0.5,
    AssetClass.EQUITY: 0.5,
    AssetClass.COMMODITY: 0.4,
}


@dataclass(frozen=True)
class SACCRTrade:
    """Representation of a single trade for SA-CCR aggregation."""

    asset_class: AssetClass
    notional: float
    direction: int = 1
    maturity: float = 1.0
    supervisory_duration: float | None = None
    supervisory_factor: float | None = None
    hedging_set: str | None = None

    def effective_notional(self) -> float:
        """Return the effective notional incorporating duration and direction."""
        duration = (
            self.supervisory_duration
            if self.supervisory_duration is not None
            else _supervisory_duration(self.maturity)
        )
        factor = (
            self.supervisory_factor
            if self.supervisory_factor is not None
            else DEFAULT_SUPERVISORY_FACTORS[self.asset_class]
        )
        return self.direction * self.notional * duration * factor

    def hedging_set_key(self) -> str:
        """Return the hedging set identifier used for aggregation."""
        if self.hedging_set:
            return self.hedging_set
        if self.asset_class == AssetClass.INTEREST_RATE:
            # For IR, currency acts as the natural hedging set.
            return "IR-General"
        return self.asset_class.value


@dataclass(frozen=True)
class SACCRResult:
    """SA-CCR capital components."""

    replacement_cost: float
    addon: float
    multiplier: float
    potential_future_exposure: float
    ead: float
    addon_by_asset_class: Dict[AssetClass, float]

    def capital_requirement(self, risk_weight: float, capital_ratio: float = 0.08) -> float:
        """Regulatory capital charge derived from EAD."""
        return self.ead * risk_weight * capital_ratio


class SACCRCalculator:
    """Compute SA-CCR exposure components for a netting set."""

    def __init__(
        self,
        *,
        alpha: float = 1.4,
        multiplier_floor: float = 0.05,
        asset_class_correlations: Dict[AssetClass, float] | None = None,
    ) -> None:
        self.alpha = alpha
        self.multiplier_floor = multiplier_floor
        self.asset_class_correlations = asset_class_correlations or DEFAULT_CORRELATIONS

    def calculate(
        self,
        trades: Iterable[SACCRTrade],
        *,
        mark_to_market: float = 0.0,
        collateral: float = 0.0,
    ) -> SACCRResult:
        trades = list(trades)
        addon_by_asset_class = self._aggregate_addons(trades)

        total_addon = float(sum(addon_by_asset_class.values()))
        replacement_cost = max(mark_to_market - collateral, 0.0)
        multiplier = self._multiplier(mark_to_market, collateral, total_addon)
        pfe = multiplier * total_addon
        ead = self.alpha * (replacement_cost + pfe)

        return SACCRResult(
            replacement_cost=replacement_cost,
            addon=total_addon,
            multiplier=multiplier,
            potential_future_exposure=pfe,
            ead=ead,
            addon_by_asset_class=addon_by_asset_class,
        )

    def _aggregate_addons(self, trades: List[SACCRTrade]) -> Dict[AssetClass, float]:
        addon_by_asset_class: Dict[AssetClass, float] = {}
        exposures_by_asset_class: Dict[AssetClass, Dict[str, float]] = {}

        for trade in trades:
            en = trade.effective_notional()
            if en == 0.0:
                continue
            hedging_sets = exposures_by_asset_class.setdefault(trade.asset_class, {})
            key = trade.hedging_set_key()
            hedging_sets[key] = hedging_sets.get(key, 0.0) + en

        for asset_class, exposures in exposures_by_asset_class.items():
            rho = self.asset_class_correlations.get(asset_class, 0.5)
            addon = _correlated_sum(list(exposures.values()), rho)
            addon_by_asset_class[asset_class] = addon

        return addon_by_asset_class

    def _multiplier(self, mtm: float, collateral: float, addon: float) -> float:
        """SA-CCR multiplier that accounts for collateralisation."""
        if addon <= 0.0:
            return 1.0

        floor = self.multiplier_floor
        numerator = mtm - collateral
        exponential_term = jnp.exp(numerator / (2.0 * (1.0 - floor) * addon))
        multiplier = floor + (1.0 - floor) * exponential_term
        return float(jnp.minimum(1.0, multiplier))


def _supervisory_duration(maturity: float) -> float:
    """Basel supervisory duration approximation for SA-CCR."""
    maturity = max(maturity, 1e-8)
    return (1.0 - jnp.exp(-maturity)) / maturity


def _correlated_sum(values: List[float], correlation: float) -> float:
    """Aggregate exposures using the supervisory correlation."""
    arr = jnp.asarray(values)
    if arr.size == 0:
        return 0.0

    total = 0.0
    for i in range(arr.size):
        for j in range(arr.size):
            corr = 1.0 if i == j else correlation
            total += arr[i] * arr[j] * corr

    return float(jnp.sqrt(jnp.maximum(total, 0.0)))
