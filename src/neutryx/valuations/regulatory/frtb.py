"""FRTB Standardized Approach (Sensitivity-Based Method) calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import jax.numpy as jnp

from neutryx.valuations.simm.risk_weights import (
    RiskClass,
    get_risk_weights,
    get_vega_risk_weight,
)
from neutryx.valuations.simm.sensitivities import RiskFactorType, SensitivityType

# Default correlation parameters (simplified calibrations)
DEFAULT_WITHIN_BUCKET_CORRELATION: Dict[RiskClass, float] = {
    RiskClass.INTEREST_RATE: 0.99,
    RiskClass.CREDIT_QUALIFYING: 0.50,
    RiskClass.CREDIT_NON_QUALIFYING: 0.50,
    RiskClass.EQUITY: 0.20,
    RiskClass.COMMODITY: 0.30,
    RiskClass.FX: 0.60,
}

DEFAULT_CROSS_BUCKET_CORRELATION: Dict[RiskClass, float] = {
    RiskClass.INTEREST_RATE: 0.35,
    RiskClass.CREDIT_QUALIFYING: 0.40,
    RiskClass.CREDIT_NON_QUALIFYING: 0.40,
    RiskClass.EQUITY: 0.30,
    RiskClass.COMMODITY: 0.25,
    RiskClass.FX: 0.50,
}

CURVATURE_SCALING = 0.30


@dataclass(frozen=True)
class FRTBSensitivity:
    """Sensitivity input for the FRTB standardized approach."""

    risk_factor_type: RiskFactorType
    sensitivity_type: SensitivityType
    bucket: str
    risk_factor: str
    amount: float
    tenor: Optional[str] = None
    risk_weight_override: Optional[float] = None


@dataclass(frozen=True)
class FRTBChargeBreakdown:
    """Per-risk-class breakdown of FRTB capital charges."""

    delta: float = 0.0
    vega: float = 0.0
    curvature: float = 0.0

    @property
    def total(self) -> float:
        """Total capital charge for the risk class."""
        return self.delta + self.vega + self.curvature


@dataclass(frozen=True)
class FRTBResult:
    """Result of the FRTB standardized approach calculation."""

    total_capital: float
    delta_charge: float
    vega_charge: float
    curvature_charge: float
    charges_by_risk_class: Dict[RiskClass, FRTBChargeBreakdown]


class FRTBStandardizedApproach:
    """Compute capital charges under the FRTB standardized approach."""

    def __init__(
        self,
        *,
        within_bucket_correlation: Dict[RiskClass, float] | None = None,
        cross_bucket_correlation: Dict[RiskClass, float] | None = None,
    ) -> None:
        self._within_bucket_correlation = (
            within_bucket_correlation or DEFAULT_WITHIN_BUCKET_CORRELATION
        )
        self._cross_bucket_correlation = (
            cross_bucket_correlation or DEFAULT_CROSS_BUCKET_CORRELATION
        )

    def calculate(
        self,
        sensitivities: Iterable[FRTBSensitivity],
    ) -> FRTBResult:
        """Compute FRTB capital charges for the provided sensitivities."""
        grouped = self._group_sensitivities(sensitivities)

        delta_total = 0.0
        vega_total = 0.0
        curvature_total = 0.0
        charges_by_risk_class: Dict[RiskClass, FRTBChargeBreakdown] = {}

        for risk_class, sens_by_type in grouped.items():
            delta_charge = self._aggregate_charge(
                risk_class,
                sens_by_type.get(SensitivityType.DELTA, {}),
            )
            vega_charge = self._aggregate_charge(
                risk_class,
                sens_by_type.get(SensitivityType.VEGA, {}),
                sensitivity_type=SensitivityType.VEGA,
            )
            curvature_charge = self._aggregate_charge(
                risk_class,
                sens_by_type.get(SensitivityType.CURVATURE, {}),
                sensitivity_type=SensitivityType.CURVATURE,
            )

            charges_by_risk_class[risk_class] = FRTBChargeBreakdown(
                delta=delta_charge,
                vega=vega_charge,
                curvature=curvature_charge,
            )

            delta_total += delta_charge
            vega_total += vega_charge
            curvature_total += curvature_charge

        total_capital = delta_total + vega_total + curvature_total

        return FRTBResult(
            total_capital=total_capital,
            delta_charge=delta_total,
            vega_charge=vega_total,
            curvature_charge=curvature_total,
            charges_by_risk_class=charges_by_risk_class,
        )

    def _group_sensitivities(
        self,
        sensitivities: Iterable[FRTBSensitivity],
    ) -> Dict[RiskClass, Dict[SensitivityType, Dict[str, List[float]]]]:
        grouped: Dict[
            RiskClass,
            Dict[SensitivityType, Dict[str, List[float]]],
        ] = {}

        for sens in sensitivities:
            risk_class = _map_risk_factor_type(sens.risk_factor_type)
            weighted_amount = self._weighted_sensitivity(risk_class, sens)

            if weighted_amount == 0.0:
                continue

            class_dict = grouped.setdefault(risk_class, {})
            type_dict = class_dict.setdefault(sens.sensitivity_type, {})
            bucket_list = type_dict.setdefault(sens.bucket, [])
            bucket_list.append(weighted_amount)

        return grouped

    def _aggregate_charge(
        self,
        risk_class: RiskClass,
        bucketed_sensitivities: Dict[str, List[float]],
        *,
        sensitivity_type: SensitivityType = SensitivityType.DELTA,
    ) -> float:
        if not bucketed_sensitivities:
            return 0.0

        within_corr = self._within_bucket_correlation.get(risk_class, 0.5)
        cross_corr = self._cross_bucket_correlation.get(risk_class, 0.25)

        bucket_capitals: List[Tuple[float, float]] = []
        for bucket, weighted_values in bucketed_sensitivities.items():
            capital, net_exposure = _bucket_capital(weighted_values, within_corr)
            # Apply curvature scaling per BCBS final rules (simplified)
            if sensitivity_type == SensitivityType.CURVATURE:
                capital *= CURVATURE_SCALING
                net_exposure *= CURVATURE_SCALING
            bucket_capitals.append((capital, net_exposure))

        total_capital = _aggregate_buckets(bucket_capitals, cross_corr)

        if sensitivity_type == SensitivityType.VEGA:
            # Vega charge scaled relative to delta (per FRTB documentation)
            total_capital *= 0.5

        return float(total_capital)

    def _weighted_sensitivity(self, risk_class: RiskClass, sens: FRTBSensitivity) -> float:
        """Return weighted sensitivity for aggregation."""
        if sens.risk_weight_override is not None:
            risk_weight = sens.risk_weight_override
        else:
            if sens.sensitivity_type == SensitivityType.VEGA:
                delta_weight = get_risk_weights(
                    risk_class,
                    bucket=sens.bucket,
                    tenor=sens.tenor,
                )
                risk_weight = get_vega_risk_weight(risk_class, delta_weight)
            else:
                risk_weight = get_risk_weights(
                    risk_class,
                    bucket=sens.bucket,
                    tenor=sens.tenor,
                )

        weighted = sens.amount * risk_weight
        if sens.sensitivity_type == SensitivityType.CURVATURE:
            weighted = abs(sens.amount) * risk_weight
        return float(weighted)


def _bucket_capital(values: List[float], correlation: float) -> Tuple[float, float]:
    """Aggregate capital within a bucket using pairwise correlations."""
    exposures = jnp.asarray(values)
    if exposures.size == 0:
        return 0.0, 0.0

    total = 0.0
    for i in range(exposures.size):
        for j in range(exposures.size):
            corr = 1.0 if i == j else correlation
            total += exposures[i] * exposures[j] * corr

    sqrt_term = jnp.sqrt(jnp.maximum(total, 0.0))
    lambda_term = jnp.sum(jnp.maximum(-exposures, 0.0))

    capital = float(sqrt_term + lambda_term)
    net_exposure = float(exposures.sum())
    return capital, net_exposure


def _aggregate_buckets(
    bucket_capitals: List[Tuple[float, float]],
    correlation: float,
) -> float:
    """Aggregate bucket-level capital across the risk class."""
    if not bucket_capitals:
        return 0.0

    capitals = jnp.asarray([cap for cap, _ in bucket_capitals])
    nets = jnp.asarray([net for _, net in bucket_capitals])

    total = 0.0
    for i in range(capitals.size):
        for j in range(capitals.size):
            corr = 1.0 if i == j else correlation
            total += capitals[i] * capitals[j] * corr

    sqrt_term = jnp.sqrt(jnp.maximum(total, 0.0))
    lambda_term = jnp.sum(jnp.maximum(-nets, 0.0))

    return float(sqrt_term + lambda_term)


def _map_risk_factor_type(risk_factor_type: RiskFactorType) -> RiskClass:
    """Map SIMM risk factor types to FRTB risk classes."""
    mapping = {
        RiskFactorType.IR: RiskClass.INTEREST_RATE,
        RiskFactorType.FX: RiskClass.FX,
        RiskFactorType.EQUITY: RiskClass.EQUITY,
        RiskFactorType.COMMODITY: RiskClass.COMMODITY,
        RiskFactorType.CREDIT_Q: RiskClass.CREDIT_QUALIFYING,
        RiskFactorType.CREDIT_NON_Q: RiskClass.CREDIT_NON_QUALIFYING,
    }
    return mapping.get(risk_factor_type, RiskClass.EQUITY)
