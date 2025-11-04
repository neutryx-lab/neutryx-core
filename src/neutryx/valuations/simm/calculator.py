"""SIMM calculation engine.

This module implements the ISDA SIMM calculation methodology for
computing risk-based initial margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import jax.numpy as jnp

from neutryx.valuations.simm.risk_weights import (
    RiskClass,
    apply_concentration_risk,
    get_concentration_threshold,
    get_correlations,
    get_vega_risk_weight,
    get_risk_weights,
)
from neutryx.valuations.simm.sensitivities import (
    BucketedSensitivities,
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType,
    bucket_sensitivities,
)


@dataclass
class SIMMResult:
    """SIMM calculation result.

    Attributes
    ----------
    total_im : float
        Total initial margin (all risk classes)
    im_by_risk_class : Dict[RiskClass, float]
        IM by risk class (before diversification)
    delta_im : float
        Delta margin component
    vega_im : float
        Vega margin component
    curvature_im : float
        Curvature margin component
    product_class_multiplier : float
        Product class multiplier applied
    """

    total_im: float
    im_by_risk_class: Dict[RiskClass, float]
    delta_im: float
    vega_im: float
    curvature_im: float = 0.0
    product_class_multiplier: float = 1.0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SIMMResult(total_im={self.total_im:,.0f}, "
            f"delta={self.delta_im:,.0f}, "
            f"vega={self.vega_im:,.0f}, "
            f"curvature={self.curvature_im:,.0f})"
        )


class SIMMCalculator:
    """ISDA SIMM calculator.

    This class implements the SIMM methodology for calculating initial margin
    from risk factor sensitivities.
    """

    def __init__(self, product_class_multiplier: float = 1.0):
        """Initialize SIMM calculator.

        Parameters
        ----------
        product_class_multiplier : float
            Product class multiplier (default 1.0)
            Some product classes may have regulatory multipliers
        """
        self.product_class_multiplier = product_class_multiplier

    def calculate(
        self,
        sensitivities: List[RiskFactorSensitivity],
    ) -> SIMMResult:
        """Calculate SIMM initial margin.

        Parameters
        ----------
        sensitivities : List[RiskFactorSensitivity]
            List of all risk factor sensitivities

        Returns
        -------
        SIMMResult
            SIMM calculation result
        """
        # Bucket sensitivities
        bucketed = bucket_sensitivities(sensitivities)

        # Calculate component margins
        delta_im_by_class: Dict[RiskClass, float] = {}
        vega_im_by_class: Dict[RiskClass, float] = {}
        curvature_im_by_class: Dict[RiskClass, float] = {}

        for (rf_type, sens_type), bucketed_sens in bucketed.items():
            risk_class = self._map_rf_type_to_risk_class(rf_type)
            if sens_type == SensitivityType.DELTA:
                im = self._calculate_margin(
                    risk_class,
                    bucketed_sens,
                    sensitivity_type=SensitivityType.DELTA,
                )
                delta_im_by_class[risk_class] = im
            elif sens_type == SensitivityType.VEGA:
                im = self._calculate_margin(
                    risk_class,
                    bucketed_sens,
                    sensitivity_type=SensitivityType.VEGA,
                )
                vega_im_by_class[risk_class] = im
            elif sens_type == SensitivityType.CURVATURE:
                im = self._calculate_margin(
                    risk_class,
                    bucketed_sens,
                    sensitivity_type=SensitivityType.CURVATURE,
                )
                curvature_im_by_class[risk_class] = im

        delta_im_total = sum(delta_im_by_class.values())
        vega_im_total = sum(vega_im_by_class.values())
        curvature_im_total = sum(curvature_im_by_class.values())

        # Combine all risk classes
        all_im_by_class: Dict[RiskClass, float] = {}
        risk_classes = set(delta_im_by_class) | set(vega_im_by_class) | set(curvature_im_by_class)
        for risk_class in risk_classes:
            delta_component = delta_im_by_class.get(risk_class, 0.0)
            vega_component = vega_im_by_class.get(risk_class, 0.0)
            curvature_component = curvature_im_by_class.get(risk_class, 0.0)
            all_im_by_class[risk_class] = delta_component + vega_component + curvature_component

        total_im = sum(all_im_by_class.values())

        # Apply product class multiplier
        total_im *= self.product_class_multiplier

        return SIMMResult(
            total_im=total_im,
            im_by_risk_class=all_im_by_class,
            delta_im=delta_im_total,
            vega_im=vega_im_total,
            curvature_im=curvature_im_total,
            product_class_multiplier=self.product_class_multiplier,
        )

    def _calculate_margin(
        self,
        risk_class: RiskClass,
        bucketed_sens: BucketedSensitivities,
        *,
        sensitivity_type: SensitivityType,
    ) -> float:
        """Calculate margin for a risk class and sensitivity type."""
        bucket_margins = []

        for bucket, sensitivities in bucketed_sens.bucket_sensitivities.items():
            ws = self._calculate_weighted_sensitivity(
                risk_class,
                bucket,
                sensitivities,
                sensitivity_type=sensitivity_type,
            )

            if sensitivity_type == SensitivityType.DELTA:
                threshold = get_concentration_threshold(risk_class, bucket)
                cr = apply_concentration_risk(ws, threshold)
                ws = jnp.sign(ws) * abs(ws) * cr
            else:
                cr = 1.0

            bucket_margins.append((bucket, ws))

        if len(bucket_margins) == 0:
            return 0.0

        rho = get_correlations(risk_class, within_bucket=False)

        total_variance = 0.0
        for i, (_, ws_i) in enumerate(bucket_margins):
            for j, (_, ws_j) in enumerate(bucket_margins):
                correlation = 1.0 if i == j else rho
                total_variance += ws_i * ws_j * correlation

        margin = jnp.sqrt(max(0.0, total_variance))

        if sensitivity_type == SensitivityType.VEGA:
            margin *= 0.5
        elif sensitivity_type == SensitivityType.CURVATURE:
            margin *= 0.3

        return float(margin)

    def _calculate_weighted_sensitivity(
        self,
        risk_class: RiskClass,
        bucket: str,
        sensitivities: List[RiskFactorSensitivity],
        *,
        sensitivity_type: SensitivityType,
    ) -> float:
        """Calculate weighted sensitivity for a bucket.

        Parameters
        ----------
        risk_class : RiskClass
            Risk class
        bucket : str
            Bucket identifier
        sensitivities : List[RiskFactorSensitivity]
            Sensitivities in the bucket

        sensitivity_type : SensitivityType
            Sensitivity type for risk-weight selection

        Returns
        -------
        float
            Weighted sensitivity (WS)
        """
        ws = 0.0
        for sens in sensitivities:
            if sensitivity_type == SensitivityType.VEGA:
                delta_rw = get_risk_weights(
                    risk_class=risk_class,
                    bucket=bucket,
                    tenor=sens.tenor,
                )
                rw = get_vega_risk_weight(risk_class, delta_rw)
            else:
                rw = get_risk_weights(
                    risk_class=risk_class,
                    bucket=bucket,
                    tenor=sens.tenor,
                )

            value = sens.sensitivity
            if sensitivity_type == SensitivityType.CURVATURE:
                value = abs(value)

            ws += value * rw

        return ws

    def _map_rf_type_to_risk_class(self, rf_type: RiskFactorType) -> RiskClass:
        """Map RiskFactorType to RiskClass.

        Parameters
        ----------
        rf_type : RiskFactorType
            Risk factor type

        Returns
        -------
        RiskClass
            Corresponding risk class
        """
        mapping = {
            RiskFactorType.IR: RiskClass.INTEREST_RATE,
            RiskFactorType.FX: RiskClass.FX,
            RiskFactorType.EQUITY: RiskClass.EQUITY,
            RiskFactorType.CREDIT_Q: RiskClass.CREDIT_QUALIFYING,
            RiskFactorType.CREDIT_NON_Q: RiskClass.CREDIT_NON_QUALIFYING,
            RiskFactorType.COMMODITY: RiskClass.COMMODITY,
        }
        return mapping.get(rf_type, RiskClass.EQUITY)


def calculate_simm(
    sensitivities: List[RiskFactorSensitivity],
    product_class_multiplier: float = 1.0,
) -> SIMMResult:
    """Calculate SIMM initial margin (convenience function).

    Parameters
    ----------
    sensitivities : List[RiskFactorSensitivity]
        List of risk factor sensitivities
    product_class_multiplier : float
        Product class multiplier (default 1.0)

    Returns
    -------
    SIMMResult
        SIMM calculation result
    """
    calculator = SIMMCalculator(product_class_multiplier=product_class_multiplier)
    return calculator.calculate(sensitivities)
