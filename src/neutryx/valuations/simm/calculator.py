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
            f"vega={self.vega_im:,.0f})"
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

        # Calculate delta margin
        delta_im_by_class = {}
        for (rf_type, sens_type), bucketed_sens in bucketed.items():
            if sens_type == SensitivityType.DELTA:
                risk_class = self._map_rf_type_to_risk_class(rf_type)
                im = self._calculate_delta_margin(risk_class, bucketed_sens)
                delta_im_by_class[risk_class] = im

        # Calculate vega margin
        vega_im_by_class = {}
        for (rf_type, sens_type), bucketed_sens in bucketed.items():
            if sens_type == SensitivityType.VEGA:
                risk_class = self._map_rf_type_to_risk_class(rf_type)
                im = self._calculate_vega_margin(risk_class, bucketed_sens)
                vega_im_by_class[risk_class] = im

        # Aggregate across risk classes
        delta_im_total = sum(delta_im_by_class.values())
        vega_im_total = sum(vega_im_by_class.values())

        # Combine all risk classes
        all_im_by_class = {}
        for risk_class in set(list(delta_im_by_class.keys()) + list(vega_im_by_class.keys())):
            delta_component = delta_im_by_class.get(risk_class, 0.0)
            vega_component = vega_im_by_class.get(risk_class, 0.0)
            # Simple sum (actual SIMM has more complex aggregation)
            all_im_by_class[risk_class] = delta_component + vega_component

        # Apply cross-risk-class aggregation (simplified: sum)
        # Full SIMM uses correlation matrix for cross-risk aggregation
        total_im = sum(all_im_by_class.values())

        # Apply product class multiplier
        total_im *= self.product_class_multiplier

        return SIMMResult(
            total_im=total_im,
            im_by_risk_class=all_im_by_class,
            delta_im=delta_im_total,
            vega_im=vega_im_total,
            product_class_multiplier=self.product_class_multiplier,
        )

    def _calculate_delta_margin(
        self,
        risk_class: RiskClass,
        bucketed_sens: BucketedSensitivities,
    ) -> float:
        """Calculate delta margin for a risk class.

        Parameters
        ----------
        risk_class : RiskClass
            Risk class
        bucketed_sens : BucketedSensitivities
            Bucketed sensitivities

        Returns
        -------
        float
            Delta margin for the risk class
        """
        # Calculate weighted sensitivities by bucket
        bucket_margins = []

        for bucket, sensitivities in bucketed_sens.bucket_sensitivities.items():
            # Calculate weighted sensitivity for bucket
            ws = self._calculate_weighted_sensitivity(
                risk_class, bucket, sensitivities
            )

            # Apply concentration risk
            threshold = get_concentration_threshold(risk_class, bucket)
            cr = apply_concentration_risk(ws, threshold)

            # Bucket margin
            bucket_margin = abs(ws) * cr
            bucket_margins.append((bucket, bucket_margin, ws))

        # Aggregate across buckets with correlation
        if len(bucket_margins) == 0:
            return 0.0

        # Simplified aggregation: sqrt(sum of squares with correlation)
        rho = get_correlations(risk_class, within_bucket=False)  # Cross-bucket correlation

        total_variance = 0.0
        for i, (bucket_i, margin_i, ws_i) in enumerate(bucket_margins):
            for j, (bucket_j, margin_j, ws_j) in enumerate(bucket_margins):
                correlation = 1.0 if i == j else rho
                total_variance += ws_i * ws_j * correlation

        delta_margin = jnp.sqrt(max(0.0, total_variance))
        return float(delta_margin)

    def _calculate_weighted_sensitivity(
        self,
        risk_class: RiskClass,
        bucket: str,
        sensitivities: List[RiskFactorSensitivity],
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

        Returns
        -------
        float
            Weighted sensitivity (WS)

        Notes
        -----
        WS = sum(sensitivity * risk_weight)
        """
        ws = 0.0
        for sens in sensitivities:
            rw = get_risk_weights(
                risk_class=risk_class,
                bucket=bucket,
                tenor=sens.tenor,
            )
            ws += sens.sensitivity * rw

        return ws

    def _calculate_vega_margin(
        self,
        risk_class: RiskClass,
        bucketed_sens: BucketedSensitivities,
    ) -> float:
        """Calculate vega margin for a risk class.

        Parameters
        ----------
        risk_class : RiskClass
            Risk class
        bucketed_sens : BucketedSensitivities
            Bucketed vega sensitivities

        Returns
        -------
        float
            Vega margin

        Notes
        -----
        Vega margin follows similar aggregation to delta, but with
        vega-specific risk weights.
        """
        # Simplified: use delta margin calculation with vega multiplier
        # Full implementation would have vega-specific risk weights
        vega_margin = self._calculate_delta_margin(risk_class, bucketed_sens)

        # Vega multiplier (simplified)
        vega_multiplier = 0.50
        return vega_margin * vega_multiplier

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
