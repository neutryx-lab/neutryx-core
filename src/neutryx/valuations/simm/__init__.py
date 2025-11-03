"""ISDA SIMM (Standard Initial Margin Model) implementation.

This module implements the ISDA SIMM methodology for calculating risk-based
initial margin for uncleared OTC derivatives.

SIMM is based on:
- Delta and vega risk sensitivities
- Product class and risk factor bucketing
- Risk weights and correlations from ISDA SIMM calibration
- Cross-margin offsets between risk classes
"""

from neutryx.valuations.simm.calculator import (
    SIMMCalculator,
    SIMMResult,
    calculate_simm,
)
from neutryx.valuations.simm.risk_weights import (
    RiskClass,
    get_risk_weights,
    get_correlations,
)
from neutryx.valuations.simm.sensitivities import (
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType,
    bucket_sensitivities,
)

__all__ = [
    "SIMMCalculator",
    "SIMMResult",
    "calculate_simm",
    "RiskClass",
    "get_risk_weights",
    "get_correlations",
    "RiskFactorSensitivity",
    "RiskFactorType",
    "SensitivityType",
    "bucket_sensitivities",
]
