"""Regulatory capital calculations (FRTB, SA-CCR, Basel III, DRC, RRAO)."""

from __future__ import annotations

from .basel import (
    BaselCapitalCalculator,
    BaselCapitalInputs,
    BaselCapitalResult,
    BaselExposure,
)
from .engine import RegulatoryCapitalEngine, RegulatoryCapitalSummary
from .frtb import (
    FRTBChargeBreakdown,
    FRTBResult,
    FRTBSensitivity,
    FRTBStandardizedApproach,
)
from .frtb_drc import (
    CreditRating,
    DefaultExposure,
    DRCResult,
    FRTBDefaultRiskCharge,
    SecuritizedExposure,
    Sector,
    SecuritizationType,
    Seniority,
    calculate_lgd_from_recovery,
    map_external_rating_to_cqs,
)
from .frtb_rrao import (
    ExoticUnderlying,
    FRTBResidualRiskAddOn,
    LiquidityClass,
    PayoffComplexity,
    RRAOExposure,
    RRAOResult,
    calculate_basis_risk_rrao,
    classify_payoff_complexity,
    estimate_hedge_effectiveness,
)
from .saccr import (
    AssetClass,
    SACCRCalculator,
    SACCRResult,
    SACCRTrade,
)

__all__ = [
    # SA-CCR
    "AssetClass",
    "SACCRCalculator",
    "SACCRResult",
    "SACCRTrade",
    # Basel III
    "BaselCapitalCalculator",
    "BaselCapitalInputs",
    "BaselCapitalResult",
    "BaselExposure",
    # FRTB Standardized Approach
    "FRTBChargeBreakdown",
    "FRTBResult",
    "FRTBSensitivity",
    "FRTBStandardizedApproach",
    # FRTB Default Risk Charge (DRC)
    "CreditRating",
    "DefaultExposure",
    "DRCResult",
    "FRTBDefaultRiskCharge",
    "SecuritizedExposure",
    "Sector",
    "SecuritizationType",
    "Seniority",
    "calculate_lgd_from_recovery",
    "map_external_rating_to_cqs",
    # FRTB Residual Risk Add-On (RRAO)
    "ExoticUnderlying",
    "FRTBResidualRiskAddOn",
    "LiquidityClass",
    "PayoffComplexity",
    "RRAOExposure",
    "RRAOResult",
    "calculate_basis_risk_rrao",
    "classify_payoff_complexity",
    "estimate_hedge_effectiveness",
    # Regulatory Engine
    "RegulatoryCapitalEngine",
    "RegulatoryCapitalSummary",
]
