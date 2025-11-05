"""Margin calculation for OTC derivatives.

This module provides calculations for:
- Variation Margin (VM): Mark-to-market collateral
- Initial Margin (IM): Risk-based collateral (SIMM and other models)
- UMR Compliance: BCBS-IOSCO uncleared margin rules
- CSA Management: Credit Support Annex terms and margin calls
- Collateral Transformation: Optimal collateral selection and cost minimization
"""

from neutryx.valuations.margin.initial_margin import (
    InitialMarginModel,
    calculate_grid_im,
    calculate_schedule_im,
)
from neutryx.valuations.margin.variation_margin import (
    calculate_variation_margin,
    calculate_vm_call,
)
from neutryx.valuations.margin.umr_compliance import (
    AANACalculation,
    CollateralMovement,
    CollateralType,
    CSAManager,
    CSAPortfolio,
    CSATerms,
    CustodianAccount,
    CustodianInterface,
    MarginCall,
    MarginType,
    UMRComplianceChecker,
    UMRPhase,
    UMRThresholds,
    generate_margin_report,
)
from neutryx.valuations.margin.collateral_transformation import (
    CollateralHolding,
    CollateralPortfolio,
    CollateralSelection,
    ConcentrationOptimizedStrategy,
    GreedyLowestCostStrategy,
    OptimalMixStrategy,
    TransformationCost,
    TransformationResult,
    TransformationStrategy,
    calculate_funding_cost,
    calculate_fx_conversion_cost,
    calculate_haircut_adjusted_value,
)

__all__ = [
    # Variation Margin
    "calculate_variation_margin",
    "calculate_vm_call",
    # Initial Margin
    "InitialMarginModel",
    "calculate_grid_im",
    "calculate_schedule_im",
    # UMR Compliance
    "UMRPhase",
    "MarginType",
    "CollateralType",
    "AANACalculation",
    "MarginCall",
    "UMRThresholds",
    "UMRComplianceChecker",
    "generate_margin_report",
    # CSA Management
    "CSATerms",
    "CSAPortfolio",
    "CSAManager",
    # Custodian Integration
    "CustodianAccount",
    "CustodianInterface",
    "CollateralMovement",
    # Collateral Transformation
    "CollateralHolding",
    "CollateralPortfolio",
    "CollateralSelection",
    "TransformationCost",
    "TransformationResult",
    "TransformationStrategy",
    "GreedyLowestCostStrategy",
    "OptimalMixStrategy",
    "ConcentrationOptimizedStrategy",
    "calculate_haircut_adjusted_value",
    "calculate_fx_conversion_cost",
    "calculate_funding_cost",
]
