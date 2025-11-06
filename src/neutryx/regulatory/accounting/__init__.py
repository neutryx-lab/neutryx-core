"""Accounting standards compliance framework.

This module provides comprehensive accounting standards compliance for:
- IFRS 9: Financial Instruments (classification, measurement, ECL, hedge accounting)
- IFRS 13: Fair Value Measurement (hierarchy, techniques, disclosures)
- Valuation adjustments (XVA: CVA, DVA, FVA, MVA, KVA)
- Hedge effectiveness testing
- Disclosure requirements
"""

from __future__ import annotations

from .ifrs9 import (
    ECLModel,
    ECLResult,
    ECLStage,
    FinancialInstrumentCategory,
    HedgeEffectivenessTest,
    HedgeRelationship,
    HedgeType,
    IFRS9Classifier,
)
from .ifrs13 import (
    FairValueHierarchy,
    FairValueInput,
    FairValueMeasurement,
    IFRS13Disclosure,
    ValuationTechnique,
)
from .xva import (
    CVACalculator,
    DVACalculator,
    FVACalculator,
    KVACalculator,
    MVACalculator,
    ValuationAdjustment,
    XVAEngine,
)

__all__ = [
    # IFRS 9
    "ECLModel",
    "ECLResult",
    "ECLStage",
    "FinancialInstrumentCategory",
    "HedgeEffectivenessTest",
    "HedgeRelationship",
    "HedgeType",
    "IFRS9Classifier",
    # IFRS 13
    "FairValueHierarchy",
    "FairValueInput",
    "FairValueMeasurement",
    "IFRS13Disclosure",
    "ValuationTechnique",
    # XVA
    "CVACalculator",
    "DVACalculator",
    "FVACalculator",
    "KVACalculator",
    "MVACalculator",
    "ValuationAdjustment",
    "XVAEngine",
]
