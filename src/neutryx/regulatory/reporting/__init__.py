"""Regulatory compliance and reporting framework.

This module provides comprehensive regulatory reporting capabilities for:
- EMIR/Dodd-Frank trade reporting
- MiFID II/MiFIR transaction and reference data reporting
- Basel III/IV capital reporting
- Regulatory submissions and reconciliation
"""

from __future__ import annotations

from .emir import (
    EMIRLifecycleEvent,
    EMIRReconciliation,
    EMIRTradeReport,
    EMIRTradeReporter,
    EMIRValuationReport,
)
from .mifid import (
    BestExecutionAnalyzer,
    BestExecutionReport,
    MiFIDReferenceDataReport,
    MiFIDTransactionReport,
    MiFIDTransactionReporter,
)
from .basel_reporting import (
    BaselCapitalReport,
    BaselCapitalReporter,
    CVACapitalReport,
    FRTBCapitalReport,
    LeverageRatioReport,
    OperationalRiskReport,
)
from .report_engine import (
    RegulatoryReport,
    RegulatoryReportEngine,
    ReportStatus,
    ReportSubmission,
)

__all__ = [
    # EMIR/Dodd-Frank
    "EMIRLifecycleEvent",
    "EMIRReconciliation",
    "EMIRTradeReport",
    "EMIRTradeReporter",
    "EMIRValuationReport",
    # MiFID II
    "BestExecutionAnalyzer",
    "BestExecutionReport",
    "MiFIDReferenceDataReport",
    "MiFIDTransactionReport",
    "MiFIDTransactionReporter",
    # Basel III/IV
    "BaselCapitalReport",
    "BaselCapitalReporter",
    "CVACapitalReport",
    "FRTBCapitalReport",
    "LeverageRatioReport",
    "OperationalRiskReport",
    # Report Engine
    "RegulatoryReport",
    "RegulatoryReportEngine",
    "ReportStatus",
    "ReportSubmission",
]
