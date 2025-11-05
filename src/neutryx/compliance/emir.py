"""EMIR (European Market Infrastructure Regulation) and Dodd-Frank trade reporting.

This module implements trade reporting requirements for:
- EMIR (EU Regulation 648/2012)
- Dodd-Frank Act (US)
- Trade repository reporting
- Lifecycle event reporting
- Valuation reporting
- Reconciliation and dispute resolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from .report_engine import RegulatoryReport, ReportStatus, ReportType


class LifecycleEventType(str, Enum):
    """EMIR lifecycle event types."""

    NEW_TRADE = "NEWT"
    MODIFICATION = "MODI"
    TERMINATION = "TERM"
    COMPRESSION = "COMP"
    POSITION_COMPONENT = "POSC"
    EARLY_TERMINATION = "ETRM"
    NOVATION = "NOVA"
    EXERCISE = "EXER"
    ALLOCATION = "ALOC"
    CLEARING = "CLRG"
    ERROR = "EROR"
    CORRECTION = "CORR"


class ProductClassification(str, Enum):
    """EMIR product classifications."""

    INTEREST_RATE = "IRSW"  # Interest Rate Swaps
    FOREIGN_EXCHANGE = "FXSW"  # FX Swaps
    EQUITY = "EQSW"  # Equity Swaps
    COMMODITY = "COSW"  # Commodity Swaps
    CREDIT = "CRDS"  # Credit Derivatives
    OTHER = "OTHR"


class ClearingStatus(str, Enum):
    """Trade clearing status."""

    CLEARED = "C"
    NON_CLEARED = "N"
    INTENT_TO_CLEAR = "I"


@dataclass
class EMIRTradeReport(RegulatoryReport):
    """EMIR trade report for submission to trade repository."""

    # Trade identification
    unique_trade_identifier: str = ""  # UTI
    unique_product_identifier: str = ""  # UPI
    prior_uti: Optional[str] = None

    # Counterparties
    reporting_counterparty_lei: str = ""
    other_counterparty_lei: str = ""
    broker_lei: Optional[str] = None
    clearing_member_lei: Optional[str] = None
    beneficiary_lei: Optional[str] = None

    # Trade details
    execution_timestamp: Optional[datetime] = None
    value_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    product_classification: ProductClassification = ProductClassification.OTHER
    product_name: str = ""
    asset_class: str = ""

    # Economics
    notional_amount: Decimal = Decimal("0")
    notional_currency: str = "USD"
    price: Optional[Decimal] = None
    price_currency: Optional[str] = None
    fixed_rate: Optional[Decimal] = None
    floating_rate_index: Optional[str] = None
    spread: Optional[Decimal] = None

    # Clearing and settlement
    clearing_status: ClearingStatus = ClearingStatus.NON_CLEARED
    ccp_lei: Optional[str] = None
    intragroup: bool = False
    confirmation_timestamp: Optional[datetime] = None

    # Collateral
    collateralization: bool = False
    collateral_portfolio_code: Optional[str] = None

    # Lifecycle
    lifecycle_event: LifecycleEventType = LifecycleEventType.NEW_TRADE

    def __post_init__(self):
        """Initialize report type and default data."""
        self.report_type = ReportType.EMIR_TRADE
        if not self.data:
            self.data = self._build_data_dict()

    def _build_data_dict(self) -> Dict[str, Any]:
        """Build the data dictionary for reporting."""
        return {
            "uti": self.unique_trade_identifier,
            "upi": self.unique_product_identifier,
            "prior_uti": self.prior_uti,
            "reporting_counterparty": self.reporting_counterparty_lei,
            "other_counterparty": self.other_counterparty_lei,
            "broker": self.broker_lei,
            "execution_timestamp": self.execution_timestamp.isoformat() if self.execution_timestamp else None,
            "value_date": self.value_date.isoformat() if self.value_date else None,
            "maturity_date": self.maturity_date.isoformat() if self.maturity_date else None,
            "product_classification": self.product_classification.value,
            "product_name": self.product_name,
            "asset_class": self.asset_class,
            "notional_amount": str(self.notional_amount),
            "notional_currency": self.notional_currency,
            "price": str(self.price) if self.price else None,
            "clearing_status": self.clearing_status.value,
            "ccp": self.ccp_lei,
            "intragroup": self.intragroup,
            "collateralization": self.collateralization,
            "lifecycle_event": self.lifecycle_event.value,
        }

    def validate(self) -> bool:
        """Validate EMIR trade report."""
        super().validate()

        # UTI is mandatory
        if not self.unique_trade_identifier:
            self.errors.append("Unique Trade Identifier (UTI) is required")

        # LEI validation (20 characters)
        if self.reporting_counterparty_lei and len(self.reporting_counterparty_lei) != 20:
            self.errors.append(f"Invalid reporting counterparty LEI: {self.reporting_counterparty_lei}")

        if self.other_counterparty_lei and len(self.other_counterparty_lei) != 20:
            self.errors.append(f"Invalid other counterparty LEI: {self.other_counterparty_lei}")

        # Execution timestamp required for new trades
        if self.lifecycle_event == LifecycleEventType.NEW_TRADE and not self.execution_timestamp:
            self.errors.append("Execution timestamp required for new trades")

        # Maturity date validation
        if self.value_date and self.maturity_date and self.maturity_date < self.value_date:
            self.errors.append("Maturity date must be after value date")

        # Notional validation
        if self.notional_amount <= 0:
            self.errors.append("Notional amount must be positive")

        # Currency code validation (ISO 4217)
        if len(self.notional_currency) != 3:
            self.errors.append(f"Invalid currency code: {self.notional_currency}")

        # Clearing validation
        if self.clearing_status == ClearingStatus.CLEARED and not self.ccp_lei:
            self.errors.append("CCP LEI required for cleared trades")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to ISO 20022 XML format for EMIR reporting."""
        # Simplified XML generation (production would use proper XML library)
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<Document xmlns="urn:iso:std:iso:20022:tech:xsd:auth.036.001.02">',
            "  <DerivsTradRpt>",
            "    <RptHdr>",
            f"      <RptgNtty><LEI>{self.reporting_counterparty_lei}</LEI></RptgNtty>",
            f"      <RptgPrd><Dt>{self.reporting_date.date().isoformat()}</Dt></RptgPrd>",
            "    </RptHdr>",
            "    <TradData>",
            f"      <CtrPty1><LEI>{self.reporting_counterparty_lei}</LEI></CtrPty1>",
            f"      <CtrPty2><LEI>{self.other_counterparty_lei}</LEI></CtrPty2>",
            "      <TxDtls>",
            f"        <UnqTradIdr>{self.unique_trade_identifier}</UnqTradIdr>",
            f"        <UnqPdctIdr>{self.unique_product_identifier}</UnqPdctIdr>",
            f"        <ExctnTmstmp>{self.execution_timestamp.isoformat() if self.execution_timestamp else ''}</ExctnTmstmp>",
            f"        <ValDt>{self.value_date.date().isoformat() if self.value_date else ''}</ValDt>",
            f"        <MtrtyDt>{self.maturity_date.date().isoformat() if self.maturity_date else ''}</MtrtyDt>",
            f"        <PdctClssfctn>{self.product_classification.value}</PdctClssfctn>",
            f"        <NotlAmt Ccy=\"{self.notional_currency}\">{self.notional_amount}</NotlAmt>",
            f"        <ClrSts>{self.clearing_status.value}</ClrSts>",
            f"        <EvtTp>{self.lifecycle_event.value}</EvtTp>",
            "      </TxDtls>",
            "    </TradData>",
            "  </DerivsTradRpt>",
            "</Document>",
        ]
        return "\n".join(xml_parts)


@dataclass
class EMIRLifecycleEvent(RegulatoryReport):
    """EMIR lifecycle event report."""

    unique_trade_identifier: str = ""
    event_type: LifecycleEventType = LifecycleEventType.MODIFICATION
    event_timestamp: Optional[datetime] = None
    prior_uti: Optional[str] = None
    new_uti: Optional[str] = None

    # Modified fields
    modified_notional: Optional[Decimal] = None
    modified_maturity: Optional[datetime] = None
    termination_amount: Optional[Decimal] = None
    termination_currency: Optional[str] = None

    def __post_init__(self):
        """Initialize lifecycle event report."""
        self.report_type = ReportType.EMIR_LIFECYCLE

    def validate(self) -> bool:
        """Validate lifecycle event report."""
        super().validate()

        if not self.unique_trade_identifier:
            self.errors.append("UTI is required for lifecycle events")

        if not self.event_timestamp:
            self.errors.append("Event timestamp is required")

        # Event-specific validation
        if self.event_type == LifecycleEventType.TERMINATION:
            if self.termination_amount is None:
                self.warnings.append("Termination amount not specified")

        if self.event_type == LifecycleEventType.NOVATION:
            if not self.new_uti:
                self.errors.append("New UTI required for novation")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert lifecycle event to XML."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<LifecycleEvent>
  <UTI>{self.unique_trade_identifier}</UTI>
  <EventType>{self.event_type.value}</EventType>
  <EventTimestamp>{self.event_timestamp.isoformat() if self.event_timestamp else ''}</EventTimestamp>
  <PriorUTI>{self.prior_uti or ''}</PriorUTI>
  <NewUTI>{self.new_uti or ''}</NewUTI>
</LifecycleEvent>"""


@dataclass
class EMIRValuationReport(RegulatoryReport):
    """EMIR valuation report for mark-to-market/mark-to-model."""

    unique_trade_identifier: str = ""
    valuation_date: Optional[datetime] = None
    valuation_amount: Decimal = Decimal("0")
    valuation_currency: str = "USD"
    valuation_type: str = "MTOM"  # MTOM or MKTM
    valuation_timestamp: Optional[datetime] = None

    # For cleared trades
    ccp_valuation: Optional[Decimal] = None
    ccp_variation_margin: Optional[Decimal] = None

    def __post_init__(self):
        """Initialize valuation report."""
        self.report_type = ReportType.EMIR_VALUATION

    def validate(self) -> bool:
        """Validate valuation report."""
        super().validate()

        if not self.unique_trade_identifier:
            self.errors.append("UTI is required")

        if not self.valuation_date:
            self.errors.append("Valuation date is required")

        if self.valuation_type not in ["MTOM", "MKTM"]:
            self.errors.append(f"Invalid valuation type: {self.valuation_type}")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert valuation to XML."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<ValuationReport>
  <UTI>{self.unique_trade_identifier}</UTI>
  <ValuationDate>{self.valuation_date.date().isoformat() if self.valuation_date else ''}</ValuationDate>
  <ValuationAmount Ccy="{self.valuation_currency}">{self.valuation_amount}</ValuationAmount>
  <ValuationType>{self.valuation_type}</ValuationType>
  <ValuationTimestamp>{self.valuation_timestamp.isoformat() if self.valuation_timestamp else ''}</ValuationTimestamp>
</ValuationReport>"""


@dataclass
class EMIRReconciliation:
    """EMIR reconciliation report for portfolio reconciliation."""

    portfolio_id: str
    reconciliation_date: datetime
    counterparty_lei: str
    total_trades: int = 0
    matched_trades: int = 0
    unmatched_trades: int = 0
    disputed_trades: int = 0
    outstanding_disputes: List[str] = field(default_factory=list)
    reconciliation_frequency: str = "DAILY"  # DAILY, WEEKLY, QUARTERLY

    def match_rate(self) -> float:
        """Calculate reconciliation match rate."""
        if self.total_trades == 0:
            return 1.0
        return self.matched_trades / self.total_trades

    def requires_dispute_resolution(self) -> bool:
        """Check if dispute resolution is required."""
        # EMIR requires dispute resolution if match rate < 95%
        return self.match_rate() < 0.95 or len(self.outstanding_disputes) > 0


class EMIRTradeReporter:
    """EMIR trade reporting engine."""

    def __init__(self, reporting_lei: str, trade_repository: str = "DTCC-GTR"):
        """Initialize EMIR trade reporter.

        Parameters
        ----------
        reporting_lei : str
            Legal Entity Identifier of reporting entity
        trade_repository : str
            Target trade repository (e.g., "DTCC-GTR", "Regis-TR")
        """
        self.reporting_lei = reporting_lei
        self.trade_repository = trade_repository
        self.reports: Dict[str, EMIRTradeReport] = {}

    def create_trade_report(
        self,
        uti: str,
        counterparty_lei: str,
        **kwargs,
    ) -> EMIRTradeReport:
        """Create a new EMIR trade report.

        Parameters
        ----------
        uti : str
            Unique Trade Identifier
        counterparty_lei : str
            Other counterparty LEI
        **kwargs
            Additional trade details

        Returns
        -------
        EMIRTradeReport
            Created trade report
        """
        report = EMIRTradeReport(
            unique_trade_identifier=uti,
            reporting_counterparty_lei=self.reporting_lei,
            other_counterparty_lei=counterparty_lei,
            **kwargs,
        )
        self.reports[uti] = report
        return report

    def create_lifecycle_event(
        self,
        uti: str,
        event_type: LifecycleEventType,
        **kwargs,
    ) -> EMIRLifecycleEvent:
        """Create a lifecycle event report.

        Parameters
        ----------
        uti : str
            Unique Trade Identifier
        event_type : LifecycleEventType
            Type of lifecycle event
        **kwargs
            Event details

        Returns
        -------
        EMIRLifecycleEvent
            Created lifecycle event report
        """
        return EMIRLifecycleEvent(
            unique_trade_identifier=uti,
            event_type=event_type,
            event_timestamp=datetime.utcnow(),
            **kwargs,
        )

    def create_valuation_report(
        self,
        uti: str,
        valuation_amount: Decimal,
        **kwargs,
    ) -> EMIRValuationReport:
        """Create a valuation report.

        Parameters
        ----------
        uti : str
            Unique Trade Identifier
        valuation_amount : Decimal
            Valuation amount
        **kwargs
            Valuation details

        Returns
        -------
        EMIRValuationReport
            Created valuation report
        """
        return EMIRValuationReport(
            unique_trade_identifier=uti,
            valuation_amount=valuation_amount,
            valuation_date=datetime.utcnow(),
            valuation_timestamp=datetime.utcnow(),
            **kwargs,
        )

    def batch_report(
        self,
        reports: List[EMIRTradeReport],
    ) -> Dict[str, Any]:
        """Submit batch of trade reports.

        Parameters
        ----------
        reports : list of EMIRTradeReport
            Reports to submit

        Returns
        -------
        dict
            Batch submission results
        """
        results = {
            "total": len(reports),
            "validated": 0,
            "failed": 0,
            "errors": [],
        }

        for report in reports:
            if report.validate():
                results["validated"] += 1
                report.status = ReportStatus.VALIDATED
            else:
                results["failed"] += 1
                results["errors"].append({
                    "uti": report.unique_trade_identifier,
                    "errors": report.errors,
                })

        return results
