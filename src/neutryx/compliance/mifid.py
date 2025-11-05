"""MiFID II/MiFIR (Markets in Financial Instruments Directive) transaction reporting.

This module implements transaction reporting requirements for:
- MiFID II/MiFIR transaction reporting (RTS 22)
- Reference data reporting (RTS 23)
- Best execution analysis
- Position reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from .report_engine import RegulatoryReport, ReportStatus, ReportType


class TransactionType(str, Enum):
    """MiFID II transaction types."""

    BUY = "BUY"
    SELL = "SELL"


class VenueType(str, Enum):
    """Trading venue types."""

    REGULATED_MARKET = "XOFF"  # Regulated market
    MTF = "MTFF"  # Multilateral Trading Facility
    OTF = "OTFF"  # Organised Trading Facility
    SI = "SINT"  # Systematic Internaliser
    OTC = "XXXX"  # Over-the-counter


class InstrumentClassification(str, Enum):
    """Financial instrument classifications."""

    EQUITY = "EQUI"
    BOND = "BOND"
    DERIVATIVE = "DERV"
    STRUCTURED = "STRU"
    COMMODITY = "COMM"
    CURRENCY = "CURR"


class CapacityType(str, Enum):
    """Trading capacity."""

    PRINCIPAL = "PRIN"
    AGENT = "AOTC"  # Any other capacity
    MATCHED_PRINCIPAL = "MTCP"


@dataclass
class MiFIDTransactionReport(RegulatoryReport):
    """MiFID II transaction report (RTS 22)."""

    # Transaction identification
    transaction_reference_number: str = ""
    trading_date_time: Optional[datetime] = None
    trading_capacity: CapacityType = CapacityType.PRINCIPAL
    venue: VenueType = VenueType.OTC

    # Instrument details
    instrument_id: str = ""  # ISIN or alternative identifier
    instrument_classification: InstrumentClassification = InstrumentClassification.DERIVATIVE
    instrument_name: str = ""
    notional_currency: str = "USD"

    # Counterparties
    buyer_lei: str = ""
    seller_lei: str = ""
    buyer_decision_maker_code: Optional[str] = None
    seller_decision_maker_code: Optional[str] = None
    transmitting_firm_lei: Optional[str] = None

    # Quantities and prices
    quantity: Decimal = Decimal("0")
    quantity_currency: str = "USD"
    price: Decimal = Decimal("0")
    price_currency: str = "USD"
    price_notation: str = "MONE"  # Monetary, Percentage, etc.
    notional_amount: Decimal = Decimal("0")

    # Transaction details
    buy_sell_indicator: TransactionType = TransactionType.BUY
    complex_trade_component_id: Optional[str] = None
    venue_transaction_id: Optional[str] = None

    # Derivatives specifics
    derivative_notional_change: bool = False
    derivative_cleared: bool = False
    ccp_lei: Optional[str] = None

    # Commodity derivatives
    commodity_derivative_indicator: bool = False

    # Flags
    short_selling_indicator: Optional[bool] = None
    waiver_indicator: Optional[str] = None
    otc_post_trade_indicator: bool = False

    def __post_init__(self):
        """Initialize transaction report."""
        self.report_type = ReportType.MIFID_TRANSACTION
        if not self.data:
            self.data = self._build_data_dict()

    def _build_data_dict(self) -> Dict[str, Any]:
        """Build data dictionary for reporting."""
        return {
            "transaction_reference": self.transaction_reference_number,
            "trading_date_time": self.trading_date_time.isoformat() if self.trading_date_time else None,
            "trading_capacity": self.trading_capacity.value,
            "venue": self.venue.value,
            "instrument_id": self.instrument_id,
            "instrument_classification": self.instrument_classification.value,
            "buyer_lei": self.buyer_lei,
            "seller_lei": self.seller_lei,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "notional_amount": str(self.notional_amount),
            "buy_sell_indicator": self.buy_sell_indicator.value,
        }

    def validate(self) -> bool:
        """Validate MiFID II transaction report."""
        super().validate()

        # Transaction reference is mandatory
        if not self.transaction_reference_number:
            self.errors.append("Transaction reference number is required")

        # Trading date/time is mandatory
        if not self.trading_date_time:
            self.errors.append("Trading date/time is required")

        # Instrument identification
        if not self.instrument_id:
            self.errors.append("Instrument identifier (ISIN) is required")

        # LEI validation
        if self.buyer_lei and len(self.buyer_lei) != 20:
            self.errors.append(f"Invalid buyer LEI: {self.buyer_lei}")

        if self.seller_lei and len(self.seller_lei) != 20:
            self.errors.append(f"Invalid seller LEI: {self.seller_lei}")

        # Quantity and price validation
        if self.quantity <= 0:
            self.errors.append("Quantity must be positive")

        if self.price <= 0:
            self.errors.append("Price must be positive")

        # Currency codes
        if len(self.quantity_currency) != 3:
            self.errors.append(f"Invalid quantity currency: {self.quantity_currency}")

        if len(self.price_currency) != 3:
            self.errors.append(f"Invalid price currency: {self.price_currency}")

        # Cleared derivatives must have CCP
        if self.derivative_cleared and not self.ccp_lei:
            self.errors.append("CCP LEI required for cleared derivatives")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to XML format for MiFID II reporting."""
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<Document xmlns="urn:iso:std:iso:20022:tech:xsd:auth.090.001.01">',
            "  <FinInstrmRptgTxRpt>",
            "    <Tx>",
            f"      <TxId>{self.transaction_reference_number}</TxId>",
            f"      <TradgDtTm>{self.trading_date_time.isoformat() if self.trading_date_time else ''}</TradgDtTm>",
            f"      <TradgCpcty>{self.trading_capacity.value}</TradgCpcty>",
            f"      <Venue>{self.venue.value}</Venue>",
            "      <FinInstrm>",
            f"        <Id><ISIN>{self.instrument_id}</ISIN></Id>",
            f"        <ClssfctnTp>{self.instrument_classification.value}</ClssfctnTp>",
            "      </FinInstrm>",
            f"      <Buyr><LEI>{self.buyer_lei}</LEI></Buyr>",
            f"      <Sellr><LEI>{self.seller_lei}</LEI></Sellr>",
            f"      <Qty>{self.quantity}</Qty>",
            f"      <Pric Ccy=\"{self.price_currency}\">{self.price}</Pric>",
            f"      <BuySellInd>{self.buy_sell_indicator.value}</BuySellInd>",
            "    </Tx>",
            "  </FinInstrmRptgTxRpt>",
            "</Document>",
        ]
        return "\n".join(xml_parts)


@dataclass
class MiFIDReferenceDataReport(RegulatoryReport):
    """MiFID II reference data report (RTS 23)."""

    # Instrument identification
    isin: str = ""
    instrument_full_name: str = ""
    instrument_classification: InstrumentClassification = InstrumentClassification.DERIVATIVE

    # Issuer/operator
    issuer_lei: Optional[str] = None
    operator_lei: Optional[str] = None

    # Trading venue
    trading_venue_mic: str = ""  # Market Identifier Code
    first_trade_date: Optional[datetime] = None
    termination_date: Optional[datetime] = None

    # Derivatives specific
    underlying_isin: Optional[str] = None
    underlying_index: Optional[str] = None
    option_type: Optional[str] = None  # CALL, PUT
    strike_price: Optional[Decimal] = None
    strike_price_currency: Optional[str] = None
    option_exercise_style: Optional[str] = None  # EURO, AMER, ASIA

    # Notional and denomination
    notional_currency: str = "USD"
    minimum_tradable_quantity: Optional[Decimal] = None

    def __post_init__(self):
        """Initialize reference data report."""
        self.report_type = ReportType.MIFID_REFERENCE_DATA

    def validate(self) -> bool:
        """Validate reference data report."""
        super().validate()

        if not self.isin or len(self.isin) != 12:
            self.errors.append("Valid ISIN (12 characters) is required")

        if not self.instrument_full_name:
            self.errors.append("Instrument full name is required")

        if not self.trading_venue_mic or len(self.trading_venue_mic) != 4:
            self.errors.append("Valid MIC code (4 characters) is required")

        if self.issuer_lei and len(self.issuer_lei) != 20:
            self.errors.append(f"Invalid issuer LEI: {self.issuer_lei}")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to XML for reference data reporting."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<ReferenceData>
  <ISIN>{self.isin}</ISIN>
  <FullName>{self.instrument_full_name}</FullName>
  <Classification>{self.instrument_classification.value}</Classification>
  <TradingVenue>{self.trading_venue_mic}</TradingVenue>
  <IssuerLEI>{self.issuer_lei or ''}</IssuerLEI>
</ReferenceData>"""


@dataclass
class BestExecutionReport:
    """MiFID II best execution analysis report."""

    reporting_period_start: datetime
    reporting_period_end: datetime
    venue: str
    instrument_class: InstrumentClassification

    # Execution quality metrics
    total_orders: int = 0
    executed_orders: int = 0
    average_execution_time_seconds: float = 0.0

    # Price metrics
    average_spread_bps: float = 0.0
    price_improvement_rate: float = 0.0
    average_price_improvement_bps: float = 0.0

    # Venue analysis
    venue_concentration: Dict[str, float] = field(default_factory=dict)  # venue -> % of volume

    def execution_rate(self) -> float:
        """Calculate order execution rate."""
        if self.total_orders == 0:
            return 0.0
        return self.executed_orders / self.total_orders

    def meets_best_execution_criteria(self) -> bool:
        """Check if venue meets best execution criteria.

        Returns
        -------
        bool
            True if execution quality meets regulatory expectations
        """
        # Example criteria (adjust based on regulatory requirements)
        return (
            self.execution_rate() >= 0.95 and  # 95%+ execution rate
            self.average_execution_time_seconds <= 10.0 and  # < 10s average execution
            self.average_spread_bps <= 5.0  # Reasonable spreads
        )


class BestExecutionAnalyzer:
    """Analyzer for MiFID II best execution requirements."""

    def __init__(self, firm_lei: str):
        """Initialize best execution analyzer.

        Parameters
        ----------
        firm_lei : str
            Legal Entity Identifier of the investment firm
        """
        self.firm_lei = firm_lei
        self.execution_data: List[Dict[str, Any]] = []

    def add_execution(
        self,
        venue: str,
        instrument_class: InstrumentClassification,
        execution_time: float,
        spread_bps: float,
        price_improvement_bps: float = 0.0,
    ) -> None:
        """Record an execution for analysis.

        Parameters
        ----------
        venue : str
            Execution venue
        instrument_class : InstrumentClassification
            Instrument classification
        execution_time : float
            Time to execution in seconds
        spread_bps : float
            Bid-ask spread in basis points
        price_improvement_bps : float
            Price improvement achieved in basis points
        """
        self.execution_data.append({
            "venue": venue,
            "instrument_class": instrument_class,
            "execution_time": execution_time,
            "spread_bps": spread_bps,
            "price_improvement_bps": price_improvement_bps,
            "timestamp": datetime.utcnow(),
        })

    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        instrument_class: Optional[InstrumentClassification] = None,
    ) -> BestExecutionReport:
        """Generate best execution report.

        Parameters
        ----------
        start_date : datetime
            Report period start
        end_date : datetime
            Report period end
        instrument_class : InstrumentClassification, optional
            Filter by instrument class

        Returns
        -------
        BestExecutionReport
            Best execution analysis report
        """
        # Filter data
        filtered_data = [
            d for d in self.execution_data
            if start_date <= d["timestamp"] <= end_date
            and (instrument_class is None or d["instrument_class"] == instrument_class)
        ]

        if not filtered_data:
            return BestExecutionReport(
                reporting_period_start=start_date,
                reporting_period_end=end_date,
                venue="ALL",
                instrument_class=instrument_class or InstrumentClassification.DERIVATIVE,
            )

        # Calculate metrics
        total = len(filtered_data)
        avg_exec_time = sum(d["execution_time"] for d in filtered_data) / total
        avg_spread = sum(d["spread_bps"] for d in filtered_data) / total

        price_improvements = [d["price_improvement_bps"] for d in filtered_data if d["price_improvement_bps"] > 0]
        avg_price_improvement = sum(price_improvements) / len(price_improvements) if price_improvements else 0.0
        price_improvement_rate = len(price_improvements) / total

        # Venue concentration
        venue_counts: Dict[str, int] = {}
        for d in filtered_data:
            venue_counts[d["venue"]] = venue_counts.get(d["venue"], 0) + 1

        venue_concentration = {
            venue: count / total
            for venue, count in venue_counts.items()
        }

        return BestExecutionReport(
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            venue="ALL",
            instrument_class=instrument_class or InstrumentClassification.DERIVATIVE,
            total_orders=total,
            executed_orders=total,  # Assuming all in data were executed
            average_execution_time_seconds=avg_exec_time,
            average_spread_bps=avg_spread,
            price_improvement_rate=price_improvement_rate,
            average_price_improvement_bps=avg_price_improvement,
            venue_concentration=venue_concentration,
        )


class MiFIDTransactionReporter:
    """MiFID II transaction reporting engine."""

    def __init__(
        self,
        firm_lei: str,
        competent_authority: str = "FCA",  # Financial Conduct Authority
    ):
        """Initialize MiFID transaction reporter.

        Parameters
        ----------
        firm_lei : str
            Investment firm LEI
        competent_authority : str
            National competent authority code
        """
        self.firm_lei = firm_lei
        self.competent_authority = competent_authority
        self.reports: Dict[str, MiFIDTransactionReport] = {}

    def create_transaction_report(
        self,
        transaction_ref: str,
        instrument_isin: str,
        **kwargs,
    ) -> MiFIDTransactionReport:
        """Create a transaction report.

        Parameters
        ----------
        transaction_ref : str
            Unique transaction reference
        instrument_isin : str
            Instrument ISIN
        **kwargs
            Additional transaction details

        Returns
        -------
        MiFIDTransactionReport
            Created transaction report
        """
        report = MiFIDTransactionReport(
            transaction_reference_number=transaction_ref,
            instrument_id=instrument_isin,
            trading_date_time=datetime.utcnow(),
            **kwargs,
        )
        self.reports[transaction_ref] = report
        return report

    def create_reference_data_report(
        self,
        isin: str,
        instrument_name: str,
        **kwargs,
    ) -> MiFIDReferenceDataReport:
        """Create a reference data report.

        Parameters
        ----------
        isin : str
            Instrument ISIN
        instrument_name : str
            Full instrument name
        **kwargs
            Additional reference data

        Returns
        -------
        MiFIDReferenceDataReport
            Created reference data report
        """
        return MiFIDReferenceDataReport(
            isin=isin,
            instrument_full_name=instrument_name,
            **kwargs,
        )

    def batch_submit(
        self,
        reports: List[MiFIDTransactionReport],
    ) -> Dict[str, Any]:
        """Submit batch of transaction reports.

        Parameters
        ----------
        reports : list of MiFIDTransactionReport
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
                    "transaction_ref": report.transaction_reference_number,
                    "errors": report.errors,
                })

        return results
