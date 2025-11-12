"""Core regulatory reporting engine and base classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Type
from uuid import UUID, uuid4

import xml.etree.ElementTree as ET

from .validators.schema import SchemaValidator, build_default_schema_definitions


class ReportStatus(str, Enum):
    """Status of a regulatory report."""

    DRAFT = "draft"
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RECONCILED = "reconciled"


class ReportType(str, Enum):
    """Types of regulatory reports."""

    EMIR_TRADE = "emir_trade"
    EMIR_LIFECYCLE = "emir_lifecycle"
    EMIR_VALUATION = "emir_valuation"
    MIFID_TRANSACTION = "mifid_transaction"
    MIFID_REFERENCE_DATA = "mifid_reference_data"
    BASEL_CAPITAL = "basel_capital"
    BASEL_CVA = "basel_cva"
    BASEL_FRTB = "basel_frtb"
    BASEL_LEVERAGE = "basel_leverage"


@dataclass
class RegulatoryReport:
    """Base class for all regulatory reports."""

    report_id: UUID = field(default_factory=uuid4)
    report_type: ReportType = field(default=ReportType.EMIR_TRADE)
    reporting_date: datetime = field(default_factory=datetime.utcnow)
    status: ReportStatus = field(default=ReportStatus.DRAFT)
    counterparty_id: Optional[str] = None
    trade_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate the report data.

        Returns
        -------
        bool
            True if validation passes, False otherwise.
            Validation errors are stored in self.errors.
        """
        self.errors.clear()
        self.warnings.clear()

        # Base validation
        if not self.report_id:
            self.errors.append("Report ID is required")

        if self.status == ReportStatus.DRAFT:
            self.status = ReportStatus.PENDING_VALIDATION

        # Subclasses should override to add specific validation
        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert report to XML format for submission.

        Returns
        -------
        str
            XML representation of the report.

        Notes
        -----
        Subclasses must implement this method for their specific format.
        """
        raise NotImplementedError("Subclasses must implement to_xml()")

    def to_json(self) -> Dict[str, Any]:
        """Convert report to JSON format."""

        return {
            "report_id": str(self.report_id),
            "report_type": self.report_type.value,
            "reporting_date": self.reporting_date.isoformat(),
            "status": self.status.value,
            "counterparty_id": self.counterparty_id,
            "trade_id": self.trade_id,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    """Indent an XML tree in-place for pretty printing."""

    indent = "  "
    whitespace = "\n" + level * indent
    if len(element):
        if not element.text or not element.text.strip():
            element.text = whitespace + indent
        for idx, child in enumerate(element):
            _indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                if idx + 1 < len(element):
                    child.tail = whitespace + indent
                else:
                    child.tail = whitespace
    else:
        if not element.text or not element.text.strip():
            element.text = ""


def _to_xml_string(root: ET.Element) -> str:
    """Render an XML element tree to a string with declaration."""

    _indent_xml(root)
    xml_body = ET.tostring(root, encoding="unicode")
    if not xml_body.endswith("\n"):
        xml_body = f"{xml_body}\n"
    return f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{xml_body}"


@dataclass
class EMIRTradeRegulatoryReport(RegulatoryReport):
    """EMIR trade report aligned with ESMA R0001 schema."""

    def __post_init__(self) -> None:  # pragma: no cover - simple assignment
        self.report_type = ReportType.EMIR_TRADE

    def to_xml(self) -> str:
        data = self.data
        reporting = data.get("reporting_counterparty", {})
        counterparty = data.get("counterparty", {})
        trade = data.get("trade", {})
        product = trade.get("product", {})
        notional = trade.get("notional", {})

        root = ET.Element("Document", {"xmlns": "urn:esma:emir:r0001"})
        header = ET.SubElement(root, "ReportHeader")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "CreationTimestamp").text = self.reporting_date.replace(microsecond=0).isoformat()
        reporting_entity = ET.SubElement(header, "ReportingEntity")
        if reporting.get("lei"):
            ET.SubElement(reporting_entity, "LEI").text = reporting.get("lei")

        trade_report = ET.SubElement(root, "TradeReport")
        reporting_cp = ET.SubElement(trade_report, "ReportingCounterparty")
        if reporting.get("lei"):
            ET.SubElement(reporting_cp, "LEI").text = reporting.get("lei")

        counterparty_elem = ET.SubElement(trade_report, "Counterparty")
        if counterparty.get("lei"):
            ET.SubElement(counterparty_elem, "LEI").text = counterparty.get("lei")

        trade_data = ET.SubElement(trade_report, "TradeData")
        if trade.get("trade_id"):
            ET.SubElement(trade_data, "TradeId").text = trade.get("trade_id")

        product_elem = None
        if product:
            product_elem = ET.SubElement(trade_data, "Product")
            if product.get("upi"):
                ET.SubElement(product_elem, "UPI").text = product.get("upi")
            if product.get("asset_class"):
                ET.SubElement(product_elem, "AssetClass").text = product.get("asset_class")

        if notional:
            notional_elem = ET.SubElement(trade_data, "Notional")
            if notional.get("amount") is not None:
                ET.SubElement(notional_elem, "Amount").text = str(notional.get("amount"))
            if notional.get("currency"):
                ET.SubElement(notional_elem, "Currency").text = notional.get("currency")

        if trade.get("execution_timestamp"):
            ET.SubElement(trade_data, "ExecutionTimestamp").text = trade.get("execution_timestamp")

        return _to_xml_string(root)


@dataclass
class EMIRLifecycleRegulatoryReport(RegulatoryReport):
    """EMIR lifecycle event report adhering to ESMA change records."""

    def __post_init__(self) -> None:  # pragma: no cover - simple assignment
        self.report_type = ReportType.EMIR_LIFECYCLE

    def to_xml(self) -> str:
        event = self.data.get("event", {})
        changes = event.get("changes", [])

        root = ET.Element("Document", {"xmlns": "urn:esma:emir:r0001:lifecycle"})
        header = ET.SubElement(root, "ReportHeader")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "CreationTimestamp").text = self.reporting_date.replace(microsecond=0).isoformat()

        lifecycle = ET.SubElement(root, "LifecycleEvent")
        if event.get("uti"):
            ET.SubElement(lifecycle, "UTI").text = event.get("uti")
        if event.get("event_type"):
            ET.SubElement(lifecycle, "EventType").text = event.get("event_type")
        if event.get("event_timestamp"):
            ET.SubElement(lifecycle, "EventTimestamp").text = event.get("event_timestamp")
        if event.get("prior_uti"):
            ET.SubElement(lifecycle, "PriorUTI").text = event.get("prior_uti")
        if event.get("new_uti"):
            ET.SubElement(lifecycle, "NewUTI").text = event.get("new_uti")

        if changes:
            changes_elem = ET.SubElement(lifecycle, "Changes")
            for change in changes:
                change_elem = ET.SubElement(changes_elem, "Change")
                if change.get("field"):
                    ET.SubElement(change_elem, "Field").text = change.get("field")
                if change.get("old_value") is not None:
                    ET.SubElement(change_elem, "OldValue").text = str(change.get("old_value"))
                if change.get("new_value") is not None:
                    ET.SubElement(change_elem, "NewValue").text = str(change.get("new_value"))

        return _to_xml_string(root)


@dataclass
class EMIRValuationRegulatoryReport(RegulatoryReport):
    """EMIR valuation report following ESMA R0010 layout."""

    def __post_init__(self) -> None:  # pragma: no cover
        self.report_type = ReportType.EMIR_VALUATION

    def to_xml(self) -> str:
        valuation = self.data.get("valuation", {})

        root = ET.Element("Document", {"xmlns": "urn:esma:emir:r0010"})
        header = ET.SubElement(root, "ReportHeader")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "CreationTimestamp").text = self.reporting_date.replace(microsecond=0).isoformat()

        valuation_elem = ET.SubElement(root, "ValuationReport")
        if valuation.get("uti"):
            ET.SubElement(valuation_elem, "UTI").text = valuation.get("uti")
        if valuation.get("valuation_date"):
            ET.SubElement(valuation_elem, "ValuationDate").text = valuation.get("valuation_date")
        if valuation.get("valuation_amount") is not None:
            amount_elem = ET.SubElement(valuation_elem, "ValuationAmount")
            amount_elem.text = str(valuation.get("valuation_amount"))
            if valuation.get("valuation_currency"):
                amount_elem.set("currency", valuation.get("valuation_currency"))
        if valuation.get("valuation_type"):
            ET.SubElement(valuation_elem, "ValuationType").text = valuation.get("valuation_type")
        if valuation.get("valuation_timestamp"):
            ET.SubElement(valuation_elem, "ValuationTimestamp").text = valuation.get("valuation_timestamp")

        return _to_xml_string(root)


@dataclass
class MiFIDTransactionRegulatoryReport(RegulatoryReport):
    """MiFID II transaction report aligned with RTS 22."""

    def __post_init__(self) -> None:  # pragma: no cover
        self.report_type = ReportType.MIFID_TRANSACTION

    def to_xml(self) -> str:
        data = self.data
        transaction = data.get("transaction", {})
        instrument = transaction.get("instrument", {})
        buyer = transaction.get("buyer", {})
        seller = transaction.get("seller", {})
        price = transaction.get("price", {})

        root = ET.Element("Document", {"xmlns": "urn:eu:esma:rr:mifid:rts22"})
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "SubmissionDate").text = self.reporting_date.date().isoformat()
        reporting_firm = ET.SubElement(header, "ReportingFirm")
        if data.get("reporting_firm"):
            ET.SubElement(reporting_firm, "LEI").text = data["reporting_firm"].get("lei", "")

        transaction_elem = ET.SubElement(root, "Transaction")
        if transaction.get("transaction_reference"):
            ET.SubElement(transaction_elem, "TransactionReference").text = transaction.get("transaction_reference")
        if transaction.get("execution_timestamp"):
            ET.SubElement(transaction_elem, "ExecutionTimestamp").text = transaction.get("execution_timestamp")
        if transaction.get("quantity") is not None:
            ET.SubElement(transaction_elem, "Quantity").text = str(transaction.get("quantity"))

        instrument_elem = None
        if instrument:
            instrument_elem = ET.SubElement(transaction_elem, "Instrument")
            if instrument.get("isin"):
                ET.SubElement(instrument_elem, "ISIN").text = instrument.get("isin")
            if instrument.get("classification"):
                ET.SubElement(instrument_elem, "Classification").text = instrument.get("classification")

        if buyer:
            buyer_elem = ET.SubElement(transaction_elem, "Buyer")
            if buyer.get("lei"):
                ET.SubElement(buyer_elem, "LEI").text = buyer.get("lei")

        if seller:
            seller_elem = ET.SubElement(transaction_elem, "Seller")
            if seller.get("lei"):
                ET.SubElement(seller_elem, "LEI").text = seller.get("lei")

        if price:
            price_elem = ET.SubElement(transaction_elem, "Price")
            if price.get("amount") is not None:
                ET.SubElement(price_elem, "Amount").text = str(price.get("amount"))
            if price.get("currency"):
                ET.SubElement(price_elem, "Currency").text = price.get("currency")

        return _to_xml_string(root)


@dataclass
class MiFIDReferenceDataRegulatoryReport(RegulatoryReport):
    """MiFID II reference data report aligned with RTS 23."""

    def __post_init__(self) -> None:  # pragma: no cover
        self.report_type = ReportType.MIFID_REFERENCE_DATA

    def to_xml(self) -> str:
        instrument = self.data.get("instrument", {})

        root = ET.Element("Document", {"xmlns": "urn:eu:esma:rr:mifid:rts23"})
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "SubmissionDate").text = self.reporting_date.date().isoformat()

        instrument_elem = ET.SubElement(root, "Instrument")
        if instrument.get("isin"):
            ET.SubElement(instrument_elem, "ISIN").text = instrument.get("isin")
        if instrument.get("full_name"):
            ET.SubElement(instrument_elem, "Name").text = instrument.get("full_name")
        if instrument.get("classification"):
            ET.SubElement(instrument_elem, "Classification").text = instrument.get("classification")
        if instrument.get("currency"):
            ET.SubElement(instrument_elem, "Currency").text = instrument.get("currency")
        if instrument.get("issuance_date"):
            ET.SubElement(instrument_elem, "IssuanceDate").text = instrument.get("issuance_date")

        return _to_xml_string(root)


@dataclass
class BaselCapitalRegulatoryReport(RegulatoryReport):
    """Basel III Pillar 3 capital report."""

    def __post_init__(self) -> None:  # pragma: no cover
        self.report_type = ReportType.BASEL_CAPITAL

    def to_xml(self) -> str:
        data = self.data
        bank = data.get("bank", {})
        capital = data.get("capital", {})
        rwa = data.get("risk_weighted_assets", {})
        ratios = data.get("ratios", {})

        root = ET.Element("Pillar3Report", {"xmlns": "urn:bis:basel:capital"})
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "ReferenceDate").text = self.reporting_date.date().isoformat()
        if bank.get("name"):
            ET.SubElement(header, "BankName").text = bank.get("name")
        if bank.get("lei"):
            ET.SubElement(header, "BankLEI").text = bank.get("lei")

        capital_elem = ET.SubElement(root, "CapitalStructure")
        if capital.get("cet1") is not None:
            ET.SubElement(capital_elem, "CET1").text = str(capital.get("cet1"))
        if capital.get("tier1") is not None:
            ET.SubElement(capital_elem, "Tier1").text = str(capital.get("tier1"))
        if capital.get("tier2") is not None:
            ET.SubElement(capital_elem, "Tier2").text = str(capital.get("tier2"))

        rwa_elem = ET.SubElement(root, "RiskWeightedAssets")
        if rwa.get("credit") is not None:
            ET.SubElement(rwa_elem, "Credit").text = str(rwa.get("credit"))
        if rwa.get("market") is not None:
            ET.SubElement(rwa_elem, "Market").text = str(rwa.get("market"))
        if rwa.get("operational") is not None:
            ET.SubElement(rwa_elem, "Operational").text = str(rwa.get("operational"))

        ratios_elem = ET.SubElement(root, "CapitalRatios")
        if ratios.get("cet1") is not None:
            ET.SubElement(ratios_elem, "CET1Ratio").text = str(ratios.get("cet1"))
        if ratios.get("tier1") is not None:
            ET.SubElement(ratios_elem, "Tier1Ratio").text = str(ratios.get("tier1"))
        if ratios.get("total") is not None:
            ET.SubElement(ratios_elem, "TotalCapitalRatio").text = str(ratios.get("total"))

        return _to_xml_string(root)


@dataclass
class BaselCVARegulatoryReport(RegulatoryReport):
    """Basel CVA risk capital report."""

    def __post_init__(self) -> None:  # pragma: no cover
        self.report_type = ReportType.BASEL_CVA

    def to_xml(self) -> str:
        data = self.data
        bank = data.get("bank", {})
        cva = data.get("cva", {})

        root = ET.Element("CVAReport", {"xmlns": "urn:bis:basel:cva"})
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "ReferenceDate").text = self.reporting_date.date().isoformat()
        if bank.get("lei"):
            ET.SubElement(header, "BankLEI").text = bank.get("lei")

        cva_charge = ET.SubElement(root, "CVACharge")
        if cva.get("capital_charge") is not None:
            ET.SubElement(cva_charge, "CapitalCharge").text = str(cva.get("capital_charge"))
        if cva.get("advanced_method") is not None:
            ET.SubElement(cva_charge, "AdvancedMethod").text = str(cva.get("advanced_method"))

        hedges = cva.get("hedges", {})
        if hedges:
            hedges_elem = ET.SubElement(root, "Hedges")
            if hedges.get("eligible") is not None:
                ET.SubElement(hedges_elem, "Eligible").text = str(hedges.get("eligible"))
            if hedges.get("non_eligible") is not None:
                ET.SubElement(hedges_elem, "NonEligible").text = str(hedges.get("non_eligible"))

        return _to_xml_string(root)


@dataclass
class BaselFRTBRegulatoryReport(RegulatoryReport):
    """Basel FRTB market risk capital report."""

    def __post_init__(self) -> None:  # pragma: no cover
        self.report_type = ReportType.BASEL_FRTB

    def to_xml(self) -> str:
        data = self.data
        bank = data.get("bank", {})
        requirement = data.get("requirement", {})

        root = ET.Element("FRTBReport", {"xmlns": "urn:bis:basel:frtb"})
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "ReferenceDate").text = self.reporting_date.date().isoformat()
        if bank.get("lei"):
            ET.SubElement(header, "BankLEI").text = bank.get("lei")

        capital_req = ET.SubElement(root, "CapitalRequirement")
        if requirement.get("standardised") is not None:
            ET.SubElement(capital_req, "Standardised").text = str(requirement.get("standardised"))
        if requirement.get("internal_model") is not None:
            ET.SubElement(capital_req, "InternalModel").text = str(requirement.get("internal_model"))
        if requirement.get("total") is not None:
            ET.SubElement(capital_req, "Total").text = str(requirement.get("total"))

        return _to_xml_string(root)


@dataclass
class BaselLeverageRegulatoryReport(RegulatoryReport):
    """Basel leverage ratio report."""

    def __post_init__(self) -> None:  # pragma: no cover
        self.report_type = ReportType.BASEL_LEVERAGE

    def to_xml(self) -> str:
        data = self.data
        bank = data.get("bank", {})
        leverage = data.get("leverage", {})

        root = ET.Element("LeverageReport", {"xmlns": "urn:bis:basel:leverage"})
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportId").text = str(self.report_id)
        ET.SubElement(header, "ReferenceDate").text = self.reporting_date.date().isoformat()
        if bank.get("lei"):
            ET.SubElement(header, "BankLEI").text = bank.get("lei")

        leverage_elem = ET.SubElement(root, "LeverageRatio")
        if leverage.get("tier1") is not None:
            ET.SubElement(leverage_elem, "Tier1Capital").text = str(leverage.get("tier1"))
        if leverage.get("exposure") is not None:
            ET.SubElement(leverage_elem, "ExposureMeasure").text = str(leverage.get("exposure"))
        if leverage.get("ratio") is not None:
            ET.SubElement(leverage_elem, "Ratio").text = str(leverage.get("ratio"))

        return _to_xml_string(root)


@dataclass
class ReportSubmission:
    """Record of a report submission to a regulatory authority."""

    submission_id: UUID = field(default_factory=uuid4)
    report_id: UUID = field(default=None)
    submission_timestamp: datetime = field(default_factory=datetime.utcnow)
    repository: str = ""  # Trade repository name (e.g., "DTCC", "Regis-TR")
    status: ReportStatus = field(default=ReportStatus.SUBMITTED)
    acknowledgment_id: Optional[str] = None
    response_message: Optional[str] = None
    rejection_reason: Optional[str] = None


class ReportValidator(Protocol):
    """Protocol for report validators."""

    def validate(self, report: RegulatoryReport) -> bool:
        """Validate a regulatory report."""
        ...


class RegulatoryReportEngine:
    """Engine for generating, validating, and submitting regulatory reports."""

    REPORT_CLASS_REGISTRY: Dict[ReportType, Type[RegulatoryReport]] = {
        ReportType.EMIR_TRADE: EMIRTradeRegulatoryReport,
        ReportType.EMIR_LIFECYCLE: EMIRLifecycleRegulatoryReport,
        ReportType.EMIR_VALUATION: EMIRValuationRegulatoryReport,
        ReportType.MIFID_TRANSACTION: MiFIDTransactionRegulatoryReport,
        ReportType.MIFID_REFERENCE_DATA: MiFIDReferenceDataRegulatoryReport,
        ReportType.BASEL_CAPITAL: BaselCapitalRegulatoryReport,
        ReportType.BASEL_CVA: BaselCVARegulatoryReport,
        ReportType.BASEL_FRTB: BaselFRTBRegulatoryReport,
        ReportType.BASEL_LEVERAGE: BaselLeverageRegulatoryReport,
    }

    def __init__(
        self,
        entity_id: str,
        lei: str,  # Legal Entity Identifier
        validators: Optional[List[ReportValidator]] = None,
        schema_validator: Optional[SchemaValidator] = None,
    ):
        """Initialize the regulatory report engine.

        Parameters
        ----------
        entity_id : str
            Reporting entity identifier
        lei : str
            Legal Entity Identifier
        validators : list of ReportValidator, optional
            Custom validators to apply
        """
        self.entity_id = entity_id
        self.lei = lei
        self.validators = validators or []
        if schema_validator is None:
            schema_definitions = build_default_schema_definitions(ReportType)
            schema_validator = SchemaValidator(schema_definitions)
        self.schema_validator = schema_validator
        self.reports: Dict[UUID, RegulatoryReport] = {}
        self.submissions: Dict[UUID, ReportSubmission] = {}

    def create_report(
        self,
        report_type: ReportType,
        data: Dict[str, Any],
        **kwargs,
    ) -> RegulatoryReport:
        """Create a new regulatory report.

        Parameters
        ----------
        report_type : ReportType
            Type of report to create
        data : dict
            Report data
        **kwargs
            Additional report fields

        Returns
        -------
        RegulatoryReport
            Created report instance
        """
        report_cls = self.REPORT_CLASS_REGISTRY.get(report_type, RegulatoryReport)
        report_kwargs = dict(kwargs)
        report_kwargs.setdefault("report_type", report_type)
        report = report_cls(
            data=data,
            **report_kwargs,
        )
        report.report_type = report_type
        self.reports[report.report_id] = report
        return report

    def validate_report(self, report: RegulatoryReport) -> bool:
        """Validate a regulatory report.

        Parameters
        ----------
        report : RegulatoryReport
            Report to validate

        Returns
        -------
        bool
            True if validation passes
        """
        # Base validation
        if not report.validate():
            return False

        # Apply XML schema validation
        schema_errors = self.schema_validator.validate(report)
        if schema_errors:
            report.errors.extend(schema_errors)
            return False

        # Apply custom validators
        for validator in self.validators:
            if not validator.validate(report):
                return False

        report.status = ReportStatus.VALIDATED
        return True

    def submit_report(
        self,
        report: RegulatoryReport,
        repository: str,
    ) -> ReportSubmission:
        """Submit a validated report to a regulatory repository.

        Parameters
        ----------
        report : RegulatoryReport
            Report to submit
        repository : str
            Target repository identifier

        Returns
        -------
        ReportSubmission
            Submission record

        Raises
        ------
        ValueError
            If report is not in validated status
        """
        if report.status != ReportStatus.VALIDATED:
            raise ValueError(
                f"Report must be validated before submission. Current status: {report.status}"
            )

        submission = ReportSubmission(
            report_id=report.report_id,
            repository=repository,
        )

        # In production, this would actually submit to the repository
        # For now, we record the submission
        self.submissions[submission.submission_id] = submission
        report.status = ReportStatus.SUBMITTED

        return submission

    def reconcile_report(
        self,
        report_id: UUID,
        counterparty_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reconcile a report with counterparty's version.

        Parameters
        ----------
        report_id : UUID
            Report to reconcile
        counterparty_report : dict
            Counterparty's version of the report

        Returns
        -------
        dict
            Reconciliation results with differences identified
        """
        report = self.reports.get(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")

        differences = {}
        for key, our_value in report.data.items():
            cp_value = counterparty_report.get(key)
            if our_value != cp_value:
                differences[key] = {
                    "our_value": our_value,
                    "counterparty_value": cp_value,
                }

        # Check for missing fields
        missing_in_our_report = set(counterparty_report.keys()) - set(report.data.keys())
        missing_in_cp_report = set(report.data.keys()) - set(counterparty_report.keys())

        return {
            "report_id": str(report_id),
            "differences": differences,
            "missing_in_our_report": list(missing_in_our_report),
            "missing_in_counterparty_report": list(missing_in_cp_report),
            "is_matched": len(differences) == 0 and len(missing_in_our_report) == 0 and len(missing_in_cp_report) == 0,
        }

    def get_report(self, report_id: UUID) -> Optional[RegulatoryReport]:
        """Retrieve a report by ID."""
        return self.reports.get(report_id)

    def get_reports_by_status(self, status: ReportStatus) -> List[RegulatoryReport]:
        """Retrieve all reports with a given status."""
        return [r for r in self.reports.values() if r.status == status]

    def get_submission(self, submission_id: UUID) -> Optional[ReportSubmission]:
        """Retrieve a submission record by ID."""
        return self.submissions.get(submission_id)


__all__ = [
    "RegulatoryReport",
    "RegulatoryReportEngine",
    "ReportStatus",
    "ReportSubmission",
    "ReportType",
    "EMIRTradeRegulatoryReport",
    "EMIRLifecycleRegulatoryReport",
    "EMIRValuationRegulatoryReport",
    "MiFIDTransactionRegulatoryReport",
    "MiFIDReferenceDataRegulatoryReport",
    "BaselCapitalRegulatoryReport",
    "BaselCVARegulatoryReport",
    "BaselFRTBRegulatoryReport",
    "BaselLeverageRegulatoryReport",
]
