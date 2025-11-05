"""Core regulatory reporting engine and base classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
from uuid import UUID, uuid4


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
        """Convert report to JSON format.

        Returns
        -------
        dict
            JSON-serializable dictionary representation.
        """
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

    def __init__(
        self,
        entity_id: str,
        lei: str,  # Legal Entity Identifier
        validators: Optional[List[ReportValidator]] = None,
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
        report = RegulatoryReport(
            report_type=report_type,
            data=data,
            **kwargs,
        )
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
