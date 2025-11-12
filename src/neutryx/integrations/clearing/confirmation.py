"""Trade confirmation matching and affirmation for clearing operations.

This module implements the confirmation lifecycle:
1. Trade confirmation generation
2. Bilateral matching between counterparties
3. Break resolution and exception handling
4. Central affirmation by CCP
5. Allocation confirmation for prime brokerage

Implements industry standards:
- FpML confirmation messages
- ISO 20022 securities confirmation
- DTCC CTM (Central Trade Manager) protocol
- Omgeo OASYS connectivity
"""

from __future__ import annotations

from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base import Party, ProductType, Trade, TradeEconomics


class ConfirmationStatus(str, Enum):
    """Confirmation status."""
    PENDING = "pending"
    SENT = "sent"
    RECEIVED = "received"
    MATCHED = "matched"
    MISMATCHED = "mismatched"
    AFFIRMED = "affirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BreakType(str, Enum):
    """Types of trade breaks (discrepancies)."""
    PRICE_MISMATCH = "price_mismatch"
    QUANTITY_MISMATCH = "quantity_mismatch"
    SETTLEMENT_DATE_MISMATCH = "settlement_date_mismatch"
    COUNTERPARTY_MISMATCH = "counterparty_mismatch"
    ACCOUNT_MISMATCH = "account_mismatch"
    PRODUCT_MISMATCH = "product_mismatch"
    MISSING_CONFIRMATION = "missing_confirmation"
    DUPLICATE_CONFIRMATION = "duplicate_confirmation"
    TIMEOUT = "timeout"
    EXTERNAL_STATUS_MISMATCH = "external_status_mismatch"


class BreakSeverity(str, Enum):
    """Break severity level."""
    LOW = "low"           # Minor discrepancies, auto-resolvable
    MEDIUM = "medium"     # Requires review
    HIGH = "high"         # Blocking issue
    CRITICAL = "critical"  # Trade validity in question


class AffirmationMethod(str, Enum):
    """Affirmation method."""
    ELECTRONIC = "electronic"  # Automated via API/messaging
    MANUAL = "manual"          # Phone/email confirmation
    CENTRAL = "central"        # CCP central affirmation
    DEEMED = "deemed"          # Affirmation by timeout


class ToleranceConfig(BaseModel):
    """Tolerance settings for matching."""

    price_tolerance: Decimal = Field(default=Decimal("0.01"), description="Price tolerance (bps)")
    quantity_tolerance: Decimal = Field(default=Decimal("0"), description="Quantity tolerance")
    rate_tolerance: Decimal = Field(default=Decimal("0.0001"), description="Rate tolerance")
    amount_tolerance: Decimal = Field(default=Decimal("100"), description="Amount tolerance")
    time_tolerance_minutes: int = Field(default=15, description="Settlement time tolerance")

    strict_matching: bool = Field(default=False, description="Require exact match")
    ignore_minor_breaks: bool = Field(default=True, description="Ignore minor breaks")


class ConfirmationDetails(BaseModel):
    """Confirmation details for a trade."""

    # Core trade details
    trade_id: str = Field(..., description="Trade identifier")
    trade_date: datetime = Field(..., description="Trade date")
    settlement_date: datetime = Field(..., description="Settlement date")

    # Counterparties
    buyer: Party = Field(..., description="Buyer")
    seller: Party = Field(..., description="Seller")
    executing_broker: Optional[str] = Field(None, description="Executing broker")
    clearing_broker: Optional[str] = Field(None, description="Clearing broker")

    # Product
    product_type: ProductType = Field(..., description="Product type")
    product_description: str = Field(..., description="Product description")

    # Economics
    quantity: Decimal = Field(..., description="Quantity")
    price: Optional[Decimal] = Field(None, description="Price")
    notional: Decimal = Field(..., description="Notional amount")
    currency: str = Field(..., description="Currency")

    # Settlement
    settlement_type: str = Field(default="DVP", description="Settlement type")
    settlement_currency: Optional[str] = Field(None, description="Settlement currency")
    settlement_amount: Optional[Decimal] = Field(None, description="Settlement amount")

    # Accounts
    buyer_account: Optional[str] = Field(None, description="Buyer account")
    seller_account: Optional[str] = Field(None, description="Seller account")
    custodian: Optional[str] = Field(None, description="Custodian")

    # Additional terms
    accrued_interest: Optional[Decimal] = Field(None, description="Accrued interest")
    fees: Optional[Decimal] = Field(None, description="Fees")
    commission: Optional[Decimal] = Field(None, description="Commission")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "trade_date": self.trade_date.isoformat(),
            "settlement_date": self.settlement_date.isoformat(),
            "buyer": self.buyer.model_dump(),
            "seller": self.seller.model_dump(),
            "product_type": self.product_type.value,
            "product_description": self.product_description,
            "quantity": float(self.quantity),
            "price": float(self.price) if self.price else None,
            "notional": float(self.notional),
            "currency": self.currency,
            "settlement_type": self.settlement_type,
            "metadata": self.metadata,
        }


class Confirmation(BaseModel):
    """Trade confirmation."""

    confirmation_id: str = Field(
        default_factory=lambda: f"CNF-{uuid4().hex[:12].upper()}"
    )
    status: ConfirmationStatus = Field(default=ConfirmationStatus.PENDING)

    # Origination
    originator: Party = Field(..., description="Party sending confirmation")
    recipient: Party = Field(..., description="Party receiving confirmation")
    direction: str = Field(..., description="buy or sell from originator perspective")

    # Details
    details: ConfirmationDetails = Field(..., description="Trade details")

    # Lifecycle
    created_time: datetime = Field(default_factory=datetime.utcnow)
    sent_time: Optional[datetime] = Field(None, description="Time sent")
    received_time: Optional[datetime] = Field(None, description="Time received")
    matched_time: Optional[datetime] = Field(None, description="Time matched")
    affirmed_time: Optional[datetime] = Field(None, description="Time affirmed")

    # Matching
    counterparty_confirmation_id: Optional[str] = Field(None, description="Matching confirmation ID")
    match_score: Optional[float] = Field(None, description="Match quality score 0-100")

    # Affirmation
    affirmation_method: Optional[AffirmationMethod] = Field(None)
    affirmed_by: Optional[str] = Field(None, description="Party that affirmed")

    # Exception handling
    breaks: List[str] = Field(default_factory=list, description="Break IDs")
    reject_reason: Optional[str] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradeBreak(BaseModel):
    """Trade break (discrepancy) between confirmations."""

    break_id: str = Field(default_factory=lambda: f"BRK-{uuid4().hex[:12].upper()}")
    break_type: BreakType = Field(..., description="Type of break")
    severity: BreakSeverity = Field(..., description="Severity level")

    # Related confirmations
    confirmation_a_id: str = Field(..., description="First confirmation")
    confirmation_b_id: str = Field(..., description="Second confirmation")

    # Field details
    field_name: str = Field(..., description="Field with discrepancy")
    expected_value: Optional[Any] = Field(None, description="Expected value")
    actual_value: Optional[Any] = Field(None, description="Actual value")
    difference: Optional[Any] = Field(None, description="Difference")

    # Resolution
    resolved: bool = Field(default=False)
    resolution: Optional[str] = Field(None, description="Resolution method")
    resolved_by: Optional[str] = Field(None)
    resolved_time: Optional[datetime] = Field(None)

    # Lifecycle
    created_time: datetime = Field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = Field(None, description="Assigned analyst")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class MatchResult(BaseModel):
    """Result of confirmation matching."""

    trade_id: str = Field(..., description="Trade ID")
    confirmation_a_id: str = Field(..., description="First confirmation")
    confirmation_b_id: str = Field(..., description="Second confirmation")

    matched: bool = Field(..., description="Whether confirmations match")
    match_score: float = Field(..., description="Match quality 0-100")

    breaks: List[TradeBreak] = Field(default_factory=list)
    num_breaks: int = Field(default=0)
    critical_breaks: int = Field(default=0)

    can_auto_affirm: bool = Field(default=False)
    requires_manual_review: bool = Field(default=False)

    matched_time: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class MatchStatistics:
    """Matching statistics."""
    total_confirmations: int = 0
    matched_confirmations: int = 0
    pending_confirmations: int = 0
    breaks_detected: int = 0
    critical_breaks: int = 0
    auto_affirmed: int = 0
    manual_reviews: int = 0
    avg_match_time_seconds: float = 0.0
    match_rate: float = 100.0

    def update_match_rate(self):
        """Update match rate."""
        if self.total_confirmations > 0:
            self.match_rate = (self.matched_confirmations / self.total_confirmations) * 100


class ConfirmationMatcher:
    """Bilateral confirmation matcher."""

    def __init__(self, tolerance: Optional[ToleranceConfig] = None):
        self.tolerance = tolerance or ToleranceConfig()
        self.confirmations: Dict[str, Confirmation] = {}
        self.breaks: Dict[str, TradeBreak] = {}
        self.matches: Dict[str, MatchResult] = {}
        self.statistics = MatchStatistics()

    def add_confirmation(self, confirmation: Confirmation) -> Confirmation:
        """Add confirmation for matching."""
        self.confirmations[confirmation.confirmation_id] = confirmation
        self.statistics.total_confirmations += 1
        self.statistics.pending_confirmations += 1

        # Try to find counterparty confirmation
        counterparty = self._find_counterparty_confirmation(confirmation)

        if counterparty:
            # Attempt matching
            result = self.match_confirmations(
                confirmation.confirmation_id,
                counterparty.confirmation_id
            )

            if result.matched:
                self.statistics.matched_confirmations += 2  # Both sides
                self.statistics.pending_confirmations -= 2

                if result.can_auto_affirm:
                    self.statistics.auto_affirmed += 2

            if result.breaks:
                self.statistics.breaks_detected += len(result.breaks)
                self.statistics.critical_breaks += result.critical_breaks

            if result.requires_manual_review:
                self.statistics.manual_reviews += 1

        self.statistics.update_match_rate()

        return confirmation

    def _find_counterparty_confirmation(
        self,
        confirmation: Confirmation
    ) -> Optional[Confirmation]:
        """Find matching counterparty confirmation."""
        trade_id = confirmation.details.trade_id

        for conf_id, conf in self.confirmations.items():
            if conf_id == confirmation.confirmation_id:
                continue

            # Must be same trade
            if conf.details.trade_id != trade_id:
                continue

            # Must be from counterparty
            if conf.originator.party_id == confirmation.originator.party_id:
                continue

            # Must be opposite side
            if conf.direction == confirmation.direction:
                continue

            # Must not already be matched
            if conf.status == ConfirmationStatus.MATCHED:
                continue

            return conf

        return None

    def match_confirmations(
        self,
        confirmation_a_id: str,
        confirmation_b_id: str
    ) -> MatchResult:
        """Match two confirmations."""
        conf_a = self.confirmations.get(confirmation_a_id)
        conf_b = self.confirmations.get(confirmation_b_id)

        if not conf_a or not conf_b:
            raise ValueError("Confirmation not found")

        breaks = []
        match_score = 100.0

        # Compare fields
        breaks.extend(self._compare_field(
            "trade_date",
            conf_a.details.trade_date,
            conf_b.details.trade_date,
            conf_a, conf_b,
            BreakType.SETTLEMENT_DATE_MISMATCH,
            BreakSeverity.HIGH
        ))

        breaks.extend(self._compare_field(
            "settlement_date",
            conf_a.details.settlement_date,
            conf_b.details.settlement_date,
            conf_a, conf_b,
            BreakType.SETTLEMENT_DATE_MISMATCH,
            BreakSeverity.HIGH
        ))

        breaks.extend(self._compare_decimal(
            "quantity",
            conf_a.details.quantity,
            conf_b.details.quantity,
            self.tolerance.quantity_tolerance,
            conf_a, conf_b,
            BreakType.QUANTITY_MISMATCH
        ))

        if conf_a.details.price and conf_b.details.price:
            breaks.extend(self._compare_decimal(
                "price",
                conf_a.details.price,
                conf_b.details.price,
                self.tolerance.price_tolerance,
                conf_a, conf_b,
                BreakType.PRICE_MISMATCH
            ))

        breaks.extend(self._compare_decimal(
            "notional",
            conf_a.details.notional,
            conf_b.details.notional,
            self.tolerance.amount_tolerance,
            conf_a, conf_b,
            BreakType.QUANTITY_MISMATCH
        ))

        # Counterparty checks
        if conf_a.details.buyer.party_id != conf_b.details.buyer.party_id:
            breaks.append(self._create_break(
                "buyer",
                conf_a.details.buyer.party_id,
                conf_b.details.buyer.party_id,
                conf_a, conf_b,
                BreakType.COUNTERPARTY_MISMATCH,
                BreakSeverity.CRITICAL
            ))

        if conf_a.details.seller.party_id != conf_b.details.seller.party_id:
            breaks.append(self._create_break(
                "seller",
                conf_a.details.seller.party_id,
                conf_b.details.seller.party_id,
                conf_a, conf_b,
                BreakType.COUNTERPARTY_MISMATCH,
                BreakSeverity.CRITICAL
            ))

        # Calculate match score
        if breaks:
            # Deduct points for each break
            severity_weights = {
                BreakSeverity.LOW: 2,
                BreakSeverity.MEDIUM: 5,
                BreakSeverity.HIGH: 15,
                BreakSeverity.CRITICAL: 40
            }

            for brk in breaks:
                match_score -= severity_weights.get(brk.severity, 10)

        match_score = max(0.0, match_score)

        # Determine if matched
        matched = match_score >= 95.0 and all(
            brk.severity != BreakSeverity.CRITICAL for brk in breaks
        )

        critical_breaks = sum(1 for b in breaks if b.severity == BreakSeverity.CRITICAL)

        can_auto_affirm = (
            matched and
            match_score == 100.0 and
            len(breaks) == 0
        )

        requires_manual_review = (
            not matched or
            critical_breaks > 0 or
            match_score < 95.0
        )

        # Update confirmation status
        if matched:
            conf_a.status = ConfirmationStatus.MATCHED
            conf_b.status = ConfirmationStatus.MATCHED
            conf_a.matched_time = datetime.utcnow()
            conf_b.matched_time = datetime.utcnow()
            conf_a.counterparty_confirmation_id = confirmation_b_id
            conf_b.counterparty_confirmation_id = confirmation_a_id
        else:
            conf_a.status = ConfirmationStatus.MISMATCHED
            conf_b.status = ConfirmationStatus.MISMATCHED

        conf_a.match_score = match_score
        conf_b.match_score = match_score
        conf_a.breaks = [b.break_id for b in breaks]
        conf_b.breaks = [b.break_id for b in breaks]

        # Store breaks
        for brk in breaks:
            self.breaks[brk.break_id] = brk

        # Create match result
        result = MatchResult(
            trade_id=conf_a.details.trade_id,
            confirmation_a_id=confirmation_a_id,
            confirmation_b_id=confirmation_b_id,
            matched=matched,
            match_score=match_score,
            breaks=breaks,
            num_breaks=len(breaks),
            critical_breaks=critical_breaks,
            can_auto_affirm=can_auto_affirm,
            requires_manual_review=requires_manual_review
        )

        self.matches[conf_a.details.trade_id] = result

        return result

    def _compare_field(
        self,
        field_name: str,
        value_a: Any,
        value_b: Any,
        conf_a: Confirmation,
        conf_b: Confirmation,
        break_type: BreakType,
        severity: BreakSeverity
    ) -> List[TradeBreak]:
        """Compare field values."""
        if value_a != value_b:
            return [self._create_break(
                field_name,
                value_a,
                value_b,
                conf_a,
                conf_b,
                break_type,
                severity
            )]
        return []

    def _compare_decimal(
        self,
        field_name: str,
        value_a: Decimal,
        value_b: Decimal,
        tolerance: Decimal,
        conf_a: Confirmation,
        conf_b: Confirmation,
        break_type: BreakType
    ) -> List[TradeBreak]:
        """Compare decimal values with tolerance."""
        if self.tolerance.strict_matching:
            if value_a != value_b:
                return [self._create_break(
                    field_name, value_a, value_b, conf_a, conf_b,
                    break_type, BreakSeverity.HIGH
                )]
        else:
            diff = abs(value_a - value_b)
            if diff > tolerance:
                severity = BreakSeverity.HIGH if diff > tolerance * 10 else BreakSeverity.MEDIUM
                return [self._create_break(
                    field_name, value_a, value_b, conf_a, conf_b,
                    break_type, severity,
                    difference=float(diff)
                )]

        return []

    def _create_break(
        self,
        field_name: str,
        value_a: Any,
        value_b: Any,
        conf_a: Confirmation,
        conf_b: Confirmation,
        break_type: BreakType,
        severity: BreakSeverity,
        difference: Optional[float] = None
    ) -> TradeBreak:
        """Create trade break."""
        return TradeBreak(
            break_type=break_type,
            severity=severity,
            confirmation_a_id=conf_a.confirmation_id,
            confirmation_b_id=conf_b.confirmation_id,
            field_name=field_name,
            expected_value=str(value_a),
            actual_value=str(value_b),
            difference=difference
        )

    def affirm_confirmation(
        self,
        confirmation_id: str,
        method: AffirmationMethod,
        affirmed_by: str
    ) -> Confirmation:
        """Affirm a confirmation."""
        conf = self.confirmations.get(confirmation_id)
        if not conf:
            raise ValueError(f"Confirmation {confirmation_id} not found")

        if conf.status != ConfirmationStatus.MATCHED:
            raise ValueError(f"Confirmation must be matched before affirmation")

        conf.status = ConfirmationStatus.AFFIRMED
        conf.affirmed_time = datetime.utcnow()
        conf.affirmation_method = method
        conf.affirmed_by = affirmed_by

        return conf

    def resolve_break(
        self,
        break_id: str,
        resolution: str,
        resolved_by: str
    ) -> TradeBreak:
        """Resolve a trade break."""
        brk = self.breaks.get(break_id)
        if not brk:
            raise ValueError(f"Break {break_id} not found")

        brk.resolved = True
        brk.resolution = resolution
        brk.resolved_by = resolved_by
        brk.resolved_time = datetime.utcnow()

        return brk

    def get_unmatched_confirmations(self) -> List[Confirmation]:
        """Get list of unmatched confirmations."""
        return [
            conf for conf in self.confirmations.values()
            if conf.status in [
                ConfirmationStatus.PENDING,
                ConfirmationStatus.SENT,
                ConfirmationStatus.RECEIVED,
                ConfirmationStatus.MISMATCHED
            ]
        ]

    def get_pending_breaks(self) -> List[TradeBreak]:
        """Get unresolved breaks."""
        return [brk for brk in self.breaks.values() if not brk.resolved]

    def get_statistics(self) -> Dict[str, Any]:
        """Get matching statistics."""
        return {
            "total_confirmations": self.statistics.total_confirmations,
            "matched_confirmations": self.statistics.matched_confirmations,
            "pending_confirmations": self.statistics.pending_confirmations,
            "match_rate": self.statistics.match_rate,
            "breaks_detected": self.statistics.breaks_detected,
            "critical_breaks": self.statistics.critical_breaks,
            "auto_affirmed": self.statistics.auto_affirmed,
            "manual_reviews": self.statistics.manual_reviews,
        }


class AllocationConfirmation(BaseModel):
    """Allocation confirmation for block trades / prime brokerage."""

    allocation_id: str = Field(
        default_factory=lambda: f"ALC-{uuid4().hex[:12].upper()}"
    )
    parent_trade_id: str = Field(..., description="Parent block trade ID")

    # Allocation details
    allocated_to: Party = Field(..., description="Allocation recipient")
    allocated_quantity: Decimal = Field(..., description="Allocated quantity")
    allocated_percentage: Decimal = Field(..., description="Percentage of parent")

    # Pricing
    allocation_price: Optional[Decimal] = Field(None, description="Allocation price")
    average_price: bool = Field(default=True, description="Use average price")

    # Accounts
    account: str = Field(..., description="Client account")
    settlement_account: Optional[str] = Field(None)

    # Status
    status: ConfirmationStatus = Field(default=ConfirmationStatus.PENDING)
    confirmed_time: Optional[datetime] = Field(None)

    created_time: datetime = Field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExternalSystem(str, Enum):
    """External systems participating in confirmation flow."""

    DTCC_CTM = "dtcc_ctm"
    MARKITWIRE = "markitwire"
    SWIFT = "swift"
    CCP = "ccp"
    PRIME_BROKER = "prime_broker"
    CLIENT_PORTAL = "client_portal"


class ExternalRecordStatus(str, Enum):
    """Status values reported by external confirmation systems."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    MATCHED = "matched"
    AFFIRMED = "affirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ReconciliationStatus(str, Enum):
    """Status of external reconciliation."""

    PENDING = "pending"
    RECONCILED = "reconciled"
    BROKEN = "broken"


class ExternalConfirmationRecord(BaseModel):
    """External status snapshot for a confirmation."""

    record_id: str = Field(default_factory=lambda: f"EXT-{uuid4().hex[:12].upper()}")
    trade_id: str = Field(..., description="Associated trade identifier")
    system: ExternalSystem = Field(..., description="External system source")
    status: ExternalRecordStatus = Field(..., description="External status")

    confirmation_id: Optional[str] = Field(
        None, description="Internal confirmation identifier if supplied"
    )
    direction: Optional[str] = Field(
        None, description="Side (buy/sell) from external perspective"
    )

    received_time: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)


class ReconciliationIssue(BaseModel):
    """Issue detected during reconciliation."""

    issue_id: str = Field(default_factory=lambda: f"ISS-{uuid4().hex[:12].upper()}")
    description: str = Field(..., description="Human readable description")
    severity: BreakSeverity = Field(..., description="Impact of the issue")
    break_id: Optional[str] = Field(None, description="Associated break identifier")


class ReconciliationResult(BaseModel):
    """Result of reconciling internal and external confirmations."""

    trade_id: str = Field(...)
    status: ReconciliationStatus = Field(...)
    issues: List[ReconciliationIssue] = Field(default_factory=list)
    confirmation_ids: List[str] = Field(default_factory=list)
    external_record_ids: List[str] = Field(default_factory=list)
    reconciled_time: datetime = Field(default_factory=datetime.utcnow)


class ConfirmationReconciliationEngine:
    """Coordinate confirmation matching with external system statuses."""

    def __init__(self, matcher: ConfirmationMatcher):
        self.matcher = matcher
        self.external_records: Dict[str, List[ExternalConfirmationRecord]] = defaultdict(list)
        self.results: Dict[str, ReconciliationResult] = {}

    def submit_confirmation(self, confirmation: Confirmation) -> Confirmation:
        """Register a confirmation with the underlying matcher."""
        return self.matcher.add_confirmation(confirmation)

    def record_external_confirmation(
        self, record: ExternalConfirmationRecord
    ) -> ExternalConfirmationRecord:
        """Record external system status and detect mismatches."""

        confirmations = self._match_confirmations(record)
        if not confirmations:
            # keep record but raise potential issue on reconciliation
            self.external_records[record.trade_id].append(record)
            self.reconcile_trade(record.trade_id)
            return record

        self.external_records[record.trade_id].append(record)

        for confirmation in confirmations:
            confirmation.metadata.setdefault("external_statuses", {})[
                record.system.value
            ] = record.status.value

            if not self._status_aligned(confirmation.status, record.status):
                brk = self._create_external_break(confirmation, record)
                self.matcher.breaks[brk.break_id] = brk

        # refresh reconciliation state with new record information
        self.reconcile_trade(record.trade_id)

        return record

    def reconcile_trade(self, trade_id: str) -> ReconciliationResult:
        """Reconcile confirmations for a trade with external statuses."""

        confirmations = self._find_confirmations(trade_id)
        external_records = self.external_records.get(trade_id, [])
        issues: List[ReconciliationIssue] = []

        if not confirmations:
            issues.append(
                ReconciliationIssue(
                    description="No internal confirmations available for trade",
                    severity=BreakSeverity.HIGH,
                )
            )

        if not external_records:
            issues.append(
                ReconciliationIssue(
                    description="No external confirmation statuses received",
                    severity=BreakSeverity.MEDIUM,
                )
            )

        status = ReconciliationStatus.RECONCILED
        if issues:
            status = ReconciliationStatus.PENDING

        for confirmation in confirmations:
            matching = self._matching_records_for_confirmation(confirmation, external_records)

            if not matching:
                issues.append(
                    ReconciliationIssue(
                        description=(
                            f"No external status for confirmation {confirmation.confirmation_id}"
                        ),
                        severity=BreakSeverity.MEDIUM,
                    )
                )
                status = ReconciliationStatus.PENDING
                continue

            latest_record = max(matching, key=lambda rec: rec.received_time)
            if not self._status_aligned(confirmation.status, latest_record.status):
                brk = self._create_external_break(confirmation, latest_record)
                self.matcher.breaks[brk.break_id] = brk

                issues.append(
                    ReconciliationIssue(
                        description=(
                            "Status mismatch between internal confirmation "
                            f"{confirmation.confirmation_id} ({confirmation.status.value}) "
                            f"and {latest_record.system.value} "
                            f"({latest_record.status.value})"
                        ),
                        severity=BreakSeverity.HIGH,
                        break_id=brk.break_id,
                    )
                )
                status = ReconciliationStatus.BROKEN

        result = ReconciliationResult(
            trade_id=trade_id,
            status=status,
            issues=issues,
            confirmation_ids=[conf.confirmation_id for conf in confirmations],
            external_record_ids=[record.record_id for record in external_records],
        )

        self.results[trade_id] = result
        return result

    def affirm_trade(
        self,
        trade_id: str,
        method: AffirmationMethod,
        affirmed_by: str,
    ) -> List[Confirmation]:
        """Affirm confirmations for a trade once reconciliation succeeds."""

        result = self.reconcile_trade(trade_id)
        if result.status != ReconciliationStatus.RECONCILED:
            raise ValueError(
                "Cannot affirm trade until reconciliation status is RECONCILED"
            )

        affirmed: List[Confirmation] = []
        for confirmation in self._find_confirmations(trade_id):
            if confirmation.status != ConfirmationStatus.AFFIRMED:
                affirmed.append(
                    self.matcher.affirm_confirmation(
                        confirmation.confirmation_id, method, affirmed_by
                    )
                )
            else:
                affirmed.append(confirmation)

        return affirmed

    def get_outstanding_affirmations(self) -> List[str]:
        """Return trade IDs that still require affirmation or reconciliation."""

        outstanding: List[str] = []
        for trade_id, result in self.results.items():
            if result.status != ReconciliationStatus.RECONCILED:
                outstanding.append(trade_id)
                continue

            confirmations = self._find_confirmations(trade_id)
            if any(conf.status != ConfirmationStatus.AFFIRMED for conf in confirmations):
                outstanding.append(trade_id)

        return outstanding

    def get_external_records(self, trade_id: str) -> List[ExternalConfirmationRecord]:
        """Return all external records captured for a trade."""

        return list(self.external_records.get(trade_id, []))

    def _match_confirmations(
        self, record: ExternalConfirmationRecord
    ) -> List[Confirmation]:
        """Find confirmations associated with an external record."""

        confirmations: List[Confirmation] = []

        if record.confirmation_id:
            confirmation = self.matcher.confirmations.get(record.confirmation_id)
            if confirmation:
                confirmations.append(confirmation)
                return confirmations

        for confirmation in self._find_confirmations(record.trade_id):
            if record.direction and confirmation.direction != record.direction:
                continue
            confirmations.append(confirmation)

        return confirmations

    def _find_confirmations(self, trade_id: str) -> List[Confirmation]:
        """Helper to fetch confirmations for a trade."""

        return [
            conf
            for conf in self.matcher.confirmations.values()
            if conf.details.trade_id == trade_id
        ]

    @staticmethod
    def _matching_records_for_confirmation(
        confirmation: Confirmation,
        external_records: Iterable[ExternalConfirmationRecord],
    ) -> List[ExternalConfirmationRecord]:
        """Filter external records relevant for a confirmation."""

        candidates = []
        for record in external_records:
            if record.confirmation_id and record.confirmation_id != confirmation.confirmation_id:
                continue

            if record.direction and record.direction != confirmation.direction:
                continue

            candidates.append(record)

        return candidates

    @staticmethod
    def _status_aligned(
        confirmation_status: ConfirmationStatus,
        external_status: ExternalRecordStatus,
    ) -> bool:
        """Determine whether internal/external statuses align."""

        alignment = {
            ConfirmationStatus.PENDING: {
                ExternalRecordStatus.PENDING,
                ExternalRecordStatus.ACKNOWLEDGED,
            },
            ConfirmationStatus.SENT: {
                ExternalRecordStatus.PENDING,
                ExternalRecordStatus.ACKNOWLEDGED,
            },
            ConfirmationStatus.RECEIVED: {
                ExternalRecordStatus.ACKNOWLEDGED,
            },
            ConfirmationStatus.MATCHED: {
                ExternalRecordStatus.MATCHED,
                ExternalRecordStatus.AFFIRMED,
            },
            ConfirmationStatus.AFFIRMED: {
                ExternalRecordStatus.AFFIRMED,
            },
            ConfirmationStatus.REJECTED: {
                ExternalRecordStatus.REJECTED,
                ExternalRecordStatus.CANCELLED,
            },
            ConfirmationStatus.CANCELLED: {
                ExternalRecordStatus.CANCELLED,
            },
            ConfirmationStatus.MISMATCHED: {
                ExternalRecordStatus.PENDING,
                ExternalRecordStatus.MATCHED,
                ExternalRecordStatus.AFFIRMED,
            },
        }

        allowed = alignment.get(confirmation_status)
        if not allowed:
            return True

        return external_status in allowed

    @staticmethod
    def _create_external_break(
        confirmation: Confirmation,
        record: ExternalConfirmationRecord,
    ) -> TradeBreak:
        """Create break entry for external mismatch."""

        return TradeBreak(
            break_type=BreakType.EXTERNAL_STATUS_MISMATCH,
            severity=BreakSeverity.HIGH,
            confirmation_a_id=confirmation.confirmation_id,
            confirmation_b_id=record.confirmation_id or record.record_id,
            field_name="status",
            expected_value=confirmation.status.value,
            actual_value=record.status.value,
            metadata={
                "system": record.system.value,
                "record_id": record.record_id,
            },
        )


__all__ = [
    "Confirmation",
    "ConfirmationStatus",
    "ConfirmationDetails",
    "TradeBreak",
    "BreakType",
    "BreakSeverity",
    "MatchResult",
    "MatchStatistics",
    "ConfirmationMatcher",
    "ToleranceConfig",
    "AffirmationMethod",
    "AllocationConfirmation",
    "ExternalSystem",
    "ExternalRecordStatus",
    "ExternalConfirmationRecord",
    "ReconciliationStatus",
    "ReconciliationIssue",
    "ReconciliationResult",
    "ConfirmationReconciliationEngine",
]
