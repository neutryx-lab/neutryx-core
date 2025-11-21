"""Trade confirmation matching and affirmation workflows.

This module handles post-trade confirmation processes:
- Confirmation generation and matching
- Affirmation workflows
- Break management and resolution
- Electronic confirmation via platforms (MarkitSERV, AcadiaSoft, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class ConfirmationStatus(Enum):
    """Status of trade confirmation."""

    PENDING = "pending"  # Awaiting counterparty confirmation
    MATCHED = "matched"  # Confirmations match
    MISMATCHED = "mismatched"  # Discrepancies found
    AFFIRMED = "affirmed"  # Both parties affirmed
    REJECTED = "rejected"  # Counterparty rejected
    CANCELLED = "cancelled"  # Confirmation cancelled
    EXPIRED = "expired"  # Confirmation expired


class BreakType(Enum):
    """Types of confirmation breaks."""

    ECONOMIC = "economic"  # Pricing or economic terms differ
    LEGAL = "legal"  # Legal terms differ
    OPERATIONAL = "operational"  # Settlement or operational details differ
    MISSING = "missing"  # Confirmation missing from counterparty


class BreakSeverity(Enum):
    """Severity of confirmation break."""

    LOW = "low"  # Minor discrepancy
    MEDIUM = "medium"  # Moderate discrepancy
    HIGH = "high"  # Material discrepancy
    CRITICAL = "critical"  # Critical issue requiring immediate attention


class AffirmationMethod(Enum):
    """Method of trade affirmation."""

    MANUAL = "manual"  # Manual review and approval
    ELECTRONIC = "electronic"  # Electronic confirmation platform
    EMAIL = "email"  # Email confirmation
    PHONE = "phone"  # Phone confirmation
    AUTO_AFFIRM = "auto_affirm"  # Automatic affirmation


class TradeConfirmation(BaseModel):
    """Trade confirmation document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    confirmation_id: str = Field(default_factory=lambda: f"CONF-{uuid4().hex[:12].upper()}")
    trade_id: str = Field(..., description="Associated trade ID")
    party_id: str = Field(..., description="Party sending confirmation")
    counterparty_id: str = Field(..., description="Counterparty receiving confirmation")
    trade_date: date = Field(..., description="Trade execution date")
    settlement_date: date = Field(..., description="Settlement date")
    product_type: str = Field(..., description="Product type")
    economic_terms: Dict[str, Any] = Field(..., description="Economic terms of the trade")
    legal_terms: Dict[str, str] = Field(default_factory=dict, description="Legal documentation")
    operational_details: Dict[str, str] = Field(default_factory=dict)
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    generated_at: datetime = Field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    matched_at: Optional[datetime] = None
    affirmed_at: Optional[datetime] = None
    expiration_date: date = Field(
        default_factory=lambda: date.today() + timedelta(days=3),
        description="Confirmation expiration date (T+3)"
    )
    metadata: Dict[str, str] = Field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if confirmation has expired."""
        return date.today() > self.expiration_date and self.status == ConfirmationStatus.PENDING


class ConfirmationBreak(BaseModel):
    """Confirmation discrepancy/break."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    break_id: str = Field(default_factory=lambda: f"BRK-{uuid4().hex[:12].upper()}")
    confirmation_id: str
    trade_id: str
    break_type: BreakType
    severity: BreakSeverity
    field_name: str = Field(..., description="Field with discrepancy")
    expected_value: Any = Field(..., description="Expected value")
    actual_value: Any = Field(..., description="Actual value from counterparty")
    description: str = Field(..., description="Description of break")
    detected_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    assigned_to: Optional[str] = None


@dataclass
class MatchingResult:
    """Result of confirmation matching."""

    is_matched: bool
    confirmation_id: str
    trade_id: str
    breaks: List[ConfirmationBreak] = field(default_factory=list)
    matched_fields: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_breaks(self) -> bool:
        """Check if matching found any breaks."""
        return len(self.breaks) > 0

    @property
    def break_count(self) -> int:
        """Get number of breaks."""
        return len(self.breaks)


class AffirmationRequest(BaseModel):
    """Request for trade affirmation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    confirmation_id: str
    trade_id: str
    method: AffirmationMethod
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.now)
    response_deadline: datetime = Field(
        default_factory=lambda: datetime.now() + timedelta(hours=24)
    )
    notes: Optional[str] = None


class AffirmationResponse(BaseModel):
    """Response to affirmation request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    confirmation_id: str
    trade_id: str
    affirmed: bool
    responded_by: str
    responded_at: datetime = Field(default_factory=datetime.now)
    comments: Optional[str] = None
    exceptions: List[str] = Field(default_factory=list)


class ConfirmationManager:
    """Manages trade confirmation and affirmation workflows.

    Example:
        >>> manager = ConfirmationManager()
        >>>
        >>> # Generate confirmation
        >>> confirmation = manager.generate_confirmation(
        ...     trade_id="TRD-001",
        ...     party_id="BANK-A",
        ...     counterparty_id="BANK-B",
        ...     trade_date=date.today(),
        ...     settlement_date=date.today() + timedelta(days=2),
        ...     product_type="interest_rate_swap",
        ...     economic_terms={
        ...         "notional": 10_000_000,
        ...         "fixed_rate": 0.05,
        ...         "tenor": 5
        ...     }
        ... )
        >>>
        >>> # Send confirmation
        >>> manager.send_confirmation(confirmation.confirmation_id)
        >>>
        >>> # Match with counterparty confirmation
        >>> result = manager.match_confirmation(
        ...     confirmation.confirmation_id,
        ...     counterparty_confirmation
        ... )
        >>>
        >>> # Affirm if matched
        >>> if result.is_matched:
        ...     manager.affirm_confirmation(confirmation.confirmation_id)
    """

    def __init__(self, tolerance: Optional[Dict[str, float]] = None):
        """Initialize confirmation manager.

        Args:
            tolerance: Tolerance thresholds for matching (e.g., {"price": 0.01})
        """
        self._confirmations: Dict[str, TradeConfirmation] = {}
        self._breaks: Dict[str, List[ConfirmationBreak]] = {}  # trade_id -> breaks
        self._affirmation_requests: Dict[str, AffirmationRequest] = {}
        self._affirmation_responses: Dict[str, AffirmationResponse] = {}
        self._tolerance = tolerance or {}

    def generate_confirmation(
        self,
        trade_id: str,
        party_id: str,
        counterparty_id: str,
        trade_date: date,
        settlement_date: date,
        product_type: str,
        economic_terms: Dict[str, Any],
        legal_terms: Optional[Dict[str, str]] = None,
        operational_details: Optional[Dict[str, str]] = None,
    ) -> TradeConfirmation:
        """Generate trade confirmation.

        Args:
            trade_id: Trade ID
            party_id: Party generating confirmation
            counterparty_id: Counterparty
            trade_date: Trade date
            settlement_date: Settlement date
            product_type: Product type
            economic_terms: Economic terms
            legal_terms: Legal terms
            operational_details: Operational details

        Returns:
            Generated confirmation
        """
        confirmation = TradeConfirmation(
            trade_id=trade_id,
            party_id=party_id,
            counterparty_id=counterparty_id,
            trade_date=trade_date,
            settlement_date=settlement_date,
            product_type=product_type,
            economic_terms=economic_terms,
            legal_terms=legal_terms or {},
            operational_details=operational_details or {},
        )

        self._confirmations[confirmation.confirmation_id] = confirmation
        return confirmation

    def send_confirmation(self, confirmation_id: str) -> None:
        """Send confirmation to counterparty.

        Args:
            confirmation_id: Confirmation ID

        Raises:
            ValueError: If confirmation not found
        """
        if confirmation_id not in self._confirmations:
            raise ValueError(f"Confirmation {confirmation_id} not found")

        confirmation = self._confirmations[confirmation_id]
        confirmation.sent_at = datetime.now()

    def match_confirmation(
        self,
        confirmation_id: str,
        counterparty_confirmation: TradeConfirmation,
    ) -> MatchingResult:
        """Match confirmation with counterparty's version.

        Args:
            confirmation_id: Our confirmation ID
            counterparty_confirmation: Counterparty's confirmation

        Returns:
            Matching result with any breaks identified

        Raises:
            ValueError: If confirmation not found
        """
        if confirmation_id not in self._confirmations:
            raise ValueError(f"Confirmation {confirmation_id} not found")

        our_conf = self._confirmations[confirmation_id]

        # Verify trade IDs match
        if our_conf.trade_id != counterparty_confirmation.trade_id:
            break_obj = ConfirmationBreak(
                confirmation_id=confirmation_id,
                trade_id=our_conf.trade_id,
                break_type=BreakType.OPERATIONAL,
                severity=BreakSeverity.CRITICAL,
                field_name="trade_id",
                expected_value=our_conf.trade_id,
                actual_value=counterparty_confirmation.trade_id,
                description="Trade ID mismatch",
            )
            return MatchingResult(
                is_matched=False,
                confirmation_id=confirmation_id,
                trade_id=our_conf.trade_id,
                breaks=[break_obj],
            )

        breaks = []
        matched_fields = set()

        # Match economic terms
        for field, expected_value in our_conf.economic_terms.items():
            actual_value = counterparty_confirmation.economic_terms.get(field)

            if self._values_match(field, expected_value, actual_value):
                matched_fields.add(f"economic_terms.{field}")
            else:
                severity = self._determine_severity(field, expected_value, actual_value)
                break_obj = ConfirmationBreak(
                    confirmation_id=confirmation_id,
                    trade_id=our_conf.trade_id,
                    break_type=BreakType.ECONOMIC,
                    severity=severity,
                    field_name=field,
                    expected_value=expected_value,
                    actual_value=actual_value,
                    description=f"Economic term mismatch: {field}",
                )
                breaks.append(break_obj)

        # Match dates
        if our_conf.trade_date == counterparty_confirmation.trade_date:
            matched_fields.add("trade_date")
        else:
            breaks.append(
                ConfirmationBreak(
                    confirmation_id=confirmation_id,
                    trade_id=our_conf.trade_id,
                    break_type=BreakType.OPERATIONAL,
                    severity=BreakSeverity.HIGH,
                    field_name="trade_date",
                    expected_value=our_conf.trade_date,
                    actual_value=counterparty_confirmation.trade_date,
                    description="Trade date mismatch",
                )
            )

        if our_conf.settlement_date == counterparty_confirmation.settlement_date:
            matched_fields.add("settlement_date")
        else:
            breaks.append(
                ConfirmationBreak(
                    confirmation_id=confirmation_id,
                    trade_id=our_conf.trade_id,
                    break_type=BreakType.OPERATIONAL,
                    severity=BreakSeverity.MEDIUM,
                    field_name="settlement_date",
                    expected_value=our_conf.settlement_date,
                    actual_value=counterparty_confirmation.settlement_date,
                    description="Settlement date mismatch",
                )
            )

        # Store breaks
        if breaks:
            self._breaks[our_conf.trade_id] = breaks

        # Update confirmation status
        if not breaks:
            our_conf.status = ConfirmationStatus.MATCHED
            our_conf.matched_at = datetime.now()
        else:
            our_conf.status = ConfirmationStatus.MISMATCHED

        return MatchingResult(
            is_matched=len(breaks) == 0,
            confirmation_id=confirmation_id,
            trade_id=our_conf.trade_id,
            breaks=breaks,
            matched_fields=matched_fields,
        )

    def _values_match(self, field_name: str, expected: Any, actual: Any) -> bool:
        """Check if values match within tolerance."""
        if expected == actual:
            return True

        # Apply tolerance for numeric fields
        if field_name in self._tolerance and isinstance(expected, (int, float)):
            tolerance = self._tolerance[field_name]
            if isinstance(actual, (int, float)):
                return abs(expected - actual) <= tolerance

        return False

    def _determine_severity(self, field_name: str, expected: Any, actual: Any) -> BreakSeverity:
        """Determine severity of a break."""
        # Critical fields
        if field_name in ["notional", "strike", "fixed_rate"]:
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                diff_pct = abs(expected - actual) / expected if expected != 0 else float("inf")
                if diff_pct > 0.10:  # >10% difference
                    return BreakSeverity.CRITICAL
                elif diff_pct > 0.05:  # >5% difference
                    return BreakSeverity.HIGH
                elif diff_pct > 0.01:  # >1% difference
                    return BreakSeverity.MEDIUM
                else:
                    return BreakSeverity.LOW

        # Default severity
        return BreakSeverity.MEDIUM

    def affirm_confirmation(
        self,
        confirmation_id: str,
        method: AffirmationMethod = AffirmationMethod.MANUAL,
        affirmed_by: Optional[str] = None,
    ) -> None:
        """Affirm a matched confirmation.

        Args:
            confirmation_id: Confirmation ID
            method: Affirmation method
            affirmed_by: User affirming

        Raises:
            ValueError: If confirmation not found or not matched
        """
        if confirmation_id not in self._confirmations:
            raise ValueError(f"Confirmation {confirmation_id} not found")

        confirmation = self._confirmations[confirmation_id]

        if confirmation.status != ConfirmationStatus.MATCHED:
            raise ValueError(
                f"Cannot affirm confirmation {confirmation_id} "
                f"with status {confirmation.status}"
            )

        confirmation.status = ConfirmationStatus.AFFIRMED
        confirmation.affirmed_at = datetime.now()

    def resolve_break(
        self,
        break_id: str,
        resolution_notes: str,
        resolved_by: str,
    ) -> None:
        """Resolve a confirmation break.

        Args:
            break_id: Break ID
            resolution_notes: Notes on resolution
            resolved_by: User resolving break

        Raises:
            ValueError: If break not found
        """
        # Find break
        break_obj = None
        for breaks in self._breaks.values():
            for b in breaks:
                if b.break_id == break_id:
                    break_obj = b
                    break
            if break_obj:
                break

        if not break_obj:
            raise ValueError(f"Break {break_id} not found")

        break_obj.resolved_at = datetime.now()
        break_obj.resolution_notes = resolution_notes
        break_obj.assigned_to = resolved_by

    def get_confirmation(self, confirmation_id: str) -> Optional[TradeConfirmation]:
        """Get confirmation by ID.

        Args:
            confirmation_id: Confirmation ID

        Returns:
            Confirmation or None
        """
        return self._confirmations.get(confirmation_id)

    def get_breaks(self, trade_id: str) -> List[ConfirmationBreak]:
        """Get breaks for a trade.

        Args:
            trade_id: Trade ID

        Returns:
            List of breaks
        """
        return self._breaks.get(trade_id, [])

    def get_unresolved_breaks(self, severity: Optional[BreakSeverity] = None) -> List[ConfirmationBreak]:
        """Get all unresolved breaks.

        Args:
            severity: Filter by severity

        Returns:
            List of unresolved breaks
        """
        unresolved = []
        for breaks in self._breaks.values():
            for break_obj in breaks:
                if break_obj.resolved_at is None:
                    if severity is None or break_obj.severity == severity:
                        unresolved.append(break_obj)
        return unresolved

    def get_statistics(self) -> Dict[str, Any]:
        """Get confirmation statistics.

        Returns:
            Dictionary with statistics
        """
        total_confirmations = len(self._confirmations)
        matched = sum(1 for c in self._confirmations.values() if c.status == ConfirmationStatus.MATCHED)
        affirmed = sum(1 for c in self._confirmations.values() if c.status == ConfirmationStatus.AFFIRMED)
        mismatched = sum(1 for c in self._confirmations.values() if c.status == ConfirmationStatus.MISMATCHED)

        total_breaks = sum(len(breaks) for breaks in self._breaks.values())
        unresolved_breaks = len(self.get_unresolved_breaks())

        return {
            "total_confirmations": total_confirmations,
            "matched": matched,
            "affirmed": affirmed,
            "mismatched": mismatched,
            "match_rate": matched / total_confirmations if total_confirmations else 0.0,
            "affirmation_rate": affirmed / total_confirmations if total_confirmations else 0.0,
            "total_breaks": total_breaks,
            "unresolved_breaks": unresolved_breaks,
        }


__all__ = [
    "ConfirmationStatus",
    "BreakType",
    "BreakSeverity",
    "AffirmationMethod",
    "TradeConfirmation",
    "ConfirmationBreak",
    "MatchingResult",
    "AffirmationRequest",
    "AffirmationResponse",
    "ConfirmationManager",
]
