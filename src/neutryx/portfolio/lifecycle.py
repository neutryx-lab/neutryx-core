"""Trade lifecycle management for amendments, novations, and events.

Provides tracking and management of trade lifecycle events:
- Amendments (rate changes, notional changes, maturity extensions)
- Novations (transfer to new counterparty)
- Terminations (early termination, partial termination)
- Version control for trade changes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from neutryx.contracts.trade import Trade, TradeStatus


class LifecycleEventType(Enum):
    """Types of trade lifecycle events."""

    AMENDMENT = "amendment"
    NOVATION = "novation"
    TERMINATION = "termination"
    PARTIAL_TERMINATION = "partial_termination"
    ASSIGNMENT = "assignment"
    COMPRESSION = "compression"
    STATUS_CHANGE = "status_change"


@dataclass
class LifecycleEvent:
    """Record of a trade lifecycle event."""

    event_id: str
    trade_id: str
    event_type: LifecycleEventType
    event_date: date
    effective_date: date
    description: str
    changes: Dict[str, Any]  # Field name -> new value
    previous_values: Dict[str, Any]  # Field name -> old value
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, str] = field(default_factory=dict)


class TradeAmendment(BaseModel):
    """Request to amend a trade."""

    trade_id: str
    amendment_date: date = Field(default_factory=date.today)
    effective_date: Optional[date] = None
    changes: Dict[str, Any] = Field(..., description="Fields to change with new values")
    reason: Optional[str] = None
    amended_by: Optional[str] = None


class TradeNovation(BaseModel):
    """Request to novate a trade to a new counterparty."""

    trade_id: str
    novation_date: date = Field(default_factory=date.today)
    effective_date: Optional[date] = None
    new_counterparty_id: str
    new_netting_set_id: Optional[str] = None
    reason: Optional[str] = None
    novated_by: Optional[str] = None


class TradeTermination(BaseModel):
    """Request to terminate a trade."""

    trade_id: str
    termination_date: date = Field(default_factory=date.today)
    termination_payment: Optional[float] = None
    reason: Optional[str] = None
    terminated_by: Optional[str] = None


@dataclass
class TradeVersion:
    """Snapshot of trade state at a point in time."""

    version_number: int
    trade_id: str
    snapshot_date: datetime
    trade_data: Dict[str, Any]  # Serialized trade state
    event_id: Optional[str] = None  # Link to lifecycle event
    comment: Optional[str] = None


class LifecycleManager:
    """Manages trade lifecycle events and versioning.

    Example:
        >>> manager = LifecycleManager()
        >>> trade = Trade(id="TRD-001", ...)
        >>> 
        >>> # Amend notional
        >>> amendment = TradeAmendment(
        ...     trade_id="TRD-001",
        ...     changes={"notional": 2_000_000},
        ...     reason="Client requested increase"
        ... )
        >>> event = manager.amend_trade(trade, amendment)
        >>> 
        >>> # View history
        >>> history = manager.get_trade_history("TRD-001")
    """

    def __init__(self):
        """Initialize lifecycle manager."""
        self._events: Dict[str, List[LifecycleEvent]] = {}  # trade_id -> events
        self._versions: Dict[str, List[TradeVersion]] = {}  # trade_id -> versions
        self._event_counter = 0

    def amend_trade(self, trade: Trade, amendment: TradeAmendment) -> LifecycleEvent:
        """Apply an amendment to a trade.

        Args:
            trade: Trade to amend
            amendment: Amendment details

        Returns:
            Lifecycle event record

        Raises:
            ValueError: If amendment is invalid
        """
        if trade.id != amendment.trade_id:
            raise ValueError(f"Trade ID mismatch: {trade.id} != {amendment.trade_id}")

        # Capture current state
        previous_values = {}
        for field_name, new_value in amendment.changes.items():
            if hasattr(trade, field_name):
                previous_values[field_name] = getattr(trade, field_name)
            else:
                raise ValueError(f"Invalid field for amendment: {field_name}")

        # Create version snapshot before changes
        self._create_version_snapshot(trade, "Before amendment")

        # Apply changes
        for field_name, new_value in amendment.changes.items():
            setattr(trade, field_name, new_value)

        # Create event record
        event = self._create_event(
            trade_id=trade.id,
            event_type=LifecycleEventType.AMENDMENT,
            event_date=amendment.amendment_date,
            effective_date=amendment.effective_date or amendment.amendment_date,
            description=amendment.reason or "Trade amended",
            changes=amendment.changes,
            previous_values=previous_values,
            created_by=amendment.amended_by,
        )

        # Create version snapshot after changes
        self._create_version_snapshot(trade, f"After amendment {event.event_id}", event.event_id)

        return event

    def novate_trade(self, trade: Trade, novation: TradeNovation) -> LifecycleEvent:
        """Novate a trade to a new counterparty.

        Args:
            trade: Trade to novate
            novation: Novation details

        Returns:
            Lifecycle event record
        """
        if trade.id != novation.trade_id:
            raise ValueError(f"Trade ID mismatch: {trade.id} != {novation.trade_id}")

        # Capture current state
        previous_values = {
            "counterparty_id": trade.counterparty_id,
            "netting_set_id": trade.netting_set_id,
            "status": trade.status,
        }

        # Create version snapshot
        self._create_version_snapshot(trade, "Before novation")

        # Apply novation
        trade.counterparty_id = novation.new_counterparty_id
        if novation.new_netting_set_id:
            trade.netting_set_id = novation.new_netting_set_id
        trade.status = TradeStatus.NOVATED

        changes = {
            "counterparty_id": novation.new_counterparty_id,
            "netting_set_id": novation.new_netting_set_id,
            "status": TradeStatus.NOVATED,
        }

        # Create event record
        event = self._create_event(
            trade_id=trade.id,
            event_type=LifecycleEventType.NOVATION,
            event_date=novation.novation_date,
            effective_date=novation.effective_date or novation.novation_date,
            description=novation.reason or f"Novated to {novation.new_counterparty_id}",
            changes=changes,
            previous_values=previous_values,
            created_by=novation.novated_by,
        )

        self._create_version_snapshot(trade, f"After novation {event.event_id}", event.event_id)

        return event

    def terminate_trade(self, trade: Trade, termination: TradeTermination) -> LifecycleEvent:
        """Terminate a trade.

        Args:
            trade: Trade to terminate
            termination: Termination details

        Returns:
            Lifecycle event record
        """
        if trade.id != termination.trade_id:
            raise ValueError(f"Trade ID mismatch: {trade.id} != {termination.trade_id}")

        previous_values = {"status": trade.status}

        self._create_version_snapshot(trade, "Before termination")

        # Mark as terminated
        trade.status = TradeStatus.TERMINATED

        changes = {"status": TradeStatus.TERMINATED}
        if termination.termination_payment:
            changes["termination_payment"] = termination.termination_payment

        event = self._create_event(
            trade_id=trade.id,
            event_type=LifecycleEventType.TERMINATION,
            event_date=termination.termination_date,
            effective_date=termination.termination_date,
            description=termination.reason or "Trade terminated",
            changes=changes,
            previous_values=previous_values,
            created_by=termination.terminated_by,
        )

        self._create_version_snapshot(trade, f"After termination {event.event_id}", event.event_id)

        return event

    def _create_event(
        self,
        trade_id: str,
        event_type: LifecycleEventType,
        event_date: date,
        effective_date: date,
        description: str,
        changes: Dict[str, Any],
        previous_values: Dict[str, Any],
        created_by: Optional[str] = None,
    ) -> LifecycleEvent:
        """Create and store a lifecycle event."""
        self._event_counter += 1
        event_id = f"EVT-{self._event_counter:06d}"

        event = LifecycleEvent(
            event_id=event_id,
            trade_id=trade_id,
            event_type=event_type,
            event_date=event_date,
            effective_date=effective_date,
            description=description,
            changes=changes,
            previous_values=previous_values,
            created_by=created_by,
        )

        if trade_id not in self._events:
            self._events[trade_id] = []
        self._events[trade_id].append(event)

        return event

    def _create_version_snapshot(self, trade: Trade, comment: str, event_id: Optional[str] = None) -> TradeVersion:
        """Create a version snapshot of a trade."""
        if trade.id not in self._versions:
            self._versions[trade.id] = []

        version_number = len(self._versions[trade.id]) + 1

        # Serialize trade state
        trade_data = trade.model_dump()

        version = TradeVersion(
            version_number=version_number,
            trade_id=trade.id,
            snapshot_date=datetime.now(),
            trade_data=trade_data,
            event_id=event_id,
            comment=comment,
        )

        self._versions[trade.id].append(version)
        return version

    def get_trade_history(self, trade_id: str) -> List[LifecycleEvent]:
        """Get all lifecycle events for a trade.

        Args:
            trade_id: Trade ID

        Returns:
            List of lifecycle events in chronological order
        """
        return self._events.get(trade_id, [])

    def get_trade_versions(self, trade_id: str) -> List[TradeVersion]:
        """Get all version snapshots for a trade.

        Args:
            trade_id: Trade ID

        Returns:
            List of trade versions in chronological order
        """
        return self._versions.get(trade_id, [])

    def get_version(self, trade_id: str, version_number: int) -> Optional[TradeVersion]:
        """Get a specific version of a trade.

        Args:
            trade_id: Trade ID
            version_number: Version number (1-indexed)

        Returns:
            Trade version or None if not found
        """
        versions = self._versions.get(trade_id, [])
        if 1 <= version_number <= len(versions):
            return versions[version_number - 1]
        return None

    def get_events_by_type(self, event_type: LifecycleEventType) -> List[LifecycleEvent]:
        """Get all events of a specific type.

        Args:
            event_type: Event type to filter by

        Returns:
            List of matching events
        """
        all_events = []
        for events in self._events.values():
            all_events.extend(e for e in events if e.event_type == event_type)
        return all_events

    def get_events_by_date_range(self, start_date: date, end_date: date) -> List[LifecycleEvent]:
        """Get all events within a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of events in date range
        """
        all_events = []
        for events in self._events.values():
            all_events.extend(e for e in events if start_date <= e.event_date <= end_date)
        return all_events


__all__ = [
    "LifecycleEventType",
    "LifecycleEvent",
    "TradeAmendment",
    "TradeNovation",
    "TradeTermination",
    "TradeVersion",
    "LifecycleManager",
]
