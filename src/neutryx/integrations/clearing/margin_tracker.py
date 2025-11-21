"""Enhanced margin tracking for CCP connectors.

This module provides advanced margin tracking capabilities for CCP connectors:
1. Historical margin requirement tracking
2. Margin call monitoring and alerting
3. Margin change analytics
4. Integration with margin aggregation service
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import CCPConnector, MarginCall


class MarginChangeType(str, Enum):
    """Type of margin change."""
    INCREASE = "increase"
    DECREASE = "decrease"
    NEW_CALL = "new_call"
    CALL_SATISFIED = "call_satisfied"
    POSITION_CHANGE = "position_change"


class MarginSnapshot(BaseModel):
    """Snapshot of margin requirements at a point in time."""

    snapshot_id: str = Field(..., description="Snapshot identifier")
    ccp_name: str = Field(..., description="CCP name")
    member_id: str = Field(..., description="Member ID")
    snapshot_time: datetime = Field(default_factory=datetime.utcnow)
    as_of_date: date = Field(..., description="Reporting date")

    # Margin components
    initial_margin: Decimal = Field(..., description="Initial margin")
    variation_margin: Decimal = Field(..., description="Variation margin")
    additional_margin: Decimal = Field(default=Decimal("0"), description="Additional margin")
    liquidity_addon: Decimal = Field(default=Decimal("0"), description="Liquidity add-on")
    concentration_addon: Decimal = Field(default=Decimal("0"), description="Concentration charge")
    total_margin: Decimal = Field(..., description="Total margin required")

    # Breakdown by portfolio/product
    margin_by_portfolio: Dict[str, Decimal] = Field(default_factory=dict)
    margin_by_product: Dict[str, Decimal] = Field(default_factory=dict)

    # Posted collateral
    posted_margin: Decimal = Field(default=Decimal("0"), description="Posted margin")
    collateral_by_type: Dict[str, Decimal] = Field(default_factory=dict)

    # Deficit/surplus
    margin_deficit: Decimal = Field(default=Decimal("0"), description="Margin deficit")

    # Portfolio metrics
    num_trades: Optional[int] = Field(None, description="Number of trades")
    gross_notional: Optional[Decimal] = Field(None, description="Gross notional")
    net_exposure: Optional[Decimal] = Field(None, description="Net exposure")

    currency: str = Field(default="USD", description="Currency")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarginChangeEvent(BaseModel):
    """Event recording a change in margin requirements."""

    event_id: str = Field(..., description="Event identifier")
    ccp_name: str = Field(..., description="CCP name")
    event_time: datetime = Field(default_factory=datetime.utcnow)
    change_type: MarginChangeType = Field(..., description="Type of change")

    # Before/after values
    previous_margin: Decimal = Field(..., description="Previous margin requirement")
    new_margin: Decimal = Field(..., description="New margin requirement")
    change_amount: Decimal = Field(..., description="Change amount")
    change_percentage: float = Field(..., description="Change percentage")

    # Reason for change
    reason: Optional[str] = Field(None, description="Reason for change")
    triggering_event: Optional[str] = Field(None, description="Triggering event")

    # Related margin call
    margin_call_id: Optional[str] = Field(None, description="Related margin call ID")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarginCallRecord(BaseModel):
    """Extended margin call record with tracking."""

    margin_call: MarginCall = Field(..., description="Original margin call")

    # Status tracking
    acknowledged_time: Optional[datetime] = Field(None)
    satisfied_time: Optional[datetime] = Field(None)
    overdue: bool = Field(default=False)

    # Response tracking
    response_time_minutes: Optional[float] = Field(None, description="Time to satisfy call")

    # Collateral posted
    collateral_posted: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Collateral posted to satisfy call"
    )

    # Follow-up calls
    follow_up_calls: List[str] = Field(default_factory=list, description="Follow-up call IDs")

    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class MarginTrackerConfig:
    """Configuration for margin tracker."""

    # Snapshot frequency
    snapshot_interval_hours: int = 6  # Take snapshot every 6 hours
    retain_snapshots_days: int = 90  # Keep 90 days of history

    # Change detection
    significant_change_threshold_pct: float = 5.0  # 5% change is significant
    large_change_threshold: Decimal = Decimal("1000000")  # $1M absolute change

    # Margin call tracking
    margin_call_alert_enabled: bool = True
    margin_call_overdue_hours: int = 4  # Mark overdue after 4 hours

    # Notification callbacks
    notification_callbacks: List[Callable] = field(default_factory=list)

    # Analytics
    enable_trend_analysis: bool = True
    trend_analysis_days: int = 30


class CCPMarginTracker:
    """Enhanced margin tracking for CCP connectors.

    This tracker can be attached to any CCP connector to provide:
    1. Historical margin snapshots
    2. Margin change detection and alerting
    3. Margin call tracking and analytics
    4. Trend analysis and forecasting
    """

    def __init__(
        self,
        ccp_name: str,
        member_id: str,
        config: Optional[MarginTrackerConfig] = None,
    ):
        """Initialize margin tracker.

        Args:
            ccp_name: CCP name
            member_id: Member ID
            config: Tracker configuration
        """
        self.ccp_name = ccp_name
        self.member_id = member_id
        self.config = config or MarginTrackerConfig()

        # Historical data
        self._snapshots: List[MarginSnapshot] = []
        self._change_events: List[MarginChangeEvent] = []
        self._margin_calls: Dict[str, MarginCallRecord] = {}

        # Current state
        self._current_margin: Optional[Decimal] = None
        self._last_snapshot_time: Optional[datetime] = None

    async def record_margin_snapshot(
        self,
        margin_data: Dict[str, Any],
        as_of_date: Optional[date] = None,
    ) -> MarginSnapshot:
        """Record a margin requirement snapshot.

        Args:
            margin_data: Margin data from CCP
            as_of_date: Reporting date (defaults to today)

        Returns:
            Margin snapshot
        """
        as_of_date = as_of_date or date.today()

        # Parse margin components
        initial_margin = Decimal(str(margin_data.get("initial_margin", 0)))
        variation_margin = Decimal(str(margin_data.get("variation_margin", 0)))
        additional_margin = Decimal(str(margin_data.get("additional_margin", 0)))
        liquidity_addon = Decimal(str(margin_data.get("liquidity_addon", 0)))
        concentration_addon = Decimal(str(margin_data.get("concentration_addon", 0)))

        total_margin = (
            initial_margin + variation_margin + additional_margin +
            liquidity_addon + concentration_addon
        )

        # Create snapshot
        snapshot = MarginSnapshot(
            snapshot_id=f"MS-{self.ccp_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            ccp_name=self.ccp_name,
            member_id=self.member_id,
            as_of_date=as_of_date,
            initial_margin=initial_margin,
            variation_margin=variation_margin,
            additional_margin=additional_margin,
            liquidity_addon=liquidity_addon,
            concentration_addon=concentration_addon,
            total_margin=total_margin,
            posted_margin=Decimal(str(margin_data.get("posted_margin", 0))),
            margin_deficit=total_margin - Decimal(str(margin_data.get("posted_margin", 0))),
            num_trades=margin_data.get("num_trades"),
            gross_notional=Decimal(str(margin_data["gross_notional"])) if margin_data.get("gross_notional") else None,
            net_exposure=Decimal(str(margin_data["net_exposure"])) if margin_data.get("net_exposure") else None,
            metadata=margin_data,
        )

        # Detect significant changes
        if self._current_margin is not None:
            await self._detect_margin_change(
                previous_margin=self._current_margin,
                new_margin=total_margin,
            )

        # Store snapshot
        self._snapshots.append(snapshot)
        self._current_margin = total_margin
        self._last_snapshot_time = datetime.utcnow()

        # Clean old snapshots
        self._cleanup_old_snapshots()

        # Notify callbacks
        await self._notify_snapshot(snapshot)

        return snapshot

    async def _detect_margin_change(
        self,
        previous_margin: Decimal,
        new_margin: Decimal,
    ):
        """Detect and record significant margin changes.

        Args:
            previous_margin: Previous margin requirement
            new_margin: New margin requirement
        """
        change_amount = new_margin - previous_margin
        change_pct = float((change_amount / previous_margin) * 100) if previous_margin > 0 else 0

        # Determine if change is significant
        is_significant = (
            abs(change_pct) >= self.config.significant_change_threshold_pct or
            abs(change_amount) >= self.config.large_change_threshold
        )

        if is_significant:
            change_type = (
                MarginChangeType.INCREASE if change_amount > 0
                else MarginChangeType.DECREASE
            )

            event = MarginChangeEvent(
                event_id=f"MCE-{self.ccp_name}-{len(self._change_events) + 1:06d}",
                ccp_name=self.ccp_name,
                change_type=change_type,
                previous_margin=previous_margin,
                new_margin=new_margin,
                change_amount=change_amount,
                change_percentage=change_pct,
                reason="Automatic detection" if abs(change_pct) >= self.config.significant_change_threshold_pct else "Large absolute change",
            )

            self._change_events.append(event)

            # Notify callbacks
            await self._notify_change_event(event)

    async def record_margin_call(
        self,
        margin_call: MarginCall,
    ) -> MarginCallRecord:
        """Record a new margin call.

        Args:
            margin_call: Margin call from CCP

        Returns:
            Margin call record
        """
        record = MarginCallRecord(
            margin_call=margin_call,
        )

        self._margin_calls[margin_call.call_id] = record

        # Create change event
        if self._current_margin:
            event = MarginChangeEvent(
                event_id=f"MCE-{self.ccp_name}-{len(self._change_events) + 1:06d}",
                ccp_name=self.ccp_name,
                change_type=MarginChangeType.NEW_CALL,
                previous_margin=self._current_margin,
                new_margin=self._current_margin + margin_call.call_amount,
                change_amount=margin_call.call_amount,
                change_percentage=float((margin_call.call_amount / self._current_margin) * 100),
                reason=f"Margin call: {margin_call.call_type}",
                margin_call_id=margin_call.call_id,
            )
            self._change_events.append(event)

        # Notify callbacks
        if self.config.margin_call_alert_enabled:
            await self._notify_margin_call(record)

        return record

    async def satisfy_margin_call(
        self,
        call_id: str,
        collateral_posted: Optional[List[Dict[str, Any]]] = None,
    ):
        """Mark a margin call as satisfied.

        Args:
            call_id: Margin call ID
            collateral_posted: Details of collateral posted
        """
        record = self._margin_calls.get(call_id)
        if not record:
            raise ValueError(f"Margin call {call_id} not found")

        record.satisfied_time = datetime.utcnow()

        # Calculate response time
        if record.margin_call.timestamp:
            delta = record.satisfied_time - record.margin_call.timestamp
            record.response_time_minutes = delta.total_seconds() / 60

        if collateral_posted:
            record.collateral_posted = collateral_posted

        # Create change event
        if self._current_margin:
            event = MarginChangeEvent(
                event_id=f"MCE-{self.ccp_name}-{len(self._change_events) + 1:06d}",
                ccp_name=self.ccp_name,
                change_type=MarginChangeType.CALL_SATISFIED,
                previous_margin=self._current_margin + record.margin_call.call_amount,
                new_margin=self._current_margin,
                change_amount=-record.margin_call.call_amount,
                change_percentage=float((-record.margin_call.call_amount / self._current_margin) * 100),
                reason=f"Margin call satisfied: {call_id}",
                margin_call_id=call_id,
            )
            self._change_events.append(event)

    def check_overdue_margin_calls(self) -> List[MarginCallRecord]:
        """Check for overdue margin calls.

        Returns:
            List of overdue margin call records
        """
        overdue_calls = []
        now = datetime.utcnow()

        for call_id, record in self._margin_calls.items():
            if record.satisfied_time:
                continue  # Already satisfied

            # Check if overdue
            if record.margin_call.due_time < now:
                record.overdue = True
                overdue_calls.append(record)

        return overdue_calls

    def get_current_margin(self) -> Optional[Decimal]:
        """Get current margin requirement."""
        return self._current_margin

    def get_latest_snapshot(self) -> Optional[MarginSnapshot]:
        """Get most recent margin snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshots(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> List[MarginSnapshot]:
        """Get historical margin snapshots.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of snapshots

        Returns:
            List of margin snapshots
        """
        snapshots = self._snapshots

        if start_date:
            snapshots = [s for s in snapshots if s.as_of_date >= start_date]

        if end_date:
            snapshots = [s for s in snapshots if s.as_of_date <= end_date]

        if limit:
            snapshots = snapshots[-limit:]

        return snapshots

    def get_change_events(
        self,
        start_time: Optional[datetime] = None,
        change_type: Optional[MarginChangeType] = None,
    ) -> List[MarginChangeEvent]:
        """Get margin change events.

        Args:
            start_time: Filter by start time
            change_type: Filter by change type

        Returns:
            List of change events
        """
        events = self._change_events

        if start_time:
            events = [e for e in events if e.event_time >= start_time]

        if change_type:
            events = [e for e in events if e.change_type == change_type]

        return events

    def get_active_margin_calls(self) -> List[MarginCallRecord]:
        """Get all active (unsatisfied) margin calls.

        Returns:
            List of active margin call records
        """
        return [
            record for record in self._margin_calls.values()
            if not record.satisfied_time
        ]

    def get_margin_call_statistics(self) -> Dict[str, Any]:
        """Get margin call statistics.

        Returns:
            Dictionary with statistics
        """
        total_calls = len(self._margin_calls)
        active_calls = len(self.get_active_margin_calls())
        satisfied_calls = total_calls - active_calls
        overdue_calls = len(self.check_overdue_margin_calls())

        # Calculate average response time
        response_times = [
            r.response_time_minutes
            for r in self._margin_calls.values()
            if r.response_time_minutes is not None
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            "total_margin_calls": total_calls,
            "active_calls": active_calls,
            "satisfied_calls": satisfied_calls,
            "overdue_calls": overdue_calls,
            "satisfaction_rate": (satisfied_calls / total_calls * 100) if total_calls > 0 else 0,
            "avg_response_time_minutes": avg_response_time,
        }

    def get_margin_trends(self, days: int = 30) -> Dict[str, List[float]]:
        """Get margin requirement trends.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trend data
        """
        if not self.config.enable_trend_analysis:
            return {}

        cutoff_date = date.today() - timedelta(days=days)
        recent_snapshots = [
            s for s in self._snapshots
            if s.as_of_date >= cutoff_date
        ]

        trends = {
            "dates": [],
            "total_margin": [],
            "initial_margin": [],
            "variation_margin": [],
            "margin_deficit": [],
        }

        for snapshot in recent_snapshots:
            trends["dates"].append(snapshot.as_of_date.isoformat())
            trends["total_margin"].append(float(snapshot.total_margin))
            trends["initial_margin"].append(float(snapshot.initial_margin))
            trends["variation_margin"].append(float(snapshot.variation_margin))
            trends["margin_deficit"].append(float(snapshot.margin_deficit))

        return trends

    def _cleanup_old_snapshots(self):
        """Remove old snapshots beyond retention period."""
        cutoff_date = date.today() - timedelta(days=self.config.retain_snapshots_days)
        self._snapshots = [
            s for s in self._snapshots
            if s.as_of_date >= cutoff_date
        ]

    async def _notify_snapshot(self, snapshot: MarginSnapshot):
        """Send snapshot notifications."""
        for callback in self.config.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("snapshot", snapshot)
                else:
                    callback("snapshot", snapshot)
            except Exception:
                pass

    async def _notify_change_event(self, event: MarginChangeEvent):
        """Send change event notifications."""
        for callback in self.config.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("change", event)
                else:
                    callback("change", event)
            except Exception:
                pass

    async def _notify_margin_call(self, record: MarginCallRecord):
        """Send margin call notifications."""
        for callback in self.config.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("margin_call", record)
                else:
                    callback("margin_call", record)
            except Exception:
                pass


# Add asyncio import
import asyncio


__all__ = [
    "CCPMarginTracker",
    "MarginTrackerConfig",
    "MarginSnapshot",
    "MarginChangeEvent",
    "MarginCallRecord",
    "MarginChangeType",
]
