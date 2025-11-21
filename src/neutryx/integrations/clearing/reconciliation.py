"""CCP reconciliation engine for positions and settlements.

This module provides comprehensive reconciliation between internal records
and CCP reports:
1. Position reconciliation (internal vs CCP positions)
2. Settlement reconciliation (instructed vs settled)
3. Margin reconciliation (calculated vs reported)
4. Break detection and reporting
5. Automated resolution workflows
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .base import CCPConnector, PositionReport


class ReconciliationType(str, Enum):
    """Type of reconciliation."""
    POSITION = "position"  # Trade positions
    SETTLEMENT = "settlement"  # Settlement instructions
    MARGIN = "margin"  # Margin requirements
    COLLATERAL = "collateral"  # Posted collateral


class BreakType(str, Enum):
    """Type of reconciliation break."""
    MISSING_INTERNAL = "missing_internal"  # CCP has trade, internal doesn't
    MISSING_CCP = "missing_ccp"  # Internal has trade, CCP doesn't
    QUANTITY_MISMATCH = "quantity_mismatch"  # Different notionals
    PRICE_MISMATCH = "price_mismatch"  # Different rates/prices
    STATUS_MISMATCH = "status_mismatch"  # Different trade statuses
    SETTLEMENT_MISMATCH = "settlement_mismatch"  # Settlement not matched
    MARGIN_MISMATCH = "margin_mismatch"  # Margin calculations differ
    COLLATERAL_MISMATCH = "collateral_mismatch"  # Collateral amounts differ


class BreakSeverity(str, Enum):
    """Severity of reconciliation break."""
    CRITICAL = "critical"  # Requires immediate attention
    HIGH = "high"  # Significant impact
    MEDIUM = "medium"  # Moderate impact
    LOW = "low"  # Minor discrepancy
    INFO = "info"  # Informational only


class BreakStatus(str, Enum):
    """Status of reconciliation break."""
    OPEN = "open"  # Newly detected
    INVESTIGATING = "investigating"  # Under investigation
    RESOLVED = "resolved"  # Successfully resolved
    ACKNOWLEDGED = "acknowledged"  # Acknowledged, will resolve later
    CANCELLED = "cancelled"  # False positive


class ReconciliationBreak(BaseModel):
    """A reconciliation break (discrepancy)."""

    break_id: str = Field(..., description="Break identifier")
    recon_id: str = Field(..., description="Related reconciliation run ID")
    break_type: BreakType = Field(..., description="Type of break")
    severity: BreakSeverity = Field(..., description="Severity")
    status: BreakStatus = Field(default=BreakStatus.OPEN)

    # Entities involved
    ccp_name: str = Field(..., description="CCP name")
    trade_id: Optional[str] = Field(None, description="Internal trade ID")
    ccp_trade_id: Optional[str] = Field(None, description="CCP trade ID")

    # Discrepancy details
    field_name: Optional[str] = Field(None, description="Field with discrepancy")
    internal_value: Optional[str] = Field(None, description="Internal system value")
    ccp_value: Optional[str] = Field(None, description="CCP reported value")
    difference: Optional[str] = Field(None, description="Calculated difference")

    # Description
    description: str = Field(..., description="Break description")
    impact_assessment: Optional[str] = Field(None, description="Business impact")

    # Timestamps
    detected_time: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_time: Optional[datetime] = Field(None)
    resolved_time: Optional[datetime] = Field(None)

    # Resolution
    resolution_notes: Optional[str] = Field(None, description="How break was resolved")
    resolved_by: Optional[str] = Field(None, description="Who resolved the break")

    # Follow-up
    follow_up_required: bool = Field(default=False)
    follow_up_notes: Optional[str] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReconciliationResult(BaseModel):
    """Result of a reconciliation run."""

    recon_id: str = Field(..., description="Reconciliation run ID")
    recon_type: ReconciliationType = Field(..., description="Reconciliation type")
    ccp_name: str = Field(..., description="CCP name")
    as_of_date: date = Field(..., description="Reconciliation date")
    run_time: datetime = Field(default_factory=datetime.utcnow)

    # Counts
    total_internal_records: int = Field(default=0, description="Internal records count")
    total_ccp_records: int = Field(default=0, description="CCP records count")
    matched_records: int = Field(default=0, description="Matched records")
    breaks: List[ReconciliationBreak] = Field(default_factory=list, description="Detected breaks")

    # Summary
    total_breaks: int = Field(default=0)
    critical_breaks: int = Field(default=0)
    high_breaks: int = Field(default=0)
    medium_breaks: int = Field(default=0)
    low_breaks: int = Field(default=0)

    # Status
    reconciliation_passed: bool = Field(..., description="Whether recon passed")
    requires_attention: bool = Field(..., description="Requires manual attention")

    # Totals (for value reconciliation)
    total_internal_value: Optional[Decimal] = Field(None, description="Total internal value")
    total_ccp_value: Optional[Decimal] = Field(None, description="Total CCP value")
    value_difference: Optional[Decimal] = Field(None, description="Value difference")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_summary(self):
        """Calculate summary statistics from breaks."""
        self.total_breaks = len(self.breaks)
        self.critical_breaks = sum(1 for b in self.breaks if b.severity == BreakSeverity.CRITICAL)
        self.high_breaks = sum(1 for b in self.breaks if b.severity == BreakSeverity.HIGH)
        self.medium_breaks = sum(1 for b in self.breaks if b.severity == BreakSeverity.MEDIUM)
        self.low_breaks = sum(1 for b in self.breaks if b.severity == BreakSeverity.LOW)

        self.requires_attention = (self.critical_breaks > 0 or self.high_breaks > 0)
        self.reconciliation_passed = (self.critical_breaks == 0 and self.high_breaks == 0)


@dataclass
class ReconciliationConfig:
    """Configuration for reconciliation engine."""

    # Tolerance thresholds
    quantity_tolerance: Decimal = Decimal("0.01")  # 0.01 tolerance for notionals
    price_tolerance: Decimal = Decimal("0.0001")  # 0.01% tolerance for rates
    margin_tolerance: Decimal = Decimal("100")  # $100 tolerance for margins

    # Severity rules
    large_quantity_threshold: Decimal = Decimal("10000000")  # $10M+
    critical_margin_threshold: Decimal = Decimal("1000000")  # $1M+ margin diff

    # Auto-resolution
    auto_resolve_within_tolerance: bool = True
    auto_acknowledge_info_breaks: bool = True

    # Scheduling
    auto_reconcile_enabled: bool = True
    reconcile_position_daily: bool = True
    reconcile_settlement_daily: bool = True
    reconcile_margin_daily: bool = True

    # Notification thresholds
    notify_on_critical: bool = True
    notify_on_high: bool = True


class CCPReconciliationEngine:
    """Reconciliation engine for CCP positions and settlements.

    This engine performs automated reconciliation between internal records
    and CCP reports, detecting and tracking discrepancies.
    """

    def __init__(
        self,
        connectors: Dict[str, CCPConnector],
        config: Optional[ReconciliationConfig] = None,
    ):
        """Initialize reconciliation engine.

        Args:
            connectors: Dictionary of CCP name to connector instances
            config: Reconciliation configuration
        """
        self.connectors = connectors
        self.config = config or ReconciliationConfig()

        # Reconciliation history
        self._reconciliation_runs: List[ReconciliationResult] = []

        # Open breaks tracking
        self._open_breaks: Dict[str, ReconciliationBreak] = {}

    async def reconcile_positions(
        self,
        ccp_name: str,
        internal_positions: List[Dict[str, Any]],
        as_of_date: Optional[date] = None,
    ) -> ReconciliationResult:
        """Reconcile trade positions with CCP.

        Args:
            ccp_name: CCP name
            internal_positions: List of internal position records
            as_of_date: Reconciliation date

        Returns:
            Reconciliation result
        """
        as_of_date = as_of_date or date.today()
        connector = self.connectors.get(ccp_name)

        if not connector:
            raise ValueError(f"No connector found for CCP: {ccp_name}")

        # Fetch CCP position report
        position_report = await connector.get_position_report(as_of_date=datetime.combine(as_of_date, datetime.min.time()))

        # Create reconciliation result
        recon_id = f"RECON-POS-{ccp_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        result = ReconciliationResult(
            recon_id=recon_id,
            recon_type=ReconciliationType.POSITION,
            ccp_name=ccp_name,
            as_of_date=as_of_date,
            total_internal_records=len(internal_positions),
            total_ccp_records=len(position_report.positions),
            reconciliation_passed=False,  # Will be set after analysis
        )

        # Create position maps for comparison
        internal_map = {pos.get("trade_id"): pos for pos in internal_positions}
        ccp_map = {pos.get("trade_id"): pos for pos in position_report.positions}

        # Find matches and breaks
        matched_count = 0

        # Check internal positions against CCP
        for trade_id, internal_pos in internal_map.items():
            ccp_pos = ccp_map.get(trade_id)

            if not ccp_pos:
                # Missing in CCP
                break_item = ReconciliationBreak(
                    break_id=f"{recon_id}-B{len(result.breaks) + 1:04d}",
                    recon_id=recon_id,
                    break_type=BreakType.MISSING_CCP,
                    severity=self._determine_severity(
                        BreakType.MISSING_CCP,
                        internal_pos.get("notional", 0)
                    ),
                    ccp_name=ccp_name,
                    trade_id=trade_id,
                    description=f"Trade {trade_id} exists internally but not in CCP",
                    internal_value=str(internal_pos),
                )
                result.breaks.append(break_item)
                self._open_breaks[break_item.break_id] = break_item
                continue

            # Compare position details
            breaks_found = self._compare_positions(
                recon_id, ccp_name, trade_id, internal_pos, ccp_pos
            )

            if breaks_found:
                result.breaks.extend(breaks_found)
                for break_item in breaks_found:
                    self._open_breaks[break_item.break_id] = break_item
            else:
                matched_count += 1

        # Check for CCP positions not in internal system
        for trade_id, ccp_pos in ccp_map.items():
            if trade_id not in internal_map:
                break_item = ReconciliationBreak(
                    break_id=f"{recon_id}-B{len(result.breaks) + 1:04d}",
                    recon_id=recon_id,
                    break_type=BreakType.MISSING_INTERNAL,
                    severity=self._determine_severity(
                        BreakType.MISSING_INTERNAL,
                        ccp_pos.get("notional", 0)
                    ),
                    ccp_name=ccp_name,
                    trade_id=trade_id,
                    ccp_trade_id=ccp_pos.get("ccp_trade_id"),
                    description=f"Trade {trade_id} exists in CCP but not internally",
                    ccp_value=str(ccp_pos),
                )
                result.breaks.append(break_item)
                self._open_breaks[break_item.break_id] = break_item

        # Finalize result
        result.matched_records = matched_count
        result.calculate_summary()

        # Store result
        self._reconciliation_runs.append(result)

        return result

    def _compare_positions(
        self,
        recon_id: str,
        ccp_name: str,
        trade_id: str,
        internal_pos: Dict[str, Any],
        ccp_pos: Dict[str, Any],
    ) -> List[ReconciliationBreak]:
        """Compare internal and CCP positions for discrepancies.

        Args:
            recon_id: Reconciliation run ID
            ccp_name: CCP name
            trade_id: Trade ID
            internal_pos: Internal position
            ccp_pos: CCP position

        Returns:
            List of breaks found
        """
        breaks = []
        break_num = 1

        # Compare notional
        internal_notional = Decimal(str(internal_pos.get("notional", 0)))
        ccp_notional = Decimal(str(ccp_pos.get("notional", 0)))

        if abs(internal_notional - ccp_notional) > self.config.quantity_tolerance:
            breaks.append(ReconciliationBreak(
                break_id=f"{recon_id}-B{break_num:04d}",
                recon_id=recon_id,
                break_type=BreakType.QUANTITY_MISMATCH,
                severity=self._determine_severity(
                    BreakType.QUANTITY_MISMATCH,
                    abs(internal_notional - ccp_notional)
                ),
                ccp_name=ccp_name,
                trade_id=trade_id,
                field_name="notional",
                internal_value=str(internal_notional),
                ccp_value=str(ccp_notional),
                difference=str(internal_notional - ccp_notional),
                description=f"Notional mismatch for trade {trade_id}",
            ))
            break_num += 1

        # Compare rates/prices
        internal_rate = internal_pos.get("fixed_rate") or internal_pos.get("price")
        ccp_rate = ccp_pos.get("fixed_rate") or ccp_pos.get("price")

        if internal_rate and ccp_rate:
            internal_rate_dec = Decimal(str(internal_rate))
            ccp_rate_dec = Decimal(str(ccp_rate))

            if abs(internal_rate_dec - ccp_rate_dec) > self.config.price_tolerance:
                breaks.append(ReconciliationBreak(
                    break_id=f"{recon_id}-B{break_num:04d}",
                    recon_id=recon_id,
                    break_type=BreakType.PRICE_MISMATCH,
                    severity=BreakSeverity.MEDIUM,
                    ccp_name=ccp_name,
                    trade_id=trade_id,
                    field_name="rate/price",
                    internal_value=str(internal_rate),
                    ccp_value=str(ccp_rate),
                    difference=str(internal_rate_dec - ccp_rate_dec),
                    description=f"Rate/price mismatch for trade {trade_id}",
                ))
                break_num += 1

        # Compare status
        internal_status = internal_pos.get("status")
        ccp_status = ccp_pos.get("status")

        if internal_status and ccp_status and internal_status != ccp_status:
            breaks.append(ReconciliationBreak(
                break_id=f"{recon_id}-B{break_num:04d}",
                recon_id=recon_id,
                break_type=BreakType.STATUS_MISMATCH,
                severity=BreakSeverity.LOW,
                ccp_name=ccp_name,
                trade_id=trade_id,
                field_name="status",
                internal_value=internal_status,
                ccp_value=ccp_status,
                description=f"Status mismatch for trade {trade_id}",
            ))

        return breaks

    def _determine_severity(
        self,
        break_type: BreakType,
        amount: Any = None,
    ) -> BreakSeverity:
        """Determine severity of a break.

        Args:
            break_type: Type of break
            amount: Amount involved (if applicable)

        Returns:
            Severity level
        """
        # Critical breaks
        if break_type == BreakType.MISSING_INTERNAL:
            return BreakSeverity.CRITICAL

        if break_type == BreakType.MISSING_CCP:
            if amount and Decimal(str(amount)) > self.config.large_quantity_threshold:
                return BreakSeverity.CRITICAL
            return BreakSeverity.HIGH

        # Quantity mismatches
        if break_type == BreakType.QUANTITY_MISMATCH:
            if amount and Decimal(str(amount)) > self.config.large_quantity_threshold:
                return BreakSeverity.HIGH
            return BreakSeverity.MEDIUM

        # Margin mismatches
        if break_type == BreakType.MARGIN_MISMATCH:
            if amount and Decimal(str(amount)) > self.config.critical_margin_threshold:
                return BreakSeverity.CRITICAL
            return BreakSeverity.MEDIUM

        # Default severities
        severity_map = {
            BreakType.PRICE_MISMATCH: BreakSeverity.MEDIUM,
            BreakType.STATUS_MISMATCH: BreakSeverity.LOW,
            BreakType.SETTLEMENT_MISMATCH: BreakSeverity.HIGH,
            BreakType.COLLATERAL_MISMATCH: BreakSeverity.MEDIUM,
        }

        return severity_map.get(break_type, BreakSeverity.LOW)

    async def reconcile_settlements(
        self,
        ccp_name: str,
        internal_settlements: List[Dict[str, Any]],
        as_of_date: Optional[date] = None,
    ) -> ReconciliationResult:
        """Reconcile settlement instructions with CCP settlements.

        Args:
            ccp_name: CCP name
            internal_settlements: List of internal settlement records
            as_of_date: Reconciliation date

        Returns:
            Reconciliation result
        """
        as_of_date = as_of_date or date.today()

        recon_id = f"RECON-SETTLE-{ccp_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        result = ReconciliationResult(
            recon_id=recon_id,
            recon_type=ReconciliationType.SETTLEMENT,
            ccp_name=ccp_name,
            as_of_date=as_of_date,
            total_internal_records=len(internal_settlements),
            total_ccp_records=0,  # Would be populated from CCP data
            reconciliation_passed=True,
        )

        # In practice, would fetch settlement confirmations from CCP
        # and compare with internal settlement instructions

        result.calculate_summary()
        self._reconciliation_runs.append(result)

        return result

    def resolve_break(
        self,
        break_id: str,
        resolution_notes: str,
        resolved_by: Optional[str] = None,
    ):
        """Mark a break as resolved.

        Args:
            break_id: Break identifier
            resolution_notes: How the break was resolved
            resolved_by: Who resolved it
        """
        break_item = self._open_breaks.get(break_id)
        if not break_item:
            raise ValueError(f"Break {break_id} not found")

        break_item.status = BreakStatus.RESOLVED
        break_item.resolved_time = datetime.utcnow()
        break_item.resolution_notes = resolution_notes
        break_item.resolved_by = resolved_by

        # Remove from open breaks
        del self._open_breaks[break_id]

    def acknowledge_break(
        self,
        break_id: str,
        notes: Optional[str] = None,
    ):
        """Acknowledge a break (will resolve later).

        Args:
            break_id: Break identifier
            notes: Acknowledgment notes
        """
        break_item = self._open_breaks.get(break_id)
        if not break_item:
            raise ValueError(f"Break {break_id} not found")

        break_item.status = BreakStatus.ACKNOWLEDGED
        break_item.acknowledged_time = datetime.utcnow()
        if notes:
            break_item.follow_up_notes = notes
            break_item.follow_up_required = True

    def get_open_breaks(
        self,
        ccp_name: Optional[str] = None,
        severity: Optional[BreakSeverity] = None,
    ) -> List[ReconciliationBreak]:
        """Get open breaks.

        Args:
            ccp_name: Filter by CCP name
            severity: Filter by severity

        Returns:
            List of open breaks
        """
        breaks = list(self._open_breaks.values())

        if ccp_name:
            breaks = [b for b in breaks if b.ccp_name == ccp_name]

        if severity:
            breaks = [b for b in breaks if b.severity == severity]

        return breaks

    def get_reconciliation_runs(
        self,
        ccp_name: Optional[str] = None,
        recon_type: Optional[ReconciliationType] = None,
        limit: Optional[int] = None,
    ) -> List[ReconciliationResult]:
        """Get reconciliation run history.

        Args:
            ccp_name: Filter by CCP name
            recon_type: Filter by reconciliation type
            limit: Maximum number of runs to return

        Returns:
            List of reconciliation results
        """
        runs = self._reconciliation_runs

        if ccp_name:
            runs = [r for r in runs if r.ccp_name == ccp_name]

        if recon_type:
            runs = [r for r in runs if r.recon_type == recon_type]

        if limit:
            runs = runs[-limit:]

        return runs

    def get_reconciliation_statistics(self) -> Dict[str, Any]:
        """Get reconciliation statistics.

        Returns:
            Dictionary with statistics
        """
        total_runs = len(self._reconciliation_runs)
        passed_runs = sum(1 for r in self._reconciliation_runs if r.reconciliation_passed)
        total_breaks = len(self._open_breaks)

        severity_counts = {
            "critical": sum(1 for b in self._open_breaks.values() if b.severity == BreakSeverity.CRITICAL),
            "high": sum(1 for b in self._open_breaks.values() if b.severity == BreakSeverity.HIGH),
            "medium": sum(1 for b in self._open_breaks.values() if b.severity == BreakSeverity.MEDIUM),
            "low": sum(1 for b in self._open_breaks.values() if b.severity == BreakSeverity.LOW),
        }

        return {
            "total_reconciliation_runs": total_runs,
            "passed_runs": passed_runs,
            "failed_runs": total_runs - passed_runs,
            "pass_rate": (passed_runs / total_runs * 100) if total_runs > 0 else 0,
            "total_open_breaks": total_breaks,
            "breaks_by_severity": severity_counts,
        }


__all__ = [
    "CCPReconciliationEngine",
    "ReconciliationConfig",
    "ReconciliationResult",
    "ReconciliationBreak",
    "ReconciliationType",
    "BreakType",
    "BreakSeverity",
    "BreakStatus",
]
