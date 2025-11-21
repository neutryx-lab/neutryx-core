"""Margin aggregation service for multi-CCP portfolio management.

This module provides consolidated margin tracking and reporting across multiple CCPs:
1. Aggregate initial margin (IM) and variation margin (VM) across all CCPs
2. Calculate portfolio-level margin requirements
3. Track collateral requirements and utilization
4. Monitor margin calls and deficits
5. Generate consolidated margin reports
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .base import CCPConnector, MarginCall


class MarginType(str, Enum):
    """Type of margin."""
    INITIAL_MARGIN = "initial_margin"  # IM - upfront margin
    VARIATION_MARGIN = "variation_margin"  # VM - mark-to-market margin
    ADDITIONAL_MARGIN = "additional_margin"  # Add-ons and buffers
    LIQUIDITY_ADDON = "liquidity_addon"  # Liquidity buffer
    CONCENTRATION_ADDON = "concentration_addon"  # Concentration charge


class CollateralType(str, Enum):
    """Type of collateral."""
    CASH = "cash"
    GOVERNMENT_BONDS = "government_bonds"
    CORPORATE_BONDS = "corporate_bonds"
    EQUITIES = "equities"
    SECURITIES = "securities"
    GOLD = "gold"
    OTHER = "other"


class MarginCallStatus(str, Enum):
    """Margin call status."""
    PENDING = "pending"
    NOTIFIED = "notified"
    ACKNOWLEDGED = "acknowledged"
    POSTED = "posted"
    SATISFIED = "satisfied"
    DISPUTED = "disputed"
    OVERDUE = "overdue"


class CollateralPosition(BaseModel):
    """Collateral position with valuation."""

    collateral_type: CollateralType = Field(..., description="Type of collateral")
    asset_id: Optional[str] = Field(None, description="Specific asset identifier")
    quantity: Decimal = Field(..., description="Quantity or amount")
    currency: str = Field(..., description="Currency")

    # Valuation
    market_value: Decimal = Field(..., description="Market value")
    haircut: Decimal = Field(default=Decimal("0"), description="Haircut percentage")
    collateral_value: Decimal = Field(..., description="Value after haircut")

    # Location
    ccp_name: Optional[str] = Field(None, description="CCP holding collateral")
    account: Optional[str] = Field(None, description="Account holding collateral")

    last_valuation_date: date = Field(..., description="Last valuation date")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CCPMarginRequirement(BaseModel):
    """Margin requirement from a single CCP."""

    ccp_name: str = Field(..., description="CCP name")
    member_id: str = Field(..., description="Member ID")
    as_of_date: date = Field(..., description="Reporting date")

    # Margin components
    initial_margin: Decimal = Field(default=Decimal("0"), description="Initial margin")
    variation_margin: Decimal = Field(default=Decimal("0"), description="Variation margin")
    additional_margin: Decimal = Field(default=Decimal("0"), description="Additional margin")
    liquidity_addon: Decimal = Field(default=Decimal("0"), description="Liquidity add-on")
    concentration_addon: Decimal = Field(default=Decimal("0"), description="Concentration charge")

    # Totals
    total_margin_required: Decimal = Field(..., description="Total margin required")
    total_margin_posted: Decimal = Field(default=Decimal("0"), description="Total margin posted")
    margin_deficit: Decimal = Field(default=Decimal("0"), description="Margin deficit/surplus")

    # Currency
    currency: str = Field(default="USD", description="Reporting currency")

    # Collateral breakdown
    collateral_positions: List[CollateralPosition] = Field(
        default_factory=list,
        description="Collateral positions posted"
    )

    # Active margin calls
    active_margin_calls: List[MarginCall] = Field(
        default_factory=list,
        description="Active margin calls"
    )

    # Portfolio metrics
    num_trades: Optional[int] = Field(None, description="Number of trades")
    gross_notional: Optional[Decimal] = Field(None, description="Gross notional")
    net_mtm: Optional[Decimal] = Field(None, description="Net mark-to-market")

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_margin_deficit(self):
        """Calculate margin deficit (negative means surplus)."""
        self.margin_deficit = self.total_margin_required - self.total_margin_posted

    def has_margin_call(self) -> bool:
        """Check if there are active margin calls."""
        return len(self.active_margin_calls) > 0

    def total_collateral_value(self) -> Decimal:
        """Calculate total collateral value after haircuts."""
        return sum(pos.collateral_value for pos in self.collateral_positions)


class AggregatedMarginReport(BaseModel):
    """Aggregated margin report across all CCPs."""

    report_id: str = Field(..., description="Report identifier")
    as_of_date: date = Field(..., description="Report date")
    generated_time: datetime = Field(default_factory=datetime.utcnow)

    # CCP breakdown
    ccp_requirements: Dict[str, CCPMarginRequirement] = Field(
        default_factory=dict,
        description="Margin requirements by CCP"
    )

    # Aggregated totals
    total_initial_margin: Decimal = Field(default=Decimal("0"), description="Total IM across CCPs")
    total_variation_margin: Decimal = Field(default=Decimal("0"), description="Total VM across CCPs")
    total_additional_margin: Decimal = Field(default=Decimal("0"), description="Total additional margin")
    total_margin_required: Decimal = Field(default=Decimal("0"), description="Total margin required")
    total_margin_posted: Decimal = Field(default=Decimal("0"), description="Total margin posted")
    total_margin_deficit: Decimal = Field(default=Decimal("0"), description="Total deficit/surplus")

    # Collateral aggregation
    total_collateral_value: Decimal = Field(default=Decimal("0"), description="Total collateral value")
    collateral_by_type: Dict[CollateralType, Decimal] = Field(
        default_factory=dict,
        description="Collateral breakdown by type"
    )

    # Portfolio metrics
    total_trades: int = Field(default=0, description="Total number of trades")
    total_gross_notional: Decimal = Field(default=Decimal("0"), description="Total gross notional")
    total_net_mtm: Decimal = Field(default=Decimal("0"), description="Total net MTM")

    # Margin calls
    total_margin_calls: int = Field(default=0, description="Number of active margin calls")
    margin_calls_by_ccp: Dict[str, int] = Field(default_factory=dict)

    # Utilization metrics
    collateral_utilization_pct: float = Field(default=0.0, description="Collateral utilization %")
    margin_coverage_ratio: float = Field(default=0.0, description="Margin coverage ratio")

    # Currency
    reporting_currency: str = Field(default="USD", description="Reporting currency")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_aggregates(self):
        """Calculate aggregate values from CCP requirements."""
        # Reset aggregates
        self.total_initial_margin = Decimal("0")
        self.total_variation_margin = Decimal("0")
        self.total_additional_margin = Decimal("0")
        self.total_margin_required = Decimal("0")
        self.total_margin_posted = Decimal("0")
        self.total_margin_deficit = Decimal("0")
        self.total_collateral_value = Decimal("0")
        self.collateral_by_type = {}
        self.total_trades = 0
        self.total_gross_notional = Decimal("0")
        self.total_net_mtm = Decimal("0")
        self.total_margin_calls = 0
        self.margin_calls_by_ccp = {}

        # Aggregate from each CCP
        for ccp_name, req in self.ccp_requirements.items():
            self.total_initial_margin += req.initial_margin
            self.total_variation_margin += req.variation_margin
            self.total_additional_margin += req.additional_margin
            self.total_margin_required += req.total_margin_required
            self.total_margin_posted += req.total_margin_posted
            self.total_margin_deficit += req.margin_deficit
            self.total_collateral_value += req.total_collateral_value()

            if req.num_trades:
                self.total_trades += req.num_trades
            if req.gross_notional:
                self.total_gross_notional += req.gross_notional
            if req.net_mtm:
                self.total_net_mtm += req.net_mtm

            # Aggregate collateral by type
            for pos in req.collateral_positions:
                if pos.collateral_type not in self.collateral_by_type:
                    self.collateral_by_type[pos.collateral_type] = Decimal("0")
                self.collateral_by_type[pos.collateral_type] += pos.collateral_value

            # Count margin calls
            num_calls = len(req.active_margin_calls)
            if num_calls > 0:
                self.total_margin_calls += num_calls
                self.margin_calls_by_ccp[ccp_name] = num_calls

        # Calculate utilization metrics
        if self.total_collateral_value > 0:
            self.collateral_utilization_pct = float(
                (self.total_margin_required / self.total_collateral_value) * 100
            )

        if self.total_margin_required > 0:
            self.margin_coverage_ratio = float(
                self.total_margin_posted / self.total_margin_required
            )


@dataclass
class MarginAggregatorConfig:
    """Configuration for margin aggregator."""

    # Reporting settings
    reporting_currency: str = "USD"
    consolidation_frequency_hours: int = 6  # How often to refresh

    # Thresholds and alerts
    margin_deficit_threshold: Decimal = Decimal("1000000")  # Alert threshold
    collateral_utilization_warning: float = 80.0  # % utilization warning
    margin_coverage_warning: float = 1.1  # Warn if coverage < 110%

    # Collateral haircuts (simplified)
    collateral_haircuts: Dict[CollateralType, Decimal] = field(default_factory=lambda: {
        CollateralType.CASH: Decimal("0"),
        CollateralType.GOVERNMENT_BONDS: Decimal("0.02"),  # 2%
        CollateralType.CORPORATE_BONDS: Decimal("0.10"),  # 10%
        CollateralType.EQUITIES: Decimal("0.20"),  # 20%
        CollateralType.GOLD: Decimal("0.15"),  # 15%
        CollateralType.OTHER: Decimal("0.30"),  # 30%
    })

    # Auto-refresh settings
    auto_refresh_enabled: bool = True


class MarginAggregationService:
    """Service for aggregating margin requirements across multiple CCPs.

    This service provides:
    1. Real-time margin aggregation across all CCPs
    2. Consolidated collateral tracking
    3. Margin call monitoring and alerting
    4. Portfolio-level margin analytics
    5. Collateral optimization recommendations
    """

    def __init__(
        self,
        connectors: Dict[str, CCPConnector],
        config: Optional[MarginAggregatorConfig] = None,
    ):
        """Initialize margin aggregation service.

        Args:
            connectors: Dictionary of CCP name to connector instances
            config: Aggregator configuration
        """
        self.connectors = connectors
        self.config = config or MarginAggregatorConfig()

        # Cached margin requirements
        self._ccp_margins: Dict[str, CCPMarginRequirement] = {}
        self._last_refresh: Optional[datetime] = None

        # Historical reports
        self._historical_reports: List[AggregatedMarginReport] = []

    async def refresh_margin_requirements(
        self,
        ccp_names: Optional[List[str]] = None,
    ) -> Dict[str, CCPMarginRequirement]:
        """Refresh margin requirements from CCPs.

        Args:
            ccp_names: List of CCP names to refresh (all if None)

        Returns:
            Dictionary of CCP name to margin requirements
        """
        ccps_to_refresh = ccp_names or list(self.connectors.keys())

        # Fetch margins from all CCPs in parallel
        tasks = []
        ccp_list = []

        for ccp_name in ccps_to_refresh:
            connector = self.connectors.get(ccp_name)
            if connector and connector.is_connected:
                tasks.append(self._fetch_ccp_margin(connector))
                ccp_list.append(ccp_name)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update cache
        for ccp_name, result in zip(ccp_list, results):
            if isinstance(result, CCPMarginRequirement):
                self._ccp_margins[ccp_name] = result

        self._last_refresh = datetime.utcnow()

        return self._ccp_margins

    async def _fetch_ccp_margin(
        self,
        connector: CCPConnector,
    ) -> CCPMarginRequirement:
        """Fetch margin requirements from a single CCP.

        Args:
            connector: CCP connector

        Returns:
            CCP margin requirement
        """
        try:
            # Fetch margin data
            margin_data = await connector.get_margin_requirements()

            # Parse margin components
            initial_margin = Decimal(str(margin_data.get("initial_margin", 0)))
            variation_margin = Decimal(str(margin_data.get("variation_margin", 0)))
            additional_margin = Decimal(str(margin_data.get("additional_margin", 0)))
            liquidity_addon = Decimal(str(margin_data.get("liquidity_addon", 0)))
            concentration_addon = Decimal(str(margin_data.get("concentration_addon", 0)))

            total_required = (
                initial_margin + variation_margin + additional_margin +
                liquidity_addon + concentration_addon
            )

            # Get position report for additional metrics
            try:
                position_report = await connector.get_position_report()
                num_trades = len(position_report.positions) if position_report.positions else None
                gross_notional = position_report.total_exposure
                net_mtm = None  # Would be calculated from positions
            except:
                num_trades = None
                gross_notional = None
                net_mtm = None

            # Create requirement object
            requirement = CCPMarginRequirement(
                ccp_name=connector.config.ccp_name,
                member_id=connector.config.member_id,
                as_of_date=date.today(),
                initial_margin=initial_margin,
                variation_margin=variation_margin,
                additional_margin=additional_margin,
                liquidity_addon=liquidity_addon,
                concentration_addon=concentration_addon,
                total_margin_required=total_required,
                total_margin_posted=Decimal(str(margin_data.get("posted_margin", 0))),
                currency=self.config.reporting_currency,
                num_trades=num_trades,
                gross_notional=gross_notional,
                net_mtm=net_mtm,
                metadata=margin_data,
            )

            requirement.calculate_margin_deficit()

            return requirement

        except Exception as e:
            # Return empty requirement on error
            return CCPMarginRequirement(
                ccp_name=connector.config.ccp_name,
                member_id=connector.config.member_id,
                as_of_date=date.today(),
                total_margin_required=Decimal("0"),
                currency=self.config.reporting_currency,
                metadata={"error": str(e)},
            )

    async def generate_aggregated_report(
        self,
        refresh: bool = True,
    ) -> AggregatedMarginReport:
        """Generate aggregated margin report.

        Args:
            refresh: Whether to refresh margin data first

        Returns:
            Aggregated margin report
        """
        # Refresh if requested
        if refresh:
            await self.refresh_margin_requirements()

        # Create report
        report = AggregatedMarginReport(
            report_id=f"MR-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            as_of_date=date.today(),
            ccp_requirements=self._ccp_margins.copy(),
            reporting_currency=self.config.reporting_currency,
        )

        # Calculate aggregates
        report.calculate_aggregates()

        # Store in history
        self._historical_reports.append(report)

        # Trim history (keep last 100 reports)
        if len(self._historical_reports) > 100:
            self._historical_reports = self._historical_reports[-100:]

        return report

    def get_ccp_margin(self, ccp_name: str) -> Optional[CCPMarginRequirement]:
        """Get cached margin requirement for a CCP.

        Args:
            ccp_name: CCP name

        Returns:
            CCP margin requirement or None if not available
        """
        return self._ccp_margins.get(ccp_name)

    def get_total_margin_deficit(self) -> Decimal:
        """Get total margin deficit across all CCPs.

        Returns:
            Total margin deficit (negative means surplus)
        """
        return sum(
            req.margin_deficit
            for req in self._ccp_margins.values()
        )

    def get_ccps_with_margin_calls(self) -> List[str]:
        """Get list of CCPs with active margin calls.

        Returns:
            List of CCP names with active margin calls
        """
        return [
            ccp_name
            for ccp_name, req in self._ccp_margins.items()
            if req.has_margin_call()
        ]

    def get_margin_alerts(self) -> List[Dict[str, Any]]:
        """Get margin-related alerts.

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Check total deficit
        total_deficit = self.get_total_margin_deficit()
        if total_deficit > self.config.margin_deficit_threshold:
            alerts.append({
                "type": "MARGIN_DEFICIT",
                "severity": "HIGH",
                "message": f"Total margin deficit exceeds threshold: {total_deficit:,.2f}",
                "deficit": float(total_deficit),
            })

        # Check per-CCP deficits
        for ccp_name, req in self._ccp_margins.items():
            if req.margin_deficit > Decimal("0"):
                alerts.append({
                    "type": "CCP_MARGIN_DEFICIT",
                    "severity": "MEDIUM",
                    "ccp": ccp_name,
                    "message": f"{ccp_name} margin deficit: {req.margin_deficit:,.2f}",
                    "deficit": float(req.margin_deficit),
                })

        # Check active margin calls
        ccps_with_calls = self.get_ccps_with_margin_calls()
        for ccp_name in ccps_with_calls:
            req = self._ccp_margins[ccp_name]
            alerts.append({
                "type": "MARGIN_CALL",
                "severity": "HIGH",
                "ccp": ccp_name,
                "message": f"{ccp_name} has {len(req.active_margin_calls)} active margin call(s)",
                "num_calls": len(req.active_margin_calls),
            })

        return alerts

    def get_historical_reports(
        self,
        limit: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[AggregatedMarginReport]:
        """Get historical margin reports.

        Args:
            limit: Maximum number of reports to return
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of historical reports
        """
        reports = self._historical_reports

        if start_date:
            reports = [r for r in reports if r.as_of_date >= start_date]

        if end_date:
            reports = [r for r in reports if r.as_of_date <= end_date]

        if limit:
            reports = reports[-limit:]

        return reports

    def get_margin_trends(self, days: int = 30) -> Dict[str, List[float]]:
        """Get margin requirement trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trend data
        """
        cutoff_date = date.today() - timedelta(days=days)
        recent_reports = [
            r for r in self._historical_reports
            if r.as_of_date >= cutoff_date
        ]

        trends = {
            "dates": [],
            "total_im": [],
            "total_vm": [],
            "total_deficit": [],
            "collateral_utilization": [],
        }

        for report in recent_reports:
            trends["dates"].append(report.as_of_date.isoformat())
            trends["total_im"].append(float(report.total_initial_margin))
            trends["total_vm"].append(float(report.total_variation_margin))
            trends["total_deficit"].append(float(report.total_margin_deficit))
            trends["collateral_utilization"].append(report.collateral_utilization_pct)

        return trends

    async def optimize_collateral_allocation(
        self,
    ) -> Dict[str, Dict[CollateralType, Decimal]]:
        """Suggest optimal collateral allocation across CCPs.

        Returns:
            Dictionary of CCP name to suggested collateral allocation
        """
        # This is a placeholder for collateral optimization logic
        # In practice, this would use optimization algorithms to:
        # 1. Minimize haircuts
        # 2. Balance concentration risk
        # 3. Maximize collateral efficiency

        suggestions = {}

        for ccp_name, req in self._ccp_margins.items():
            if req.margin_deficit > Decimal("0"):
                # Suggest posting cash (lowest haircut)
                suggestions[ccp_name] = {
                    CollateralType.CASH: req.margin_deficit
                }

        return suggestions


__all__ = [
    "MarginAggregationService",
    "MarginAggregatorConfig",
    "AggregatedMarginReport",
    "CCPMarginRequirement",
    "CollateralPosition",
    "MarginType",
    "CollateralType",
    "MarginCallStatus",
]
