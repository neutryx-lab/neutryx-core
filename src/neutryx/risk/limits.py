"""Position limits and risk controls framework.

This module provides comprehensive limit management for trading operations:

1. **Position Limits**: Notional limits by product, desk, trader
2. **VaR Limits**: Value-at-Risk limits and utilization tracking
3. **Concentration Limits**: Single-name, sector, geography limits
4. **Issuer Exposure Limits**: Credit exposure by counterparty
5. **Pre-Trade Controls**: Real-time limit checking before execution
6. **What-If Analysis**: Scenario-based limit impact assessment
7. **Breach Notifications**: Alert generation and escalation

Applications:
- Front-office trading controls
- Middle-office risk monitoring
- Regulatory compliance (FRTB, Basel)
- Real-time risk management
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

import jax.numpy as jnp
from jax import Array


# ==============================================================================
# Limit Types and Status
# ==============================================================================


class LimitType(Enum):
    """Types of trading limits."""

    NOTIONAL = "notional"  # Absolute notional exposure
    VAR = "var"  # Value at Risk
    CONCENTRATION = "concentration"  # Single-name/sector concentration
    ISSUER = "issuer"  # Credit exposure to single issuer
    DELTA = "delta"  # Delta exposure
    VEGA = "vega"  # Vega exposure
    GAMMA = "gamma"  # Gamma exposure
    TENOR = "tenor"  # Exposure by maturity bucket


class LimitStatus(Enum):
    """Status of limit utilization."""

    OK = "ok"  # Within limits
    WARNING = "warning"  # Approaching limit (e.g., >80%)
    SOFT_BREACH = "soft_breach"  # Exceeded soft limit (tradeable with approval)
    HARD_BREACH = "hard_breach"  # Exceeded hard limit (not tradeable)


class BreachSeverity(Enum):
    """Severity level for limit breaches."""

    INFO = "info"  # Informational
    WARNING = "warning"  # Warning level
    CRITICAL = "critical"  # Critical breach requiring immediate action


# ==============================================================================
# Limit Definitions
# ==============================================================================


@dataclass
class Limit:
    """Base class for all limit types.

    Attributes
    ----------
    name : str
        Limit identifier
    limit_type : LimitType
        Type of limit
    hard_limit : float
        Hard limit value (cannot exceed)
    soft_limit : float, optional
        Soft limit value (warning threshold)
    warning_threshold : float
        Percentage of hard limit that triggers warning (default 0.8)
    scope : Dict[str, str]
        Scope definition (e.g., {"desk": "rates", "product": "swaps"})
    """

    name: str
    limit_type: LimitType
    hard_limit: float
    soft_limit: Optional[float] = None
    warning_threshold: float = 0.8
    scope: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.soft_limit is None:
            self.soft_limit = self.hard_limit * self.warning_threshold

    def check_utilization(self, current_exposure: float) -> LimitStatus:
        """Check limit status given current exposure.

        Parameters
        ----------
        current_exposure : float
            Current exposure value

        Returns
        -------
        LimitStatus
            Status of limit utilization
        """
        if current_exposure > self.hard_limit:
            return LimitStatus.HARD_BREACH
        elif current_exposure > self.soft_limit:
            return LimitStatus.SOFT_BREACH
        elif current_exposure > self.hard_limit * self.warning_threshold:
            return LimitStatus.WARNING
        else:
            return LimitStatus.OK

    def utilization_pct(self, current_exposure: float) -> float:
        """Calculate utilization as percentage of hard limit.

        Parameters
        ----------
        current_exposure : float
            Current exposure value

        Returns
        -------
        float
            Utilization percentage (0-100+)
        """
        return 100.0 * current_exposure / self.hard_limit

    def available_capacity(self, current_exposure: float) -> float:
        """Calculate remaining capacity before hard limit.

        Parameters
        ----------
        current_exposure : float
            Current exposure value

        Returns
        -------
        float
            Remaining capacity
        """
        return max(0.0, self.hard_limit - current_exposure)


@dataclass
class NotionalLimit(Limit):
    """Notional exposure limit by product/desk.

    Examples
    --------
    >>> limit = NotionalLimit(
    ...     name="Rates Swaps Desk",
    ...     hard_limit=1e9,  # $1B
    ...     soft_limit=800e6,  # $800M
    ...     scope={"desk": "rates", "product": "swaps"}
    ... )
    """

    def __init__(
        self,
        name: str,
        hard_limit: float,
        soft_limit: Optional[float] = None,
        warning_threshold: float = 0.8,
        scope: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name=name,
            limit_type=LimitType.NOTIONAL,
            hard_limit=hard_limit,
            soft_limit=soft_limit,
            warning_threshold=warning_threshold,
            scope=scope or {},
        )


@dataclass
class VaRLimit(Limit):
    """Value-at-Risk limit.

    Attributes
    ----------
    confidence_level : float
        VaR confidence level (e.g., 0.99)
    horizon_days : int
        VaR time horizon in days

    Examples
    --------
    >>> limit = VaRLimit(
    ...     name="Equity Desk 99% 1-day VaR",
    ...     hard_limit=10e6,  # $10M VaR limit
    ...     confidence_level=0.99,
    ...     horizon_days=1,
    ...     scope={"desk": "equity"}
    ... )
    """

    confidence_level: float = 0.99
    horizon_days: int = 1

    def __init__(
        self,
        name: str,
        hard_limit: float,
        confidence_level: float = 0.99,
        horizon_days: int = 1,
        soft_limit: Optional[float] = None,
        warning_threshold: float = 0.8,
        scope: Optional[Dict[str, str]] = None,
    ):
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
        super().__init__(
            name=name,
            limit_type=LimitType.VAR,
            hard_limit=hard_limit,
            soft_limit=soft_limit,
            warning_threshold=warning_threshold,
            scope=scope or {},
        )


@dataclass
class ConcentrationLimit(Limit):
    """Concentration limit for single-name/sector exposure.

    Attributes
    ----------
    concentration_type : str
        Type of concentration ("single_name", "sector", "geography", etc.)
    max_percentage : float
        Maximum percentage of portfolio in single entity

    Examples
    --------
    >>> limit = ConcentrationLimit(
    ...     name="Single Name Concentration",
    ...     hard_limit=0.10,  # 10% max in single name
    ...     concentration_type="single_name",
    ...     scope={"portfolio": "credit"}
    ... )
    """

    concentration_type: str = "single_name"
    max_percentage: float = 0.10

    def __init__(
        self,
        name: str,
        hard_limit: float,
        concentration_type: str = "single_name",
        max_percentage: Optional[float] = None,
        soft_limit: Optional[float] = None,
        warning_threshold: float = 0.8,
        scope: Optional[Dict[str, str]] = None,
    ):
        self.concentration_type = concentration_type
        self.max_percentage = max_percentage or hard_limit
        super().__init__(
            name=name,
            limit_type=LimitType.CONCENTRATION,
            hard_limit=hard_limit,
            soft_limit=soft_limit,
            warning_threshold=warning_threshold,
            scope=scope or {},
        )


@dataclass
class IssuerLimit(Limit):
    """Credit exposure limit to single issuer/counterparty.

    Attributes
    ----------
    issuer_id : str
        Issuer/counterparty identifier
    credit_rating : Optional[str]
        Credit rating of issuer

    Examples
    --------
    >>> limit = IssuerLimit(
    ...     name="Bank XYZ Exposure",
    ...     issuer_id="BANK_XYZ",
    ...     hard_limit=500e6,  # $500M exposure limit
    ...     credit_rating="AA",
    ... )
    """

    issuer_id: str = ""
    credit_rating: Optional[str] = None

    def __init__(
        self,
        name: str,
        issuer_id: str,
        hard_limit: float,
        credit_rating: Optional[str] = None,
        soft_limit: Optional[float] = None,
        warning_threshold: float = 0.8,
        scope: Optional[Dict[str, str]] = None,
    ):
        self.issuer_id = issuer_id
        self.credit_rating = credit_rating
        super().__init__(
            name=name,
            limit_type=LimitType.ISSUER,
            hard_limit=hard_limit,
            soft_limit=soft_limit,
            warning_threshold=warning_threshold,
            scope=scope or {},
        )


# ==============================================================================
# Limit Breach and Notification
# ==============================================================================


@dataclass
class LimitBreach:
    """Record of a limit breach.

    Attributes
    ----------
    limit : Limit
        The limit that was breached
    current_exposure : float
        Exposure at time of breach
    excess_amount : float
        Amount by which limit was exceeded
    status : LimitStatus
        Breach severity
    timestamp : datetime
        Time of breach
    trade_details : Optional[Dict]
        Details of trade causing breach
    """

    limit: Limit
    current_exposure: float
    excess_amount: float
    status: LimitStatus
    timestamp: datetime = field(default_factory=datetime.now)
    trade_details: Optional[Dict] = None

    @property
    def severity(self) -> BreachSeverity:
        """Determine breach severity."""
        if self.status == LimitStatus.HARD_BREACH:
            return BreachSeverity.CRITICAL
        elif self.status == LimitStatus.SOFT_BREACH:
            return BreachSeverity.WARNING
        else:
            return BreachSeverity.INFO

    def to_alert(self) -> Dict:
        """Convert breach to alert message."""
        return {
            "limit_name": self.limit.name,
            "limit_type": self.limit.limit_type.value,
            "hard_limit": self.limit.hard_limit,
            "current_exposure": self.current_exposure,
            "excess_amount": self.excess_amount,
            "utilization_pct": self.limit.utilization_pct(self.current_exposure),
            "status": self.status.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "scope": self.limit.scope,
            "trade_details": self.trade_details,
        }


# ==============================================================================
# Limit Manager
# ==============================================================================


@dataclass
class LimitManager:
    """Central limit management and monitoring.

    Manages all limits, tracks utilization, and generates breach alerts.

    Attributes
    ----------
    limits : Dict[str, Limit]
        Dictionary of all defined limits
    breaches : List[LimitBreach]
        History of limit breaches
    """

    limits: Dict[str, Limit] = field(default_factory=dict)
    breaches: List[LimitBreach] = field(default_factory=list)

    def add_limit(self, limit: Limit):
        """Add a limit to the manager.

        Parameters
        ----------
        limit : Limit
            Limit to add
        """
        self.limits[limit.name] = limit

    def remove_limit(self, limit_name: str):
        """Remove a limit from the manager.

        Parameters
        ----------
        limit_name : str
            Name of limit to remove
        """
        if limit_name in self.limits:
            del self.limits[limit_name]

    def check_limit(
        self,
        limit_name: str,
        current_exposure: float,
        trade_details: Optional[Dict] = None,
    ) -> LimitBreach | None:
        """Check if exposure breaches a limit.

        Parameters
        ----------
        limit_name : str
            Name of limit to check
        current_exposure : float
            Current exposure value
        trade_details : Optional[Dict]
            Details of trade being checked

        Returns
        -------
        LimitBreach | None
            Breach record if limit violated, None otherwise
        """
        if limit_name not in self.limits:
            raise ValueError(f"Unknown limit: {limit_name}")

        limit = self.limits[limit_name]
        status = limit.check_utilization(current_exposure)

        if status in (LimitStatus.SOFT_BREACH, LimitStatus.HARD_BREACH):
            breach = LimitBreach(
                limit=limit,
                current_exposure=current_exposure,
                excess_amount=current_exposure - limit.hard_limit,
                status=status,
                trade_details=trade_details,
            )
            self.breaches.append(breach)
            return breach

        return None

    def check_all_limits(
        self,
        exposures: Dict[str, float],
        trade_details: Optional[Dict] = None,
    ) -> List[LimitBreach]:
        """Check all limits against current exposures.

        Parameters
        ----------
        exposures : Dict[str, float]
            Dictionary mapping limit names to current exposures
        trade_details : Optional[Dict]
            Details of trade being checked

        Returns
        -------
        List[LimitBreach]
            List of all breaches
        """
        breaches = []
        for limit_name, exposure in exposures.items():
            if limit_name in self.limits:
                breach = self.check_limit(limit_name, exposure, trade_details)
                if breach:
                    breaches.append(breach)
        return breaches

    def get_limit_status_summary(self, exposures: Dict[str, float]) -> Dict:
        """Generate summary of all limit statuses.

        Parameters
        ----------
        exposures : Dict[str, float]
            Current exposures for each limit

        Returns
        -------
        Dict
            Summary of limit utilizations and statuses
        """
        summary = {
            "limits": {},
            "breaches": [],
            "warnings": [],
            "total_limits": len(self.limits),
            "breached_limits": 0,
            "warning_limits": 0,
        }

        for limit_name, limit in self.limits.items():
            exposure = exposures.get(limit_name, 0.0)
            status = limit.check_utilization(exposure)

            limit_info = {
                "name": limit_name,
                "type": limit.limit_type.value,
                "hard_limit": limit.hard_limit,
                "soft_limit": limit.soft_limit,
                "current_exposure": exposure,
                "utilization_pct": limit.utilization_pct(exposure),
                "available_capacity": limit.available_capacity(exposure),
                "status": status.value,
                "scope": limit.scope,
            }

            summary["limits"][limit_name] = limit_info

            if status in (LimitStatus.SOFT_BREACH, LimitStatus.HARD_BREACH):
                summary["breaches"].append(limit_name)
                summary["breached_limits"] += 1
            elif status == LimitStatus.WARNING:
                summary["warnings"].append(limit_name)
                summary["warning_limits"] += 1

        return summary

    def get_available_capacity(
        self,
        limit_name: str,
        current_exposure: float,
    ) -> float:
        """Get available capacity for a limit.

        Parameters
        ----------
        limit_name : str
            Name of limit
        current_exposure : float
            Current exposure

        Returns
        -------
        float
            Available capacity before hard limit
        """
        if limit_name not in self.limits:
            raise ValueError(f"Unknown limit: {limit_name}")

        return self.limits[limit_name].available_capacity(current_exposure)

    def get_breaches(
        self,
        severity: Optional[BreachSeverity] = None,
        limit_type: Optional[LimitType] = None,
    ) -> List[LimitBreach]:
        """Get breach history with optional filtering.

        Parameters
        ----------
        severity : Optional[BreachSeverity]
            Filter by severity
        limit_type : Optional[LimitType]
            Filter by limit type

        Returns
        -------
        List[LimitBreach]
            Filtered list of breaches
        """
        filtered = self.breaches

        if severity is not None:
            filtered = [b for b in filtered if b.severity == severity]

        if limit_type is not None:
            filtered = [b for b in filtered if b.limit.limit_type == limit_type]

        return filtered

    def clear_breach_history(self):
        """Clear all breach history."""
        self.breaches = []


# ==============================================================================
# Pre-Trade Control
# ==============================================================================


@dataclass
class PreTradeCheck:
    """Result of pre-trade limit check.

    Attributes
    ----------
    approved : bool
        Whether trade is approved
    breaches : List[LimitBreach]
        List of limits breached by trade
    warnings : List[str]
        List of warning messages
    available_capacity : Dict[str, float]
        Available capacity for each checked limit
    """

    approved: bool
    breaches: List[LimitBreach] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    available_capacity: Dict[str, float] = field(default_factory=dict)

    @property
    def has_hard_breaches(self) -> bool:
        """Check if any hard limit breaches exist."""
        return any(b.status == LimitStatus.HARD_BREACH for b in self.breaches)

    @property
    def has_soft_breaches(self) -> bool:
        """Check if any soft limit breaches exist."""
        return any(b.status == LimitStatus.SOFT_BREACH for b in self.breaches)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "approved": self.approved,
            "has_hard_breaches": self.has_hard_breaches,
            "has_soft_breaches": self.has_soft_breaches,
            "num_breaches": len(self.breaches),
            "breaches": [b.to_alert() for b in self.breaches],
            "warnings": self.warnings,
            "available_capacity": self.available_capacity,
        }


def pre_trade_control(
    limit_manager: LimitManager,
    current_exposures: Dict[str, float],
    proposed_trade_impact: Dict[str, float],
    trade_details: Optional[Dict] = None,
    allow_soft_breaches: bool = True,
) -> PreTradeCheck:
    """Perform pre-trade limit check.

    Parameters
    ----------
    limit_manager : LimitManager
        Limit manager with all limits
    current_exposures : Dict[str, float]
        Current exposures before trade
    proposed_trade_impact : Dict[str, float]
        Impact of proposed trade on each limit
    trade_details : Optional[Dict]
        Details of proposed trade
    allow_soft_breaches : bool
        Whether soft breaches are allowed (require approval)

    Returns
    -------
    PreTradeCheck
        Result of pre-trade check

    Example
    -------
    >>> manager = LimitManager()
    >>> manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9))
    >>> current = {"Swaps": 800e6}
    >>> impact = {"Swaps": 250e6}  # Proposed $250M trade
    >>> check = pre_trade_control(manager, current, impact)
    >>> if check.approved:
    ...     execute_trade()
    """
    # Calculate post-trade exposures
    post_trade_exposures = {}
    for limit_name in set(current_exposures.keys()) | set(proposed_trade_impact.keys()):
        current = current_exposures.get(limit_name, 0.0)
        impact = proposed_trade_impact.get(limit_name, 0.0)
        post_trade_exposures[limit_name] = current + impact

    # Check all limits
    breaches = limit_manager.check_all_limits(post_trade_exposures, trade_details)

    # Determine approval
    has_hard_breaches = any(b.status == LimitStatus.HARD_BREACH for b in breaches)
    has_soft_breaches = any(b.status == LimitStatus.SOFT_BREACH for b in breaches)

    if has_hard_breaches:
        approved = False
    elif has_soft_breaches and not allow_soft_breaches:
        approved = False
    else:
        approved = True

    # Generate warnings
    warnings = []
    for limit_name, post_exposure in post_trade_exposures.items():
        if limit_name in limit_manager.limits:
            limit = limit_manager.limits[limit_name]
            status = limit.check_utilization(post_exposure)
            if status == LimitStatus.WARNING:
                utilization = limit.utilization_pct(post_exposure)
                warnings.append(
                    f"{limit_name}: {utilization:.1f}% utilized (approaching limit)"
                )

    # Calculate available capacity
    available_capacity = {}
    for limit_name in post_trade_exposures.keys():
        if limit_name in limit_manager.limits:
            available_capacity[limit_name] = limit_manager.get_available_capacity(
                limit_name, post_trade_exposures[limit_name]
            )

    return PreTradeCheck(
        approved=approved,
        breaches=breaches,
        warnings=warnings,
        available_capacity=available_capacity,
    )


# ==============================================================================
# What-If Analysis
# ==============================================================================


@dataclass
class WhatIfScenario:
    """What-if scenario for limit impact analysis.

    Attributes
    ----------
    name : str
        Scenario name
    trade_impacts : Dict[str, float]
        Impact on each limit
    description : Optional[str]
        Scenario description
    """

    name: str
    trade_impacts: Dict[str, float]
    description: Optional[str] = None


def what_if_analysis(
    limit_manager: LimitManager,
    current_exposures: Dict[str, float],
    scenarios: List[WhatIfScenario],
) -> Dict[str, PreTradeCheck]:
    """Analyze impact of multiple scenarios on limits.

    Parameters
    ----------
    limit_manager : LimitManager
        Limit manager
    current_exposures : Dict[str, float]
        Current exposures
    scenarios : List[WhatIfScenario]
        List of scenarios to analyze

    Returns
    -------
    Dict[str, PreTradeCheck]
        Results for each scenario

    Example
    -------
    >>> scenarios = [
    ...     WhatIfScenario("Small Trade", {"Swaps": 100e6}),
    ...     WhatIfScenario("Large Trade", {"Swaps": 500e6}),
    ... ]
    >>> results = what_if_analysis(manager, current_exposures, scenarios)
    >>> for name, check in results.items():
    ...     print(f"{name}: {'APPROVED' if check.approved else 'REJECTED'}")
    """
    results = {}

    for scenario in scenarios:
        check = pre_trade_control(
            limit_manager,
            current_exposures,
            scenario.trade_impacts,
            trade_details={"scenario": scenario.name},
        )
        results[scenario.name] = check

    return results


__all__ = [
    # Enums
    "LimitType",
    "LimitStatus",
    "BreachSeverity",
    # Limit Classes
    "Limit",
    "NotionalLimit",
    "VaRLimit",
    "ConcentrationLimit",
    "IssuerLimit",
    # Breach and Management
    "LimitBreach",
    "LimitManager",
    # Pre-Trade Control
    "PreTradeCheck",
    "pre_trade_control",
    # What-If Analysis
    "WhatIfScenario",
    "what_if_analysis",
]
