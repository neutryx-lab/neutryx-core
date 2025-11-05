"""Base classes and protocols for CCP (Central Counterparty) integrations.

This module provides the foundational architecture for integrating with
clearing houses and CCPs. It defines common interfaces, message types,
and protocols that all CCP implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class TradeStatus(str, Enum):
    """Trade lifecycle status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CLEARED = "cleared"
    SETTLED = "settled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MessageType(str, Enum):
    """CCP message types."""
    TRADE_SUBMISSION = "trade_submission"
    TRADE_CONFIRMATION = "trade_confirmation"
    TRADE_REJECTION = "trade_rejection"
    MARGIN_CALL = "margin_call"
    SETTLEMENT = "settlement"
    POSITION_REPORT = "position_report"
    RISK_REPORT = "risk_report"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"


class ProductType(str, Enum):
    """Clearable product types."""
    IRS = "interest_rate_swap"
    CDS = "credit_default_swap"
    FX_FORWARD = "fx_forward"
    FX_SWAP = "fx_swap"
    EQUITY_OPTION = "equity_option"
    COMMODITY_FUTURE = "commodity_future"
    REPO = "repo"
    SWAPTION = "swaption"


class CCPError(Exception):
    """Base exception for CCP operations."""
    pass


class CCPConnectionError(CCPError):
    """Connection-related errors."""
    pass


class CCPAuthenticationError(CCPError):
    """Authentication failures."""
    pass


class CCPTradeRejectionError(CCPError):
    """Trade rejection by CCP."""

    def __init__(self, message: str, rejection_code: Optional[str] = None):
        super().__init__(message)
        self.rejection_code = rejection_code


class CCPTimeoutError(CCPError):
    """Operation timeout."""
    pass


class Party(BaseModel):
    """Trading party information."""
    model_config = ConfigDict(frozen=True)

    party_id: str = Field(..., description="Unique party identifier")
    name: str = Field(..., description="Party legal name")
    lei: Optional[str] = Field(None, description="Legal Entity Identifier")
    bic: Optional[str] = Field(None, description="Bank Identifier Code")
    member_id: Optional[str] = Field(None, description="CCP member ID")


class TradeEconomics(BaseModel):
    """Trade economic terms."""

    notional: Decimal = Field(..., description="Trade notional amount")
    currency: str = Field(..., description="Trade currency")
    fixed_rate: Optional[Decimal] = Field(None, description="Fixed rate for swaps")
    spread: Optional[Decimal] = Field(None, description="Spread over benchmark")
    strike: Optional[Decimal] = Field(None, description="Option strike")
    price: Optional[Decimal] = Field(None, description="Trade price")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with floats."""
        return {
            "notional": float(self.notional),
            "currency": self.currency,
            "fixed_rate": float(self.fixed_rate) if self.fixed_rate else None,
            "spread": float(self.spread) if self.spread else None,
            "strike": float(self.strike) if self.strike else None,
            "price": float(self.price) if self.price else None,
        }


class Trade(BaseModel):
    """Standardized trade representation."""

    trade_id: str = Field(..., description="Unique trade identifier")
    product_type: ProductType = Field(..., description="Product classification")
    trade_date: datetime = Field(..., description="Trade execution date")
    effective_date: datetime = Field(..., description="Trade start date")
    maturity_date: datetime = Field(..., description="Trade maturity date")
    buyer: Party = Field(..., description="Buyer party")
    seller: Party = Field(..., description="Seller party")
    economics: TradeEconomics = Field(..., description="Trade economics")

    # Optional fields
    uti: Optional[str] = Field(None, description="Unique Transaction Identifier")
    clearing_broker: Optional[str] = Field(None, description="Clearing broker ID")
    collateral_currency: Optional[str] = Field(None, description="Collateral currency")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "product_type": self.product_type.value,
            "trade_date": self.trade_date.isoformat(),
            "effective_date": self.effective_date.isoformat(),
            "maturity_date": self.maturity_date.isoformat(),
            "buyer": self.buyer.model_dump(),
            "seller": self.seller.model_dump(),
            "economics": self.economics.to_dict(),
            "uti": self.uti,
            "clearing_broker": self.clearing_broker,
            "collateral_currency": self.collateral_currency,
            "metadata": self.metadata,
        }


class TradeSubmissionResponse(BaseModel):
    """Response from trade submission."""

    submission_id: str = Field(..., description="Submission reference ID")
    trade_id: str = Field(..., description="Original trade ID")
    status: TradeStatus = Field(..., description="Submission status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ccp_trade_id: Optional[str] = Field(None, description="CCP-assigned trade ID")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason if applicable")
    rejection_code: Optional[str] = Field(None, description="Rejection code")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarginCall(BaseModel):
    """Margin call notification."""

    call_id: str = Field(..., description="Margin call ID")
    member_id: str = Field(..., description="Member receiving call")
    call_amount: Decimal = Field(..., description="Amount required")
    currency: str = Field(..., description="Call currency")
    call_type: str = Field(..., description="Type: initial/variation/additional")
    due_time: datetime = Field(..., description="Payment deadline")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    portfolio_im: Optional[Decimal] = Field(None, description="Portfolio IM")
    trade_im: Optional[Decimal] = Field(None, description="Trade-level IM")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PositionReport(BaseModel):
    """Position report from CCP."""

    report_id: str = Field(..., description="Report identifier")
    member_id: str = Field(..., description="Member ID")
    as_of_date: datetime = Field(..., description="Report valuation date")
    positions: List[Dict[str, Any]] = Field(default_factory=list)
    total_exposure: Optional[Decimal] = Field(None, description="Total exposure")
    initial_margin: Optional[Decimal] = Field(None, description="Initial margin requirement")
    variation_margin: Optional[Decimal] = Field(None, description="Variation margin")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CCPConfig(BaseModel):
    """Configuration for CCP connectivity."""

    ccp_name: str = Field(..., description="CCP name")
    member_id: str = Field(..., description="Member ID")
    api_endpoint: str = Field(..., description="API endpoint URL")

    # Authentication
    api_key: Optional[str] = Field(None, description="API key")
    api_secret: Optional[str] = Field(None, description="API secret")
    certificate_path: Optional[str] = Field(None, description="Certificate file path")
    private_key_path: Optional[str] = Field(None, description="Private key path")

    # Connection settings
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")

    # Environment
    environment: str = Field(default="production", description="Environment: production/uat/test")
    use_sandbox: bool = Field(default=False, description="Use sandbox environment")

    # Protocol settings
    protocol: str = Field(default="REST", description="Protocol: REST/FIX/SWIFT")
    message_format: str = Field(default="JSON", description="Message format: JSON/XML/FIX")

    # Additional settings
    enable_heartbeat: bool = Field(default=True, description="Enable heartbeat messages")
    heartbeat_interval: int = Field(default=60, description="Heartbeat interval in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CCPConnector(ABC):
    """Abstract base class for CCP connectivity.

    All CCP implementations must inherit from this class and implement
    the required methods for trade submission, status checking, and
    lifecycle management.
    """

    def __init__(self, config: CCPConfig):
        """Initialize CCP connector.

        Args:
            config: CCP configuration
        """
        self.config = config
        self._connected = False
        self._session_id: Optional[str] = None

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to CCP.

        Returns:
            True if connection successful

        Raises:
            CCPConnectionError: If connection fails
            CCPAuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from CCP.

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        """Submit trade for clearing.

        Args:
            trade: Trade to submit

        Returns:
            Submission response with status

        Raises:
            CCPTradeRejectionError: If trade is rejected
            CCPConnectionError: If connection fails
            CCPTimeoutError: If request times out
        """
        pass

    @abstractmethod
    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        """Get current status of a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Current trade status
        """
        pass

    @abstractmethod
    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel a pending trade.

        Args:
            trade_id: Trade identifier

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    async def get_margin_requirements(
        self,
        member_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current margin requirements.

        Args:
            member_id: Member ID (defaults to configured member)

        Returns:
            Dictionary with margin requirements
        """
        pass

    @abstractmethod
    async def get_position_report(
        self,
        as_of_date: Optional[datetime] = None
    ) -> PositionReport:
        """Get position report.

        Args:
            as_of_date: Report date (defaults to today)

        Returns:
            Position report
        """
        pass

    @abstractmethod
    async def healthcheck(self) -> bool:
        """Check CCP connectivity health.

        Returns:
            True if connection healthy
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._session_id

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"ccp={self.config.ccp_name}, "
            f"member={self.config.member_id}, "
            f"connected={self._connected})"
        )


class CCPMessageProtocol(Protocol):
    """Protocol for CCP message formatting."""

    def format_trade_submission(self, trade: Trade) -> bytes:
        """Format trade submission message."""
        ...

    def parse_submission_response(self, response: bytes) -> TradeSubmissionResponse:
        """Parse submission response."""
        ...

    def format_status_request(self, trade_id: str) -> bytes:
        """Format status request."""
        ...

    def parse_status_response(self, response: bytes) -> TradeStatus:
        """Parse status response."""
        ...


@dataclass
class CCPMetrics:
    """Metrics for CCP operations."""

    total_submissions: int = 0
    successful_submissions: int = 0
    rejected_submissions: int = 0
    failed_submissions: int = 0
    avg_response_time_ms: float = 0.0
    connection_uptime_pct: float = 100.0
    last_heartbeat: Optional[datetime] = None

    response_times: List[float] = field(default_factory=list)

    def record_submission(
        self,
        success: bool,
        response_time_ms: float,
        rejected: bool = False
    ):
        """Record submission metrics."""
        self.total_submissions += 1
        if success:
            self.successful_submissions += 1
        elif rejected:
            self.rejected_submissions += 1
        else:
            self.failed_submissions += 1

        self.response_times.append(response_time_ms)
        if len(self.response_times) > 1000:
            self.response_times.pop(0)

        self.avg_response_time_ms = float(np.mean(self.response_times))

    def success_rate(self) -> float:
        """Calculate submission success rate."""
        if self.total_submissions == 0:
            return 0.0
        return self.successful_submissions / self.total_submissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_submissions": self.total_submissions,
            "successful_submissions": self.successful_submissions,
            "rejected_submissions": self.rejected_submissions,
            "failed_submissions": self.failed_submissions,
            "success_rate": self.success_rate(),
            "avg_response_time_ms": self.avg_response_time_ms,
            "connection_uptime_pct": self.connection_uptime_pct,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
        }


__all__ = [
    "CCPConnector",
    "CCPConfig",
    "CCPError",
    "CCPConnectionError",
    "CCPAuthenticationError",
    "CCPTradeRejectionError",
    "CCPTimeoutError",
    "CCPMessageProtocol",
    "CCPMetrics",
    "Trade",
    "TradeEconomics",
    "TradeSubmissionResponse",
    "TradeStatus",
    "Party",
    "MarginCall",
    "PositionReport",
    "MessageType",
    "ProductType",
]
