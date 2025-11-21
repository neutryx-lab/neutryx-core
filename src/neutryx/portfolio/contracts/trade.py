"""Trade and position abstractions for portfolio management.

This module provides base classes for representing individual trades and positions,
supporting portfolio aggregation, netting, and XVA calculations.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TradeStatus(str, Enum):
    """Lifecycle status of a trade."""

    PENDING = "Pending"  # Awaiting confirmation
    ACTIVE = "Active"  # Live and accruing exposure
    TERMINATED = "Terminated"  # Early termination
    MATURED = "Matured"  # Reached maturity
    CANCELLED = "Cancelled"  # Cancelled before execution
    NOVATED = "Novated"  # Transferred to another party


class ProductType(str, Enum):
    """High-level product classification."""

    EQUITY_OPTION = "EquityOption"
    FX_OPTION = "FxOption"
    INTEREST_RATE_SWAP = "InterestRateSwap"
    CREDIT_DEFAULT_SWAP = "CreditDefaultSwap"
    VARIANCE_SWAP = "VarianceSwap"
    SWAPTION = "Swaption"
    FORWARD = "Forward"
    FUTURE = "Future"
    OTHER = "Other"


class SettlementType(str, Enum):
    """Settlement method for derivatives."""

    CASH = "Cash"
    PHYSICAL = "Physical"
    ELECTION = "Election"  # Counterparty election


class Trade(BaseModel):
    """Base trade representation for risk management.

    Attributes
    ----------
    id : str
        Unique internal trade identifier
    trade_number : str, optional
        Systematic trade number (e.g., TRD-20250315-0001)
    external_id : str, optional
        External trade ID (e.g., from front office system)
    usi : str, optional
        Unique Swap Identifier (regulatory reporting)
    counterparty_id : str
        ID of the counterparty (references Counterparty.id)
    netting_set_id : str, optional
        ID of the netting set this trade belongs to
    book_id : str, optional
        ID of the book this trade is assigned to
    desk_id : str, optional
        ID of the trading desk (derived from book hierarchy)
    trader_id : str, optional
        ID of the trader responsible for this trade
    product_type : ProductType
        High-level product classification
    trade_date : date
        Date the trade was executed
    effective_date : date, optional
        Date the trade becomes effective (may differ from trade_date)
    maturity_date : date, optional
        Maturity or expiration date
    status : TradeStatus
        Current lifecycle status
    notional : float, optional
        Notional amount in base currency
    currency : str, optional
        Currency of the trade (ISO 4217)
    settlement_type : SettlementType, optional
        Cash or physical settlement
    product_details : dict, optional
        Product-specific details (FpML object, custom structure, etc.)
    mtm : float, optional
        Mark-to-market value (positive = owe to us, negative = we owe)
    last_valuation_date : date, optional
        Date of last MTM calculation
    """

    id: str
    trade_number: Optional[str] = Field(
        default=None,
        description="Systematic trade number (e.g., TRD-20250315-0001)",
    )
    external_id: Optional[str] = None
    usi: Optional[str] = Field(
        default=None,
        description="Unique Swap Identifier for regulatory reporting",
    )
    counterparty_id: str
    netting_set_id: Optional[str] = None
    book_id: Optional[str] = Field(
        default=None,
        description="ID of the book this trade is assigned to",
    )
    desk_id: Optional[str] = Field(
        default=None,
        description="ID of the trading desk",
    )
    trader_id: Optional[str] = Field(
        default=None,
        description="ID of the trader responsible for this trade",
    )
    product_type: ProductType
    trade_date: date
    effective_date: Optional[date] = None
    maturity_date: Optional[date] = None
    status: TradeStatus = TradeStatus.ACTIVE
    notional: Optional[float] = Field(default=None, gt=0.0)
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)
    settlement_type: Optional[SettlementType] = None
    product_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flexible storage for product-specific data",
    )
    convention_profile_id: Optional[str] = Field(
        default=None,
        description="ID of the convention profile used to generate this trade",
    )
    generated_from_convention: bool = Field(
        default=False,
        description="Whether this trade was generated from market conventions",
    )
    mtm: Optional[float] = None
    last_valuation_date: Optional[date] = None

    def __repr__(self) -> str:
        """String representation."""
        maturity_str = f", maturity={self.maturity_date}" if self.maturity_date else ""
        notional_str = (
            f", notional={self.notional:,.0f} {self.currency}"
            if self.notional and self.currency
            else ""
        )
        return (
            f"Trade(id='{self.id}', "
            f"product={self.product_type.value}, "
            f"counterparty='{self.counterparty_id}'"
            f"{maturity_str}{notional_str})"
        )

    def is_active(self) -> bool:
        """Check if trade is currently active."""
        return self.status == TradeStatus.ACTIVE

    def is_expired(self, as_of_date: date) -> bool:
        """Check if trade has expired as of a given date.

        Parameters
        ----------
        as_of_date : date
            Valuation date

        Returns
        -------
        bool
            True if maturity_date is before as_of_date
        """
        if self.maturity_date is None:
            return False
        return self.maturity_date < as_of_date

    def time_to_maturity(self, as_of_date: date) -> Optional[float]:
        """Calculate time to maturity in years.

        Parameters
        ----------
        as_of_date : date
            Valuation date

        Returns
        -------
        float or None
            Time to maturity in years (Act/365), None if no maturity date
        """
        if self.maturity_date is None:
            return None
        days = (self.maturity_date - as_of_date).days
        return max(0.0, days / 365.0)

    def get_mtm(self, default: float = 0.0) -> float:
        """Get mark-to-market value with a default fallback.

        Parameters
        ----------
        default : float
            Default value if MTM is not set

        Returns
        -------
        float
            MTM value or default
        """
        if self.mtm is None:
            return default
        return self.mtm

    def update_mtm(self, mtm_value: float, valuation_date: date) -> None:
        """Update MTM and valuation date.

        Parameters
        ----------
        mtm_value : float
            New mark-to-market value
        valuation_date : date
            Valuation date
        """
        self.mtm = mtm_value
        self.last_valuation_date = valuation_date

    def belongs_to_netting_set(self, netting_set_id: str) -> bool:
        """Check if trade belongs to a specific netting set.

        Parameters
        ----------
        netting_set_id : str
            Netting set ID to check

        Returns
        -------
        bool
            True if trade belongs to the netting set
        """
        return self.netting_set_id == netting_set_id

    def get_effective_date(self) -> date:
        """Get effective date, falling back to trade date if not set.

        Returns
        -------
        date
            Effective date or trade date
        """
        return self.effective_date if self.effective_date else self.trade_date
