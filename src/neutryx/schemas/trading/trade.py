"""Trade and position schemas for portfolio management.

This module provides the canonical trade representation used across
all Neutryx domains including portfolio, clearing, and risk management.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from neutryx.schemas.core.base import AuditedSchema, NeutryxSchema
from neutryx.schemas.core.enums import (
    CurrencyCode,
    ProductType,
    SettlementType,
    TradeStatus,
)
from neutryx.schemas.core.identifiers import Party


class TradeEconomics(NeutryxSchema):
    """Economic terms of a trade.

    Contains the core financial parameters that define a trade's value.
    """

    __schema_version__ = "1.0.0"
    __schema_domain__ = "trading"

    notional: Decimal = Field(
        ...,
        gt=0,
        description="Trade notional amount",
    )
    currency: CurrencyCode = Field(
        ...,
        description="Trade currency",
    )
    fixed_rate: Optional[Decimal] = Field(
        default=None,
        description="Fixed rate for swaps/FRAs",
    )
    floating_spread: Optional[Decimal] = Field(
        default=None,
        description="Spread over floating rate benchmark",
    )
    strike: Optional[Decimal] = Field(
        default=None,
        description="Option strike price",
    )
    price: Optional[Decimal] = Field(
        default=None,
        description="Trade price or premium",
    )
    quantity: Optional[Decimal] = Field(
        default=None,
        description="Number of units/contracts",
    )


class Trade(AuditedSchema):
    """Canonical trade representation for all Neutryx modules.

    This is the single source of truth for trade data, consolidating
    definitions from portfolio.contracts.trade and integrations.clearing.base.

    Attributes:
        trade_id: Unique internal identifier
        trade_number: Systematic trade number (e.g., TRD-20250315-0001)
        external_id: External system reference
        usi: Unique Swap Identifier for regulatory reporting
        uti: Unique Transaction Identifier for regulatory reporting
        product_type: Classification of the derivative product
        status: Current lifecycle status
        trade_date: Date the trade was executed
        effective_date: Date the trade becomes effective
        maturity_date: Maturity or expiration date
        buyer: Buying party
        seller: Selling party
        counterparty_id: ID reference to counterparty (alternative to full Party)
        economics: Financial terms of the trade
        book_id: Book the trade is assigned to
        desk_id: Trading desk
        trader_id: Trader responsible
        netting_set_id: Netting set for this trade
        settlement_type: Cash or physical settlement
        clearing_broker: Clearing broker identifier
        mtm: Current mark-to-market value
        last_valuation_date: Date of last MTM calculation
    """

    __schema_version__ = "1.0.0"
    __schema_domain__ = "trading"

    # Core identifiers
    trade_id: str = Field(
        ...,
        min_length=1,
        description="Unique trade identifier",
    )
    trade_number: Optional[str] = Field(
        default=None,
        description="Systematic trade number (e.g., TRD-20250315-0001)",
    )
    external_id: Optional[str] = Field(
        default=None,
        description="External system trade ID",
    )
    usi: Optional[str] = Field(
        default=None,
        description="Unique Swap Identifier (CFTC)",
    )
    uti: Optional[str] = Field(
        default=None,
        description="Unique Transaction Identifier (EMIR/MiFID)",
    )

    # Product classification
    product_type: ProductType = Field(
        ...,
        description="Product classification",
    )
    status: TradeStatus = Field(
        default=TradeStatus.PENDING,
        description="Trade lifecycle status",
    )

    # Dates
    trade_date: date = Field(
        ...,
        description="Trade execution date",
    )
    effective_date: Optional[date] = Field(
        default=None,
        description="Trade start/effective date",
    )
    maturity_date: Optional[date] = Field(
        default=None,
        description="Maturity or expiration date",
    )

    # Parties - support both embedded and reference patterns
    buyer: Optional[Party] = Field(
        default=None,
        description="Buying party (full details)",
    )
    seller: Optional[Party] = Field(
        default=None,
        description="Selling party (full details)",
    )
    counterparty_id: Optional[str] = Field(
        default=None,
        description="Counterparty ID reference",
    )

    # Economic terms
    economics: Optional[TradeEconomics] = Field(
        default=None,
        description="Trade economic terms",
    )
    # Flat economics for simpler use cases
    notional: Optional[Decimal] = Field(
        default=None,
        gt=0,
        description="Trade notional (alternative to economics.notional)",
    )
    currency: Optional[CurrencyCode] = Field(
        default=None,
        description="Trade currency (alternative to economics.currency)",
    )

    # Organization
    book_id: Optional[str] = Field(
        default=None,
        description="Book assignment",
    )
    desk_id: Optional[str] = Field(
        default=None,
        description="Trading desk",
    )
    trader_id: Optional[str] = Field(
        default=None,
        description="Responsible trader",
    )
    netting_set_id: Optional[str] = Field(
        default=None,
        description="Netting set ID for aggregation",
    )

    # Settlement and clearing
    settlement_type: Optional[SettlementType] = Field(
        default=None,
        description="Settlement method",
    )
    clearing_broker: Optional[str] = Field(
        default=None,
        description="Clearing broker ID",
    )
    collateral_currency: Optional[CurrencyCode] = Field(
        default=None,
        description="Collateral posting currency",
    )

    # Valuation
    mtm: Optional[float] = Field(
        default=None,
        description="Mark-to-market value (positive = asset)",
    )
    last_valuation_date: Optional[date] = Field(
        default=None,
        description="Date of last MTM calculation",
    )

    # Convention tracking
    convention_profile_id: Optional[str] = Field(
        default=None,
        description="Convention profile used for generation",
    )
    generated_from_convention: bool = Field(
        default=False,
        description="Whether trade was auto-generated from conventions",
    )

    # Flexible extension
    product_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Product-specific details (FpML object, etc.)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @field_validator("maturity_date")
    @classmethod
    def maturity_after_trade_date(
        cls, v: Optional[date], info
    ) -> Optional[date]:
        """Validate maturity date is after trade date."""
        if v is not None and "trade_date" in info.data:
            trade_date = info.data["trade_date"]
            if v < trade_date:
                raise ValueError("maturity_date must be >= trade_date")
        return v

    @field_validator("effective_date")
    @classmethod
    def effective_on_or_after_trade(
        cls, v: Optional[date], info
    ) -> Optional[date]:
        """Validate effective date is on or after trade date."""
        if v is not None and "trade_date" in info.data:
            trade_date = info.data["trade_date"]
            if v < trade_date:
                raise ValueError("effective_date must be >= trade_date")
        return v

    def __repr__(self) -> str:
        """Concise string representation."""
        parts = [
            f"Trade(id='{self.trade_id}'",
            f"product={self.product_type.value}",
        ]
        if self.counterparty_id:
            parts.append(f"cpty='{self.counterparty_id}'")
        if self.notional and self.currency:
            parts.append(f"notional={self.notional:,.0f} {self.currency.value}")
        if self.maturity_date:
            parts.append(f"mat={self.maturity_date}")
        return ", ".join(parts) + ")"


class TradeSubmissionResponse(NeutryxSchema):
    """Response from submitting a trade to a CCP or counterparty.

    Contains submission status and any assigned identifiers.
    """

    __schema_version__ = "1.0.0"
    __schema_domain__ = "trading"

    submission_id: str = Field(
        ...,
        description="Submission reference ID",
    )
    trade_id: str = Field(
        ...,
        description="Original trade ID",
    )
    status: TradeStatus = Field(
        ...,
        description="Submission status",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )
    ccp_trade_id: Optional[str] = Field(
        default=None,
        description="CCP-assigned trade ID",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Rejection reason if applicable",
    )
    rejection_code: Optional[str] = Field(
        default=None,
        description="Rejection code from CCP",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response data",
    )
