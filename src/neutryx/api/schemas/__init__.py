"""Shared API schema models for Neutryx.

This module contains Pydantic models for API requests and responses,
shared across different components to avoid circular dependencies.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

import jax.numpy as jnp
from fastapi import HTTPException
from pydantic import BaseModel, Field

from neutryx.integrations.clearing.base import Party, ProductType
from neutryx.integrations.clearing.rfq import (
    AuctionResult,
    AuctionType,
    OrderSide,
    Quote,
    RFQ,
    RFQSpecification,
    RFQStatus,
    TimeInForce,
)


class VanillaOptionRequest(BaseModel):
    """Request model for vanilla option pricing."""

    spot: float = Field(..., description="Current underlying spot level")
    strike: float = Field(..., description="Option strike price")
    maturity: float = Field(..., description="Time to maturity in years")
    rate: float = Field(0.0, description="Risk-free rate")
    dividend: float = Field(0.0, description="Dividend yield")
    volatility: float = Field(..., description="Volatility of the underlying")
    steps: int = Field(64, gt=0, description="Number of time steps")
    paths: int = Field(8192, gt=0, description="Number of Monte Carlo paths")
    antithetic: bool = Field(False, description="Use antithetic sampling")
    call: bool = Field(True, description="Price a call (False for put)")
    seed: int | None = Field(None, description="PRNG seed for simulation determinism")


class ProfileRequest(BaseModel):
    """Request model for exposure or discount profiles."""

    values: List[float]

    def to_array(self) -> jnp.ndarray:
        """Convert values to JAX array."""
        if not self.values:
            raise HTTPException(status_code=400, detail="Expected non-empty sequence")
        return jnp.asarray(self.values, dtype=jnp.float32)


class CVARequest(BaseModel):
    """Request model for CVA calculation."""

    epe: ProfileRequest
    discount: ProfileRequest
    default_probability: ProfileRequest
    lgd: float = Field(0.6, description="Loss given default")


class FVARequest(BaseModel):
    """Request model for FVA calculation."""

    epe: ProfileRequest
    discount: ProfileRequest
    funding_spread: Sequence[float] | float


class MVARequest(BaseModel):
    """Request model for MVA calculation."""

    initial_margin: ProfileRequest
    discount: ProfileRequest
    spread: Sequence[float] | float


class PortfolioXVARequest(BaseModel):
    """Request to compute XVA for a portfolio or netting set."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    netting_set_id: str | None = Field(
        None, description="Netting set ID (if None, compute for entire portfolio)"
    )
    valuation_date: str = Field(..., description="Valuation date (ISO format)")
    compute_cva: bool = Field(True, description="Compute CVA")
    compute_dva: bool = Field(False, description="Compute DVA")
    compute_fva: bool = Field(False, description="Compute FVA")
    compute_mva: bool = Field(False, description="Compute MVA")
    lgd: float = Field(0.6, description="Loss given default (if not in counterparty data)")
    funding_spread_bps: float = Field(50.0, description="Funding spread in bps for FVA")
    time_grid: List[float] | None = Field(
        None, description="Optional exposure time grid overriding Monte Carlo steps"
    )
    market_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market data bundle used for exposure simulation",
    )
    monte_carlo: Dict[str, int] = Field(
        default_factory=lambda: {"steps": 32, "paths": 2048, "seed": 0},
        description="Monte Carlo configuration (steps, paths, seed)",
    )
    discount_curve: ProfileRequest | None = Field(
        None, description="Discount factors aligned with the exposure grid"
    )
    counterparty_pd: ProfileRequest | None = Field(
        None, description="Counterparty cumulative default probabilities"
    )
    own_pd: ProfileRequest | None = Field(
        None, description="Own cumulative default probabilities for DVA"
    )
    lgd_curve: ProfileRequest | None = Field(
        None, description="Term structure of counterparty LGD"
    )
    own_lgd_curve: ProfileRequest | None = Field(
        None, description="Term structure of own LGD"
    )
    funding_curve: ProfileRequest | None = Field(
        None, description="Funding spread profile for FVA (decimal, not bps)"
    )
    initial_margin: ProfileRequest | None = Field(
        None, description="Initial margin profile used for MVA"
    )


class PortfolioSummaryRequest(BaseModel):
    """Request to get portfolio summary statistics."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    valuation_date: str | None = Field(None, description="Valuation date (ISO format)")


class FpMLPriceRequest(BaseModel):
    """Request to price an FpML document."""

    fpml_xml: str = Field(..., description="FpML XML document")
    market_data: dict[str, Any] = Field(
        ..., description="Market data: spot, volatility, rate, dividend, etc."
    )
    steps: int = Field(252, gt=0, description="Number of time steps")
    paths: int = Field(100_000, gt=0, description="Number of Monte Carlo paths")
    seed: int = Field(42, description="Random seed")


class FpMLParseRequest(BaseModel):
    """Request to parse an FpML document."""

    fpml_xml: str = Field(..., description="FpML XML document")


class FpMLValidateRequest(BaseModel):
    """Request to validate an FpML document."""

    fpml_xml: str = Field(..., description="FpML XML document")


class PartyPayload(BaseModel):
    """API payload for a trading party."""

    party_id: str = Field(..., description="Unique party identifier")
    name: str = Field(..., description="Party legal name")
    lei: Optional[str] = Field(None, description="Legal Entity Identifier")
    bic: Optional[str] = Field(None, description="Bank Identifier Code")
    member_id: Optional[str] = Field(None, description="CCP member ID")

    def to_party(self) -> Party:
        """Convert payload to :class:`Party`."""

        return Party(
            party_id=self.party_id,
            name=self.name,
            lei=self.lei,
            bic=self.bic,
            member_id=self.member_id,
        )


class RFQSpecificationPayload(BaseModel):
    """API payload mirroring :class:`RFQSpecification`."""

    product_type: str = Field(..., description="Product type identifier")
    notional: Decimal = Field(..., description="Notional amount")
    currency: str = Field(..., description="Currency")
    effective_date: datetime = Field(..., description="Effective date")
    maturity_date: datetime = Field(..., description="Maturity date")
    fixed_rate: Optional[Decimal] = Field(None, description="Fixed rate for swaps")
    strike: Optional[Decimal] = Field(None, description="Strike for options")
    underlying: Optional[str] = Field(None, description="Underlying asset")
    tenor: Optional[str] = Field(None, description="Tenor (e.g., 5Y)")
    settlement_type: str = Field(default="physical", description="Settlement type")
    day_count: str = Field(default="ACT/360", description="Day count convention")
    payment_frequency: Optional[str] = Field(None, description="Payment frequency")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_specification(self) -> RFQSpecification:
        """Convert payload to :class:`RFQSpecification`."""

        return RFQSpecification(
            product_type=ProductType(self.product_type),
            notional=self.notional,
            currency=self.currency,
            effective_date=self.effective_date,
            maturity_date=self.maturity_date,
            fixed_rate=self.fixed_rate,
            strike=self.strike,
            underlying=self.underlying,
            tenor=self.tenor,
            settlement_type=self.settlement_type,
            day_count=self.day_count,
            payment_frequency=self.payment_frequency,
            metadata=self.metadata,
        )


class RFQCreateRequest(BaseModel):
    """Request body for RFQ creation."""

    requester: PartyPayload
    specification: RFQSpecificationPayload
    side: OrderSide
    auction_type: AuctionType = Field(default=AuctionType.SINGLE_PRICE)
    quote_deadline: datetime
    expiry_time: Optional[datetime] = Field(None, description="Optional explicit expiry time")
    clearing_broker: Optional[str] = Field(None, description="Clearing broker identifier")
    max_participants: Optional[int] = Field(None, description="Max number of dealers")
    invited_dealers: Optional[List[str]] = Field(None, description="List of invited dealer IDs")
    allow_partial_fills: bool = Field(False, description="Allow partial fills")
    require_all_or_none: bool = Field(False, description="All-or-none execution")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_rfq_kwargs(self) -> Dict[str, Any]:
        """Convert to RFQ creation keyword arguments."""

        kwargs: Dict[str, Any] = {
            "requester": self.requester.to_party(),
            "specification": self.specification.to_specification(),
            "side": self.side,
            "quote_deadline": self.quote_deadline,
            "auction_type": self.auction_type,
            "clearing_broker": self.clearing_broker,
            "max_participants": self.max_participants,
            "invited_dealers": self.invited_dealers,
            "allow_partial_fills": self.allow_partial_fills,
            "require_all_or_none": self.require_all_or_none,
            "metadata": self.metadata,
        }
        if self.expiry_time is not None:
            kwargs["expiry_time"] = self.expiry_time
        return kwargs


class QuoteSubmissionRequest(BaseModel):
    """Request body for quote submission."""

    quoter: PartyPayload
    quoter_member_id: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    yield_value: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    min_quantity: Optional[Decimal] = None
    time_in_force: str = Field("good_till_cancel", description="Time in force policy")
    good_till: Optional[datetime] = None
    priority: int = Field(0, description="Quote priority (lower = better)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_quote_kwargs(self) -> Dict[str, Any]:
        """Convert to :class:`Quote` arguments."""

        tif = TimeInForce(self.time_in_force)

        return {
            "quoter": self.quoter.to_party(),
            "quoter_member_id": self.quoter_member_id,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "yield_value": self.yield_value,
            "spread": self.spread,
            "min_quantity": self.min_quantity,
            "time_in_force": tif,
            "good_till": self.good_till,
            "priority": self.priority,
            "metadata": self.metadata,
        }


class RFQStatusUpdateRequest(BaseModel):
    """Payload for manual RFQ status updates."""

    status: RFQStatus
    reason: Optional[str] = Field(None, description="Reason for status change")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RFQStatusEvent(BaseModel):
    """Status transition entry for RFQ history."""

    status: RFQStatus
    timestamp: datetime
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuoteSummary(BaseModel):
    """Summary of a submitted quote."""

    quote_id: str
    rfq_id: str
    status: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal]
    priority: int
    submitted_at: datetime

    @classmethod
    def from_quote(cls, quote: Quote) -> "QuoteSummary":
        return cls(
            quote_id=quote.quote_id,
            rfq_id=quote.rfq_id,
            status=quote.status.value,
            side=quote.side,
            quantity=quote.quantity,
            price=quote.price,
            priority=quote.priority,
            submitted_at=quote.submission_time,
        )


class RFQSummary(BaseModel):
    """Summary view for an RFQ including status history."""

    rfq_id: str
    status: RFQStatus
    auction_type: AuctionType
    side: OrderSide
    quote_deadline: datetime
    expiry_time: datetime
    quotes_received: int
    execution_price: Optional[Decimal]
    best_quote_id: Optional[str]
    status_history: List[RFQStatusEvent]

    @classmethod
    def from_rfq(cls, rfq: RFQ) -> "RFQSummary":
        history = [
            RFQStatusEvent(
                status=entry["status"] if isinstance(entry["status"], RFQStatus) else RFQStatus(entry["status"]),
                timestamp=entry["timestamp"],
                reason=entry.get("reason"),
                metadata=entry.get("metadata", {}),
            )
            for entry in rfq.status_history
        ]

        return cls(
            rfq_id=rfq.rfq_id,
            status=rfq.status,
            auction_type=rfq.auction_type,
            side=rfq.side,
            quote_deadline=rfq.quote_deadline,
            expiry_time=rfq.expiry_time,
            quotes_received=rfq.quotes_received,
            execution_price=rfq.execution_price,
            best_quote_id=rfq.best_quote_id,
            status_history=history,
        )


class AuctionResultPayload(BaseModel):
    """Serializable representation of :class:`AuctionResult`."""

    result_id: str
    rfq_id: str
    auction_type: AuctionType
    clearing_price: Optional[Decimal]
    total_quantity_filled: Decimal
    num_participants: int
    winning_quotes: List[str]
    allocations: List[Dict[str, Any]]
    price_improvement: Optional[Decimal] = None
    spread_tightness: Optional[Decimal] = None
    competition_score: float
    execution_time: datetime
    execution_duration_ms: float
    metadata: Dict[str, Any]

    @classmethod
    def from_result(cls, result: AuctionResult) -> "AuctionResultPayload":
        data = result.model_dump(mode="json")
        return cls(**data)


__all__ = [
    "VanillaOptionRequest",
    "ProfileRequest",
    "CVARequest",
    "FVARequest",
    "MVARequest",
    "PortfolioXVARequest",
    "PortfolioSummaryRequest",
    "FpMLPriceRequest",
    "FpMLParseRequest",
    "FpMLValidateRequest",
    "PartyPayload",
    "RFQSpecificationPayload",
    "RFQCreateRequest",
    "QuoteSubmissionRequest",
    "RFQStatusUpdateRequest",
    "RFQStatusEvent",
    "RFQSummary",
    "QuoteSummary",
    "AuctionResultPayload",
]
