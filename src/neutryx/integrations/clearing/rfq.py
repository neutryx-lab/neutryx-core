"""RFQ (Request for Quote) workflow and auction mechanisms for clearing.

This module implements the RFQ lifecycle for cleared products:
1. RFQ creation and submission
2. Quote collection from market makers
3. Auction execution (single-price, multi-price, continuous)
4. Best execution analysis
5. Trade allocation and novation

Supports various auction protocols:
- Single-price sealed-bid (uniform price)
- Multi-price sealed-bid (discriminatory)
- Continuous double auction (order book)
- Dutch auction (descending price)
- Vickrey auction (second-price sealed-bid)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base import Party, ProductType, Trade, TradeEconomics


class RFQStatus(str, Enum):
    """RFQ lifecycle status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    OPEN = "open"
    QUOTING = "quoting"
    CLOSED = "closed"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class QuoteStatus(str, Enum):
    """Quote status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class AuctionType(str, Enum):
    """Auction mechanism type."""

    SINGLE_PRICE = "single_price"  # Uniform price clearing
    MULTI_PRICE = "multi_price"  # Discriminatory pricing
    CONTINUOUS = "continuous"  # Order book matching
    DUTCH = "dutch"  # Descending price
    VICKREY = "vickrey"  # Second-price sealed-bid
    SECOND_PRICE = "second_price"  # Multi-unit second price
    MULTI_ROUND = "multi_round"  # Sequential rounds with price improvement


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Time in force for quotes."""
    GTC = "good_till_cancel"
    GTD = "good_till_date"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


class RFQSpecification(BaseModel):
    """RFQ product specification."""

    product_type: ProductType = Field(..., description="Product type")
    notional: Decimal = Field(..., description="Notional amount")
    currency: str = Field(..., description="Currency")
    effective_date: datetime = Field(..., description="Effective date")
    maturity_date: datetime = Field(..., description="Maturity date")

    # Optional product-specific fields
    fixed_rate: Optional[Decimal] = Field(None, description="Fixed rate for swaps")
    strike: Optional[Decimal] = Field(None, description="Strike for options")
    underlying: Optional[str] = Field(None, description="Underlying asset")
    tenor: Optional[str] = Field(None, description="Tenor (e.g., 5Y)")

    # Additional terms
    settlement_type: str = Field(default="physical", description="Settlement type")
    day_count: str = Field(default="ACT/360", description="Day count convention")
    payment_frequency: Optional[str] = Field(None, description="Payment frequency")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('notional')
    @classmethod
    def validate_notional(cls, v):
        if v <= 0:
            raise ValueError("Notional must be positive")
        return v


class RFQ(BaseModel):
    """Request for Quote."""

    rfq_id: str = Field(default_factory=lambda: f"RFQ-{uuid4().hex[:12].upper()}")
    status: RFQStatus = Field(default=RFQStatus.DRAFT)
    status_history: List[Dict[str, Any]] = Field(default_factory=list, description="Lifecycle state transitions")

    # Requester information
    requester: Party = Field(..., description="Party requesting quote")
    clearing_broker: Optional[str] = Field(None, description="Clearing broker")

    # Product specification
    specification: RFQSpecification = Field(..., description="Product specification")
    side: OrderSide = Field(..., description="Buy or sell")

    # Auction parameters
    auction_type: AuctionType = Field(default=AuctionType.SINGLE_PRICE)
    max_participants: Optional[int] = Field(None, description="Max number of quote providers")
    invited_dealers: Optional[List[str]] = Field(None, description="Invited dealer IDs")

    # Timing
    submission_time: datetime = Field(default_factory=datetime.utcnow)
    quote_deadline: datetime = Field(..., description="Quote submission deadline")
    expiry_time: datetime = Field(..., description="RFQ expiry time")

    # Execution preferences
    min_quote_size: Optional[Decimal] = Field(None, description="Minimum quote size")
    allow_partial_fills: bool = Field(default=False, description="Allow partial fills")
    require_all_or_none: bool = Field(default=False, description="All-or-none requirement")

    # Results
    quotes_received: int = Field(default=0)
    best_quote_id: Optional[str] = Field(None)
    execution_price: Optional[Decimal] = Field(None)
    execution_time: Optional[datetime] = Field(None)
    allocated_trades: List[str] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def record_status(
        self,
        status: RFQStatus,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a status transition to the history and update current status."""

        transition_entry = {
            "status": status,
            "timestamp": datetime.utcnow(),
        }
        if reason:
            transition_entry["reason"] = reason
        if metadata:
            transition_entry["metadata"] = metadata

        self.status_history.append(transition_entry)
        self.status = status

    @field_validator('quote_deadline', 'expiry_time')
    @classmethod
    def validate_times(cls, v, info):
        if 'submission_time' in info.data:
            if v <= info.data['submission_time']:
                raise ValueError(f"{info.field_name} must be after submission_time")
        return v


class Quote(BaseModel):
    """Quote submission from market maker."""

    quote_id: str = Field(default_factory=lambda: f"QTE-{uuid4().hex[:12].upper()}")
    rfq_id: str = Field(..., description="Reference RFQ ID")
    status: QuoteStatus = Field(default=QuoteStatus.PENDING)

    # Quoter information
    quoter: Party = Field(..., description="Party providing quote")
    quoter_member_id: str = Field(..., description="CCP member ID")

    # Quote terms
    side: OrderSide = Field(..., description="Buy or sell")
    quantity: Decimal = Field(..., description="Quoted quantity")
    price: Optional[Decimal] = Field(None, description="Price (rate, spread, etc.)")
    yield_value: Optional[Decimal] = Field(None, description="Yield")
    spread: Optional[Decimal] = Field(None, description="Spread over benchmark")

    # Constraints
    min_quantity: Optional[Decimal] = Field(None, description="Minimum fill quantity")
    time_in_force: TimeInForce = Field(default=TimeInForce.GTC)
    good_till: Optional[datetime] = Field(None, description="Expiry for GTD orders")

    # Metadata
    submission_time: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=0, description="Quote priority (lower = higher priority)")
    quote_quality_score: Optional[float] = Field(None, description="Quality score 0-100")

    # Execution
    filled_quantity: Decimal = Field(default=Decimal("0"))
    average_fill_price: Optional[Decimal] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('quantity', 'min_quantity')
    @classmethod
    def validate_quantities(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    def is_expired(self) -> bool:
        """Check if quote has expired."""
        if self.time_in_force == TimeInForce.GTD and self.good_till:
            return datetime.utcnow() > self.good_till
        return False

    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity."""
        return self.quantity - self.filled_quantity


class AuctionResult(BaseModel):
    """Auction execution result."""

    result_id: str = Field(default_factory=lambda: f"AUC-{uuid4().hex[:12].upper()}")
    rfq_id: str = Field(..., description="Reference RFQ ID")
    auction_type: AuctionType = Field(..., description="Auction type used")

    # Execution results
    clearing_price: Optional[Decimal] = Field(None, description="Clearing price for single-price")
    total_quantity_filled: Decimal = Field(..., description="Total quantity executed")
    num_participants: int = Field(..., description="Number of participants")

    # Allocations
    winning_quotes: List[str] = Field(default_factory=list, description="Winning quote IDs")
    allocations: List[Dict[str, Any]] = Field(default_factory=list, description="Quantity allocations")

    # Metrics
    price_improvement: Optional[Decimal] = Field(None, description="vs reference price")
    spread_tightness: Optional[Decimal] = Field(None, description="Bid-ask spread")
    competition_score: float = Field(..., description="Competition metric 0-100")

    # Timing
    execution_time: datetime = Field(default_factory=datetime.utcnow)
    execution_duration_ms: float = Field(..., description="Auction duration in ms")

    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class OrderBookLevel:
    """Order book price level."""
    price: Decimal
    quantity: Decimal
    num_orders: int
    quote_ids: List[str] = field(default_factory=list)


class OrderBook:
    """Order book for continuous auction."""

    def __init__(self, rfq_id: str):
        self.rfq_id = rfq_id
        self.bids: Dict[Decimal, OrderBookLevel] = {}  # Buy orders
        self.offers: Dict[Decimal, OrderBookLevel] = {}  # Sell orders
        self.last_update: datetime = datetime.utcnow()

    def add_quote(self, quote: Quote) -> None:
        """Add quote to order book."""
        if quote.price is None:
            return

        book = self.bids if quote.side == OrderSide.BUY else self.offers

        if quote.price in book:
            level = book[quote.price]
            level.quantity += quote.quantity
            level.num_orders += 1
            level.quote_ids.append(quote.quote_id)
        else:
            book[quote.price] = OrderBookLevel(
                price=quote.price,
                quantity=quote.quantity,
                num_orders=1,
                quote_ids=[quote.quote_id]
            )

        self.last_update = datetime.utcnow()

    def remove_quote(self, quote: Quote) -> None:
        """Remove quote from order book."""
        if quote.price is None:
            return

        book = self.bids if quote.side == OrderSide.BUY else self.offers

        if quote.price in book:
            level = book[quote.price]
            level.quantity -= quote.quantity
            level.num_orders -= 1
            if quote.quote_id in level.quote_ids:
                level.quote_ids.remove(quote.quote_id)

            if level.quantity <= 0 or level.num_orders == 0:
                del book[quote.price]

        self.last_update = datetime.utcnow()

    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        if not self.bids:
            return None
        return max(self.bids.keys())

    def best_offer(self) -> Optional[Decimal]:
        """Get best offer price."""
        if not self.offers:
            return None
        return min(self.offers.keys())

    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        bid = self.best_bid()
        offer = self.best_offer()
        if bid is not None and offer is not None:
            return offer - bid
        return None

    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        bid = self.best_bid()
        offer = self.best_offer()
        if bid is not None and offer is not None:
            return (bid + offer) / Decimal("2")
        return None

    def depth(self, side: OrderSide, num_levels: int = 5) -> List[OrderBookLevel]:
        """Get order book depth."""
        book = self.bids if side == OrderSide.BUY else self.offers

        if side == OrderSide.BUY:
            # Sort bids descending (best first)
            sorted_prices = sorted(book.keys(), reverse=True)
        else:
            # Sort offers ascending (best first)
            sorted_prices = sorted(book.keys())

        return [book[p] for p in sorted_prices[:num_levels]]


class AuctionEngine(ABC):
    """Abstract auction execution engine."""

    def __init__(self, rfq: RFQ):
        self.rfq = rfq
        self.quotes: List[Quote] = []
        self.result: Optional[AuctionResult] = None

    def add_quote(self, quote: Quote) -> None:
        """Add quote to auction."""
        if quote.rfq_id != self.rfq.rfq_id:
            raise ValueError(f"Quote RFQ ID {quote.rfq_id} does not match {self.rfq.rfq_id}")

        # Validate quote hasn't expired
        if quote.is_expired():
            quote.status = QuoteStatus.EXPIRED
            return

        self.quotes.append(quote)

    @abstractmethod
    def execute(self) -> AuctionResult:
        """Execute auction and determine winners."""
        pass

    def _validate_execution(self) -> None:
        """Validate auction can be executed."""
        if not self.quotes:
            raise ValueError("No quotes to execute")

        if datetime.utcnow() < self.rfq.quote_deadline:
            raise ValueError("Quote deadline has not passed")

    def _calculate_competition_score(self) -> float:
        """Calculate competition metric (0-100)."""
        if len(self.quotes) <= 1:
            return 0.0

        # Score based on number of participants and price dispersion
        participant_score = min(len(self.quotes) / 10.0, 1.0) * 50

        # Price dispersion (standard deviation of prices)
        prices = [float(q.price) for q in self.quotes if q.price is not None]
        if len(prices) > 1:
            import statistics
            price_std = statistics.stdev(prices)
            price_mean = statistics.mean(prices)
            cv = price_std / price_mean if price_mean != 0 else 0
            dispersion_score = min(cv * 100, 50)
        else:
            dispersion_score = 0.0

        return participant_score + dispersion_score


class SinglePriceAuction(AuctionEngine):
    """Single-price sealed-bid auction (uniform price clearing).

    All winning bidders pay the same clearing price, which is typically:
    - For buy side: highest losing bid price (or lowest winning offer)
    - For sell side: lowest losing offer (or highest winning bid)

    Also known as uniform price auction or competitive auction.
    """

    def execute(self) -> AuctionResult:
        """Execute single-price auction."""
        self._validate_execution()
        start_time = datetime.utcnow()

        # Sort quotes by price (best first)
        if self.rfq.side == OrderSide.BUY:
            # For buy orders, best = lowest offer
            sorted_quotes = sorted(
                [q for q in self.quotes if q.price is not None],
                key=lambda q: (q.price, q.priority)
            )
        else:
            # For sell orders, best = highest bid
            sorted_quotes = sorted(
                [q for q in self.quotes if q.price is not None],
                key=lambda q: (-q.price, q.priority)
            )

        # Allocate quantity to quotes until filled
        target_quantity = self.rfq.specification.notional
        remaining = target_quantity
        winning_quotes = []
        allocations = []
        clearing_price = None

        for quote in sorted_quotes:
            if remaining <= 0:
                break

            fill_qty = min(quote.quantity, remaining)

            allocations.append({
                "quote_id": quote.quote_id,
                "quoter_id": quote.quoter.party_id,
                "quantity": float(fill_qty),
                "price": float(quote.price)
            })

            quote.filled_quantity = fill_qty
            quote.status = QuoteStatus.ACCEPTED
            winning_quotes.append(quote.quote_id)

            remaining -= fill_qty
            clearing_price = quote.price

        # Mark rejected quotes
        for quote in sorted_quotes:
            if quote.quote_id not in winning_quotes:
                quote.status = QuoteStatus.REJECTED

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        self.result = AuctionResult(
            rfq_id=self.rfq.rfq_id,
            auction_type=AuctionType.SINGLE_PRICE,
            clearing_price=clearing_price,
            total_quantity_filled=target_quantity - remaining,
            num_participants=len(self.quotes),
            winning_quotes=winning_quotes,
            allocations=allocations,
            competition_score=self._calculate_competition_score(),
            execution_duration_ms=duration_ms
        )

        return self.result


class MultiPriceAuction(AuctionEngine):
    """Multi-price sealed-bid auction (discriminatory pricing).

    Winners pay their bid price (discriminatory pricing).
    Allocates to best quotes first until order is filled.
    """

    def execute(self) -> AuctionResult:
        """Execute multi-price auction."""
        self._validate_execution()
        start_time = datetime.utcnow()

        # Sort quotes by price (best first)
        if self.rfq.side == OrderSide.BUY:
            sorted_quotes = sorted(
                [q for q in self.quotes if q.price is not None],
                key=lambda q: (q.price, q.priority)
            )
        else:
            sorted_quotes = sorted(
                [q for q in self.quotes if q.price is not None],
                key=lambda q: (-q.price, q.priority)
            )

        # Allocate
        target_quantity = self.rfq.specification.notional
        remaining = target_quantity
        winning_quotes = []
        allocations = []

        for quote in sorted_quotes:
            if remaining <= 0:
                break

            fill_qty = min(quote.quantity, remaining)

            allocations.append({
                "quote_id": quote.quote_id,
                "quoter_id": quote.quoter.party_id,
                "quantity": float(fill_qty),
                "price": float(quote.price)  # Each pays their bid
            })

            quote.filled_quantity = fill_qty
            quote.average_fill_price = quote.price
            quote.status = QuoteStatus.ACCEPTED
            winning_quotes.append(quote.quote_id)

            remaining -= fill_qty

        # Mark rejected
        for quote in sorted_quotes:
            if quote.quote_id not in winning_quotes:
                quote.status = QuoteStatus.REJECTED

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        self.result = AuctionResult(
            rfq_id=self.rfq.rfq_id,
            auction_type=AuctionType.MULTI_PRICE,
            clearing_price=None,  # No single clearing price
            total_quantity_filled=target_quantity - remaining,
            num_participants=len(self.quotes),
            winning_quotes=winning_quotes,
            allocations=allocations,
            competition_score=self._calculate_competition_score(),
            execution_duration_ms=duration_ms
        )

        return self.result


class DutchAuction(AuctionEngine):
    """Descending price auction execution.

    Starts from the highest quoted price and descends until the requested
    quantity is filled. Records the price path and allocates at the clearing
    price once sufficient liquidity is available.
    """

    def execute(self) -> AuctionResult:
        """Execute Dutch auction."""
        self._validate_execution()
        start_time = datetime.utcnow()

        priced_quotes = [q for q in self.quotes if q.price is not None]
        if not priced_quotes:
            raise ValueError("No priced quotes available for Dutch auction")

        # Determine ordering and price levels
        if self.rfq.side == OrderSide.BUY:
            sorted_quotes = sorted(
                priced_quotes,
                key=lambda q: (q.price, q.priority)
            )
            is_active = lambda quote, level: quote.price <= level
        else:
            sorted_quotes = sorted(
                priced_quotes,
                key=lambda q: (-q.price, q.priority)
            )
            is_active = lambda quote, level: quote.price >= level

        price_levels = sorted({q.price for q in priced_quotes}, reverse=True)

        target_quantity = self.rfq.specification.notional
        remaining = target_quantity
        winning_quotes: List[str] = []
        allocations: List[Dict[str, Any]] = []
        clearing_price: Optional[Decimal] = None
        price_path: List[Dict[str, Any]] = []

        for step, level in enumerate(price_levels):
            price_path.append({"step": step, "price": float(level)})

            eligible_quotes = [q for q in sorted_quotes if is_active(q, level)]
            cumulative = sum((q.quantity for q in eligible_quotes), Decimal("0"))

            if cumulative >= target_quantity:
                clearing_price = level
                remaining = target_quantity
                for quote in eligible_quotes:
                    if remaining <= 0:
                        break
                    fill_qty = min(quote.quantity, remaining)
                    if fill_qty <= 0:
                        continue

                    allocations.append({
                        "quote_id": quote.quote_id,
                        "quoter_id": quote.quoter.party_id,
                        "quantity": float(fill_qty),
                        "price": float(level)
                    })
                    quote.filled_quantity = fill_qty
                    quote.average_fill_price = level
                    quote.status = QuoteStatus.ACCEPTED
                    if quote.quote_id not in winning_quotes:
                        winning_quotes.append(quote.quote_id)
                    remaining -= fill_qty
                break

        if clearing_price is None:
            # Not enough liquidity; allocate what is available at final level
            final_level = price_levels[-1]
            clearing_price = final_level
            eligible_quotes = [q for q in sorted_quotes if is_active(q, final_level)]
            remaining = target_quantity

            for quote in eligible_quotes:
                if remaining <= 0:
                    break
                fill_qty = min(quote.quantity, remaining)
                if fill_qty <= 0:
                    continue

                allocations.append({
                    "quote_id": quote.quote_id,
                    "quoter_id": quote.quoter.party_id,
                    "quantity": float(fill_qty),
                    "price": float(final_level)
                })
                quote.filled_quantity = fill_qty
                quote.average_fill_price = final_level
                quote.status = QuoteStatus.ACCEPTED
                if quote.quote_id not in winning_quotes:
                    winning_quotes.append(quote.quote_id)
                remaining -= fill_qty

        # Mark non-winning quotes as rejected
        for quote in priced_quotes:
            if quote.quote_id not in winning_quotes:
                quote.status = QuoteStatus.REJECTED

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        self.result = AuctionResult(
            rfq_id=self.rfq.rfq_id,
            auction_type=AuctionType.DUTCH,
            clearing_price=clearing_price,
            total_quantity_filled=target_quantity - remaining,
            num_participants=len(self.quotes),
            winning_quotes=winning_quotes,
            allocations=allocations,
            competition_score=self._calculate_competition_score(),
            execution_duration_ms=duration_ms,
            metadata={
                "price_path": price_path,
                "start_price": float(price_levels[0]) if price_levels else None
            }
        )

        return self.result


class VickreyAuction(AuctionEngine):
    """Second-price sealed-bid auction implementation."""

    def execute(self) -> AuctionResult:
        """Execute Vickrey auction."""
        self._validate_execution()
        start_time = datetime.utcnow()

        priced_quotes = [q for q in self.quotes if q.price is not None]
        if not priced_quotes:
            raise ValueError("No priced quotes available for Vickrey auction")

        if self.rfq.side == OrderSide.BUY:
            sorted_quotes = sorted(
                priced_quotes,
                key=lambda q: (q.price, q.priority)
            )
        else:
            sorted_quotes = sorted(
                priced_quotes,
                key=lambda q: (-q.price, q.priority)
            )

        target_quantity = self.rfq.specification.notional
        remaining = target_quantity
        winning_quotes: List[str] = []
        allocations: List[Dict[str, Any]] = []

        for quote in sorted_quotes:
            if remaining <= 0:
                break
            fill_qty = min(quote.quantity, remaining)
            if fill_qty <= 0:
                continue

            allocations.append({
                "quote_id": quote.quote_id,
                "quoter_id": quote.quoter.party_id,
                "quantity": float(fill_qty)
            })
            quote.filled_quantity = fill_qty
            quote.status = QuoteStatus.ACCEPTED
            winning_quotes.append(quote.quote_id)
            remaining -= fill_qty

        if not winning_quotes:
            raise ValueError("No winning quotes identified in Vickrey auction")

        losing_quotes = [q for q in sorted_quotes if q.quote_id not in winning_quotes]
        if losing_quotes:
            reference_quote = losing_quotes[0]
            clearing_price = reference_quote.price
            reference_quote_id = reference_quote.quote_id
        else:
            # Fallback to last winning quote price when no losing quotes exist
            winning_quote_objs = [q for q in sorted_quotes if q.quote_id in winning_quotes]
            clearing_price = winning_quote_objs[-1].price
            reference_quote_id = None

        # Update allocation prices and winner fill prices
        for allocation in allocations:
            allocation["price"] = float(clearing_price)

        for quote in priced_quotes:
            if quote.quote_id in winning_quotes:
                quote.average_fill_price = clearing_price
            else:
                quote.status = QuoteStatus.REJECTED

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        self.result = AuctionResult(
            rfq_id=self.rfq.rfq_id,
            auction_type=AuctionType.VICKREY,
            clearing_price=clearing_price,
            total_quantity_filled=target_quantity - remaining,
            num_participants=len(self.quotes),
            winning_quotes=winning_quotes,
            allocations=allocations,
            competition_score=self._calculate_competition_score(),
            execution_duration_ms=duration_ms,
            metadata={
                "second_price_quote_id": reference_quote_id
            }
        )

        return self.result


class SecondPriceAuction(AuctionEngine):
    """Multi-unit second price auction.

    Winning quotes receive allocations at the price of the best losing quote
    (or the worst winning price when no losing quote is available).
    """

    def execute(self) -> AuctionResult:
        """Execute the second price auction."""

        self._validate_execution()
        start_time = datetime.utcnow()

        priced_quotes = [q for q in self.quotes if q.price is not None]
        if not priced_quotes:
            raise ValueError("No priced quotes available for second price auction")

        if self.rfq.side == OrderSide.BUY:
            sorted_quotes = sorted(
                priced_quotes,
                key=lambda q: (q.price, q.priority),
            )
        else:
            sorted_quotes = sorted(
                priced_quotes,
                key=lambda q: (-q.price, q.priority),
            )

        target_quantity = self.rfq.specification.notional
        remaining = target_quantity
        winning_quotes: List[str] = []
        allocations: List[Dict[str, Any]] = []

        for quote in sorted_quotes:
            if remaining <= 0:
                break
            fill_qty = min(quote.quantity, remaining)
            if fill_qty <= 0:
                continue

            allocations.append(
                {
                    "quote_id": quote.quote_id,
                    "quoter_id": quote.quoter.party_id,
                    "quantity": float(fill_qty),
                }
            )
            quote.filled_quantity = fill_qty
            quote.status = QuoteStatus.ACCEPTED
            winning_quotes.append(quote.quote_id)
            remaining -= fill_qty

        if not winning_quotes:
            raise ValueError("No winning quotes identified in second price auction")

        losing_quotes = [q for q in sorted_quotes if q.quote_id not in winning_quotes]
        if losing_quotes:
            reference_quote = losing_quotes[0]
            clearing_price = reference_quote.price
            reference_quote_id = reference_quote.quote_id
        else:
            winning_quote_objs = [q for q in sorted_quotes if q.quote_id in winning_quotes]
            clearing_price = winning_quote_objs[-1].price
            reference_quote_id = None

        for allocation in allocations:
            allocation["price"] = float(clearing_price)

        for quote in priced_quotes:
            if quote.quote_id in winning_quotes:
                quote.average_fill_price = clearing_price
            else:
                quote.status = QuoteStatus.REJECTED

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        self.result = AuctionResult(
            rfq_id=self.rfq.rfq_id,
            auction_type=AuctionType.SECOND_PRICE,
            clearing_price=clearing_price,
            total_quantity_filled=target_quantity - remaining,
            num_participants=len(self.quotes),
            winning_quotes=winning_quotes,
            allocations=allocations,
            competition_score=self._calculate_competition_score(),
            execution_duration_ms=duration_ms,
            metadata={
                "second_price_quote_id": reference_quote_id
            },
        )

        return self.result


class MultiRoundAuction(AuctionEngine):
    """Auction executed over multiple sequential rounds.

    Quotes should include a ``metadata["round"]`` value to indicate the
    submission round. Rounds are processed in ascending order and allocations
    occur per round using price-time priority within each round.
    """

    def execute(self) -> AuctionResult:
        """Execute a multi-round auction."""

        self._validate_execution()
        start_time = datetime.utcnow()

        priced_quotes = [q for q in self.quotes if q.price is not None]
        if not priced_quotes:
            raise ValueError("No priced quotes available for multi-round auction")

        target_quantity = self.rfq.specification.notional
        remaining = target_quantity
        winning_quotes: List[str] = []
        allocations: List[Dict[str, Any]] = []
        round_details: List[Dict[str, Any]] = []
        final_clearing_price: Optional[Decimal] = None

        rounds = sorted({int(q.metadata.get("round", 1)) for q in priced_quotes})

        for round_number in rounds:
            round_quotes = [q for q in priced_quotes if int(q.metadata.get("round", 1)) == round_number]
            if not round_quotes:
                continue

            if self.rfq.side == OrderSide.BUY:
                sorted_round = sorted(
                    round_quotes,
                    key=lambda q: (q.price, q.priority),
                )
            else:
                sorted_round = sorted(
                    round_quotes,
                    key=lambda q: (-q.price, q.priority),
                )

            round_allocations: List[Dict[str, Any]] = []
            clearing_price: Optional[Decimal] = None

            for quote in sorted_round:
                if remaining <= 0:
                    break

                fill_qty = min(quote.quantity, remaining)
                if fill_qty <= 0:
                    continue

                allocation = {
                    "quote_id": quote.quote_id,
                    "quoter_id": quote.quoter.party_id,
                    "quantity": float(fill_qty),
                    "price": float(quote.price),
                    "round": round_number,
                }
                round_allocations.append(allocation)
                allocations.append(allocation)
                quote.filled_quantity = fill_qty
                quote.average_fill_price = quote.price
                quote.status = QuoteStatus.ACCEPTED
                if quote.quote_id not in winning_quotes:
                    winning_quotes.append(quote.quote_id)
                remaining -= fill_qty
                clearing_price = quote.price
                final_clearing_price = quote.price

            # Mark non-winning quotes in this round as rejected when auction completes
            for quote in sorted_round:
                if quote.quote_id not in winning_quotes:
                    quote.status = QuoteStatus.REJECTED

            round_details.append(
                {
                    "round": round_number,
                    "allocations": round_allocations,
                    "clearing_price": float(clearing_price) if clearing_price is not None else None,
                }
            )

            if remaining <= 0:
                break

        for quote in priced_quotes:
            if quote.quote_id not in winning_quotes:
                quote.status = QuoteStatus.REJECTED

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        self.result = AuctionResult(
            rfq_id=self.rfq.rfq_id,
            auction_type=AuctionType.MULTI_ROUND,
            clearing_price=final_clearing_price,
            total_quantity_filled=target_quantity - remaining,
            num_participants=len(self.quotes),
            winning_quotes=winning_quotes,
            allocations=allocations,
            competition_score=self._calculate_competition_score(),
            execution_duration_ms=duration_ms,
            metadata={
                "rounds": round_details,
            },
        )

        return self.result


class ContinuousAuction(AuctionEngine):
    """Continuous double auction with order book matching.

    Quotes are matched continuously as they arrive.
    Implements price-time priority matching.
    """

    def __init__(self, rfq: RFQ):
        super().__init__(rfq)
        self.order_book = OrderBook(rfq.rfq_id)
        self.trades: List[Dict[str, Any]] = []

    def add_quote(self, quote: Quote) -> None:
        """Add quote and attempt matching."""
        super().add_quote(quote)
        self.order_book.add_quote(quote)
        self._try_match()

    def _try_match(self) -> None:
        """Try to match orders in book."""
        while True:
            best_bid = self.order_book.best_bid()
            best_offer = self.order_book.best_offer()

            if best_bid is None or best_offer is None:
                break

            # Check if crossing
            if best_bid >= best_offer:
                # Execute trade at offer price (price-time priority)
                self._execute_match(best_bid, best_offer)
            else:
                break

    def _execute_match(self, bid_price: Decimal, offer_price: Decimal) -> None:
        """Execute a match between bid and offer."""
        # Simplified matching - in production would need full order matching
        execution_price = offer_price  # Price-time priority favors passive side

        bid_level = self.order_book.bids.get(bid_price)
        offer_level = self.order_book.offers.get(offer_price)

        if bid_level and offer_level:
            match_qty = min(bid_level.quantity, offer_level.quantity)

            self.trades.append({
                "price": float(execution_price),
                "quantity": float(match_qty),
                "bid_quotes": bid_level.quote_ids[:],
                "offer_quotes": offer_level.quote_ids[:],
                "timestamp": datetime.utcnow().isoformat()
            })

            # Update book
            bid_level.quantity -= match_qty
            offer_level.quantity -= match_qty

            if bid_level.quantity <= 0:
                del self.order_book.bids[bid_price]
            if offer_level.quantity <= 0:
                del self.order_book.offers[offer_price]

    def execute(self) -> AuctionResult:
        """Finalize continuous auction."""
        self._validate_execution()
        start_time = datetime.utcnow()

        total_filled = sum(Decimal(str(t["quantity"])) for t in self.trades)

        # Volume-weighted average price
        if self.trades:
            vwap = sum(
                Decimal(str(t["price"])) * Decimal(str(t["quantity"]))
                for t in self.trades
            ) / total_filled if total_filled > 0 else Decimal("0")
        else:
            vwap = None

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Determine winning quotes from trades
        winning_quotes = set()
        for trade in self.trades:
            winning_quotes.update(trade.get("bid_quotes", []))
            winning_quotes.update(trade.get("offer_quotes", []))

        self.result = AuctionResult(
            rfq_id=self.rfq.rfq_id,
            auction_type=AuctionType.CONTINUOUS,
            clearing_price=vwap,
            total_quantity_filled=total_filled,
            num_participants=len(self.quotes),
            winning_quotes=list(winning_quotes),
            allocations=self.trades,
            competition_score=self._calculate_competition_score(),
            execution_duration_ms=duration_ms,
            metadata={
                "final_spread": float(self.order_book.spread()) if self.order_book.spread() else None,
                "num_trades": len(self.trades)
            }
        )

        return self.result


class RFQManager:
    """RFQ lifecycle manager."""

    _ALLOWED_TRANSITIONS: Dict[RFQStatus, set[RFQStatus]] = {
        RFQStatus.DRAFT: {RFQStatus.SUBMITTED, RFQStatus.CANCELLED},
        RFQStatus.SUBMITTED: {RFQStatus.OPEN, RFQStatus.CANCELLED},
        RFQStatus.OPEN: {RFQStatus.QUOTING, RFQStatus.CLOSED, RFQStatus.CANCELLED, RFQStatus.EXPIRED},
        RFQStatus.QUOTING: {RFQStatus.CLOSED, RFQStatus.EXECUTED, RFQStatus.CANCELLED, RFQStatus.EXPIRED},
        RFQStatus.CLOSED: {RFQStatus.EXECUTED, RFQStatus.FAILED, RFQStatus.CANCELLED},
        RFQStatus.EXECUTED: set(),
        RFQStatus.CANCELLED: set(),
        RFQStatus.EXPIRED: set(),
        RFQStatus.FAILED: set(),
    }

    def __init__(self):
        self.rfqs: Dict[str, RFQ] = {}
        self.quotes: Dict[str, List[Quote]] = {}
        self.auctions: Dict[str, AuctionEngine] = {}
        self.results: Dict[str, AuctionResult] = {}
        self._state_listeners: Dict[str, List[asyncio.Queue]] = {}

    def create_rfq(
        self,
        requester: Party,
        specification: RFQSpecification,
        side: OrderSide,
        quote_deadline: datetime,
        auction_type: AuctionType = AuctionType.SINGLE_PRICE,
        **kwargs
    ) -> RFQ:
        """Create new RFQ."""
        # Set expiry if not provided
        expiry = kwargs.pop('expiry_time', quote_deadline + timedelta(hours=1))

        rfq = RFQ(
            requester=requester,
            specification=specification,
            side=side,
            quote_deadline=quote_deadline,
            expiry_time=expiry,
            auction_type=auction_type,
            **kwargs
        )

        self.rfqs[rfq.rfq_id] = rfq
        self.quotes[rfq.rfq_id] = []

        rfq.record_status(RFQStatus.DRAFT, reason="RFQ created")
        self._notify_state(rfq, reason="RFQ created")

        return rfq

    def submit_rfq(self, rfq_id: str) -> RFQ:
        """Submit RFQ to market."""
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        if rfq.status != RFQStatus.DRAFT:
            raise ValueError(f"RFQ {rfq_id} already submitted")

        self._transition(rfq, RFQStatus.SUBMITTED, reason="RFQ submitted")
        self._transition(rfq, RFQStatus.OPEN, reason="RFQ opened for quoting")
        return rfq

    def submit_quote(self, rfq_id: str, quote: Quote) -> Quote:
        """Submit quote for RFQ."""
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        if rfq.status not in [RFQStatus.OPEN, RFQStatus.QUOTING]:
            raise ValueError(f"RFQ {rfq_id} not accepting quotes")

        if datetime.utcnow() > rfq.quote_deadline:
            raise ValueError(f"Quote deadline passed for RFQ {rfq_id}")

        quote.rfq_id = rfq_id
        quote.status = QuoteStatus.SUBMITTED
        self.quotes[rfq_id].append(quote)
        rfq.quotes_received += 1
        if rfq.status == RFQStatus.OPEN:
            self._transition(rfq, RFQStatus.QUOTING, reason="Quotes received")
        elif rfq.status != RFQStatus.QUOTING:
            raise ValueError(f"RFQ {rfq_id} cannot accept quotes in status {rfq.status}")
        else:
            # Notify listeners for additional quote submissions
            self._notify_state(rfq, reason="Quote received")

        return quote

    def execute_auction(self, rfq_id: str) -> AuctionResult:
        """Execute auction for RFQ."""
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        quotes = self.quotes.get(rfq_id, [])

        # Create appropriate auction engine
        if rfq.auction_type == AuctionType.SINGLE_PRICE:
            engine = SinglePriceAuction(rfq)
        elif rfq.auction_type == AuctionType.MULTI_PRICE:
            engine = MultiPriceAuction(rfq)
        elif rfq.auction_type == AuctionType.CONTINUOUS:
            engine = ContinuousAuction(rfq)
        elif rfq.auction_type == AuctionType.DUTCH:
            engine = DutchAuction(rfq)
        elif rfq.auction_type == AuctionType.VICKREY:
            engine = VickreyAuction(rfq)
        elif rfq.auction_type == AuctionType.SECOND_PRICE:
            engine = SecondPriceAuction(rfq)
        elif rfq.auction_type == AuctionType.MULTI_ROUND:
            engine = MultiRoundAuction(rfq)
        else:
            raise NotImplementedError(f"Auction type {rfq.auction_type} not implemented")

        # Add quotes to engine
        for quote in quotes:
            engine.add_quote(quote)

        # Execute
        if rfq.status in {RFQStatus.OPEN, RFQStatus.QUOTING}:
            try:
                self._transition(rfq, RFQStatus.CLOSED, reason="RFQ closed for execution")
            except ValueError:
                # Allow execution if already closed or in terminal state
                pass
        result = engine.execute()

        # Update RFQ
        self._transition(
            rfq,
            RFQStatus.EXECUTED,
            reason="RFQ executed",
            metadata=result.model_dump(mode="json"),
        )
        rfq.execution_price = result.clearing_price
        rfq.execution_time = result.execution_time
        rfq.best_quote_id = result.winning_quotes[0] if result.winning_quotes else None

        self.auctions[rfq_id] = engine
        self.results[rfq_id] = result

        return result

    def get_rfq(self, rfq_id: str) -> Optional[RFQ]:
        """Get RFQ by ID."""
        return self.rfqs.get(rfq_id)

    def get_quotes(self, rfq_id: str) -> List[Quote]:
        """Get quotes for RFQ."""
        return self.quotes.get(rfq_id, [])

    def get_result(self, rfq_id: str) -> Optional[AuctionResult]:
        """Get auction result."""
        return self.results.get(rfq_id)

    def cancel_rfq(self, rfq_id: str) -> RFQ:
        """Cancel RFQ."""
        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        if rfq.status in [RFQStatus.EXECUTED, RFQStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel RFQ {rfq_id} in status {rfq.status}")

        self._transition(rfq, RFQStatus.CANCELLED, reason="RFQ cancelled")
        return rfq

    def update_status(
        self,
        rfq_id: str,
        new_status: RFQStatus,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RFQ:
        """Manually update RFQ status following allowed transitions."""

        rfq = self.rfqs.get(rfq_id)
        if not rfq:
            raise ValueError(f"RFQ {rfq_id} not found")

        self._transition(rfq, new_status, reason=reason, metadata=metadata)
        return rfq

    def register_state_listener(self, rfq_id: str) -> asyncio.Queue:
        """Register a listener for RFQ state updates."""

        queue: asyncio.Queue = asyncio.Queue()
        self._state_listeners.setdefault(rfq_id, []).append(queue)

        rfq = self.rfqs.get(rfq_id)
        if rfq:
            queue.put_nowait(self._serialize_status(rfq))

        return queue

    def unregister_state_listener(self, rfq_id: str, queue: asyncio.Queue) -> None:
        """Remove a previously registered state listener."""

        listeners = self._state_listeners.get(rfq_id, [])
        if queue in listeners:
            listeners.remove(queue)
        if not listeners and rfq_id in self._state_listeners:
            del self._state_listeners[rfq_id]

    def _transition(
        self,
        rfq: RFQ,
        new_status: RFQStatus,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Validate and apply a state transition."""

        current = rfq.status
        allowed = self._ALLOWED_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid RFQ status transition from {current} to {new_status}"
            )

        rfq.record_status(new_status, reason=reason, metadata=metadata)
        self._notify_state(rfq, reason=reason, metadata=metadata)

    def _notify_state(
        self,
        rfq: RFQ,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify registered listeners of a state change."""

        payload = self._serialize_status(rfq, reason=reason, metadata=metadata)
        for queue in self._state_listeners.get(rfq.rfq_id, []):
            queue.put_nowait(payload)

    @staticmethod
    def _serialize_status(
        rfq: RFQ,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Serialize RFQ status information for API delivery."""

        serialized = {
            "rfq_id": rfq.rfq_id,
            "status": rfq.status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "quotes_received": rfq.quotes_received,
            "execution_price": float(rfq.execution_price) if rfq.execution_price else None,
        }
        if reason:
            serialized["reason"] = reason
        if metadata:
            serialized["metadata"] = metadata
        return serialized


__all__ = [
    "RFQ",
    "RFQStatus",
    "RFQSpecification",
    "Quote",
    "QuoteStatus",
    "AuctionType",
    "OrderSide",
    "TimeInForce",
    "AuctionResult",
    "OrderBook",
    "OrderBookLevel",
    "AuctionEngine",
    "SinglePriceAuction",
    "MultiPriceAuction",
    "DutchAuction",
    "VickreyAuction",
    "SecondPriceAuction",
    "MultiRoundAuction",
    "ContinuousAuction",
    "RFQManager",
]
