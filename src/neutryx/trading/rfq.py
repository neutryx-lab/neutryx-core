"""Request for Quote (RFQ) workflow and auction mechanisms.

This module implements RFQ workflows for derivatives pricing:
- RFQ creation and submission
- Quote generation and submission
- Multi-dealer auctions
- Best execution tracking
- Quote acceptance/rejection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class RFQStatus(Enum):
    """Status of an RFQ."""

    DRAFT = "draft"  # Being prepared
    SUBMITTED = "submitted"  # Sent to dealers
    QUOTED = "quoted"  # Received at least one quote
    EXECUTED = "executed"  # Quote accepted and trade executed
    EXPIRED = "expired"  # Timeout reached
    CANCELLED = "cancelled"  # Cancelled by requester
    REJECTED = "rejected"  # Rejected by all dealers


class AuctionType(Enum):
    """Type of auction mechanism."""

    SINGLE_DEALER = "single_dealer"  # Request to single dealer
    MULTI_DEALER = "multi_dealer"  # Competitive bidding
    BLIND_AUCTION = "blind_auction"  # Sealed bids
    OPEN_AUCTION = "open_auction"  # Visible bids with revisions


class QuoteStatus(Enum):
    """Status of a dealer quote."""

    PENDING = "pending"  # Quote requested but not submitted
    SUBMITTED = "submitted"  # Quote submitted to client
    ACCEPTED = "accepted"  # Client accepted quote
    REJECTED = "rejected"  # Client rejected quote
    EXPIRED = "expired"  # Quote validity expired
    WITHDRAWN = "withdrawn"  # Dealer withdrew quote


class RFQRequest(BaseModel):
    """Request for Quote from a client.

    Example:
        >>> rfq = RFQRequest(
        ...     client_id="CLIENT-001",
        ...     product_type="interest_rate_swap",
        ...     parameters={
        ...         "notional": 10_000_000,
        ...         "fixed_rate": 0.05,
        ...         "tenor": 5,
        ...         "currency": "USD"
        ...     },
        ...     dealer_ids=["DEALER-001", "DEALER-002", "DEALER-003"]
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rfq_id: str = Field(default_factory=lambda: f"RFQ-{uuid4().hex[:12].upper()}")
    client_id: str = Field(..., description="Client requesting quote")
    product_type: str = Field(..., description="Type of product (e.g., IRS, swaption, FX option)")
    parameters: Dict[str, Any] = Field(..., description="Product parameters")
    auction_type: AuctionType = Field(default=AuctionType.MULTI_DEALER)
    dealer_ids: List[str] = Field(..., description="List of dealers to request quotes from")
    quantity: Optional[float] = Field(None, description="Quantity/notional amount")
    side: Optional[str] = Field(None, description="Buy or Sell")
    settlement_date: Optional[datetime] = None
    validity_period: timedelta = Field(
        default=timedelta(minutes=30), description="How long quotes remain valid"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: RFQStatus = RFQStatus.DRAFT
    metadata: Dict[str, str] = Field(default_factory=dict)

    def __post_init__(self):
        """Set expiration time."""
        if self.expires_at is None:
            self.expires_at = self.created_at + self.validity_period


class RFQQuote(BaseModel):
    """Quote from a dealer in response to RFQ."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    quote_id: str = Field(default_factory=lambda: f"QTE-{uuid4().hex[:12].upper()}")
    rfq_id: str = Field(..., description="RFQ this quote responds to")
    dealer_id: str = Field(..., description="Dealer providing quote")
    bid_price: Optional[float] = Field(None, description="Bid price")
    ask_price: Optional[float] = Field(None, description="Ask price")
    mid_price: Optional[float] = Field(None, description="Mid price")
    spread: Optional[float] = Field(None, description="Bid-ask spread")
    all_in_price: float = Field(..., description="Total price including fees")
    quote_time: datetime = Field(default_factory=datetime.now)
    valid_until: datetime = Field(..., description="Quote expiration time")
    status: QuoteStatus = QuoteStatus.SUBMITTED
    greeks: Optional[Dict[str, float]] = Field(None, description="Risk sensitivities")
    fees: Optional[Dict[str, float]] = Field(None, description="Breakdown of fees")
    notes: Optional[str] = Field(None, description="Additional notes from dealer")
    metadata: Dict[str, str] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if quote is still valid."""
        return datetime.now() < self.valid_until and self.status == QuoteStatus.SUBMITTED

    @property
    def is_executable(self) -> bool:
        """Check if quote can be executed."""
        return self.is_valid and self.status == QuoteStatus.SUBMITTED


class RFQResponse(BaseModel):
    """Client's response to dealer quotes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rfq_id: str
    accepted_quote_id: Optional[str] = None
    rejected_quote_ids: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None
    response_time: datetime = Field(default_factory=datetime.now)
    executed_trade_id: Optional[str] = None


@dataclass
class RFQAuction:
    """Auction session for an RFQ."""

    rfq_id: str
    auction_type: AuctionType
    start_time: datetime
    end_time: datetime
    quotes: List[RFQQuote] = field(default_factory=list)
    best_bid: Optional[RFQQuote] = None
    best_ask: Optional[RFQQuote] = None
    winner: Optional[RFQQuote] = None
    completed: bool = False

    def add_quote(self, quote: RFQQuote) -> None:
        """Add a quote to the auction.

        Args:
            quote: Dealer quote to add

        Raises:
            ValueError: If auction has ended or quote is for wrong RFQ
        """
        if self.completed:
            raise ValueError("Auction has already completed")

        if datetime.now() > self.end_time:
            raise ValueError("Auction has expired")

        if quote.rfq_id != self.rfq_id:
            raise ValueError(f"Quote RFQ ID {quote.rfq_id} does not match auction {self.rfq_id}")

        self.quotes.append(quote)
        self._update_best_quotes()

    def _update_best_quotes(self) -> None:
        """Update best bid and ask quotes."""
        valid_quotes = [q for q in self.quotes if q.is_valid]

        if not valid_quotes:
            return

        # Best bid: highest bid price
        bids = [q for q in valid_quotes if q.bid_price is not None]
        if bids:
            self.best_bid = max(bids, key=lambda q: q.bid_price)

        # Best ask: lowest ask price
        asks = [q for q in valid_quotes if q.ask_price is not None]
        if asks:
            self.best_ask = min(asks, key=lambda q: q.ask_price)

    def get_best_execution(self, side: str) -> Optional[RFQQuote]:
        """Get best quote for execution.

        Args:
            side: "buy" or "sell"

        Returns:
            Best quote for the specified side
        """
        if side.lower() == "buy":
            return self.best_ask  # Best price to buy at (lowest ask)
        elif side.lower() == "sell":
            return self.best_bid  # Best price to sell at (highest bid)
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")

    def complete_auction(self, accepted_quote_id: Optional[str] = None) -> None:
        """Complete the auction.

        Args:
            accepted_quote_id: ID of accepted quote, if any
        """
        self.completed = True

        for quote in self.quotes:
            if quote.quote_id == accepted_quote_id:
                quote.status = QuoteStatus.ACCEPTED
                self.winner = quote
            elif quote.is_valid:
                quote.status = QuoteStatus.REJECTED


class RFQManager:
    """Manages RFQ workflows and auctions.

    Example:
        >>> manager = RFQManager()
        >>>
        >>> # Create and submit RFQ
        >>> rfq = RFQRequest(
        ...     client_id="CLIENT-001",
        ...     product_type="interest_rate_swap",
        ...     parameters={"notional": 10_000_000, "tenor": 5},
        ...     dealer_ids=["DEALER-001", "DEALER-002"]
        ... )
        >>> manager.submit_rfq(rfq)
        >>>
        >>> # Dealers submit quotes
        >>> quote1 = RFQQuote(
        ...     rfq_id=rfq.rfq_id,
        ...     dealer_id="DEALER-001",
        ...     bid_price=100.5,
        ...     ask_price=101.0,
        ...     all_in_price=100.75,
        ...     valid_until=datetime.now() + timedelta(minutes=30)
        ... )
        >>> manager.submit_quote(quote1)
        >>>
        >>> # Get best quotes
        >>> best_quotes = manager.get_best_quotes(rfq.rfq_id)
        >>>
        >>> # Accept quote
        >>> response = RFQResponse(
        ...     rfq_id=rfq.rfq_id,
        ...     accepted_quote_id=quote1.quote_id
        ... )
        >>> trade_id = manager.respond_to_rfq(response)
    """

    def __init__(self):
        """Initialize RFQ manager."""
        self._rfqs: Dict[str, RFQRequest] = {}
        self._quotes: Dict[str, List[RFQQuote]] = {}  # rfq_id -> quotes
        self._auctions: Dict[str, RFQAuction] = {}  # rfq_id -> auction
        self._responses: Dict[str, RFQResponse] = {}  # rfq_id -> response

    def submit_rfq(self, rfq: RFQRequest) -> RFQAuction:
        """Submit an RFQ to dealers.

        Args:
            rfq: RFQ request to submit

        Returns:
            Created auction session

        Raises:
            ValueError: If RFQ already exists
        """
        if rfq.rfq_id in self._rfqs:
            raise ValueError(f"RFQ {rfq.rfq_id} already exists")

        # Update status
        rfq.status = RFQStatus.SUBMITTED

        # Store RFQ
        self._rfqs[rfq.rfq_id] = rfq
        self._quotes[rfq.rfq_id] = []

        # Create auction
        auction = RFQAuction(
            rfq_id=rfq.rfq_id,
            auction_type=rfq.auction_type,
            start_time=datetime.now(),
            end_time=rfq.expires_at or datetime.now() + rfq.validity_period,
        )
        self._auctions[rfq.rfq_id] = auction

        return auction

    def submit_quote(self, quote: RFQQuote) -> None:
        """Submit a dealer quote.

        Args:
            quote: Dealer quote

        Raises:
            ValueError: If RFQ not found or invalid
        """
        if quote.rfq_id not in self._rfqs:
            raise ValueError(f"RFQ {quote.rfq_id} not found")

        rfq = self._rfqs[quote.rfq_id]

        # Check RFQ status
        if rfq.status not in [RFQStatus.SUBMITTED, RFQStatus.QUOTED]:
            raise ValueError(f"RFQ {quote.rfq_id} is not accepting quotes (status: {rfq.status})")

        # Check dealer authorization
        if quote.dealer_id not in rfq.dealer_ids:
            raise ValueError(f"Dealer {quote.dealer_id} not authorized for RFQ {quote.rfq_id}")

        # Add to quotes
        self._quotes[quote.rfq_id].append(quote)

        # Update RFQ status
        if rfq.status == RFQStatus.SUBMITTED:
            rfq.status = RFQStatus.QUOTED

        # Add to auction
        auction = self._auctions[quote.rfq_id]
        auction.add_quote(quote)

    def get_rfq(self, rfq_id: str) -> Optional[RFQRequest]:
        """Get RFQ by ID.

        Args:
            rfq_id: RFQ ID

        Returns:
            RFQ or None if not found
        """
        return self._rfqs.get(rfq_id)

    def get_quotes(self, rfq_id: str) -> List[RFQQuote]:
        """Get all quotes for an RFQ.

        Args:
            rfq_id: RFQ ID

        Returns:
            List of quotes
        """
        return self._quotes.get(rfq_id, [])

    def get_best_quotes(self, rfq_id: str) -> Dict[str, Optional[RFQQuote]]:
        """Get best bid and ask quotes for an RFQ.

        Args:
            rfq_id: RFQ ID

        Returns:
            Dictionary with 'best_bid' and 'best_ask' quotes
        """
        auction = self._auctions.get(rfq_id)
        if not auction:
            return {"best_bid": None, "best_ask": None}

        return {"best_bid": auction.best_bid, "best_ask": auction.best_ask}

    def respond_to_rfq(self, response: RFQResponse) -> Optional[str]:
        """Respond to RFQ by accepting or rejecting quotes.

        Args:
            response: Client response

        Returns:
            Trade ID if quote was accepted, None otherwise

        Raises:
            ValueError: If RFQ not found or response invalid
        """
        if response.rfq_id not in self._rfqs:
            raise ValueError(f"RFQ {response.rfq_id} not found")

        rfq = self._rfqs[response.rfq_id]
        auction = self._auctions[response.rfq_id]

        # Store response
        self._responses[response.rfq_id] = response

        if response.accepted_quote_id:
            # Find accepted quote
            accepted_quote = None
            for quote in self._quotes[response.rfq_id]:
                if quote.quote_id == response.accepted_quote_id:
                    accepted_quote = quote
                    break

            if not accepted_quote:
                raise ValueError(f"Quote {response.accepted_quote_id} not found")

            if not accepted_quote.is_executable:
                raise ValueError(f"Quote {response.accepted_quote_id} is not executable")

            # Update statuses
            accepted_quote.status = QuoteStatus.ACCEPTED
            rfq.status = RFQStatus.EXECUTED

            # Complete auction
            auction.complete_auction(accepted_quote.quote_id)

            # Generate trade ID
            trade_id = f"TRD-{uuid4().hex[:12].upper()}"
            response.executed_trade_id = trade_id

            return trade_id
        else:
            # All quotes rejected
            rfq.status = RFQStatus.REJECTED
            auction.complete_auction()

            return None

    def cancel_rfq(self, rfq_id: str, reason: Optional[str] = None) -> None:
        """Cancel an RFQ.

        Args:
            rfq_id: RFQ ID to cancel
            reason: Cancellation reason

        Raises:
            ValueError: If RFQ not found or already completed
        """
        if rfq_id not in self._rfqs:
            raise ValueError(f"RFQ {rfq_id} not found")

        rfq = self._rfqs[rfq_id]

        if rfq.status in [RFQStatus.EXECUTED, RFQStatus.EXPIRED, RFQStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel RFQ {rfq_id} with status {rfq.status}")

        # Update status
        rfq.status = RFQStatus.CANCELLED

        # Expire all quotes
        for quote in self._quotes[rfq_id]:
            if quote.status == QuoteStatus.SUBMITTED:
                quote.status = QuoteStatus.EXPIRED

        # Complete auction
        auction = self._auctions[rfq_id]
        auction.complete_auction()

    def expire_rfqs(self) -> List[str]:
        """Expire RFQs that have exceeded their validity period.

        Returns:
            List of expired RFQ IDs
        """
        expired_ids = []
        now = datetime.now()

        for rfq_id, rfq in self._rfqs.items():
            if rfq.status in [RFQStatus.SUBMITTED, RFQStatus.QUOTED]:
                if rfq.expires_at and now > rfq.expires_at:
                    rfq.status = RFQStatus.EXPIRED
                    expired_ids.append(rfq_id)

                    # Expire quotes
                    for quote in self._quotes[rfq_id]:
                        if quote.status == QuoteStatus.SUBMITTED:
                            quote.status = QuoteStatus.EXPIRED

        return expired_ids

    def get_statistics(self, dealer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get RFQ statistics.

        Args:
            dealer_id: Optional dealer ID to filter by

        Returns:
            Dictionary with statistics
        """
        total_rfqs = len(self._rfqs)
        total_quotes = sum(len(quotes) for quotes in self._quotes.values())

        if dealer_id:
            dealer_quotes = [
                q for quotes in self._quotes.values() for q in quotes if q.dealer_id == dealer_id
            ]
            accepted_quotes = [q for q in dealer_quotes if q.status == QuoteStatus.ACCEPTED]

            return {
                "dealer_id": dealer_id,
                "total_quotes_submitted": len(dealer_quotes),
                "quotes_accepted": len(accepted_quotes),
                "win_rate": len(accepted_quotes) / len(dealer_quotes) if dealer_quotes else 0.0,
            }
        else:
            executed_rfqs = [rfq for rfq in self._rfqs.values() if rfq.status == RFQStatus.EXECUTED]

            return {
                "total_rfqs": total_rfqs,
                "total_quotes": total_quotes,
                "executed_rfqs": len(executed_rfqs),
                "execution_rate": len(executed_rfqs) / total_rfqs if total_rfqs else 0.0,
                "avg_quotes_per_rfq": total_quotes / total_rfqs if total_rfqs else 0.0,
            }


__all__ = [
    "RFQStatus",
    "AuctionType",
    "QuoteStatus",
    "RFQRequest",
    "RFQQuote",
    "RFQResponse",
    "RFQAuction",
    "RFQManager",
]
