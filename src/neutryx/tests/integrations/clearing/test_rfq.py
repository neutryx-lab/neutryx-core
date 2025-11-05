"""Tests for RFQ workflow and auction mechanisms."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from neutryx.integrations.clearing.base import Party, ProductType
from neutryx.integrations.clearing.rfq import (
    RFQ,
    RFQSpecification,
    RFQStatus,
    Quote,
    QuoteStatus,
    AuctionType,
    OrderSide,
    TimeInForce,
    RFQManager,
    SinglePriceAuction,
    MultiPriceAuction,
    ContinuousAuction,
    OrderBook,
)


@pytest.fixture
def requester():
    """Sample requester party."""
    return Party(
        party_id="BUYER001",
        name="Buyer Bank",
        lei="1234567890ABCDEFGH12",
        bic="BUYRBANKXXX"
    )


@pytest.fixture
def quoter1():
    """First quote provider."""
    return Party(
        party_id="DEALER001",
        name="Dealer One",
        lei="DEALER1234567890AB",
        bic="DEALR001XXX"
    )


@pytest.fixture
def quoter2():
    """Second quote provider."""
    return Party(
        party_id="DEALER002",
        name="Dealer Two",
        lei="DEALER0987654321AB",
        bic="DEALR002XXX"
    )


@pytest.fixture
def rfq_spec():
    """Sample RFQ specification."""
    return RFQSpecification(
        product_type=ProductType.IRS,
        notional=Decimal("10000000"),
        currency="USD",
        effective_date=datetime.now() + timedelta(days=2),
        maturity_date=datetime.now() + timedelta(days=365*5),
        tenor="5Y",
        day_count="ACT/360"
    )


class TestRFQBasics:
    """Test basic RFQ functionality."""

    def test_rfq_creation(self, requester, rfq_spec):
        """Test RFQ creation."""
        rfq = RFQ(
            requester=requester,
            specification=rfq_spec,
            side=OrderSide.BUY,
            quote_deadline=datetime.utcnow() + timedelta(hours=2),
            expiry_time=datetime.utcnow() + timedelta(hours=4)
        )

        assert rfq.rfq_id.startswith("RFQ-")
        assert rfq.status == RFQStatus.DRAFT
        assert rfq.requester == requester
        assert rfq.side == OrderSide.BUY

    def test_rfq_spec_validation(self):
        """Test RFQ specification validation."""
        # Negative notional should fail
        with pytest.raises(ValueError, match="Notional must be positive"):
            RFQSpecification(
                product_type=ProductType.IRS,
                notional=Decimal("-1000000"),
                currency="USD",
                effective_date=datetime.now(),
                maturity_date=datetime.now() + timedelta(days=365)
            )

    def test_quote_creation(self, quoter1):
        """Test quote creation."""
        quote = Quote(
            rfq_id="RFQ-TEST123",
            quoter=quoter1,
            quoter_member_id="MEM001",
            side=OrderSide.SELL,
            quantity=Decimal("10000000"),
            price=Decimal("2.5"),
            time_in_force=TimeInForce.GTC
        )

        assert quote.quote_id.startswith("QTE-")
        assert quote.status == QuoteStatus.PENDING
        assert quote.remaining_quantity() == quote.quantity

    def test_quote_expiry(self, quoter1):
        """Test quote expiry check."""
        # GTD quote that has expired
        quote = Quote(
            rfq_id="RFQ-TEST",
            quoter=quoter1,
            quoter_member_id="MEM001",
            side=OrderSide.SELL,
            quantity=Decimal("1000000"),
            time_in_force=TimeInForce.GTD,
            good_till=datetime.utcnow() - timedelta(hours=1)
        )

        assert quote.is_expired()


class TestRFQManager:
    """Test RFQ Manager."""

    def test_create_and_submit_rfq(self, requester, rfq_spec):
        """Test creating and submitting RFQ."""
        manager = RFQManager()

        rfq = manager.create_rfq(
            requester=requester,
            specification=rfq_spec,
            side=OrderSide.BUY,
            quote_deadline=datetime.utcnow() + timedelta(hours=2)
        )

        assert rfq.status == RFQStatus.DRAFT

        # Submit RFQ
        rfq = manager.submit_rfq(rfq.rfq_id)
        assert rfq.status == RFQStatus.OPEN

    def test_submit_quote(self, requester, rfq_spec, quoter1):
        """Test submitting quotes to RFQ."""
        manager = RFQManager()

        rfq = manager.create_rfq(
            requester=requester,
            specification=rfq_spec,
            side=OrderSide.BUY,
            quote_deadline=datetime.utcnow() + timedelta(hours=2)
        )
        manager.submit_rfq(rfq.rfq_id)

        # Submit quote
        quote = Quote(
            rfq_id=rfq.rfq_id,
            quoter=quoter1,
            quoter_member_id="MEM001",
            side=OrderSide.SELL,
            quantity=rfq_spec.notional,
            price=Decimal("2.5")
        )

        submitted_quote = manager.submit_quote(rfq.rfq_id, quote)
        assert submitted_quote.status == QuoteStatus.SUBMITTED
        assert rfq.quotes_received == 1

    def test_cancel_rfq(self, requester, rfq_spec):
        """Test cancelling RFQ."""
        manager = RFQManager()

        rfq = manager.create_rfq(
            requester=requester,
            specification=rfq_spec,
            side=OrderSide.BUY,
            quote_deadline=datetime.utcnow() + timedelta(hours=2)
        )
        manager.submit_rfq(rfq.rfq_id)

        # Cancel
        rfq = manager.cancel_rfq(rfq.rfq_id)
        assert rfq.status == RFQStatus.CANCELLED


class TestSinglePriceAuction:
    """Test single-price auction."""

    def test_single_price_auction_execution(self, requester, rfq_spec, quoter1, quoter2):
        """Test single-price auction execution."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        rfq = RFQ(
            requester=requester,
            specification=rfq_spec,
            side=OrderSide.BUY,
            submission_time=past_time,
            quote_deadline=datetime.utcnow() - timedelta(seconds=1),  # Expired
            expiry_time=datetime.utcnow() + timedelta(hours=1),
            auction_type=AuctionType.SINGLE_PRICE
        )

        auction = SinglePriceAuction(rfq)

        # Add quotes
        quote1 = Quote(
            rfq_id=rfq.rfq_id,
            quoter=quoter1,
            quoter_member_id="MEM001",
            side=OrderSide.SELL,
            quantity=Decimal("6000000"),
            price=Decimal("2.5")  # Best price
        )

        quote2 = Quote(
            rfq_id=rfq.rfq_id,
            quoter=quoter2,
            quoter_member_id="MEM002",
            side=OrderSide.SELL,
            quantity=Decimal("5000000"),
            price=Decimal("2.6")  # Second best
        )

        auction.add_quote(quote1)
        auction.add_quote(quote2)

        # Execute
        result = auction.execute()

        assert result.clearing_price is not None
        assert result.total_quantity_filled == rfq_spec.notional
        assert len(result.winning_quotes) >= 1
        assert result.num_participants == 2


class TestMultiPriceAuction:
    """Test multi-price auction."""

    def test_multi_price_auction(self, requester, rfq_spec, quoter1, quoter2):
        """Test multi-price auction execution."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        rfq = RFQ(
            requester=requester,
            specification=rfq_spec,
            side=OrderSide.BUY,
            submission_time=past_time,
            quote_deadline=datetime.utcnow() - timedelta(seconds=1),
            expiry_time=datetime.utcnow() + timedelta(hours=1),
            auction_type=AuctionType.MULTI_PRICE
        )

        auction = MultiPriceAuction(rfq)

        quote1 = Quote(
            rfq_id=rfq.rfq_id,
            quoter=quoter1,
            quoter_member_id="MEM001",
            side=OrderSide.SELL,
            quantity=Decimal("6000000"),
            price=Decimal("2.5")
        )

        quote2 = Quote(
            rfq_id=rfq.rfq_id,
            quoter=quoter2,
            quoter_member_id="MEM002",
            side=OrderSide.SELL,
            quantity=Decimal("5000000"),
            price=Decimal("2.6")
        )

        auction.add_quote(quote1)
        auction.add_quote(quote2)

        result = auction.execute()

        # In multi-price, each winner pays their own price
        assert result.clearing_price is None  # No single clearing price
        assert result.total_quantity_filled == rfq_spec.notional
        assert len(result.allocations) >= 1


class TestOrderBook:
    """Test order book functionality."""

    def test_order_book_basics(self, quoter1, quoter2):
        """Test order book operations."""
        book = OrderBook("RFQ-TEST")

        quote1 = Quote(
            rfq_id="RFQ-TEST",
            quoter=quoter1,
            quoter_member_id="MEM001",
            side=OrderSide.BUY,
            quantity=Decimal("1000000"),
            price=Decimal("2.5")
        )

        quote2 = Quote(
            rfq_id="RFQ-TEST",
            quoter=quoter2,
            quoter_member_id="MEM002",
            side=OrderSide.SELL,
            quantity=Decimal("1000000"),
            price=Decimal("2.6")
        )

        book.add_quote(quote1)
        book.add_quote(quote2)

        assert book.best_bid() == Decimal("2.5")
        assert book.best_offer() == Decimal("2.6")
        assert book.spread() == Decimal("0.1")
        assert book.mid_price() == Decimal("2.55")

    def test_order_book_depth(self, quoter1):
        """Test order book depth."""
        book = OrderBook("RFQ-TEST")

        # Add multiple levels
        for i in range(5):
            quote = Quote(
                rfq_id="RFQ-TEST",
                quoter=quoter1,
                quoter_member_id="MEM001",
                side=OrderSide.BUY,
                quantity=Decimal("1000000"),
                price=Decimal(f"2.{5-i}")
            )
            book.add_quote(quote)

        depth = book.depth(OrderSide.BUY, num_levels=3)
        assert len(depth) == 3
        # Best bid should be 2.5
        assert depth[0].price == Decimal("2.5")


class TestContinuousAuction:
    """Test continuous auction."""

    def test_continuous_auction(self, requester, rfq_spec, quoter1, quoter2):
        """Test continuous double auction."""
        rfq = RFQ(
            requester=requester,
            specification=rfq_spec,
            side=OrderSide.BUY,
            quote_deadline=datetime.utcnow() + timedelta(hours=1),
            expiry_time=datetime.utcnow() + timedelta(hours=2),
            auction_type=AuctionType.CONTINUOUS
        )

        auction = ContinuousAuction(rfq)

        # Add crossing quotes (bid >= offer)
        bid_quote = Quote(
            rfq_id=rfq.rfq_id,
            quoter=quoter1,
            quoter_member_id="MEM001",
            side=OrderSide.BUY,
            quantity=Decimal("5000000"),
            price=Decimal("2.6")  # High bid
        )

        offer_quote = Quote(
            rfq_id=rfq.rfq_id,
            quoter=quoter2,
            quoter_member_id="MEM002",
            side=OrderSide.SELL,
            quantity=Decimal("5000000"),
            price=Decimal("2.5")  # Low offer (should cross)
        )

        auction.add_quote(bid_quote)
        auction.add_quote(offer_quote)

        # Should have executed trades
        assert len(auction.trades) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
