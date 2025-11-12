from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from neutryx.integrations.clearing.base import Party, ProductType
from neutryx.integrations.clearing.rfq import (
    AuctionType,
    OrderSide,
    Quote,
    RFQManager,
    RFQSpecification,
)


def _make_party(idx: int) -> Party:
    return Party(party_id=f"PTY-{idx}", name=f"Dealer {idx}")


def _prepare_rfq(manager: RFQManager, auction_type: AuctionType, side: OrderSide, notional: str) -> str:
    spec = RFQSpecification(
        product_type=ProductType.IRS,
        notional=Decimal(notional),
        currency="USD",
        effective_date=datetime.utcnow(),
        maturity_date=datetime.utcnow() + timedelta(days=365),
    )
    rfq = manager.create_rfq(
        requester=_make_party(0),
        specification=spec,
        side=side,
        quote_deadline=datetime.utcnow() + timedelta(minutes=1),
        auction_type=auction_type,
    )
    manager.submit_rfq(rfq.rfq_id)
    return rfq.rfq_id


def _close_quotes(manager: RFQManager, rfq_id: str) -> None:
    rfq = manager.get_rfq(rfq_id)
    assert rfq is not None
    rfq.quote_deadline = datetime.utcnow() - timedelta(seconds=1)


def test_dutch_auction_price_path_and_allocation():
    manager = RFQManager()
    rfq_id = _prepare_rfq(manager, AuctionType.DUTCH, OrderSide.SELL, "100")

    quotes = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.BUY,
            quantity=Decimal("40"),
            price=Decimal("105"),
            priority=1,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.BUY,
            quantity=Decimal("60"),
            price=Decimal("103"),
            priority=0,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(3),
            quoter_member_id="M3",
            side=OrderSide.BUY,
            quantity=Decimal("40"),
            price=Decimal("101"),
            priority=0,
        ),
    ]

    for quote in quotes:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.auction_type is AuctionType.DUTCH
    assert result.clearing_price == Decimal("103")
    assert result.total_quantity_filled == Decimal("100")

    price_path = result.metadata.get("price_path")
    assert price_path is not None
    assert price_path[0]["price"] == 105.0
    assert price_path[1]["price"] == 103.0

    winning_ids = set(result.winning_quotes)
    assert quotes[0].quote_id in winning_ids
    assert quotes[1].quote_id in winning_ids
    assert quotes[2].status.name == "REJECTED"


def test_dutch_auction_partial_fill_when_liquidity_insufficient():
    manager = RFQManager()
    rfq_id = _prepare_rfq(manager, AuctionType.DUTCH, OrderSide.SELL, "150")

    quotes = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("105"),
            priority=0,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.BUY,
            quantity=Decimal("30"),
            price=Decimal("103"),
            priority=1,
        ),
    ]

    for quote in quotes:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.total_quantity_filled == Decimal("80")
    assert result.clearing_price == Decimal("103")
    assert quotes[1].status.name == "ACCEPTED"


def test_vickrey_auction_second_price_logic_for_buy_side():
    manager = RFQManager()
    rfq_id = _prepare_rfq(manager, AuctionType.VICKREY, OrderSide.BUY, "70")

    quotes = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.SELL,
            quantity=Decimal("40"),
            price=Decimal("100"),
            priority=1,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.SELL,
            quantity=Decimal("40"),
            price=Decimal("102"),
            priority=0,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(3),
            quoter_member_id="M3",
            side=OrderSide.SELL,
            quantity=Decimal("50"),
            price=Decimal("104"),
            priority=0,
        ),
    ]

    for quote in quotes:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.auction_type is AuctionType.VICKREY
    assert result.total_quantity_filled == Decimal("70")
    assert result.clearing_price == Decimal("104")
    assert result.metadata.get("second_price_quote_id") == quotes[2].quote_id

    for allocation in result.allocations:
        assert allocation["price"] == float(Decimal("104"))

    assert quotes[2].status.name == "REJECTED"


def test_vickrey_auction_prefers_lower_priority_on_ties():
    manager = RFQManager()
    rfq_id = _prepare_rfq(manager, AuctionType.VICKREY, OrderSide.SELL, "60")

    quotes = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.BUY,
            quantity=Decimal("40"),
            price=Decimal("101"),
            priority=0,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.BUY,
            quantity=Decimal("30"),
            price=Decimal("101"),
            priority=2,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(3),
            quoter_member_id="M3",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("99"),
            priority=1,
        ),
    ]

    for quote in quotes:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.total_quantity_filled == Decimal("60")
    assert result.clearing_price == Decimal("99")
    # Highest priority (lowest value) should be filled first among tied prices
    assert quotes[0].status.name == "ACCEPTED"
    assert quotes[1].status.name == "ACCEPTED"
    assert quotes[2].status.name == "REJECTED"

    # Second-price should reference the losing quote when available
    assert result.metadata.get("second_price_quote_id") == quotes[2].quote_id
