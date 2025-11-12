import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from neutryx.integrations.clearing.base import Party, ProductType
from neutryx.integrations.clearing.rfq import (  # noqa: E402
    AuctionType,
    OrderSide,
    Quote,
    QuoteStatus,
    RFQManager,
    RFQSpecification,
    RFQStatus,
)


@pytest.fixture
def manager() -> RFQManager:
    return RFQManager()


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


def test_single_price_auction_uniform_clearing(manager: RFQManager) -> None:
    rfq_id = _prepare_rfq(manager, AuctionType.SINGLE_PRICE, OrderSide.SELL, "80")

    quotes: List[Quote] = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("105"),
            priority=1,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.BUY,
            quantity=Decimal("40"),
            price=Decimal("104"),
            priority=0,
        ),
    ]

    for quote in quotes:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.auction_type is AuctionType.SINGLE_PRICE
    assert result.clearing_price == Decimal("104")
    assert result.total_quantity_filled == Decimal("80")

    winning_ids = set(result.winning_quotes)
    assert quotes[0].quote_id in winning_ids
    assert quotes[1].quote_id in winning_ids

    rfq = manager.get_rfq(rfq_id)
    assert rfq is not None
    statuses = [entry["status"].value for entry in rfq.status_history]
    assert statuses[:3] == ["draft", "submitted", "open"]
    assert statuses[-2:] == ["closed", "executed"]


def test_multi_price_auction_discriminatory(manager: RFQManager) -> None:
    rfq_id = _prepare_rfq(manager, AuctionType.MULTI_PRICE, OrderSide.SELL, "70")

    quotes = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.BUY,
            quantity=Decimal("40"),
            price=Decimal("105"),
            priority=0,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.BUY,
            quantity=Decimal("50"),
            price=Decimal("103"),
            priority=1,
        ),
    ]

    for quote in quotes:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.auction_type is AuctionType.MULTI_PRICE
    assert [alloc["price"] for alloc in result.allocations] == [105.0, 103.0]

    assert quotes[0].status == QuoteStatus.ACCEPTED
    assert quotes[1].status == QuoteStatus.ACCEPTED


def test_second_price_auction_uses_reference_quote(manager: RFQManager) -> None:
    rfq_id = _prepare_rfq(manager, AuctionType.SECOND_PRICE, OrderSide.BUY, "60")

    quotes = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.SELL,
            quantity=Decimal("30"),
            price=Decimal("99"),
            priority=1,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.SELL,
            quantity=Decimal("30"),
            price=Decimal("100"),
            priority=0,
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(3),
            quoter_member_id="M3",
            side=OrderSide.SELL,
            quantity=Decimal("40"),
            price=Decimal("102"),
            priority=0,
        ),
    ]

    for quote in quotes:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.auction_type is AuctionType.SECOND_PRICE
    assert result.clearing_price == Decimal("102")
    assert all(alloc["price"] == 102.0 for alloc in result.allocations)
    assert quotes[2].status == QuoteStatus.REJECTED


def test_multi_round_auction_tracks_rounds(manager: RFQManager) -> None:
    rfq_id = _prepare_rfq(manager, AuctionType.MULTI_ROUND, OrderSide.SELL, "90")

    round_one = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(1),
            quoter_member_id="M1",
            side=OrderSide.BUY,
            quantity=Decimal("30"),
            price=Decimal("106"),
            priority=0,
            metadata={"round": 1},
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(2),
            quoter_member_id="M2",
            side=OrderSide.BUY,
            quantity=Decimal("20"),
            price=Decimal("104"),
            priority=0,
            metadata={"round": 1},
        ),
    ]

    round_two = [
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(3),
            quoter_member_id="M3",
            side=OrderSide.BUY,
            quantity=Decimal("40"),
            price=Decimal("103"),
            priority=1,
            metadata={"round": 2},
        ),
        Quote(
            rfq_id=rfq_id,
            quoter=_make_party(4),
            quoter_member_id="M4",
            side=OrderSide.BUY,
            quantity=Decimal("40"),
            price=Decimal("101"),
            priority=0,
            metadata={"round": 2},
        ),
    ]

    for quote in [*round_one, *round_two]:
        manager.submit_quote(rfq_id, quote)

    _close_quotes(manager, rfq_id)

    result = manager.execute_auction(rfq_id)

    assert result.auction_type is AuctionType.MULTI_ROUND
    assert result.total_quantity_filled == Decimal("90")

    rounds = result.metadata.get("rounds")
    assert rounds is not None
    assert len(rounds) == 2
    assert rounds[0]["round"] == 1
    assert rounds[1]["round"] == 2

    rfq = manager.get_rfq(rfq_id)
    assert rfq is not None
    assert rfq.status == RFQStatus.EXECUTED
    history_statuses = [entry["status"].value for entry in rfq.status_history]
    assert history_statuses[-1] == "executed"
    assert "closed" in history_statuses

