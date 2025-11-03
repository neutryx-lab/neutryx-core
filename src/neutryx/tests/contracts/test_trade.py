"""Tests for Trade models."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from neutryx.contracts.trade import (
    ProductType,
    SettlementType,
    Trade,
    TradeStatus,
)


def test_trade_basic():
    """Test basic Trade creation."""
    trade = Trade(
        id="TRD001",
        counterparty_id="CP001",
        product_type=ProductType.EQUITY_OPTION,
        trade_date=date(2024, 1, 15),
        maturity_date=date(2025, 1, 15),
        notional=1_000_000.0,
        currency="USD",
    )
    assert trade.id == "TRD001"
    assert trade.counterparty_id == "CP001"
    assert trade.product_type == ProductType.EQUITY_OPTION
    assert trade.status == TradeStatus.ACTIVE
    assert trade.is_active()


def test_trade_with_netting_set():
    """Test Trade with netting set assignment."""
    trade = Trade(
        id="TRD002",
        counterparty_id="CP001",
        netting_set_id="NS001",
        product_type=ProductType.INTEREST_RATE_SWAP,
        trade_date=date(2024, 1, 1),
    )
    assert trade.netting_set_id == "NS001"
    assert trade.belongs_to_netting_set("NS001")
    assert not trade.belongs_to_netting_set("NS002")


def test_trade_with_external_ids():
    """Test Trade with external identifiers."""
    trade = Trade(
        id="TRD003",
        external_id="EXT12345",
        usi="USI-ABC-123456",
        counterparty_id="CP001",
        product_type=ProductType.CREDIT_DEFAULT_SWAP,
        trade_date=date(2024, 1, 1),
    )
    assert trade.external_id == "EXT12345"
    assert trade.usi == "USI-ABC-123456"


def test_trade_effective_date():
    """Test effective date handling."""
    # With explicit effective date
    trade1 = Trade(
        id="TRD004",
        counterparty_id="CP001",
        product_type=ProductType.FX_OPTION,
        trade_date=date(2024, 1, 15),
        effective_date=date(2024, 1, 17),
    )
    assert trade1.get_effective_date() == date(2024, 1, 17)

    # Without effective date (fallback to trade date)
    trade2 = Trade(
        id="TRD005",
        counterparty_id="CP001",
        product_type=ProductType.FX_OPTION,
        trade_date=date(2024, 1, 15),
    )
    assert trade2.get_effective_date() == date(2024, 1, 15)


def test_trade_is_expired():
    """Test is_expired method."""
    trade = Trade(
        id="TRD006",
        counterparty_id="CP001",
        product_type=ProductType.SWAPTION,
        trade_date=date(2024, 1, 1),
        maturity_date=date(2024, 12, 31),
    )

    assert not trade.is_expired(date(2024, 6, 1))
    assert not trade.is_expired(date(2024, 12, 31))
    assert trade.is_expired(date(2025, 1, 1))


def test_trade_time_to_maturity():
    """Test time_to_maturity calculation."""
    trade = Trade(
        id="TRD007",
        counterparty_id="CP001",
        product_type=ProductType.FORWARD,
        trade_date=date(2024, 1, 1),
        maturity_date=date(2025, 1, 1),
    )

    # Exactly 1 year
    ttm = trade.time_to_maturity(date(2024, 1, 1))
    assert abs(ttm - 1.0) < 0.01  # Allow small rounding error

    # 6 months
    ttm_6m = trade.time_to_maturity(date(2024, 7, 2))
    assert 0.4 < ttm_6m < 0.6

    # After maturity
    ttm_past = trade.time_to_maturity(date(2025, 6, 1))
    assert ttm_past == 0.0

    # No maturity date
    trade_no_mat = Trade(
        id="TRD008",
        counterparty_id="CP001",
        product_type=ProductType.OTHER,
        trade_date=date(2024, 1, 1),
    )
    assert trade_no_mat.time_to_maturity(date(2024, 6, 1)) is None


def test_trade_mtm_handling():
    """Test MTM value handling."""
    trade = Trade(
        id="TRD009",
        counterparty_id="CP001",
        product_type=ProductType.VARIANCE_SWAP,
        trade_date=date(2024, 1, 1),
    )

    # Initially no MTM
    assert trade.mtm is None
    assert trade.get_mtm(default=0.0) == 0.0
    assert trade.get_mtm(default=100.0) == 100.0

    # Update MTM
    trade.update_mtm(mtm_value=50_000.0, valuation_date=date(2024, 6, 1))
    assert trade.mtm == 50_000.0
    assert trade.last_valuation_date == date(2024, 6, 1)
    assert trade.get_mtm() == 50_000.0


def test_trade_status_lifecycle():
    """Test trade status transitions."""
    trade = Trade(
        id="TRD010",
        counterparty_id="CP001",
        product_type=ProductType.FUTURE,
        trade_date=date(2024, 1, 1),
        status=TradeStatus.PENDING,
    )
    assert not trade.is_active()

    # Activate trade
    trade.status = TradeStatus.ACTIVE
    assert trade.is_active()

    # Terminate trade
    trade.status = TradeStatus.TERMINATED
    assert not trade.is_active()


def test_trade_with_product_details():
    """Test Trade with product-specific details."""
    product_details = {
        "option_type": "Call",
        "strike": 100.0,
        "barrier": 120.0,
        "underlying": "SPX",
    }
    trade = Trade(
        id="TRD011",
        counterparty_id="CP001",
        product_type=ProductType.EQUITY_OPTION,
        trade_date=date(2024, 1, 1),
        product_details=product_details,
    )
    assert trade.product_details is not None
    assert trade.product_details["option_type"] == "Call"
    assert trade.product_details["strike"] == 100.0


def test_trade_settlement_type():
    """Test settlement type specification."""
    trade_cash = Trade(
        id="TRD012",
        counterparty_id="CP001",
        product_type=ProductType.FX_OPTION,
        trade_date=date(2024, 1, 1),
        settlement_type=SettlementType.CASH,
    )
    assert trade_cash.settlement_type == SettlementType.CASH

    trade_physical = Trade(
        id="TRD013",
        counterparty_id="CP001",
        product_type=ProductType.FX_OPTION,
        trade_date=date(2024, 1, 1),
        settlement_type=SettlementType.PHYSICAL,
    )
    assert trade_physical.settlement_type == SettlementType.PHYSICAL


def test_trade_repr():
    """Test string representation."""
    trade = Trade(
        id="TRD014",
        counterparty_id="CP001",
        product_type=ProductType.INTEREST_RATE_SWAP,
        trade_date=date(2024, 1, 1),
        maturity_date=date(2034, 1, 1),
        notional=10_000_000.0,
        currency="USD",
    )
    repr_str = repr(trade)
    assert "TRD014" in repr_str
    assert "InterestRateSwap" in repr_str
    assert "CP001" in repr_str
    assert "2034-01-01" in repr_str


def test_product_type_enum():
    """Test ProductType enum values."""
    assert ProductType.EQUITY_OPTION.value == "EquityOption"
    assert ProductType.INTEREST_RATE_SWAP.value == "InterestRateSwap"
    assert ProductType.CREDIT_DEFAULT_SWAP.value == "CreditDefaultSwap"
    assert ProductType.SWAPTION.value == "Swaption"


def test_trade_status_enum():
    """Test TradeStatus enum values."""
    assert TradeStatus.PENDING.value == "Pending"
    assert TradeStatus.ACTIVE.value == "Active"
    assert TradeStatus.TERMINATED.value == "Terminated"
    assert TradeStatus.MATURED.value == "Matured"
    assert TradeStatus.NOVATED.value == "Novated"


def test_settlement_type_enum():
    """Test SettlementType enum values."""
    assert SettlementType.CASH.value == "Cash"
    assert SettlementType.PHYSICAL.value == "Physical"
    assert SettlementType.ELECTION.value == "Election"
