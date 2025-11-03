"""Tests for Portfolio class and hierarchy management."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from neutryx.contracts.counterparty import Counterparty, EntityType
from neutryx.contracts.csa import CSA, CollateralTerms, EligibleCollateral, CollateralType
from neutryx.contracts.master_agreement import MasterAgreement, AgreementType
from neutryx.contracts.trade import Trade, ProductType, TradeStatus
from neutryx.portfolio.netting_set import NettingSet
from neutryx.portfolio.portfolio import Portfolio


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio with counterparties, agreements, and trades."""
    portfolio = Portfolio(name="Test Portfolio", base_currency="USD")

    # Add counterparties
    cp1 = Counterparty(id="CP001", name="Bank ABC", entity_type=EntityType.FINANCIAL)
    cp2 = Counterparty(id="CP002", name="Corp XYZ", entity_type=EntityType.CORPORATE)
    portfolio.add_counterparty(cp1)
    portfolio.add_counterparty(cp2)

    # Add master agreements
    ma1 = MasterAgreement(
        id="MA001",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="OWN",
        party_b_id="CP001",
        effective_date=date(2020, 1, 1),
    )
    portfolio.add_master_agreement(ma1)

    # Add CSA
    csa1 = CSA(
        id="CSA001",
        party_a_id="OWN",
        party_b_id="CP001",
        effective_date="2020-01-01",
        collateral_terms=CollateralTerms(
            base_currency="USD",
            eligible_collateral=[
                EligibleCollateral(collateral_type=CollateralType.CASH, currency="USD", haircut=0.0)
            ],
        ),
    )
    portfolio.add_csa(csa1)

    # Add netting sets
    ns1 = NettingSet(
        id="NS001",
        master_agreement_id="MA001",
        counterparty_id="CP001",
        csa_id="CSA001",
    )
    ns2 = NettingSet(
        id="NS002",
        master_agreement_id="MA001",
        counterparty_id="CP002",
    )
    portfolio.add_netting_set(ns1)
    portfolio.add_netting_set(ns2)

    # Add trades
    trade1 = Trade(
        id="TRD001",
        counterparty_id="CP001",
        netting_set_id="NS001",
        product_type=ProductType.EQUITY_OPTION,
        trade_date=date(2024, 1, 1),
        maturity_date=date(2025, 1, 1),
        notional=1_000_000.0,
        currency="USD",
        mtm=50_000.0,
    )
    trade2 = Trade(
        id="TRD002",
        counterparty_id="CP001",
        netting_set_id="NS001",
        product_type=ProductType.INTEREST_RATE_SWAP,
        trade_date=date(2024, 1, 1),
        maturity_date=date(2034, 1, 1),
        notional=10_000_000.0,
        currency="USD",
        mtm=-30_000.0,
    )
    trade3 = Trade(
        id="TRD003",
        counterparty_id="CP002",
        netting_set_id="NS002",
        product_type=ProductType.FX_OPTION,
        trade_date=date(2024, 1, 1),
        maturity_date=date(2024, 12, 31),
        notional=5_000_000.0,
        currency="EUR",
        mtm=100_000.0,
    )
    portfolio.add_trade(trade1)
    portfolio.add_trade(trade2)
    portfolio.add_trade(trade3)

    return portfolio


def test_portfolio_creation():
    """Test basic portfolio creation."""
    portfolio = Portfolio(name="My Portfolio")
    assert portfolio.name == "My Portfolio"
    assert portfolio.base_currency == "USD"
    assert portfolio.num_counterparties() == 0
    assert portfolio.num_trades() == 0


def test_portfolio_add_entities(sample_portfolio):
    """Test adding entities to portfolio."""
    assert sample_portfolio.num_counterparties() == 2
    assert sample_portfolio.num_netting_sets() == 2
    assert sample_portfolio.num_trades() == 3


def test_portfolio_get_entities(sample_portfolio):
    """Test retrieving entities from portfolio."""
    cp = sample_portfolio.get_counterparty("CP001")
    assert cp is not None
    assert cp.name == "Bank ABC"

    ns = sample_portfolio.get_netting_set("NS001")
    assert ns is not None
    assert ns.counterparty_id == "CP001"

    trade = sample_portfolio.get_trade("TRD001")
    assert trade is not None
    assert trade.product_type == ProductType.EQUITY_OPTION


def test_portfolio_trade_auto_adds_to_netting_set(sample_portfolio):
    """Test that trades are automatically added to their netting set."""
    ns = sample_portfolio.get_netting_set("NS001")
    assert ns.num_trades() == 2
    assert ns.contains_trade("TRD001")
    assert ns.contains_trade("TRD002")


def test_portfolio_remove_trade(sample_portfolio):
    """Test removing trades from portfolio."""
    # Check initial state
    assert sample_portfolio.num_trades() == 3
    ns = sample_portfolio.get_netting_set("NS001")
    assert ns.num_trades() == 2

    # Remove trade
    removed = sample_portfolio.remove_trade("TRD001")
    assert removed
    assert sample_portfolio.num_trades() == 2
    assert sample_portfolio.get_trade("TRD001") is None

    # Verify it's removed from netting set
    assert ns.num_trades() == 1
    assert not ns.contains_trade("TRD001")

    # Remove non-existent trade
    removed_none = sample_portfolio.remove_trade("TRD999")
    assert not removed_none


def test_portfolio_get_trades_by_counterparty(sample_portfolio):
    """Test filtering trades by counterparty."""
    trades_cp1 = sample_portfolio.get_trades_by_counterparty("CP001")
    assert len(trades_cp1) == 2
    assert all(t.counterparty_id == "CP001" for t in trades_cp1)

    trades_cp2 = sample_portfolio.get_trades_by_counterparty("CP002")
    assert len(trades_cp2) == 1
    assert trades_cp2[0].id == "TRD003"


def test_portfolio_get_trades_by_netting_set(sample_portfolio):
    """Test filtering trades by netting set."""
    trades_ns1 = sample_portfolio.get_trades_by_netting_set("NS001")
    assert len(trades_ns1) == 2
    assert all(t.netting_set_id == "NS001" for t in trades_ns1)


def test_portfolio_get_trades_by_product_type(sample_portfolio):
    """Test filtering trades by product type."""
    equity_opts = sample_portfolio.get_trades_by_product_type(ProductType.EQUITY_OPTION)
    assert len(equity_opts) == 1
    assert equity_opts[0].id == "TRD001"

    irs_trades = sample_portfolio.get_trades_by_product_type(ProductType.INTEREST_RATE_SWAP)
    assert len(irs_trades) == 1
    assert irs_trades[0].id == "TRD002"


def test_portfolio_get_active_trades(sample_portfolio):
    """Test filtering active trades."""
    active = sample_portfolio.get_active_trades()
    assert len(active) == 3

    # Terminate one trade
    trade = sample_portfolio.get_trade("TRD001")
    trade.status = TradeStatus.TERMINATED

    active_after = sample_portfolio.get_active_trades()
    assert len(active_after) == 2


def test_portfolio_get_netting_sets_by_counterparty(sample_portfolio):
    """Test getting netting sets by counterparty."""
    ns_cp1 = sample_portfolio.get_netting_sets_by_counterparty("CP001")
    assert len(ns_cp1) == 1
    assert ns_cp1[0].id == "NS001"


def test_portfolio_get_collateralized_netting_sets(sample_portfolio):
    """Test filtering collateralized netting sets."""
    collateralized = sample_portfolio.get_collateralized_netting_sets()
    assert len(collateralized) == 1
    assert collateralized[0].id == "NS001"


def test_portfolio_get_uncollateralized_netting_sets(sample_portfolio):
    """Test filtering uncollateralized netting sets."""
    uncollateralized = sample_portfolio.get_uncollateralized_netting_sets()
    assert len(uncollateralized) == 1
    assert uncollateralized[0].id == "NS002"


def test_portfolio_calculate_net_mtm_by_netting_set(sample_portfolio):
    """Test calculating net MTM by netting set."""
    # NS001: TRD001 (50K) + TRD002 (-30K) = 20K
    net_mtm = sample_portfolio.calculate_net_mtm_by_netting_set("NS001")
    assert net_mtm == 20_000.0

    # NS002: TRD003 (100K)
    net_mtm_ns2 = sample_portfolio.calculate_net_mtm_by_netting_set("NS002")
    assert net_mtm_ns2 == 100_000.0


def test_portfolio_calculate_net_mtm_by_counterparty(sample_portfolio):
    """Test calculating net MTM by counterparty."""
    # CP001: TRD001 (50K) + TRD002 (-30K) = 20K
    net_mtm_cp1 = sample_portfolio.calculate_net_mtm_by_counterparty("CP001")
    assert net_mtm_cp1 == 20_000.0

    # CP002: TRD003 (100K)
    net_mtm_cp2 = sample_portfolio.calculate_net_mtm_by_counterparty("CP002")
    assert net_mtm_cp2 == 100_000.0


def test_portfolio_calculate_total_mtm(sample_portfolio):
    """Test calculating total portfolio MTM."""
    # 50K - 30K + 100K = 120K
    total_mtm = sample_portfolio.calculate_total_mtm()
    assert total_mtm == 120_000.0


def test_portfolio_calculate_gross_notional(sample_portfolio):
    """Test calculating gross notional."""
    # 1M + 10M + 5M = 16M
    gross_notional = sample_portfolio.calculate_gross_notional()
    assert gross_notional == 16_000_000.0


def test_portfolio_summary(sample_portfolio):
    """Test portfolio summary statistics."""
    summary = sample_portfolio.summary()
    assert summary["counterparties"] == 2
    assert summary["netting_sets"] == 2
    assert summary["trades"] == 3
    assert summary["active_trades"] == 3


def test_portfolio_get_trades_maturing_before(sample_portfolio):
    """Test filtering trades maturing before a date."""
    cutoff = date(2025, 6, 1)
    maturing = sample_portfolio.get_trades_maturing_before(cutoff)
    # TRD003 matures 2024-12-31, TRD001 matures 2025-01-01
    assert len(maturing) == 2


def test_portfolio_get_trades_maturing_in_range(sample_portfolio):
    """Test filtering trades maturing in a date range."""
    start = date(2024, 6, 1)
    end = date(2025, 6, 1)
    maturing = sample_portfolio.get_trades_maturing_in_range(start, end)
    # TRD003 (2024-12-31) and TRD001 (2025-01-01)
    assert len(maturing) == 2


def test_portfolio_empty_filters():
    """Test filters on empty portfolio."""
    portfolio = Portfolio(name="Empty")

    assert len(portfolio.get_active_trades()) == 0
    assert len(portfolio.get_trades_by_counterparty("CP001")) == 0
    assert len(portfolio.get_collateralized_netting_sets()) == 0
    assert portfolio.calculate_total_mtm() == 0.0
    assert portfolio.calculate_gross_notional() == 0.0


def test_portfolio_repr(sample_portfolio):
    """Test string representation."""
    repr_str = repr(sample_portfolio)
    assert "Test Portfolio" in repr_str
    assert "counterparties=2" in repr_str
    assert "netting_sets=2" in repr_str
    assert "trades=3" in repr_str


def test_portfolio_with_base_currency():
    """Test portfolio with custom base currency."""
    portfolio = Portfolio(name="EUR Portfolio", base_currency="EUR")
    assert portfolio.base_currency == "EUR"


def test_portfolio_num_active_trades(sample_portfolio):
    """Test counting active trades."""
    assert sample_portfolio.num_active_trades() == 3

    # Terminate a trade
    trade = sample_portfolio.get_trade("TRD001")
    trade.status = TradeStatus.MATURED

    assert sample_portfolio.num_active_trades() == 2
