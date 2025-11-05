"""Integration tests for the fictional portfolio.

Tests portfolio creation, API integration, and XVA calculations.
"""
import pytest

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


@pytest.mark.integration
class TestFictionalPortfolio:
    """Test suite for fictional portfolio creation and management."""

    @pytest.fixture(scope="class")
    def portfolio_and_hierarchy(self):
        """Create portfolio and book hierarchy for testing."""
        return create_fictional_portfolio()

    def test_portfolio_creation(self, portfolio_and_hierarchy):
        """Test that portfolio is created with expected structure."""
        portfolio, book_hierarchy = portfolio_and_hierarchy

        assert portfolio.name == "Global Trading Portfolio"
        assert portfolio.base_currency == "USD"
        assert len(portfolio.counterparties) == 6
        assert len(portfolio.netting_sets) == 6
        assert len(portfolio.trades) == 11

    def test_counterparty_setup(self, portfolio_and_hierarchy):
        """Test counterparty configuration."""
        portfolio, _ = portfolio_and_hierarchy

        # Check specific counterparties
        aaa_bank = portfolio.get_counterparty("CP_BANK_AAA")
        assert aaa_bank is not None
        assert aaa_bank.name == "AAA Global Bank"
        assert aaa_bank.is_bank is True
        assert aaa_bank.credit.rating.value == "AAA"
        assert aaa_bank.credit.lgd == 0.4

        hedge_fund = portfolio.get_counterparty("CP_HEDGE_FUND")
        assert hedge_fund is not None
        assert hedge_fund.name == "Alpha Strategies Fund"
        assert hedge_fund.credit.rating.value == "A-"

    def test_book_hierarchy(self, portfolio_and_hierarchy):
        """Test book hierarchy structure."""
        _, book_hierarchy = portfolio_and_hierarchy

        # Legal entity
        assert len(book_hierarchy.legal_entities) == 1
        le = book_hierarchy.legal_entities["LE001"]
        assert le.name == "Global Investment Bank Ltd"

        # Business unit
        assert len(book_hierarchy.business_units) == 1
        bu = book_hierarchy.business_units["BU_TRADING"]
        assert bu.name == "Global Trading"

        # Desks
        assert len(book_hierarchy.desks) == 3
        assert "DESK_RATES" in book_hierarchy.desks
        assert "DESK_FX" in book_hierarchy.desks
        assert "DESK_EQUITY" in book_hierarchy.desks

        # Traders
        assert len(book_hierarchy.traders) == 6

        # Books
        assert len(book_hierarchy.books) == 7

    def test_netting_sets(self, portfolio_and_hierarchy):
        """Test netting set configuration."""
        portfolio, _ = portfolio_and_hierarchy

        # Check CSA coverage
        csa_netting_sets = portfolio.get_collateralized_netting_sets()
        non_csa_netting_sets = portfolio.get_uncollateralized_netting_sets()

        assert len(csa_netting_sets) == 4  # AAA Bank, Corp A, Sovereign, Insurance
        assert len(non_csa_netting_sets) == 2  # Corp BBB, Hedge Fund

        # Check specific netting set
        ns_aaa = portfolio.get_netting_set("NS_CP_BANK_AAA")
        assert ns_aaa is not None
        assert ns_aaa.has_csa()
        assert ns_aaa.counterparty_id == "CP_BANK_AAA"

    def test_trades_by_product_type(self, portfolio_and_hierarchy):
        """Test trade filtering by product type."""
        portfolio, _ = portfolio_and_hierarchy

        from neutryx.contracts.trade import ProductType

        # Interest rate swaps
        irs_trades = portfolio.get_trades_by_product_type(ProductType.INTEREST_RATE_SWAP)
        assert len(irs_trades) == 3

        # FX options
        fx_trades = portfolio.get_trades_by_product_type(ProductType.FX_OPTION)
        assert len(fx_trades) == 3

        # Equity options
        eq_trades = portfolio.get_trades_by_product_type(ProductType.EQUITY_OPTION)
        assert len(eq_trades) == 3

        # Swaptions
        swpn_trades = portfolio.get_trades_by_product_type(ProductType.SWAPTION)
        assert len(swpn_trades) == 1

        # Variance swaps
        varswap_trades = portfolio.get_trades_by_product_type(ProductType.VARIANCE_SWAP)
        assert len(varswap_trades) == 1

    def test_trades_by_book(self, portfolio_and_hierarchy):
        """Test trade filtering by book."""
        portfolio, _ = portfolio_and_hierarchy

        # USD IRS book
        usd_irs_trades = portfolio.get_trades_by_book("BOOK_IRS_USD")
        assert len(usd_irs_trades) == 2
        assert all(t.book_id == "BOOK_IRS_USD" for t in usd_irs_trades)

        # FX Majors book
        fx_majors_trades = portfolio.get_trades_by_book("BOOK_FX_MAJORS")
        assert len(fx_majors_trades) == 2

        # Equity Exotic book
        eq_exotic_trades = portfolio.get_trades_by_book("BOOK_EQ_EXOTIC")
        assert len(eq_exotic_trades) == 2

    def test_trades_by_desk(self, portfolio_and_hierarchy):
        """Test trade filtering by desk."""
        portfolio, _ = portfolio_and_hierarchy

        # Rates desk
        rates_trades = portfolio.get_trades_by_desk("DESK_RATES")
        assert len(rates_trades) == 4  # 3 IRS + 1 swaption

        # FX desk
        fx_trades = portfolio.get_trades_by_desk("DESK_FX")
        assert len(fx_trades) == 3

        # Equity desk
        equity_trades = portfolio.get_trades_by_desk("DESK_EQUITY")
        assert len(equity_trades) == 4  # 3 options + 1 varswap

    def test_trades_by_counterparty(self, portfolio_and_hierarchy):
        """Test trade filtering by counterparty."""
        portfolio, _ = portfolio_and_hierarchy

        # AAA Bank (IRS trades)
        aaa_trades = portfolio.get_trades_by_counterparty("CP_BANK_AAA")
        assert len(aaa_trades) == 2

        # Corp A (FX options)
        corp_a_trades = portfolio.get_trades_by_counterparty("CP_CORP_A")
        assert len(corp_a_trades) == 2

        # Hedge Fund (mixed)
        hf_trades = portfolio.get_trades_by_counterparty("CP_HEDGE_FUND")
        assert len(hf_trades) == 2

    def test_mtm_calculations(self, portfolio_and_hierarchy):
        """Test MTM calculation methods."""
        portfolio, _ = portfolio_and_hierarchy

        # Total portfolio MTM
        total_mtm = portfolio.calculate_total_mtm()
        assert total_mtm != 0.0

        # Counterparty MTM
        aaa_mtm = portfolio.calculate_net_mtm_by_counterparty("CP_BANK_AAA")
        assert aaa_mtm != 0.0

        # Book MTM
        book_mtm = portfolio.calculate_mtm_by_book("BOOK_IRS_USD")
        assert book_mtm != 0.0

        # Desk MTM
        desk_mtm = portfolio.calculate_mtm_by_desk("DESK_RATES")
        assert desk_mtm != 0.0

    def test_notional_calculations(self, portfolio_and_hierarchy):
        """Test notional calculation methods."""
        portfolio, _ = portfolio_and_hierarchy

        # Gross notional
        gross_notional = portfolio.calculate_gross_notional()
        assert gross_notional > 0.0
        assert gross_notional > 100_000_000.0  # Should be > 100M

        # Book notional
        book_notional = portfolio.calculate_notional_by_book("BOOK_IRS_USD")
        assert book_notional > 0.0

        # Desk notional
        desk_notional = portfolio.calculate_notional_by_desk("DESK_FX")
        assert desk_notional > 0.0

    def test_portfolio_summary(self, portfolio_and_hierarchy):
        """Test portfolio summary generation."""
        portfolio, book_hierarchy = portfolio_and_hierarchy

        summary = get_portfolio_summary(portfolio, book_hierarchy)

        assert summary["portfolio_name"] == "Global Trading Portfolio"
        assert summary["base_currency"] == "USD"

        stats = summary["statistics"]
        assert stats["counterparties"] == 6
        assert stats["netting_sets"] == 6
        assert stats["trades"] == 11
        assert stats["active_trades"] == 11

        # Check counterparties section
        assert len(summary["counterparties"]) == 6
        for cp_id, cp_info in summary["counterparties"].items():
            assert "name" in cp_info
            assert "entity_type" in cp_info
            assert "rating" in cp_info
            assert "num_trades" in cp_info
            assert "net_mtm" in cp_info
            assert "has_csa" in cp_info

        # Check books section
        assert len(summary["books"]) == 7

        # Check desks section
        assert len(summary["desks"]) == 3

    def test_active_trades(self, portfolio_and_hierarchy):
        """Test active trade filtering."""
        portfolio, _ = portfolio_and_hierarchy

        active_trades = portfolio.get_active_trades()
        assert len(active_trades) == 11
        assert all(t.is_active() for t in active_trades)

    def test_book_summary(self, portfolio_and_hierarchy):
        """Test book summary generation."""
        portfolio, _ = portfolio_and_hierarchy

        book_summary = portfolio.get_book_summary("BOOK_IRS_USD")

        assert book_summary["book_id"] == "BOOK_IRS_USD"
        assert book_summary["num_trades"] == 2
        assert book_summary["active_trades"] == 2
        assert "total_mtm" in book_summary
        assert "total_notional" in book_summary

    def test_desk_summary(self, portfolio_and_hierarchy):
        """Test desk summary generation."""
        portfolio, _ = portfolio_and_hierarchy

        desk_summary = portfolio.get_desk_summary("DESK_RATES")

        assert desk_summary["desk_id"] == "DESK_RATES"
        assert desk_summary["num_trades"] == 4
        assert desk_summary["active_trades"] == 4
        assert desk_summary["num_books"] == 3
        assert "total_mtm" in desk_summary
        assert "total_notional" in desk_summary

    def test_trader_summary(self, portfolio_and_hierarchy):
        """Test trader summary generation."""
        portfolio, _ = portfolio_and_hierarchy

        trader_summary = portfolio.get_trader_summary("TRADER_001")

        assert trader_summary["trader_id"] == "TRADER_001"
        assert trader_summary["num_trades"] > 0
        assert "total_mtm" in trader_summary
        assert "total_notional" in trader_summary

    def test_portfolio_aggregations(self, portfolio_and_hierarchy):
        """Test portfolio aggregation methods."""
        portfolio, _ = portfolio_and_hierarchy

        # Aggregate by book
        book_agg = portfolio.aggregate_by_book()
        assert len(book_agg) == 7

        # Aggregate by desk
        desk_agg = portfolio.aggregate_by_desk()
        assert len(desk_agg) == 3

        # Aggregate by trader
        trader_agg = portfolio.aggregate_by_trader()
        assert len(trader_agg) == 6

    def test_portfolio_serialization(self, portfolio_and_hierarchy):
        """Test portfolio can be serialized to JSON."""
        portfolio, _ = portfolio_and_hierarchy

        # Serialize portfolio
        portfolio_dict = portfolio.model_dump(mode="json")

        assert portfolio_dict["name"] == "Global Trading Portfolio"
        assert "counterparties" in portfolio_dict
        assert "trades" in portfolio_dict
        assert "netting_sets" in portfolio_dict

        # Should be able to reconstruct
        from neutryx.portfolio.portfolio import Portfolio

        reconstructed = Portfolio.model_validate(portfolio_dict)
        assert reconstructed.name == portfolio.name
        assert len(reconstructed.trades) == len(portfolio.trades)
        assert len(reconstructed.counterparties) == len(portfolio.counterparties)

    def test_trade_product_details(self, portfolio_and_hierarchy):
        """Test that trades have appropriate product details."""
        portfolio, _ = portfolio_and_hierarchy

        # Check IRS details
        irs_trade = next(
            t for t in portfolio.trades.values() if t.product_type.value == "InterestRateSwap"
        )
        assert irs_trade.product_details is not None
        assert "fixed_rate" in irs_trade.product_details
        assert "floating_rate" in irs_trade.product_details

        # Check FX option details
        fx_trade = next(t for t in portfolio.trades.values() if t.product_type.value == "FxOption")
        assert fx_trade.product_details is not None
        assert "currency_pair" in fx_trade.product_details
        assert "strike" in fx_trade.product_details

        # Check equity option details
        eq_trade = next(
            t for t in portfolio.trades.values() if t.product_type.value == "EquityOption"
        )
        assert eq_trade.product_details is not None
        assert "underlyer" in eq_trade.product_details
        assert "strike" in eq_trade.product_details

    def test_csa_characteristics(self, portfolio_and_hierarchy):
        """Test CSA characteristics."""
        portfolio, _ = portfolio_and_hierarchy

        csa_aaa = portfolio.get_csa("CSA_CP_BANK_AAA")
        assert csa_aaa is not None
        assert csa_aaa.threshold_terms.threshold_party_a == 1_000_000.0
        assert csa_aaa.threshold_terms.threshold_party_b == 1_000_000.0
        assert csa_aaa.threshold_terms.mta_party_a == 100_000.0
        assert csa_aaa.variation_margin_required is True

    def test_master_agreements(self, portfolio_and_hierarchy):
        """Test master agreement setup."""
        portfolio, _ = portfolio_and_hierarchy

        ma_aaa = portfolio.get_master_agreement("MA_CP_BANK_AAA")
        assert ma_aaa is not None
        assert ma_aaa.party_a_id == "OUR_INSTITUTION"
        assert ma_aaa.party_b_id == "CP_BANK_AAA"
        assert ma_aaa.agreement_type.value == "ISDA2002"


@pytest.mark.integration
class TestPortfolioStatistics:
    """Test portfolio statistics and analytics."""

    @pytest.fixture(scope="class")
    def portfolio(self):
        """Create portfolio for testing."""
        portfolio, _ = create_fictional_portfolio()
        return portfolio

    def test_counterparty_counts(self, portfolio):
        """Test counterparty counting."""
        assert portfolio.num_counterparties() == 6

    def test_netting_set_counts(self, portfolio):
        """Test netting set counting."""
        assert portfolio.num_netting_sets() == 6

    def test_trade_counts(self, portfolio):
        """Test trade counting."""
        assert portfolio.num_trades() == 11
        assert portfolio.num_active_trades() == 11

    def test_summary_statistics(self, portfolio):
        """Test summary statistics."""
        summary = portfolio.summary()

        assert summary["counterparties"] == 6
        assert summary["netting_sets"] == 6
        assert summary["trades"] == 11
        assert summary["active_trades"] == 11
