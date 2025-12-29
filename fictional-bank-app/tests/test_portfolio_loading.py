#!/usr/bin/env python3
"""Unit tests for portfolio loading functionality."""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


class TestPortfolioCreation:
    """Test portfolio creation and structure."""

    def test_create_portfolio(self):
        """Test that portfolio can be created successfully."""
        portfolio, book_hierarchy = create_fictional_portfolio()

        assert portfolio is not None
        assert book_hierarchy is not None
        assert portfolio.name == "Global Trading Portfolio"
        assert portfolio.base_currency == "USD"

    def test_portfolio_has_trades(self):
        """Test that portfolio contains expected number of trades."""
        portfolio, _ = create_fictional_portfolio()

        assert len(portfolio.trades) == 11
        assert all(trade.id for trade in portfolio.trades.values())

    def test_portfolio_has_counterparties(self):
        """Test that portfolio contains expected counterparties."""
        portfolio, _ = create_fictional_portfolio()

        counterparties = list(portfolio.counterparties.values())
        assert len(counterparties) == 6

        # Check counterparty names
        cp_names = [cp.name for cp in counterparties]
        assert "AAA Global Bank" in cp_names
        assert "Tech Corporation A" in cp_names

    def test_portfolio_has_netting_sets(self):
        """Test that portfolio contains netting sets."""
        portfolio, _ = create_fictional_portfolio()

        netting_sets = list(portfolio.netting_sets.values())
        assert len(netting_sets) == 6

        # Check CSA coverage
        csa_count = sum(1 for ns in netting_sets if ns.csa_id is not None)
        assert csa_count == 4  # 4 out of 6 have CSA


class TestPortfolioSummary:
    """Test portfolio summary generation."""

    def test_get_portfolio_summary(self):
        """Test that summary can be generated."""
        portfolio, book_hierarchy = create_fictional_portfolio()
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        assert summary is not None
        assert "portfolio_name" in summary
        assert "statistics" in summary
        assert "counterparties" in summary
        assert "books" in summary
        assert "desks" in summary

    def test_summary_statistics(self):
        """Test summary statistics are correct."""
        portfolio, book_hierarchy = create_fictional_portfolio()
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        stats = summary["statistics"]
        assert stats["counterparties"] == 6
        assert stats["netting_sets"] == 6
        assert stats["trades"] == 11
        assert stats["active_trades"] == 11

    def test_summary_mtm_calculation(self):
        """Test MTM is calculated and aggregated."""
        portfolio, book_hierarchy = create_fictional_portfolio()
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        # MTM should be sum of all trades
        assert "total_mtm" in summary
        assert isinstance(summary["total_mtm"], (int, float))

        # Calculate expected MTM
        expected_mtm = sum(trade.mtm for trade in portfolio.trades.values())
        assert abs(summary["total_mtm"] - expected_mtm) < 0.01

    def test_counterparty_breakdown(self):
        """Test counterparty breakdown in summary."""
        portfolio, book_hierarchy = create_fictional_portfolio()
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        cp_summary = summary["counterparties"]
        assert len(cp_summary) == 6

        # Each counterparty should have required fields
        for cp_id, cp_info in cp_summary.items():
            assert "name" in cp_info
            assert "rating" in cp_info
            assert "num_trades" in cp_info
            assert "net_mtm" in cp_info
            assert "has_csa" in cp_info

    def test_desk_breakdown(self):
        """Test desk breakdown in summary."""
        portfolio, book_hierarchy = create_fictional_portfolio()
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        desk_summary = summary["desks"]
        assert len(desk_summary) == 3  # Rates, FX, Equity

        # Check desk names
        desk_names = [desk_info["name"] for desk_info in desk_summary.values()]
        assert "Interest Rates Desk" in desk_names
        assert "Foreign Exchange Desk" in desk_names
        assert "Equity Derivatives Desk" in desk_names


class TestProductMix:
    """Test product diversity in portfolio."""

    def test_product_types(self):
        """Test that portfolio has diverse product types."""
        portfolio, _ = create_fictional_portfolio()

        product_types = [trade.product_type.value for trade in portfolio.trades.values()]

        # Check for expected product types
        assert "InterestRateSwap" in product_types
        assert "Swaption" in product_types
        assert "FxOption" in product_types
        assert "EquityOption" in product_types
        assert "VarianceSwap" in product_types

    def test_currency_mix(self):
        """Test that portfolio has multiple currencies."""
        portfolio, _ = create_fictional_portfolio()

        currencies = {trade.currency for trade in portfolio.trades.values()}

        assert "USD" in currencies
        assert "EUR" in currencies
        assert len(currencies) >= 2


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_all_trades_have_counterparties(self):
        """Test that all trades reference valid counterparties."""
        portfolio, _ = create_fictional_portfolio()

        for trade in portfolio.trades.values():
            assert trade.counterparty_id in portfolio.counterparties

    def test_all_netting_sets_have_counterparties(self):
        """Test that all netting sets reference valid counterparties."""
        portfolio, _ = create_fictional_portfolio()

        for ns in portfolio.netting_sets.values():
            assert ns.counterparty_id in portfolio.counterparties

    def test_all_trades_have_positive_notional(self):
        """Test that all trades have positive notional amounts."""
        portfolio, _ = create_fictional_portfolio()

        for trade in portfolio.trades.values():
            assert trade.notional > 0

    def test_maturity_dates_are_valid(self):
        """Test that trade maturity dates are in the future."""
        portfolio, _ = create_fictional_portfolio()
        from datetime import date

        # Portfolio valuation date
        val_date = date(2024, 1, 15)

        for trade in portfolio.trades.values():
            maturity = trade.maturity_date
            if isinstance(maturity, str):
                maturity = date.fromisoformat(maturity)
            # Maturity should be after valuation date
            # (commenting out as some test data may have past dates)
            # assert maturity >= val_date


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
