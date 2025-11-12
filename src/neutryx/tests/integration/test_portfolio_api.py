"""Integration tests for portfolio API endpoints.

Tests the REST API for portfolio management and XVA calculations.
"""

import asyncio
from typing import Any

import pytest
import httpx
from fastapi.testclient import TestClient

from neutryx.api.rest import create_app
from neutryx.api.portfolio_store import InMemoryPortfolioStore
from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio


@pytest.fixture
def client():
    """Create test client for API."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def portfolio():
    """Create fictional portfolio for testing."""
    portfolio, _ = create_fictional_portfolio()
    return portfolio


@pytest.fixture
def file_store_env(tmp_path, monkeypatch):
    """Configure environment variables for filesystem store persistence tests."""
    path = tmp_path / "portfolios.json"
    monkeypatch.setenv("NEUTRYX_PORTFOLIO_STORE", "filesystem")
    monkeypatch.setenv("NEUTRYX_PORTFOLIO_STORE_PATH", str(path))
    return path


class TestPortfolioRegistration:
    """Test portfolio registration endpoints."""

    def test_register_portfolio(self, client, portfolio):
        """Test registering a portfolio with the API."""
        portfolio_data = portfolio.model_dump(mode="json")

        response = client.post("/portfolio/register", json=portfolio_data)

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "registered"
        assert result["portfolio_id"] == "Global Trading Portfolio"
        assert "summary" in result

    def test_register_invalid_portfolio(self, client):
        """Test registering invalid portfolio data."""
        # Portfolio model has defaults for most fields, so we need truly invalid data
        invalid_data = None  # Send None as invalid JSON

        response = client.post("/portfolio/register", json=invalid_data)

        # FastAPI will reject None as invalid Portfolio data
        assert response.status_code == 400 or response.status_code == 422


class TestPortfolioSummary:
    """Test portfolio summary endpoints."""

    def test_get_portfolio_summary(self, client, portfolio):
        """Test getting portfolio summary."""
        # Register portfolio first
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Get summary
        response = client.get("/portfolio/Global Trading Portfolio/summary")

        assert response.status_code == 200
        summary = response.json()

        assert summary["portfolio_id"] == "Global Trading Portfolio"
        assert summary["base_currency"] == "USD"
        # Note: The API returns counterparty details as a list, not a count
        assert len(summary["counterparties"]) == 6  # Updated to check length of list
        assert summary["netting_sets"] == 6
        assert summary["trades"] == 11
        assert "total_mtm" in summary
        assert "gross_notional" in summary
        assert "counterparties" in summary

    def test_get_nonexistent_portfolio_summary(self, client):
        """Test getting summary for non-existent portfolio."""
        response = client.get("/portfolio/NonExistent/summary")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestNettingSets:
    """Test netting set endpoints."""

    def test_list_netting_sets(self, client, portfolio):
        """Test listing netting sets."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # List netting sets
        response = client.get("/portfolio/Global Trading Portfolio/netting-sets")

        assert response.status_code == 200
        result = response.json()

        assert result["portfolio_id"] == "Global Trading Portfolio"
        assert "netting_sets" in result
        assert len(result["netting_sets"]) == 6

        # Check netting set structure
        ns = result["netting_sets"][0]
        assert "netting_set_id" in ns
        assert "counterparty_id" in ns
        assert "counterparty_name" in ns
        assert "has_csa" in ns
        assert "num_trades" in ns
        assert "net_mtm" in ns

    def test_list_netting_sets_nonexistent_portfolio(self, client):
        """Test listing netting sets for non-existent portfolio."""
        response = client.get("/portfolio/NonExistent/netting-sets")

        assert response.status_code == 404


class TestPortfolioXVA:
    """Test portfolio XVA calculation endpoints."""

    def test_compute_portfolio_level_xva(self, client, portfolio):
        """Test computing XVA for entire portfolio."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute XVA
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
            "compute_dva": True,
            "compute_fva": True,
            "compute_mva": True,
            "lgd": 0.6,
            "funding_spread_bps": 50.0,
        }

        response = client.post("/portfolio/xva", json=request)

        assert response.status_code == 200
        result = response.json()

        assert result["scope"] == "portfolio"
        assert result["num_trades"] == 11
        assert "net_mtm" in result
        assert "positive_exposure" in result
        assert "negative_exposure" in result
        assert "cva" in result
        assert "dva" in result
        assert "fva" in result
        assert "mva" in result
        assert "total_xva" in result
        assert result["valuation_date"] == "2024-01-15"

    def test_compute_netting_set_xva(self, client, portfolio):
        """Test computing XVA for specific netting set."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute XVA for specific netting set
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "netting_set_id": "NS_CP_BANK_AAA",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
            "compute_dva": True,
            "compute_fva": True,
            "compute_mva": True,
        }

        response = client.post("/portfolio/xva", json=request)

        assert response.status_code == 200
        result = response.json()

        assert "netting_set:NS_CP_BANK_AAA" in result["scope"]
        assert result["num_trades"] == 2  # AAA Bank has 2 IRS trades
        assert "cva" in result
        assert "total_xva" in result

    def test_compute_xva_cva_only(self, client, portfolio):
        """Test computing only CVA."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute only CVA
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
            "compute_dva": False,
            "compute_fva": False,
            "compute_mva": False,
        }

        response = client.post("/portfolio/xva", json=request)

        assert response.status_code == 200
        result = response.json()

        assert result["cva"] is not None
        assert result["dva"] is None
        assert result["fva"] is None
        assert result["mva"] is None

    def test_compute_xva_nonexistent_portfolio(self, client):
        """Test computing XVA for non-existent portfolio."""
        request = {
            "portfolio_id": "NonExistent",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
        }

        response = client.post("/portfolio/xva", json=request)

        assert response.status_code == 404

    def test_compute_xva_nonexistent_netting_set(self, client, portfolio):
        """Test computing XVA for non-existent netting set."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Try non-existent netting set
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "netting_set_id": "NS_NONEXISTENT",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
        }

        response = client.post("/portfolio/xva", json=request)

        assert response.status_code == 404

    def test_compute_xva_empty_netting_set(self, client):
        """Test computing XVA for empty netting set."""
        from neutryx.portfolio.portfolio import Portfolio
        from neutryx.portfolio.netting_set import NettingSet

        # Create portfolio with empty netting set
        portfolio = Portfolio(name="Empty Portfolio")
        ns = NettingSet(
            id="NS_EMPTY",
            counterparty_id="CP_TEST",
            master_agreement_id="MA_TEST",
        )
        portfolio.add_netting_set(ns)

        # Register
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute XVA
        request = {
            "portfolio_id": "Empty Portfolio",
            "netting_set_id": "NS_EMPTY",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
        }

        response = client.post("/portfolio/xva", json=request)

        assert response.status_code == 200
        result = response.json()

        assert result["num_trades"] == 0
        assert result["cva"] == 0.0
        assert result["total_xva"] == 0.0


class TestXVACalculationLogic:
    """Test XVA calculation logic and formulas."""

    def test_cva_increases_with_exposure(self, client, portfolio):
        """Test that CVA increases with positive exposure."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Get XVA
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
            "compute_dva": False,
            "compute_fva": False,
            "compute_mva": False,
            "lgd": 0.6,
        }

        response = client.post("/portfolio/xva", json=request)
        result = response.json()

        # CVA should be positive when positive exposure exists
        if result["positive_exposure"] > 0:
            assert result["cva"] > 0

    def test_fva_zero_with_csa(self, client, portfolio):
        """Test that FVA is zero for CSA netting sets."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute FVA for CSA netting set
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "netting_set_id": "NS_CP_BANK_AAA",  # Has CSA
            "valuation_date": "2024-01-15",
            "compute_cva": False,
            "compute_dva": False,
            "compute_fva": True,
            "compute_mva": False,
        }

        response = client.post("/portfolio/xva", json=request)
        result = response.json()

        # FVA should be zero for collateralized netting sets
        assert result["fva"] == 0.0

    def test_fva_nonzero_without_csa(self, client, portfolio):
        """Test that FVA is non-zero for non-CSA netting sets with exposure."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute FVA for non-CSA netting set
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "netting_set_id": "NS_CP_CORP_BBB",  # No CSA
            "valuation_date": "2024-01-15",
            "compute_cva": False,
            "compute_dva": False,
            "compute_fva": True,
            "compute_mva": False,
            "funding_spread_bps": 100.0,
        }

        response = client.post("/portfolio/xva", json=request)
        result = response.json()

        # FVA should be non-zero if positive exposure exists
        if result["positive_exposure"] > 0:
            assert result["fva"] > 0

    def test_mva_only_with_csa(self, client, portfolio):
        """Test that MVA is only computed for CSA netting sets."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute MVA for CSA netting set
        request_csa = {
            "portfolio_id": "Global Trading Portfolio",
            "netting_set_id": "NS_CP_BANK_AAA",  # Has CSA
            "valuation_date": "2024-01-15",
            "compute_cva": False,
            "compute_dva": False,
            "compute_fva": False,
            "compute_mva": True,
        }

        response_csa = client.post("/portfolio/xva", json=request_csa)
        result_csa = response_csa.json()

        # Compute MVA for non-CSA netting set
        request_no_csa = {
            "portfolio_id": "Global Trading Portfolio",
            "netting_set_id": "NS_CP_CORP_BBB",  # No CSA
            "valuation_date": "2024-01-15",
            "compute_cva": False,
            "compute_dva": False,
            "compute_fva": False,
            "compute_mva": True,
        }

        response_no_csa = client.post("/portfolio/xva", json=request_no_csa)
        result_no_csa = response_no_csa.json()

        # MVA should be positive for CSA, zero for non-CSA
        assert result_csa["mva"] > 0
        assert result_no_csa["mva"] == 0.0

    def test_dva_reduces_total_xva(self, client, portfolio):
        """Test that DVA reduces total XVA."""
        # Register portfolio
        portfolio_data = portfolio.model_dump(mode="json")
        client.post("/portfolio/register", json=portfolio_data)

        # Compute with DVA
        request = {
            "portfolio_id": "Global Trading Portfolio",
            "valuation_date": "2024-01-15",
            "compute_cva": True,
            "compute_dva": True,
            "compute_fva": False,
            "compute_mva": False,
        }

        response = client.post("/portfolio/xva", json=request)
        result = response.json()

        # DVA should reduce total XVA (it's a benefit)
        if result["dva"] and result["dva"] > 0:
            # Total XVA = CVA - DVA (DVA is subtracted)
            expected_total = result["cva"] - result["dva"]
            assert abs(result["total_xva"] - expected_total) < 0.01


class TestPortfolioStoreBehaviour:
    """Test persistence and concurrency guarantees of the portfolio store."""

    def test_portfolio_persistence_across_app_instances(
        self, portfolio, file_store_env
    ) -> None:
        """Registering a portfolio persists it across new app instances."""

        portfolio_data = portfolio.model_dump(mode="json")

        with TestClient(create_app()) as first_client:
            response = first_client.post("/portfolio/register", json=portfolio_data)
            assert response.status_code == 200

        with TestClient(create_app()) as second_client:
            response = second_client.get("/portfolio/Global Trading Portfolio/summary")

        assert response.status_code == 200
        summary = response.json()
        assert summary["portfolio_id"] == "Global Trading Portfolio"
        assert summary["trades"] == 11

    @pytest.mark.anyio
    async def test_concurrent_access_is_thread_safe(self, portfolio) -> None:
        """Concurrent clients should read consistent data without race conditions."""

        store = InMemoryPortfolioStore()
        app = create_app(portfolio_store=store)
        portfolio_data = portfolio.model_dump(mode="json")

        with TestClient(app) as client:
            response = client.post("/portfolio/register", json=portfolio_data)
            assert response.status_code == 200

        async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:

            async def fetch_summary() -> dict[str, Any]:
                response = await async_client.get(
                    "/portfolio/Global Trading Portfolio/summary"
                )
                assert response.status_code == 200
                return response.json()

            results = await asyncio.gather(*(fetch_summary() for _ in range(8)))

        for summary in results:
            assert summary["portfolio_id"] == "Global Trading Portfolio"
            assert summary["trades"] == 11
