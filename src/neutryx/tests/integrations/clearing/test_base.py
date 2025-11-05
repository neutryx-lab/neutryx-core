"""Tests for CCP base classes and protocols."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from neutryx.integrations.clearing import (
    CCPConfig,
    CCPConnector,
    CCPMetrics,
    Party,
    Trade,
    TradeEconomics,
    TradeStatus,
    ProductType,
)


class TestCCPConfig:
    """Test CCP configuration."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        config = CCPConfig(
            ccp_name="TEST_CCP",
            member_id="MEMBER123",
            api_endpoint="https://api.test.com",
        )

        assert config.ccp_name == "TEST_CCP"
        assert config.member_id == "MEMBER123"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_config_with_auth(self):
        """Test configuration with authentication."""
        config = CCPConfig(
            ccp_name="TEST_CCP",
            member_id="MEMBER123",
            api_endpoint="https://api.test.com",
            api_key="key123",
            api_secret="secret456",
        )

        assert config.api_key == "key123"
        assert config.api_secret == "secret456"

    def test_config_timeout_settings(self):
        """Test custom timeout settings."""
        config = CCPConfig(
            ccp_name="TEST_CCP",
            member_id="MEMBER123",
            api_endpoint="https://api.test.com",
            timeout=60,
            max_retries=5,
            retry_delay=10,
        )

        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.retry_delay == 10


class TestParty:
    """Test Party model."""

    def test_party_creation(self):
        """Test party creation with required fields."""
        party = Party(
            party_id="PARTY001",
            name="Test Corporation",
        )

        assert party.party_id == "PARTY001"
        assert party.name == "Test Corporation"
        assert party.lei is None

    def test_party_with_lei(self):
        """Test party with LEI."""
        party = Party(
            party_id="PARTY001",
            name="Test Corporation",
            lei="123456789012345678XX",
            bic="TESTUS33",
        )

        assert party.lei == "123456789012345678XX"
        assert party.bic == "TESTUS33"


class TestTradeEconomics:
    """Test TradeEconomics model."""

    def test_basic_economics(self):
        """Test basic trade economics."""
        economics = TradeEconomics(
            notional=Decimal("1000000"),
            currency="USD",
        )

        assert economics.notional == Decimal("1000000")
        assert economics.currency == "USD"
        assert economics.fixed_rate is None

    def test_swap_economics(self):
        """Test swap economics with fixed rate."""
        economics = TradeEconomics(
            notional=Decimal("5000000"),
            currency="EUR",
            fixed_rate=Decimal("0.025"),
            spread=Decimal("0.001"),
        )

        assert economics.fixed_rate == Decimal("0.025")
        assert economics.spread == Decimal("0.001")

    def test_economics_to_dict(self):
        """Test economics conversion to dictionary."""
        economics = TradeEconomics(
            notional=Decimal("1000000"),
            currency="USD",
            fixed_rate=Decimal("0.03"),
        )

        econ_dict = economics.to_dict()
        assert econ_dict["notional"] == 1000000.0
        assert econ_dict["currency"] == "USD"
        assert econ_dict["fixed_rate"] == 0.03


class TestTrade:
    """Test Trade model."""

    def test_trade_creation(self):
        """Test trade creation."""
        buyer = Party(party_id="BUYER1", name="Buyer Corp", lei="BUYER123")
        seller = Party(party_id="SELLER1", name="Seller Corp", lei="SELLER456")
        economics = TradeEconomics(
            notional=Decimal("10000000"),
            currency="USD",
            fixed_rate=Decimal("0.025"),
        )

        trade = Trade(
            trade_id="TRADE001",
            product_type=ProductType.IRS,
            trade_date=datetime.utcnow(),
            effective_date=datetime.utcnow() + timedelta(days=2),
            maturity_date=datetime.utcnow() + timedelta(days=365 * 5),
            buyer=buyer,
            seller=seller,
            economics=economics,
        )

        assert trade.trade_id == "TRADE001"
        assert trade.product_type == ProductType.IRS
        assert trade.buyer.party_id == "BUYER1"
        assert trade.seller.party_id == "SELLER1"

    def test_trade_with_uti(self):
        """Test trade with UTI."""
        buyer = Party(party_id="BUYER1", name="Buyer Corp")
        seller = Party(party_id="SELLER1", name="Seller Corp")
        economics = TradeEconomics(notional=Decimal("1000000"), currency="USD")

        trade = Trade(
            trade_id="TRADE001",
            product_type=ProductType.IRS,
            trade_date=datetime.utcnow(),
            effective_date=datetime.utcnow(),
            maturity_date=datetime.utcnow() + timedelta(days=365),
            buyer=buyer,
            seller=seller,
            economics=economics,
            uti="UTI123456789",
        )

        assert trade.uti == "UTI123456789"

    def test_trade_to_dict(self):
        """Test trade conversion to dictionary."""
        buyer = Party(party_id="BUYER1", name="Buyer Corp")
        seller = Party(party_id="SELLER1", name="Seller Corp")
        economics = TradeEconomics(notional=Decimal("1000000"), currency="USD")

        trade = Trade(
            trade_id="TRADE001",
            product_type=ProductType.IRS,
            trade_date=datetime.utcnow(),
            effective_date=datetime.utcnow(),
            maturity_date=datetime.utcnow() + timedelta(days=365),
            buyer=buyer,
            seller=seller,
            economics=economics,
        )

        trade_dict = trade.to_dict()
        assert trade_dict["trade_id"] == "TRADE001"
        assert trade_dict["product_type"] == "interest_rate_swap"
        assert "buyer" in trade_dict
        assert "economics" in trade_dict


class TestCCPMetrics:
    """Test CCPMetrics."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CCPMetrics()

        assert metrics.total_submissions == 0
        assert metrics.successful_submissions == 0
        assert metrics.rejected_submissions == 0
        assert metrics.success_rate() == 0.0

    def test_metrics_record_success(self):
        """Test recording successful submission."""
        metrics = CCPMetrics()
        metrics.record_submission(success=True, response_time_ms=150.0)

        assert metrics.total_submissions == 1
        assert metrics.successful_submissions == 1
        assert metrics.success_rate() == 1.0
        assert metrics.avg_response_time_ms == 150.0

    def test_metrics_record_rejection(self):
        """Test recording rejection."""
        metrics = CCPMetrics()
        metrics.record_submission(success=False, response_time_ms=50.0, rejected=True)

        assert metrics.total_submissions == 1
        assert metrics.rejected_submissions == 1
        assert metrics.success_rate() == 0.0

    def test_metrics_multiple_submissions(self):
        """Test multiple submissions."""
        metrics = CCPMetrics()
        metrics.record_submission(True, 100.0)
        metrics.record_submission(True, 200.0)
        metrics.record_submission(False, 50.0, rejected=True)
        metrics.record_submission(True, 150.0)

        assert metrics.total_submissions == 4
        assert metrics.successful_submissions == 3
        assert metrics.rejected_submissions == 1
        assert metrics.success_rate() == 0.75
        assert 100.0 < metrics.avg_response_time_ms < 150.0

    def test_metrics_to_dict(self):
        """Test metrics to dictionary conversion."""
        metrics = CCPMetrics()
        metrics.record_submission(True, 100.0)

        metrics_dict = metrics.to_dict()
        assert "total_submissions" in metrics_dict
        assert "success_rate" in metrics_dict
        assert "avg_response_time_ms" in metrics_dict


class TestTradeStatus:
    """Test TradeStatus enum."""

    def test_trade_status_values(self):
        """Test trade status enum values."""
        assert TradeStatus.PENDING == "pending"
        assert TradeStatus.SUBMITTED == "submitted"
        assert TradeStatus.ACCEPTED == "accepted"
        assert TradeStatus.REJECTED == "rejected"
        assert TradeStatus.CLEARED == "cleared"
        assert TradeStatus.SETTLED == "settled"


class TestProductType:
    """Test ProductType enum."""

    def test_product_type_values(self):
        """Test product type enum values."""
        assert ProductType.IRS == "interest_rate_swap"
        assert ProductType.CDS == "credit_default_swap"
        assert ProductType.FX_FORWARD == "fx_forward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
