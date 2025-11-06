"""Tests for CCP base classes and common functionality."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from neutryx.integrations.clearing.base import (
    CCPConfig,
    CCPConnector,
    CCPError,
    CCPAuthenticationError,
    CCPConnectionError,
    CCPTradeRejectionError,
    CCPTimeoutError,
    CCPMetrics,
    Trade,
    TradeEconomics,
    TradeStatus,
    TradeSubmissionResponse,
    Party,
    ProductType,
    MarginCall,
    PositionReport,
)


class MockCCPConnector(CCPConnector):
    """Mock CCP connector for testing."""

    async def connect(self) -> bool:
        self._connected = True
        self._session_id = "mock_session_123"
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        self._session_id = None
        return True

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        if not self._connected:
            raise CCPConnectionError("Not connected")
        return TradeSubmissionResponse(
            submission_id="sub_123",
            trade_id=trade.trade_id,
            status=TradeStatus.ACCEPTED,
            ccp_trade_id="ccp_456",
        )

    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        return TradeStatus.CLEARED

    async def cancel_trade(self, trade_id: str) -> bool:
        return True

    async def get_margin_requirements(self, member_id=None):
        return {"initial_margin": 1000000, "variation_margin": 50000}

    async def get_position_report(self, as_of_date=None):
        return PositionReport(
            report_id="rpt_123",
            member_id=self.config.member_id,
            as_of_date=datetime.utcnow(),
            positions=[],
            total_exposure=Decimal("10000000"),
            initial_margin=Decimal("1000000"),
            variation_margin=Decimal("50000"),
        )

    async def healthcheck(self) -> bool:
        return self._connected


@pytest.fixture
def ccp_config():
    """Create test CCP configuration."""
    return CCPConfig(
        ccp_name="TestCCP",
        member_id="TEST_MEMBER",
        api_endpoint="https://api.test.ccp",
        api_key="test_key",
        api_secret="test_secret",
        timeout=30,
        max_retries=3,
        environment="test",
    )


@pytest.fixture
def sample_party():
    """Create sample party."""
    return Party(
        party_id="PARTY123",
        name="Test Party Inc",
        lei="549300ABCDEF12345678",
        bic="TESTGB2L",
        member_id="MEMBER123",
    )


@pytest.fixture
def sample_trade(sample_party):
    """Create sample trade."""
    return Trade(
        trade_id="TRADE_001",
        product_type=ProductType.IRS,
        trade_date=datetime(2025, 1, 1),
        effective_date=datetime(2025, 1, 3),
        maturity_date=datetime(2030, 1, 3),
        buyer=sample_party,
        seller=Party(
            party_id="PARTY456",
            name="Counterparty Corp",
            lei="549300ZYXWVU98765432",
            bic="COUNTGB2L",
            member_id="MEMBER456",
        ),
        economics=TradeEconomics(
            notional=Decimal("10000000"),
            currency="USD",
            fixed_rate=Decimal("0.025"),
        ),
        uti="UTI_123456789",
    )


class TestCCPConfig:
    """Test CCP configuration."""

    def test_config_creation(self, ccp_config):
        """Test configuration creation."""
        assert ccp_config.ccp_name == "TestCCP"
        assert ccp_config.member_id == "TEST_MEMBER"
        assert ccp_config.timeout == 30
        assert ccp_config.max_retries == 3
        assert ccp_config.environment == "test"

    def test_config_defaults(self):
        """Test default configuration values."""
        config = CCPConfig(
            ccp_name="TestCCP",
            member_id="MEMBER",
            api_endpoint="https://api.test",
        )
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay == 5
        assert config.environment == "production"
        assert config.protocol == "REST"


class TestParty:
    """Test Party model."""

    def test_party_creation(self, sample_party):
        """Test party creation."""
        assert sample_party.party_id == "PARTY123"
        assert sample_party.name == "Test Party Inc"
        assert sample_party.lei == "549300ABCDEF12345678"
        assert sample_party.bic == "TESTGB2L"

    def test_party_immutable(self, sample_party):
        """Test that party is immutable (frozen)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            sample_party.party_id = "NEW_ID"


class TestTradeEconomics:
    """Test trade economics model."""

    def test_economics_creation(self):
        """Test economics creation."""
        econ = TradeEconomics(
            notional=Decimal("5000000"),
            currency="EUR",
            fixed_rate=Decimal("0.03"),
            spread=Decimal("0.005"),
        )
        assert econ.notional == Decimal("5000000")
        assert econ.currency == "EUR"
        assert econ.fixed_rate == Decimal("0.03")
        assert econ.spread == Decimal("0.005")

    def test_economics_to_dict(self):
        """Test conversion to dictionary."""
        econ = TradeEconomics(
            notional=Decimal("1000000"),
            currency="USD",
            fixed_rate=Decimal("0.025"),
        )
        d = econ.to_dict()
        assert d["notional"] == 1000000.0
        assert d["currency"] == "USD"
        assert d["fixed_rate"] == 0.025


class TestTrade:
    """Test Trade model."""

    def test_trade_creation(self, sample_trade):
        """Test trade creation."""
        assert sample_trade.trade_id == "TRADE_001"
        assert sample_trade.product_type == ProductType.IRS
        assert sample_trade.economics.notional == Decimal("10000000")

    def test_trade_to_dict(self, sample_trade):
        """Test trade serialization."""
        d = sample_trade.to_dict()
        assert d["trade_id"] == "TRADE_001"
        assert d["product_type"] == "interest_rate_swap"
        assert d["economics"]["notional"] == 10000000.0
        assert "buyer" in d
        assert "seller" in d


class TestTradeSubmissionResponse:
    """Test trade submission response."""

    def test_submission_response_accepted(self):
        """Test accepted submission."""
        response = TradeSubmissionResponse(
            submission_id="sub_123",
            trade_id="trade_456",
            status=TradeStatus.ACCEPTED,
            ccp_trade_id="ccp_789",
        )
        assert response.status == TradeStatus.ACCEPTED
        assert response.ccp_trade_id == "ccp_789"
        assert response.rejection_reason is None

    def test_submission_response_rejected(self):
        """Test rejected submission."""
        response = TradeSubmissionResponse(
            submission_id="sub_123",
            trade_id="trade_456",
            status=TradeStatus.REJECTED,
            rejection_reason="Invalid notional amount",
            rejection_code="ERR_NOTIONAL",
        )
        assert response.status == TradeStatus.REJECTED
        assert response.rejection_reason == "Invalid notional amount"
        assert response.rejection_code == "ERR_NOTIONAL"


class TestMarginCall:
    """Test margin call model."""

    def test_margin_call_creation(self):
        """Test margin call creation."""
        call = MarginCall(
            call_id="CALL_001",
            member_id="MEMBER123",
            call_amount=Decimal("500000"),
            currency="USD",
            call_type="variation",
            due_time=datetime(2025, 1, 15, 17, 0),
            portfolio_im=Decimal("2000000"),
        )
        assert call.call_amount == Decimal("500000")
        assert call.call_type == "variation"
        assert call.portfolio_im == Decimal("2000000")


class TestPositionReport:
    """Test position report model."""

    def test_position_report_creation(self):
        """Test position report creation."""
        report = PositionReport(
            report_id="RPT_001",
            member_id="MEMBER123",
            as_of_date=datetime(2025, 1, 15),
            positions=[{"trade_id": "T1", "notional": 1000000}],
            total_exposure=Decimal("5000000"),
            initial_margin=Decimal("500000"),
            variation_margin=Decimal("25000"),
        )
        assert report.total_exposure == Decimal("5000000")
        assert len(report.positions) == 1


class TestCCPErrors:
    """Test CCP error classes."""

    def test_ccp_error(self):
        """Test base CCP error."""
        error = CCPError("Base error message")
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)

    def test_connection_error(self):
        """Test connection error."""
        error = CCPConnectionError("Connection failed")
        assert isinstance(error, CCPError)
        assert str(error) == "Connection failed"

    def test_authentication_error(self):
        """Test authentication error."""
        error = CCPAuthenticationError("Invalid credentials")
        assert isinstance(error, CCPError)

    def test_trade_rejection_error(self):
        """Test trade rejection error."""
        error = CCPTradeRejectionError("Trade rejected", rejection_code="ERR_123")
        assert error.rejection_code == "ERR_123"
        assert isinstance(error, CCPError)

    def test_timeout_error(self):
        """Test timeout error."""
        error = CCPTimeoutError("Request timed out")
        assert isinstance(error, CCPError)


class TestCCPMetrics:
    """Test CCP metrics tracking."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = CCPMetrics()
        assert metrics.total_submissions == 0
        assert metrics.successful_submissions == 0
        assert metrics.avg_response_time_ms == 0.0

    def test_record_successful_submission(self):
        """Test recording successful submission."""
        metrics = CCPMetrics()
        metrics.record_submission(success=True, response_time_ms=150.0)
        assert metrics.total_submissions == 1
        assert metrics.successful_submissions == 1
        assert metrics.avg_response_time_ms == 150.0

    def test_record_rejected_submission(self):
        """Test recording rejected submission."""
        metrics = CCPMetrics()
        metrics.record_submission(success=False, response_time_ms=100.0, rejected=True)
        assert metrics.total_submissions == 1
        assert metrics.rejected_submissions == 1
        assert metrics.successful_submissions == 0

    def test_record_failed_submission(self):
        """Test recording failed submission."""
        metrics = CCPMetrics()
        metrics.record_submission(success=False, response_time_ms=50.0)
        assert metrics.failed_submissions == 1

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = CCPMetrics()
        metrics.record_submission(True, 100.0)
        metrics.record_submission(True, 120.0)
        metrics.record_submission(False, 80.0, rejected=True)
        assert metrics.total_submissions == 3
        assert metrics.success_rate() == pytest.approx(2/3)

    def test_average_response_time(self):
        """Test average response time calculation."""
        metrics = CCPMetrics()
        metrics.record_submission(True, 100.0)
        metrics.record_submission(True, 200.0)
        metrics.record_submission(True, 150.0)
        assert metrics.avg_response_time_ms == pytest.approx(150.0)

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = CCPMetrics()
        metrics.record_submission(True, 100.0)
        d = metrics.to_dict()
        assert "total_submissions" in d
        assert "success_rate" in d
        assert "avg_response_time_ms" in d


@pytest.mark.asyncio
class TestCCPConnector:
    """Test CCP connector base functionality."""

    async def test_connector_initialization(self, ccp_config):
        """Test connector initialization."""
        connector = MockCCPConnector(ccp_config)
        assert connector.config == ccp_config
        assert not connector.is_connected
        assert connector.session_id is None

    async def test_connect(self, ccp_config):
        """Test connection."""
        connector = MockCCPConnector(ccp_config)
        result = await connector.connect()
        assert result is True
        assert connector.is_connected
        assert connector.session_id == "mock_session_123"

    async def test_disconnect(self, ccp_config):
        """Test disconnection."""
        connector = MockCCPConnector(ccp_config)
        await connector.connect()
        assert connector.is_connected
        result = await connector.disconnect()
        assert result is True
        assert not connector.is_connected
        assert connector.session_id is None

    async def test_submit_trade(self, ccp_config, sample_trade):
        """Test trade submission."""
        connector = MockCCPConnector(ccp_config)
        await connector.connect()
        response = await connector.submit_trade(sample_trade)
        assert response.status == TradeStatus.ACCEPTED
        assert response.trade_id == sample_trade.trade_id
        assert response.ccp_trade_id == "ccp_456"

    async def test_submit_trade_not_connected(self, ccp_config, sample_trade):
        """Test trade submission when not connected."""
        connector = MockCCPConnector(ccp_config)
        with pytest.raises(CCPConnectionError):
            await connector.submit_trade(sample_trade)

    async def test_get_trade_status(self, ccp_config):
        """Test getting trade status."""
        connector = MockCCPConnector(ccp_config)
        await connector.connect()
        status = await connector.get_trade_status("trade_123")
        assert status == TradeStatus.CLEARED

    async def test_cancel_trade(self, ccp_config):
        """Test trade cancellation."""
        connector = MockCCPConnector(ccp_config)
        await connector.connect()
        result = await connector.cancel_trade("trade_123")
        assert result is True

    async def test_get_margin_requirements(self, ccp_config):
        """Test getting margin requirements."""
        connector = MockCCPConnector(ccp_config)
        await connector.connect()
        margin = await connector.get_margin_requirements()
        assert "initial_margin" in margin
        assert "variation_margin" in margin

    async def test_get_position_report(self, ccp_config):
        """Test getting position report."""
        connector = MockCCPConnector(ccp_config)
        await connector.connect()
        report = await connector.get_position_report()
        assert report.member_id == ccp_config.member_id
        assert report.total_exposure == Decimal("10000000")

    async def test_healthcheck(self, ccp_config):
        """Test health check."""
        connector = MockCCPConnector(ccp_config)
        assert not await connector.healthcheck()
        await connector.connect()
        assert await connector.healthcheck()

    async def test_connector_repr(self, ccp_config):
        """Test connector string representation."""
        connector = MockCCPConnector(ccp_config)
        repr_str = repr(connector)
        assert "MockCCPConnector" in repr_str
        assert "TestCCP" in repr_str
        assert "TEST_MEMBER" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
