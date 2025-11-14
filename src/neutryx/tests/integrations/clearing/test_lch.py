"""Tests for LCH SwapClear integration."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from neutryx.integrations.clearing.lch import (
    LCHSwapClearConnector,
    LCHSwapClearConfig,
    LCHTradeDetails,
)
from neutryx.integrations.clearing.base import (
    Trade,
    TradeEconomics,
    Party,
    ProductType,
    TradeStatus,
    CCPConnectionError,
    CCPTradeRejectionError,
)


@pytest.fixture
def lch_config():
    """Create LCH SwapClear configuration."""
    return LCHSwapClearConfig(
        ccp_name="LCH SwapClear",
        member_id="LCH_MEMBER_123",
        api_endpoint="https://api.lch.com",
        api_key="test_lch_key",
        api_secret="test_lch_secret",
        clearing_service="SwapClear",
        compression_enabled=True,
        settlement_currency="USD",
        margin_method="PAI",
        environment="test",
    )


@pytest.fixture
def sample_irs_trade():
    """Create sample interest rate swap trade."""
    return Trade(
        trade_id="IRS_001",
        product_type=ProductType.IRS,
        trade_date=datetime(2025, 1, 1),
        effective_date=datetime(2025, 1, 3),
        maturity_date=datetime(2035, 1, 3),
        buyer=Party(
            party_id="PARTY_A",
            name="Bank A",
            lei="5493001234567890ABCD",
            bic="BANKA2L",
            member_id="LCH_MEMBER_123",
        ),
        seller=Party(
            party_id="PARTY_B",
            name="Bank B",
            lei="5493009876543210ZYXW",
            bic="BANKB2L",
            member_id="LCH_MEMBER_456",
        ),
        economics=TradeEconomics(
            notional=Decimal("50000000"),
            currency="USD",
            fixed_rate=Decimal("0.0325"),
        ),
        uti="UTI_LCH_123456789",
    )


class TestLCHSwapClearConfig:
    """Test LCH SwapClear configuration."""

    def test_config_creation(self, lch_config):
        """Test configuration creation."""
        assert lch_config.ccp_name == "LCH SwapClear"
        assert lch_config.clearing_service == "SwapClear"
        assert lch_config.compression_enabled is True
        assert lch_config.margin_method == "PAI"

    def test_config_defaults(self):
        """Test default configuration values."""
        config = LCHSwapClearConfig(
            ccp_name="LCH",
            member_id="MEMBER",
            api_endpoint="https://api.lch.com",
        )
        assert config.clearing_service == "SwapClear"
        assert config.compression_enabled is False
        assert config.settlement_currency == "USD"
        assert config.margin_method == "PAI"


class TestLCHTradeDetails:
    """Test LCH trade details model."""

    def test_trade_details_creation(self):
        """Test trade details creation."""
        details = LCHTradeDetails(
            usi="USI_123456",
            collateralization="fully_collateralized",
            swap_stream_buyer={"leg_type": "fixed", "rate": 0.025},
            swap_stream_seller={"leg_type": "floating", "index": "SOFR"},
        )
        assert details.usi == "USI_123456"
        assert details.collateralization == "fully_collateralized"
        assert details.confirmation_method == "electronic"


@pytest.mark.asyncio
class TestLCHSwapClearConnector:
    """Test LCH SwapClear connector."""

    async def test_connector_initialization(self, lch_config):
        """Test connector initialization."""
        connector = LCHSwapClearConnector(lch_config)
        assert connector.config == lch_config
        assert not connector.is_connected
        assert connector.metrics.total_submissions == 0

    @patch("httpx.AsyncClient")
    async def test_connect_success(self, mock_client_class, lch_config):
        """Test successful connection."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_id": "lch_session_abc"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        result = await connector.connect()

        assert result is True
        assert connector.is_connected
        assert connector.session_id == "lch_session_abc"

    @patch("httpx.AsyncClient")
    async def test_connect_authentication_failure(self, mock_client_class, lch_config):
        """Test connection with authentication failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)

        from neutryx.integrations.clearing.base import CCPAuthenticationError
        with pytest.raises(CCPAuthenticationError):
            await connector.connect()

    @patch("httpx.AsyncClient")
    async def test_disconnect(self, mock_client_class, lch_config):
        """Test disconnection."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        connector._connected = True
        connector._client = mock_client

        result = await connector.disconnect()

        assert result is True
        assert not connector.is_connected
        assert connector.session_id is None
        mock_client.aclose.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_submit_trade_success(self, mock_client_class, lch_config, sample_irs_trade):
        """Test successful trade submission."""
        # Mock authentication response
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json.return_value = {"session_id": "session_123"}

        # Mock trade submission response
        submit_response = MagicMock()
        submit_response.status_code = 201
        submit_response.json.return_value = {
            "submission_id": "sub_789",
            "lch_trade_id": "LCH_TRADE_001",
            "compression_eligible": True,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[auth_response, submit_response])
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        await connector.connect()

        response = await connector.submit_trade(sample_irs_trade)

        assert response.status == TradeStatus.ACCEPTED
        assert response.trade_id == sample_irs_trade.trade_id
        assert response.ccp_trade_id == "LCH_TRADE_001"
        assert response.metadata["clearing_house"] == "LCH"
        assert response.metadata["service"] == "SwapClear"
        assert connector.metrics.successful_submissions == 1

    @patch("httpx.AsyncClient")
    async def test_submit_trade_rejection(self, mock_client_class, lch_config, sample_irs_trade):
        """Test trade rejection."""
        # Mock authentication
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json.return_value = {"session_id": "session_123"}

        # Mock rejection response
        reject_response = MagicMock()
        reject_response.status_code = 400
        reject_response.json.return_value = {
            "error_code": "ERR_NOTIONAL",
            "message": "Notional amount exceeds limit",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[auth_response, reject_response])
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        await connector.connect()

        with pytest.raises(CCPTradeRejectionError) as exc_info:
            await connector.submit_trade(sample_irs_trade)

        assert "Notional amount exceeds limit" in str(exc_info.value)
        assert exc_info.value.rejection_code == "ERR_NOTIONAL"
        assert connector.metrics.rejected_submissions == 1

    async def test_submit_trade_not_connected(self, lch_config, sample_irs_trade):
        """Test trade submission when not connected."""
        connector = LCHSwapClearConnector(lch_config)

        with pytest.raises(CCPConnectionError):
            await connector.submit_trade(sample_irs_trade)

    @patch("httpx.AsyncClient")
    async def test_get_trade_status(self, mock_client_class, lch_config):
        """Test getting trade status."""
        # Mock authentication
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json.return_value = {"session_id": "session_123"}

        # Mock status response
        status_response = MagicMock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "cleared"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=auth_response)
        mock_client.get = AsyncMock(return_value=status_response)
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        await connector.connect()

        status = await connector.get_trade_status("TRADE_123")
        assert status == TradeStatus.CLEARED

    @patch("httpx.AsyncClient")
    async def test_get_margin_requirements(self, mock_client_class, lch_config):
        """Test getting margin requirements."""
        # Mock authentication
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json.return_value = {"session_id": "session_123"}

        # Mock margin response
        margin_response = MagicMock()
        margin_response.status_code = 200
        margin_response.json.return_value = {
            "initial_margin": 2500000,
            "variation_margin": 125000,
            "margin_method": "PAI",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=auth_response)
        mock_client.get = AsyncMock(return_value=margin_response)
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        await connector.connect()

        margin = await connector.get_margin_requirements()
        assert margin["initial_margin"] == 2500000
        assert margin["margin_method"] == "PAI"

    @patch("httpx.AsyncClient")
    async def test_request_compression(self, mock_client_class, lch_config):
        """Test compression request."""
        # Mock authentication
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json.return_value = {"session_id": "session_123"}

        # Mock compression response
        compression_response = MagicMock()
        compression_response.status_code = 200
        compression_response.json.return_value = {
            "compression_id": "COMP_001",
            "trades_compressed": 25,
            "notional_reduced": 500000000,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[auth_response, compression_response])
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        await connector.connect()

        result = await connector.request_compression()
        assert result["compression_id"] == "COMP_001"
        assert result["trades_compressed"] == 25

    @patch("httpx.AsyncClient")
    async def test_healthcheck(self, mock_client_class, lch_config):
        """Test health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        connector = LCHSwapClearConnector(lch_config)
        connector._connected = True
        connector._client = mock_client

        result = await connector.healthcheck()
        assert result is True


class TestLCHSwapClearHelpers:
    """Test LCH SwapClear helper methods (sync tests)."""

    def test_format_lch_trade(self, lch_config, sample_irs_trade):
        """Test trade formatting for LCH."""
        connector = LCHSwapClearConnector(lch_config)
        formatted = connector._format_lch_trade(sample_irs_trade)

        assert "usi" in formatted
        assert formatted["product_type"] == "interest_rate_swap"
        assert formatted["notional"]["amount"] == 50000000.0
        assert formatted["notional"]["currency"] == "USD"
        assert formatted["fixed_rate"] == 0.0325
        assert formatted["settlement_currency"] == "USD"
        assert formatted["margin_method"] == "PAI"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
