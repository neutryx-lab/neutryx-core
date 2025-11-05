"""Tests for CLS (Continuous Linked Settlement) integration."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from neutryx.integrations.clearing.base import CCPConfig
from neutryx.integrations.clearing.cls import (
    CLSConnector,
    CLSSettlementInstruction,
    CLSSettlementService,
    CLSCurrency,
    CLSStatus,
)
from neutryx.integrations.clearing.cls.messages import CLSSettlementStatus


@pytest.fixture
def cls_config():
    """Create CLS configuration."""
    return CCPConfig(
        ccp_name="CLS",
        member_id="CLSMEMBER001",
        api_endpoint="https://api.cls-group.com/v1",
        api_key="test_api_key",
        environment="test",
        use_sandbox=True,
    )


@pytest.fixture
async def cls_connector(cls_config):
    """Create and connect CLS connector."""
    connector = CLSConnector(cls_config)
    await connector.connect()
    yield connector
    await connector.disconnect()


class TestCLSSettlementInstruction:
    """Tests for CLS settlement instruction."""

    def test_create_instruction(self):
        """Test creating CLS settlement instruction."""
        instruction = CLSSettlementInstruction(
            instruction_id="CLSI001",
            trade_id="TRADE001",
            settlement_session_id="CLS_20250117",
            buy_currency=CLSCurrency.USD,
            buy_amount=Decimal("1000000.00"),
            sell_currency=CLSCurrency.EUR,
            sell_amount=Decimal("900000.00"),
            fx_rate=Decimal("1.1111"),
            value_date=date(2025, 1, 17),
            settlement_date=date(2025, 1, 17),
            submitter_bic="BANKGB2LXXX",
            counterparty_bic="BANKUS33XXX",
            settlement_member="MEMBER001",
        )

        assert instruction.buy_currency == CLSCurrency.USD
        assert instruction.sell_currency == CLSCurrency.EUR
        assert instruction.validate_currencies()

    def test_same_currency_validation(self):
        """Test validation fails for same buy/sell currency."""
        instruction = CLSSettlementInstruction(
            instruction_id="CLSI002",
            trade_id="TRADE002",
            settlement_session_id="CLS_20250117",
            buy_currency=CLSCurrency.USD,
            buy_amount=Decimal("1000000.00"),
            sell_currency=CLSCurrency.USD,  # Same as buy_currency
            sell_amount=Decimal("1000000.00"),
            fx_rate=Decimal("1.0000"),
            value_date=date(2025, 1, 17),
            settlement_date=date(2025, 1, 17),
            submitter_bic="BANKGB2LXXX",
            counterparty_bic="BANKUS33XXX",
            settlement_member="MEMBER001",
        )

        with pytest.raises(ValueError, match="Buy and sell currencies must be different"):
            instruction.validate_currencies()

    def test_to_cls_message(self):
        """Test converting instruction to CLS message format."""
        instruction = CLSSettlementInstruction(
            instruction_id="CLSI003",
            trade_id="TRADE003",
            settlement_session_id="CLS_20250117",
            buy_currency=CLSCurrency.GBP,
            buy_amount=Decimal("800000.00"),
            sell_currency=CLSCurrency.USD,
            sell_amount=Decimal("1000000.00"),
            fx_rate=Decimal("1.2500"),
            value_date=date(2025, 1, 17),
            settlement_date=date(2025, 1, 17),
            submitter_bic="BANKGB2LXXX",
            counterparty_bic="BANKUS33XXX",
            settlement_member="MEMBER001",
        )

        msg = instruction.to_cls_message()
        assert "MSG_TYPE=SETTLEMENT_INSTRUCTION" in msg
        assert "BUY_CCY=GBP" in msg
        assert "SELL_CCY=USD" in msg


class TestCLSConnector:
    """Tests for CLS connector."""

    @pytest.mark.asyncio
    async def test_connect(self, cls_connector):
        """Test connecting to CLS."""
        assert cls_connector.is_connected
        assert cls_connector.session_id is not None

    @pytest.mark.asyncio
    async def test_submit_settlement_instruction(self, cls_connector):
        """Test submitting settlement instruction."""
        instruction = CLSSettlementInstruction(
            instruction_id="CLSI004",
            trade_id="TRADE004",
            settlement_session_id="CLS_20250117",
            buy_currency=CLSCurrency.USD,
            buy_amount=Decimal("1000000.00"),
            sell_currency=CLSCurrency.JPY,
            sell_amount=Decimal("110000000.00"),
            fx_rate=Decimal("110.0000"),
            value_date=date(2025, 1, 17),
            settlement_date=date(2025, 1, 17),
            submitter_bic="BANKGB2LXXX",
            counterparty_bic="BANKJP2TXXX",
            settlement_member="MEMBER001",
        )

        confirmation = await cls_connector.submit_settlement_instruction(instruction)

        assert confirmation.instruction_id == "CLSI004"
        assert confirmation.status == CLSSettlementStatus.MATCHED
        assert confirmation.is_successful() is False  # Not yet settled

    @pytest.mark.asyncio
    async def test_get_settlement_status(self, cls_connector):
        """Test getting settlement status."""
        # First submit instruction
        instruction = CLSSettlementInstruction(
            instruction_id="CLSI005",
            trade_id="TRADE005",
            settlement_session_id="CLS_20250117",
            buy_currency=CLSCurrency.EUR,
            buy_amount=Decimal("900000.00"),
            sell_currency=CLSCurrency.USD,
            sell_amount=Decimal("1000000.00"),
            fx_rate=Decimal("1.1111"),
            value_date=date(2025, 1, 17),
            settlement_date=date(2025, 1, 17),
            submitter_bic="BANKGB2LXXX",
            counterparty_bic="BANKUS33XXX",
            settlement_member="MEMBER001",
        )

        await cls_connector.submit_settlement_instruction(instruction)

        # Then get status
        status = await cls_connector.get_settlement_status("CLSI005")

        assert status.instruction_id == "CLSI005"
        assert status.trade_id == "TRADE005"
        assert status.pay_in_complete

    @pytest.mark.asyncio
    async def test_cancel_settlement_instruction(self, cls_connector):
        """Test cancelling settlement instruction."""
        # Submit instruction
        instruction = CLSSettlementInstruction(
            instruction_id="CLSI006",
            trade_id="TRADE006",
            settlement_session_id="CLS_20250117",
            buy_currency=CLSCurrency.CHF,
            buy_amount=Decimal("950000.00"),
            sell_currency=CLSCurrency.EUR,
            sell_amount=Decimal("900000.00"),
            fx_rate=Decimal("0.9474"),
            value_date=date(2025, 1, 17),
            settlement_date=date(2025, 1, 17),
            submitter_bic="BANKGB2LXXX",
            counterparty_bic="BANKCH2ZXXX",
            settlement_member="MEMBER001",
        )

        await cls_connector.submit_settlement_instruction(instruction)

        # Cancel it
        result = await cls_connector.cancel_settlement_instruction("CLSI006")
        assert result is True

        # Verify status is cancelled
        status = await cls_connector.get_settlement_status("CLSI006")
        assert status.status == CLSSettlementStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_healthcheck(self, cls_connector):
        """Test CLS healthcheck."""
        healthy = await cls_connector.healthcheck()
        assert healthy is True


class TestCLSSettlementService:
    """Tests for CLS settlement service."""

    @pytest.mark.asyncio
    async def test_settle_fx_trade(self, cls_connector):
        """Test settling FX trade."""
        service = CLSSettlementService(cls_connector)

        confirmation = await service.settle_fx_trade(
            trade_id="TRADE007",
            buy_currency="USD",
            buy_amount=Decimal("1000000.00"),
            sell_currency="EUR",
            sell_amount=Decimal("900000.00"),
            value_date=date(2025, 1, 17),
            submitter_bic="BANKGB2LXXX",
            counterparty_bic="BANKUS33XXX",
            settlement_member="MEMBER001",
        )

        assert confirmation.trade_id == "TRADE007"
        assert confirmation.status == CLSSettlementStatus.MATCHED

    @pytest.mark.asyncio
    async def test_settle_fx_trade_invalid_currency(self, cls_connector):
        """Test settling FX trade with invalid currency."""
        service = CLSSettlementService(cls_connector)

        with pytest.raises(ValueError, match="Invalid or non-CLS-eligible currency"):
            await service.settle_fx_trade(
                trade_id="TRADE008",
                buy_currency="ZZZ",  # Invalid currency
                buy_amount=Decimal("1000000.00"),
                sell_currency="EUR",
                sell_amount=Decimal("900000.00"),
                value_date=date(2025, 1, 17),
                submitter_bic="BANKGB2LXXX",
                counterparty_bic="BANKUS33XXX",
                settlement_member="MEMBER001",
            )

    @pytest.mark.asyncio
    async def test_calculate_settlement_exposure(self, cls_connector):
        """Test calculating net settlement exposure."""
        service = CLSSettlementService(cls_connector)

        # Submit multiple instructions
        instructions = []
        for i in range(3):
            confirmation = await service.settle_fx_trade(
                trade_id=f"TRADE{100+i}",
                buy_currency="USD",
                buy_amount=Decimal("1000000.00"),
                sell_currency="EUR",
                sell_amount=Decimal("900000.00"),
                value_date=date(2025, 1, 17),
                submitter_bic="BANKGB2LXXX",
                counterparty_bic="BANKUS33XXX",
                settlement_member="MEMBER001",
            )

        pending = await service.get_pending_instructions()
        exposures = service.calculate_settlement_exposure(pending)

        assert "USD" in exposures
        assert "EUR" in exposures
        assert exposures["USD"] == Decimal("3000000.00")  # 3 x 1M
        assert exposures["EUR"] == Decimal("-2700000.00")  # 3 x -900K
