"""Tests for Euroclear integration."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from neutryx.integrations.clearing.base import CCPConfig
from neutryx.integrations.clearing.euroclear import (
    EuroclearConnector,
    EuroclearSettlementInstruction,
    EuroclearSettlementService,
    SettlementType,
)
from neutryx.integrations.clearing.euroclear.messages import SettlementStatus


@pytest.fixture
def euroclear_config():
    """Create Euroclear configuration."""
    return CCPConfig(
        ccp_name="Euroclear",
        member_id="EUROCLEAR001",
        api_endpoint="https://api.euroclear.com/v1",
        api_key="test_api_key",
        environment="test",
        use_sandbox=True,
    )


@pytest.fixture
async def euroclear_connector(euroclear_config):
    """Create and connect Euroclear connector."""
    connector = EuroclearConnector(euroclear_config)
    await connector.connect()
    yield connector
    await connector.disconnect()


class TestEuroclearSettlementInstruction:
    """Tests for Euroclear settlement instruction."""

    def test_create_instruction_dvp(self):
        """Test creating DVP settlement instruction."""
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI001",
            sender_reference="SENDER001",
            settlement_type=SettlementType.DVP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US0378331005",
            security_name="Apple Inc",
            quantity=Decimal("1000"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
            settlement_amount=Decimal("150000.00"),
            settlement_currency="USD",
        )

        assert instruction.settlement_type == SettlementType.DVP
        assert instruction.isin == "US0378331005"
        assert instruction.validate_dvp_fields()

    def test_create_instruction_fop(self):
        """Test creating FOP (Free of Payment) instruction."""
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI002",
            sender_reference="SENDER002",
            settlement_type=SettlementType.FOP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US5949181045",
            quantity=Decimal("500"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
        )

        assert instruction.settlement_type == SettlementType.FOP
        # FOP doesn't require settlement amount
        assert instruction.settlement_amount is None

    def test_dvp_validation_missing_amount(self):
        """Test DVP validation fails without settlement amount."""
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI003",
            sender_reference="SENDER003",
            settlement_type=SettlementType.DVP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US0378331005",
            quantity=Decimal("1000"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
            # Missing settlement_amount and settlement_currency
        )

        with pytest.raises(ValueError, match="DVP requires settlement amount and currency"):
            instruction.validate_dvp_fields()

    def test_invalid_isin(self):
        """Test invalid ISIN format."""
        with pytest.raises(ValueError, match="Invalid ISIN"):
            EuroclearSettlementInstruction(
                instruction_id="ECSI004",
                sender_reference="SENDER004",
                settlement_type=SettlementType.FOP,
                settlement_date=date(2025, 1, 17),
                trade_date=date(2025, 1, 15),
                isin="INVALID123",
                quantity=Decimal("1000"),
                delivering_party="PARTY001",
                receiving_party="PARTY002",
                participant_bic="BANKGB2LXXX",
            )

    def test_to_mt540(self):
        """Test converting instruction to MT540 format."""
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI005",
            sender_reference="SENDER005",
            settlement_type=SettlementType.RFP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US0378331005",
            security_name="Apple Inc",
            quantity=Decimal("1000"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
            safekeeping_account="SAFE001",
        )

        mt540_msg = instruction.to_mt540()
        assert ":20:SENDER005" in mt540_msg
        assert ":35B:ISIN US0378331005" in mt540_msg


class TestEuroclearConnector:
    """Tests for Euroclear connector."""

    @pytest.mark.anyio
    async def test_connect(self, euroclear_connector):
        """Test connecting to Euroclear."""
        assert euroclear_connector.is_connected
        assert euroclear_connector.session_id is not None

    @pytest.mark.anyio
    async def test_submit_settlement_instruction(self, euroclear_connector):
        """Test submitting settlement instruction."""
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI006",
            sender_reference="SENDER006",
            settlement_type=SettlementType.DVP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US0378331005",
            quantity=Decimal("1000"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
            settlement_amount=Decimal("150000.00"),
            settlement_currency="USD",
        )

        confirmation = await euroclear_connector.submit_settlement_instruction(instruction)

        assert confirmation.instruction_id == "ECSI006"
        assert confirmation.status == SettlementStatus.MATCHED
        assert confirmation.euroclear_reference is not None

    @pytest.mark.anyio
    async def test_get_settlement_status(self, euroclear_connector):
        """Test getting settlement status."""
        # First submit instruction
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI007",
            sender_reference="SENDER007",
            settlement_type=SettlementType.FOP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US5949181045",
            quantity=Decimal("500"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
        )

        await euroclear_connector.submit_settlement_instruction(instruction)

        # Then get status
        status = await euroclear_connector.get_settlement_status("ECSI007")

        assert status.instruction_id == "ECSI007"
        assert status.sender_reference == "SENDER007"
        assert status.matched

    @pytest.mark.anyio
    async def test_cancel_settlement_instruction(self, euroclear_connector):
        """Test cancelling settlement instruction."""
        # Submit instruction
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI008",
            sender_reference="SENDER008",
            settlement_type=SettlementType.FOP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US0378331005",
            quantity=Decimal("1000"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
        )

        await euroclear_connector.submit_settlement_instruction(instruction)

        # Cancel it
        result = await euroclear_connector.cancel_settlement_instruction("ECSI008")
        assert result is True

        # Verify status is cancelled
        status = await euroclear_connector.get_settlement_status("ECSI008")
        assert status.status == SettlementStatus.CANCELLED

    @pytest.mark.anyio
    async def test_amend_settlement_instruction(self, euroclear_connector):
        """Test amending settlement instruction."""
        # Submit instruction
        instruction = EuroclearSettlementInstruction(
            instruction_id="ECSI009",
            sender_reference="SENDER009",
            settlement_type=SettlementType.FOP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US0378331005",
            quantity=Decimal("1000"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
        )

        await euroclear_connector.submit_settlement_instruction(instruction)

        # Amend it
        new_date = date(2025, 1, 20)
        new_quantity = Decimal("1200")

        confirmation = await euroclear_connector.amend_settlement_instruction(
            instruction_id="ECSI009",
            new_settlement_date=new_date,
            new_quantity=new_quantity,
        )

        assert confirmation.instruction_id == "ECSI009"

    @pytest.mark.anyio
    async def test_get_holdings(self, euroclear_connector):
        """Test getting securities holdings."""
        holdings = await euroclear_connector.get_holdings("ACCOUNT001")

        assert holdings["account_id"] == "ACCOUNT001"
        assert "holdings" in holdings
        assert isinstance(holdings["holdings"], list)

    @pytest.mark.anyio
    async def test_healthcheck(self, euroclear_connector):
        """Test Euroclear healthcheck."""
        healthy = await euroclear_connector.healthcheck()
        assert healthy is True


class TestEuroclearSettlementService:
    """Tests for Euroclear settlement service."""

    @pytest.mark.anyio
    async def test_settle_securities_trade(self, euroclear_connector):
        """Test settling securities trade."""
        service = EuroclearSettlementService(euroclear_connector)

        confirmation = await service.settle_securities_trade(
            sender_reference="SENDER010",
            settlement_type=SettlementType.DVP,
            settlement_date=date(2025, 1, 17),
            trade_date=date(2025, 1, 15),
            isin="US0378331005",
            quantity=Decimal("1000"),
            delivering_party="PARTY001",
            receiving_party="PARTY002",
            participant_bic="BANKGB2LXXX",
            settlement_amount=Decimal("150000.00"),
            settlement_currency="USD",
        )

        assert confirmation.sender_reference == "SENDER010"
        assert confirmation.status == SettlementStatus.MATCHED

    @pytest.mark.anyio
    async def test_get_pending_instructions(self, euroclear_connector):
        """Test getting pending instructions."""
        service = EuroclearSettlementService(euroclear_connector)

        # Submit a few instructions
        for i in range(3):
            await service.settle_securities_trade(
                sender_reference=f"SENDER{200+i}",
                settlement_type=SettlementType.FOP,
                settlement_date=date(2025, 1, 17),
                trade_date=date(2025, 1, 15),
                isin="US0378331005",
                quantity=Decimal("1000"),
                delivering_party="PARTY001",
                receiving_party="PARTY002",
                participant_bic="BANKGB2LXXX",
            )

        pending = await service.get_pending_instructions(settlement_date=date(2025, 1, 17))
        assert len(pending) >= 3

    @pytest.mark.anyio
    async def test_reconcile_settlements(self, euroclear_connector):
        """Test reconciling settlements."""
        service = EuroclearSettlementService(euroclear_connector)

        # Submit instructions
        for i in range(2):
            await service.settle_securities_trade(
                sender_reference=f"SENDER{300+i}",
                settlement_type=SettlementType.FOP,
                settlement_date=date(2025, 1, 17),
                trade_date=date(2025, 1, 15),
                isin="US0378331005",
                quantity=Decimal("1000"),
                delivering_party="PARTY001",
                receiving_party="PARTY002",
                participant_bic="BANKGB2LXXX",
            )

        # Reconcile
        report = await service.reconcile_settlements(date(2025, 1, 17))

        assert "settlement_date" in report
        assert "total_instructions" in report
        assert "settlement_rate" in report
        assert report["total_instructions"] >= 2
