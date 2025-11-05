"""Tests for settlement instructions and corporate actions."""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal

from neutryx.integrations.clearing.base import Party, ProductType
from neutryx.integrations.clearing.settlement_instructions import (
    SettlementInstruction,
    SettlementType,
    SettlementStatus,
    SettlementMethod,
    FailReason,
    CashFlow,
    SecuritiesMovement,
    SettlementInstructionGenerator,
)
from neutryx.integrations.clearing.corporate_actions import (
    CorporateActionType,
    CorporateActionStatus,
    CorporateActionEvent,
    ElectionType,
    PaymentType,
    DividendTerms,
    SplitTerms,
    Position,
    CorporateActionProcessor,
)


@pytest.fixture
def deliverer():
    """Delivering party."""
    return Party(party_id="SELLER001", name="Seller Bank", bic="SELLBANKXXX")


@pytest.fixture
def receiver():
    """Receiving party."""
    return Party(party_id="BUYER001", name="Buyer Bank", bic="BUYRBANKXXX")


class TestSettlementInstructions:
    """Test settlement instruction generation."""

    def test_generate_instruction(self, deliverer, receiver):
        """Test generating settlement instruction."""
        generator = SettlementInstructionGenerator()

        instruction = generator.generate_instruction(
            trade_id="TRD-001",
            trade_date=date.today(),
            settlement_date=date.today() + timedelta(days=2),
            deliverer=deliverer,
            receiver=receiver,
            settlement_type=SettlementType.DVP
        )

        assert instruction.instruction_id.startswith("SI-")
        assert instruction.status == SettlementStatus.PENDING
        assert instruction.trade_id == "TRD-001"

    def test_add_cash_flow(self, deliverer, receiver):
        """Test adding cash flow to instruction."""
        generator = SettlementInstructionGenerator()

        instruction = generator.generate_instruction(
            trade_id="TRD-001",
            trade_date=date.today(),
            settlement_date=date.today() + timedelta(days=2),
            deliverer=deliverer,
            receiver=receiver
        )

        cash_flow = CashFlow(
            amount=Decimal("1000000"),
            currency="USD",
            direction="pay",
            payment_date=date.today() + timedelta(days=2),
            value_date=date.today() + timedelta(days=2),
            payer_account="ACC001",
            receiver_account="ACC002"
        )

        generator.add_cash_flow(instruction.instruction_id, cash_flow)

        assert len(instruction.cash_flows) == 1

    def test_settle_instruction(self, deliverer, receiver):
        """Test settling instruction."""
        generator = SettlementInstructionGenerator()

        instruction = generator.generate_instruction(
            trade_id="TRD-001",
            trade_date=date.today(),
            settlement_date=date.today() + timedelta(days=2),
            deliverer=deliverer,
            receiver=receiver
        )

        cash_flow = CashFlow(
            amount=Decimal("1000000"),
            currency="USD",
            direction="pay",
            payment_date=date.today(),
            value_date=date.today(),
            payer_account="ACC001",
            receiver_account="ACC002"
        )
        generator.add_cash_flow(instruction.instruction_id, cash_flow)

        # Settle
        settled = generator.settle_instruction(instruction.instruction_id)

        assert settled.status == SettlementStatus.SETTLED
        assert settled.actual_settlement_date is not None

    def test_fail_instruction(self, deliverer, receiver):
        """Test failing instruction."""
        generator = SettlementInstructionGenerator()

        instruction = generator.generate_instruction(
            trade_id="TRD-001",
            trade_date=date.today(),
            settlement_date=date.today() + timedelta(days=2),
            deliverer=deliverer,
            receiver=receiver
        )

        failed = generator.fail_instruction(
            instruction.instruction_id,
            FailReason.INSUFFICIENT_CASH,
            "Insufficient funds in account"
        )

        assert failed.status == SettlementStatus.FAILED
        assert failed.fail_reason == FailReason.INSUFFICIENT_CASH

    def test_retry_instruction(self, deliverer, receiver):
        """Test retrying failed instruction."""
        generator = SettlementInstructionGenerator()

        instruction = generator.generate_instruction(
            trade_id="TRD-001",
            trade_date=date.today(),
            settlement_date=date.today(),
            deliverer=deliverer,
            receiver=receiver
        )

        generator.fail_instruction(
            instruction.instruction_id,
            FailReason.INSUFFICIENT_CASH
        )

        # Retry
        retried = generator.retry_instruction(
            instruction.instruction_id,
            new_settlement_date=date.today() + timedelta(days=1)
        )

        assert retried.status == SettlementStatus.PENDING
        assert retried.retry_count == 1


class TestCorporateActions:
    """Test corporate action processing."""

    def test_dividend_event_creation(self):
        """Test creating dividend event."""
        event = CorporateActionEvent(
            event_type=CorporateActionType.CASH_DIVIDEND,
            security_id="US0378331005",
            security_name="Apple Inc",
            issuer="Apple Inc",
            announcement_date=date.today(),
            ex_date=date.today() + timedelta(days=7),
            record_date=date.today() + timedelta(days=8),
            payment_date=date.today() + timedelta(days=30),
            election_type=ElectionType.MANDATORY,
            terms={
                "dividend_rate": "0.25",
                "currency": "USD",
                "payment_type": "cash"
            },
            description="Quarterly dividend"
        )

        assert event.event_id.startswith("CA-")
        assert event.event_type == CorporateActionType.CASH_DIVIDEND

    def test_dividend_entitlement_calculation(self):
        """Test calculating dividend entitlements."""
        processor = CorporateActionProcessor()

        # Create dividend event
        event = CorporateActionEvent(
            event_type=CorporateActionType.CASH_DIVIDEND,
            security_id="US0378331005",
            security_name="Apple Inc",
            issuer="Apple Inc",
            announcement_date=date.today(),
            ex_date=date.today() + timedelta(days=7),
            record_date=date.today() + timedelta(days=8),
            payment_date=date.today() + timedelta(days=30),
            election_type=ElectionType.MANDATORY,
            terms={},
            description="Quarterly dividend"
        )
        processor.add_event(event)

        # Create position
        holder = Party(party_id="HOLD001", name="Holder", bic="HOLDXXXX")
        position = Position(
            security_id="US0378331005",
            holder=holder,
            quantity=Decimal("1000"),
            account="ACC001",
            as_of_date=date.today() + timedelta(days=8),
            record_date_position=True
        )
        processor.add_position(position)

        # Calculate entitlement
        terms = DividendTerms(
            dividend_rate=Decimal("0.25"),
            currency="USD",
            payment_type=PaymentType.CASH
        )

        entitlement = processor.calculate_dividend_entitlement(
            event.event_id,
            position,
            terms
        )

        assert entitlement.entitled_quantity == Decimal("1000")
        assert entitlement.cash_entitlement == Decimal("250")  # 1000 * 0.25

    def test_stock_split(self):
        """Test stock split processing."""
        processor = CorporateActionProcessor()

        # Create split event
        event = CorporateActionEvent(
            event_type=CorporateActionType.STOCK_SPLIT,
            security_id="US0378331005",
            security_name="Apple Inc",
            issuer="Apple Inc",
            announcement_date=date.today(),
            ex_date=date.today() + timedelta(days=7),
            record_date=date.today() + timedelta(days=7),
            payment_date=date.today() + timedelta(days=8),
            election_type=ElectionType.MANDATORY,
            terms={},
            description="2-for-1 stock split"
        )
        processor.add_event(event)

        # Create position
        holder = Party(party_id="HOLD001", name="Holder", bic="HOLDXXXX")
        position = Position(
            security_id="US0378331005",
            holder=holder,
            quantity=Decimal("100"),
            account="ACC001",
            as_of_date=date.today() + timedelta(days=7),
            record_date_position=True
        )
        processor.add_position(position)

        # Apply split
        terms = SplitTerms(
            split_ratio="2:1",
            old_shares=1,
            new_shares=2
        )

        new_qty, cash_in_lieu = processor.calculate_split_adjustment(
            event.event_id,
            position,
            terms
        )

        assert new_qty == Decimal("200")  # 100 * 2
        assert position.quantity == Decimal("200")

    def test_submit_election(self):
        """Test submitting election for voluntary event."""
        processor = CorporateActionProcessor()

        # Create voluntary event
        event = CorporateActionEvent(
            event_type=CorporateActionType.TENDER_OFFER,
            security_id="US0378331005",
            security_name="Apple Inc",
            issuer="Apple Inc",
            announcement_date=date.today(),
            ex_date=date.today() + timedelta(days=7),
            record_date=date.today() + timedelta(days=7),
            payment_date=date.today() + timedelta(days=30),
            election_type=ElectionType.VOLUNTARY,
            election_deadline=date.today() + timedelta(days=20),
            default_option="no_action",
            requires_election=True,
            terms={},
            description="Tender offer at $100/share"
        )
        processor.add_event(event)

        # Create position
        holder = Party(party_id="HOLD001", name="Holder", bic="HOLDXXXX")
        position = Position(
            security_id="US0378331005",
            holder=holder,
            quantity=Decimal("500"),
            account="ACC001",
            as_of_date=date.today() + timedelta(days=7),
            record_date_position=True
        )
        processor.add_position(position)

        # Submit election
        election = processor.submit_election(
            event.event_id,
            position.position_id,
            elected_option="tender",
            elected_quantity=Decimal("300")
        )

        assert election.elected_option == "tender"
        assert election.elected_quantity == Decimal("300")
        assert not election.is_default


class TestStatistics:
    """Test statistics tracking."""

    def test_settlement_statistics(self, deliverer, receiver):
        """Test settlement statistics."""
        generator = SettlementInstructionGenerator()

        # Create and settle instruction
        instruction = generator.generate_instruction(
            trade_id="TRD-001",
            trade_date=date.today(),
            settlement_date=date.today(),
            deliverer=deliverer,
            receiver=receiver
        )

        cash_flow = CashFlow(
            amount=Decimal("1000000"),
            currency="USD",
            direction="pay",
            payment_date=date.today(),
            value_date=date.today(),
            payer_account="ACC001",
            receiver_account="ACC002"
        )
        generator.add_cash_flow(instruction.instruction_id, cash_flow)
        generator.settle_instruction(instruction.instruction_id)

        stats = generator.get_statistics()

        assert stats["total_instructions"] == 1
        assert stats["settled_instructions"] == 1
        assert stats["settlement_rate"] == 100.0

    def test_corporate_action_statistics(self):
        """Test corporate action statistics."""
        processor = CorporateActionProcessor()

        event = CorporateActionEvent(
            event_type=CorporateActionType.CASH_DIVIDEND,
            security_id="US0378331005",
            security_name="Apple Inc",
            issuer="Apple Inc",
            announcement_date=date.today(),
            ex_date=date.today() + timedelta(days=7),
            record_date=date.today() + timedelta(days=8),
            payment_date=date.today() + timedelta(days=30),
            election_type=ElectionType.MANDATORY,
            terms={},
            description="Dividend"
        )
        processor.add_event(event)

        stats = processor.get_statistics()

        assert stats["total_events"] == 1
        assert stats["pending_events"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
