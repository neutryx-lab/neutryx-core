import pytest
from datetime import date, datetime
from decimal import Decimal

from neutryx.integrations.clearing.swift.mt import MT543, MT544, SwiftValidationError
from neutryx.integrations.clearing.swift.mx import SETR002
from neutryx.integrations.clearing.euroclear.messages import (
    EuroclearSettlementInstruction,
    SettlementStatus,
    SettlementType,
)


def test_mt543_round_trip():
    mt543 = MT543(
        sender_bic="FOOBUS33XXX",
        receiver_bic="MGTCBEBEECL",
        message_ref="REF123",
        sender_reference="REF123",
        trade_date=date(2024, 1, 10),
        settlement_date=date(2024, 1, 12),
        isin="US1234567890",
        quantity=Decimal("1500.25"),
        settlement_amount=Decimal("100000.50"),
        settlement_currency="USD",
        account_owner="DELIVERACCT",
        safekeeping_account="SAFE12345",
        place_of_settlement="MGTCBEBEECL",
    )

    swift_text = mt543.to_swift()
    parsed = MT543.from_swift(swift_text)

    assert parsed.sender_reference == mt543.sender_reference
    assert parsed.trade_date == mt543.trade_date
    assert parsed.settlement_date == mt543.settlement_date
    assert parsed.quantity == mt543.quantity
    assert parsed.settlement_amount == mt543.settlement_amount
    assert parsed.settlement_currency == mt543.settlement_currency
    assert parsed.place_of_settlement == mt543.place_of_settlement
    assert parsed.validate()

    without_amount = "\n".join(
        line for line in swift_text.splitlines() if not line.startswith(":19A")
    )
    with pytest.raises(SwiftValidationError):
        MT543.from_swift(without_amount)


def test_mt544_round_trip():
    mt544 = MT544(
        sender_bic="BARCUS44XXX",
        receiver_bic="MGTCBEBEECL",
        message_ref="CONF001",
        sender_reference="CONF001",
        related_reference="REF123",
        settlement_date=date(2024, 1, 12),
        isin="US1234567890",
        quantity=Decimal("1500.25"),
        status="SETT",
        settlement_amount=Decimal("100000.50"),
        settlement_currency="USD",
    )

    swift_text = mt544.to_swift()
    parsed = MT544.from_swift(swift_text)

    assert parsed.sender_reference == mt544.sender_reference
    assert parsed.related_reference == mt544.related_reference
    assert parsed.settlement_date == mt544.settlement_date
    assert parsed.quantity == mt544.quantity
    assert parsed.status == mt544.status
    assert parsed.settlement_amount == mt544.settlement_amount
    assert parsed.settlement_currency == mt544.settlement_currency
    assert parsed.validate()


def test_setr002_round_trip():
    redemption = SETR002(
        sender_bic="INVESTBANKXX",
        receiver_bic="FUNDADMINXX",
        message_ref="ORDER001",
        order_reference="ORDER001",
        isin="LU1234567890",
        units_number=Decimal("250.00"),
        trade_date=date(2024, 3, 5),
        settlement_date=date(2024, 3, 12),
        investor_name="Jane Doe",
        investor_account="ACC1234567",
        settlement_currency="EUR",
        settlement_account="BE12345678901234",
    )
    redemption.creation_date = datetime(2024, 3, 5, 10, 30, 0)

    xml_text = redemption.to_swift()
    parsed = SETR002.from_swift(xml_text)

    assert parsed.order_reference == redemption.order_reference
    assert parsed.isin == redemption.isin
    assert parsed.units_number == redemption.units_number
    assert parsed.trade_date == redemption.trade_date
    assert parsed.settlement_date == redemption.settlement_date
    assert parsed.investor_name == redemption.investor_name
    assert parsed.investor_account == redemption.investor_account
    assert parsed.settlement_currency == redemption.settlement_currency
    assert parsed.settlement_account == redemption.settlement_account


def test_euroclear_to_mt543_and_mt544_generation():
    dvp_instruction = EuroclearSettlementInstruction(
        instruction_id="INST001",
        sender_reference="REF001",
        settlement_type=SettlementType.DVP,
        settlement_date=date(2024, 4, 15),
        trade_date=date(2024, 4, 12),
        isin="FR1234567890",
        quantity=Decimal("100"),
        delivering_party="DELIVBICXXX",
        receiving_party="RECEIVBICXX",
        participant_bic="PARTBICXXXX",
        settlement_amount=Decimal("500000"),
        settlement_currency="EUR",
        payment_bank_bic="PAYMENTBIC",
        safekeeping_account="SAFE9999",
    )

    mt543_text = dvp_instruction.to_mt540()
    assert ":19A:" in mt543_text
    parsed_mt543 = MT543.from_swift(mt543_text)
    assert parsed_mt543.settlement_amount == dvp_instruction.settlement_amount
    assert parsed_mt543.settlement_currency == dvp_instruction.settlement_currency
    assert parsed_mt543.account_owner == dvp_instruction.delivering_party

    rvp_instruction = EuroclearSettlementInstruction(
        instruction_id="INST002",
        sender_reference="REF002",
        linked_reference="INST001",
        settlement_type=SettlementType.RVP,
        settlement_date=date(2024, 4, 15),
        trade_date=date(2024, 4, 12),
        isin="FR1234567890",
        quantity=Decimal("100"),
        delivering_party="DELIVBICXXX",
        receiving_party="RECEIVBICXX",
        participant_bic="PARTBICXXXX",
        settlement_amount=Decimal("500000"),
        settlement_currency="EUR",
        status=SettlementStatus.MATCHED,
    )

    mt544_text = rvp_instruction.to_mt540()
    assert ":19A:" in mt544_text
    parsed_mt544 = MT544.from_swift(mt544_text)
    assert parsed_mt544.settlement_amount == rvp_instruction.settlement_amount
    assert parsed_mt544.settlement_currency == rvp_instruction.settlement_currency
    assert parsed_mt544.related_reference == rvp_instruction.linked_reference
