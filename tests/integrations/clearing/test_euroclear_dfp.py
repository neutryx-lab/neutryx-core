"""Tests for Euroclear DFP settlement instructions."""

from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from neutryx.integrations.clearing.euroclear.messages import (  # noqa: E402
    EuroclearSettlementInstruction,
    SettlementStatus,
    SettlementType,
)
from neutryx.integrations.clearing.swift.base import SwiftValidationError  # noqa: E402


def _build_base_instruction(**overrides) -> EuroclearSettlementInstruction:
    """Helper to build a baseline DFP instruction for tests."""

    defaults = {
        "instruction_id": "DFP123456",
        "sender_reference": "REF123456789",
        "settlement_type": SettlementType.DFP,
        "settlement_date": date(2024, 5, 17),
        "trade_date": date(2024, 5, 15),
        "isin": "US1234567890",
        "quantity": Decimal("1500"),
        "delivering_party": "DELVUS33XXX",
        "receiving_party": "RECVUS33XXX",
        "participant_bic": "PARTEBICXXX",
        "safekeeping_account": "SAFE123456",
        "status": SettlementStatus.MATCHED,
    }

    defaults.update(overrides)
    return EuroclearSettlementInstruction(**defaults)


def test_dfp_instruction_generates_mt542_swift():
    """Ensure DFP instructions produce an MT542 message without payment fields."""

    instruction = _build_base_instruction()

    swift_message = instruction.to_mt540()

    assert "{2:O542" in swift_message
    assert ":23:DFRE" in swift_message
    assert ":36B::SETT//1500.00" in swift_message
    assert ":95P::ACCW//DELVUS33XXX" in swift_message
    assert ":19A:" not in swift_message  # No settlement amount for DFP


def test_dfp_instruction_rejects_payment_amount():
    """DFP instructions must not include settlement payment fields."""

    instruction = _build_base_instruction(settlement_amount=Decimal("100"))

    with pytest.raises(SwiftValidationError):
        instruction.to_mt540()


def test_dfp_instruction_requires_valid_status():
    """DFP instructions should enforce allowed status values."""

    instruction = _build_base_instruction(status=SettlementStatus.SETTLED)

    with pytest.raises(SwiftValidationError):
        instruction.to_mt540()

