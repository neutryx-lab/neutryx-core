"""Tests for SWIFT messaging (MT and MX formats)."""

import pytest
from datetime import date, datetime
from decimal import Decimal

from neutryx.integrations.clearing.swift.base import SwiftValidationError
from neutryx.integrations.clearing.swift.mt import MT540, MT542, MT543, MT544
from neutryx.integrations.clearing.swift.mx import PACS008, SETR002


class TestMT540:
    """Tests for MT540 (Receive Free) message."""

    def test_create_mt540(self):
        """Test creating MT540 message."""
        mt540 = MT540(
            sender_bic="BANKGB2LXXX",
            receiver_bic="MGTCBEBEECL",
            message_ref="REF123456",
            sender_reference="SENDER001",
            trade_date=date(2025, 1, 15),
            settlement_date=date(2025, 1, 17),
            isin="US0378331005",
            quantity=Decimal("1000"),
            security_description="Apple Inc Common Stock",
            account_owner="CLIENT001",
            safekeeping_account="SAFE001",
            place_of_settlement="EUROCLEAR",
        )

        assert mt540.message_type == "MT540"
        assert mt540.isin == "US0378331005"
        assert mt540.quantity == Decimal("1000")
        assert mt540.validate()

    def test_mt540_to_swift(self):
        """Test converting MT540 to SWIFT format."""
        mt540 = MT540(
            sender_bic="BANKGB2LXXX",
            receiver_bic="MGTCBEBEECL",
            message_ref="REF123456",
            sender_reference="SENDER001",
            trade_date=date(2025, 1, 15),
            settlement_date=date(2025, 1, 17),
            isin="US0378331005",
            quantity=Decimal("1000"),
            account_owner="CLIENT001",
            safekeeping_account="SAFE001",
            place_of_settlement="EUROCLEAR",
        )

        swift_msg = mt540.to_swift()
        assert ":20:SENDER001" in swift_msg
        assert ":35B:ISIN US0378331005" in swift_msg
        assert ":23:RFRE" in swift_msg

    def test_mt540_invalid_isin(self):
        """Test MT540 with invalid ISIN."""
        with pytest.raises(ValueError, match="Invalid ISIN"):
            MT540(
                sender_bic="BANKGB2LXXX",
                receiver_bic="MGTCBEBEECL",
                message_ref="REF123456",
                sender_reference="SENDER001",
                trade_date=date(2025, 1, 15),
                settlement_date=date(2025, 1, 17),
                isin="INVALID",
                quantity=Decimal("1000"),
                account_owner="CLIENT001",
                safekeeping_account="SAFE001",
                place_of_settlement="EUROCLEAR",
            )

    def test_mt540_validation_settlement_before_trade(self):
        """Test validation fails when settlement date before trade date."""
        mt540 = MT540(
            sender_bic="BANKGB2LXXX",
            receiver_bic="MGTCBEBEECL",
            message_ref="REF123456",
            sender_reference="SENDER001",
            trade_date=date(2025, 1, 17),
            settlement_date=date(2025, 1, 15),  # Before trade date
            isin="US0378331005",
            quantity=Decimal("1000"),
            account_owner="CLIENT001",
            safekeeping_account="SAFE001",
            place_of_settlement="EUROCLEAR",
        )

        with pytest.raises(SwiftValidationError, match="Settlement date cannot be before trade date"):
            mt540.validate()


class TestMT542:
    """Tests for MT542 (Deliver Free) message."""

    def test_create_mt542(self):
        """Test creating MT542 message."""
        mt542 = MT542(
            sender_bic="BANKGB2LXXX",
            receiver_bic="MGTCBEBEECL",
            message_ref="REF123456",
            sender_reference="SENDER002",
            trade_date=date(2025, 1, 15),
            settlement_date=date(2025, 1, 17),
            isin="US5949181045",
            quantity=Decimal("500"),
            account_owner="CLIENT001",
            safekeeping_account="SAFE001",
            place_of_settlement="EUROCLEAR",
            receiving_agent="BANKUS33XXX",
        )

        assert mt542.message_type == "MT542"
        assert mt542.isin == "US5949181045"
        assert mt542.validate()


class TestMT543:
    """Tests for MT543 (Deliver Against Payment) message."""

    def test_create_mt543(self):
        """Test creating MT543 message."""
        mt543 = MT543(
            sender_bic="BANKGB2LXXX",
            receiver_bic="MGTCBEBEECL",
            message_ref="REF123456",
            sender_reference="SENDER003",
            trade_date=date(2025, 1, 15),
            settlement_date=date(2025, 1, 17),
            isin="US0378331005",
            quantity=Decimal("1000"),
            settlement_amount=Decimal("150000.00"),
            settlement_currency="USD",
            account_owner="CLIENT001",
            safekeeping_account="SAFE001",
            place_of_settlement="EUROCLEAR",
        )

        assert mt543.message_type == "MT543"
        assert mt543.settlement_amount == Decimal("150000.00")
        assert mt543.validate()


class TestPACS008:
    """Tests for pacs.008 (FI to FI Customer Credit Transfer) message."""

    def test_create_pacs008(self):
        """Test creating pacs.008 message."""
        pacs008 = PACS008(
            sender_bic="BANKGB2LXXX",
            receiver_bic="BANKUS33XXX",
            message_ref="MSG123456",
            instruction_id="INSTR001",
            end_to_end_id="E2E001",
            payment_amount=Decimal("100000.00"),
            payment_currency="USD",
            value_date=date(2025, 1, 17),
            debtor_name="ABC Corporation",
            debtor_account="GB29NWBK60161331926819",
            debtor_agent_bic="BANKGB2LXXX",
            creditor_name="XYZ Limited",
            creditor_account="US64SVBKUS6S3300958879",
            creditor_agent_bic="BANKUS33XXX",
        )

        assert pacs008.message_type == "pacs.008"
        assert pacs008.payment_currency == "USD"
        assert pacs008.validate()

    def test_pacs008_to_xml(self):
        """Test converting pacs.008 to XML format."""
        pacs008 = PACS008(
            sender_bic="BANKGB2LXXX",
            receiver_bic="BANKUS33XXX",
            message_ref="MSG123456",
            instruction_id="INSTR001",
            end_to_end_id="E2E001",
            payment_amount=Decimal("100000.00"),
            payment_currency="USD",
            value_date=date(2025, 1, 17),
            debtor_name="ABC Corporation",
            debtor_account="GB29NWBK60161331926819",
            debtor_agent_bic="BANKGB2LXXX",
            creditor_name="XYZ Limited",
            creditor_account="US64SVBKUS6S3300958879",
            creditor_agent_bic="BANKUS33XXX",
        )

        xml_msg = pacs008.to_swift()
        assert '<?xml version="1.0"' in xml_msg
        assert '<Document' in xml_msg
        assert '<InstrId>INSTR001</InstrId>' in xml_msg
        assert 'Ccy="USD"' in xml_msg


class TestSETR002:
    """Tests for setr.002 (Redemption Order) message."""

    def test_create_setr002(self):
        """Test creating setr.002 message."""
        setr002 = SETR002(
            sender_bic="FUNDMGRGB2L",
            receiver_bic="CUSTODYUS33",
            message_ref="MSG789012",
            order_reference="ORD001",
            isin="IE00B4L5Y983",
            units_number=Decimal("1000.00"),
            trade_date=date(2025, 1, 15),
            settlement_date=date(2025, 1, 17),
            investor_name="John Doe",
            investor_account="ACCT001",
            settlement_currency="USD",
            settlement_account="US64SVBKUS6S3300958879",
        )

        assert setr002.message_type == "setr.002"
        assert setr002.isin == "IE00B4L5Y983"
        assert setr002.validate()

    def test_setr002_invalid_isin(self):
        """Test setr.002 with invalid ISIN."""
        with pytest.raises(ValueError, match="Invalid ISIN"):
            SETR002(
                sender_bic="FUNDMGRGB2L",
                receiver_bic="CUSTODYUS33",
                message_ref="MSG789012",
                order_reference="ORD001",
                isin="INVALID123",
                units_number=Decimal("1000.00"),
                trade_date=date(2025, 1, 15),
                settlement_date=date(2025, 1, 17),
                investor_name="John Doe",
                investor_account="ACCT001",
                settlement_currency="USD",
                settlement_account="US64SVBKUS6S3300958879",
            )
