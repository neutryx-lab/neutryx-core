"""SWIFT MX (ISO 20022 XML) format implementation.

MX messages are the modern XML-based SWIFT format (ISO 20022) that
are replacing MT messages. This module implements key MX message types
for payment and settlement instructions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, datetime
from decimal import Decimal, DecimalException
from typing import Any, Dict, Optional

# Use defusedxml for parsing untrusted XML to prevent XXE attacks
import defusedxml.ElementTree as DefusedET

from pydantic import Field, field_validator

from .base import SwiftMessage, SwiftMessageType, SwiftParseError, SwiftValidationError


class MXMessage(SwiftMessage):
    """Base class for MX (ISO 20022 XML) SWIFT messages."""

    namespace: str = "urn:iso:std:iso:20022:tech:xsd"

    def _create_root(self, doc_name: str) -> ET.Element:
        """Create root XML element with namespace.

        Args:
            doc_name: Document name (e.g., FIToFICstmrCdtTrf)

        Returns:
            Root XML element
        """
        root = ET.Element(
            f"{{{self.namespace}:{doc_name.split('.')[0]}}}{doc_name}",
            attrib={
                "xmlns": f"{self.namespace}:{doc_name.split('.')[0]}",
            }
        )
        return root

    def _add_element(self, parent: ET.Element, tag: str, text: Optional[str] = None) -> ET.Element:
        """Add child element to parent.

        Args:
            parent: Parent XML element
            tag: Child element tag
            text: Optional text content

        Returns:
            Created child element
        """
        child = ET.SubElement(parent, tag)
        if text is not None:
            child.text = text
        return child

    def _format_date(self, d: date) -> str:
        """Format date as ISO 8601 (YYYY-MM-DD)."""
        return d.strftime("%Y-%m-%d")

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime as ISO 8601."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _format_amount(self, amount: Decimal, currency: str) -> Dict[str, str]:
        """Format monetary amount.

        Args:
            amount: Amount value
            currency: Currency code

        Returns:
            Dictionary with Ccy and amount
        """
        return {
            "Ccy": currency,
            "amount": f"{float(amount):.2f}"
        }


class PACS008(MXMessage):
    """pacs.008 - FIToFICustomerCreditTransfer.

    Financial Institution to Financial Institution Customer Credit Transfer.
    Used for cross-border payments and FX settlement.
    """

    message_type: str = Field(default=SwiftMessageType.PACS_008.value, frozen=True)

    # Group Header
    instruction_id: str = Field(..., description="Unique instruction ID")
    end_to_end_id: str = Field(..., description="End-to-end ID")

    # Payment details
    payment_amount: Decimal = Field(..., description="Payment amount")
    payment_currency: str = Field(..., description="Payment currency (ISO 4217)")
    value_date: date = Field(..., description="Value date")

    # Debtor (payer)
    debtor_name: str = Field(..., description="Debtor/payer name")
    debtor_account: str = Field(..., description="Debtor account (IBAN)")
    debtor_agent_bic: str = Field(..., description="Debtor agent BIC")

    # Creditor (receiver)
    creditor_name: str = Field(..., description="Creditor/receiver name")
    creditor_account: str = Field(..., description="Creditor account (IBAN)")
    creditor_agent_bic: str = Field(..., description="Creditor agent BIC")

    # Optional
    remittance_info: Optional[str] = Field(None, description="Remittance information")
    charge_bearer: str = Field(default="SHAR", description="Charge bearer: DEBT/CRED/SHAR")

    @field_validator("payment_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        if len(v) != 3:
            raise ValueError(f"Invalid currency code: {v}")
        return v.upper()

    def to_swift(self) -> str:
        """Convert to pacs.008 XML format."""
        # Create root element
        root = ET.Element("Document", xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08")
        fi_to_fi_cct = ET.SubElement(root, "FIToFICstmrCdtTrf")

        # Group Header
        grp_hdr = ET.SubElement(fi_to_fi_cct, "GrpHdr")
        self._add_element(grp_hdr, "MsgId", self.message_ref)
        self._add_element(grp_hdr, "CreDtTm", self._format_datetime(self.creation_date))
        self._add_element(grp_hdr, "NbOfTxs", "1")

        # Credit Transfer Transaction Information
        cdt_trf_tx_inf = ET.SubElement(fi_to_fi_cct, "CdtTrfTxInf")

        # Payment Identification
        pmt_id = ET.SubElement(cdt_trf_tx_inf, "PmtId")
        self._add_element(pmt_id, "InstrId", self.instruction_id)
        self._add_element(pmt_id, "EndToEndId", self.end_to_end_id)

        # Interbank Settlement Amount
        intrbnk_sttlm_amt = self._add_element(cdt_trf_tx_inf, "IntrBkSttlmAmt", f"{float(self.payment_amount):.2f}")
        intrbnk_sttlm_amt.set("Ccy", self.payment_currency)

        # Interbank Settlement Date
        self._add_element(cdt_trf_tx_inf, "IntrBkSttlmDt", self._format_date(self.value_date))

        # Charge Bearer
        self._add_element(cdt_trf_tx_inf, "ChrgBr", self.charge_bearer)

        # Debtor Agent
        dbtr_agt = ET.SubElement(cdt_trf_tx_inf, "DbtrAgt")
        fin_instn_id = ET.SubElement(dbtr_agt, "FinInstnId")
        bicfi = ET.SubElement(fin_instn_id, "BICFI")
        bicfi.text = self.debtor_agent_bic

        # Debtor
        dbtr = ET.SubElement(cdt_trf_tx_inf, "Dbtr")
        self._add_element(dbtr, "Nm", self.debtor_name)

        # Debtor Account
        dbtr_acct = ET.SubElement(cdt_trf_tx_inf, "DbtrAcct")
        id_elem = ET.SubElement(dbtr_acct, "Id")
        self._add_element(id_elem, "IBAN", self.debtor_account)

        # Creditor Agent
        cdtr_agt = ET.SubElement(cdt_trf_tx_inf, "CdtrAgt")
        fin_instn_id2 = ET.SubElement(cdtr_agt, "FinInstnId")
        bicfi2 = ET.SubElement(fin_instn_id2, "BICFI")
        bicfi2.text = self.creditor_agent_bic

        # Creditor
        cdtr = ET.SubElement(cdt_trf_tx_inf, "Cdtr")
        self._add_element(cdtr, "Nm", self.creditor_name)

        # Creditor Account
        cdtr_acct = ET.SubElement(cdt_trf_tx_inf, "CdtrAcct")
        id_elem2 = ET.SubElement(cdtr_acct, "Id")
        self._add_element(id_elem2, "IBAN", self.creditor_account)

        # Remittance Information
        if self.remittance_info:
            rmt_inf = ET.SubElement(cdt_trf_tx_inf, "RmtInf")
            self._add_element(rmt_inf, "Ustrd", self.remittance_info)

        # Convert to string
        ET.indent(root, space="  ")
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")

    @classmethod
    def from_swift(cls, swift_text: str) -> PACS008:
        """Parse pacs.008 from XML."""
        try:
            root = DefusedET.fromstring(swift_text)

            # Navigate XML structure
            ns = {"ns": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08"}
            cdt_trf = root.find(".//ns:CdtTrfTxInf", ns)

            if cdt_trf is None:
                raise SwiftParseError("Invalid pacs.008 structure")

            # Extract fields
            fields = {
                "sender_bic": "UNKNOWN",
                "receiver_bic": "UNKNOWN",
                "message_ref": root.find(".//ns:GrpHdr/ns:MsgId", ns).text,  # type: ignore
                "instruction_id": cdt_trf.find(".//ns:PmtId/ns:InstrId", ns).text,  # type: ignore
                "end_to_end_id": cdt_trf.find(".//ns:PmtId/ns:EndToEndId", ns).text,  # type: ignore
                "payment_amount": Decimal(cdt_trf.find(".//ns:IntrBkSttlmAmt", ns).text),  # type: ignore
                "payment_currency": cdt_trf.find(".//ns:IntrBkSttlmAmt", ns).get("Ccy"),  # type: ignore
                "value_date": datetime.strptime(
                    cdt_trf.find(".//ns:IntrBkSttlmDt", ns).text,  # type: ignore
                    "%Y-%m-%d"
                ).date(),
                "debtor_name": cdt_trf.find(".//ns:Dbtr/ns:Nm", ns).text,  # type: ignore
                "debtor_account": cdt_trf.find(".//ns:DbtrAcct/ns:Id/ns:IBAN", ns).text,  # type: ignore
                "debtor_agent_bic": cdt_trf.find(".//ns:DbtrAgt/ns:FinInstnId/ns:BICFI", ns).text,  # type: ignore
                "creditor_name": cdt_trf.find(".//ns:Cdtr/ns:Nm", ns).text,  # type: ignore
                "creditor_account": cdt_trf.find(".//ns:CdtrAcct/ns:Id/ns:IBAN", ns).text,  # type: ignore
                "creditor_agent_bic": cdt_trf.find(".//ns:CdtrAgt/ns:FinInstnId/ns:BICFI", ns).text,  # type: ignore
            }

            return cls(**fields)

        except (ET.ParseError, AttributeError) as e:
            raise SwiftParseError(f"Failed to parse pacs.008: {e}")

    def validate(self) -> bool:
        """Validate pacs.008 message."""
        if self.payment_amount <= 0:
            raise SwiftValidationError("Payment amount must be positive")

        if self.charge_bearer not in ("DEBT", "CRED", "SHAR"):
            raise SwiftValidationError(f"Invalid charge bearer: {self.charge_bearer}")

        return True


class SETR002(MXMessage):
    """setr.002 - Redemption Order.

    Securities Trade Redemption Order used for fund redemptions.
    """

    message_type: str = Field(default=SwiftMessageType.SETR_002.value, frozen=True)

    # Order details
    order_reference: str = Field(..., description="Order reference")
    isin: str = Field(..., description="Fund ISIN")
    units_number: Decimal = Field(..., description="Number of units to redeem")
    trade_date: date = Field(..., description="Trade date")
    settlement_date: date = Field(..., description="Expected settlement date")

    # Investor details
    investor_name: str = Field(..., description="Investor name")
    investor_account: str = Field(..., description="Investor account")

    # Settlement details
    settlement_currency: str = Field(..., description="Settlement currency")
    settlement_account: str = Field(..., description="Settlement account (IBAN)")

    @field_validator("isin")
    @classmethod
    def validate_isin(cls, v: str) -> str:
        """Validate ISIN format."""
        import re
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", v):
            raise ValueError(f"Invalid ISIN format: {v}")
        return v

    def to_swift(self) -> str:
        """Convert to setr.002 XML format."""
        root = ET.Element("Document", xmlns="urn:iso:std:iso:20022:tech:xsd:setr.002.001.03")
        rmptn_ordr = ET.SubElement(root, "RedOrdr")

        # Message Identification
        msg_id = ET.SubElement(rmptn_ordr, "MsgId")
        self._add_element(msg_id, "Id", self.message_ref)
        self._add_element(msg_id, "CreDtTm", self._format_datetime(self.creation_date))

        # Order Details
        ordr_dtls = ET.SubElement(rmptn_ordr, "OrdrDtls")
        self._add_element(ordr_dtls, "OrdrRef", self.order_reference)
        self._add_element(ordr_dtls, "TradDt", self._format_date(self.trade_date))

        # Financial Instrument
        fin_instrm = ET.SubElement(ordr_dtls, "FinInstrm")
        id_elem = ET.SubElement(fin_instrm, "Id")
        self._add_element(id_elem, "ISIN", self.isin)

        # Units Number
        units_dtls = ET.SubElement(ordr_dtls, "UnitsDtls")
        self._add_element(units_dtls, "UnitsNb", f"{float(self.units_number):.2f}")

        # Investor details
        invstr = ET.SubElement(ordr_dtls, "Invstr")
        self._add_element(invstr, "Nm", self.investor_name)
        invstr_acct = ET.SubElement(invstr, "Acct")
        self._add_element(invstr_acct, "Id", self.investor_account)

        # Settlement Details
        sttlm_dtls = ET.SubElement(ordr_dtls, "SttlmDtls")
        self._add_element(sttlm_dtls, "SttlmDt", self._format_date(self.settlement_date))
        self._add_element(sttlm_dtls, "SttlmCcy", self.settlement_currency)
        sttlm_acct = ET.SubElement(sttlm_dtls, "SttlmAcct")
        acct_id = ET.SubElement(sttlm_acct, "Id")
        self._add_element(acct_id, "IBAN", self.settlement_account)

        # Convert to string
        ET.indent(root, space="  ")
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")

    @classmethod
    def from_swift(cls, swift_text: str) -> SETR002:
        """Parse setr.002 from XML."""
        try:
            root = DefusedET.fromstring(swift_text)
        except (ET.ParseError, DefusedET.ParseError) as exc:
            raise SwiftParseError(f"Failed to parse setr.002 XML: {exc}") from exc

        ns = {"ns": "urn:iso:std:iso:20022:tech:xsd:setr.002.001.03"}
        rmptn_ordr = root.find("ns:RedOrdr", ns)
        if rmptn_ordr is None:
            raise SwiftParseError("Invalid setr.002 structure")

        def get_text(path: str) -> str:
            elem = rmptn_ordr.find(path, ns)
            if elem is None or (elem.text is None or elem.text.strip() == ""):
                raise SwiftValidationError(f"Missing required setr.002 field: {path}")
            return elem.text.strip()

        try:
            creation_date_text = rmptn_ordr.findtext("ns:MsgId/ns:CreDtTm", default="", namespaces=ns)
            fields: Dict[str, Any] = {
                "sender_bic": "UNKNOWN",
                "receiver_bic": "UNKNOWN",
                "message_ref": get_text("ns:MsgId/ns:Id"),
                "order_reference": get_text("ns:OrdrDtls/ns:OrdrRef"),
                "isin": get_text("ns:OrdrDtls/ns:FinInstrm/ns:Id/ns:ISIN"),
                "units_number": Decimal(get_text("ns:OrdrDtls/ns:UnitsDtls/ns:UnitsNb")),
                "trade_date": datetime.strptime(get_text("ns:OrdrDtls/ns:TradDt"), "%Y-%m-%d").date(),
                "settlement_date": datetime.strptime(
                    get_text("ns:OrdrDtls/ns:SttlmDtls/ns:SttlmDt"), "%Y-%m-%d"
                ).date(),
                "settlement_currency": get_text("ns:OrdrDtls/ns:SttlmDtls/ns:SttlmCcy"),
                "settlement_account": get_text(
                    "ns:OrdrDtls/ns:SttlmDtls/ns:SttlmAcct/ns:Id/ns:IBAN"
                ),
                "investor_name": get_text("ns:OrdrDtls/ns:Invstr/ns:Nm"),
                "investor_account": get_text("ns:OrdrDtls/ns:Invstr/ns:Acct/ns:Id"),
            }

            if creation_date_text:
                fields["creation_date"] = datetime.strptime(creation_date_text, "%Y-%m-%dT%H:%M:%S")

            return cls(**fields)
        except (ValueError, DecimalException) as exc:
            raise SwiftValidationError(f"Invalid value in setr.002 message: {exc}") from exc

    def validate(self) -> bool:
        """Validate setr.002 message."""
        if self.units_number <= 0:
            raise SwiftValidationError("Units number must be positive")

        if self.settlement_date < self.trade_date:
            raise SwiftValidationError("Settlement date cannot be before trade date")

        return True


__all__ = [
    "MXMessage",
    "PACS008",
    "SETR002",
]
