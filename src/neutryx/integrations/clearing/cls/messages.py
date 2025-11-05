"""CLS message formats and data models."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CLSSettlementStatus(str, Enum):
    """CLS settlement status."""
    PENDING = "pending"
    MATCHED = "matched"
    SETTLED = "settled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class CLSCurrency(str, Enum):
    """CLS-eligible currencies (18 major currencies)."""
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    CNY = "CNY"
    DKK = "DKK"
    EUR = "EUR"
    GBP = "GBP"
    HKD = "HKD"
    HUF = "HUF"
    ILS = "ILS"
    JPY = "JPY"
    KRW = "KRW"
    MXN = "MXN"
    NOK = "NOK"
    NZD = "NZD"
    SEK = "SEK"
    SGD = "SGD"
    USD = "USD"


class CLSSettlementInstruction(BaseModel):
    """CLS settlement instruction for FX trade.

    Represents a single side (pay or receive) of an FX settlement.
    """

    # Instruction identification
    instruction_id: str = Field(..., description="Unique instruction ID")
    trade_id: str = Field(..., description="Related trade ID")
    settlement_session_id: str = Field(..., description="CLS settlement session ID")

    # Trade details
    buy_currency: CLSCurrency = Field(..., description="Currency being bought")
    buy_amount: Decimal = Field(..., description="Amount being bought")
    sell_currency: CLSCurrency = Field(..., description="Currency being sold")
    sell_amount: Decimal = Field(..., description="Amount being sold")
    fx_rate: Decimal = Field(..., description="FX rate")

    # Settlement details
    value_date: date = Field(..., description="Settlement value date")
    settlement_date: date = Field(..., description="CLS settlement date")

    # Party information
    submitter_bic: str = Field(..., description="Submitting party BIC")
    counterparty_bic: str = Field(..., description="Counterparty BIC")
    settlement_member: str = Field(..., description="CLS settlement member code")

    # Optional fields
    execution_timestamp: Optional[datetime] = Field(None, description="Trade execution time")
    settlement_method: str = Field(default="PVP", description="Settlement method (always PVP for CLS)")
    original_instruction_id: Optional[str] = Field(None, description="Original instruction if amended")

    # Status tracking
    status: CLSSettlementStatus = Field(default=CLSSettlementStatus.PENDING)
    submission_timestamp: datetime = Field(default_factory=datetime.utcnow)

    def validate_currencies(self) -> bool:
        """Validate that currencies are CLS-eligible and different."""
        if self.buy_currency == self.sell_currency:
            raise ValueError("Buy and sell currencies must be different")
        return True

    def calculate_notional_usd(self, usd_rates: dict[str, Decimal]) -> Decimal:
        """Calculate notional value in USD equivalent.

        Args:
            usd_rates: Dictionary of currency to USD conversion rates

        Returns:
            Notional value in USD
        """
        if self.buy_currency == CLSCurrency.USD:
            return self.buy_amount
        elif self.sell_currency == CLSCurrency.USD:
            return self.sell_amount
        else:
            # Convert buy currency to USD
            buy_ccy = self.buy_currency.value
            if buy_ccy in usd_rates:
                return self.buy_amount * usd_rates[buy_ccy]
            return Decimal(0)

    def to_cls_message(self) -> str:
        """Convert to CLS proprietary message format.

        Returns:
            CLS-formatted message string
        """
        lines = [
            f"MSG_TYPE=SETTLEMENT_INSTRUCTION",
            f"INSTRUCTION_ID={self.instruction_id}",
            f"TRADE_ID={self.trade_id}",
            f"SESSION_ID={self.settlement_session_id}",
            f"BUY_CCY={self.buy_currency.value}",
            f"BUY_AMT={float(self.buy_amount):.2f}",
            f"SELL_CCY={self.sell_currency.value}",
            f"SELL_AMT={float(self.sell_amount):.2f}",
            f"FX_RATE={float(self.fx_rate):.6f}",
            f"VALUE_DATE={self.value_date.strftime('%Y%m%d')}",
            f"SETTLEMENT_DATE={self.settlement_date.strftime('%Y%m%d')}",
            f"SUBMITTER_BIC={self.submitter_bic}",
            f"COUNTERPARTY_BIC={self.counterparty_bic}",
            f"SETTLEMENT_MEMBER={self.settlement_member}",
            f"SETTLEMENT_METHOD={self.settlement_method}",
            f"STATUS={self.status.value}",
        ]
        return "\n".join(lines)


class CLSConfirmation(BaseModel):
    """CLS settlement confirmation message."""

    confirmation_id: str = Field(..., description="Confirmation ID")
    instruction_id: str = Field(..., description="Related instruction ID")
    trade_id: str = Field(..., description="Trade ID")

    # Settlement details
    settlement_date: date = Field(..., description="Actual settlement date")
    settlement_time: datetime = Field(..., description="Settlement timestamp")

    # Amounts
    buy_currency: str = Field(..., description="Currency bought")
    buy_amount: Decimal = Field(..., description="Amount bought")
    sell_currency: str = Field(..., description="Currency sold")
    sell_amount: Decimal = Field(..., description="Amount sold")

    # Status
    status: CLSSettlementStatus = Field(..., description="Final settlement status")
    confirmation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional rejection details
    rejection_reason: Optional[str] = Field(None, description="Rejection reason if applicable")
    rejection_code: Optional[str] = Field(None, description="Rejection code")

    def is_successful(self) -> bool:
        """Check if settlement was successful."""
        return self.status == CLSSettlementStatus.SETTLED

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "confirmation_id": self.confirmation_id,
            "instruction_id": self.instruction_id,
            "trade_id": self.trade_id,
            "settlement_date": self.settlement_date.isoformat(),
            "settlement_time": self.settlement_time.isoformat(),
            "buy_currency": self.buy_currency,
            "buy_amount": float(self.buy_amount),
            "sell_currency": self.sell_currency,
            "sell_amount": float(self.sell_amount),
            "status": self.status.value,
            "is_successful": self.is_successful(),
            "rejection_reason": self.rejection_reason,
            "rejection_code": self.rejection_code,
        }


class CLSStatus(BaseModel):
    """CLS settlement status query response."""

    instruction_id: str = Field(..., description="Instruction ID")
    trade_id: str = Field(..., description="Trade ID")
    status: CLSSettlementStatus = Field(..., description="Current status")
    last_updated: datetime = Field(..., description="Last status update time")

    # Payment details
    pay_in_complete: bool = Field(default=False, description="Pay-in complete")
    pay_out_complete: bool = Field(default=False, description="Pay-out complete")

    # Optional message
    status_message: Optional[str] = Field(None, description="Status message")

    def is_final(self) -> bool:
        """Check if status is final (settled, failed, cancelled)."""
        return self.status in (
            CLSSettlementStatus.SETTLED,
            CLSSettlementStatus.FAILED,
            CLSSettlementStatus.CANCELLED,
            CLSSettlementStatus.REJECTED,
        )


__all__ = [
    "CLSSettlementInstruction",
    "CLSConfirmation",
    "CLSStatus",
    "CLSSettlementStatus",
    "CLSCurrency",
]
