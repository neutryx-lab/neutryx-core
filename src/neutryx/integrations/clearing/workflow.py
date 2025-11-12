"""RFQ workflow orchestration and post-trade processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple

from .base import Party, Trade, TradeEconomics
from .confirmation import (
    AffirmationMethod,
    Confirmation,
    ConfirmationDetails,
    ConfirmationStatus,
    ToleranceConfig,
)
from .rfq import AuctionResult, OrderSide, Quote, RFQ, RFQManager, RFQSpecification
from .settlement_instructions import (
    CashFlow,
    SettlementInstruction,
    SettlementMethod,
    SettlementStatus,
    SettlementType,
)


class RFQWorkflowError(Exception):
    """Raised when the RFQ workflow enters an invalid state."""


class RFQWorkflowState(str, Enum):
    """Enumerates the coarse workflow milestones for cleared RFQs."""

    RFQ_CREATED = "rfq_created"
    RFQ_SUBMITTED = "rfq_submitted"
    QUOTE_COLLECTION = "quote_collection"
    AUCTION_EXECUTED = "auction_executed"
    TRADE_BOOKED = "trade_booked"
    CONFIRMATION_MATCHED = "confirmation_matched"
    AFFIRMATION_COMPLETE = "affirmation_complete"
    SETTLEMENT_INSTRUCTED = "settlement_instructed"
    SETTLED = "settled"


@dataclass
class WorkflowSnapshot:
    """Lightweight snapshot returned to API clients and tests."""

    rfq: RFQ
    state: RFQWorkflowState
    auction_result: Optional[AuctionResult] = None
    trade: Optional[Trade] = None
    confirmations: Optional[Tuple[Confirmation, Confirmation]] = None
    settlement_instruction: Optional[SettlementInstruction] = None


class RFQWorkflowService:
    """High-level orchestrator composing RFQ lifecycle primitives."""

    def __init__(self) -> None:
        self.manager = RFQManager()
        self._states: Dict[str, RFQWorkflowState] = {}
        self._trades: Dict[str, Trade] = {}
        self._confirmations: Dict[str, Tuple[Confirmation, Confirmation]] = {}
        self._settlements: Dict[str, SettlementInstruction] = {}

    # ------------------------------------------------------------------
    # RFQ and quote handling
    # ------------------------------------------------------------------
    def create_rfq(
        self,
        requester: Party,
        specification: RFQSpecification,
        side: OrderSide,
        quote_deadline: datetime,
        **kwargs,
    ) -> RFQ:
        rfq = self.manager.create_rfq(
            requester=requester,
            specification=specification,
            side=side,
            quote_deadline=quote_deadline,
            **kwargs,
        )
        self._states[rfq.rfq_id] = RFQWorkflowState.RFQ_CREATED
        return rfq

    def submit_rfq(self, rfq_id: str) -> RFQ:
        rfq = self.manager.submit_rfq(rfq_id)
        self._states[rfq_id] = RFQWorkflowState.RFQ_SUBMITTED
        return rfq

    def submit_quote(self, rfq_id: str, quote: Quote) -> Quote:
        submitted = self.manager.submit_quote(rfq_id, quote)
        self._states[rfq_id] = RFQWorkflowState.QUOTE_COLLECTION
        return submitted

    def execute_auction(self, rfq_id: str, *, force: bool = False) -> AuctionResult:
        rfq = self.manager.get_rfq(rfq_id)
        if not rfq:
            raise RFQWorkflowError(f"RFQ {rfq_id} not found")

        if force and datetime.utcnow() < rfq.quote_deadline:
            # Testing/replay helper – mimic deadline expiry.
            rfq.quote_deadline = datetime.utcnow() - timedelta(milliseconds=1)

        result = self.manager.execute_auction(rfq_id)
        self._states[rfq_id] = RFQWorkflowState.AUCTION_EXECUTED
        return result

    # ------------------------------------------------------------------
    # Trade and state helpers
    # ------------------------------------------------------------------
    def book_trade(
        self,
        rfq_id: str,
        *,
        trade_id: str,
        buyer: Party,
        seller: Party,
        execution_price: Decimal,
        trade_date: datetime,
        effective_date: datetime,
        maturity_date: datetime,
        currency: str,
    ) -> Trade:
        result = self.manager.get_result(rfq_id)
        if not result:
            raise RFQWorkflowError("Auction must be executed before booking trade")

        rfq = self.manager.get_rfq(rfq_id)
        if not rfq:
            raise RFQWorkflowError(f"RFQ {rfq_id} not found")

        economics = TradeEconomics(
            notional=rfq.specification.notional,
            currency=currency,
            price=execution_price,
            fixed_rate=rfq.specification.fixed_rate,
        )

        trade = Trade(
            trade_id=trade_id,
            product_type=rfq.specification.product_type,
            trade_date=trade_date,
            effective_date=effective_date,
            maturity_date=maturity_date,
            buyer=buyer,
            seller=seller,
            economics=economics,
        )

        self._trades[rfq_id] = trade
        self._states[rfq_id] = RFQWorkflowState.TRADE_BOOKED
        return trade

    def attach_confirmations(
        self,
        rfq_id: str,
        confirmations: Tuple[Confirmation, Confirmation],
    ) -> None:
        if self._states.get(rfq_id) != RFQWorkflowState.TRADE_BOOKED:
            raise RFQWorkflowError("Trade must be booked before confirmations are attached")

        self._confirmations[rfq_id] = confirmations
        if all(conf.status == ConfirmationStatus.MATCHED for conf in confirmations):
            self._states[rfq_id] = RFQWorkflowState.CONFIRMATION_MATCHED

    def mark_affirmed(self, rfq_id: str) -> None:
        confirmations = self._confirmations.get(rfq_id)
        if not confirmations:
            raise RFQWorkflowError("No confirmations recorded for RFQ")

        if not all(conf.status == ConfirmationStatus.AFFIRMED for conf in confirmations):
            raise RFQWorkflowError("Confirmations must be affirmed before updating workflow")

        self._states[rfq_id] = RFQWorkflowState.AFFIRMATION_COMPLETE

    def attach_settlement_instruction(
        self, rfq_id: str, instruction: SettlementInstruction
    ) -> None:
        if self._states.get(rfq_id) not in {
            RFQWorkflowState.AFFIRMATION_COMPLETE,
            RFQWorkflowState.SETTLEMENT_INSTRUCTED,
        }:
            raise RFQWorkflowError("Confirmations must be affirmed before settlement instruction")

        self._settlements[rfq_id] = instruction
        self._states[rfq_id] = RFQWorkflowState.SETTLEMENT_INSTRUCTED

    def mark_settled(self, rfq_id: str) -> None:
        instruction = self._settlements.get(rfq_id)
        if not instruction:
            raise RFQWorkflowError("Settlement instruction missing")

        if instruction.status != SettlementStatus.SETTLED:
            raise RFQWorkflowError("Settlement instruction must be marked settled")

        self._states[rfq_id] = RFQWorkflowState.SETTLED

    def snapshot(self, rfq_id: str) -> WorkflowSnapshot:
        rfq = self.manager.get_rfq(rfq_id)
        if not rfq:
            raise RFQWorkflowError(f"RFQ {rfq_id} not found")

        state = self._states.get(rfq_id, RFQWorkflowState.RFQ_CREATED)
        result = self.manager.get_result(rfq_id)
        trade = self._trades.get(rfq_id)
        confirmations = self._confirmations.get(rfq_id)
        settlement = self._settlements.get(rfq_id)

        return WorkflowSnapshot(
            rfq=rfq,
            state=state,
            auction_result=result,
            trade=trade,
            confirmations=confirmations,
            settlement_instruction=settlement,
        )


class PostTradeProcessingService:
    """Generates confirmations and settlement instructions for booked trades."""

    def __init__(self, tolerance: Optional[ToleranceConfig] = None) -> None:
        self.tolerance = tolerance or ToleranceConfig()

    def generate_confirmations(
        self,
        trade: Trade,
        *,
        execution_price: Decimal,
        settlement_date: datetime,
    ) -> Tuple[Confirmation, Confirmation]:
        details = ConfirmationDetails(
            trade_id=trade.trade_id,
            trade_date=trade.trade_date,
            settlement_date=settlement_date,
            buyer=trade.buyer,
            seller=trade.seller,
            product_type=trade.product_type,
            product_description=trade.product_type.value,
            quantity=trade.economics.notional,
            price=execution_price,
            notional=trade.economics.notional,
            currency=trade.economics.currency,
        )

        buy_confirmation = Confirmation(
            originator=trade.buyer,
            recipient=trade.seller,
            direction="buy",
            details=details,
            status=ConfirmationStatus.SENT,
        )

        sell_confirmation = Confirmation(
            originator=trade.seller,
            recipient=trade.buyer,
            direction="sell",
            details=details,
            status=ConfirmationStatus.RECEIVED,
        )

        self._match_confirmations(buy_confirmation, sell_confirmation)
        return buy_confirmation, sell_confirmation

    def _match_confirmations(
        self, confirmation_a: Confirmation, confirmation_b: Confirmation
    ) -> None:
        # Simplified field-by-field comparison respecting tolerances.
        amount_diff = abs(
            confirmation_a.details.notional - confirmation_b.details.notional
        )
        matches = [
            confirmation_a.details.trade_id == confirmation_b.details.trade_id,
            amount_diff <= self.tolerance.amount_tolerance,
            confirmation_a.details.currency == confirmation_b.details.currency,
        ]

        if all(matches):
            confirmation_a.status = ConfirmationStatus.MATCHED
            confirmation_b.status = ConfirmationStatus.MATCHED
            confirmation_a.match_score = 100.0
            confirmation_b.match_score = 100.0
            confirmation_a.matched_time = datetime.utcnow()
            confirmation_b.matched_time = confirmation_a.matched_time
        else:
            confirmation_a.status = ConfirmationStatus.MISMATCHED
            confirmation_b.status = ConfirmationStatus.MISMATCHED
            confirmation_a.match_score = confirmation_b.match_score = 0.0

    def affirm_confirmations(
        self, confirmations: Iterable[Confirmation], *, method: AffirmationMethod
    ) -> None:
        for confirmation in confirmations:
            if confirmation.status != ConfirmationStatus.MATCHED:
                raise RFQWorkflowError("Confirmation must be matched before affirmation")
            confirmation.status = ConfirmationStatus.AFFIRMED
            confirmation.affirmation_method = method
            confirmation.affirmed_time = datetime.utcnow()

    def generate_settlement_instruction(
        self,
        trade: Trade,
        confirmation: Confirmation,
        *,
        settlement_date: datetime,
        settlement_type: SettlementType = SettlementType.DVP,
        settlement_method: SettlementMethod = SettlementMethod.CCP,
    ) -> SettlementInstruction:
        cash_flow = CashFlow(
            amount=trade.economics.notional,
            currency=trade.economics.currency,
            direction="pay" if confirmation.direction == "buy" else "receive",
            payment_date=settlement_date.date(),
            value_date=settlement_date.date(),
            payer_account=f"{trade.buyer.party_id}-CASH",
            receiver_account=f"{trade.seller.party_id}-CASH",
        )

        instruction = SettlementInstruction(
            trade_id=trade.trade_id,
            confirmation_id=confirmation.confirmation_id,
            settlement_type=settlement_type,
            settlement_method=settlement_method,
            settlement_date=settlement_date.date(),
            deliverer=trade.seller,
            receiver=trade.buyer,
            cash_flows=[cash_flow],
            trade_date=trade.trade_date.date(),
            intended_settlement_date=settlement_date.date(),
        )

        instruction.status = SettlementStatus.INSTRUCTED
        return instruction

    def mark_instruction_settled(self, instruction: SettlementInstruction) -> None:
        instruction.status = SettlementStatus.SETTLED
        instruction.metadata.setdefault("settled_at", datetime.utcnow().isoformat())


__all__ = [
    "PostTradeProcessingService",
    "RFQWorkflowError",
    "RFQWorkflowService",
    "RFQWorkflowState",
    "WorkflowSnapshot",
]

