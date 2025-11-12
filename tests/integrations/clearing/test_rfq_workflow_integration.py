"""Workflow-level integration test for the RFQ → settlement lifecycle."""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from neutryx.integrations.clearing import (
    AffirmationMethod,
    OrderSide,
    Party,
    PostTradeProcessingService,
    ProductType,
    RFQWorkflowService,
    RFQWorkflowState,
    SettlementStatus,
    SettlementType,
)
from neutryx.integrations.clearing.rfq import RFQSpecification, Quote, TimeInForce


def _make_party(party_id: str, name: str) -> Party:
    return Party(party_id=party_id, name=name, lei=f"LEI-{party_id}")


def test_rfq_workflow_end_to_end():
    workflow = RFQWorkflowService()
    post_trade = PostTradeProcessingService()

    requester = _make_party("REQ1", "Requesting Asset Manager")
    dealer_a = _make_party("DLR1", "Dealer One")
    dealer_b = _make_party("DLR2", "Dealer Two")

    spec = RFQSpecification(
        product_type=ProductType.IRS,
        notional=Decimal("1000000"),
        currency="USD",
        effective_date=datetime.utcnow() + timedelta(days=2),
        maturity_date=datetime.utcnow() + timedelta(days=365 * 5),
        fixed_rate=Decimal("0.025"),
    )

    quote_deadline = datetime.utcnow() + timedelta(minutes=1)
    rfq = workflow.create_rfq(
        requester=requester,
        specification=spec,
        side=OrderSide.BUY,
        quote_deadline=quote_deadline,
        submission_time=datetime.utcnow() - timedelta(minutes=1),
    )

    workflow.submit_rfq(rfq.rfq_id)

    quote_a = Quote(
        rfq_id=rfq.rfq_id,
        quoter=dealer_a,
        quoter_member_id="MEMBER-A",
        side=OrderSide.SELL,
        quantity=spec.notional,
        price=Decimal("99.85"),
        time_in_force=TimeInForce.GTC,
    )

    quote_b = Quote(
        rfq_id=rfq.rfq_id,
        quoter=dealer_b,
        quoter_member_id="MEMBER-B",
        side=OrderSide.SELL,
        quantity=spec.notional,
        price=Decimal("99.80"),
        time_in_force=TimeInForce.GTC,
    )

    workflow.submit_quote(rfq.rfq_id, quote_a)
    workflow.submit_quote(rfq.rfq_id, quote_b)

    auction_result = workflow.execute_auction(rfq.rfq_id, force=True)
    assert auction_result.total_quantity_filled == spec.notional
    assert auction_result.clearing_price == quote_b.price

    trade = workflow.book_trade(
        rfq.rfq_id,
        trade_id="TRD-001",
        buyer=requester,
        seller=dealer_b,
        execution_price=auction_result.clearing_price or Decimal("0"),
        trade_date=datetime.utcnow(),
        effective_date=spec.effective_date,
        maturity_date=spec.maturity_date,
        currency=spec.currency,
    )

    settlement_date = datetime.utcnow() + timedelta(days=2)
    confirmations = post_trade.generate_confirmations(
        trade, execution_price=auction_result.clearing_price or Decimal("0"), settlement_date=settlement_date
    )

    assert all(conf.status.name == "MATCHED" for conf in confirmations)
    workflow.attach_confirmations(rfq.rfq_id, confirmations)
    snapshot_after_confirmations = workflow.snapshot(rfq.rfq_id)
    assert snapshot_after_confirmations.state == RFQWorkflowState.CONFIRMATION_MATCHED

    post_trade.affirm_confirmations(confirmations, method=AffirmationMethod.ELECTRONIC)
    workflow.mark_affirmed(rfq.rfq_id)
    snapshot_after_affirm = workflow.snapshot(rfq.rfq_id)
    assert snapshot_after_affirm.state == RFQWorkflowState.AFFIRMATION_COMPLETE

    instruction = post_trade.generate_settlement_instruction(
        trade,
        confirmations[0],
        settlement_date=settlement_date,
        settlement_type=SettlementType.DVP,
    )

    workflow.attach_settlement_instruction(rfq.rfq_id, instruction)
    post_trade.mark_instruction_settled(instruction)
    assert instruction.status == SettlementStatus.SETTLED
    workflow.mark_settled(rfq.rfq_id)

    final_snapshot = workflow.snapshot(rfq.rfq_id)
    assert final_snapshot.state == RFQWorkflowState.SETTLED
    assert final_snapshot.trade == trade
    assert final_snapshot.auction_result == auction_result

