"""Tests for corporate action processing flows."""

from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from neutryx.integrations.clearing.base import Party
from neutryx.integrations.clearing.corporate_actions import (
    CorporateActionEvent,
    CorporateActionProcessor,
    CorporateActionStatus,
    CorporateActionType,
    DividendTerms,
    ElectionType,
    MergerTerms,
    PaymentType,
    Position,
    RightsTerms,
)


def _make_party(party_id: str) -> Party:
    return Party(party_id=party_id, name=f"Party {party_id}")


def _base_event(
    event_type: CorporateActionType,
    terms: dict,
    requires_election: bool = False,
    new_security_id: str | None = None,
) -> CorporateActionEvent:
    announcement = date(2024, 1, 2)
    record = date(2024, 1, 15)

    return CorporateActionEvent(
        event_type=event_type,
        status=CorporateActionStatus.PENDING,
        security_id="XYZ123",
        security_name="XYZ Corp",
        issuer="XYZ Corp",
        announcement_date=announcement,
        ex_date=record,
        record_date=record,
        payment_date=date(2024, 1, 20),
        election_type=ElectionType.VOLUNTARY if requires_election else ElectionType.MANDATORY,
        requires_election=requires_election,
        default_option="cash" if requires_election else None,
        terms=terms,
        new_security_id=new_security_id,
        description=f"{event_type.value} event",
    )


def _position(quantity: Decimal, record_date: date) -> Position:
    return Position(
        security_id="XYZ123",
        holder=_make_party("BUY1"),
        quantity=quantity,
        account="ACC-123",
        as_of_date=record_date,
        record_date_position=True,
    )


def test_cash_dividend_with_withholding():
    processor = CorporateActionProcessor()

    dividend_terms = DividendTerms(
        dividend_rate=Decimal("1.50"),
        currency="USD",
        payment_type=PaymentType.CASH,
        withholding_rate=Decimal("0.15"),
    )

    event = _base_event(
        CorporateActionType.CASH_DIVIDEND,
        dividend_terms.model_dump(exclude_none=True),
    )
    processor.add_event(event)

    position = _position(Decimal("200"), event.record_date)
    processor.add_position(position)

    entitlements = processor.process_event(event.event_id)

    assert len(entitlements) == 1
    entitlement = entitlements[0]

    expected_gross = Decimal("300.00")  # 200 * 1.50
    expected_tax = expected_gross * dividend_terms.withholding_rate
    expected_net = expected_gross - expected_tax

    assert entitlement.cash_entitlement == expected_net
    assert entitlement.gross_amount == expected_gross
    assert entitlement.tax_withheld == expected_tax
    assert entitlement.currency == "USD"


def test_rights_issue_entitlement_calculation():
    processor = CorporateActionProcessor()

    rights_terms = RightsTerms(
        subscription_ratio="1:4",
        subscription_price=Decimal("10"),
        currency="USD",
        oversubscription_allowed=True,
    )

    event = _base_event(
        CorporateActionType.RIGHTS_ISSUE,
        rights_terms.model_dump(exclude_none=True),
        new_security_id="RGT999",
    )
    processor.add_event(event)

    position = _position(Decimal("120"), event.record_date)
    processor.add_position(position)

    entitlements = processor.process_event(event.event_id)

    assert len(entitlements) == 1
    entitlement = entitlements[0]

    assert entitlement.stock_entitlement == Decimal("30")  # 120 * 0.25
    # Subscription outflow should be stored as negative net amount
    assert entitlement.net_amount == Decimal("-300")
    assert entitlement.metadata["oversubscription_allowed"] is True


def test_merger_entitlement_updates_position_quantity():
    processor = CorporateActionProcessor()

    merger_terms = MergerTerms(
        acquirer_security_id="NEW-001",
        acquirer_name="New Corp",
        exchange_ratio="3:2",
        cash_consideration=Decimal("5"),
        election_available=False,
        proration_possible=False,
    )

    event = _base_event(
        CorporateActionType.MERGER,
        merger_terms.model_dump(exclude_none=True),
        new_security_id="NEW-001",
    )
    processor.add_event(event)

    position = _position(Decimal("200"), event.record_date)
    processor.add_position(position)

    entitlements = processor.process_event(event.event_id)

    assert len(entitlements) == 1
    entitlement = entitlements[0]

    # 200 * (3/2) = 300 new shares
    assert entitlement.stock_entitlement == Decimal("300")
    assert entitlement.cash_entitlement == Decimal("1000")
    # Position should reflect new security and quantity
    assert position.security_id == "NEW-001"
    assert position.quantity == Decimal("300")


def test_tender_offer_requires_election_and_reduces_position():
    processor = CorporateActionProcessor()

    offer_terms = {"offer_price": "12.5", "currency": "USD"}

    event = _base_event(
        CorporateActionType.TENDER_OFFER,
        offer_terms,
        requires_election=True,
    )
    processor.add_event(event)

    position = _position(Decimal("80"), event.record_date)
    processor.add_position(position)

    # Submit election for part of the position
    processor.submit_election(
        event.event_id,
        position.position_id,
        elected_option="cash",
        elected_quantity=Decimal("50"),
    )

    entitlements = processor.process_event(event.event_id)

    assert len(entitlements) == 1
    entitlement = entitlements[0]

    assert entitlement.cash_entitlement == Decimal("625")  # 50 * 12.5
    assert position.quantity == Decimal("30")  # 80 - 50 tendered


def test_tender_offer_without_elections_raises():
    processor = CorporateActionProcessor()

    event = _base_event(
        CorporateActionType.TENDER_OFFER,
        {"offer_price": "10", "currency": "USD"},
        requires_election=True,
    )
    processor.add_event(event)

    position = _position(Decimal("40"), event.record_date)
    processor.add_position(position)

    with pytest.raises(ValueError):
        processor.process_event(event.event_id)
