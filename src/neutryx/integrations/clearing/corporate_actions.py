"""Corporate action processing for securities clearing.

This module handles corporate actions that affect cleared positions:
1. Dividend processing (cash/stock)
2. Stock splits and reverse splits
3. Mergers and acquisitions
4. Rights issues and warrants
5. Spin-offs and demergers
6. Mandatory/voluntary tender offers
7. Redemptions and calls

Implements:
- ISO 20022 CAEV (Corporate Action Event) messages
- DTCC corporate action processing
- Automatic position adjustments
- Entitlement calculations
- Election processing for voluntary events
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .base import Party


class CorporateActionType(str, Enum):
    """Corporate action types."""
    # Distributions
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    RETURN_OF_CAPITAL = "return_of_capital"

    # Capital changes
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    BONUS_ISSUE = "bonus_issue"
    RIGHTS_ISSUE = "rights_issue"

    # Reorganizations
    MERGER = "merger"
    ACQUISITION = "acquisition"
    SPIN_OFF = "spin_off"
    DEMERGER = "demerger"

    # Tender offers
    TENDER_OFFER = "tender_offer"
    EXCHANGE_OFFER = "exchange_offer"

    # Redemptions
    CALL = "call"
    PUT = "put"
    REDEMPTION = "redemption"
    MATURITY = "maturity"

    # Other
    NAME_CHANGE = "name_change"
    DELISTING = "delisting"
    BANKRUPTCY = "bankruptcy"


class CorporateActionStatus(str, Enum):
    """Corporate action status."""
    ANNOUNCED = "announced"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    WITHDRAWN = "withdrawn"
    LAPSED = "lapsed"


class ElectionType(str, Enum):
    """Election type for voluntary events."""
    MANDATORY = "mandatory"
    VOLUNTARY = "voluntary"
    MANDATORY_WITH_OPTIONS = "mandatory_with_options"


class PaymentType(str, Enum):
    """Payment type."""
    CASH = "cash"
    STOCK = "stock"
    CASH_AND_STOCK = "cash_and_stock"
    RIGHTS = "rights"
    WARRANTS = "warrants"


class TaxTreatment(str, Enum):
    """Tax treatment."""
    QUALIFIED_DIVIDEND = "qualified"
    ORDINARY_DIVIDEND = "ordinary"
    RETURN_OF_CAPITAL = "return_of_capital"
    CAPITAL_GAIN = "capital_gain"
    TAX_FREE = "tax_free"


class CorporateActionEvent(BaseModel):
    """Corporate action event."""

    event_id: str = Field(
        default_factory=lambda: f"CA-{uuid4().hex[:12].upper()}"
    )
    event_type: CorporateActionType = Field(..., description="Type of corporate action")
    status: CorporateActionStatus = Field(default=CorporateActionStatus.ANNOUNCED)

    # Security information
    security_id: str = Field(..., description="Affected security (ISIN/CUSIP)")
    security_name: str = Field(..., description="Security name")
    issuer: str = Field(..., description="Issuer name")

    # Key dates
    announcement_date: date = Field(..., description="Announcement date")
    ex_date: date = Field(..., description="Ex-dividend/ex-date")
    record_date: date = Field(..., description="Record date")
    payment_date: date = Field(..., description="Payment/execution date")

    # Election dates (for voluntary events)
    election_deadline: Optional[date] = Field(None, description="Election deadline")
    withdrawal_deadline: Optional[date] = Field(None, description="Withdrawal deadline")

    # Event details
    election_type: ElectionType = Field(..., description="Mandatory or voluntary")
    default_option: Optional[str] = Field(None, description="Default option if no election")

    # Terms
    terms: Dict[str, Any] = Field(..., description="Event-specific terms")

    # Related securities (for reorganizations)
    new_security_id: Optional[str] = Field(None, description="New security ISIN")
    new_security_name: Optional[str] = Field(None, description="New security name")

    # Descriptions
    description: str = Field(..., description="Event description")
    terms_and_conditions: Optional[str] = Field(None, description="Full T&Cs")

    # Processing
    automatic_processing: bool = Field(default=True, description="Auto-process if possible")
    requires_election: bool = Field(default=False, description="Requires holder election")

    # External references
    cusip_event_id: Optional[str] = Field(None, description="CUSIP event ID")
    dtcc_event_id: Optional[str] = Field(None, description="DTCC event ID")
    isin: Optional[str] = Field(None, description="ISIN")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('ex_date', 'record_date', 'payment_date')
    @classmethod
    def validate_dates(cls, v, info):
        if 'announcement_date' in info.data:
            if v < info.data['announcement_date']:
                raise ValueError(f"{info.field_name} cannot be before announcement_date")
        return v


class DividendTerms(BaseModel):
    """Dividend payment terms."""

    dividend_rate: Decimal = Field(..., description="Dividend per share")
    currency: str = Field(..., description="Currency")
    payment_type: PaymentType = Field(default=PaymentType.CASH)

    # Stock dividend terms
    stock_dividend_ratio: Optional[str] = Field(None, description="e.g., '1:10'")
    new_shares_per_old: Optional[Decimal] = Field(None, description="New shares per old")

    # Mixed payment terms
    cash_dividend_rate: Optional[Decimal] = Field(
        None,
        description="Cash portion per share for mixed payments",
    )

    # Rights issue terms
    rights_ratio: Optional[Decimal] = Field(
        None, description="Rights per existing share"
    )
    rights_subscription_price: Optional[Decimal] = Field(
        None, description="Subscription price for rights"
    )
    rights_max_allocation: Optional[Decimal] = Field(
        None, description="Maximum rights that can be allocated"
    )

    # Warrant distribution terms
    warrant_ratio: Optional[Decimal] = Field(
        None, description="Warrants per existing share"
    )
    warrant_exercise_price: Optional[Decimal] = Field(
        None, description="Exercise price for warrants"
    )
    warrant_max_allocation: Optional[Decimal] = Field(
        None, description="Maximum warrants that can be allocated"
    )

    # Tax information
    tax_treatment: TaxTreatment = Field(default=TaxTreatment.QUALIFIED_DIVIDEND)
    withholding_rate: Optional[Decimal] = Field(None, description="Withholding tax rate")
    country_of_taxation: Optional[str] = Field(None, description="Tax jurisdiction")

    # Reinvestment
    drip_available: bool = Field(default=False, description="DRIP available")
    drip_price: Optional[Decimal] = Field(None, description="DRIP reinvestment price")
    drip_discount: Optional[Decimal] = Field(None, description="DRIP discount %")


class SplitTerms(BaseModel):
    """Stock split terms."""

    split_ratio: str = Field(..., description="Split ratio (e.g., '2:1', '1:3')")
    old_shares: int = Field(..., description="Old shares in ratio")
    new_shares: int = Field(..., description="New shares in ratio")

    # Cash in lieu for fractional shares
    fractional_share_treatment: str = Field(
        default="cash_in_lieu",
        description="How to handle fractions"
    )
    cash_in_lieu_price: Optional[Decimal] = Field(None, description="Price for fractional shares")


class MergerTerms(BaseModel):
    """Merger/acquisition terms."""

    acquirer_security_id: str = Field(..., description="Acquirer security ID")
    acquirer_name: str = Field(..., description="Acquirer name")

    exchange_ratio: Optional[str] = Field(None, description="Exchange ratio")
    cash_consideration: Optional[Decimal] = Field(None, description="Cash per share")
    stock_consideration: Optional[Decimal] = Field(None, description="Shares per share")

    # Mixed consideration
    election_available: bool = Field(default=False, description="Can elect cash or stock")
    proration_possible: bool = Field(default=False, description="Proration may occur")


class RightsTerms(BaseModel):
    """Rights issue terms."""

    subscription_ratio: str = Field(..., description="Rights ratio (e.g., '1:5')")
    subscription_price: Decimal = Field(..., description="Subscription price")
    currency: str = Field(..., description="Currency")

    # Trading
    rights_trading_start: Optional[date] = Field(None, description="Rights trading start")
    rights_trading_end: Optional[date] = Field(None, description="Rights trading end")
    rights_isin: Optional[str] = Field(None, description="Rights ISIN")

    # Oversubscription
    oversubscription_allowed: bool = Field(default=False)


def _parse_ratio(ratio: str) -> Decimal:
    """Parse ratio strings like '1:5' into Decimal values."""

    if ratio is None:
        raise ValueError("Ratio value is required")

    if isinstance(ratio, Decimal):
        return ratio

    ratio_str = str(ratio)
    if ":" in ratio_str:
        numerator, denominator = ratio_str.split(":", 1)
        return Decimal(numerator.strip()) / Decimal(denominator.strip())

    return Decimal(ratio_str)


class Position(BaseModel):
    """Security position for corporate action processing."""

    position_id: str = Field(
        default_factory=lambda: f"POS-{uuid4().hex[:12].upper()}"
    )

    # Position details
    security_id: str = Field(..., description="Security ID")
    holder: Party = Field(..., description="Position holder")
    quantity: Decimal = Field(..., description="Position quantity")
    account: str = Field(..., description="Account number")

    # Record date position
    as_of_date: date = Field(..., description="Position as of date")
    record_date_position: bool = Field(default=False, description="Is this record date position")

    # Cost basis (for tax purposes)
    cost_basis: Optional[Decimal] = Field(None, description="Total cost basis")
    acquisition_date: Optional[date] = Field(None, description="Acquisition date")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class Entitlement(BaseModel):
    """Corporate action entitlement calculation."""

    entitlement_id: str = Field(
        default_factory=lambda: f"ENT-{uuid4().hex[:12].upper()}"
    )
    event_id: str = Field(..., description="Corporate action event ID")
    position_id: str = Field(..., description="Position ID")

    # Holder
    holder: Party = Field(..., description="Entitlement holder")
    account: str = Field(..., description="Account")

    # Entitlement
    entitled_quantity: Decimal = Field(..., description="Entitled shares/units")

    # Cash entitlement
    cash_entitlement: Optional[Decimal] = Field(None, description="Cash amount entitled")
    currency: Optional[str] = Field(None, description="Currency")

    # Stock entitlement
    stock_entitlement: Optional[Decimal] = Field(None, description="Stock quantity entitled")
    new_security_id: Optional[str] = Field(None, description="New security ID")

    # Fractional shares
    fractional_shares: Decimal = Field(default=Decimal("0"), description="Fractional shares")
    cash_in_lieu: Optional[Decimal] = Field(None, description="Cash for fractional shares")

    # Tax
    gross_amount: Optional[Decimal] = Field(None, description="Gross amount")
    tax_withheld: Optional[Decimal] = Field(None, description="Tax withheld")
    net_amount: Optional[Decimal] = Field(None, description="Net amount")

    # Status
    calculated_time: datetime = Field(default_factory=datetime.utcnow)
    paid: bool = Field(default=False)
    payment_time: Optional[datetime] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class Election(BaseModel):
    """Holder election for voluntary corporate action."""

    election_id: str = Field(
        default_factory=lambda: f"ELE-{uuid4().hex[:12].upper()}"
    )
    event_id: str = Field(..., description="Corporate action event ID")
    position_id: str = Field(..., description="Position ID")

    # Holder
    holder: Party = Field(..., description="Election holder")
    account: str = Field(..., description="Account")

    # Election
    elected_option: str = Field(..., description="Elected option")
    elected_quantity: Decimal = Field(..., description="Quantity for election")

    # Timing
    election_time: datetime = Field(default_factory=datetime.utcnow)
    is_default: bool = Field(default=False, description="Is default election")

    # Processing
    processed: bool = Field(default=False)
    processed_time: Optional[datetime] = Field(None)

    # Withdrawal
    withdrawn: bool = Field(default=False)
    withdrawal_time: Optional[datetime] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class CorporateActionStatistics:
    """Corporate action processing statistics."""
    total_events: int = 0
    pending_events: int = 0
    completed_events: int = 0
    total_entitlements: int = 0
    total_entitlement_value: Decimal = Decimal("0")
    elections_received: int = 0
    default_elections: int = 0


class CorporateActionProcessor:
    """Process corporate actions and calculate entitlements."""

    def __init__(self):
        self.events: Dict[str, CorporateActionEvent] = {}
        self.positions: Dict[str, Position] = {}
        self.entitlements: Dict[str, Entitlement] = {}
        self.elections: Dict[str, Election] = {}
        self.statistics = CorporateActionStatistics()

    def add_event(self, event: CorporateActionEvent) -> CorporateActionEvent:
        """Add corporate action event."""
        self.events[event.event_id] = event
        self.statistics.total_events += 1
        self.statistics.pending_events += 1

        return event

    def add_position(self, position: Position) -> Position:
        """Add position for processing."""
        self.positions[position.position_id] = position
        return position

    def _store_entitlement(self, entitlement: Entitlement) -> Entitlement:
        """Persist entitlement and update statistics."""

        self.entitlements[entitlement.entitlement_id] = entitlement
        self.statistics.total_entitlements += 1

        if entitlement.net_amount:
            self.statistics.total_entitlement_value += entitlement.net_amount

        return entitlement

    def calculate_dividend_entitlement(
        self,
        event_id: str,
        position: Position,
        terms: DividendTerms
    ) -> Entitlement:
        """Calculate dividend entitlement."""
        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        # Cash dividend calculation
        if terms.payment_type == PaymentType.CASH:
            gross_amount = position.quantity * terms.dividend_rate

            # Calculate tax
            tax_withheld = Decimal("0")
            if terms.withholding_rate:
                tax_withheld = gross_amount * terms.withholding_rate

            net_amount = gross_amount - tax_withheld

            entitlement = Entitlement(
                event_id=event_id,
                position_id=position.position_id,
                holder=position.holder,
                account=position.account,
                entitled_quantity=position.quantity,
                cash_entitlement=net_amount,
                currency=terms.currency,
                gross_amount=gross_amount,
                tax_withheld=tax_withheld,
                net_amount=net_amount
            )

        # Stock dividend calculation
        elif terms.payment_type == PaymentType.STOCK:
            if not terms.new_shares_per_old:
                raise ValueError("Stock dividend requires new_shares_per_old")

            total_new_shares = position.quantity * terms.new_shares_per_old
            whole_shares = int(total_new_shares)
            fractional = total_new_shares - Decimal(whole_shares)

            # Cash in lieu for fractional shares
            cash_in_lieu = Decimal("0")
            if fractional > 0 and terms.drip_price:
                cash_in_lieu = fractional * terms.drip_price

            entitlement = Entitlement(
                event_id=event_id,
                position_id=position.position_id,
                holder=position.holder,
                account=position.account,
                entitled_quantity=position.quantity,
                stock_entitlement=Decimal(whole_shares),
                new_security_id=event.new_security_id,
                fractional_shares=fractional,
                cash_in_lieu=cash_in_lieu,
                currency=terms.currency
            )

        elif terms.payment_type == PaymentType.CASH_AND_STOCK:
            cash_rate = terms.cash_dividend_rate or terms.dividend_rate
            if cash_rate is None:
                raise ValueError(
                    "Cash and stock dividend requires cash_dividend_rate or dividend_rate"
                )
            if not terms.new_shares_per_old:
                raise ValueError(
                    "Cash and stock dividend requires new_shares_per_old for stock component"
                )

            gross_cash = position.quantity * cash_rate
            tax_withheld = Decimal("0")
            if terms.withholding_rate:
                tax_withheld = gross_cash * terms.withholding_rate

            net_cash = gross_cash - tax_withheld

            total_new_shares = position.quantity * terms.new_shares_per_old
            whole_shares = int(total_new_shares)
            fractional = total_new_shares - Decimal(whole_shares)

            cash_in_lieu = Decimal("0")
            if fractional > 0 and terms.drip_price:
                cash_in_lieu = fractional * terms.drip_price

            entitlement = Entitlement(
                event_id=event_id,
                position_id=position.position_id,
                holder=position.holder,
                account=position.account,
                entitled_quantity=position.quantity,
                cash_entitlement=net_cash,
                stock_entitlement=Decimal(whole_shares),
                new_security_id=event.new_security_id,
                fractional_shares=fractional,
                cash_in_lieu=cash_in_lieu,
                currency=terms.currency,
                gross_amount=gross_cash,
                tax_withheld=tax_withheld,
                net_amount=net_cash,
                metadata={
                    "cash_component": {
                        "rate_per_share": cash_rate,
                        "gross_amount": gross_cash,
                        "tax_withheld": tax_withheld,
                    },
                    "stock_component": {
                        "ratio": terms.new_shares_per_old,
                        "total_new_shares": total_new_shares,
                        "cash_in_lieu_rate": terms.drip_price,
                    },
                },
            )

        elif terms.payment_type == PaymentType.RIGHTS:
            if terms.rights_ratio is None:
                raise ValueError("Rights distribution requires rights_ratio")
            if terms.rights_subscription_price is None:
                raise ValueError("Rights distribution requires rights_subscription_price")

            total_rights = position.quantity * terms.rights_ratio
            if terms.rights_max_allocation is not None:
                if terms.rights_max_allocation <= 0:
                    raise ValueError("rights_max_allocation must be positive")
                total_rights = min(total_rights, terms.rights_max_allocation)

            subscription_value = total_rights * terms.rights_subscription_price

            entitlement = Entitlement(
                event_id=event_id,
                position_id=position.position_id,
                holder=position.holder,
                account=position.account,
                entitled_quantity=position.quantity,
                stock_entitlement=total_rights,
                new_security_id=event.new_security_id,
                currency=terms.currency,
                net_amount=Decimal("0"),
                metadata={
                    "rights": {
                        "ratio": terms.rights_ratio,
                        "subscription_price": terms.rights_subscription_price,
                        "subscription_value": subscription_value,
                        "max_allocation": terms.rights_max_allocation,
                    }
                },
            )

        elif terms.payment_type == PaymentType.WARRANTS:
            if terms.warrant_ratio is None:
                raise ValueError("Warrant distribution requires warrant_ratio")
            if terms.warrant_exercise_price is None:
                raise ValueError("Warrant distribution requires warrant_exercise_price")

            total_warrants = position.quantity * terms.warrant_ratio
            if terms.warrant_max_allocation is not None:
                if terms.warrant_max_allocation <= 0:
                    raise ValueError("warrant_max_allocation must be positive")
                total_warrants = min(total_warrants, terms.warrant_max_allocation)

            exercise_value = total_warrants * terms.warrant_exercise_price

            entitlement = Entitlement(
                event_id=event_id,
                position_id=position.position_id,
                holder=position.holder,
                account=position.account,
                entitled_quantity=position.quantity,
                stock_entitlement=total_warrants,
                new_security_id=event.new_security_id,
                currency=terms.currency,
                net_amount=Decimal("0"),
                metadata={
                    "warrants": {
                        "ratio": terms.warrant_ratio,
                        "exercise_price": terms.warrant_exercise_price,
                        "exercise_value": exercise_value,
                        "max_allocation": terms.warrant_max_allocation,
                    }
                },
            )

        else:
            raise NotImplementedError(f"Payment type {terms.payment_type} not implemented")

        return self._store_entitlement(entitlement)

    def calculate_split_adjustment(
        self,
        event_id: str,
        position: Position,
        terms: SplitTerms
    ) -> Tuple[Decimal, Decimal]:
        """Calculate position adjustment for stock split.

        Returns:
            (new_quantity, cash_in_lieu_for_fractionals)
        """
        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        # Calculate new quantity
        ratio = Decimal(terms.new_shares) / Decimal(terms.old_shares)
        new_quantity_total = position.quantity * ratio

        # Separate whole and fractional shares
        new_quantity_whole = int(new_quantity_total)
        fractional = new_quantity_total - Decimal(new_quantity_whole)

        # Cash in lieu for fractional shares
        cash_in_lieu = Decimal("0")
        if fractional > 0 and terms.cash_in_lieu_price:
            cash_in_lieu = fractional * terms.cash_in_lieu_price

        # Update position
        position.quantity = Decimal(new_quantity_whole)
        position.metadata['split_adjustment'] = {
            'event_id': event_id,
            'old_quantity': float(position.quantity / ratio),
            'ratio': terms.split_ratio,
            'fractional_shares': float(fractional),
            'cash_in_lieu': float(cash_in_lieu)
        }

        return Decimal(new_quantity_whole), cash_in_lieu

    def calculate_rights_issue(
        self,
        event_id: str,
        position: Position,
        terms: RightsTerms,
    ) -> Entitlement:
        """Calculate entitlements for a rights issue event."""

        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        ratio = _parse_ratio(terms.subscription_ratio)
        total_rights = position.quantity * ratio
        subscription_value = total_rights * terms.subscription_price

        entitlement = Entitlement(
            event_id=event_id,
            position_id=position.position_id,
            holder=position.holder,
            account=position.account,
            entitled_quantity=position.quantity,
            stock_entitlement=total_rights,
            new_security_id=event.new_security_id,
            currency=terms.currency,
            net_amount=-subscription_value,
            metadata={
                "subscription_price": str(terms.subscription_price),
                "subscription_value": str(subscription_value),
                "oversubscription_allowed": terms.oversubscription_allowed,
            },
        )

        return self._store_entitlement(entitlement)

    def calculate_merger_entitlement(
        self,
        event_id: str,
        position: Position,
        terms: MergerTerms,
    ) -> Entitlement:
        """Calculate entitlements for merger or acquisition events."""

        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        stock_ratio = Decimal("0")
        if terms.exchange_ratio:
            stock_ratio = _parse_ratio(terms.exchange_ratio)
        elif terms.stock_consideration:
            stock_ratio = terms.stock_consideration

        new_shares = position.quantity * stock_ratio
        cash_amount = Decimal("0")
        if terms.cash_consideration:
            cash_amount = position.quantity * terms.cash_consideration

        new_security_id = terms.acquirer_security_id or event.new_security_id

        entitlement = Entitlement(
            event_id=event_id,
            position_id=position.position_id,
            holder=position.holder,
            account=position.account,
            entitled_quantity=position.quantity,
            stock_entitlement=new_shares if new_shares > 0 else None,
            new_security_id=new_security_id,
            cash_entitlement=cash_amount if cash_amount > 0 else None,
            currency=(event.metadata or {}).get("consideration_currency"),
            net_amount=cash_amount if cash_amount > 0 else None,
            metadata={
                "exchange_ratio": terms.exchange_ratio,
                "cash_consideration": str(terms.cash_consideration)
                if terms.cash_consideration is not None
                else None,
                "election_available": terms.election_available,
                "proration_possible": terms.proration_possible,
            },
        )

        if new_shares > 0:
            position.metadata.setdefault("merger_exchanges", []).append({
                "event_id": event_id,
                "old_quantity": float(position.quantity),
                "new_shares": float(new_shares),
                "new_security_id": new_security_id,
            })
            position.quantity = new_shares
            if new_security_id:
                position.security_id = new_security_id

        return self._store_entitlement(entitlement)

    def process_spin_off_entitlement(
        self,
        event_id: str,
        position: Position,
        spin_terms: Dict[str, Any],
    ) -> Entitlement:
        """Calculate entitlements for spin-off or demerger events."""

        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        ratio_str = spin_terms.get("spin_off_ratio") or spin_terms.get("distribution_ratio")
        if not ratio_str:
            raise ValueError("Spin-off terms must include spin_off_ratio")

        ratio = _parse_ratio(ratio_str)
        total_new_shares = position.quantity * ratio
        whole_shares = Decimal(int(total_new_shares))
        fractional = total_new_shares - whole_shares

        cash_in_lieu = Decimal("0")
        cash_price = spin_terms.get("cash_in_lieu_price")
        if cash_price is not None:
            cash_in_lieu = fractional * Decimal(str(cash_price))

        entitlement = Entitlement(
            event_id=event_id,
            position_id=position.position_id,
            holder=position.holder,
            account=position.account,
            entitled_quantity=position.quantity,
            stock_entitlement=whole_shares,
            new_security_id=spin_terms.get("new_security_id") or event.new_security_id,
            fractional_shares=fractional,
            cash_in_lieu=cash_in_lieu if cash_in_lieu > 0 else None,
            currency=spin_terms.get("currency"),
            metadata={
                "spin_off_ratio": ratio_str,
                "cash_in_lieu_price": str(cash_price) if cash_price is not None else None,
            },
        )

        return self._store_entitlement(entitlement)

    def process_redemption_event(
        self,
        event_id: str,
        position: Position,
        redemption_price: Decimal,
        currency: Optional[str],
    ) -> Entitlement:
        """Process redemption, call or maturity events."""

        cash_amount = position.quantity * redemption_price

        entitlement = Entitlement(
            event_id=event_id,
            position_id=position.position_id,
            holder=position.holder,
            account=position.account,
            entitled_quantity=position.quantity,
            cash_entitlement=cash_amount,
            currency=currency,
            net_amount=cash_amount,
        )

        position.metadata.setdefault("redemptions", []).append({
            "event_id": event_id,
            "redemption_price": float(redemption_price),
            "cash_amount": float(cash_amount),
        })
        position.quantity = Decimal("0")

        return self._store_entitlement(entitlement)

    def _event_elections(self, event_id: str) -> List[Election]:
        """Retrieve non-withdrawn elections for an event."""

        return [
            election
            for election in self.elections.values()
            if election.event_id == event_id and not election.withdrawn
        ]

    def process_tender_offer(
        self,
        event_id: str,
        terms: Dict[str, Any],
    ) -> List[Entitlement]:
        """Create entitlements from tender or exchange offers."""

        elections = self._event_elections(event_id)
        if not elections:
            raise ValueError("Tender offer requires elections")

        offer_price = Decimal(str(terms.get("offer_price", "0")))
        currency = terms.get("currency")

        entitlements: List[Entitlement] = []

        for election in elections:
            position = self.positions.get(election.position_id)
            if not position:
                continue

            quantity = election.elected_quantity
            cash_amount = offer_price * quantity if offer_price else None

            entitlement = Entitlement(
                event_id=event_id,
                position_id=election.position_id,
                holder=position.holder,
                account=position.account,
                entitled_quantity=quantity,
                cash_entitlement=cash_amount,
                currency=currency,
                net_amount=cash_amount,
                metadata={
                    "elected_option": election.elected_option,
                    "is_default": election.is_default,
                },
            )

            entitlements.append(self._store_entitlement(entitlement))
            position.quantity -= quantity
            position.metadata.setdefault("tender_offers", []).append({
                "event_id": event_id,
                "quantity_tendered": float(quantity),
                "cash_amount": float(cash_amount) if cash_amount else None,
            })

        return entitlements

    def submit_election(
        self,
        event_id: str,
        position_id: str,
        elected_option: str,
        elected_quantity: Optional[Decimal] = None
    ) -> Election:
        """Submit holder election for voluntary event."""
        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        if event.election_type == ElectionType.MANDATORY:
            raise ValueError("Event is mandatory, no election required")

        position = self.positions.get(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")

        # Check deadline
        if event.election_deadline and date.today() > event.election_deadline:
            raise ValueError("Election deadline has passed")

        # Use full position if quantity not specified
        if elected_quantity is None:
            elected_quantity = position.quantity

        if elected_quantity > position.quantity:
            raise ValueError("Elected quantity exceeds position")

        election = Election(
            event_id=event_id,
            position_id=position_id,
            holder=position.holder,
            account=position.account,
            elected_option=elected_option,
            elected_quantity=elected_quantity
        )

        self.elections[election.election_id] = election
        self.statistics.elections_received += 1

        return election

    def apply_default_election(
        self,
        event_id: str,
        position_id: str
    ) -> Election:
        """Apply default election for positions without election."""
        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        if not event.default_option:
            raise ValueError("No default option specified for event")

        position = self.positions.get(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")

        election = Election(
            event_id=event_id,
            position_id=position_id,
            holder=position.holder,
            account=position.account,
            elected_option=event.default_option,
            elected_quantity=position.quantity,
            is_default=True
        )

        self.elections[election.election_id] = election
        self.statistics.default_elections += 1

        return election

    def process_event(self, event_id: str) -> List[Entitlement]:
        """Process corporate action event for all eligible positions."""
        event = self.events.get(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        if event.status != CorporateActionStatus.PENDING:
            raise ValueError(f"Event {event_id} not in PENDING status")

        # Find eligible positions (as of record date)
        eligible_positions = [
            pos for pos in self.positions.values()
            if pos.security_id == event.security_id
            and pos.record_date_position
            and pos.as_of_date == event.record_date
        ]

        entitlements = []

        # Process based on event type
        if event.event_type in [
            CorporateActionType.CASH_DIVIDEND,
            CorporateActionType.STOCK_DIVIDEND,
            CorporateActionType.SPECIAL_DIVIDEND
        ]:
            terms = DividendTerms(**event.terms)
            for position in eligible_positions:
                ent = self.calculate_dividend_entitlement(event_id, position, terms)
                entitlements.append(ent)

        elif event.event_type in [
            CorporateActionType.STOCK_SPLIT,
            CorporateActionType.REVERSE_SPLIT
        ]:
            terms = SplitTerms(**event.terms)
            for position in eligible_positions:
                new_qty, cash = self.calculate_split_adjustment(event_id, position, terms)
                # Could create entitlement for cash_in_lieu if needed

        elif event.event_type == CorporateActionType.RIGHTS_ISSUE:
            terms = RightsTerms(**event.terms)
            for position in eligible_positions:
                entitlements.append(self.calculate_rights_issue(event_id, position, terms))

        elif event.event_type in [
            CorporateActionType.MERGER,
            CorporateActionType.ACQUISITION,
        ]:
            terms = MergerTerms(**event.terms)
            for position in eligible_positions:
                entitlements.append(
                    self.calculate_merger_entitlement(event_id, position, terms)
                )

        elif event.event_type in [
            CorporateActionType.SPIN_OFF,
            CorporateActionType.DEMERGER,
        ]:
            for position in eligible_positions:
                entitlements.append(
                    self.process_spin_off_entitlement(event_id, position, event.terms)
                )

        elif event.event_type in [
            CorporateActionType.TENDER_OFFER,
            CorporateActionType.EXCHANGE_OFFER,
        ]:
            entitlements.extend(self.process_tender_offer(event_id, event.terms))

        elif event.event_type in [
            CorporateActionType.CALL,
            CorporateActionType.REDEMPTION,
            CorporateActionType.MATURITY,
        ]:
            redemption_price = event.terms.get("redemption_price")
            if redemption_price is None:
                raise ValueError("Redemption events require redemption_price term")

            redemption_price = Decimal(str(redemption_price))
            currency = event.terms.get("currency")

            for position in eligible_positions:
                entitlements.append(
                    self.process_redemption_event(
                        event_id, position, redemption_price, currency
                    )
                )

        event.status = CorporateActionStatus.COMPLETED
        self.statistics.completed_events += 1
        self.statistics.pending_events -= 1

        return entitlements

    def get_position_entitlements(
        self,
        position_id: str
    ) -> List[Entitlement]:
        """Get all entitlements for a position."""
        return [
            ent for ent in self.entitlements.values()
            if ent.position_id == position_id
        ]

    def get_pending_elections(self) -> List[Tuple[CorporateActionEvent, List[Position]]]:
        """Get events requiring elections and positions without elections."""
        pending = []

        for event in self.events.values():
            if event.requires_election and event.status == CorporateActionStatus.PENDING:
                # Find positions without elections
                eligible_positions = [
                    pos for pos in self.positions.values()
                    if pos.security_id == event.security_id
                    and pos.record_date_position
                ]

                positions_with_elections = {
                    elec.position_id for elec in self.elections.values()
                    if elec.event_id == event.event_id and not elec.withdrawn
                }

                positions_without_election = [
                    pos for pos in eligible_positions
                    if pos.position_id not in positions_with_elections
                ]

                if positions_without_election:
                    pending.append((event, positions_without_election))

        return pending

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_events": self.statistics.total_events,
            "pending_events": self.statistics.pending_events,
            "completed_events": self.statistics.completed_events,
            "total_entitlements": self.statistics.total_entitlements,
            "total_entitlement_value": float(self.statistics.total_entitlement_value),
            "elections_received": self.statistics.elections_received,
            "default_elections": self.statistics.default_elections,
        }


@dataclass
class ScheduledCorporateAction:
    """Represents a corporate action scheduled for processing."""

    event: CorporateActionEvent
    execution_date: date
    status: CorporateActionStatus = CorporateActionStatus.ANNOUNCED
    attempts: int = 0
    last_error: Optional[str] = None

    def mark_in_progress(self) -> None:
        if self.status == CorporateActionStatus.COMPLETED:
            return
        self.status = CorporateActionStatus.IN_PROGRESS
        self.attempts += 1

    def mark_completed(self) -> None:
        self.status = CorporateActionStatus.COMPLETED
        self.last_error = None

    def mark_failed(self, error: str) -> None:
        self.status = CorporateActionStatus.PENDING
        self.last_error = error


class CorporateActionScheduler:
    """Schedule and orchestrate corporate action processing."""

    def __init__(self) -> None:
        self._schedule: Dict[date, List[ScheduledCorporateAction]] = defaultdict(list)

    def schedule(self, event: CorporateActionEvent) -> ScheduledCorporateAction:
        """Schedule an event based on record/payment date."""

        execution_date = event.payment_date or event.record_date
        scheduled = ScheduledCorporateAction(event=event, execution_date=execution_date)
        self._schedule[execution_date].append(scheduled)
        return scheduled

    def consume_due_events(self, as_of: date) -> List[ScheduledCorporateAction]:
        """Return due events up to ``as_of`` marking them in progress."""

        due: List[ScheduledCorporateAction] = []
        for execution_date in sorted(self._schedule):
            if execution_date > as_of:
                continue
            for scheduled in self._schedule[execution_date]:
                if scheduled.status in {
                    CorporateActionStatus.ANNOUNCED,
                    CorporateActionStatus.PENDING,
                }:
                    scheduled.mark_in_progress()
                    due.append(scheduled)
        return due

    def mark_completed(self, event_id: str) -> None:
        """Mark an event as completed."""

        for scheduled in self._iterate_all():
            if scheduled.event.event_id == event_id:
                scheduled.mark_completed()
                break

    def mark_failed(self, event_id: str, error: str) -> None:
        """Record a processing failure for retry."""

        for scheduled in self._iterate_all():
            if scheduled.event.event_id == event_id:
                scheduled.mark_failed(error)
                break

    def pending(self) -> List[ScheduledCorporateAction]:
        """Return all events not yet completed."""

        return [
            scheduled
            for scheduled in self._iterate_all()
            if scheduled.status != CorporateActionStatus.COMPLETED
        ]

    def _iterate_all(self) -> Iterable[ScheduledCorporateAction]:
        for entries in self._schedule.values():
            for scheduled in entries:
                yield scheduled


__all__ = [
    "CorporateActionType",
    "CorporateActionStatus",
    "CorporateActionEvent",
    "ElectionType",
    "PaymentType",
    "TaxTreatment",
    "DividendTerms",
    "SplitTerms",
    "MergerTerms",
    "RightsTerms",
    "Position",
    "Entitlement",
    "Election",
    "CorporateActionProcessor",
    "CorporateActionStatistics",
    "ScheduledCorporateAction",
    "CorporateActionScheduler",
]
