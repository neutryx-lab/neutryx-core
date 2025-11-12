"""Portfolio update pipeline integrating scheduled corporate actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from neutryx.data.security_master import SecurityMaster
from neutryx.integrations.clearing.corporate_actions import (
    CorporateActionEvent,
    CorporateActionScheduler,
    CorporateActionType,
    DividendTerms,
    MergerTerms,
    ScheduledCorporateAction,
    SplitTerms,
)
from neutryx.portfolio.portfolio import Portfolio
from neutryx.portfolio.positions import PortfolioPosition


@dataclass
class PortfolioCorporateActionResult:
    """Summary of how a corporate action impacted the portfolio."""

    event_id: str
    applied: bool
    details: Dict[str, Any] = field(default_factory=dict)


class PortfolioUpdatePipeline:
    """Apply scheduled corporate actions to portfolio positions."""

    def __init__(
        self,
        portfolio: Portfolio,
        *,
        scheduler: CorporateActionScheduler,
        security_master: SecurityMaster,
    ) -> None:
        self._portfolio = portfolio
        self._scheduler = scheduler
        self._security_master = security_master

    def process_corporate_actions(self, as_of: date) -> List[PortfolioCorporateActionResult]:
        """Consume due corporate actions from the scheduler."""

        results: List[PortfolioCorporateActionResult] = []
        due_events = self._scheduler.consume_due_events(as_of)
        for scheduled in due_events:
            try:
                result = self._apply_event(scheduled)
            except Exception as exc:  # pragma: no cover - defensive guard
                scheduled.mark_failed(str(exc))
                results.append(
                    PortfolioCorporateActionResult(
                        event_id=scheduled.event.event_id,
                        applied=False,
                        details={"error": str(exc)},
                    )
                )
            else:
                scheduled.mark_completed()
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _apply_event(self, scheduled: ScheduledCorporateAction) -> PortfolioCorporateActionResult:
        event = scheduled.event
        if event.event_type == CorporateActionType.STOCK_SPLIT:
            return self._apply_stock_split(event)
        if event.event_type == CorporateActionType.CASH_DIVIDEND:
            return self._apply_dividend(event)
        if event.event_type in {
            CorporateActionType.MERGER,
            CorporateActionType.ACQUISITION,
            CorporateActionType.SPIN_OFF,
            CorporateActionType.DEMERGER,
        }:
            return self._apply_reorganisation(event)

        return PortfolioCorporateActionResult(
            event_id=event.event_id,
            applied=False,
            details={"reason": f"Unhandled corporate action type {event.event_type.value}"},
        )

    def _apply_stock_split(self, event: CorporateActionEvent) -> PortfolioCorporateActionResult:
        record = self._security_master.ensure_active(
            event.security_id, as_of=event.record_date
        )
        position = self._portfolio.get_position(event.security_id)
        if position is None:
            return PortfolioCorporateActionResult(
                event_id=event.event_id,
                applied=False,
                details={"reason": "no_position"},
            )

        terms = event.terms
        if not isinstance(terms, SplitTerms):
            terms = SplitTerms(**terms)

        ratio = Decimal(terms.new_shares) / Decimal(terms.old_shares)
        old_quantity = position.quantity
        new_total = old_quantity * ratio
        whole_quantity = Decimal(int(new_total))
        fractional = new_total - whole_quantity

        position.quantity = whole_quantity
        position.metadata.setdefault("corporate_actions", []).append(
            {
                "event_id": event.event_id,
                "type": event.event_type.value,
                "ratio": terms.split_ratio,
                "old_quantity": str(old_quantity),
                "new_quantity": str(whole_quantity),
                "fractional": str(fractional),
            }
        )

        cash_in_lieu = Decimal("0")
        if fractional > Decimal("0") and terms.cash_in_lieu_price:
            cash_in_lieu = fractional * Decimal(terms.cash_in_lieu_price)
            self._portfolio.record_cash_flow(record.metadata.get("currency", "USD"), cash_in_lieu)

        return PortfolioCorporateActionResult(
            event_id=event.event_id,
            applied=True,
            details={
                "security_id": record.security_id,
                "new_quantity": str(position.quantity),
                "cash_in_lieu": str(cash_in_lieu),
            },
        )

    def _apply_dividend(self, event: CorporateActionEvent) -> PortfolioCorporateActionResult:
        record = self._security_master.ensure_active(
            event.security_id, as_of=event.record_date
        )
        position = self._portfolio.get_position(event.security_id)
        if position is None:
            return PortfolioCorporateActionResult(
                event_id=event.event_id,
                applied=False,
                details={"reason": "no_position"},
            )

        terms = event.terms
        if not isinstance(terms, DividendTerms):
            terms = DividendTerms(**terms)

        rate = Decimal(terms.dividend_rate)
        cash_amount = position.quantity * rate
        currency = terms.currency
        self._portfolio.record_cash_flow(currency, cash_amount)
        position.metadata.setdefault("corporate_actions", []).append(
            {
                "event_id": event.event_id,
                "type": event.event_type.value,
                "dividend_rate": str(rate),
                "cash_amount": str(cash_amount),
                "currency": currency,
            }
        )

        return PortfolioCorporateActionResult(
            event_id=event.event_id,
            applied=True,
            details={
                "security_id": record.security_id,
                "cash_amount": str(cash_amount),
                "currency": currency,
            },
        )

    def _apply_reorganisation(self, event: CorporateActionEvent) -> PortfolioCorporateActionResult:
        record = self._security_master.ensure_active(
            event.security_id, as_of=event.record_date
        )
        position = self._portfolio.get_position(event.security_id)
        if position is None:
            return PortfolioCorporateActionResult(
                event_id=event.event_id,
                applied=False,
                details={"reason": "no_position"},
            )

        terms = event.terms
        if not isinstance(terms, MergerTerms):
            terms = MergerTerms(**terms)

        old_quantity = position.quantity
        new_ratio = self._parse_ratio(terms.exchange_ratio) if terms.exchange_ratio else None
        if terms.stock_consideration is not None:
            new_ratio = Decimal(terms.stock_consideration)

        new_quantity = Decimal("0")
        if new_ratio is not None:
            new_quantity = old_quantity * new_ratio

        cash_currency = event.metadata.get("cash_currency", "USD") if event.metadata else "USD"
        cash_amount = Decimal("0")
        if terms.cash_consideration is not None:
            cash_amount = old_quantity * Decimal(terms.cash_consideration)
            self._portfolio.record_cash_flow(cash_currency, cash_amount)

        # Remove old position quantity while preserving metadata trail
        position.metadata.setdefault("corporate_actions", []).append(
            {
                "event_id": event.event_id,
                "type": event.event_type.value,
                "old_quantity": str(old_quantity),
                "new_quantity": str(new_quantity),
                "cash_amount": str(cash_amount),
                "cash_currency": cash_currency,
            }
        )
        position.quantity = Decimal("0")

        new_position: Optional[PortfolioPosition] = None
        if event.new_security_id and new_quantity > Decimal("0"):
            self._security_master.ensure_active(
                event.new_security_id, as_of=event.payment_date or event.record_date
            )
            new_position = self._portfolio.adjust_position(
                event.new_security_id,
                new_quantity,
            )
            new_position.metadata.setdefault("corporate_actions", []).append(
                {
                    "event_id": event.event_id,
                    "source_security": record.security_id,
                    "quantity_received": str(new_quantity),
                }
            )

        details = {
            "security_id": record.security_id,
            "quantity_removed": str(old_quantity),
            "cash_amount": str(cash_amount),
            "cash_currency": cash_currency,
        }
        if new_position:
            details["new_security_id"] = new_position.security_id
            details["new_quantity"] = str(new_position.quantity)

        return PortfolioCorporateActionResult(
            event_id=event.event_id,
            applied=True,
            details=details,
        )

    @staticmethod
    def _parse_ratio(ratio: str) -> Decimal:
        if ":" in ratio:
            left, right = ratio.split(":", 1)
            return Decimal(left) / Decimal(right)
        return Decimal(ratio)


__all__ = [
    "PortfolioUpdatePipeline",
    "PortfolioCorporateActionResult",
]
