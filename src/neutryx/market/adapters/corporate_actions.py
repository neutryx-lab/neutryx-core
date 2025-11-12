"""Vendor specific corporate action normalization utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, Any, Iterable, List

from ..storage.security_master import CorporateActionEvent, CorporateActionType


class CorporateActionParser(ABC):
    """Abstract base parser converting raw vendor payloads into events."""

    @abstractmethod
    def parse(self, raw_event: Dict[str, Any]) -> CorporateActionEvent:
        """Normalize a raw event payload."""


@dataclass
class BloombergCorporateActionParser(CorporateActionParser):
    """Parser for Bloomberg style payloads."""

    def parse(self, raw_event: Dict[str, Any]) -> CorporateActionEvent:
        action_map = {
            "DVD": CorporateActionType.DIVIDEND,
            "SPLIT": CorporateActionType.SPLIT,
            "MERGER": CorporateActionType.MERGER,
            "SPIN": CorporateActionType.SPIN_OFF,
            "NAME/TICKER CHANGE": CorporateActionType.SYMBOL_CHANGE,
        }
        action_type = action_map.get(raw_event.get("event_type", "").upper())
        if not action_type:
            raise ValueError("Unsupported Bloomberg corporate action type")

        effective = raw_event.get("effective_date")
        if isinstance(effective, date):
            effective_date = effective
        else:
            effective_date = date.fromisoformat(str(effective))

        details = {
            "amount": raw_event.get("amount"),
            "currency": raw_event.get("currency"),
            "split_ratio": raw_event.get("split_ratio"),
            "old_ticker": raw_event.get("old_ticker"),
            "new_ticker": raw_event.get("new_ticker"),
            "pay_date": raw_event.get("pay_date"),
        }
        description = raw_event.get("description") or f"{action_type.value} event"
        return CorporateActionEvent(
            action_type=action_type,
            effective_date=effective_date,
            description=description,
            details={k: v for k, v in details.items() if v is not None},
        )


@dataclass
class RefinitivCorporateActionParser(CorporateActionParser):
    """Parser for Refinitiv style payloads."""

    def parse(self, raw_event: Dict[str, Any]) -> CorporateActionEvent:
        action_map = {
            "DVD_CASH": CorporateActionType.DIVIDEND,
            "STOCK_SPLIT": CorporateActionType.SPLIT,
            "MERGER": CorporateActionType.MERGER,
            "SPINOFF": CorporateActionType.SPIN_OFF,
            "TICKER_CHANGE": CorporateActionType.SYMBOL_CHANGE,
        }
        action_type = action_map.get(raw_event.get("type"))
        if not action_type:
            raise ValueError("Unsupported Refinitiv corporate action type")

        effective = raw_event.get("effectiveDate")
        if isinstance(effective, date):
            effective_date = effective
        else:
            effective_date = date.fromisoformat(str(effective))

        details = {
            "amount": raw_event.get("cashAmount"),
            "currency": raw_event.get("currency"),
            "split_ratio": raw_event.get("ratio"),
            "old_ticker": raw_event.get("fromSymbol"),
            "new_ticker": raw_event.get("toSymbol"),
            "pay_date": raw_event.get("payDate"),
        }
        description = raw_event.get("text") or f"{action_type.value} event"
        return CorporateActionEvent(
            action_type=action_type,
            effective_date=effective_date,
            description=description,
            details={k: v for k, v in details.items() if v is not None},
        )


def normalize_events(
    raw_events: Iterable[Dict[str, Any]], parser: CorporateActionParser
) -> List[CorporateActionEvent]:
    """Normalize a collection of raw events using the provided parser."""

    return [parser.parse(event) for event in raw_events]
