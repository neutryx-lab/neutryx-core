"""Corporate action normalization utilities for vendor adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

from ..storage.security_master import CorporateActionEvent, CorporateActionType


def _ensure_datetime(value: Any) -> datetime:
    """Convert dates, datetimes, or ISO strings to :class:`datetime`."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Unsupported date value: {value!r}")


def _ensure_optional_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    return _ensure_datetime(value)


class CorporateActionParser(ABC):
    """Base parser for vendor-specific corporate action payloads."""

    vendor: str

    def __init__(self, vendor: str) -> None:
        self.vendor = vendor

    @abstractmethod
    def normalize(self, payload: Dict[str, Any]) -> CorporateActionEvent:
        """Normalize a raw vendor payload into a :class:`CorporateActionEvent`."""

    def normalize_many(
        self, payloads: Iterable[Dict[str, Any]]
    ) -> List[CorporateActionEvent]:
        """Normalize multiple payloads."""

        return [self.normalize(payload) for payload in payloads]


class BloombergCorporateActionParser(CorporateActionParser):
    """Normalize Bloomberg corporate action payloads."""

    EVENT_MAP = {
        "DVD_CASH": CorporateActionType.DIVIDEND,
        "DVD_STOCK": CorporateActionType.DIVIDEND,
        "SPLIT": CorporateActionType.SPLIT,
        "CHG_NAME": CorporateActionType.NAME_CHANGE,
        "CHG_TICKER": CorporateActionType.IDENTIFIER_CHANGE,
        "MERGER": CorporateActionType.MERGER,
        "SPIN": CorporateActionType.SPIN_OFF,
        "RIGHTS": CorporateActionType.RIGHTS_ISSUE,
        "DELIST": CorporateActionType.DELISTING,
    }

    def __init__(self) -> None:
        super().__init__(vendor="bloomberg")

    def normalize(self, payload: Dict[str, Any]) -> CorporateActionEvent:
        event_code = payload.get("EVENT_TYPE") or payload.get("eventType")
        action_type = self.EVENT_MAP.get(event_code, CorporateActionType.OTHER)
        security_id = str(payload.get("SECURITY_ID") or payload.get("securityId"))
        event_id = str(payload.get("EVENT_ID") or payload.get("eventId") or f"{security_id}:{event_code}")
        effective_date = _ensure_datetime(
            payload.get("EFFECTIVE_DATE")
            or payload.get("effectiveDate")
            or payload.get("PAY_DATE")
        )
        details: Dict[str, Any] = {}

        if action_type == CorporateActionType.DIVIDEND:
            details["amount"] = payload.get("DVD_AMOUNT") or payload.get("amount")
            details["currency"] = payload.get("DVD_CRNCY") or payload.get("currency")
        elif action_type == CorporateActionType.SPLIT:
            details["split_ratio"] = payload.get("SPLIT_RATIO") or payload.get("splitRatio")
        elif action_type == CorporateActionType.NAME_CHANGE:
            if payload.get("NEW_NAME") or payload.get("newName"):
                details["new_name"] = payload.get("NEW_NAME") or payload.get("newName")
        elif action_type == CorporateActionType.IDENTIFIER_CHANGE:
            new_ticker = payload.get("NEW_TICKER") or payload.get("newTicker")
            if new_ticker:
                details.setdefault("identifiers", {})["ticker"] = new_ticker
            old_ticker = payload.get("OLD_TICKER") or payload.get("oldTicker")
            if old_ticker:
                details.setdefault("removed_identifiers", []).append("ticker")
                details.setdefault("previous_values", {})["ticker"] = old_ticker
        elif action_type == CorporateActionType.MERGER:
            details["partner"] = payload.get("MERGER_PARTNER") or payload.get("partner")
        elif action_type == CorporateActionType.SPIN_OFF:
            details["spun_company"] = payload.get("SPIN_NAME") or payload.get("spinName")

        details = {key: value for key, value in details.items() if value is not None}

        return CorporateActionEvent(
            event_id=event_id,
            security_id=security_id,
            action_type=action_type,
            effective_date=effective_date,
            details=details,
            announcement_date=_ensure_optional_datetime(
                payload.get("ANNOUNCE_DATE") or payload.get("announcementDate")
            ),
            source=self.vendor,
            raw_payload=payload,
        )


class RefinitivCorporateActionParser(CorporateActionParser):
    """Normalize Refinitiv corporate action payloads."""

    EVENT_MAP = {
        "CA_DIV": CorporateActionType.DIVIDEND,
        "CA_SPLT": CorporateActionType.SPLIT,
        "CA_NAME": CorporateActionType.NAME_CHANGE,
        "CA_TICK": CorporateActionType.IDENTIFIER_CHANGE,
        "CA_MERG": CorporateActionType.MERGER,
        "CA_DELIST": CorporateActionType.DELISTING,
    }

    def __init__(self) -> None:
        super().__init__(vendor="refinitiv")

    def normalize(self, payload: Dict[str, Any]) -> CorporateActionEvent:
        event_code = payload.get("eventCode") or payload.get("EVENT_CODE")
        action_type = self.EVENT_MAP.get(event_code, CorporateActionType.OTHER)
        security_id = str(payload.get("instrumentId") or payload.get("INSTRUMENT_ID"))
        event_id = str(payload.get("eventId") or payload.get("EVENT_ID") or f"{security_id}:{event_code}")
        effective_date = _ensure_datetime(payload.get("effectiveDate") or payload.get("EFFECTIVE_DATE"))
        details: Dict[str, Any] = {}

        if action_type == CorporateActionType.DIVIDEND:
            details["amount"] = payload.get("grossAmount") or payload.get("GROSS_AMOUNT")
            details["currency"] = payload.get("currency") or payload.get("CURRENCY")
        elif action_type == CorporateActionType.SPLIT:
            details["split_ratio"] = payload.get("splitRatio") or payload.get("SPLIT_RATIO")
        elif action_type == CorporateActionType.NAME_CHANGE:
            new_name = payload.get("newName") or payload.get("NEW_NAME")
            if new_name:
                details["new_name"] = new_name
        elif action_type == CorporateActionType.IDENTIFIER_CHANGE:
            identifiers = payload.get("identifiers") or {}
            if identifiers:
                details["identifiers"] = identifiers
        elif action_type == CorporateActionType.DELISTING:
            details["reason"] = payload.get("reason") or payload.get("REASON")

        details = {key: value for key, value in details.items() if value is not None}

        return CorporateActionEvent(
            event_id=event_id,
            security_id=security_id,
            action_type=action_type,
            effective_date=effective_date,
            details=details,
            announcement_date=_ensure_optional_datetime(
                payload.get("announcementDate") or payload.get("ANNOUNCEMENT_DATE")
            ),
            source=self.vendor,
            raw_payload=payload,
        )


class SimulatedCorporateActionParser(CorporateActionParser):
    """Parser used for tests and synthetic data feeds."""

    def __init__(self) -> None:
        super().__init__(vendor="simulated")

    def normalize(self, payload: Dict[str, Any]) -> CorporateActionEvent:
        action_value = str(payload.get("action_type", "other")).lower()
        try:
            action_type = CorporateActionType(action_value)
        except ValueError:
            action_type = CorporateActionType.OTHER
        security_id = str(payload["security_id"])
        event_id = str(payload.get("event_id") or f"{security_id}:{payload['effective_date']}")
        effective_date = _ensure_datetime(payload["effective_date"])
        details = payload.get("details", {})

        return CorporateActionEvent(
            event_id=event_id,
            security_id=security_id,
            action_type=action_type,
            effective_date=effective_date,
            details=details,
            announcement_date=_ensure_optional_datetime(payload.get("announcement_date")),
            source=self.vendor,
            raw_payload=payload,
        )


__all__ = [
    "CorporateActionParser",
    "BloombergCorporateActionParser",
    "RefinitivCorporateActionParser",
    "SimulatedCorporateActionParser",
]
