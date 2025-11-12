"""Security master data model with identifier mapping and version control."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Any, Iterable
import bisect


class SecurityIdentifierType(Enum):
    """Supported security identifier types."""

    TICKER = "ticker"
    ISIN = "isin"
    CUSIP = "cusip"
    SEDOL = "sedol"
    FIGI = "figi"


@dataclass(frozen=True)
class SecurityIdentifier:
    """Security identifier representation."""

    id_type: SecurityIdentifierType
    value: str
    primary: bool = False

    def normalized(self) -> str:
        """Return normalized value used for mappings."""

        return self.value.strip().upper()


class CorporateActionType(Enum):
    """Supported corporate action event types."""

    DIVIDEND = "dividend"
    SPLIT = "split"
    MERGER = "merger"
    SPIN_OFF = "spin_off"
    SYMBOL_CHANGE = "symbol_change"


@dataclass
class CorporateActionEvent:
    """Normalized corporate action event."""

    action_type: CorporateActionType
    effective_date: date
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityVersion:
    """Immutable snapshot of security attributes at an effective date."""

    version: int
    effective_date: date
    attributes: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)


@dataclass
class SecurityRecord:
    """Represents a security master record with version history."""

    security_id: str
    name: str
    asset_class: str
    identifiers: Dict[SecurityIdentifierType, SecurityIdentifier] = field(
        default_factory=dict
    )
    versions: List[SecurityVersion] = field(default_factory=list)
    corporate_actions: List[CorporateActionEvent] = field(default_factory=list)

    def add_identifier(self, identifier: SecurityIdentifier) -> None:
        self.identifiers[identifier.id_type] = identifier

    def latest_version(self) -> Optional[SecurityVersion]:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: (v.effective_date, v.version))

    def get_attributes(self, as_of: Optional[date] = None) -> Dict[str, Any]:
        if not self.versions:
            return {}

        if as_of is None:
            return dict(self.latest_version().attributes)

        # Versions sorted by effective date to allow binary search
        versions_sorted = sorted(
            self.versions, key=lambda v: (v.effective_date, v.version)
        )
        effective_dates = [v.effective_date.toordinal() for v in versions_sorted]
        idx = bisect.bisect_right(effective_dates, as_of.toordinal()) - 1
        if idx < 0:
            return {}
        return dict(versions_sorted[idx].attributes)


class SecurityMaster:
    """Security master service with identifier mapping and versioning."""

    def __init__(self) -> None:
        self._records: Dict[str, SecurityRecord] = {}
        self._identifier_map: Dict[
            SecurityIdentifierType, Dict[str, str]
        ] = {id_type: {} for id_type in SecurityIdentifierType}

    # ------------------------------------------------------------------
    # Registration & Retrieval
    # ------------------------------------------------------------------
    def register_security(
        self,
        security_id: str,
        name: str,
        asset_class: str,
        identifiers: Iterable[SecurityIdentifier],
        attributes: Optional[Dict[str, Any]] = None,
        effective_date: Optional[date] = None,
    ) -> SecurityRecord:
        if security_id in self._records:
            raise ValueError(f"Security {security_id} already registered")

        record = SecurityRecord(
            security_id=security_id,
            name=name,
            asset_class=asset_class,
        )

        for identifier in identifiers:
            self._add_identifier_mapping(security_id, identifier)
            record.add_identifier(identifier)

        version = SecurityVersion(
            version=1,
            effective_date=effective_date or date.today(),
            attributes=dict(attributes or {}),
        )
        record.versions.append(version)
        self._records[security_id] = record
        return record

    def update_security(
        self,
        security_id: str,
        updates: Dict[str, Any],
        effective_date: Optional[date] = None,
    ) -> SecurityVersion:
        record = self._get_required_record(security_id)
        base_attributes = {}
        latest = record.latest_version()
        if latest:
            base_attributes.update(latest.attributes)

        base_attributes.update(updates)
        version_number = (latest.version + 1) if latest else 1
        new_version = SecurityVersion(
            version=version_number,
            effective_date=effective_date or date.today(),
            attributes=base_attributes,
        )
        record.versions.append(new_version)
        return new_version

    def add_identifier(
        self, security_id: str, identifier: SecurityIdentifier
    ) -> None:
        record = self._get_required_record(security_id)
        self._add_identifier_mapping(security_id, identifier)
        record.add_identifier(identifier)

    def get_security(self, security_id: str) -> Optional[SecurityRecord]:
        return self._records.get(security_id)

    def get_security_by_identifier(
        self, identifier_value: str, identifier_type: SecurityIdentifierType
    ) -> Optional[SecurityRecord]:
        normalized = identifier_value.strip().upper()
        security_id = self._identifier_map[identifier_type].get(normalized)
        if not security_id:
            return None
        return self._records.get(security_id)

    def list_securities(self) -> List[SecurityRecord]:
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Corporate Actions
    # ------------------------------------------------------------------
    def apply_corporate_action(
        self, security_id: str, event: CorporateActionEvent
    ) -> None:
        record = self._get_required_record(security_id)
        record.corporate_actions.append(event)

        updates: Dict[str, Any] = {}
        if event.action_type == CorporateActionType.SYMBOL_CHANGE:
            new_ticker = event.details.get("new_ticker")
            old_ticker = event.details.get("old_ticker")
            if new_ticker:
                updates["ticker"] = new_ticker
                self._update_identifier_value(
                    record,
                    SecurityIdentifierType.TICKER,
                    old_ticker,
                    new_ticker,
                )
        elif event.action_type == CorporateActionType.SPLIT:
            if "split_ratio" in event.details:
                updates["split_ratio"] = event.details["split_ratio"]
        elif event.action_type == CorporateActionType.DIVIDEND:
            latest = record.latest_version()
            dividends = list(latest.attributes.get("dividends", [])) if latest else []
            dividends.append(
                {
                    "amount": event.details.get("amount"),
                    "currency": event.details.get("currency"),
                    "pay_date": event.details.get("pay_date"),
                }
            )
            updates["dividends"] = dividends

        if updates:
            self.update_security(security_id, updates, event.effective_date)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_identifier_mapping(
        self, security_id: str, identifier: SecurityIdentifier
    ) -> None:
        normalized = identifier.normalized()
        mapping = self._identifier_map[identifier.id_type]
        if normalized in mapping and mapping[normalized] != security_id:
            raise ValueError(
                f"Identifier {identifier.id_type.value}:{identifier.value} "
                "already mapped to another security"
            )
        mapping[normalized] = security_id

    def _update_identifier_value(
        self,
        record: SecurityRecord,
        id_type: SecurityIdentifierType,
        old_value: Optional[str],
        new_value: str,
    ) -> None:
        if old_value:
            old_normalized = old_value.strip().upper()
            self._identifier_map[id_type].pop(old_normalized, None)
        identifier = SecurityIdentifier(id_type=id_type, value=new_value, primary=True)
        self._identifier_map[id_type][identifier.normalized()] = record.security_id
        record.add_identifier(identifier)

    def _get_required_record(self, security_id: str) -> SecurityRecord:
        try:
            return self._records[security_id]
        except KeyError as exc:
            raise KeyError(f"Security {security_id} not found") from exc
