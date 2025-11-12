"""Security master data model with identifier management and versioning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..data_models import AssetClass


class CorporateActionType(Enum):
    """Supported corporate action event types."""

    DIVIDEND = "dividend"
    SPLIT = "split"
    MERGER = "merger"
    SPIN_OFF = "spin_off"
    RIGHTS_ISSUE = "rights_issue"
    NAME_CHANGE = "name_change"
    IDENTIFIER_CHANGE = "identifier_change"
    DELISTING = "delisting"
    OTHER = "other"


@dataclass(frozen=True)
class CorporateActionEvent:
    """Normalized corporate action event."""

    event_id: str
    security_id: str
    action_type: CorporateActionType
    effective_date: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    announcement_date: Optional[datetime] = None
    source: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass
class SecurityMasterRecord:
    """Versioned record of a security."""

    security_id: str
    asset_class: AssetClass
    name: str
    identifiers: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    effective_from: datetime = field(default_factory=datetime.utcnow)
    effective_to: Optional[datetime] = None
    status: str = "active"
    events: List[CorporateActionEvent] = field(default_factory=list)

    def is_effective(self, when: datetime) -> bool:
        """Check if the record is effective for the given timestamp."""

        return self.effective_from <= when and (
            self.effective_to is None or when < self.effective_to
        )


class SecurityMasterError(Exception):
    """Base error for security master operations."""


class SecurityNotFoundError(SecurityMasterError):
    """Raised when a security record is not found."""


class SecurityMaster:
    """In-memory security master with identifier mapping and versioning."""

    def __init__(self) -> None:
        self._records: Dict[str, List[SecurityMasterRecord]] = {}
        self._identifier_index: Dict[Tuple[str, str], str] = {}

    def register_security(
        self,
        security_id: str,
        asset_class: AssetClass,
        name: str,
        identifiers: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None,
        effective_from: Optional[datetime] = None,
        status: str = "active",
    ) -> SecurityMasterRecord:
        """Register a new security in the master."""

        if security_id in self._records:
            raise SecurityMasterError(f"Security '{security_id}' already exists")

        effective = effective_from or datetime.utcnow()
        record = SecurityMasterRecord(
            security_id=security_id,
            asset_class=asset_class,
            name=name,
            identifiers=_sanitize_identifiers(identifiers),
            metadata=deepcopy(metadata) if metadata else {},
            version=1,
            created_at=effective,
            updated_at=effective,
            effective_from=effective,
            status=status,
        )
        self._records[security_id] = [record]
        self._rebuild_identifier_index(security_id, record)
        return record

    def update_security(
        self,
        security_id: str,
        *,
        name: Optional[str] = None,
        identifiers: Optional[Dict[str, Optional[str]]] = None,
        remove_identifiers: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        effective_from: Optional[datetime] = None,
        event: Optional[CorporateActionEvent] = None,
    ) -> SecurityMasterRecord:
        """Update an existing security, creating a new version."""

        versions = self._records.get(security_id)
        if not versions:
            raise SecurityNotFoundError(f"Security '{security_id}' not found")

        latest = versions[-1]
        effective = effective_from or datetime.utcnow()
        if effective < latest.effective_from:
            raise SecurityMasterError("Effective date cannot be before current version")

        latest.effective_to = effective
        latest.updated_at = effective

        new_identifiers = dict(latest.identifiers)
        if remove_identifiers is not None:
            for key in remove_identifiers:
                new_identifiers.pop(_normalize_identifier_key(key), None)
        if identifiers is not None:
            for key, value in identifiers.items():
                norm_key = _normalize_identifier_key(key)
                if value is None:
                    new_identifiers.pop(norm_key, None)
                else:
                    new_identifiers[norm_key] = str(value).strip()

        new_metadata = deepcopy(latest.metadata)
        if metadata is not None:
            for key, value in metadata.items():
                if value is None:
                    new_metadata.pop(key, None)
                else:
                    new_metadata[key] = value

        new_events = list(latest.events)
        if event is not None:
            new_events.append(event)

        new_record = SecurityMasterRecord(
            security_id=security_id,
            asset_class=latest.asset_class,
            name=name if name is not None else latest.name,
            identifiers=new_identifiers,
            metadata=new_metadata,
            version=latest.version + 1,
            created_at=latest.created_at,
            updated_at=effective,
            effective_from=effective,
            status=status if status is not None else latest.status,
            events=new_events,
        )

        versions.append(new_record)
        self._rebuild_identifier_index(security_id, new_record)
        return new_record

    def apply_corporate_action(
        self, event: CorporateActionEvent
    ) -> SecurityMasterRecord:
        """Apply a corporate action and create a new security version."""

        security_id = event.security_id
        if security_id not in self._records:
            raise SecurityNotFoundError(
                f"Cannot apply corporate action to unknown security '{security_id}'"
            )
        latest_record = self.get_security_by_id(security_id)
        if latest_record is None:
            raise SecurityNotFoundError(
                f"No active record found for security '{security_id}'"
            )

        name_update: Optional[str] = None
        status_update: Optional[str] = None
        metadata_updates: Dict[str, Any] = {}
        identifier_updates: Dict[str, Optional[str]] = {}

        if event.action_type == CorporateActionType.NAME_CHANGE:
            name_update = event.details.get("new_name")
        elif event.action_type == CorporateActionType.IDENTIFIER_CHANGE:
            identifier_updates.update(event.details.get("identifiers", {}))
        elif event.action_type == CorporateActionType.SPLIT:
            if "split_ratio" in event.details:
                metadata_updates["last_split_ratio"] = event.details["split_ratio"]
        elif event.action_type == CorporateActionType.DELISTING:
            status_update = "inactive"
            if event.details:
                metadata_updates["delisting_details"] = event.details
        elif event.action_type == CorporateActionType.MERGER:
            status_update = "merged"
            if event.details:
                metadata_updates["merger_details"] = event.details
        elif event.action_type == CorporateActionType.DIVIDEND:
            current = deepcopy(latest_record.metadata.get("dividends", []))
            current.append(event.details)
            metadata_updates["dividends"] = current
        else:
            if event.details:
                current = deepcopy(latest_record.metadata.get("corporate_actions", []))
                current.append(event.details)
                metadata_updates["corporate_actions"] = current

        return self.update_security(
            security_id,
            name=name_update,
            identifiers=identifier_updates if identifier_updates else None,
            remove_identifiers=event.details.get("removed_identifiers")
            if event.action_type == CorporateActionType.IDENTIFIER_CHANGE
            else None,
            metadata=metadata_updates if metadata_updates else None,
            status=status_update,
            effective_from=event.effective_date,
            event=event,
        )

    def get_security_by_id(
        self, security_id: str, as_of: Optional[datetime] = None
    ) -> Optional[SecurityMasterRecord]:
        """Retrieve security by its identifier, optionally as of a timestamp."""

        versions = self._records.get(security_id)
        if not versions:
            return None

        if as_of is None:
            return versions[-1]

        for record in reversed(versions):
            if record.is_effective(as_of):
                return record
        return None

    def get_security_by_identifier(
        self,
        id_type: str,
        value: str,
        as_of: Optional[datetime] = None,
    ) -> Optional[SecurityMasterRecord]:
        """Lookup a security by one of its identifiers."""

        norm_key = _normalize_identifier_key(id_type)
        norm_value = _normalize_identifier_value(value)

        if as_of is None:
            security_id = self._identifier_index.get((norm_key, norm_value))
            if security_id:
                return self.get_security_by_id(security_id)

        for security_id, versions in self._records.items():
            for record in reversed(versions):
                identifier_value = record.identifiers.get(norm_key)
                if identifier_value and _normalize_identifier_value(identifier_value) == norm_value:
                    if as_of is None or record.is_effective(as_of):
                        return record
        return None

    def get_versions(self, security_id: str) -> List[SecurityMasterRecord]:
        """Return all versions for a security."""

        return list(self._records.get(security_id, []))

    def _rebuild_identifier_index(
        self, security_id: str, record: SecurityMasterRecord
    ) -> None:
        """Update identifier index for the latest version."""

        keys_to_remove = [key for key, value in self._identifier_index.items() if value == security_id]
        for key in keys_to_remove:
            del self._identifier_index[key]

        for key, value in record.identifiers.items():
            self._identifier_index[(key, _normalize_identifier_value(value))] = security_id


def _normalize_identifier_key(identifier: Any) -> str:
    return str(identifier).strip().lower()


def _normalize_identifier_value(value: str) -> str:
    return str(value).strip().upper()


def _sanitize_identifiers(identifiers: Dict[str, str]) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for key, value in identifiers.items():
        if value is None:
            continue
        sanitized[_normalize_identifier_key(key)] = str(value).strip()
    return sanitized


__all__ = [
    "CorporateActionType",
    "CorporateActionEvent",
    "SecurityMasterRecord",
    "SecurityMasterError",
    "SecurityNotFoundError",
    "SecurityMaster",
]
