"""Security master storage abstractions and retrieval APIs.

The security master centralises reference data for tradable instruments and
provides a uniform lookup interface for validation and portfolio workflows.
The design favours a storage-agnostic façade so the same API can be backed by
an in-memory dictionary for tests or an external database in production.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

from neutryx.market.data_models import AssetClass


class SecurityMasterError(Exception):
    """Base exception for security master issues."""


class SecurityNotFoundError(SecurityMasterError):
    """Raised when a requested security cannot be located."""


class SecurityAlreadyExistsError(SecurityMasterError):
    """Raised when attempting to register a duplicate security."""


class SecurityInactiveError(SecurityMasterError):
    """Raised when an inactive security is requested as active."""


@dataclass(frozen=True)
class SecurityRecord:
    """Reference data for a tradable security."""

    security_id: str
    asset_class: AssetClass
    ticker: Optional[str] = None
    isin: Optional[str] = None
    cusip: Optional[str] = None
    exchange: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None
    status: str = "active"
    active_from: Optional[date] = None
    active_to: Optional[date] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_active(self, *, as_of: Optional[date] = None) -> bool:
        """Return True if the security is considered active."""

        if self.status.lower() not in {"active", "trading"}:
            return False

        if as_of is None:
            return True

        if self.active_from and as_of < self.active_from:
            return False

        if self.active_to and as_of > self.active_to:
            return False

        return True

    def to_summary(self) -> Dict[str, Optional[str]]:
        """Return a JSON-serialisable representation."""

        return {
            "security_id": self.security_id,
            "asset_class": self.asset_class.value,
            "ticker": self.ticker,
            "isin": self.isin,
            "cusip": self.cusip,
            "exchange": self.exchange,
            "country": self.country,
            "description": self.description,
            "status": self.status,
        }


class SecurityMasterStorage(ABC):
    """Abstract storage backend for security reference data."""

    @abstractmethod
    def save(self, record: SecurityRecord) -> None:
        """Persist or update a security record."""

    @abstractmethod
    def get(self, security_id: str) -> Optional[SecurityRecord]:
        """Retrieve a security by its primary identifier."""

    @abstractmethod
    def find_by_identifier(
        self, identifier_type: str, identifier_value: str
    ) -> Optional[SecurityRecord]:
        """Return the first matching record for an identifier."""

    @abstractmethod
    def all(self) -> Iterable[SecurityRecord]:
        """Return all security records."""


class InMemorySecurityMasterStorage(SecurityMasterStorage):
    """Simple in-memory storage suitable for tests and prototyping."""

    def __init__(self) -> None:
        self._records: Dict[str, SecurityRecord] = {}
        self._index: MutableMapping[str, Dict[str, str]] = {}

    def _normalise(self, identifier_type: str, identifier_value: str) -> str:
        value = identifier_value.strip()
        if identifier_type in {"ticker", "exchange", "security_id"}:
            return value.upper()
        return value.upper()

    def _index_record(self, record: SecurityRecord) -> None:
        for identifier_type in ("security_id", "ticker", "isin", "cusip"):
            value = getattr(record, identifier_type, None)
            if value:
                bucket = self._index.setdefault(identifier_type, {})
                bucket[self._normalise(identifier_type, value)] = record.security_id

    def save(self, record: SecurityRecord) -> None:
        if record.security_id in self._records:
            raise SecurityAlreadyExistsError(
                f"Security '{record.security_id}' already registered"
            )

        self._records[record.security_id] = record
        self._index_record(record)

    def get(self, security_id: str) -> Optional[SecurityRecord]:
        return self._records.get(security_id)

    def find_by_identifier(
        self, identifier_type: str, identifier_value: str
    ) -> Optional[SecurityRecord]:
        bucket = self._index.get(identifier_type)
        if not bucket:
            return None

        normalised = self._normalise(identifier_type, identifier_value)
        security_id = bucket.get(normalised)
        if security_id is None:
            return None
        return self._records.get(security_id)

    def all(self) -> Iterator[SecurityRecord]:
        return iter(self._records.values())


class SecurityMaster:
    """High-level retrieval API sitting on top of storage backends."""

    def __init__(self, storage: SecurityMasterStorage) -> None:
        self._storage = storage

    def register(self, record: SecurityRecord) -> None:
        """Register a new security in the master."""

        self._storage.save(record)

    def get(self, security_id: str, *, as_of: Optional[date] = None) -> SecurityRecord:
        """Fetch a security record by its identifier."""

        record = self._storage.get(security_id)
        if record is None:
            raise SecurityNotFoundError(security_id)

        if as_of and not record.is_active(as_of=as_of):
            raise SecurityInactiveError(security_id)

        return record

    def lookup(
        self,
        identifier_value: str,
        *,
        identifier_type: str = "ticker",
        as_of: Optional[date] = None,
        require_active: bool = True,
    ) -> SecurityRecord:
        """Lookup a security by an alternate identifier."""

        record = self._storage.find_by_identifier(identifier_type, identifier_value)
        if record is None:
            raise SecurityNotFoundError(
                f"{identifier_type}={identifier_value} not found in security master"
            )

        if require_active and as_of and not record.is_active(as_of=as_of):
            raise SecurityInactiveError(record.security_id)

        return record

    def ensure_active(
        self, security_id: str, *, as_of: Optional[date] = None
    ) -> SecurityRecord:
        """Validate that a security is active as of the provided date."""

        record = self.get(security_id, as_of=as_of)
        if as_of and not record.is_active(as_of=as_of):
            raise SecurityInactiveError(security_id)
        return record

    def snapshot(self) -> Mapping[str, SecurityRecord]:
        """Return an immutable snapshot of all registered securities."""

        return {record.security_id: record for record in self._storage.all()}


__all__ = [
    "SecurityMaster",
    "SecurityMasterStorage",
    "InMemorySecurityMasterStorage",
    "SecurityRecord",
    "SecurityMasterError",
    "SecurityNotFoundError",
    "SecurityAlreadyExistsError",
    "SecurityInactiveError",
]
