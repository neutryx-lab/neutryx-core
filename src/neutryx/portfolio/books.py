"""Book hierarchy models for trade organization.

Implements organizational structure:
LegalEntity → BusinessUnit → Desk → Book → Trader

Books represent trading portfolios assigned to specific desks and traders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field


class EntityStatus(Enum):
    """Status of organizational entities."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    CLOSED = "closed"


@dataclass
class LegalEntity:
    """Legal entity representing a company or subsidiary."""

    id: str
    name: str
    lei: Optional[str] = None  # Legal Entity Identifier
    jurisdiction: Optional[str] = None
    status: EntityStatus = EntityStatus.ACTIVE
    business_units: List[str] = field(default_factory=list)  # BU IDs
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if legal entity is active."""
        return self.status == EntityStatus.ACTIVE

    def add_business_unit(self, bu_id: str) -> None:
        """Add a business unit to this legal entity."""
        if bu_id not in self.business_units:
            self.business_units.append(bu_id)


@dataclass
class BusinessUnit:
    """Business unit within a legal entity."""

    id: str
    name: str
    legal_entity_id: str
    status: EntityStatus = EntityStatus.ACTIVE
    desks: List[str] = field(default_factory=list)  # Desk IDs
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if business unit is active."""
        return self.status == EntityStatus.ACTIVE

    def add_desk(self, desk_id: str) -> None:
        """Add a desk to this business unit."""
        if desk_id not in self.desks:
            self.desks.append(desk_id)


@dataclass
class Desk:
    """Trading desk within a business unit."""

    id: str
    name: str
    business_unit_id: str
    desk_type: Optional[str] = None  # e.g., "rates", "equity", "fx"
    status: EntityStatus = EntityStatus.ACTIVE
    books: List[str] = field(default_factory=list)  # Book IDs
    traders: Set[str] = field(default_factory=set)  # Trader IDs
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if desk is active."""
        return self.status == EntityStatus.ACTIVE

    def add_book(self, book_id: str) -> None:
        """Add a book to this desk."""
        if book_id not in self.books:
            self.books.append(book_id)

    def add_trader(self, trader_id: str) -> None:
        """Add a trader to this desk."""
        self.traders.add(trader_id)

    def remove_trader(self, trader_id: str) -> None:
        """Remove a trader from this desk."""
        self.traders.discard(trader_id)


@dataclass
class Book:
    """Trading book for organizing trades."""

    id: str
    name: str
    desk_id: str
    book_type: Optional[str] = None  # e.g., "proprietary", "flow", "hedge"
    status: EntityStatus = EntityStatus.ACTIVE
    primary_trader_id: Optional[str] = None
    trade_ids: Set[str] = field(default_factory=set)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    created_date: Optional[date] = None
    closed_date: Optional[date] = None

    def is_active(self) -> bool:
        """Check if book is active."""
        return self.status == EntityStatus.ACTIVE

    def add_trade(self, trade_id: str) -> None:
        """Add a trade to this book."""
        self.trade_ids.add(trade_id)

    def remove_trade(self, trade_id: str) -> None:
        """Remove a trade from this book."""
        self.trade_ids.discard(trade_id)

    def get_trade_count(self) -> int:
        """Get number of trades in book."""
        return len(self.trade_ids)

    def set_risk_limit(self, limit_type: str, value: float) -> None:
        """Set a risk limit for the book.
        
        Args:
            limit_type: Type of risk limit (e.g., "dv01", "var", "notional")
            value: Limit value
        """
        self.risk_limits[limit_type] = value

    def get_risk_limit(self, limit_type: str) -> Optional[float]:
        """Get a risk limit value."""
        return self.risk_limits.get(limit_type)


@dataclass
class Trader:
    """Individual trader."""

    id: str
    name: str
    email: Optional[str] = None
    desk_id: Optional[str] = None
    status: EntityStatus = EntityStatus.ACTIVE
    books: List[str] = field(default_factory=list)  # Book IDs trader manages
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)
    hire_date: Optional[date] = None

    def is_active(self) -> bool:
        """Check if trader is active."""
        return self.status == EntityStatus.ACTIVE

    def add_book(self, book_id: str) -> None:
        """Assign a book to this trader."""
        if book_id not in self.books:
            self.books.append(book_id)

    def remove_book(self, book_id: str) -> None:
        """Remove a book from this trader."""
        if book_id in self.books:
            self.books.remove(book_id)

    def has_permission(self, permission: str) -> bool:
        """Check if trader has a specific permission."""
        return permission in self.permissions

    def grant_permission(self, permission: str) -> None:
        """Grant a permission to trader."""
        self.permissions.add(permission)

    def revoke_permission(self, permission: str) -> None:
        """Revoke a permission from trader."""
        self.permissions.discard(permission)


class BookHierarchy:
    """Manages the complete book hierarchy structure."""

    def __init__(self):
        self.legal_entities: Dict[str, LegalEntity] = {}
        self.business_units: Dict[str, BusinessUnit] = {}
        self.desks: Dict[str, Desk] = {}
        self.books: Dict[str, Book] = {}
        self.traders: Dict[str, Trader] = {}

    def add_legal_entity(self, entity: LegalEntity) -> None:
        """Add a legal entity to the hierarchy."""
        self.legal_entities[entity.id] = entity

    def add_business_unit(self, bu: BusinessUnit) -> None:
        """Add a business unit and link to legal entity."""
        self.business_units[bu.id] = bu
        if bu.legal_entity_id in self.legal_entities:
            self.legal_entities[bu.legal_entity_id].add_business_unit(bu.id)

    def add_desk(self, desk: Desk) -> None:
        """Add a desk and link to business unit."""
        self.desks[desk.id] = desk
        if desk.business_unit_id in self.business_units:
            self.business_units[desk.business_unit_id].add_desk(desk.id)

    def add_book(self, book: Book) -> None:
        """Add a book and link to desk."""
        self.books[book.id] = book
        if book.desk_id in self.desks:
            self.desks[book.desk_id].add_book(book.id)

    def add_trader(self, trader: Trader) -> None:
        """Add a trader and link to desk."""
        self.traders[trader.id] = trader
        if trader.desk_id and trader.desk_id in self.desks:
            self.desks[trader.desk_id].add_trader(trader.id)

    def get_book_path(self, book_id: str) -> Optional[Dict[str, str]]:
        """Get full hierarchy path for a book.
        
        Returns dict with: legal_entity_id, business_unit_id, desk_id, book_id
        """
        book = self.books.get(book_id)
        if not book:
            return None

        desk = self.desks.get(book.desk_id)
        if not desk:
            return None

        bu = self.business_units.get(desk.business_unit_id)
        if not bu:
            return None

        return {
            "legal_entity_id": bu.legal_entity_id,
            "business_unit_id": bu.id,
            "desk_id": desk.id,
            "book_id": book.id,
        }

    def get_books_by_desk(self, desk_id: str) -> List[Book]:
        """Get all books for a desk."""
        desk = self.desks.get(desk_id)
        if not desk:
            return []
        return [self.books[book_id] for book_id in desk.books if book_id in self.books]

    def get_books_by_trader(self, trader_id: str) -> List[Book]:
        """Get all books managed by a trader."""
        trader = self.traders.get(trader_id)
        if not trader:
            return []
        return [self.books[book_id] for book_id in trader.books if book_id in self.books]

    def get_desks_by_business_unit(self, bu_id: str) -> List[Desk]:
        """Get all desks for a business unit."""
        bu = self.business_units.get(bu_id)
        if not bu:
            return []
        return [self.desks[desk_id] for desk_id in bu.desks if desk_id in self.desks]

    def validate_book_assignment(self, book_id: str, trader_id: str) -> bool:
        """Validate if a book can be assigned to a trader.
        
        Checks:
        - Both book and trader exist
        - Both are active
        - Trader is on the same desk as the book
        """
        book = self.books.get(book_id)
        trader = self.traders.get(trader_id)

        if not book or not trader:
            return False

        if not book.is_active() or not trader.is_active():
            return False

        if trader.desk_id != book.desk_id:
            return False

        return True


__all__ = [
    "EntityStatus",
    "LegalEntity",
    "BusinessUnit",
    "Desk",
    "Book",
    "Trader",
    "BookHierarchy",
]
