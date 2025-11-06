"""Repository pattern for trade and entity persistence.

Provides abstract interfaces and in-memory implementations for:
- Trade storage and retrieval
- Book hierarchy persistence
- Counterparty management
- Query capabilities
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional

from neutryx.portfolio.contracts.counterparty import Counterparty
from neutryx.portfolio.contracts.trade import ProductType, Trade, TradeStatus
from neutryx.portfolio.books import Book, Desk, LegalEntity, BusinessUnit, Trader


class TradeRepository(ABC):
    """Abstract repository for trade persistence."""

    @abstractmethod
    def save(self, trade: Trade) -> None:
        """Save a trade."""
        pass

    @abstractmethod
    def find_by_id(self, trade_id: str) -> Optional[Trade]:
        """Find a trade by ID."""
        pass

    @abstractmethod
    def find_all(self) -> List[Trade]:
        """Find all trades."""
        pass

    @abstractmethod
    def find_by_counterparty(self, counterparty_id: str) -> List[Trade]:
        """Find trades by counterparty."""
        pass

    @abstractmethod
    def find_by_book(self, book_id: str) -> List[Trade]:
        """Find trades by book."""
        pass

    @abstractmethod
    def find_by_status(self, status: TradeStatus) -> List[Trade]:
        """Find trades by status."""
        pass

    @abstractmethod
    def delete(self, trade_id: str) -> bool:
        """Delete a trade."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total trades."""
        pass


class BookRepository(ABC):
    """Abstract repository for book hierarchy persistence."""

    @abstractmethod
    def save_legal_entity(self, entity: LegalEntity) -> None:
        """Save a legal entity."""
        pass

    @abstractmethod
    def save_business_unit(self, bu: BusinessUnit) -> None:
        """Save a business unit."""
        pass

    @abstractmethod
    def save_desk(self, desk: Desk) -> None:
        """Save a desk."""
        pass

    @abstractmethod
    def save_book(self, book: Book) -> None:
        """Save a book."""
        pass

    @abstractmethod
    def save_trader(self, trader: Trader) -> None:
        """Save a trader."""
        pass

    @abstractmethod
    def find_book_by_id(self, book_id: str) -> Optional[Book]:
        """Find a book by ID."""
        pass

    @abstractmethod
    def find_books_by_desk(self, desk_id: str) -> List[Book]:
        """Find books by desk."""
        pass

    @abstractmethod
    def find_desk_by_id(self, desk_id: str) -> Optional[Desk]:
        """Find a desk by ID."""
        pass


class CounterpartyRepository(ABC):
    """Abstract repository for counterparty persistence."""

    @abstractmethod
    def save(self, counterparty: Counterparty) -> None:
        """Save a counterparty."""
        pass

    @abstractmethod
    def find_by_id(self, counterparty_id: str) -> Optional[Counterparty]:
        """Find a counterparty by ID."""
        pass

    @abstractmethod
    def find_by_lei(self, lei: str) -> Optional[Counterparty]:
        """Find a counterparty by LEI."""
        pass

    @abstractmethod
    def find_all(self) -> List[Counterparty]:
        """Find all counterparties."""
        pass


class InMemoryTradeRepository(TradeRepository):
    """In-memory implementation of trade repository.

    Suitable for testing and small-scale applications.
    """

    def __init__(self):
        """Initialize in-memory repository."""
        self._trades: Dict[str, Trade] = {}

    def save(self, trade: Trade) -> None:
        """Save a trade."""
        self._trades[trade.id] = trade

    def find_by_id(self, trade_id: str) -> Optional[Trade]:
        """Find a trade by ID."""
        return self._trades.get(trade_id)

    def find_all(self) -> List[Trade]:
        """Find all trades."""
        return list(self._trades.values())

    def find_by_counterparty(self, counterparty_id: str) -> List[Trade]:
        """Find trades by counterparty."""
        return [t for t in self._trades.values() if t.counterparty_id == counterparty_id]

    def find_by_book(self, book_id: str) -> List[Trade]:
        """Find trades by book."""
        return [t for t in self._trades.values() if t.book_id == book_id]

    def find_by_desk(self, desk_id: str) -> List[Trade]:
        """Find trades by desk."""
        return [t for t in self._trades.values() if t.desk_id == desk_id]

    def find_by_trader(self, trader_id: str) -> List[Trade]:
        """Find trades by trader."""
        return [t for t in self._trades.values() if t.trader_id == trader_id]

    def find_by_status(self, status: TradeStatus) -> List[Trade]:
        """Find trades by status."""
        return [t for t in self._trades.values() if t.status == status]

    def find_by_product_type(self, product_type: ProductType) -> List[Trade]:
        """Find trades by product type."""
        return [t for t in self._trades.values() if t.product_type == product_type]

    def find_by_date_range(self, start_date: date, end_date: date) -> List[Trade]:
        """Find trades by trade date range."""
        return [
            t
            for t in self._trades.values()
            if start_date <= t.trade_date <= end_date
        ]

    def find_maturing_before(self, cutoff_date: date) -> List[Trade]:
        """Find trades maturing before a date."""
        return [
            t
            for t in self._trades.values()
            if t.maturity_date and t.maturity_date < cutoff_date
        ]

    def delete(self, trade_id: str) -> bool:
        """Delete a trade."""
        if trade_id in self._trades:
            del self._trades[trade_id]
            return True
        return False

    def count(self) -> int:
        """Count total trades."""
        return len(self._trades)

    def clear(self) -> None:
        """Clear all trades (useful for testing)."""
        self._trades.clear()


class InMemoryBookRepository(BookRepository):
    """In-memory implementation of book repository."""

    def __init__(self):
        """Initialize in-memory repository."""
        self._legal_entities: Dict[str, LegalEntity] = {}
        self._business_units: Dict[str, BusinessUnit] = {}
        self._desks: Dict[str, Desk] = {}
        self._books: Dict[str, Book] = {}
        self._traders: Dict[str, Trader] = {}

    def save_legal_entity(self, entity: LegalEntity) -> None:
        """Save a legal entity."""
        self._legal_entities[entity.id] = entity

    def save_business_unit(self, bu: BusinessUnit) -> None:
        """Save a business unit."""
        self._business_units[bu.id] = bu

    def save_desk(self, desk: Desk) -> None:
        """Save a desk."""
        self._desks[desk.id] = desk

    def save_book(self, book: Book) -> None:
        """Save a book."""
        self._books[book.id] = book

    def save_trader(self, trader: Trader) -> None:
        """Save a trader."""
        self._traders[trader.id] = trader

    def find_book_by_id(self, book_id: str) -> Optional[Book]:
        """Find a book by ID."""
        return self._books.get(book_id)

    def find_books_by_desk(self, desk_id: str) -> List[Book]:
        """Find books by desk."""
        return [b for b in self._books.values() if b.desk_id == desk_id]

    def find_desk_by_id(self, desk_id: str) -> Optional[Desk]:
        """Find a desk by ID."""
        return self._desks.get(desk_id)

    def find_desks_by_business_unit(self, bu_id: str) -> List[Desk]:
        """Find desks by business unit."""
        return [d for d in self._desks.values() if d.business_unit_id == bu_id]

    def find_trader_by_id(self, trader_id: str) -> Optional[Trader]:
        """Find a trader by ID."""
        return self._traders.get(trader_id)

    def find_traders_by_desk(self, desk_id: str) -> List[Trader]:
        """Find traders by desk."""
        return [t for t in self._traders.values() if t.desk_id == desk_id]

    def find_all_books(self) -> List[Book]:
        """Find all books."""
        return list(self._books.values())

    def find_all_desks(self) -> List[Desk]:
        """Find all desks."""
        return list(self._desks.values())

    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        self._legal_entities.clear()
        self._business_units.clear()
        self._desks.clear()
        self._books.clear()
        self._traders.clear()


class InMemoryCounterpartyRepository(CounterpartyRepository):
    """In-memory implementation of counterparty repository."""

    def __init__(self):
        """Initialize in-memory repository."""
        self._counterparties: Dict[str, Counterparty] = {}
        self._lei_index: Dict[str, str] = {}  # LEI -> counterparty_id

    def save(self, counterparty: Counterparty) -> None:
        """Save a counterparty."""
        self._counterparties[counterparty.id] = counterparty
        if counterparty.lei:
            self._lei_index[counterparty.lei] = counterparty.id

    def find_by_id(self, counterparty_id: str) -> Optional[Counterparty]:
        """Find a counterparty by ID."""
        return self._counterparties.get(counterparty_id)

    def find_by_lei(self, lei: str) -> Optional[Counterparty]:
        """Find a counterparty by LEI."""
        counterparty_id = self._lei_index.get(lei)
        if counterparty_id:
            return self._counterparties.get(counterparty_id)
        return None

    def find_all(self) -> List[Counterparty]:
        """Find all counterparties."""
        return list(self._counterparties.values())

    def delete(self, counterparty_id: str) -> bool:
        """Delete a counterparty."""
        counterparty = self._counterparties.get(counterparty_id)
        if counterparty:
            if counterparty.lei:
                self._lei_index.pop(counterparty.lei, None)
            del self._counterparties[counterparty_id]
            return True
        return False

    def count(self) -> int:
        """Count total counterparties."""
        return len(self._counterparties)

    def clear(self) -> None:
        """Clear all counterparties (useful for testing)."""
        self._counterparties.clear()
        self._lei_index.clear()


class RepositoryFactory:
    """Factory for creating repository instances."""

    @staticmethod
    def create_in_memory_repositories() -> tuple[TradeRepository, BookRepository, CounterpartyRepository]:
        """Create a set of in-memory repositories.

        Returns:
            Tuple of (trade_repo, book_repo, counterparty_repo)
        """
        return (
            InMemoryTradeRepository(),
            InMemoryBookRepository(),
            InMemoryCounterpartyRepository(),
        )


__all__ = [
    "TradeRepository",
    "BookRepository",
    "CounterpartyRepository",
    "InMemoryTradeRepository",
    "InMemoryBookRepository",
    "InMemoryCounterpartyRepository",
    "RepositoryFactory",
]
