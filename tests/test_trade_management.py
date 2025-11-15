"""Tests for trade management system."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from neutryx.portfolio.contracts.counterparty import Counterparty
from neutryx.portfolio.contracts.counterparty_codes import (
    CounterpartyCodeGenerator,
    CounterpartyType,
    create_simple_counterparty_code_generator,
)
from neutryx.portfolio.contracts.trade import ProductType, Trade, TradeStatus
from neutryx.portfolio.books import (
    Book,
    BookHierarchy,
    BusinessUnit,
    Desk,
    EntityStatus,
    LegalEntity,
    Trader,
)
from neutryx.portfolio.id_generator import (
    IDPattern,
    TradeIDGenerator,
    create_book_id_generator,
    create_trade_id_generator,
)
from neutryx.portfolio.lifecycle import (
    LifecycleEventType,
    LifecycleManager,
    TradeAmendment,
    TradeNovation,
    TradeTermination,
)
from neutryx.portfolio.portfolio import Portfolio
from neutryx.portfolio.pricing_bridge import MarketData, PricingBridge
from neutryx.portfolio.repository import (
    InMemoryBookRepository,
    InMemoryCounterpartyRepository,
    InMemoryTradeRepository,
    RepositoryFactory,
)


class TestIDGenerator:
    """Test ID generation system."""

    def test_sequential_id_generation(self):
        """Test sequential ID pattern."""
        generator = create_trade_id_generator(prefix="TRD", pattern=IDPattern.SEQUENTIAL)
        id1 = generator.generate()
        id2 = generator.generate()

        assert id1 == "TRD-0001"
        assert id2 == "TRD-0002"
        assert generator.is_valid(id1)
        assert generator.is_valid(id2)

    def test_date_sequential_id_generation(self):
        """Test date-based sequential ID pattern."""
        generator = create_trade_id_generator(pattern=IDPattern.DATE_SEQUENTIAL)
        test_date = date(2025, 3, 15)

        id1 = generator.generate(test_date)
        id2 = generator.generate(test_date)

        assert "20250315" in id1
        assert id1.endswith("-0001")
        assert id2.endswith("-0002")

    def test_uuid_id_generation(self):
        """Test UUID-based ID pattern."""
        generator = create_trade_id_generator(pattern=IDPattern.UUID)
        id1 = generator.generate()
        id2 = generator.generate()

        assert id1.startswith("TRD-")
        assert id2.startswith("TRD-")
        assert id1 != id2
        assert generator.is_valid(id1)

    def test_book_id_generator(self):
        """Test book ID generator."""
        generator = create_book_id_generator()
        book_id = generator.generate()

        assert book_id.startswith("BK-")
        assert generator.is_valid(book_id)


class TestCounterpartyCodeSystem:
    """Test counterparty code generation."""

    def test_simple_code_generation(self):
        """Test simple counterparty code generation."""
        generator = create_simple_counterparty_code_generator()

        code1 = generator.generate("CPTY-001")
        code2 = generator.generate("CPTY-002")

        assert code1 == "CP-0001"
        assert code2 == "CP-0002"
        assert generator.is_valid(code1)

    def test_code_with_lei(self):
        """Test code generation with LEI."""
        generator = CounterpartyCodeGenerator()

        code = generator.generate_from_lei(
            lei="549300ABCDEF12345678", counterparty_id="CPTY-001", name="Bank ABC"
        )

        assert code.startswith("CP-")
        assert generator.is_valid(code)

    def test_code_lookup(self):
        """Test code lookup functionality."""
        generator = create_simple_counterparty_code_generator()

        code = generator.generate("CPTY-001", name="Test Corp")
        mapping = generator.get_mapping(code)

        assert mapping is not None
        assert mapping.counterparty_id == "CPTY-001"
        assert mapping.name == "Test Corp"

    def test_duplicate_code_rejection(self):
        """Test that duplicate codes are rejected."""
        generator = create_simple_counterparty_code_generator()
        code = generator.generate("CPTY-001")

        # Registering the same code should fail
        with pytest.raises(ValueError, match="already registered"):
            generator.register_code(code, "CPTY-002")


class TestBookHierarchy:
    """Test book hierarchy models."""

    def test_legal_entity_creation(self):
        """Test legal entity creation."""
        entity = LegalEntity(
            id="LE-001", name="Neutryx Corp", lei="549300ABCDEF12345678", jurisdiction="US"
        )

        assert entity.is_active()
        assert entity.lei == "549300ABCDEF12345678"

    def test_book_hierarchy_structure(self):
        """Test complete book hierarchy."""
        hierarchy = BookHierarchy()

        # Create hierarchy: LegalEntity -> BusinessUnit -> Desk -> Book
        le = LegalEntity(id="LE-001", name="Neutryx Corp")
        bu = BusinessUnit(id="BU-001", name="Trading", legal_entity_id="LE-001")
        desk = Desk(id="DSK-001", name="Rates Desk", business_unit_id="BU-001", desk_type="rates")
        book = Book(id="BK-001", name="USD Rates", desk_id="DSK-001", book_type="flow")
        trader = Trader(id="TRD-001", name="John Doe", desk_id="DSK-001")

        hierarchy.add_legal_entity(le)
        hierarchy.add_business_unit(bu)
        hierarchy.add_desk(desk)
        hierarchy.add_book(book)
        hierarchy.add_trader(trader)

        # Verify relationships
        assert "BU-001" in le.business_units
        assert "DSK-001" in bu.desks
        assert "BK-001" in desk.books
        assert "TRD-001" in desk.traders

        # Get book path
        path = hierarchy.get_book_path("BK-001")
        assert path["legal_entity_id"] == "LE-001"
        assert path["business_unit_id"] == "BU-001"
        assert path["desk_id"] == "DSK-001"
        assert path["book_id"] == "BK-001"

    def test_book_validation(self):
        """Test book assignment validation."""
        hierarchy = BookHierarchy()

        desk = Desk(id="DSK-001", name="Rates Desk", business_unit_id="BU-001")
        book = Book(id="BK-001", name="USD Rates", desk_id="DSK-001")
        trader = Trader(id="TRD-001", name="John Doe", desk_id="DSK-001")

        hierarchy.add_desk(desk)
        hierarchy.add_book(book)
        hierarchy.add_trader(trader)

        # Valid assignment (same desk)
        assert hierarchy.validate_book_assignment("BK-001", "TRD-001")

        # Invalid assignment (different desk)
        trader2 = Trader(id="TRD-002", name="Jane Smith", desk_id="DSK-002")
        hierarchy.add_trader(trader2)
        assert not hierarchy.validate_book_assignment("BK-001", "TRD-002")


class TestExtendedTradeModel:
    """Test extended Trade model with book fields."""

    def test_trade_with_book_fields(self):
        """Test creating trade with book-related fields."""
        trade = Trade(
            id="TRD-001",
            trade_number="TRD-20250315-0001",
            counterparty_id="CP-001",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            book_id="BK-001",
            desk_id="DSK-001",
            trader_id="TRD-001",
            notional=10_000_000,
            currency="USD",
        )

        assert trade.trade_number == "TRD-20250315-0001"
        assert trade.book_id == "BK-001"
        assert trade.desk_id == "DSK-001"
        assert trade.trader_id == "TRD-001"


class TestPortfolioBookAggregation:
    """Test portfolio book-level aggregation."""

    def test_get_trades_by_book(self):
        """Test filtering trades by book."""
        portfolio = Portfolio(name="Test Portfolio")

        # Create trades in different books
        trade1 = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.EQUITY_OPTION,
            trade_date=date.today(),
            book_id="BK-001",
            mtm=10000.0,
        )
        trade2 = Trade(
            id="TRD-002",
            counterparty_id="CP-001",
            product_type=ProductType.FX_OPTION,
            trade_date=date.today(),
            book_id="BK-001",
            mtm=5000.0,
        )
        trade3 = Trade(
            id="TRD-003",
            counterparty_id="CP-002",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            book_id="BK-002",
            mtm=15000.0,
        )

        portfolio.add_trade(trade1)
        portfolio.add_trade(trade2)
        portfolio.add_trade(trade3)

        # Test book filtering
        bk1_trades = portfolio.get_trades_by_book("BK-001")
        assert len(bk1_trades) == 2

        bk2_trades = portfolio.get_trades_by_book("BK-002")
        assert len(bk2_trades) == 1

    def test_calculate_mtm_by_book(self):
        """Test MTM calculation by book."""
        portfolio = Portfolio(name="Test Portfolio")

        trade1 = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.EQUITY_OPTION,
            trade_date=date.today(),
            book_id="BK-001",
            mtm=10000.0,
        )
        trade2 = Trade(
            id="TRD-002",
            counterparty_id="CP-001",
            product_type=ProductType.FX_OPTION,
            trade_date=date.today(),
            book_id="BK-001",
            mtm=5000.0,
        )

        portfolio.add_trade(trade1)
        portfolio.add_trade(trade2)

        mtm = portfolio.calculate_mtm_by_book("BK-001")
        assert mtm == 15000.0

    def test_get_book_summary(self):
        """Test book summary statistics."""
        portfolio = Portfolio(name="Test Portfolio")

        trade1 = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.EQUITY_OPTION,
            trade_date=date.today(),
            book_id="BK-001",
            mtm=10000.0,
            notional=1000000,
        )
        trade2 = Trade(
            id="TRD-002",
            counterparty_id="CP-001",
            product_type=ProductType.FX_OPTION,
            trade_date=date.today(),
            book_id="BK-001",
            status=TradeStatus.TERMINATED,
            mtm=5000.0,
            notional=500000,
        )

        portfolio.add_trade(trade1)
        portfolio.add_trade(trade2)

        summary = portfolio.get_book_summary("BK-001")
        assert summary["num_trades"] == 2
        assert summary["active_trades"] == 1
        assert summary["total_mtm"] == 15000.0
        assert summary["total_notional"] == 1500000

    def test_aggregate_by_book(self):
        """Test portfolio aggregation by book."""
        portfolio = Portfolio(name="Test Portfolio")

        trade1 = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.EQUITY_OPTION,
            trade_date=date.today(),
            book_id="BK-001",
            mtm=10000.0,
        )
        trade2 = Trade(
            id="TRD-002",
            counterparty_id="CP-001",
            product_type=ProductType.FX_OPTION,
            trade_date=date.today(),
            book_id="BK-002",
            mtm=20000.0,
        )

        portfolio.add_trade(trade1)
        portfolio.add_trade(trade2)

        aggregation = portfolio.aggregate_by_book()
        assert "BK-001" in aggregation
        assert "BK-002" in aggregation
        assert aggregation["BK-001"]["total_mtm"] == 10000.0
        assert aggregation["BK-002"]["total_mtm"] == 20000.0


class TestLifecycleManagement:
    """Test trade lifecycle management."""

    def test_trade_amendment(self):
        """Test trade amendment."""
        manager = LifecycleManager()

        trade = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            notional=1_000_000,
        )

        # Amend notional
        amendment = TradeAmendment(
            trade_id="TRD-001", changes={"notional": 2_000_000}, reason="Client requested increase"
        )

        event = manager.amend_trade(trade, amendment)

        assert event.event_type == LifecycleEventType.AMENDMENT
        assert trade.notional == 2_000_000
        assert event.previous_values["notional"] == 1_000_000

        # Check history
        history = manager.get_trade_history("TRD-001")
        assert len(history) == 1

    def test_trade_novation(self):
        """Test trade novation."""
        manager = LifecycleManager()

        trade = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.EQUITY_OPTION,
            trade_date=date.today(),
        )

        novation = TradeNovation(
            trade_id="TRD-001", new_counterparty_id="CP-002", reason="Risk transfer"
        )

        event = manager.novate_trade(trade, novation)

        assert event.event_type == LifecycleEventType.NOVATION
        assert trade.counterparty_id == "CP-002"
        assert trade.status == TradeStatus.NOVATED
        assert event.previous_values["counterparty_id"] == "CP-001"

    def test_trade_termination(self):
        """Test trade termination."""
        manager = LifecycleManager()

        trade = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            status=TradeStatus.ACTIVE,
        )

        termination = TradeTermination(
            trade_id="TRD-001", termination_payment=50000.0, reason="Early termination"
        )

        event = manager.terminate_trade(trade, termination)

        assert event.event_type == LifecycleEventType.TERMINATION
        assert trade.status == TradeStatus.TERMINATED
        assert event.changes["termination_payment"] == 50000.0

    def test_version_tracking(self):
        """Test trade version tracking."""
        manager = LifecycleManager()

        trade = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            notional=1_000_000,
        )

        # Make multiple amendments
        amendment1 = TradeAmendment(trade_id="TRD-001", changes={"notional": 2_000_000})
        manager.amend_trade(trade, amendment1)

        amendment2 = TradeAmendment(trade_id="TRD-001", changes={"notional": 3_000_000})
        manager.amend_trade(trade, amendment2)

        # Check versions
        versions = manager.get_trade_versions("TRD-001")
        assert len(versions) >= 4  # Before amendment 1, after amendment 1, before amendment 2, after amendment 2


class TestRepository:
    """Test repository pattern."""

    def test_trade_repository(self):
        """Test trade repository operations."""
        repo = InMemoryTradeRepository()

        trade = Trade(
            id="TRD-001",
            counterparty_id="CP-001",
            product_type=ProductType.EQUITY_OPTION,
            trade_date=date.today(),
        )

        # Save and retrieve
        repo.save(trade)
        found = repo.find_by_id("TRD-001")
        assert found is not None
        assert found.id == "TRD-001"

        # Find by counterparty
        trades = repo.find_by_counterparty("CP-001")
        assert len(trades) == 1

        # Count
        assert repo.count() == 1

        # Delete
        assert repo.delete("TRD-001")
        assert repo.count() == 0

    def test_book_repository(self):
        """Test book repository operations."""
        repo = InMemoryBookRepository()

        book = Book(id="BK-001", name="USD Rates", desk_id="DSK-001")
        desk = Desk(id="DSK-001", name="Rates Desk", business_unit_id="BU-001")

        repo.save_book(book)
        repo.save_desk(desk)

        # Find book
        found_book = repo.find_book_by_id("BK-001")
        assert found_book is not None
        assert found_book.name == "USD Rates"

        # Find books by desk
        books = repo.find_books_by_desk("DSK-001")
        assert len(books) == 1

    def test_counterparty_repository(self):
        """Test counterparty repository operations."""
        from neutryx.portfolio.contracts.counterparty import EntityType

        repo = InMemoryCounterpartyRepository()

        cp = Counterparty(
            id="CP-001",
            name="Bank ABC",
            entity_type=EntityType.FINANCIAL,
            lei="549300ABCDEF12345678",
            jurisdiction="US",
        )

        repo.save(cp)

        # Find by ID
        found = repo.find_by_id("CP-001")
        assert found is not None

        # Find by LEI
        found_by_lei = repo.find_by_lei("549300ABCDEF12345678")
        assert found_by_lei is not None
        assert found_by_lei.id == "CP-001"

    def test_repository_factory(self):
        """Test repository factory."""
        trade_repo, book_repo, cp_repo = RepositoryFactory.create_in_memory_repositories()

        assert isinstance(trade_repo, InMemoryTradeRepository)
        assert isinstance(book_repo, InMemoryBookRepository)
        assert isinstance(cp_repo, InMemoryCounterpartyRepository)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
