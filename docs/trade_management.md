# Trade Management System

Neutryx provides a comprehensive trade management system for organizing and managing derivatives portfolios with enterprise-grade features.

## Overview

The trade management system includes:

- **ID Generation**: Systematic generation of trade numbers, counterparty codes, and entity IDs
- **Book Hierarchy**: Organizational structure (Legal Entity → Business Unit → Desk → Book → Trader)
- **Trade Assignment**: Assign trades to books, desks, and traders
- **Lifecycle Management**: Track amendments, novations, terminations with full version history
- **Portfolio Aggregation**: Book-level, desk-level, and trader-level aggregation
- **Pricing Bridge**: Connect trades to pricing engines
- **Repository Pattern**: Abstract persistence layer for trades and entities

## ID Generation

### Trade ID Generation

Generate systematic trade identifiers with configurable patterns:

```python
from neutryx.portfolio.id_generator import create_trade_id_generator, IDPattern
from datetime import date

# Date-based sequential IDs (default)
generator = create_trade_id_generator()
trade_id = generator.generate()  # TRD-20250315-0001
trade_id = generator.generate()  # TRD-20250315-0002

# Simple sequential IDs
from neutryx.portfolio.id_generator import IDPattern
generator = create_trade_id_generator(pattern=IDPattern.SEQUENTIAL)
trade_id = generator.generate()  # TRD-0001

# UUID-based IDs for distributed systems
generator = create_trade_id_generator(pattern=IDPattern.UUID)
trade_id = generator.generate()  # TRD-550e8400-e29b-41d4-a716-446655440000

# Custom patterns
from neutryx.portfolio.id_generator import IDGeneratorConfig, TradeIDGenerator

config = IDGeneratorConfig(
    pattern=IDPattern.CUSTOM,
    custom_pattern="{prefix}_{date}_{seq:06d}",
    prefix="TRADE",
)
generator = TradeIDGenerator(config)
trade_id = generator.generate()  # TRADE_20250315_000001
```

### Counterparty Code Generation

Generate systematic counterparty codes:

```python
from neutryx.contracts.counterparty_codes import (
    create_simple_counterparty_code_generator,
    create_lei_based_counterparty_code_generator,
    create_typed_counterparty_code_generator,
    CounterpartyType,
)

# Simple sequential codes
generator = create_simple_counterparty_code_generator()
code = generator.generate("CPTY-001")  # CP-0001
code = generator.generate("CPTY-002")  # CP-0002

# LEI-based codes
generator = create_lei_based_counterparty_code_generator()
code = generator.generate_from_lei(
    lei="549300ABCDEF12345678",
    counterparty_id="CPTY-001",
    name="Bank ABC"
)  # CP-549300-0001

# Type-based codes
generator = create_typed_counterparty_code_generator()
code = generator.generate(
    counterparty_id="CPTY-001",
    counterparty_type=CounterpartyType.BANK
)  # CP-BNK-0001

# Lookup functionality
mapping = generator.get_mapping(code)
print(f"Counterparty: {mapping.name}, LEI: {mapping.lei}")
```

## Book Hierarchy

### Creating Organizational Structure

Build a complete organizational hierarchy:

```python
from neutryx.portfolio.books import (
    LegalEntity,
    BusinessUnit,
    Desk,
    Book,
    Trader,
    BookHierarchy,
    EntityStatus,
)

# Create hierarchy
hierarchy = BookHierarchy()

# Legal Entity
le = LegalEntity(
    id="LE-001",
    name="Neutryx Trading Corp",
    lei="549300ABCDEF12345678",
    jurisdiction="US",
)
hierarchy.add_legal_entity(le)

# Business Unit
bu = BusinessUnit(
    id="BU-001",
    name="Fixed Income Trading",
    legal_entity_id="LE-001",
)
hierarchy.add_business_unit(bu)

# Trading Desk
desk = Desk(
    id="DSK-001",
    name="USD Rates Desk",
    business_unit_id="BU-001",
    desk_type="rates",
)
hierarchy.add_desk(desk)

# Trading Book
book = Book(
    id="BK-001",
    name="USD IRS Flow",
    desk_id="DSK-001",
    book_type="flow",
)
hierarchy.add_book(book)

# Trader
trader = Trader(
    id="TRD-001",
    name="John Doe",
    email="john.doe@neutryx.com",
    desk_id="DSK-001",
)
trader.add_book("BK-001")
trader.grant_permission("trade_entry")
trader.grant_permission("price_approval")
hierarchy.add_trader(trader)

# Get full hierarchy path for a book
path = hierarchy.get_book_path("BK-001")
print(f"Legal Entity: {path['legal_entity_id']}")
print(f"Business Unit: {path['business_unit_id']}")
print(f"Desk: {path['desk_id']}")
print(f"Book: {path['book_id']}")

# Validate book assignments
is_valid = hierarchy.validate_book_assignment("BK-001", "TRD-001")
```

### Book Management

Manage trading books with risk limits:

```python
# Set risk limits
book.set_risk_limit("dv01", 1_000_000.0)
book.set_risk_limit("var_95", 500_000.0)
book.set_risk_limit("notional", 100_000_000.0)

# Add trades to book
book.add_trade("TRD-001")
book.add_trade("TRD-002")

# Check limits
dv01_limit = book.get_risk_limit("dv01")
trade_count = book.get_trade_count()
```

## Trade Assignment

### Creating Trades with Book Information

Assign trades to books, desks, and traders:

```python
from neutryx.contracts.trade import Trade, ProductType, TradeStatus
from datetime import date

trade = Trade(
    id="TRD-001",
    trade_number="TRD-20250315-0001",  # Systematic trade number
    counterparty_id="CP-001",
    product_type=ProductType.INTEREST_RATE_SWAP,
    trade_date=date.today(),
    book_id="BK-001",     # Book assignment
    desk_id="DSK-001",     # Desk assignment
    trader_id="TRD-001",   # Trader assignment
    notional=10_000_000,
    currency="USD",
    status=TradeStatus.ACTIVE,
)
```

## Portfolio Aggregation

### Book-Level Aggregation

Aggregate portfolio metrics by book:

```python
from neutryx.portfolio.portfolio import Portfolio

portfolio = Portfolio(name="Global Trading Portfolio")

# Add trades...
portfolio.add_trade(trade1)
portfolio.add_trade(trade2)

# Get trades by book
book_trades = portfolio.get_trades_by_book("BK-001")
desk_trades = portfolio.get_trades_by_desk("DSK-001")
trader_trades = portfolio.get_trades_by_trader("TRD-001")

# Calculate MTM by book
book_mtm = portfolio.calculate_mtm_by_book("BK-001")
desk_mtm = portfolio.calculate_mtm_by_desk("DSK-001")

# Get book summary
summary = portfolio.get_book_summary("BK-001")
print(f"Book: {summary['book_id']}")
print(f"Trades: {summary['num_trades']}")
print(f"Active: {summary['active_trades']}")
print(f"MTM: ${summary['total_mtm']:,.2f}")
print(f"Notional: ${summary['total_notional']:,.2f}")

# Aggregate across all books
book_aggregation = portfolio.aggregate_by_book()
for book_id, book_summary in book_aggregation.items():
    print(f"{book_id}: MTM=${book_summary['total_mtm']:,.2f}")

# Desk-level aggregation
desk_aggregation = portfolio.aggregate_by_desk()

# Trader-level aggregation
trader_aggregation = portfolio.aggregate_by_trader()
```

## Lifecycle Management

### Trade Amendments

Track and apply trade amendments with full history:

```python
from neutryx.portfolio.lifecycle import (
    LifecycleManager,
    TradeAmendment,
    TradeNovation,
    TradeTermination,
)

manager = LifecycleManager()

# Amend notional
amendment = TradeAmendment(
    trade_id="TRD-001",
    changes={"notional": 20_000_000},
    reason="Client requested notional increase",
    amended_by="TRD-001",
)

event = manager.amend_trade(trade, amendment)
print(f"Event ID: {event.event_id}")
print(f"Previous notional: ${event.previous_values['notional']:,.0f}")
print(f"New notional: ${trade.notional:,.0f}")

# View trade history
history = manager.get_trade_history("TRD-001")
for event in history:
    print(f"{event.event_date}: {event.event_type.value} - {event.description}")

# View version history
versions = manager.get_trade_versions("TRD-001")
for version in versions:
    print(f"Version {version.version_number}: {version.comment}")
```

### Trade Novations

Transfer trades to new counterparties:

```python
novation = TradeNovation(
    trade_id="TRD-001",
    new_counterparty_id="CP-002",
    new_netting_set_id="NS-002",
    reason="Risk transfer to affiliate",
    novated_by="TRD-001",
)

event = manager.novate_trade(trade, novation)
assert trade.counterparty_id == "CP-002"
assert trade.status == TradeStatus.NOVATED
```

### Trade Terminations

Terminate trades with termination payments:

```python
termination = TradeTermination(
    trade_id="TRD-001",
    termination_payment=50_000.0,
    reason="Early termination at client request",
    terminated_by="TRD-001",
)

event = manager.terminate_trade(trade, termination)
assert trade.status == TradeStatus.TERMINATED
```

## Pricing Bridge

### Connect Trades to Pricing Engines

Extract pricing parameters and price trades:

```python
from neutryx.portfolio.pricing_bridge import PricingBridge, MarketData
from datetime import date

bridge = PricingBridge(seed=42)

# Create market data snapshot
market_data = MarketData(
    pricing_date=date.today(),
    spot_prices={"AAPL": 150.0, "GOOGL": 140.0},
    volatilities={"AAPL": 0.25, "GOOGL": 0.28},
    interest_rates={"USD": 0.05, "EUR": 0.03},
    dividend_yields={"AAPL": 0.01, "GOOGL": 0.0},
)

# Price a single trade
result = bridge.price_trade(trade, market_data)
if result.success:
    print(f"Trade {result.trade_id}: ${result.price:,.2f}")
else:
    print(f"Pricing failed: {result.error_message}")

# Price entire portfolio
results = bridge.price_portfolio(portfolio.get_active_trades(), market_data, update_mtm=True)

successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"Successfully priced: {len(successful)} trades")
print(f"Failed: {len(failed)} trades")

# Extract pricing parameters from a trade
params = bridge.extract_pricing_parameters(trade)
```

## Repository Pattern

### Persist and Retrieve Entities

Use repositories for data persistence:

```python
from neutryx.portfolio.repository import RepositoryFactory

# Create repositories
trade_repo, book_repo, cp_repo = RepositoryFactory.create_in_memory_repositories()

# Trade repository
trade_repo.save(trade)
found_trade = trade_repo.find_by_id("TRD-001")
book_trades = trade_repo.find_by_book("BK-001")
desk_trades = trade_repo.find_by_desk("DSK-001")
active_trades = trade_repo.find_by_status(TradeStatus.ACTIVE)

# Book repository
book_repo.save_book(book)
book_repo.save_desk(desk)
book_repo.save_trader(trader)

found_book = book_repo.find_book_by_id("BK-001")
desk_books = book_repo.find_books_by_desk("DSK-001")

# Counterparty repository
cp_repo.save(counterparty)
found_cp = cp_repo.find_by_id("CP-001")
found_by_lei = cp_repo.find_by_lei("549300ABCDEF12345678")
```

## Complete Workflow Example

Here's a complete example combining all features:

```python
from datetime import date
from neutryx.portfolio.id_generator import create_trade_id_generator, create_book_id_generator
from neutryx.contracts.counterparty_codes import create_simple_counterparty_code_generator
from neutryx.portfolio.books import BookHierarchy, LegalEntity, BusinessUnit, Desk, Book, Trader
from neutryx.contracts.trade import Trade, ProductType
from neutryx.portfolio.portfolio import Portfolio
from neutryx.portfolio.lifecycle import LifecycleManager, TradeAmendment
from neutryx.portfolio.pricing_bridge import PricingBridge, MarketData
from neutryx.portfolio.repository import RepositoryFactory

# Setup ID generators
trade_id_gen = create_trade_id_generator()
book_id_gen = create_book_id_generator()
cp_code_gen = create_simple_counterparty_code_generator()

# Setup hierarchy
hierarchy = BookHierarchy()
le = LegalEntity(id="LE-001", name="Neutryx Corp")
bu = BusinessUnit(id="BU-001", name="Trading", legal_entity_id="LE-001")
desk = Desk(id="DSK-001", name="Rates Desk", business_unit_id="BU-001")
book = Book(id="BK-001", name="USD Rates", desk_id="DSK-001")
trader = Trader(id="TRD-001", name="John Doe", desk_id="DSK-001")

hierarchy.add_legal_entity(le)
hierarchy.add_business_unit(bu)
hierarchy.add_desk(desk)
hierarchy.add_book(book)
hierarchy.add_trader(trader)

# Create trades
trade_number = trade_id_gen.generate()
trade = Trade(
    id=trade_number,
    trade_number=trade_number,
    counterparty_id="CP-001",
    product_type=ProductType.EQUITY_OPTION,
    trade_date=date.today(),
    maturity_date=date(2026, 3, 15),
    book_id="BK-001",
    desk_id="DSK-001",
    trader_id="TRD-001",
    notional=1_000_000,
    currency="USD",
    product_details={
        "underlying": "AAPL",
        "strike": 155.0,
        "is_call": True,
    },
)

# Add to portfolio
portfolio = Portfolio(name="Main Portfolio")
portfolio.add_trade(trade)

# Price the portfolio
bridge = PricingBridge()
market_data = MarketData(
    pricing_date=date.today(),
    spot_prices={"AAPL": 150.0},
    volatilities={"AAPL": 0.25},
    interest_rates={"USD": 0.05},
    dividend_yields={"AAPL": 0.01},
)

results = bridge.price_portfolio([trade], market_data, update_mtm=True)

# Amend trade
lifecycle_mgr = LifecycleManager()
amendment = TradeAmendment(
    trade_id=trade.id,
    changes={"notional": 2_000_000},
    reason="Increased position size",
)
lifecycle_mgr.amend_trade(trade, amendment)

# Get book summary
summary = portfolio.get_book_summary("BK-001")
print(f"Book MTM: ${summary['total_mtm']:,.2f}")
print(f"Book Notional: ${summary['total_notional']:,.2f}")

# Persist to repository
trade_repo, book_repo, cp_repo = RepositoryFactory.create_in_memory_repositories()
trade_repo.save(trade)
book_repo.save_book(book)
```

## Best Practices

1. **Use Systematic IDs**: Always generate trade numbers and counterparty codes using the ID generators
2. **Validate Assignments**: Use `BookHierarchy.validate_book_assignment()` before assigning trades
3. **Track Lifecycle**: Use `LifecycleManager` for all trade modifications to maintain audit trail
4. **Set Risk Limits**: Define risk limits on books and monitor against them
5. **Aggregate Regularly**: Use portfolio aggregation methods for risk reporting
6. **Version Control**: Leverage version tracking for trade history and compliance
7. **Persist Data**: Use repositories to abstract data persistence

## API Reference

For detailed API documentation, see:

- [ID Generator API](../src/neutryx/portfolio/id_generator.py)
- [Book Hierarchy API](../src/neutryx/portfolio/books.py)
- [Counterparty Codes API](../src/neutryx/contracts/counterparty_codes.py)
- [Lifecycle Management API](../src/neutryx/portfolio/lifecycle.py)
- [Pricing Bridge API](../src/neutryx/portfolio/pricing_bridge.py)
- [Repository Pattern API](../src/neutryx/portfolio/repository.py)
- [Portfolio API](../src/neutryx/portfolio/portfolio.py)

## Testing

Comprehensive tests are available in [test_trade_management.py](../src/neutryx/tests/test_trade_management.py).

Run tests with:
```bash
pytest src/neutryx/tests/test_trade_management.py -v
```
