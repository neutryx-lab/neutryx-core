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
- **RFQ Workflow** (NEW): Multi-dealer request-for-quote with competitive bidding and best execution tracking
- **Convention-Based Trade Generation** (NEW): Generate market-standard trades using currency-specific conventions
- **Confirmation Matching** (NEW): Automated matching and affirmation of trade confirmations
- **Settlement Instructions** (NEW): Generate settlement instructions and SWIFT messages with payment netting

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

## RFQ Workflow (NEW)

### Request for Quote System

The RFQ (Request for Quote) system enables multi-dealer competitive bidding with auction mechanisms:

```python
from neutryx.trading.rfq import (
    RFQ,
    RFQRequest,
    Quote,
    RFQStatus,
    AuctionType,
    RFQManager,
)
from datetime import datetime, timedelta

# Create RFQ request
rfq_request = RFQRequest(
    product_type="IRS",
    notional=10_000_000,
    currency="USD",
    maturity_years=5,
    fixed_rate=None,  # Requesting quotes
    additional_terms={"payment_frequency": "quarterly"},
)

# Initialize RFQ manager
manager = RFQManager()

# Create RFQ with multiple dealers
rfq = manager.create_rfq(
    request=rfq_request,
    dealer_ids=["DEALER-001", "DEALER-002", "DEALER-003"],
    auction_type=AuctionType.BLIND,  # or AuctionType.OPEN
    expiry=datetime.now() + timedelta(hours=2),
)

# Dealers submit quotes
quote1 = Quote(
    rfq_id=rfq.rfq_id,
    dealer_id="DEALER-001",
    price=10_250_000,
    fixed_rate=0.0425,
    spread=0.0025,
    quote_time=datetime.now(),
)

manager.submit_quote(rfq.rfq_id, quote1)

# Get best execution
best_quote = manager.get_best_quote(rfq.rfq_id)
print(f"Best quote from {best_quote.dealer_id}: {best_quote.price}")

# Accept quote
manager.accept_quote(rfq.rfq_id, best_quote.quote_id)

# Track dealer statistics
stats = manager.get_dealer_statistics("DEALER-001")
print(f"Win rate: {stats.win_rate:.2%}")
print(f"Average spread: {stats.avg_spread:.4f}")
```

### Auction Types

- **Blind Auction**: Dealers cannot see other quotes
- **Open Auction**: All quotes are visible to participants

### Best Execution Tracking

The RFQ system automatically tracks:
- Quote response times
- Dealer win rates
- Average spreads by dealer
- Historical quote quality

## Convention-Based Trade Generation (NEW)

### Market Convention Profiles

Generate trades using market-standard conventions for different currencies and products:

```python
from neutryx.portfolio.trade_generation import (
    ConventionProfile,
    TradeGenerator,
    get_convention_profile,
)
from neutryx.market.convention_profiles import CurrencyConvention

# Get USD IRS convention
usd_irs_convention = get_convention_profile(
    currency="USD",
    product="IRS",
)

print(f"Day count: {usd_irs_convention.day_count}")
print(f"Payment frequency: {usd_irs_convention.payment_frequency}")
print(f"Business day convention: {usd_irs_convention.business_day_convention}")
print(f"Calendar: {usd_irs_convention.calendar}")

# Generate trade using conventions
generator = TradeGenerator()

irs_trade = generator.generate_irs(
    currency="USD",
    notional=10_000_000,
    maturity_years=5,
    fixed_rate=0.045,
    use_conventions=True,  # Apply USD market conventions
)

# Override specific conventions if needed
custom_irs = generator.generate_irs(
    currency="USD",
    notional=10_000_000,
    maturity_years=5,
    fixed_rate=0.045,
    use_conventions=True,
    overrides={
        "payment_frequency": "monthly",  # Non-standard
        "day_count": "ACT/ACT",
    },
)

# Validate convention compliance
validation = generator.validate_conventions(custom_irs)
if not validation.is_compliant:
    print(f"Warnings: {validation.warnings}")
```

### Supported Conventions

**Currencies:**
- USD: SOFR-based conventions
- EUR: ESTR-based conventions
- GBP: SONIA-based conventions
- JPY: TONA-based conventions
- CHF: SARON-based conventions

**Products:**
- IRS (Interest Rate Swaps)
- OIS (Overnight Index Swaps)
- CCS (Cross-Currency Swaps)
- Basis Swaps (tenor and currency basis)
- FRA (Forward Rate Agreements)
- Caps/Floors

### Convention Profile Details

```python
from neutryx.market.convention_profiles import (
    get_usd_irs_convention,
    get_eur_ois_convention,
    get_gbp_basis_convention,
)

# USD IRS convention details
usd_irs = get_usd_irs_convention()
# - Day count: ACT/360
# - Payment frequency: Semi-annual (6M)
# - Business day: Modified Following
# - Calendar: US (Fed holidays)
# - Floating index: SOFR

# EUR OIS convention details
eur_ois = get_eur_ois_convention()
# - Day count: ACT/360
# - Payment frequency: Annual
# - Business day: Modified Following
# - Calendar: TARGET
# - Floating index: ESTR
```

## Confirmation Matching (NEW)

### Match Trade Confirmations

Automated confirmation matching and affirmation workflow:

```python
from neutryx.trading.confirmation import (
    Confirmation,
    ConfirmationMatcher,
    MatchStatus,
    ConfirmationType,
)

# Create internal trade confirmation
internal_conf = Confirmation(
    trade_id="TRD-001",
    confirmation_type=ConfirmationType.INTERNAL,
    product_type="IRS",
    notional=10_000_000,
    currency="USD",
    fixed_rate=0.045,
    maturity_date="2030-03-15",
    counterparty_id="CP-001",
)

# Receive counterparty confirmation
counterparty_conf = Confirmation(
    trade_id="TRD-001-CP",
    confirmation_type=ConfirmationType.EXTERNAL,
    product_type="IRS",
    notional=10_000_000,
    currency="USD",
    fixed_rate=0.045,
    maturity_date="2030-03-15",
    counterparty_id="CP-001",
)

# Match confirmations
matcher = ConfirmationMatcher(tolerance=0.01)  # 1% tolerance
match_result = matcher.match(internal_conf, counterparty_conf)

if match_result.status == MatchStatus.MATCHED:
    print("Confirmations match - ready for settlement")
elif match_result.status == MatchStatus.BREAK:
    print(f"Confirmation break: {match_result.breaks}")
    for field, values in match_result.breaks.items():
        print(f"  {field}: {values['internal']} vs {values['external']}")

# Affirm trade after matching
if match_result.status == MatchStatus.MATCHED:
    affirmed = matcher.affirm_trade("TRD-001")
    print(f"Trade affirmed: {affirmed}")
```

### Matching Rules

The confirmation matcher checks:
- Product type and key terms
- Notional amounts (with tolerance)
- Currencies and rates
- Payment dates and schedules
- Counterparty information

### Break Management

```python
# Get all unmatched trades
breaks = matcher.get_breaks()

# Resolve break manually
matcher.resolve_break(
    trade_id="TRD-001",
    resolution="Counterparty confirmed rate should be 0.046",
    resolved_by="ops-team",
)
```

## Settlement Instructions (NEW)

### Generate Settlement Instructions

Automatically generate settlement instructions for affirmed trades:

```python
from neutryx.trading.settlement import (
    SettlementInstruction,
    SettlementInstructionGenerator,
    SettlementType,
    PaymentDetails,
)

# Create settlement instruction generator
generator = SettlementInstructionGenerator()

# Generate settlement instruction
instruction = generator.generate_instruction(
    trade_id="TRD-001",
    settlement_type=SettlementType.DVP,  # Delivery vs Payment
    settlement_date="2025-03-17",
    currency="USD",
    amount=10_000_000,
    counterparty_id="CP-001",
)

print(f"Instruction ID: {instruction.instruction_id}")
print(f"Settlement date: {instruction.settlement_date}")
print(f"Currency: {instruction.currency}")
print(f"Amount: {instruction.amount}")

# Add payment details
payment_details = PaymentDetails(
    beneficiary_name="Neutryx Trading Corp",
    beneficiary_account="123456789",
    beneficiary_bank="JPMORGAN CHASE",
    swift_code="CHASUS33",
    reference="TRD-001-SETTLEMENT",
)

instruction.add_payment_details(payment_details)

# Generate SWIFT message
swift_message = generator.generate_swift_message(instruction, message_type="MT202")
print(swift_message)

# Track settlement status
generator.update_status(instruction.instruction_id, "PENDING")
generator.update_status(instruction.instruction_id, "SETTLED")
```

### Settlement Types

- **DVP**: Delivery versus Payment
- **RVP**: Receive versus Payment
- **FOP**: Free of Payment
- **PVP**: Payment versus Payment (for FX)

### Payment Netting

```python
# Enable payment netting for multiple trades
netting_result = generator.net_payments(
    trade_ids=["TRD-001", "TRD-002", "TRD-003"],
    counterparty_id="CP-001",
    settlement_date="2025-03-17",
    currency="USD",
)

print(f"Net amount: {netting_result.net_amount}")
print(f"Gross amount: {netting_result.gross_amount}")
print(f"Netting efficiency: {netting_result.efficiency:.2%}")

# Generate single settlement instruction for netted amount
netted_instruction = generator.generate_instruction(
    trade_id=netting_result.netting_set_id,
    settlement_type=SettlementType.DVP,
    settlement_date="2025-03-17",
    currency="USD",
    amount=netting_result.net_amount,
    counterparty_id="CP-001",
)
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
8. **Use RFQ for Price Discovery**: Leverage the RFQ workflow for competitive multi-dealer pricing
9. **Apply Market Conventions**: Use convention-based trade generation for market-standard trades
10. **Match Confirmations**: Always match internal and external confirmations before settlement
11. **Net Payments**: Use payment netting to reduce settlement risk and operational costs

## API Reference

For detailed API documentation, see:

- **ID Generator API**: `neutryx.portfolio.id_generator`
- **Book Hierarchy API**: `neutryx.portfolio.books`
- **Counterparty Codes API**: `neutryx.contracts.counterparty_codes`
- **Lifecycle Management API**: `neutryx.portfolio.lifecycle`
- **Pricing Bridge API**: `neutryx.portfolio.pricing_bridge`
- **Repository Pattern API**: `neutryx.portfolio.repository`
- **Portfolio API**: `neutryx.portfolio.portfolio`
- **RFQ Workflow API**: `neutryx.trading.rfq` (NEW)
- **Trade Generation API**: `neutryx.portfolio.trade_generation` (NEW)
- **Convention Profiles API**: `neutryx.market.convention_profiles` (NEW)
- **Confirmation Matching API**: `neutryx.trading.confirmation` (NEW)
- **Settlement Instructions API**: `neutryx.trading.settlement` (NEW)

## Testing

Comprehensive tests are available in `src/neutryx/tests/test_trade_management.py`.

Run tests with:
```bash
pytest src/neutryx/tests/test_trade_management.py -v
```
