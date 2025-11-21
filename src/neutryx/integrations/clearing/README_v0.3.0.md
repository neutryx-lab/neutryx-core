# CCP Integration and Settlement Systems (v0.3.0)

Complete trading infrastructure for CCP integration and settlement automation.

## 🎯 Overview

Version 0.3.0 introduces comprehensive CCP (Central Counterparty) and settlement system integration with automated workflows for:
- Intelligent CCP routing and trade submission
- Automatic settlement instruction generation
- Multi-CCP margin aggregation
- Trade lifecycle event handling
- Position and settlement reconciliation

## 📦 New Components

### 1. CCP Router (`ccp_router.py`)
Intelligent routing service that selects the optimal CCP for each trade.

**Features:**
- Multiple routing strategies (lowest margin, lowest fees, best execution, netting efficiency)
- Automatic eligibility checking across all CCPs
- Parallel margin quote requests
- Comprehensive routing decision tracking

**Usage:**
```python
from neutryx.integrations.clearing import CCPRouter, RoutingStrategy

router = CCPRouter(
    connectors={"LCH SwapClear": lch_connector, "CME Clearing": cme_connector},
    default_strategy=RoutingStrategy.LOWEST_MARGIN,
)

# Route and submit trade
decision, response = await router.route_and_submit(trade)
print(f"Selected CCP: {decision.selected_ccp}")
print(f"Rationale: {decision.rationale}")
```

**Supported CCPs:**
- **LCH SwapClear**: IRS, Swaptions (G10 currencies, 50Y max maturity)
- **CME Clearing**: IRS, FX Forwards/Swaps (Multi-asset with SPAN)
- **ICE Clear**: CDS, IRS, Swaptions (Credit & European derivatives)
- **Eurex Clearing**: IRS, Repo, Swaptions (Prisma margining)

### 2. Automatic Settlement Workflow (`settlement_workflow.py`)
Orchestrates end-to-end settlement from CCP confirmation to settlement completion.

**Features:**
- Automatic settlement instruction generation from CCP confirmations
- Settlement date calculation (T+X with holiday adjustment)
- Routing to appropriate settlement systems (CLS for FX, Euroclear for securities)
- Status tracking and retry management
- Event-driven workflow with notifications

**Usage:**
```python
from neutryx.integrations.clearing import (
    AutomaticSettlementService,
    SettlementWorkflowConfig,
)

config = SettlementWorkflowConfig(
    default_settlement_cycle=2,  # T+2
    auto_retry_enabled=True,
    cls_enabled=True,
    euroclear_enabled=True,
)

service = AutomaticSettlementService(config=config)

# Process CCP confirmation
workflow = await service.process_ccp_confirmation(
    trade=trade,
    ccp_response=ccp_response,
    buyer_party=buyer_party,
    seller_party=seller_party,
)

print(f"Settlement instruction: {workflow.instruction_id}")
print(f"Settlement date: {workflow.settlement_date}")
print(f"Status: {workflow.status}")
```

**Settlement Methods:**
- **CCP**: CCP-managed settlement
- **CLS**: Continuous Linked Settlement (FX)
- **Euroclear**: Securities settlement
- **Bilateral**: Direct bilateral settlement

### 3. Margin Aggregation Service (`margin_aggregator.py`)
Aggregates margin requirements and collateral across all CCPs.

**Features:**
- Real-time margin aggregation from multiple CCPs
- Breakdown by margin type (IM, VM, additional, liquidity, concentration)
- Collateral tracking with haircuts
- Margin deficit/surplus calculation
- Historical trending and analytics
- Automated alerting for margin deficits

**Usage:**
```python
from neutryx.integrations.clearing import MarginAggregationService

service = MarginAggregationService(
    connectors={"LCH SwapClear": lch, "CME Clearing": cme},
)

# Generate consolidated report
report = await service.generate_aggregated_report()

print(f"Total IM: ${report.total_initial_margin:,.2f}")
print(f"Total VM: ${report.total_variation_margin:,.2f}")
print(f"Total deficit: ${report.total_margin_deficit:,.2f}")
print(f"Collateral utilization: {report.collateral_utilization_pct:.1f}%")

# Check for alerts
alerts = service.get_margin_alerts()
for alert in alerts:
    print(f"[{alert['severity']}] {alert['message']}")
```

**Margin Components:**
- **Initial Margin (IM)**: Upfront margin requirement
- **Variation Margin (VM)**: Mark-to-market margin
- **Additional Margin**: Buffers and add-ons
- **Liquidity Add-on**: Liquidity risk charge
- **Concentration Add-on**: Concentration risk charge

### 4. Lifecycle Settlement Mapper (`lifecycle_settlement_mapper.py`)
Automatically updates settlement instructions when trade lifecycle events occur.

**Features:**
- Automatic settlement updates for amendments
- New settlement generation for novations
- Close-out settlement for terminations
- Partial termination handling
- Event-to-settlement linkage tracking

**Usage:**
```python
from neutryx.integrations.clearing import LifecycleSettlementMapper
from neutryx.portfolio.lifecycle import LifecycleEvent, LifecycleEventType

mapper = LifecycleSettlementMapper()

# Process amendment
amendment_event = LifecycleEvent(
    event_id="AMN-001",
    trade_id="IRS-001",
    event_type=LifecycleEventType.AMENDMENT,
    event_date=date.today(),
    effective_date=date.today(),
    description="Increase notional",
    changes={"notional": 15000000},
    previous_values={"notional": 10000000},
)

impact = await mapper.process_lifecycle_event(
    event=amendment_event,
    trade=trade,
    parties={"buyer": buyer, "seller": seller},
)

print(f"Settlement action: {impact.settlement_action}")
print(f"Affected instructions: {impact.affected_instructions}")
```

**Supported Lifecycle Events:**
- **Amendment**: Update notional, rate, maturity → Updates existing settlements
- **Novation**: Transfer to new counterparty → Generates new settlement
- **Termination**: Early termination → Generates close-out settlement
- **Partial Termination**: Partial close-out → Updates remaining + close-out

### 5. Margin Tracker (`margin_tracker.py`)
Enhanced margin tracking with historical snapshots and analytics.

**Features:**
- Historical margin snapshots
- Automatic change detection (>5% or >$1M)
- Margin call tracking and satisfaction
- Response time analytics
- Overdue margin call detection
- Trend analysis

**Usage:**
```python
from neutryx.integrations.clearing import CCPMarginTracker, MarginTrackerConfig

config = MarginTrackerConfig(
    snapshot_interval_hours=6,
    retain_snapshots_days=90,
    margin_call_alert_enabled=True,
)

tracker = CCPMarginTracker(
    ccp_name="LCH SwapClear",
    member_id="MEMBER123",
    config=config,
)

# Record margin snapshot
margin_data = await connector.get_margin_requirements()
snapshot = await tracker.record_margin_snapshot(margin_data)

# Track margin call
margin_call = MarginCall(...)
await tracker.record_margin_call(margin_call)

# Later, mark as satisfied
await tracker.satisfy_margin_call(
    call_id=margin_call.call_id,
    collateral_posted=[{"type": "cash", "amount": 1000000}],
)

# Get trends
trends = tracker.get_margin_trends(days=30)
```

### 6. Reconciliation Engine (`reconciliation.py`)
Reconciles internal positions and settlements with CCP reports.

**Features:**
- Position reconciliation (internal vs CCP)
- Settlement reconciliation (instructed vs settled)
- Margin reconciliation (calculated vs reported)
- Break detection with severity classification
- Resolution workflow and tracking
- Reconciliation statistics and reporting

**Usage:**
```python
from neutryx.integrations.clearing import CCPReconciliationEngine

engine = CCPReconciliationEngine(connectors=connectors)

# Reconcile positions
internal_positions = [
    {"trade_id": "IRS-001", "notional": 10000000, "fixed_rate": 0.045},
    {"trade_id": "IRS-002", "notional": 5000000, "fixed_rate": 0.038},
]

result = await engine.reconcile_positions(
    ccp_name="LCH SwapClear",
    internal_positions=internal_positions,
)

print(f"Matched: {result.matched_records}")
print(f"Breaks: {result.total_breaks}")
print(f"Critical: {result.critical_breaks}")

# Get open breaks
breaks = engine.get_open_breaks(severity=BreakSeverity.CRITICAL)
for break_item in breaks:
    print(f"[{break_item.severity}] {break_item.description}")

# Resolve break
engine.resolve_break(
    break_id=breaks[0].break_id,
    resolution_notes="Corrected internal notional",
    resolved_by="TraderJohn",
)
```

**Break Types:**
- **MISSING_INTERNAL**: Trade exists in CCP but not internally (Critical)
- **MISSING_CCP**: Trade exists internally but not in CCP (High/Critical)
- **QUANTITY_MISMATCH**: Different notionals (Medium/High)
- **PRICE_MISMATCH**: Different rates/prices (Medium)
- **STATUS_MISMATCH**: Different trade statuses (Low)
- **MARGIN_MISMATCH**: Margin calculation differences (Medium/Critical)

## 🔄 End-to-End Workflow

```python
from neutryx.integrations.clearing import (
    CCPRouter,
    AutomaticSettlementService,
    MarginAggregationService,
    LifecycleSettlementMapper,
    CCPReconciliationEngine,
)

# 1. Route trade to optimal CCP
router = CCPRouter(connectors=connectors)
decision, response = await router.route_and_submit(trade)

# 2. Generate settlement instruction automatically
settlement_service = AutomaticSettlementService()
workflow = await settlement_service.process_ccp_confirmation(
    trade=trade,
    ccp_response=response,
    buyer_party=buyer,
    seller_party=seller,
)

# 3. Update margin aggregation
margin_service = MarginAggregationService(connectors=connectors)
report = await margin_service.generate_aggregated_report()

# 4. Process lifecycle event (if any)
mapper = LifecycleSettlementMapper()
if lifecycle_event:
    impact = await mapper.process_lifecycle_event(
        event=lifecycle_event,
        trade=trade,
        parties=parties,
    )

# 5. Reconcile positions
recon_engine = CCPReconciliationEngine(connectors=connectors)
recon_result = await recon_engine.reconcile_positions(
    ccp_name=decision.selected_ccp,
    internal_positions=internal_positions,
)
```

## 📊 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trade Generation                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CCP Router                               │
│  • Eligibility check                                             │
│  • Margin quotes (parallel)                                      │
│  • Routing decision (LOWEST_MARGIN, BEST_EXECUTION, etc.)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CCP Connectors (LCH/CME/ICE/Eurex)            │
│  • Trade submission                                              │
│  • Status tracking                                               │
│  • Margin requirements                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               Automatic Settlement Service                       │
│  • Calculate settlement date (T+X)                               │
│  • Generate settlement instruction                               │
│  • Route to settlement system (CLS/Euroclear/CCP)               │
│  • Track workflow status                                         │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Margin Aggregation Service                      │
│  • Fetch margins from all CCPs (parallel)                        │
│  • Aggregate by component (IM/VM/Additional)                     │
│  • Track collateral and deficits                                 │
│  • Generate alerts                                               │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           Lifecycle Settlement Mapper (if event)                 │
│  • Amendment → Update settlements                                │
│  • Novation → New settlement with new CP                         │
│  • Termination → Close-out settlement                            │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Reconciliation Engine                           │
│  • Position reconciliation (internal vs CCP)                     │
│  • Break detection and classification                            │
│  • Resolution workflow                                           │
└─────────────────────────────────────────────────────────────────┘
```

## 🧪 Testing

Comprehensive integration tests available in:
```
tests/integrations/clearing/test_ccp_integration_workflow.py
```

Run tests:
```bash
pytest tests/integrations/clearing/test_ccp_integration_workflow.py -v
```

Test coverage includes:
- CCP routing with multiple strategies
- Automatic settlement workflow
- Margin aggregation across CCPs
- Lifecycle event processing
- Position reconciliation
- Complete end-to-end workflow

## 📝 Configuration Examples

### CCP Router Configuration
```python
from neutryx.integrations.clearing import CCPRouter, CCPEligibilityRule

# Custom eligibility rules
rules = [
    CCPEligibilityRule(
        ccp_name="LCH SwapClear",
        clearable_products={ProductType.IRS, ProductType.SWAPTION},
        supported_currencies={"USD", "EUR", "GBP", "JPY"},
        min_notional=Decimal("1000000"),
        max_maturity_days=365 * 50,
    ),
]

router = CCPRouter(
    connectors=connectors,
    eligibility_rules=rules,
    default_strategy=RoutingStrategy.LOWEST_MARGIN,
)
```

### Settlement Workflow Configuration
```python
config = SettlementWorkflowConfig(
    default_settlement_cycle=2,
    settlement_cycles={"IRS": 2, "CDS": 1, "REPO": 0},
    auto_retry_enabled=True,
    max_retry_attempts=3,
    cls_enabled=True,
    cls_eligible_currencies={"USD", "EUR", "GBP", "JPY"},
    euroclear_enabled=True,
)
```

### Margin Aggregator Configuration
```python
config = MarginAggregatorConfig(
    reporting_currency="USD",
    consolidation_frequency_hours=6,
    margin_deficit_threshold=Decimal("1000000"),
    collateral_utilization_warning=80.0,
    margin_coverage_warning=1.1,
    auto_refresh_enabled=True,
)
```

### Reconciliation Configuration
```python
config = ReconciliationConfig(
    quantity_tolerance=Decimal("0.01"),
    price_tolerance=Decimal("0.0001"),
    margin_tolerance=Decimal("100"),
    auto_resolve_within_tolerance=True,
    auto_reconcile_enabled=True,
    reconcile_position_daily=True,
    notify_on_critical=True,
)
```

## 🎯 Key Features Summary

### CCP Integration
✅ Intelligent routing across LCH, CME, ICE, Eurex
✅ Automatic eligibility checking
✅ Parallel margin quote requests
✅ Multiple routing strategies
✅ Comprehensive decision tracking

### Settlement Automation
✅ Automatic instruction generation from CCP confirmations
✅ Smart settlement date calculation (T+X with holidays)
✅ CLS routing for FX trades
✅ Euroclear routing for securities
✅ Workflow status tracking and retry

### Margin Management
✅ Multi-CCP margin aggregation
✅ Real-time margin tracking with history
✅ Automated margin call tracking
✅ Deficit/surplus monitoring
✅ Collateral optimization suggestions

### Lifecycle Integration
✅ Automatic settlement updates for amendments
✅ New settlements for novations
✅ Close-out settlements for terminations
✅ Event-to-settlement linkage
✅ Full audit trail

### Reconciliation
✅ Position reconciliation (internal vs CCP)
✅ Settlement reconciliation
✅ Break detection with severity
✅ Resolution workflow
✅ Comprehensive reporting

## 📚 Documentation

- [CCP Base Classes](./base.py)
- [LCH Integration](./lch.py)
- [CME Integration](./cme.py)
- [ICE Integration](./ice.py)
- [Eurex Integration](./eurex.py)
- [Settlement Instructions](./settlement_instructions.py)

## 🚀 Next Steps

Future enhancements (v0.4.0+):
- [ ] Real-time CCP event streaming
- [ ] Machine learning for routing optimization
- [ ] Predictive margin forecasting
- [ ] Automated collateral optimization
- [ ] Advanced compression workflows
- [ ] Cross-CCP netting analysis
- [ ] Regulatory reporting integration (EMIR, Dodd-Frank)

## 📄 License

Copyright © 2025 Neutryx. All rights reserved.
