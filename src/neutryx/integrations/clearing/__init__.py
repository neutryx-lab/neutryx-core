"""Central Counterparty (CCP) clearing house integrations.

This package provides connectivity to major global clearing houses for
derivatives and securities clearing:

- **LCH SwapClear**: Leading IRS clearing (London Clearing House)
- **CME Clearing**: Multi-asset clearing with SPAN margining
- **ICE Clear**: Credit and European derivatives (Credit/Europe/US/Singapore)
- **Eurex Clearing**: European derivatives with Prisma margining

All integrations follow a common interface defined in the base module and
support:
- Trade submission and confirmation
- Real-time margin calculations
- Position reporting
- Trade lifecycle management
- Risk analytics

Quick Start
-----------

LCH SwapClear Example:

    >>> from neutryx.integrations.clearing import (
    ...     LCHSwapClearConnector,
    ...     LCHSwapClearConfig,
    ...     Trade,
    ...     TradeEconomics,
    ...     Party,
    ...     ProductType,
    ... )
    >>> from datetime import datetime, timedelta
    >>> from decimal import Decimal
    >>>
    >>> # Configure LCH connection
    >>> config = LCHSwapClearConfig(
    ...     ccp_name="LCH",
    ...     member_id="MEMBER123",
    ...     api_endpoint="https://api.lch.com",
    ...     api_key="your_api_key",
    ...     api_secret="your_api_secret",
    ... )
    >>>
    >>> # Create connector
    >>> connector = LCHSwapClearConnector(config)
    >>>
    >>> # Connect
    >>> await connector.connect()
    >>>
    >>> # Submit trade
    >>> trade = Trade(
    ...     trade_id="TRADE001",
    ...     product_type=ProductType.IRS,
    ...     trade_date=datetime.utcnow(),
    ...     effective_date=datetime.utcnow() + timedelta(days=2),
    ...     maturity_date=datetime.utcnow() + timedelta(days=365*5),
    ...     buyer=Party(party_id="BUYER1", name="Buyer Corp", lei="123456"),
    ...     seller=Party(party_id="SELLER1", name="Seller Corp", lei="789012"),
    ...     economics=TradeEconomics(
    ...         notional=Decimal("10000000"),
    ...         currency="USD",
    ...         fixed_rate=Decimal("0.025"),
    ...     ),
    ... )
    >>>
    >>> response = await connector.submit_trade(trade)
    >>> print(response.status)  # TradeStatus.ACCEPTED
    >>>
    >>> # Get margin requirements
    >>> margin = await connector.get_margin_requirements()
    >>> print(margin["initial_margin"])
    >>>
    >>> # Disconnect
    >>> await connector.disconnect()

CME Clearing Example:

    >>> from neutryx.integrations.clearing import (
    ...     CMEClearingConnector,
    ...     CMEClearingConfig,
    ... )
    >>>
    >>> config = CMEClearingConfig(
    ...     ccp_name="CME",
    ...     member_id="MEMBER456",
    ...     clearing_firm_id="FIRM123",
    ...     api_endpoint="https://api.cmegroup.com/clearing",
    ...     api_key="your_api_key",
    ...     span_enabled=True,
    ...     core_enabled=True,
    ... )
    >>>
    >>> connector = CMEClearingConnector(config)
    >>> await connector.connect()
    >>> response = await connector.submit_trade(trade)

Common Patterns
--------------

Error Handling:

    >>> from neutryx.integrations.clearing import (
    ...     CCPConnectionError,
    ...     CCPTradeRejectionError,
    ...     CCPTimeoutError,
    ... )
    >>>
    >>> try:
    ...     response = await connector.submit_trade(trade)
    ... except CCPTradeRejectionError as e:
    ...     print(f"Trade rejected: {e.rejection_code} - {e}")
    ... except CCPConnectionError as e:
    ...     print(f"Connection failed: {e}")
    ... except CCPTimeoutError as e:
    ...     print(f"Request timed out: {e}")

Health Monitoring:

    >>> # Check connection health
    >>> healthy = await connector.healthcheck()
    >>> if not healthy:
    ...     await connector.connect()
    >>>
    >>> # Get metrics
    >>> metrics = connector.metrics
    >>> print(f"Success rate: {metrics.success_rate():.2%}")
    >>> print(f"Avg response time: {metrics.avg_response_time_ms:.0f}ms")

Position Reporting:

    >>> from datetime import datetime
    >>>
    >>> # Get today's positions
    >>> report = await connector.get_position_report()
    >>> print(f"Total exposure: {report.total_exposure}")
    >>> print(f"Initial margin: {report.initial_margin}")
    >>>
    >>> # Get historical positions
    >>> report = await connector.get_position_report(
    ...     as_of_date=datetime(2024, 1, 1)
    ... )

Architecture
-----------

All CCP connectors inherit from `CCPConnector` base class and implement:

- `connect()`: Establish connection and authenticate
- `disconnect()`: Clean shutdown
- `submit_trade()`: Submit trade for clearing
- `get_trade_status()`: Query trade status
- `cancel_trade()`: Cancel pending trade
- `get_margin_requirements()`: Fetch margin requirements
- `get_position_report()`: Get position report
- `healthcheck()`: Connection health check

Each connector maintains:

- Connection state and session management
- Metrics tracking (success rates, response times)
- CCP-specific protocol handling
- Error handling and retry logic

Module Structure
---------------

- `base`: Base classes, protocols, and common types
- `lch`: LCH SwapClear integration
- `cme`: CME Clearing integration
- `ice`: ICE Clear Credit/Europe integration
- `eurex`: Eurex Clearing integration

Notes
-----

- All dates/times use UTC
- All amounts use Decimal for precision
- Connections support async/await patterns
- Each CCP has specific configuration requirements
- Sandbox/test environments available via config
"""

from __future__ import annotations

from .base import (
    CCPAuthenticationError,
    CCPConfig,
    CCPConnectionError,
    CCPConnector,
    CCPError,
    CCPMetrics,
    CCPTimeoutError,
    CCPTradeRejectionError,
    MarginCall,
    MessageType,
    Party,
    PositionReport,
    ProductType,
    Trade,
    TradeEconomics,
    TradeStatus,
    TradeSubmissionResponse,
)
from .cme import CMEClearingConfig, CMEClearingConnector, CMECOREAnalytics, CMESPANMargin
from .confirmation import AffirmationMethod
from .eurex import (
    EurexAssetClass,
    EurexClearingConfig,
    EurexClearingConnector,
    PrismaMarginBreakdown,
)
from .ice import (
    ICEClearConfig,
    ICEClearConnector,
    ICEClearService,
    ICECreditProduct,
    ICEMarginBreakdown,
)
from .lch import LCHSwapClearConfig, LCHSwapClearConnector, LCHTradeDetails
from .settlement_instructions import SettlementMethod, SettlementStatus, SettlementType
from .rfq import OrderSide
from .workflow import (
    PostTradeProcessingService,
    RFQWorkflowError,
    RFQWorkflowService,
    RFQWorkflowState,
    WorkflowSnapshot,
)

# New v0.3.0 components
from .ccp_router import (
    CCPRouter,
    RoutingStrategy,
    RoutingDecision,
    MarginQuote,
    CCPEligibilityRule,
    CCPCapability,
)
from .rfq_ccp_integration import (
    RFQCCPIntegrationService,
    RFQCCPIntegrationConfig,
    RFQExecutionResult,
    RFQExecutionStatus,
    TradeAllocation,
)
from .settlement_workflow import (
    AutomaticSettlementService,
    SettlementWorkflow,
    SettlementWorkflowConfig,
    SettlementWorkflowEvent,
    SettlementRoutingStrategy,
    WorkflowStatus,
)
from .margin_aggregator import (
    MarginAggregationService,
    MarginAggregatorConfig,
    AggregatedMarginReport,
    CCPMarginRequirement,
    CollateralPosition,
    MarginType,
    CollateralType,
    MarginCallStatus,
)
from .lifecycle_settlement_mapper import (
    LifecycleSettlementMapper,
    LifecycleSettlementConfig,
    LifecycleSettlementImpact,
    SettlementAction,
)
from .margin_tracker import (
    CCPMarginTracker,
    MarginTrackerConfig,
    MarginSnapshot,
    MarginChangeEvent,
    MarginCallRecord,
    MarginChangeType,
)
from .reconciliation import (
    CCPReconciliationEngine,
    ReconciliationConfig,
    ReconciliationResult,
    ReconciliationBreak,
    ReconciliationType,
    BreakType,
    BreakSeverity,
    BreakStatus,
)

__version__ = "0.1.0"

__all__ = [
    # Base classes and common types
    "CCPConnector",
    "CCPConfig",
    "CCPError",
    "CCPConnectionError",
    "CCPAuthenticationError",
    "CCPTradeRejectionError",
    "CCPTimeoutError",
    "CCPMetrics",
    "Trade",
    "TradeEconomics",
    "TradeSubmissionResponse",
    "TradeStatus",
    "Party",
    "MarginCall",
    "PositionReport",
    "MessageType",
    "ProductType",
    "AffirmationMethod",
    "OrderSide",
    "PostTradeProcessingService",
    "RFQWorkflowError",
    "RFQWorkflowService",
    "RFQWorkflowState",
    "WorkflowSnapshot",
    "SettlementMethod",
    "SettlementStatus",
    "SettlementType",
    # LCH SwapClear
    "LCHSwapClearConnector",
    "LCHSwapClearConfig",
    "LCHTradeDetails",
    # CME Clearing
    "CMEClearingConnector",
    "CMEClearingConfig",
    "CMESPANMargin",
    "CMECOREAnalytics",
    # ICE Clear
    "ICEClearConnector",
    "ICEClearConfig",
    "ICEClearService",
    "ICECreditProduct",
    "ICEMarginBreakdown",
    # Eurex Clearing
    "EurexClearingConnector",
    "EurexClearingConfig",
    "EurexAssetClass",
    "PrismaMarginBreakdown",
    # CCP Routing (v0.3.0)
    "CCPRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "MarginQuote",
    "CCPEligibilityRule",
    "CCPCapability",
    # RFQ-CCP Integration (v0.3.0)
    "RFQCCPIntegrationService",
    "RFQCCPIntegrationConfig",
    "RFQExecutionResult",
    "RFQExecutionStatus",
    "TradeAllocation",
    # Settlement Workflow (v0.3.0)
    "AutomaticSettlementService",
    "SettlementWorkflow",
    "SettlementWorkflowConfig",
    "SettlementWorkflowEvent",
    "SettlementRoutingStrategy",
    "WorkflowStatus",
    # Margin Aggregation (v0.3.0)
    "MarginAggregationService",
    "MarginAggregatorConfig",
    "AggregatedMarginReport",
    "CCPMarginRequirement",
    "CollateralPosition",
    "MarginType",
    "CollateralType",
    "MarginCallStatus",
    # Lifecycle Settlement Mapping (v0.3.0)
    "LifecycleSettlementMapper",
    "LifecycleSettlementConfig",
    "LifecycleSettlementImpact",
    "SettlementAction",
    # Margin Tracking (v0.3.0)
    "CCPMarginTracker",
    "MarginTrackerConfig",
    "MarginSnapshot",
    "MarginChangeEvent",
    "MarginCallRecord",
    "MarginChangeType",
    # Reconciliation (v0.3.0)
    "CCPReconciliationEngine",
    "ReconciliationConfig",
    "ReconciliationResult",
    "ReconciliationBreak",
    "ReconciliationType",
    "BreakType",
    "BreakSeverity",
    "BreakStatus",
]
