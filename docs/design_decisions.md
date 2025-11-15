# Design Decisions

This document captures the key architectural and implementation decisions that shape Neutryx Core. Each decision includes the context, rationale, and trade-offs considered.

## Table of Contents

1. [Core Technology Choices](#core-technology-choices)
2. [Architectural Patterns](#architectural-patterns)
3. [Performance & Scalability](#performance-scalability)
4. [Security & Governance](#security-governance)
5. [Data Management](#data-management)
6. [Regulatory Compliance](#regulatory-compliance)
7. [API Design](#api-design)
8. [Testing Strategy](#testing-strategy)

---

## Core Technology Choices

### 1. JAX as the Computational Foundation

**Decision:** Build the entire computational stack on JAX rather than traditional NumPy/SciPy.

**Context:**
Modern quantitative finance requires high-performance computing with automatic differentiation for Greeks calculation, model calibration, and risk analytics. Traditional approaches using finite differences are slow and numerically unstable.

**Rationale:**
- **JIT Compilation**: XLA-based compilation provides 10-100x speedup for repeated calculations
- **Automatic Differentiation**: Accurate Greeks without finite differences
- **Hardware Acceleration**: Seamless GPU/TPU support without code changes
- **Functional Programming**: Pure functions enable reproducibility and parallelization
- **Composability**: Build complex workflows from simple, reusable components

**Trade-offs:**
- **Learning Curve**: JAX requires functional programming mindset (no mutation)
- **Debugging**: JIT-compiled code can be harder to debug than plain Python
- **Ecosystem**: Smaller ecosystem compared to NumPy/PyTorch

**Alternatives Considered:**
- **NumPy/SciPy**: Too slow for production workloads, no AD
- **PyTorch**: Designed for ML, not quantitative finance; mutable state model conflicts with functional purity
- **TensorFlow**: Complex API, less Pythonic than JAX

**Impact:**
- All pricing functions are `@jax.jit` decorated for performance
- Greeks calculated via `jax.grad` and `jax.hessian`
- Multi-GPU support via `jax.pmap` and `jax.pjit`
- Reproducible PRNG with `jax.random.PRNGKey`

**References:**
- [Performance Benchmarks](performance_tuning.md)
- [API Reference](api_reference.md)

---

### 2. Pure Functional Programming Paradigm

**Decision:** Enforce pure functional programming throughout the codebase - no side effects, no mutation.

**Context:**
Financial calculations must be reproducible, testable, and parallelizable. Side effects and mutable state make this difficult.

**Rationale:**
- **Reproducibility**: Same inputs always produce same outputs
- **Testability**: Pure functions are easy to test in isolation
- **Parallelization**: No shared state means safe parallel execution
- **JIT Compatibility**: JAX requires pure functions for compilation
- **Mathematical Clarity**: Code reads like mathematical formulas

**Trade-offs:**
- **Memory Usage**: Creating new arrays instead of mutation increases memory
- **API Design**: Users must understand immutability (e.g., `updated_array = array.at[i].set(value)`)
- **Performance**: Naive pure code can be slower (mitigated by JIT)

**Implementation Guidelines:**
```python
# ❌ Bad: Mutation
def simulate_path(path, dt, sigma):
    for i in range(len(path)):
        path[i] *= jnp.exp(sigma * jnp.sqrt(dt))
    return path

# ✅ Good: Pure function
def simulate_path(S0, dt, sigma, steps):
    increments = jnp.exp(sigma * jnp.sqrt(dt) * jax.random.normal(key, (steps,)))
    return S0 * jnp.cumprod(increments)
```

**Impact:**
- All core functions are pure and JAX-compatible
- State management uses immutable data structures
- Configuration passed explicitly rather than global state

---

### 3. Type Safety with Python Type Hints

**Decision:** Use comprehensive type hints throughout the codebase with `mypy` strict mode.

**Context:**
Financial software requires high reliability. Dynamic typing can hide bugs that manifest in production.

**Rationale:**
- **Early Error Detection**: Catch type errors at development time
- **Better IDE Support**: Autocomplete and inline documentation
- **Self-Documenting Code**: Types serve as inline documentation
- **Refactoring Safety**: Type checker verifies refactoring correctness

**Trade-offs:**
- **Verbosity**: More code to write and maintain
- **Generic Types**: Complex generic types can be hard to read
- **JAX Integration**: JAX arrays have complex type signatures

**Implementation:**
```python
from typing import Tuple, Optional
import jax.numpy as jnp
from jax import Array

def bs_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str
) -> float:
    """Black-Scholes option pricing."""
    ...

def monte_carlo_price(
    key: Array,
    spot: float,
    config: MCConfig
) -> Tuple[float, Array]:
    """Monte Carlo pricing with path output."""
    ...
```

**Tools:**
- `mypy` for static type checking in CI
- `ruff` for consistent type hint formatting
- Pydantic for runtime validation of config objects

---

## Architectural Patterns

### 4. Layered Architecture

**Decision:** Organize the system into four distinct layers: Presentation, Service, Domain, and Infrastructure.

**Context:**
Enterprise financial systems are complex. Clear separation of concerns is essential for maintainability.

**Architecture:**
```
┌────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  REST API │ gRPC │ Python SDK │ CLI │ Jupyter Notebooks       │
└────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
│  Pricing │ Risk │ XVA │ Calibration │ Market Data │ Portfolio │
└────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────────────┐
│                      Domain Layer                               │
│  Models │ Products │ Risk Metrics │ Market Data │ Portfolio    │
└────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
│  JAX Core │ PDE Solvers │ Monte Carlo │ FFT Methods           │
└────────────────────────────────────────────────────────────────┘
```

**Rationale:**
- **Separation of Concerns**: Each layer has a single responsibility
- **Testability**: Layers can be tested independently
- **Flexibility**: Can swap implementations within a layer
- **Scalability**: Layers can be deployed separately in microservices

**Layer Responsibilities:**
1. **Presentation**: User interaction, request/response handling
2. **Service**: Business workflows, orchestration
3. **Domain**: Business logic, models, products
4. **Infrastructure**: Low-level computing, numerical methods

**Trade-offs:**
- **Complexity**: More structure requires more planning
- **Indirection**: May need to pass through multiple layers
- **Overhead**: Layer boundaries add function call overhead (mitigated by JIT)

---

### 5. Strategy Pattern for Model Selection

**Decision:** Use Strategy pattern for pricing model abstraction with a unified interface.

**Context:**
Different products can be priced with different models (Black-Scholes, Heston, SABR, etc.). Users need to switch models easily.

**Implementation:**
```python
from abc import ABC, abstractmethod

class Model(ABC):
    """Base class for all pricing models."""

    @abstractmethod
    def price(self, spot, strike, maturity, **kwargs):
        """Price a derivative."""
        pass

    @abstractmethod
    def calibrate(self, market_data, **kwargs):
        """Calibrate model to market data."""
        pass

    def greeks(self, spot, strike, maturity, **kwargs):
        """Calculate Greeks using automatic differentiation."""
        price_fn = lambda S: self.price(S, strike, maturity, **kwargs)
        delta = jax.grad(price_fn)(spot)
        gamma = jax.grad(jax.grad(price_fn))(spot)
        return delta, gamma

# Usage
class PricingStrategy:
    def __init__(self, model: Model):
        self.model = model

    def price(self, product, market_data):
        return self.model.price(product, market_data)

# Client code
bs_strategy = PricingStrategy(BlackScholesModel())
heston_strategy = PricingStrategy(HestonModel())

price_bs = bs_strategy.price(option, market_data)
price_heston = heston_strategy.price(option, market_data)
```

**Benefits:**
- Easy to add new models without changing client code
- Consistent interface for all models
- Automatic Greek calculation via AD
- Model comparison and selection

---

### 6. Repository Pattern for Data Access

**Decision:** Abstract all data access behind repository interfaces to decouple business logic from storage.

**Context:**
Market data comes from multiple sources (Bloomberg, Refinitiv, databases). Risk limits and portfolios are stored in databases. Business logic should not depend on specific storage implementations.

**Implementation:**
```python
class MarketDataRepository(ABC):
    """Abstraction for market data storage."""

    @abstractmethod
    async def get_spot_price(self, instrument: str, timestamp: datetime):
        """Retrieve spot price."""
        pass

    @abstractmethod
    async def get_curve(self, curve_name: str, date: date):
        """Retrieve yield curve."""
        pass

# PostgreSQL implementation
class PostgreSQLMarketDataRepository(MarketDataRepository):
    async def get_spot_price(self, instrument, timestamp):
        return await self.db.fetchone("""
            SELECT price FROM market_data
            WHERE instrument = $1 AND timestamp <= $2
            ORDER BY timestamp DESC LIMIT 1
        """, instrument, timestamp)

# TimescaleDB implementation with compression
class TimescaleDBRepository(MarketDataRepository):
    # Optimized for time-series data with 90% compression
    ...
```

**Benefits:**
- **Testability**: Mock repositories for unit tests
- **Flexibility**: Swap storage implementations without changing business logic
- **Performance**: Optimize queries without affecting clients
- **Multi-Source**: Aggregate data from multiple vendors transparently

**Trade-offs:**
- **Abstraction Overhead**: Additional layer of indirection
- **Leaky Abstractions**: Storage-specific features may leak through interface

---

## Performance & Scalability

### 7. JIT Compilation for Hot Paths

**Decision:** Apply `@jax.jit` decoration to all performance-critical functions.

**Context:**
Monte Carlo simulations and PDE solvers are called millions of times. Python overhead is unacceptable.

**Implementation Strategy:**
```python
@jax.jit
def simulate_gbm(key, S0, mu, sigma, T, config):
    """Simulate Geometric Brownian Motion paths."""
    dt = T / config.steps
    dW = jax.random.normal(key, (config.paths, config.steps)) * jnp.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * dt
    log_S = jnp.log(S0) + jnp.cumsum(drift + sigma * dW, axis=1)
    return jnp.exp(log_S)

@jax.jit
def present_value(payoffs, times, rate):
    """Calculate present value of cash flows."""
    discount_factors = jnp.exp(-rate * times)
    return jnp.sum(payoffs * discount_factors)
```

**Performance Gains:**
- **Black-Scholes (1M options)**: 10x speedup (500ms → 50ms on CPU)
- **Monte Carlo (100K paths)**: 10x speedup (2000ms → 200ms on CPU)
- **Heston Calibration**: 10x speedup (5000ms → 500ms)

**Best Practices:**
1. JIT entire functions, not just loops
2. Avoid Python control flow inside JIT (use `jax.lax.cond`, `jax.lax.fori_loop`)
3. Pre-allocate arrays outside JIT when possible
4. Use `static_argnums` for configuration parameters

**Trade-offs:**
- **Compilation Time**: First call is slow due to XLA compilation
- **Debugging**: Harder to debug JIT-compiled code
- **Memory**: JIT cache can consume significant memory

---

### 8. Multi-GPU Parallelization

**Decision:** Use `jax.pmap` for data parallelism across multiple GPUs/TPUs.

**Context:**
Portfolio risk calculations require pricing thousands of options. Single GPU is insufficient.

**Implementation:**
```python
from functools import partial

@partial(jax.pmap, axis_name='devices')
def distributed_pricing(keys, spots, strikes, maturities, config):
    """Price options in parallel across devices."""
    prices = jax.vmap(price_option)(
        spots, strikes, maturities, config
    )
    return prices

# Split work across devices
num_devices = jax.device_count()
keys = jax.random.split(jax.random.PRNGKey(42), num_devices)
prices = distributed_pricing(keys, spots, strikes, maturities, config)
```

**Scalability:**
- **Linear Scaling**: Near-linear speedup up to 8 GPUs
- **Batch Sizes**: Optimal at 10K-100K options per GPU
- **Memory**: Limited by GPU memory (32GB per A100)

**Deployment:**
- Kubernetes deployment with resource management
- Auto-scaling based on queue depth
- Multi-region deployment for disaster recovery

---

### 9. Adaptive Mesh Refinement for PDEs

**Decision:** Implement Adaptive Mesh Refinement (AMR) for PDE solvers to optimize accuracy vs. performance.

**Context:**
Option pricing PDEs require fine grids near barriers and strikes but can use coarse grids elsewhere. Fixed grids waste computation.

**Rationale:**
- **Accuracy**: Refine grid only where needed (barriers, strike, discontinuities)
- **Performance**: Reduce grid points by 50-80% vs. uniform grid
- **Memory**: Lower memory footprint

**Implementation:**
- Error estimation at each time step
- Hierarchical grid structure with local refinement
- Conservative interpolation for smooth transitions

**Trade-offs:**
- **Complexity**: More complex implementation than uniform grid
- **Overhead**: Grid adaptation adds computational cost
- **Parallelization**: Irregular grids harder to parallelize

---

## Security & Governance

### 10. Multi-Tenancy with Strict Isolation

**Decision:** Implement multi-tenancy at the infrastructure level with strict desk/legal entity isolation.

**Context:**
Investment banks have multiple trading desks and legal entities. Data must be isolated for compliance and security.

**Architecture:**
```python
class TenantContext:
    """Tenant context for multi-tenancy."""
    tenant_id: str
    desk: str
    legal_entity: str
    geography: str
    permissions: Set[str]

@require_tenant_context
async def price_portfolio(portfolio_id: str, context: TenantContext):
    """Price portfolio with tenant isolation."""
    # Verify tenant owns portfolio
    await verify_ownership(portfolio_id, context.tenant_id)

    # Apply tenant-specific limits
    limits = await get_tenant_limits(context.tenant_id)

    # Calculate with tenant quota
    with compute_quota(context.tenant_id):
        prices = await calculate_prices(portfolio_id)

    return prices
```

**Features:**
- **Data Isolation**: Row-level security in database
- **Compute Quotas**: CPU/GPU allocation per tenant
- **Cost Allocation**: Track usage and costs per desk
- **SLA Monitoring**: Per-tenant SLA tracking

**Enforcement:**
- Database row-level security (RLS)
- Application-level permission checks
- Network policies in K8s deployments
- Audit logging of all access

---

### 11. Role-Based Access Control (RBAC)

**Decision:** Implement fine-grained RBAC with hierarchical roles and permissions.

**Context:**
Different users have different responsibilities. Junior traders should not modify risk limits. Compliance officers need read-only access.

**Permission Model:**
```python
class Permission(Enum):
    PRICING_EXECUTE = "pricing.execute"
    PRICING_VIEW = "pricing.view"
    RISK_VIEW = "risk.view"
    RISK_MODIFY = "risk.modify"
    LIMITS_VIEW = "limits.view"
    LIMITS_MODIFY = "limits.modify"
    PORTFOLIO_VIEW = "portfolio.view"
    PORTFOLIO_TRADE = "portfolio.trade"

class Role(Enum):
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMIN = "admin"

# Role definitions
ROLE_PERMISSIONS = {
    Role.TRADER: {
        Permission.PRICING_EXECUTE,
        Permission.PRICING_VIEW,
        Permission.RISK_VIEW,
        Permission.PORTFOLIO_VIEW,
        Permission.PORTFOLIO_TRADE,
    },
    Role.RISK_MANAGER: {
        Permission.PRICING_VIEW,
        Permission.RISK_VIEW,
        Permission.RISK_MODIFY,
        Permission.LIMITS_VIEW,
        Permission.LIMITS_MODIFY,
    },
    Role.COMPLIANCE_OFFICER: {
        Permission.PRICING_VIEW,
        Permission.RISK_VIEW,
        Permission.LIMITS_VIEW,
        Permission.PORTFOLIO_VIEW,
    },
    Role.ADMIN: set(Permission),  # All permissions
}

# Usage in API
@app.post("/price")
async def price_option(
    request: PricingRequest,
    user: User = Depends(require_permission(Permission.PRICING_EXECUTE))
):
    return pricing_service.price(request)
```

**Authentication:**
- SSO (Single Sign-On) with OAuth 2.0/OpenID Connect
- Multi-factor authentication (MFA) required for production
- LDAP/Active Directory integration for enterprise

---

### 12. Immutable Audit Trail

**Decision:** Log all user actions to an immutable audit trail for compliance.

**Context:**
Regulatory requirements (MiFID II, Dodd-Frank) mandate comprehensive audit logging. Logs must be tamper-proof.

**Implementation:**
```python
class AuditLogger:
    async def log_action(
        self,
        user: User,
        action: str,
        resource: str,
        result: str,
        metadata: dict
    ):
        """Log user action to immutable audit trail."""
        await self.db.execute("""
            INSERT INTO audit_log (
                timestamp, user_id, action, resource, result, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6)
        """, datetime.utcnow(), user.id, action, resource, result, metadata)

# Decorator for automatic audit logging
def audited(action: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User, **kwargs):
            try:
                result = await func(*args, user=user, **kwargs)
                await audit_logger.log_action(
                    user, action, kwargs, "SUCCESS", {"result": result}
                )
                return result
            except Exception as e:
                await audit_logger.log_action(
                    user, action, kwargs, "FAILURE", {"error": str(e)}
                )
                raise
        return wrapper
    return decorator
```

**Logged Events:**
- All pricing requests
- Risk limit modifications
- Trade bookings and amendments
- Configuration changes
- User authentication events

**Storage:**
- Write-only append log (no updates or deletes)
- Cryptographic signing for tamper detection
- Long-term retention (7-10 years for regulatory compliance)

---

## Data Management

### 13. Multi-Source Market Data Architecture

**Decision:** Build vendor-agnostic market data layer supporting Bloomberg, Refinitiv, and custom sources.

**Context:**
Investment banks use multiple market data vendors. System must aggregate data from all sources transparently.

**Architecture:**
```python
class MarketDataAdapter(ABC):
    """Base adapter for market data vendors."""

    @abstractmethod
    async def connect(self):
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def subscribe(self, instruments):
        """Subscribe to real-time data."""
        pass

    @abstractmethod
    async def fetch_historical(self, instrument, start, end):
        """Fetch historical data."""
        pass

# Bloomberg implementation
class BloombergAdapter(MarketDataAdapter):
    async def connect(self):
        self.session = await self._create_bbg_session()

    async def subscribe(self, instruments):
        for instrument in instruments:
            await self.session.subscribe(instrument, self._on_data)

# Refinitiv implementation
class RefinitivAdapter(MarketDataAdapter):
    async def connect(self):
        self.session = await refinitiv.open_platform_session()

    async def subscribe(self, instruments):
        await self.session.stream_pricing(instruments, self._on_data)
```

**Feed Management:**
- Automatic failover between vendors
- Data normalization (different vendors use different formats)
- Conflation and throttling to manage update rates
- Quality validation before publishing to clients

**Benefits:**
- **Vendor Independence**: Not locked into single vendor
- **Redundancy**: Automatic failover improves reliability
- **Cost Optimization**: Use cheapest source for each data type

---

### 14. TimescaleDB for Time-Series Storage

**Decision:** Use TimescaleDB (PostgreSQL extension) for market data storage with automatic compression.

**Context:**
Market data is inherently time-series. Millions of ticks per day require efficient storage and fast queries.

**Rationale:**
- **Compression**: 90% compression ratio for historical data
- **Query Performance**: Hypertables optimize time-series queries
- **PostgreSQL Compatibility**: Standard SQL, ACID transactions
- **Continuous Aggregates**: Pre-computed rollups for fast analytics

**Implementation:**
```sql
-- Create hypertable for market data
CREATE TABLE market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    instrument VARCHAR(50) NOT NULL,
    price DOUBLE PRECISION,
    volume BIGINT,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION
);

SELECT create_hypertable('market_data', 'timestamp');

-- Enable compression (90% space savings)
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument'
);

SELECT add_compression_policy('market_data', INTERVAL '7 days');

-- Continuous aggregate for OHLC bars
CREATE MATERIALIZED VIEW ohlc_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucket,
    instrument,
    FIRST(price, timestamp) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, timestamp) AS close,
    SUM(volume) AS volume
FROM market_data
GROUP BY bucket, instrument;
```

**Performance:**
- **Ingestion**: 100K+ ticks/second on standard hardware
- **Compression**: 1TB raw data → 100GB compressed
- **Query**: Millisecond response for date range queries

**Trade-offs:**
- **Write Overhead**: Compression adds latency (mitigated by background jobs)
- **Complexity**: More complex than plain PostgreSQL
- **Lock-In**: TimescaleDB-specific features not portable

---

### 15. Data Validation Pipeline

**Decision:** Implement comprehensive data validation before using market data in pricing.

**Context:**
Bad market data causes incorrect pricing and risk calculations. Must detect and reject outliers.

**Validation Rules:**
```python
class ValidationPipeline:
    def __init__(self):
        self.validators = []

    def add_validator(self, validator):
        self.validators.append(validator)

    async def validate(self, tick):
        for validator in self.validators:
            result = await validator.validate(tick)
            if not result.is_valid:
                await self.handle_invalid(tick, result)
                return False
        return True

# Validators
class PriceRangeValidator:
    def validate(self, tick):
        if tick.price <= 0:
            return ValidationResult(False, "Price must be positive")
        if abs(tick.price - tick.prev_price) / tick.prev_price > 0.20:
            return ValidationResult(False, "Price jump > 20%")
        return ValidationResult(True, None)

class SpreadValidator:
    def validate(self, tick):
        spread = tick.ask - tick.bid
        if spread < 0:
            return ValidationResult(False, "Negative spread")
        if spread / tick.mid > 0.05:
            return ValidationResult(False, "Spread > 5% of mid")
        return ValidationResult(True, None)

class VolumeValidator:
    def validate(self, tick):
        if tick.volume < 0:
            return ValidationResult(False, "Negative volume")
        if tick.volume > tick.avg_volume * 10:
            return ValidationResult(False, "Volume spike > 10x average")
        return ValidationResult(True, None)
```

**Actions on Invalid Data:**
- Log warning with details
- Alert market data team
- Use last known good price
- Mark data as stale in cache

---

## Regulatory Compliance

### 16. Regulatory-First Design

**Decision:** Design all risk systems to meet regulatory requirements (FRTB, SA-CCR, SIMM) from the start.

**Context:**
Post-2008 regulations (Basel III/IV, Dodd-Frank) impose strict capital requirements. Retrofitting compliance is expensive.

**Regulatory Frameworks:**

1. **FRTB (Fundamental Review of the Trading Book)**
   - Standardized Approach (SA): Delta, vega, curvature risk charges
   - Internal Models Approach (IMA): ES 97.5%, P&L attribution, backtesting
   - Default Risk Charge (DRC) for credit-sensitive instruments
   - Residual Risk Add-On (RRAO) for exotic payoffs

2. **SA-CCR (Standardized Approach for Counterparty Credit Risk)**
   - Replacement cost (RC) for current exposure
   - Potential Future Exposure (PFE) add-on by asset class
   - Hedging set construction with netting benefits

3. **ISDA SIMM (Standard Initial Margin Model)**
   - Risk-based initial margin for uncleared derivatives
   - Product class calculations (RatesFX, Credit, Equity, Commodity)
   - Concentration thresholds and risk weights

**Implementation:**
```python
# FRTB Standardized Approach
class FRTBCalculator:
    def calculate_capital(self, portfolio):
        delta_charge = self.delta_risk_charge(portfolio)
        vega_charge = self.vega_risk_charge(portfolio)
        curvature_charge = self.curvature_risk_charge(portfolio)
        drc = self.default_risk_charge(portfolio)
        rrao = self.residual_risk_add_on(portfolio)

        return delta_charge + vega_charge + curvature_charge + drc + rrao

# SA-CCR
class SACCRCalculator:
    def calculate_ead(self, netting_set):
        rc = self.replacement_cost(netting_set)
        pfe = self.potential_future_exposure(netting_set)
        return alpha * (rc + pfe)  # alpha = 1.4

# ISDA SIMM
class SIMMCalculator:
    def calculate_im(self, portfolio):
        sensitivities = self.calculate_sensitivities(portfolio)
        weighted = self.apply_risk_weights(sensitivities)
        aggregated = self.aggregate_across_buckets(weighted)
        return aggregated.total_im
```

**Benefits:**
- **Regulatory Compliance**: Meet capital requirements
- **Risk Management**: More accurate risk measurement
- **Capital Efficiency**: Optimize capital allocation

---

### 17. Comprehensive Regulatory Reporting

**Decision:** Build automated regulatory reporting for EMIR, MiFID II, Dodd-Frank, Basel III/IV.

**Context:**
Manual regulatory reporting is error-prone and time-consuming. Automation ensures accuracy and timeliness.

**Report Types:**
1. **EMIR Trade Reporting**: All OTC derivatives to trade repositories
2. **MiFID II Transaction Reporting**: RTS 22 format, RTS 23 reference data
3. **Basel III/IV Capital Reporting**: RWA, leverage ratio, CVA capital, FRTB market risk
4. **Dodd-Frank**: Swap Data Repository (SDR) reporting

**Implementation:**
```python
class RegulatoryReporter:
    async def generate_emir_report(self, trades, report_date):
        """Generate EMIR trade report in ISO 20022 XML."""
        report = EMIRReport()
        for trade in trades:
            report.add_trade(
                uti=trade.uti,
                counterparty=trade.counterparty,
                notional=trade.notional,
                maturity=trade.maturity,
                # ... all required fields
            )
        xml = report.to_xml()
        await self.submit_to_repository(xml)

    async def generate_mifid_report(self, transactions, report_date):
        """Generate MiFID II transaction report (RTS 22)."""
        report = MiFIDTransactionReport()
        for txn in transactions:
            report.add_transaction(
                instrument_id=txn.isin,
                price=txn.price,
                quantity=txn.quantity,
                venue=txn.venue,
                timestamp=txn.timestamp,
                # ... all RTS 22 fields
            )
        xml = report.to_xml()
        await self.submit_to_regulator(xml)
```

**Automation:**
- Scheduled daily/weekly/monthly reports
- Real-time validation before submission
- Reconciliation with trade repository
- Audit trail of all submissions

---

## API Design

### 18. REST and gRPC Dual API

**Decision:** Provide both REST and gRPC APIs for different use cases.

**Context:**
External clients prefer REST. Internal services need high-performance gRPC.

**REST API (FastAPI):**
```python
from fastapi import FastAPI, Depends

app = FastAPI(title="Neutryx Pricing API")

@app.post("/api/v1/price/option")
async def price_option(
    request: OptionPricingRequest,
    user: User = Depends(get_current_user)
):
    """Price a vanilla option."""
    result = await pricing_service.price_option(
        spot=request.spot,
        strike=request.strike,
        maturity=request.maturity,
        volatility=request.volatility,
        rate=request.rate,
        option_type=request.option_type
    )
    return OptionPricingResponse(
        price=result.price,
        delta=result.delta,
        gamma=result.gamma,
        vega=result.vega,
        theta=result.theta,
        rho=result.rho
    )
```

**gRPC API:**
```protobuf
syntax = "proto3";

service PricingService {
    rpc PriceOption(OptionPricingRequest) returns (OptionPricingResponse);
    rpc PricePortfolio(PortfolioPricingRequest) returns (stream PriceUpdate);
}

message OptionPricingRequest {
    double spot = 1;
    double strike = 2;
    double maturity = 3;
    double volatility = 4;
    double rate = 5;
    string option_type = 6;
}

message OptionPricingResponse {
    double price = 1;
    double delta = 2;
    double gamma = 3;
    double vega = 4;
    double theta = 5;
    double rho = 6;
}
```

**Use Cases:**
- **REST**: External clients, web UIs, exploratory analysis
- **gRPC**: Internal microservices, high-frequency updates, streaming

**Benefits:**
- **REST**: Human-readable, browser-friendly, easier debugging
- **gRPC**: Binary protocol, 5-10x faster, bidirectional streaming

---

## Testing Strategy

### 19. Comprehensive Test Suite with 500+ Tests

**Decision:** Maintain extensive test coverage across unit, integration, and regression tests.

**Context:**
Financial software bugs can cause million-dollar losses. High test coverage is essential.

**Test Categories:**

1. **Unit Tests (~300 tests)**
   - Individual function correctness
   - Edge cases and error handling
   - Numerical accuracy vs. closed-form solutions

2. **Integration Tests (~100 tests)**
   - End-to-end workflows
   - Multi-component interactions
   - Database and external service integration

3. **Product Tests (~100 tests)**
   - All product types (IR, FX, Equity, Credit, Commodity)
   - Vanilla and exotic payoffs
   - Greeks accuracy

4. **Regulatory Tests (~120 tests)**
   - FRTB SA/IMA calculations
   - SA-CCR exposure calculations
   - SIMM initial margin
   - Compliance reporting formats

5. **Performance Tests (~20 tests)**
   - Benchmarking vs. baseline
   - GPU acceleration verification
   - Memory usage profiling

**Testing Tools:**
```bash
# Run all tests
pytest -v

# Run specific test category
pytest src/neutryx/tests/products/ -v
pytest src/neutryx/tests/market/ -v
pytest src/neutryx/tests/regulatory/ -v

# Coverage report
pytest --cov=neutryx --cov-report=html

# Parallel execution for speed
pytest -n auto  # Use all CPU cores
```

**Quality Gates:**
- All tests must pass before merge
- Coverage must be ≥85%
- No regressions in performance benchmarks
- All security scans clean (bandit, pip-audit)

---

### 20. Property-Based Testing with Hypothesis

**Decision:** Use Hypothesis for property-based testing of mathematical properties.

**Context:**
Traditional unit tests check specific examples. Property-based tests check mathematical properties across many random inputs.

**Example:**
```python
from hypothesis import given, strategies as st

@given(
    spot=st.floats(min_value=1.0, max_value=1000.0),
    strike=st.floats(min_value=1.0, max_value=1000.0),
    volatility=st.floats(min_value=0.01, max_value=1.0),
    rate=st.floats(min_value=-0.01, max_value=0.20),
    maturity=st.floats(min_value=0.01, max_value=10.0)
)
def test_call_put_parity(spot, strike, volatility, rate, maturity):
    """Test call-put parity: C - P = S - K*exp(-rT)"""
    call = bs_price(spot, strike, maturity, rate, 0.0, volatility, "call")
    put = bs_price(spot, strike, maturity, rate, 0.0, volatility, "put")
    pv_strike = strike * jnp.exp(-rate * maturity)

    assert jnp.isclose(call - put, spot - pv_strike, rtol=1e-6)

@given(
    spot=st.floats(min_value=1.0, max_value=1000.0),
    strike=st.floats(min_value=1.0, max_value=1000.0),
    volatility=st.floats(min_value=0.01, max_value=1.0),
)
def test_option_prices_positive(spot, strike, volatility):
    """Test that option prices are always non-negative."""
    call = bs_price(spot, strike, 1.0, 0.05, 0.0, volatility, "call")
    put = bs_price(spot, strike, 1.0, 0.05, 0.0, volatility, "put")

    assert call >= 0
    assert put >= 0

@given(
    spot=st.floats(min_value=1.0, max_value=1000.0),
    volatility=st.floats(min_value=0.01, max_value=1.0),
)
def test_delta_bounds(spot, volatility):
    """Test that delta is bounded: 0 <= call delta <= 1, -1 <= put delta <= 0"""
    call_delta = bs_delta(spot, 100.0, 1.0, 0.05, 0.0, volatility, "call")
    put_delta = bs_delta(spot, 100.0, 1.0, 0.05, 0.0, volatility, "put")

    assert 0 <= call_delta <= 1
    assert -1 <= put_delta <= 0
```

**Benefits:**
- **Edge Case Discovery**: Finds bugs in corner cases
- **Mathematical Correctness**: Verifies pricing properties
- **Regression Prevention**: Properties must hold for all inputs

---

## Decision Log

| ID | Decision | Date | Status | Reviewed |
|----|----------|------|--------|----------|
| DD-001 | JAX as computational foundation | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-002 | Pure functional programming | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-003 | Type safety with mypy | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-004 | Layered architecture | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-005 | Strategy pattern for models | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-006 | Repository pattern for data | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-007 | JIT compilation for hot paths | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-008 | Multi-GPU parallelization | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-009 | AMR for PDE solvers | 2025-Q2 | ✅ Accepted | 2025-06 |
| DD-010 | Multi-tenancy isolation | 2025-Q1 | ✅ Accepted | 2025-03 |
| DD-011 | RBAC with SSO/MFA | 2025-Q2 | ✅ Accepted | 2025-06 |
| DD-012 | Immutable audit trail | 2025-Q1 | ✅ Accepted | 2025-03 |
| DD-013 | Multi-source market data | 2025-Q1 | ✅ Accepted | 2025-02 |
| DD-014 | TimescaleDB for time-series | 2025-Q1 | ✅ Accepted | 2025-02 |
| DD-015 | Data validation pipeline | 2025-Q1 | ✅ Accepted | 2025-02 |
| DD-016 | Regulatory-first design | 2025-Q1 | ✅ Accepted | 2025-03 |
| DD-017 | Automated regulatory reporting | 2025-Q1 | ✅ Accepted | 2025-03 |
| DD-018 | REST and gRPC dual API | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-019 | Comprehensive test suite | 2024-Q4 | ✅ Accepted | 2025-01 |
| DD-020 | Property-based testing | 2024-Q4 | ✅ Accepted | 2025-01 |

---

## Future Considerations

### Under Evaluation

1. **Rust FFI for Critical Paths**
   - Evaluate Rust for ultra-low latency pricing
   - Trade-off: Development complexity vs. marginal performance gains

2. **Graph-Based Workflow Engine**
   - DAG-based workflow orchestration (Airflow, Prefect, Temporal)
   - Trade-off: Flexibility vs. operational complexity

3. **Real-Time Streaming with Kafka**
   - Replace polling with streaming for market data
   - Trade-off: Low latency vs. infrastructure complexity

4. **Machine Learning for Pricing**
   - Neural network surrogates for slow models (Heston, SABR)
   - Trade-off: Speed vs. accuracy and explainability

5. **Quantum Computing Integration**
   - Explore quantum algorithms for Monte Carlo (Amplitude Estimation)
   - Trade-off: Potential speedup vs. hardware availability and maturity

---

## References

- [Architecture Guide](architecture.md) - Detailed system architecture
- [Performance Tuning](performance_tuning.md) - Optimization strategies
- [Developer Guide](developer_guide.md) - Contribution guidelines
- [API Reference](api_reference.md) - Complete API documentation
- [Test Coverage Report](test_coverage.md) - Test statistics and coverage

---

**Document Ownership:** Neutryx Core Architecture Team
**Last Updated:** November 2025
**Next Review:** Q1 2026
