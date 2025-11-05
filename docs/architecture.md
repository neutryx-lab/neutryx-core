# Neutryx Core Architecture

This document provides a comprehensive overview of Neutryx Core's architecture, design patterns, and implementation details.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Design Patterns](#design-patterns)
4. [Data Flow](#data-flow)
5. [Scalability & Performance](#scalability-performance)
6. [Security Architecture](#security-architecture)
7. [Deployment Architecture](#deployment-architecture)

## System Architecture

### Layered Architecture

Neutryx Core follows a layered architecture pattern:

```
┌────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │REST API  │  │  gRPC    │  │   CLI    │  │Notebooks │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Pricing     │  │     Risk     │  │     XVA      │        │
│  │  Service     │  │   Service    │  │   Service    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Calibration  │  │   Market     │  │  Portfolio   │        │
│  │  Service     │  │    Data      │  │   Service    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────────────┐
│                      Domain Layer                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │ Models  │  │Products │  │  Risk   │  │  Market │          │
│  │(BS,Hest)│  │(Options)│  │Metrics  │  │  Data   │          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
└────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   JAX    │  │   PDE    │  │  Monte   │  │   FFT    │      │
│  │  Core    │  │ Solvers  │  │  Carlo   │  │ Methods  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────────────────────────────────────────┐
│                      Cross-Cutting Concerns                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Observ.  │  │ Security │  │  Config  │  │  Cache   │      │
│  │(Prom/Jae)│  │(RBAC/Aud)│  │   Mgmt   │  │  Layer   │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────────┘
```

### Component Overview

#### 1. Presentation Layer
- **REST API**: FastAPI-based RESTful endpoints
- **gRPC**: High-performance RPC for internal services
- **CLI**: Command-line interface for batch operations
- **Notebooks**: Jupyter integration for interactive analysis

#### 2. Service Layer
- **Pricing Service**: Orchestrates pricing workflows
- **Risk Service**: Manages risk calculations and limits
- **XVA Service**: Handles counterparty credit risk
- **Calibration Service**: Model parameter estimation
- **Market Data Service**: Real-time data ingestion
- **Portfolio Service**: Portfolio analytics and optimization

#### 3. Domain Layer
- **Models**: Stochastic models (Black-Scholes, Heston, SABR)
- **Products**: Multi-asset class derivatives
- **Risk Metrics**: VaR, Greeks, exposure profiles
- **Market Data**: Curves, surfaces, volatilities

#### 4. Infrastructure Layer
- **JAX Core**: JIT compilation, automatic differentiation
- **PDE Solvers**: Crank-Nicolson, finite differences
- **Monte Carlo**: Path simulation, variance reduction
- **FFT Methods**: Fast Fourier transforms for pricing

## Core Components

### 1. Pricing Engine (`neutryx.core.engine`)

The pricing engine is the computational heart of Neutryx Core.

```python
# Key classes and interfaces
class MCConfig:
    """Monte Carlo configuration"""
    steps: int          # Number of timesteps
    paths: int          # Number of simulations
    seed: int          # PRNG seed
    antithetic: bool   # Variance reduction

class PDEConfig:
    """PDE solver configuration"""
    grid_size: int     # Spatial grid points
    timesteps: int     # Temporal grid points
    theta: float       # Crank-Nicolson parameter

# Core functions
@jax.jit
def simulate_gbm(key, S0, mu, sigma, T, config):
    """Simulate Geometric Brownian Motion paths"""
    ...

@jax.jit
def present_value(payoffs, times, rate):
    """Calculate present value of cash flows"""
    ...
```

**Design Principles**:
- Pure functional programming (no side effects)
- JAX JIT compilation for performance
- Automatic differentiation for Greeks
- GPU/TPU acceleration support

### 2. Model Framework (`neutryx.models`)

Unified interface for all pricing models.

```python
from abc import ABC, abstractmethod

class Model(ABC):
    """Base class for all models"""

    @abstractmethod
    def price(self, spot, strike, maturity, **kwargs):
        """Price a derivative"""
        pass

    @abstractmethod
    def calibrate(self, market_data, **kwargs):
        """Calibrate model to market data"""
        pass

    def greeks(self, spot, strike, maturity, **kwargs):
        """Calculate Greeks using automatic differentiation"""
        price_fn = lambda S: self.price(S, strike, maturity, **kwargs)
        delta = jax.grad(price_fn)(spot)
        gamma = jax.grad(jax.grad(price_fn))(spot)
        return delta, gamma

# Example implementation
class BlackScholesModel(Model):
    def price(self, spot, strike, maturity, r, q, sigma, option_type):
        # Black-Scholes formula
        ...

    def calibrate(self, market_data):
        # Implied volatility calibration
        ...
```

**Key Features**:
- Consistent interface across all models
- Automatic Greek calculation via AD
- Built-in calibration support
- Parameter validation

### 3. Product Library (`neutryx.products`)

Multi-asset class product hierarchy.

```python
class Product(ABC):
    """Base product interface"""

    @abstractmethod
    def payoff(self, paths):
        """Calculate payoff given price paths"""
        pass

    def price(self, model, market_data, config):
        """Price product using specified model"""
        key = jax.random.PRNGKey(config.seed)
        paths = model.simulate(key, market_data, config)
        payoff = self.payoff(paths)
        return self.present_value(payoff, market_data)

# Product hierarchy
class Option(Product):
    """Base class for options"""
    pass

class VanillaOption(Option):
    """European vanilla option"""
    def payoff(self, paths):
        ST = paths[:, -1]
        if self.option_type == "call":
            return jnp.maximum(ST - self.strike, 0.0)
        else:
            return jnp.maximum(self.strike - ST, 0.0)

class AsianOption(Option):
    """Asian option with path averaging"""
    def payoff(self, paths):
        avg = jnp.mean(paths, axis=1)
        return jnp.maximum(avg - self.strike, 0.0)
```

### 4. Risk Framework (`neutryx.risk`)

Comprehensive risk analytics.

```python
# VaR calculation
@jax.jit
def historical_var(returns, confidence=0.95):
    """Historical simulation VaR"""
    sorted_returns = jnp.sort(returns)
    index = int((1 - confidence) * len(returns))
    return -sorted_returns[index]

@jax.jit
def parametric_var(portfolio_value, mean_return, vol, confidence=0.95):
    """Parametric VaR using variance-covariance"""
    from scipy.stats import norm
    z_score = norm.ppf(1 - confidence)
    return portfolio_value * (mean_return - z_score * vol)

# Position limits
class LimitChecker:
    def check_trade(self, proposed_trade, current_state):
        """Pre-trade limit checking"""
        post_trade_state = self.calculate_post_trade(
            proposed_trade, current_state
        )

        breaches = []
        if post_trade_state.notional > self.limits.notional_limit:
            breaches.append("notional_limit")
        if post_trade_state.var > self.limits.var_limit:
            breaches.append("var_limit")

        return LimitCheckResult(
            approved=len(breaches) == 0,
            breaches=breaches
        )
```

### 5. Market Data Infrastructure (`neutryx.market`)

Real-time market data pipeline.

```python
class MarketDataAdapter(ABC):
    """Base adapter for market data vendors"""

    @abstractmethod
    async def connect(self):
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def subscribe(self, instruments):
        """Subscribe to real-time data"""
        pass

    @abstractmethod
    async def fetch_historical(self, instrument, start, end):
        """Fetch historical data"""
        pass

# Bloomberg implementation
class BloombergAdapter(MarketDataAdapter):
    async def connect(self):
        self.session = await self._create_session()

    async def subscribe(self, instruments):
        for instrument in instruments:
            await self.session.subscribe(instrument, self._on_data)

# Storage layer
class TimescaleDBStorage:
    async def store_tick(self, tick_data):
        """Store tick data with automatic compression"""
        await self.execute("""
            INSERT INTO market_data (timestamp, instrument, price, volume)
            VALUES ($1, $2, $3, $4)
        """, tick_data)
```

## Design Patterns

### 1. Strategy Pattern (Model Selection)

```python
class PricingStrategy:
    def __init__(self, model):
        self.model = model

    def price(self, product, market_data):
        return self.model.price(product, market_data)

# Usage
bs_strategy = PricingStrategy(BlackScholesModel())
heston_strategy = PricingStrategy(HestonModel())

price_bs = bs_strategy.price(option, market_data)
price_heston = heston_strategy.price(option, market_data)
```

### 2. Factory Pattern (Product Creation)

```python
class ProductFactory:
    @staticmethod
    def create(product_type, **kwargs):
        if product_type == "european_call":
            return VanillaOption("call", **kwargs)
        elif product_type == "asian":
            return AsianOption(**kwargs)
        elif product_type == "barrier":
            return BarrierOption(**kwargs)
        else:
            raise ValueError(f"Unknown product: {product_type}")

# Usage
option = ProductFactory.create("european_call", strike=100, maturity=1.0)
```

### 3. Decorator Pattern (Observability)

```python
def track_metrics(func):
    """Decorator to track pricing metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            metrics.pricing_duration.observe(time.time() - start)
            metrics.pricing_success.inc()
            return result
        except Exception as e:
            metrics.pricing_errors.inc()
            raise
    return wrapper

@track_metrics
def price_option(S, K, T, r, sigma):
    return bs_price(S, K, T, r, 0.0, sigma, "call")
```

### 4. Observer Pattern (Market Data Updates)

```python
class MarketDataObserver(ABC):
    @abstractmethod
    def on_tick(self, tick):
        pass

class PricingEngine(MarketDataObserver):
    def on_tick(self, tick):
        # Reprice portfolio on market data update
        self.update_prices(tick)

class RiskEngine(MarketDataObserver):
    def on_tick(self, tick):
        # Recalculate risk metrics
        self.update_risk(tick)

# Market data subject
class MarketDataFeed:
    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, tick):
        for observer in self.observers:
            observer.on_tick(tick)
```

### 5. Repository Pattern (Data Access)

```python
class MarketDataRepository:
    """Abstraction for market data storage"""

    async def get_spot_price(self, instrument, timestamp):
        """Retrieve spot price"""
        pass

    async def get_curve(self, curve_name, date):
        """Retrieve yield curve"""
        pass

    async def get_volatility_surface(self, underlying, date):
        """Retrieve vol surface"""
        pass

# PostgreSQL implementation
class PostgreSQLMarketDataRepository(MarketDataRepository):
    async def get_spot_price(self, instrument, timestamp):
        return await self.db.fetchone("""
            SELECT price FROM market_data
            WHERE instrument = $1 AND timestamp <= $2
            ORDER BY timestamp DESC LIMIT 1
        """, instrument, timestamp)
```

## Scalability & Performance

### Horizontal Scaling

```python
# Distributed pricing across multiple GPUs
@partial(jax.pmap, axis_name='devices')
def distributed_pricing(keys, spots, strikes, maturities, config):
    """Price options in parallel across devices"""
    prices = jax.vmap(price_option)(
        spots, strikes, maturities, config
    )
    return prices

# Split work across devices
num_devices = jax.device_count()
keys = jax.random.split(jax.random.PRNGKey(42), num_devices)
prices = distributed_pricing(keys, spots, strikes, maturities, config)
```

### Vertical Scaling (GPU Acceleration)

```python
# JIT compilation for GPU
@jax.jit
def monte_carlo_pricer(key, S0, K, T, r, sigma, paths):
    """GPU-accelerated Monte Carlo pricing"""
    paths = simulate_gbm(key, S0, r, sigma, T, paths)
    payoff = jnp.maximum(paths[:, -1] - K, 0.0)
    return jnp.mean(payoff) * jnp.exp(-r * T)

# Automatically runs on GPU if available
price = monte_carlo_pricer(key, 100, 100, 1.0, 0.05, 0.20, 100_000)
```

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_yield_curve(date, currency):
    """Cache yield curves to avoid repeated DB queries"""
    return fetch_curve_from_db(date, currency)

# Redis for distributed caching
class DistributedCache:
    async def get_or_compute(self, key, compute_fn):
        cached = await self.redis.get(key)
        if cached:
            return cached

        value = await compute_fn()
        await self.redis.set(key, value, ex=3600)
        return value
```

## Data Flow

### Pricing Workflow

```
User Request
    │
    ├─> Parse Parameters
    │
    ├─> Load Market Data
    │   ├─> Spot prices
    │   ├─> Yield curves
    │   └─> Volatility surfaces
    │
    ├─> Select Model
    │   ├─> Black-Scholes
    │   ├─> Heston
    │   └─> SABR
    │
    ├─> Configure Engine
    │   ├─> Monte Carlo (paths, steps)
    │   └─> PDE (grid size)
    │
    ├─> Price Product
    │   ├─> Simulate paths (MC)
    │   ├─> Calculate payoff
    │   └─> Discount to present
    │
    ├─> Calculate Greeks
    │   ├─> Delta (∂V/∂S)
    │   ├─> Gamma (∂²V/∂S²)
    │   └─> Vega (∂V/∂σ)
    │
    └─> Return Result
        ├─> Price
        ├─> Greeks
        └─> Metrics
```

### Risk Calculation Workflow

```
Portfolio
    │
    ├─> Load Positions
    │
    ├─> Get Market Data
    │   ├─> Current prices
    │   ├─> Historical returns
    │   └─> Correlations
    │
    ├─> Calculate VaR
    │   ├─> Historical simulation
    │   ├─> Monte Carlo
    │   └─> Parametric
    │
    ├─> Calculate Greeks
    │   ├─> Position-level Greeks
    │   └─> Portfolio-level Greeks
    │
    ├─> Check Limits
    │   ├─> Notional limits
    │   ├─> VaR limits
    │   └─> Concentration limits
    │
    └─> Generate Report
        ├─> Risk metrics
        ├─> Limit utilization
        └─> Breach alerts
```

## Scalability & Performance

### Horizontal Scaling

```python
# Distributed pricing across multiple GPUs
@partial(jax.pmap, axis_name='devices')
def distributed_pricing(keys, spots, strikes, maturities, config):
    """Price options in parallel across devices"""
    prices = jax.vmap(price_option)(
        spots, strikes, maturities, config
    )
    return prices

# Split work across devices
num_devices = jax.device_count()
keys = jax.random.split(jax.random.PRNGKey(42), num_devices)
prices = distributed_pricing(keys, spots, strikes, maturities, config)
```

### Vertical Scaling (GPU Acceleration)

```python
# JIT compilation for GPU
@jax.jit
def monte_carlo_pricer(key, S0, K, T, r, sigma, paths):
    """GPU-accelerated Monte Carlo pricing"""
    paths = simulate_gbm(key, S0, r, sigma, T, paths)
    payoff = jnp.maximum(paths[:, -1] - K, 0.0)
    return jnp.mean(payoff) * jnp.exp(-r * T)

# Automatically runs on GPU if available
price = monte_carlo_pricer(key, 100, 100, 1.0, 0.05, 0.20, 100_000)
```

### Caching Strategy

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_yield_curve(date, currency):
    """Cache yield curves to avoid repeated DB queries"""
    return fetch_curve_from_db(date, currency)

# Redis for distributed caching
class DistributedCache:
    async def get_or_compute(self, key, compute_fn):
        cached = await self.redis.get(key)
        if cached:
            return cached

        value = await compute_fn()
        await self.redis.set(key, value, ex=3600)
        return value
```

## Security Architecture

### Authentication & Authorization

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate JWT token and extract user"""
    credentials = decode_jwt(token)
    if not credentials:
        raise HTTPException(status_code=401)
    return User(**credentials)

async def require_permission(permission: str):
    """Check if user has required permission"""
    def dependency(user: User = Depends(get_current_user)):
        if permission not in user.permissions:
            raise HTTPException(status_code=403)
        return user
    return dependency

# Usage in endpoint
@app.post("/price")
async def price_option(
    request: PricingRequest,
    user: User = Depends(require_permission("pricing.execute"))
):
    return pricing_service.price(request)
```

### Audit Logging

```python
class AuditLogger:
    async def log_action(self, user, action, resource, result):
        """Log user actions for audit trail"""
        await self.db.execute("""
            INSERT INTO audit_log (timestamp, user_id, action, resource, result)
            VALUES ($1, $2, $3, $4, $5)
        """, datetime.utcnow(), user.id, action, resource, result)

# Decorator for automatic audit logging
def audited(action: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User, **kwargs):
            result = await func(*args, user=user, **kwargs)
            await audit_logger.log_action(user, action, kwargs, result)
            return result
        return wrapper
    return decorator
```

## Deployment Architecture

### Microservices Architecture

```yaml
# Docker Compose example
version: '3.8'

services:
  pricing-service:
    image: neutryx/pricing:latest
    replicas: 3
    environment:
      - JAX_PLATFORM_NAME=gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1

  risk-service:
    image: neutryx/risk:latest
    replicas: 2

  market-data-service:
    image: neutryx/market-data:latest
    depends_on:
      - timescaledb
      - redis

  timescaledb:
    image: timescale/timescaledb:latest-pg14
    volumes:
      - timescaledb-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neutryx-pricing
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neutryx-pricing
  template:
    metadata:
      labels:
        app: neutryx-pricing
    spec:
      containers:
      - name: pricing
        image: neutryx/pricing:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: JAX_PLATFORM_NAME
          value: "gpu"
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: neutryx-pricing-service
spec:
  selector:
    app: neutryx-pricing
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Next Steps

- [Performance Tuning Guide](performance_tuning.md) - Optimize for speed
- [Troubleshooting Guide](troubleshooting.md) - Debug common issues
- [Developer Guide](developer_guide.md) - Contribute to Neutryx Core
- [API Reference](api_reference.md) - Complete API documentation
