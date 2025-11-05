# Neutryx Core - Project Overview

## Vision

Neutryx Core is a **next-generation quantitative finance platform** built to meet the demands of modern investment banks, hedge funds, and quantitative researchers. It unifies derivatives pricing, risk management, and regulatory compliance into a single, high-performance, differentiable computing framework powered by JAX.

Our vision is to provide a **complete derivatives lifecycle platform** - from real-time market data ingestion to regulatory capital calculation - all within one continuous computational graph that is JIT-compiled, GPU-accelerated, and production-ready.

## Core Philosophy

### 1. JAX-First Architecture

Everything in Neutryx Core is built with JAX at its foundation, enabling:

- **Just-In-Time (JIT) Compilation**: 10-100x performance improvements through XLA optimization
- **Automatic Differentiation (AD)**: Accurate Greeks without finite differences
- **Hardware Acceleration**: Seamless GPU/TPU scaling with minimal code changes
- **Functional Programming**: Pure functions enabling reproducibility and parallelization
- **Composability**: Build complex workflows from simple, reusable components

### 2. Production-First Design

Neutryx Core is designed for production use from day one:

- **Type Safety**: Comprehensive type hints throughout the codebase
- **Testing**: 160+ tests covering unit, integration, and regression scenarios
- **Configuration**: YAML-based configuration for reproducible workflows
- **Observability**: Built-in Prometheus metrics, Grafana dashboards, and distributed tracing
- **APIs**: REST and gRPC interfaces for enterprise integration
- **Documentation**: Extensive documentation with examples and tutorials

### 3. Enterprise-Grade Quality

Built for mission-critical financial systems:

- **Security**: Static analysis with bandit, dependency scanning with Dependabot
- **Performance**: Profiling, benchmarking, and optimization tooling
- **Scalability**: Distributed computing support with multi-GPU/TPU capabilities
- **Compliance**: Audit logging, RBAC, multi-tenancy controls
- **Monitoring**: Real-time performance tracking and alerting

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  REST API │ gRPC │ Python SDK │ CLI │ Jupyter Notebooks         │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Pricing  │  │   Risk   │  │   XVA    │  │ Calibr.  │       │
│  │ Services │  │ Services │  │ Services │  │ Services │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                        Domain Layer                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ Models  │  │Products │  │Portfolio│  │ Market  │           │
│  │(BS,Hest)│  │(Options)│  │Analytics│  │  Data   │           │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                      Computation Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   JAX    │  │   PDE    │  │   Monte  │  │   FFT    │       │
│  │  Core    │  │ Solvers  │  │  Carlo   │  │  Methods │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Observab. │  │ Storage  │  │ Security │  │  Config  │       │
│  │(Prom/Jae)│  │(TSDB/PG) │  │(RBAC/Aud)│  │  Mgmt    │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Core Engine (`neutryx.core`)

The computational heart of Neutryx Core:

- **Monte Carlo Engine**: High-performance simulation with variance reduction
- **PDE Solvers**: Crank-Nicolson and finite difference methods
- **Numerical Methods**: FFT, COS method, tree methods
- **Automatic Differentiation**: Adjoint and pathwise Greeks
- **Random Number Generation**: Reproducible PRNG with seeding

**Key Features:**
- JIT compilation for repeated calculations
- Mixed-precision support (float32/float64)
- Vectorized operations for batch pricing
- GPU/TPU acceleration via pmap/pjit

#### 2. Models (`neutryx.models`)

Comprehensive model library:

- **Black-Scholes**: Analytic pricing and Greeks
- **Heston**: Stochastic volatility with FFT/MC pricing
- **SABR**: Stochastic Alpha Beta Rho model
- **Jump Diffusion**: Merton, Kou, Variance Gamma
- **Rough Volatility**: rBergomi and rough Heston
- **Local Volatility**: Dupire PDE
- **Interest Rate Models**: Hull-White, Black-Karasinski (planned)

**Design Principles:**
- Unified interface for all models
- Calibration-ready with differentiable pricing
- Parameter validation and constraints
- Model comparison and selection tools

#### 3. Products (`neutryx.products`)

Multi-asset class product coverage:

**Equity Derivatives:**
- Vanilla options (European, American)
- Exotics (Asian, Barrier, Lookback)
- Forwards, dividend swaps, variance swaps
- Total return swaps, equity-linked notes

**Fixed Income:**
- Bonds (zero-coupon, coupon, FRN)
- Interest rate swaps, caps, floors
- Swaptions (European, American, Bermudan)
- CMS products, range accruals, TARN

**Credit Derivatives:**
- CDS pricing with ISDA model
- Hazard models, structural models
- Index products (planned)

**Commodities:**
- Forwards with storage and convenience yield
- Options, swaps, spread options

**Volatility Products:**
- VIX futures and options
- Variance swaps, corridor swaps
- Gamma swaps

**Design Principles:**
- Consistent pricing interface
- Full Greek calculation support
- Lifecycle event handling
- Corporate action support

#### 4. Risk Management (`neutryx.risk`)

Comprehensive risk analytics:

**VaR Methodologies:**
- Historical simulation VaR
- Monte Carlo VaR
- Parametric VaR with Cornish-Fisher
- Expected Shortfall (ES/CVaR)
- Incremental VaR (IVaR)
- Component VaR

**Position Limits:**
- Notional limits by product/desk
- VaR limits with utilization tracking
- Concentration limits (single-name, sector)
- Issuer exposure limits with credit ratings

**Pre-Trade Controls:**
- Real-time limit checking
- What-if scenario analysis
- Hierarchical breach thresholds (hard/soft/warning)
- Approval workflows

**Stress Testing:**
- Historical scenario analysis
- Hypothetical scenarios
- Reverse stress testing
- Concentration risk metrics

#### 5. XVA Framework (`neutryx.valuations`)

Enterprise XVA calculations:

- **CVA**: Credit Valuation Adjustment
- **DVA**: Debit Valuation Adjustment
- **FVA**: Funding Valuation Adjustment
- **MVA**: Margin Valuation Adjustment
- **KVA**: Capital Valuation Adjustment

**Features:**
- Exposure profile calculation (EE, PFE, EPE)
- Wrong-way risk modeling
- Collateral simulation
- Multi-netting set aggregation
- P&L attribution

#### 6. Market Data (`neutryx.market`)

Real-time market data infrastructure:

**Vendor Integration:**
- Bloomberg Terminal/API
- Refinitiv Data Platform (RDP)
- Refinitiv Eikon Desktop

**Storage Solutions:**
- PostgreSQL: Time-series optimized
- MongoDB: Flexible document storage
- TimescaleDB: Hypertables with 90% compression

**Data Quality:**
- Price range validation
- Spread validation
- Volume spike detection
- Time-series consistency checks
- Real-time quality scoring

**Feed Management:**
- Real-time orchestration
- Automatic failover
- Buffering and rate limiting
- Subscription management

#### 7. Calibration (`neutryx.calibration`)

Advanced calibration framework:

**Methods:**
- Differentiable optimization (Adam, LBFGS)
- Joint calibration across instruments
- Regularization (Tikhonov, L1/L2)
- Constraint handling (bounds, arbitrage-free)

**Model Selection:**
- Information criteria (AIC, BIC, AICc, HQIC)
- Cross-validation (k-fold, time-series)
- Sensitivity analysis (local, global Sobol)
- Identifiability analysis

**Diagnostics:**
- Convergence monitoring
- Parameter uncertainty
- Residual analysis
- Model comparison

#### 8. Infrastructure (`neutryx.infrastructure`)

Enterprise infrastructure components:

**Observability:**
- Prometheus: Custom business metrics
- Grafana: Pre-built dashboards
- OpenTelemetry: Distributed tracing with Jaeger
- Profiling: Automatic performance profiling

**Governance:**
- Multi-tenancy: Desk/entity isolation
- RBAC: Role-based access control
- Audit logging: Immutable audit trail
- Compliance: Regulatory reporting

**Workflows:**
- Task orchestration
- Batch processing
- Scheduled jobs
- Event-driven workflows

## Data Flow

### Pricing Workflow

```
Market Data → Model Calibration → Product Pricing → Risk Calculation
     │              │                    │                │
     ├─ Validation  ├─ Diagnostics      ├─ Greeks        ├─ VaR
     ├─ Storage     ├─ Selection        ├─ Scenarios     ├─ Limits
     └─ Quality     └─ Constraints      └─ Attribution   └─ Reporting
```

### Real-Time Risk Workflow

```
Trade Request → Pre-Trade Check → Limit Validation → Risk Update
     │               │                  │                  │
     ├─ Pricing      ├─ VaR Impact     ├─ Hard Limits    ├─ Dashboard
     ├─ Greeks       ├─ Concentration  ├─ Soft Limits    ├─ Alerts
     └─ Scenarios    └─ Issuer Exp.    └─ Approvals      └─ Reports
```

## Technology Stack

### Core Technologies

- **Language**: Python 3.10+
- **Computation**: JAX 0.4.26+ (XLA, JIT, AD)
- **Numerical**: NumPy, SciPy
- **Data**: Pandas, Polars
- **APIs**: FastAPI, gRPC
- **Configuration**: YAML, Pydantic

### Infrastructure

- **Databases**: PostgreSQL, MongoDB, TimescaleDB
- **Caching**: Redis (planned)
- **Messaging**: RabbitMQ (planned)
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Deployment**: Docker, Kubernetes (planned)

### Development Tools

- **Testing**: pytest, hypothesis
- **Linting**: ruff, mypy
- **Security**: bandit, pip-audit
- **Documentation**: MkDocs, Sphinx
- **CI/CD**: GitHub Actions

## Performance Characteristics

### Benchmarks

**Black-Scholes Pricing (1M options):**
- NumPy: ~500ms
- JAX (CPU): ~50ms (10x faster)
- JAX (GPU): ~5ms (100x faster)

**Monte Carlo Simulation (100K paths, 252 steps):**
- NumPy: ~2000ms
- JAX (CPU): ~200ms (10x faster)
- JAX (GPU): ~20ms (100x faster)

**Heston Calibration (25 data points):**
- scipy.optimize: ~5000ms
- JAX + Adam: ~500ms (10x faster)

### Scalability

- **Batch Pricing**: Linear scaling up to GPU memory limits
- **Distributed**: Multi-GPU support via pmap/pjit
- **Cloud**: Auto-scaling for compute-intensive workloads

## Use Cases

### 1. Front Office Trading

- Real-time option pricing and Greeks
- Scenario analysis and stress testing
- Pre-trade analytics and limit checking
- Structured product design

### 2. Risk Management

- Daily VaR calculation across books
- Position limit monitoring
- Stress testing and scenario analysis
- Regulatory risk reporting

### 3. Model Validation

- Model calibration and backtesting
- Parameter sensitivity analysis
- Model comparison and selection
- Identifiability testing

### 4. Regulatory Compliance

- FRTB standardized approach (planned)
- SA-CCR counterparty risk (planned)
- SIMM initial margin calculation (planned)
- Regulatory reporting (EMIR, Dodd-Frank) (planned)

### 5. Research & Development

- New model development and testing
- Pricing methodology research
- Benchmark comparison
- Academic research

## Extensibility

### Custom Models

Extend the model framework:

```python
from neutryx.models.base import Model
import jax.numpy as jnp

class MyCustomModel(Model):
    def __init__(self, params):
        self.params = params

    def price(self, spot, strike, maturity):
        # Custom pricing logic
        return price

    def calibrate(self, market_data):
        # Custom calibration logic
        return calibrated_params
```

### Custom Products

Define new products:

```python
from neutryx.products.base import Product

class MyExotic(Product):
    def payoff(self, paths):
        # Define exotic payoff
        return payoff

    def price(self, model, market):
        # Pricing logic
        return price
```

### Plugin Architecture

Extend functionality with plugins:

```python
# plugins/my_plugin.py
from neutryx.plugins import Plugin

class MyPlugin(Plugin):
    def initialize(self):
        # Setup logic
        pass

    def execute(self, context):
        # Plugin logic
        pass
```

## Deployment Options

### 1. Standalone Python

```bash
pip install neutryx-core
python my_pricing_script.py
```

### 2. REST API

```bash
uvicorn neutryx.api.rest:app --host 0.0.0.0 --port 8000
```

### 3. gRPC Service

```bash
python -m neutryx.api.grpc.server
```

### 4. Docker Container

```dockerfile
FROM python:3.10
COPY . /app
RUN pip install -e /app
CMD ["uvicorn", "neutryx.api.rest:app", "--host", "0.0.0.0"]
```

### 5. Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neutryx-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: neutryx
        image: neutryx/core:latest
        ports:
        - containerPort: 8000
```

## Roadmap Summary

**Completed (v0.1.0)**:
- Core pricing engines and models
- Multi-asset class products
- Real-time market data infrastructure
- Observability and monitoring
- Risk analytics and limits

**Q2 2025 (v0.2)**: Interest rate derivatives, FX exotics
**Q3 2025 (v0.3)**: Advanced models, enhanced calibration
**Q4 2025 (v0.4)**: FRTB, SA-CCR compliance
**Q1 2026 (v1.0)**: Production enterprise platform

See [roadmap.md](roadmap.md) for detailed milestones.

## Getting Started

Ready to start? Check out these resources:

1. **[Getting Started Guide](getting_started.md)** - Installation and first examples
2. **[Tutorials](tutorials.md)** - Hands-on learning
3. **[API Reference](api_reference.md)** - Complete API documentation
4. **[Architecture Guide](architecture.md)** - Deep dive into system design

## Community and Support

- **Documentation**: [https://neutryx-lab.github.io/neutryx-core](https://neutryx-lab.github.io/neutryx-core)
- **GitHub**: [https://github.com/neutryx-lab/neutryx-core](https://github.com/neutryx-lab/neutryx-core)
- **Issues**: [https://github.com/neutryx-lab/neutryx-core/issues](https://github.com/neutryx-lab/neutryx-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neutryx-lab/neutryx-core/discussions)

## License

Neutryx Core is released under the MIT License. See [LICENSE](../LICENSE) for details.
