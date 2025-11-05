# 🚀 Neutryx Core — The JAX-Driven Frontier of Quantitative Finance

> **Lightning-fast. Differentiable. Production-Grade.**
> Neutryx Core fuses modern computational science with financial engineering — bringing together
> **JAX**, **automatic differentiation**, and **high-performance computing** to redefine how we price derivatives,
> measure risk, and model complex markets.

<p align="center">
  <img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge"/>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge"/>
  <img alt="JAX Version" src="https://img.shields.io/badge/jax-0.4.26+-orange?style=for-the-badge"/>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge"/>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue?style=for-the-badge"/>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-features">Features</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🌌 Why Neutryx?

Neutryx Core is a **next-generation JAX-first quantitative finance library** —
designed for **investment banks**, **hedge funds**, and **AI-for-science teams** building enterprise-grade
derivatives pricing, risk management, and regulatory compliance systems at scale.

It unifies stochastic models, PDE solvers, market data infrastructure, and regulatory frameworks
into a single, differentiable platform. Every component — from yield curves to XVA calculations —
is **JIT-compiled**, **GPU-accelerated**, and **production-ready**.

> *Complete derivatives lifecycle: From real-time market data to regulatory capital calculation — all within one continuous computational graph.*

---

## ✨ Features

### Core Capabilities

- **Models:** Analytic Black-Scholes, stochastic volatility (Heston, SABR), jump diffusion, rough volatility
- **Products:** Comprehensive multi-asset class coverage including vanilla, exotic, and structured products
  - **Derivatives:** European, Asian, Barrier, Lookback, American (Longstaff-Schwartz)
  - **Equity:** Forwards, dividend swaps, variance swaps, TRS, equity-linked notes
  - **Commodities:** Forwards with convenience yield, options, swaps, spread options
  - **Fixed Income:** Bonds (zero-coupon, coupon, FRN), inflation-linked securities
  - **Interest Rate Derivatives:**
    - Linear: IRS with multi-curve framework, OIS (SOFR/ESTR/SONIA), cross-currency swaps with FX reset, basis swaps (tenor/currency), FRAs, caps/floors/collars
    - Vanilla: Swaptions (European/American/Bermudan)
    - CMS: CMS products, spread options, caplets/floorlets with convexity adjustments
    - Exotic: Range accruals, TARN, snowball notes, autocallable notes, ratchet caps/floors
    - SOFR: Post-LIBOR transition ready with daily compounding
  - **Credit:** CDS pricing, hazard models, structural models
  - **Volatility:** VIX futures/options, variance swaps, corridor swaps, gamma swaps
  - **Convertibles:** Convertible bonds, mandatory convertibles, exchangeable bonds
- **Risk:** Pathwise & bump Greeks, stress testing, VaR, ES, and adjoint-based sensitivity analysis
  - **VaR & Risk Metrics:** Historical simulation, Monte Carlo, parametric VaR, expected shortfall (CVaR), incremental VaR, component VaR
  - **Position Limits:** Notional limits, VaR limits, concentration limits, issuer exposure limits with hierarchical breach thresholds
  - **Pre-Trade Controls:** Real-time limit checking, what-if scenario analysis, approval workflows, limit breach notifications
- **Market:** Multi-curve framework with OIS discounting, tenor basis, FX volatility surfaces
- **XVA:** Exposure models (CVA, DVA, FVA, MVA, KVA) for counterparty risk
- **Calibration:** Differentiable calibration framework with diagnostics and identifiability checks
  - **Model Selection:** Information criteria (AIC, BIC, AICc, HQIC), k-fold and time-series cross-validation
  - **Sensitivity Analysis:** Local sensitivity via finite differences, global Sobol indices with Saltelli sampling
- **Regulatory Reporting:** Enterprise-grade compliance reporting with XML generation
  - **EMIR/Dodd-Frank:** Trade reporting to repositories, lifecycle events, reconciliation, valuation reporting
  - **MiFID II/MiFIR:** Transaction reporting (RTS 22), reference data (RTS 23), best execution analysis
  - **Basel III/IV:** CVA capital, FRTB market risk (SA delta/vega/curvature), operational risk, leverage ratio

### Market Data Infrastructure (NEW)

- **Bloomberg Integration**: Native Bloomberg Terminal/API connectivity with real-time data feeds
- **Refinitiv Integration**: Refinitiv Data Platform (RDP) and Eikon Desktop support
- **Database Storage**:
  - **PostgreSQL**: High-performance relational storage with time-series optimization
  - **MongoDB**: Flexible document storage for heterogeneous market data
  - **TimescaleDB**: Purpose-built time-series database with automatic compression (up to 90%)
- **Data Validation**: Comprehensive quality checks with anomaly detection
  - Price range validation, spread validation, volume spike detection
  - Time-series consistency checks, volatility bounds
  - Real-time quality scoring and reporting
- **Feed Management**: Real-time orchestration with automatic failover and buffering

### Observability & Monitoring (NEW)

- **Prometheus Metrics**: Custom business metrics for pricing, risk, and XVA operations
- **Grafana Dashboards**: Pre-built dashboards for system monitoring and performance analysis
- **Distributed Tracing**: OpenTelemetry integration with Jaeger for request tracing
- **Performance Profiling**: Automatic profiling of slow requests with cProfile
- **Alerting**: Intelligent alerting with configurable thresholds and notification channels

### Technical Highlights

- **JAX-Native:** Full JIT compilation, automatic differentiation, and XLA optimization
- **GPU/TPU Ready:** Seamless acceleration on modern hardware with `pmap`/`pjit`
- **High Performance:** Optimized numerical algorithms with 10-100x speedup for repeated calculations
- **Reproducible:** Unified configuration via YAML, consistent PRNG seeding
- **Production-Ready:** FastAPI/gRPC APIs, comprehensive test suite (370+ tests), quality tooling (ruff, bandit)
- **Enterprise-Grade:**
  - Multi-tenancy controls and RBAC
  - Audit logging and compliance reporting
  - Real-time market data feeds with validation
  - Production monitoring and observability stack
  - SLA monitoring and cost allocation
- **Extensible:** FFI bridges to QuantLib/Eigen, plugin architecture for custom models

---

## ⚡ Quickstart

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest -q
```

### Optional Dependencies

```bash
# For market data integration
pip install asyncpg motor  # PostgreSQL, MongoDB connectors

# For QuantLib integration
pip install -e ".[quantlib]"

# For Eigen bindings
pip install -e ".[eigen]"

# Install all optional dependencies
pip install -e ".[native,marketdata]"
```

### Your First Pricing Example

```python
import jax
import jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value
from neutryx.models.bs import price as bs_price

# Setup
key = jax.random.PRNGKey(42)
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.01, 0.00, 0.2

# Monte Carlo pricing
cfg = MCConfig(steps=252, paths=100_000)
paths = simulate_gbm(key, S, r - q, sigma, T, cfg)
ST = paths[:, -1]
call_mc = present_value(jnp.maximum(ST - K, 0.0), jnp.array(T), r)

# Analytical pricing
call_an = bs_price(S, K, T, r, q, sigma, "call")

print(f"Call (MC): {float(call_mc):.4f}")
print(f"Call (BS): {float(call_an):.4f}")
```

### Real-Time Market Data Pipeline

```python
from neutryx.market.adapters import BloombergAdapter, BloombergConfig
from neutryx.market.storage import TimescaleDBStorage, TimescaleDBConfig
from neutryx.market.validation import ValidationPipeline, PriceRangeValidator
from neutryx.market.feeds import FeedManager

# Configure Bloomberg
bloomberg_config = BloombergConfig(
    adapter_name="bloomberg",
    host="localhost",
    port=8194
)

# Configure TimescaleDB with compression
storage_config = TimescaleDBConfig(
    host="localhost",
    database="market_data",
    compression_enabled=True,
    retention_policy_days=90
)

# Setup validation pipeline
pipeline = ValidationPipeline()
pipeline.add_validator(PriceRangeValidator(max_jump_pct=0.20))

# Create feed manager with automatic failover
adapter = BloombergAdapter(bloomberg_config)
storage = TimescaleDBStorage(storage_config)

manager = FeedManager(
    adapters=[adapter],
    storage=storage,
    validation_pipeline=pipeline
)

# Start real-time feed
await manager.start()
await manager.subscribe("equity", ["AAPL", "MSFT", "GOOGL"])
```

### Multi-Asset Class Pricing

```python
from neutryx.products.equity import equity_forward_price, variance_swap_value
from neutryx.products.commodity import commodity_forward_price
from neutryx.products.inflation import inflation_linked_bond_price
from neutryx.products.volatility import vix_futures_price

# Equity forward with dividends
eq_forward = equity_forward_price(
    spot=100.0, maturity=1.0, risk_free_rate=0.05, dividend_yield=0.02
)

# Commodity forward with convenience yield
commodity_forward = commodity_forward_price(
    spot=50.0, maturity=1.0, risk_free_rate=0.05,
    storage_cost=0.02, convenience_yield=0.03
)

# VIX futures with mean reversion
vix_future = vix_futures_price(
    vix_spot=20.0, maturity=0.25, mean_reversion=1.5, long_term_vol=18.0
)

# Inflation-linked bond (TIPS)
tips_price = inflation_linked_bond_price(
    face_value=1000.0, real_coupon_rate=0.005, real_yield=0.003,
    maturity=10.0, index_ratio=1.25, frequency=2
)
```

### Interest Rate Derivatives (Swaptions & Exotic IR)

```python
from neutryx.products.swaptions import EuropeanSwaption, european_swaption_black
from neutryx.products.advanced_rates import BermudanSwaption, TargetRedemptionNote
from neutryx.products.interest_rate import cms_caplet_price, price_cms_spread_option

# European swaption with Black pricing and Greeks
payer_swaption_price = european_swaption_black(
    strike=0.05, option_maturity=1.0, swap_maturity=5.0,
    volatility=0.20, notional=1_000_000, is_payer=True
)

# Bermudan swaption with Longstaff-Schwartz Monte Carlo
bermudan = BermudanSwaption(
    T=1.0, K=0.05, notional=1_000_000,
    exercise_dates=jnp.array([0.25, 0.5, 0.75, 1.0]),
    option_type='payer', tenor=5.0
)
bermudan_price = bermudan.price_lsm(rate_paths, discount_factors)

# CMS caplet with convexity adjustment
cms_caplet = cms_caplet_price(
    cms_forward=0.04, strike=0.045, time_to_expiry=1.0,
    volatility=0.25, discount_factor=0.97, annuity=9.0,
    convexity_adjustment=0.002
)

# CMS spread option (10Y - 2Y)
spread_option_price = price_cms_spread_option(
    cms1_forward=0.045, cms2_forward=0.035, strike=0.01,
    time_to_expiry=1.0, spread_volatility=0.30,
    discount_factor=0.97, annuity=4.5, is_call=True
)

# Target Redemption Note (TARN)
tarn = TargetRedemptionNote(
    T=5.0, notional=1_000_000, target_coupon=100_000,
    coupon_rate=0.05, payment_freq=4
)
```

---

## 🧭 Project Structure

```text
neutryx-core/
├── .github/              # CI/CD workflows and automation
├── config/               # YAML configuration presets
├── docs/                 # Documentation and guides
│   ├── monitoring.md     # Observability and monitoring guide
│   └── market_data.md    # Market data infrastructure guide
├── demos/                # Examples, dashboards, and tutorials
├── dev/                  # Developer tooling
│   └── monitoring/       # Prometheus, Grafana, Jaeger stack
└── src/neutryx/
    ├── api/              # REST and gRPC services
    ├── core/             # Pricing engines and infrastructure
    ├── infrastructure/   # Observability, governance, workflows
    │   ├── observability/    # Prometheus, tracing, profiling, alerting
    │   └── governance/       # Multi-tenancy, RBAC, compliance
    ├── market/           # Market data and analytics
    │   ├── adapters/     # Bloomberg, Refinitiv integrations
    │   ├── storage/      # PostgreSQL, MongoDB, TimescaleDB
    │   ├── validation/   # Data quality and validation
    │   └── feeds/        # Real-time feed management
    ├── models/           # Stochastic models (BS, Heston, SABR, etc.)
    ├── products/         # Multi-asset class product library
    ├── portfolio/        # Portfolio analytics and optimization
    ├── valuations/       # XVA and exposure analytics
    └── tests/            # 100+ comprehensive tests
```

---

## 📚 Documentation

Comprehensive documentation covering all aspects of Neutryx Core:

- **[docs/overview.md](docs/overview.md)** — High-level introduction and architecture
- **[docs/api_reference.md](docs/api_reference.md)** — Complete API reference
- **[docs/design_decisions.md](docs/design_decisions.md)** — Architecture and design rationale
- **[docs/monitoring.md](docs/monitoring.md)** — Observability and monitoring guide (NEW)
- **[docs/market_data.md](docs/market_data.md)** — Market data infrastructure guide (NEW)
- **[docs/roadmap.md](docs/roadmap.md)** — Development roadmap and milestones

Generate documentation site:

```bash
mkdocs serve  # Local preview at http://127.0.0.1:8000/
mkdocs build  # Static site generation
```

---

## 💻 Examples

### Multi-Asset Class Showcase

```bash
python demos/asset_class_showcase.py
```

Demonstrates:
- **Equity:** Forwards, dividend swaps, variance swaps, TRS
- **Commodities:** Forwards with storage costs, options, swaps
- **Inflation:** TIPS pricing, inflation swaps, breakeven analysis
- **Volatility:** VIX futures, variance swaps, VVIX
- **Convertibles:** Convertible bond analytics

### Swaptions & Exotic Interest Rate Derivatives

```bash
python examples/swaptions_and_exotic_ir_demo.py
```

Demonstrates:
- **European Swaptions:** Black pricing with Greeks (delta, vega), implied volatility
- **American Swaptions:** Trinomial tree pricing with early exercise
- **Bermudan Swaptions:** Longstaff-Schwartz Monte Carlo with regression
- **CMS Products:** Caplets/floorlets with convexity adjustments, CMS swaps
- **CMS Spread Options:** Black formula and Monte Carlo pricing
- **Range Accruals:** Path-dependent accrual based on rate ranges
- **TARN:** Target redemption notes with automatic early redemption
- **Callable/Puttable Bonds:** Multiple exercise dates with optimal strategy
- **Additional Exotics:** Snowball notes, autocallable notes, ratchet caps/floors

### Basic Examples

```bash
python demos/01_bs_vanilla.py        # Vanilla options
python demos/02_path_dependents.py   # Exotic options
python demos/03_american_lsm.py      # American options
python examples/swaptions_and_exotic_ir_demo.py  # Swaptions & exotic IR (NEW)
```

### Dash Dashboard

```bash
cd demos/dashboard && python app.py
```

Interactive pricing, Greeks, and scenario analysis at `http://localhost:8050`

---

## 🗺️ Development Roadmap

> **Building the complete enterprise derivatives platform: From core pricing to AI-driven trading**

### 📅 Roadmap Timeline

```
v0.1.0 (Released) ────────────────────────────────────────────┐
                                                               │
    ✅ Foundation: Multi-asset derivatives, risk, XVA         │
    ✅ 370+ tests, Bloomberg/Refinitiv, Observability         │
                                                               │
v0.2.0 (Q2-Q3 2025) ──────────────────────────────────────────┤
                                                               │
    🎯 Advanced calibration & model enhancements              │
    🎯 Joint calibration, regularization, credit models       │
                                                               │
v0.3.0 (Q4 2025) ─────────────────────────────────────────────┤
                                                               │
    🏗️ Trading platform infrastructure                        │
    🏗️ Lifecycle management, CCP integration, FpML           │
                                                               │
v0.4.0 (Q1 2026) ─────────────────────────────────────────────┤
                                                               │
    📋 Regulatory compliance enhancement                      │
    📋 FRTB, SA-CCR, ISDA SIMM 3.0+, IFRS 9/13              │
                                                               │
v1.0.0 (Q2 2026) ─────────────────────────────────────────────┤
                                                               │
    🚀 Production enterprise platform                         │
    🚀 Security, multi-tenancy, distributed computing        │
                                                               │
v1.x (2026-2027) ─────────────────────────────────────────────┤
                                                               │
    📊 Advanced analytics & AI integration                    │
    📊 Portfolio optimization, backtesting, factor analysis        │
                                                               │

### 🎯 Key Milestones

| Version | Focus | Timeline | Status |
|---------|-------|----------|--------|
| **v0.1.0** | Foundation & Core Pricing | Jan 2025 | ✅ **Released** |
| **v0.2.0** | Advanced Calibration | Q2-Q3 2025 | 🔄 **75% Complete** |
| **v0.3.0** | Trading Infrastructure | Q4 2025 | 🔄 **50% Complete** |
| **v0.4.0** | Regulatory Compliance | Q1 2026 | 🔄 **75% Complete** |
| **v1.0.0** | Enterprise Platform | Q2 2026 | 🔄 **68% Complete** |
| **v1.x** | Analytics & Portfolio | 2026-2027 | 🔮 Future |

---

### ✅ **v0.1.0 — Foundation Release** (Current - Released)

**Status:** Complete and production-ready with 370+ tests

#### Core Capabilities Delivered

**Multi-Asset Class Product Coverage:**
- ✅ **Interest Rate Derivatives** (87 tests)
  - Linear: IRS, OIS (SOFR/ESTR/SONIA), cross-currency swaps, basis swaps, FRAs, caps/floors/collars
  - Swaptions: European, American, Bermudan with LSM Monte Carlo
  - CMS: Products, spread options, caplets/floorlets with convexity adjustments
  - Exotic IR: Range accruals, TARN, snowball notes, autocallable notes, ratchet caps/floors
- ✅ **FX Derivatives** (Complete)
  - Vanilla: Forwards, NDFs, European/American options, digitals
  - Exotic: Barriers (single/double/window), Asians, lookbacks
  - Structured: TARFs, accumulators, FX variance swaps, quanto products
- ✅ **Equity Derivatives** (Complete)
  - Options: European, American, Asian, barrier, lookback, ladder
  - Structured: Autocallables (Phoenix), reverse convertibles, basket options, cliquets
  - Volatility: Variance swaps (index/single-name), correlation swaps, dispersion strategies
- ✅ **Credit Derivatives** (Complete)
  - Single-name: CDS (ISDA model), CDS options, CLNs, recovery locks/swaps
  - Portfolio: CDX/iTraxx indices, index tranches, bespoke CDOs, nth-to-default baskets
- ✅ **Commodity Derivatives** (Complete)
  - Energy: Oil, natural gas, power, spark/dark spreads
  - Metals & Agriculture: Precious/base metals, agricultural commodities, weather derivatives
- ✅ **Fixed Income, Inflation, Volatility, Convertibles**

**Advanced Models & Calibration:**
- ✅ **IR Models:** Hull-White (1F/2F), Black-Karasinski, Cheyette, LGM, LMM/BGM, HJM, CIR, Vasicek
- ✅ **Equity Models:** Local vol (Dupire), Heston, rough vol, jump-diffusion (Merton, Kou, Variance Gamma)
- ✅ **FX Models:** Garman-Kohlhagen, FX Heston, FX SABR, FX Bates, two-factor FX
- ✅ **Credit Models:** Gaussian copula, hazard rate (Jarrow-Turnbull, Duffie-Singleton)
- ✅ **Calibration:** Differentiable framework with diagnostics, model selection (AIC/BIC/AICc/HQIC)
- ✅ **Sensitivity Analysis:** Local finite differences, global Sobol indices

**Risk Management & P&L Attribution:**
- ✅ **VaR:** Historical, Monte Carlo, parametric, ES/CVaR, incremental/component VaR
- ✅ **Stress Testing:** 25+ historical scenarios, hypothetical scenarios, reverse stress testing
- ✅ **Greeks:** DV01, CS01, vega bucketing, FX delta/gamma, higher-order Greeks (vanna, volga, charm, veta, speed, zomma, color)
- ✅ **P&L Attribution:** Daily explain (carry, delta, gamma, vega, theta, rho), risk factor attribution, FRTB test
- ✅ **CCR & XVA:** EE/PFE/EPE profiles, CVA/DVA/FVA/MVA/KVA, collateral optimization, WWR modeling
- ✅ **Position Limits & Pre-Trade Controls:** Hierarchical limits, real-time checking, what-if analysis

**Infrastructure & Operations:**
- ✅ **Market Data:** Bloomberg/Refinitiv integration, PostgreSQL/MongoDB/TimescaleDB storage, validation pipeline
- ✅ **Observability:** Prometheus metrics, Grafana dashboards, Jaeger tracing, automatic profiling
- ✅ **Regulatory Reporting:** EMIR/Dodd-Frank, MiFID II/MiFIR, Basel III/IV (70 tests)
- ✅ **APIs:** REST/gRPC endpoints, interactive dashboards
- ✅ **Performance:** JIT compilation (10-100x speedup), GPU/TPU support, mixed-precision

---

### 🎯 **v0.2.0 — Advanced Calibration & Model Enhancements** (Q2-Q3 2025)

**Focus:** Enhanced calibration techniques, joint calibration, and model stability improvements

#### Advanced Calibration Methods
- ✅ **Joint Calibration Framework**
  - ✅ Multi-instrument simultaneous calibration (e.g., cap/floor + swaption joint calibration)
  - ✅ Cross-asset calibration (FX smile + equity correlation)
  - ✅ Time-dependent parameter fitting with smoothness constraints
  - [ ] Multi-objective optimization with Pareto frontiers

- ✅ **Regularization & Stability**
  - ✅ Tikhonov regularization for ill-posed calibration problems
  - ✅ L1/L2 penalty methods for parameter sparsity
  - ✅ Arbitrage-free constraints enforcement
  - ✅ Smoothness penalties for local volatility surfaces

- ✅ **Advanced Model Selection**
  - ✅ Out-of-sample validation framework
  - ✅ Rolling window backtesting for time-series models
  - [ ] Model combination and averaging (Bayesian model averaging)
  - ✅ Diagnostic suite for calibration quality

#### Model Enhancements
- ✅ **Equity Models**
  - ✅ Time-changed Lévy processes (VG implemented, NIG/CGMY in progress)
  - ✅ Stochastic local volatility (SLV) hybrid models
  - [ ] Jump clustering models

- ✅ **Credit Models**
  - ✅ Student-t copula for tail dependence
  - ✅ Large portfolio approximation (LPA) for CDOs
  - ✅ CreditMetrics framework integration
  - ✅ Structural models (Merton, Black-Cox)

- [ ] **Interest Rate Models**
  - [ ] G2++ (two-factor Gaussian) model
  - [ ] Quasi-Gaussian (QG) models
  - [ ] Cross-currency basis modeling

**Target Release:** Q3 2025
**Key Deliverables:** 50+ new tests, joint calibration framework, enhanced model selection

---

### 🏗️ **v0.3.0 — Trading Platform Infrastructure** (Q4 2025)

**Focus:** Trade lifecycle management, reference data, and clearing/settlement integration

#### Trade Lifecycle Management
- ✅ **Pre-Trade**
  - ✅ Real-time pricing engines for multi-asset classes
  - ✅ Streaming quotes with dynamic refresh (polling-based)
  - [ ] RFQ (Request for Quote) workflow and auction mechanisms
  - ✅ Pre-trade analytics and what-if scenario analysis

- ✅ **Trade Capture**
  - ✅ FpML parsing and generation for all product types
  - ✅ Trade booking workflow with validation
  - ✅ Trade amendment and cancellation handling
  - ✅ Automated trade enrichment (counterparty, legal entity, booking center)

- ✅ **Post-Trade**
  - [ ] Confirmation matching and affirmation
  - [ ] Settlement instruction generation
  - ✅ Payment calculation and netting
  - [ ] Corporate action processing
  - ✅ Novation and assignment workflows

#### Reference Data Management
- [ ] **Security Master**
  - [ ] Centralized security master database
  - ✅ ISIN/CUSIP/SEDOL cross-reference (in vendor adapters)
  - [ ] Corporate actions processing and adjustments
  - ✅ Real-time reference data updates

- ✅ **Market Conventions**
  - ✅ Curve definitions and construction methodologies
  - ✅ Holiday calendars for all major markets (TARGET, US, UK, Japan, joint calendars)
  - ✅ Day count conventions (ACT/360, 30/360, ACT/ACT, etc.)
  - ✅ Business day adjustment rules
  - ✅ Payment and settlement conventions by currency

#### Vendor Integration
- ✅ **Market Data Vendors**
  - ✅ Bloomberg BPIPE integration for ultra-low latency (architecture ready)
  - ✅ Refinitiv RTDS (Real-Time Distribution System) (architecture ready)
  - [ ] ICE Data Services connectivity
  - [ ] CME Market Data direct feeds

- [ ] **CCP Integration**
  - [ ] LCH SwapClear connectivity and trade submission
  - [ ] CME Clearing integration
  - [ ] ICE Clear Credit/Europe
  - [ ] Eurex Clearing

- [ ] **Settlement Systems**
  - [ ] CLS (Continuous Linked Settlement) for FX
  - [ ] Euroclear/Clearstream integration
  - [ ] SWIFT messaging (MT and MX formats)

**Target Release:** Q4 2025
**Key Deliverables:** 80+ new tests, FpML integration, LCH SwapClear connectivity

---

### 🎯 **v0.4.0 — Regulatory Compliance Enhancement** (Q1 2026)

**Focus:** FRTB, SA-CCR, SIMM implementation for full regulatory compliance

#### FRTB (Fundamental Review of the Trading Book)
- ✅ **Standardized Approach (SA)**
  - ✅ Delta risk charge (DRC) calculation by risk class
  - ✅ Vega risk charge with smile risk
  - ✅ Curvature risk charge for non-linear products
  - [ ] Default risk charge (DRC) for credit-sensitive instruments
  - [ ] Residual risk add-on (RRAO) for exotic payoffs

- [ ] **Internal Models Approach (IMA)**
  - [ ] Expected shortfall (ES) at 97.5% confidence level
  - [ ] P&L attribution test (regulatory backtesting)
  - [ ] Backtesting framework with traffic light approach
  - [ ] Non-modellable risk factors (NMRF) identification and treatment

#### SA-CCR (Standardized Approach for Counterparty Credit Risk)
- ✅ **Exposure Calculation**
  - ✅ Replacement cost (RC) for mark-to-market exposure
  - ✅ Potential future exposure (PFE) add-on by asset class
  - ✅ Asset class specific calculations (IR, FX, Credit, Equity, Commodity)
  - ✅ Margined vs unmargined netting set treatment

- ✅ **Trade Bucketing & Hedging**
  - ✅ Maturity buckets and supervisory durations
  - ✅ Hedging set construction with offset recognition
  - ✅ Basis risk recognition and treatment
  - ✅ Cross-currency basis handling

#### Initial Margin (SIMM & UMR)
- ✅ **ISDA SIMM Methodology**
  - ✅ SIMM 2.6 implementation (v3.0+ upgrade needed)
  - ✅ Risk factor sensitivities calculation (delta, vega, curvature)
  - ✅ Correlation matrices by product class
  - ✅ Concentration thresholds and risk weights
  - ✅ Product class calculations (RatesFX, Credit, Equity, Commodity)

- ✅ **UMR Compliance**
  - ✅ Uncleared margin rules (bilateral OTC derivatives)
  - ✅ Variation margin (VM) calculation and dispute resolution
  - ✅ Initial margin (IM) posting and collection workflows
  - ✅ Custodian integration and pledge tracking
  - ✅ Threshold monitoring (AANA and MTA)

#### Accounting Standards
- ✅ **IFRS 9/13 Compliance**
  - ✅ Fair value hierarchy (Level 1/2/3) classification
  - ✅ Valuation adjustments (CVA, DVA, FVA)
  - ✅ Expected Credit Loss (ECL) for derivatives
  - ✅ Hedge effectiveness testing (prospective and retrospective)
  - ✅ Disclosure requirements and financial statement impact

**Target Release:** Q1 2026
**Key Deliverables:** FRTB SA implementation, SA-CCR calculator, ISDA SIMM 3.0+, 100+ new regulatory tests

---

### 🚀 **v1.0.0 — Production Enterprise Platform** (Q2 2026)

**Milestone:** Complete enterprise-grade derivatives platform with full regulatory compliance

#### Enterprise Features
- ✅ **Security & Access Control**
  - [ ] SSO (Single Sign-On) with OAuth 2.0/OpenID Connect
  - ✅ Role-based access control (RBAC) and fine-grained permissions
  - [ ] Multi-factor authentication (MFA)
  - [ ] LDAP/Active Directory integration

- ✅ **Audit & Compliance**
  - ✅ Immutable audit trail with user action tracking
  - [ ] Data lineage and provenance tracking
  - ✅ Maker-checker workflow with 4-eyes principle (generic workflow)
  - ✅ Approval workflows and compliance attestation (reporting framework)

- ✅ **Multi-Tenancy**
  - ✅ Multi-desk/legal entity isolation
  - ✅ Geography-based segregation and data residency (metadata support)
  - ✅ Compute quota management and cost allocation
  - ✅ SLA monitoring and reporting by tenant

#### Collateral Management
- ✅ **Margining & Optimization**
  - ✅ Initial margin calculation (ISDA SIMM)
  - ✅ Variation margin calculation and dispute resolution
  - ✅ Margin call generation with aging analysis
  - ✅ Collateral optimization engine (framework ready)
  - [ ] Collateral transformation strategies
  - ✅ Pledge vs rehypothecation tracking (CSA framework)

#### Performance & Scalability
- ✅ **Distributed Computing**
  - [ ] Kubernetes orchestration with auto-scaling
  - ✅ Risk grid architecture for distributed calculations (framework ready)
  - [ ] Multi-region deployment with disaster recovery
  - ✅ Fault tolerance and automatic recovery (workflow checkpointing)

- ✅ **GPU/TPU Acceleration**
  - ✅ Multi-GPU Monte Carlo with pmap/pjit
  - ✅ PDE solver GPU acceleration
  - ✅ Batch pricing optimization
  - ✅ Parallel Greeks calculation across devices

- ✅ **Algorithmic Improvements**
  - ✅ Adjoint AAD for all product types
  - ✅ Variance reduction (antithetic, control variates, importance sampling)
  - ✅ Quasi-random numbers (Sobol, Halton sequences)
  - ✅ Multilevel Monte Carlo (MLMC)
  - [ ] Adaptive mesh refinement for PDEs

**Target Release:** Q2 2026
**Key Deliverables:** Production-ready with enterprise security, distributed computing, comprehensive test suite (500+ tests)

---

### 📊 **v1.x — Advanced Analytics & AI Integration** (2026-2027)

**Focus:** Portfolio optimization and research tools

#### Portfolio Optimization
- [ ] **Classical Methods**
  - [ ] Mean-variance optimization (Markowitz)
  - [ ] Black-Litterman model with views integration
  - [ ] Risk parity portfolios
  - [ ] Minimum variance and maximum Sharpe ratio

- [ ] **Advanced Optimization**
  - [ ] CVaR/ES optimization for tail risk
  - [ ] Robust optimization with uncertainty sets
  - [ ] Dynamic programming for multi-period allocation
  - [ ] Reinforcement learning for adaptive allocation (PPO, A3C)

#### Research & Backtesting Tools
- [ ] **Strategy Backtesting**
  - [ ] Historical strategy simulation with realistic execution
  - [ ] Walk-forward analysis and optimization
  - [ ] Transaction cost modeling (spread, slippage, market impact)
  - [ ] Performance attribution and risk decomposition

- [ ] **Factor Analysis**
  - [ ] Principal component analysis (PCA) for dimension reduction
  - [ ] Factor risk models (Barra-style)
  - [ ] Style attribution (value, growth, momentum)
  - [ ] Factor timing and allocation

**Target Releases:** v1.1 (Q3 2026), v1.2 (Q4 2026), v1.3 (Q1 2027)

---

## 🧪 Testing

370+ comprehensive tests covering:

- **Unit tests:** Core functionality and model correctness
- **Integration tests:** End-to-end workflows
- **Product tests:** Multi-asset class coverage
  - Linear IR products: IRS, OIS, CCS, basis swaps, FRAs, caps/floors (39 tests)
  - Swaptions and exotic IR: European/American/Bermudan swaptions, CMS, exotic rates (48 tests)
- **Market data tests:** Storage, validation, feed management
- **Calibration tests:** Model selection and sensitivity analysis (24 tests)
- **Risk tests:** VaR methodologies, limits and controls (57 tests)
- **Regression tests:** Numerical stability checks
- **Performance tests:** Benchmarking and profiling

```bash
pytest -v                                    # All tests
pytest src/neutryx/tests/products/ -v       # Product tests
pytest src/neutryx/tests/market/ -v         # Market data tests
pytest --cov=neutryx --cov-report=html      # Coverage report
```

---

## 🤝 Contributing

We welcome contributions from the quant finance community!

### Contribution Areas

- 🎯 **High Priority:**
  - Interest rate derivatives (swaps, swaptions)
  - FX exotics and structured products
  - Credit derivatives and CDOs
  - FRTB and regulatory capital
  - Performance optimizations (GPU/TPU)

- 📊 **Medium Priority:**
  - Additional stochastic models
  - Market microstructure tools
  - Backtesting frameworks
  - ML/AI pricing methods

- 📝 **Always Welcome:**
  - Documentation improvements
  - Bug fixes and tests
  - Examples and tutorials
  - Performance benchmarks

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📋 Version History & Changelog

### **v0.1.0** (Current - Released January 2025)

**Foundation Release** - Production-ready platform with 370+ tests

**Key Achievements:**
- ✅ Multi-asset class product coverage (IR, FX, Equity, Credit, Commodity)
- ✅ Advanced stochastic models (Heston, SABR, Hull-White, LMM, etc.)
- ✅ Comprehensive risk management (VaR, stress testing, P&L attribution)
- ✅ XVA framework (CVA, DVA, FVA, MVA, KVA)
- ✅ Market data infrastructure (Bloomberg, Refinitiv)
- ✅ Observability stack (Prometheus, Grafana, Jaeger)
- ✅ Regulatory reporting (EMIR, MiFID II, Basel III/IV)

**Product Coverage:**
- **Interest Rate:** IRS, OIS, CCS, swaptions (European/American/Bermudan), CMS products, TARN, range accruals, snowball/autocallable notes (87 tests)
- **FX:** Forwards, NDFs, vanilla/digital options, barriers, TARFs, accumulators, FX variance swaps, quantos
- **Equity:** Listed/OTC options, autocallables, basket options, variance/correlation swaps, dispersion strategies
- **Credit:** CDS, CDS options, CDX/iTraxx, index tranches, bespoke CDOs, nth-to-default baskets
- **Commodity:** Energy (oil, gas, power, spreads), metals, agriculture, weather derivatives

**Risk & Analytics:**
- VaR methodologies: Historical, Monte Carlo, parametric, ES/CVaR, incremental/component
- Stress testing: 25+ historical scenarios, hypothetical scenarios, reverse stress testing
- Greeks: DV01, CS01, vega bucketing, higher-order Greeks (vanna, volga, charm, veta, speed, zomma, color)
- P&L attribution: Daily explain (carry, delta, gamma, vega, theta, rho), risk factor attribution, FRTB test
- CCR & XVA: EE/PFE/EPE profiles, CVA/DVA/FVA/MVA/KVA, collateral optimization, WWR modeling

**Infrastructure:**
- Market data: Bloomberg/Refinitiv integration, TimescaleDB (90% compression)
- Observability: Prometheus, Grafana, Jaeger tracing, automatic profiling
- Performance: JIT compilation (10-100x speedup), GPU/TPU support

---

### **v0.2.0** (In Progress - 75% Complete)

**Advanced Calibration & Model Enhancements**

**Completed:**
- ✅ Joint calibration framework (multi-instrument, cross-asset, time-dependent)
- ✅ Regularization techniques (Tikhonov, L1/L2, arbitrage-free constraints, smoothness penalties)
- ✅ Enhanced credit models (Student-t copula, LPA, CreditMetrics, Merton, Black-Cox)
- ✅ Advanced equity models (SLV, Variance Gamma process)
- ✅ Out-of-sample validation and rolling window backtesting

**In Progress:**
- 🔄 Additional Lévy processes (NIG, CGMY)
- 🔄 Jump clustering models
- 🔄 Bayesian model averaging
- 🔄 IR model extensions (G2++, Quasi-Gaussian)

**Delivered:** 50+ new tests, comprehensive calibration framework with production-ready implementations

---

### **v0.3.0** (In Progress - 50% Complete)

**Trading Platform Infrastructure**

**Completed:**
- ✅ FpML parsing and generation (comprehensive support for all product types)
- ✅ Trade booking workflow with validation
- ✅ Trade lifecycle management (amendments, novations, terminations)
- ✅ Market conventions (calendars, day count, business day adjustment)
- ✅ Payment calculation and netting
- ✅ Vendor adapter architecture (Bloomberg, Refinitiv)

**In Progress:**
- 🔄 RFQ workflow and auction mechanisms
- 🔄 Confirmation matching and affirmation
- 🔄 Settlement instruction generation
- 🔄 Corporate action processing
- 🔄 CCP integration (LCH SwapClear, CME Clearing, ICE Clear, Eurex)
- 🔄 Settlement systems (CLS, Euroclear/Clearstream, SWIFT messaging)

**Delivered:** FpML integration, trade lifecycle framework, comprehensive market conventions

---

### **v0.4.0** (In Progress - 75% Complete)

**Regulatory Compliance Enhancement**

**Completed:**
- ✅ FRTB SA delta/vega/curvature charges (full implementation)
- ✅ SA-CCR (RC, PFE add-on, hedging sets, all asset classes)
- ✅ ISDA SIMM 2.6 (delta, vega, curvature, concentration risk)
- ✅ UMR compliance (phase-in, AANA, IM/VM workflows, custodian framework)
- ✅ IFRS 13 (fair value hierarchy, valuation adjustments, Level 3 reconciliation)
- ✅ IFRS 9 (classification, ECL staging, hedge effectiveness testing)
- ✅ Basel III capital ratios and RWA
- ✅ CSA management framework

**In Progress:**
- 🔄 FRTB default risk charge (DRC) and residual risk add-on (RRAO)
- 🔄 FRTB Internal Models Approach (ES 97.5%, P&L attribution, backtesting)
- 🔄 SIMM upgrade to version 3.0+
- 🔄 Comprehensive test coverage for all regulatory modules

**Delivered:** Production-ready regulatory frameworks with SA-CCR, SIMM, UMR, and IFRS 9/13

---

### **v1.0.0** (In Progress - 68% Complete)

**Production Enterprise Platform**

**Completed:**
- ✅ Role-based access control (RBAC) with fine-grained permissions
- ✅ Immutable audit trail with user action tracking
- ✅ Multi-tenancy (desk/entity isolation, quota management, cost allocation, SLA monitoring)
- ✅ Collateral management (ISDA SIMM, VM, margin calls with aging, CSA framework)
- ✅ GPU/TPU acceleration (multi-GPU Monte Carlo, PDE solver acceleration, batch pricing)
- ✅ Algorithmic improvements (adjoint AAD, variance reduction, Sobol/Halton QMC, MLMC)
- ✅ Maker-checker workflow framework
- ✅ Compliance reporting framework

**In Progress:**
- 🔄 SSO with OAuth 2.0/OpenID Connect
- 🔄 Multi-factor authentication (MFA)
- 🔄 LDAP/Active Directory integration
- 🔄 Data lineage and provenance tracking
- 🔄 Kubernetes orchestration and auto-scaling
- 🔄 Multi-region deployment with disaster recovery
- 🔄 Collateral transformation strategies

**Delivered:** Production-ready governance, multi-tenancy, collateral management, and high-performance computing infrastructure

---

### **v1.x** (2026-2027)

**Advanced Analytics & Portfolio Optimization**

- Portfolio optimization (Markowitz, Black-Litterman, risk parity, CVaR optimization)
- Backtesting framework (strategy simulation, transaction costs, performance attribution)
- Factor analysis (PCA, Barra-style risk models, style attribution)

---

## 🔒 Security

- **Vulnerability Reports:** `dev@neutryx.tech`
- **Security Audits:** [docs/security_audit.md](docs/security_audit.md)
- **Dependency Monitoring:** Automated with Dependabot
- **Static Analysis:** Continuous scanning with bandit
- **SBOM:** Generated in CI pipeline

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **JAX Team** at Google Research
- **QuantLib** contributors
- **Bloomberg** and **Refinitiv** for market data APIs
- **SciPy/NumPy** communities
- Quantitative finance practitioners worldwide

---

## 🔗 Links

- **Documentation:** [https://neutryx-lab.github.io/neutryx-core](https://neutryx-lab.github.io/neutryx-core)
- **Repository:** [https://github.com/neutryx-lab/neutryx-core](https://github.com/neutryx-lab/neutryx-core)
- **Issues:** [https://github.com/neutryx-lab/neutryx-core/issues](https://github.com/neutryx-lab/neutryx-core/issues)
- **Website:** [https://neutryx.tech](https://neutryx.tech)

---

<p align="center">
  <strong>Built for Investment Banks, Hedge Funds, and Quantitative Researchers</strong>
</p>

<p align="center">
  <sub>Accelerating quantitative finance with differentiable computing and enterprise-grade infrastructure</sub>
</p>
