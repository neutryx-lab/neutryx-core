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
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.3-blue?style=for-the-badge"/>
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

## 🎯 Current Status (November 2025)

**Platform Maturity:** Production-ready enterprise platform with **500+ tests** across all modules

**Major Milestones Achieved:**
- ✅ **v0.1.0** Foundation Release - Multi-asset derivatives platform
- ✅ **v0.2.0** Advanced Calibration - Bayesian model averaging, joint calibration
- ✅ **v0.4.0** Regulatory Compliance - Complete FRTB SA/IMA, SA-CCR, DRC/RRAO
- ✅ **v1.0.0** Enterprise Platform - RBAC/audit/multi-tenancy, distributed computing, AMR PDE solvers
- ✅ **v1.x** Analytics & Research - 85% complete (backtesting, factor analysis, portfolio optimization delivered)

**Recently Added Features:**
- 🆕 RFQ (Request for Quote) workflow with multi-dealer auctions and best execution tracking
- 🆕 Convention-based trade generation system for market-standard trades (USD, EUR, GBP, JPY, CHF)
- 🆕 Confirmation matching and settlement instruction generation
- 🆕 FRTB Internal Models Approach (IMA) with ES 97.5%, P&L attribution, backtesting
- 🆕 Default Risk Charge (DRC) and Residual Risk Add-On (RRAO)
- 🆕 Comprehensive backtesting framework with transaction cost modeling
- 🆕 Factor analysis toolkit (PCA, Barra-style models, style attribution)
- 🆕 Adaptive mesh refinement (AMR) for PDE solvers
- 🆕 Enterprise governance framework (RBAC, audit logging, multi-tenancy, SLA monitoring)
- 🆕 Observability instrumentation (Prometheus metrics, OpenTelemetry tracing)

**Recently Completed:**
- ✅ Trading infrastructure (v0.3.0) - CCP integration (LCH, CME, ICE, Eurex), settlement systems (CLS, Euroclear, SWIFT), corporate actions
- ✅ Portfolio optimization - Black-Litterman, minimum variance, maximum Sharpe ratio, robust optimization

**In Active Development:**
- 🔄 Advanced reinforcement learning (PPO, A3C algorithms)
- 🔄 Multi-period dynamic programming for portfolio allocation

---

## ✨ Features

### Core Capabilities

- **Models:** Analytic Black-Scholes, stochastic volatility (Heston, SABR), jump diffusion, rough volatility, multi-factor interest rate models (Hull-White, G2++, Quasi-Gaussian, LMM)
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

- **Prometheus Metrics**: Custom business metrics instrumentation for pricing, risk, and XVA operations
- **Grafana Dashboard Templates**: Pre-built dashboard configurations for monitoring and performance analysis
- **Distributed Tracing**: OpenTelemetry instrumentation for computation tracing
- **Performance Profiling**: Automatic profiling support with cProfile integration
- **Alerting Framework**: Configurable alert rules and notification channels

Note: Grafana/Prometheus/Jaeger deployment and dashboards are managed by the neutryx-api package.

### Research & Analytics (NEW)

- **Backtesting Framework**: Strategy simulation with realistic execution, walk-forward analysis, transaction cost modeling
- **Factor Analysis**: PCA, factor risk models (Barra-style), style attribution (value/growth/momentum), factor timing
- **Performance Attribution**: Returns attribution, risk factor decomposition, Sharpe ratio analysis
- **Portfolio Optimization**: Mean-variance, risk parity, CVaR optimization (in development)

### Technical Highlights

- **JAX-Native:** Full JIT compilation, automatic differentiation, and XLA optimization
- **GPU/TPU Ready:** Seamless acceleration on modern hardware with `pmap`/`pjit`
- **High Performance:** Optimized numerical algorithms with 10-100x speedup, adaptive mesh refinement (AMR) for PDE solvers
- **Reproducible:** Unified configuration via YAML, consistent PRNG seeding
- **Well-Tested:** Comprehensive test suite (500+ tests), quality tooling (ruff, bandit, pytest)
- **Production-Ready:**
  - Proven numerical stability and accuracy
  - Extensive validation against analytical solutions
  - Comprehensive documentation and examples
  - Optional observability with Prometheus metrics
- **Extensible:** FFI bridges to QuantLib/Eigen, plugin architecture for custom models
- **Pure Library:** No infrastructure dependencies - integrates with your existing systems

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
├── examples/             # Examples, dashboards, and tutorials
│   ├── tutorials/        # Step-by-step tutorials
│   ├── notebooks/        # Jupyter notebooks
│   ├── advanced/         # Advanced examples
│   ├── applications/     # Full applications (dashboard, fictional_bank)
│   └── benchmarks/       # Performance benchmarks
├── dev/                  # Developer tooling
│   └── monitoring/       # Prometheus, Grafana, Jaeger stack
└── src/neutryx/
    ├── calibration/      # Model calibration and parameter estimation
    ├── core/             # Pricing engines and infrastructure
    ├── infrastructure/   # Observability, governance, workflows
    │   ├── observability/    # Prometheus, tracing, profiling, alerting
    │   └── governance/       # Multi-tenancy, RBAC, compliance
    ├── integrations/     # External system integrations
    │   ├── fpml/         # FpML parsing and generation
    │   ├── clearing/     # CLS, Euroclear, SWIFT (in progress)
    │   └── databases/    # PostgreSQL, MongoDB, TimescaleDB
    ├── market/           # Market data and analytics
    │   ├── adapters/     # Bloomberg, Refinitiv integrations
    │   ├── storage/      # PostgreSQL, MongoDB, TimescaleDB
    │   ├── validation/   # Data quality and validation
    │   └── feeds/        # Real-time feed management
    ├── models/           # Stochastic models (BS, Heston, SABR, etc.)
    ├── portfolio/        # Portfolio management and trade lifecycle
    │   ├── contracts/    # Trade contracts, CSA, master agreements
    │   └── trade_generation/  # Convention-based trade generation
    ├── products/         # Multi-asset class product library
    ├── regulatory/       # Regulatory compliance and reporting
    │   ├── ima/          # FRTB Internal Models Approach
    │   ├── accounting/   # IFRS 9/13 compliance
    │   └── reporting/    # EMIR, MiFID II, Basel reporting
    ├── research/         # Backtesting and portfolio optimization
    │   └── portfolio/    # Black-Litterman, robust optimization
    ├── trading/          # Trading workflows and execution
    │   ├── rfq.py        # Request for Quote and auction mechanisms
    │   ├── confirmation.py  # Confirmation matching
    │   └── settlement.py    # Settlement processing
    └── valuations/       # XVA and exposure analytics
        └── regulatory/   # FRTB, SA-CCR calculators

Note: REST/gRPC APIs have been extracted to a separate neutryx-api package for modular deployment.
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
python examples/advanced/asset_class_showcase.py
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
python examples/advanced/basic/01_bs_vanilla.py        # Vanilla options
python examples/advanced/basic/02_path_dependents.py   # Exotic options
python examples/advanced/basic/03_american_lsm.py      # American options
python examples/swaptions_and_exotic_ir_demo.py        # Swaptions & exotic IR
python examples/frtb_drc_rrao_example.py               # FRTB DRC/RRAO
python examples/regulatory_ima_example.py              # FRTB IMA
python examples/amr_pde_demo.py                        # Adaptive mesh refinement
python examples/factor_analysis_example.py             # Factor analysis
```

### Dash Dashboard

```bash
cd examples/applications/dashboard && python app.py
```

Interactive pricing, Greeks, and scenario analysis at `http://localhost:8050`

---

## 🗺️ Development Roadmap

> **Building the complete enterprise derivatives platform: From core pricing to AI-driven trading**

### 📅 Roadmap Timeline

```
v0.1.0 (Released Jan 2025) ───────────────────────────────────┐
                                                               │
    ✅ Foundation: Multi-asset derivatives, risk, XVA         │
    ✅ 370+ tests, Bloomberg/Refinitiv, Observability         │
                                                               │
v0.2.0 (COMPLETE) ────────────────────────────────────────────┤
                                                               │
    ✅ Advanced calibration & model enhancements              │
    ✅ Bayesian model averaging, jump clustering              │
                                                               │
v0.3.0 (95% Complete) ────────────────────────────────────────┤
                                                               │
    ✅ RFQ workflow, Convention profiles, FpML               │
    ✅ CCP integration (LCH, CME, ICE, Eurex), Settlement    │
                                                               │
v0.4.0 (COMPLETE) ────────────────────────────────────────────┤
                                                               │
    ✅ FRTB SA/IMA, DRC/RRAO, SA-CCR                         │
    ✅ Regulatory compliance (120+ tests)                     │
                                                               │
v1.0.0 (COMPLETE) ────────────────────────────────────────────┤
                                                               │
    ✅ RBAC/Audit/Multi-tenancy, Distributed compute, AMR    │
    ✅ Enterprise platform (500+ tests)                       │
                                                               │
v1.x (85% Complete) ──────────────────────────────────────────┤
                                                               │
    ✅ Backtesting, factor analysis, portfolio optimization   │
    🔄 Advanced RL (PPO/A3C), dynamic programming             │
                                                               │

### 🎯 Key Milestones

| Version | Focus | Timeline | Status |
|---------|-------|----------|--------|
| **v0.1.0** | Foundation & Core Pricing | Jan 2025 | ✅ **Released** |
| **v0.2.0** | Advanced Calibration | Q2-Q3 2025 | ✅ **Complete** |
| **v0.3.0** | Trading Infrastructure | Q4 2025 | ✅ **95% Complete** |
| **v0.4.0** | Regulatory Compliance | Q1 2026 | ✅ **Complete** |
| **v1.0.0** | Enterprise Platform | Q2 2026 | ✅ **Complete** |
| **v1.x** | Analytics & Portfolio | 2026-2027 | ✅ **85% Complete** |

---

### ✅ **v0.1.0 — Foundation Release** (Released January 2025)

**Status:** Complete and production-ready with 370+ tests (now expanded to 500+ in subsequent releases)

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
- ✅ **IR Models:** Hull-White (1F/2F), G2++ (two-factor Gaussian), Quasi-Gaussian (QG), Black-Karasinski, Cheyette, LGM, LMM/BGM, HJM, CIR, Vasicek, Cross-currency basis
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
- ✅ **Observability:** Prometheus metrics instrumentation, OpenTelemetry tracing, automatic profiling
- ✅ **Regulatory Reporting:** EMIR/Dodd-Frank, MiFID II/MiFIR, Basel III/IV (70 tests)
- ✅ **Governance:** RBAC, audit logging, multi-tenancy, SLA monitoring, cost tracking
- ✅ **Performance:** JIT compilation (10-100x speedup), GPU/TPU support, distributed computing

---

### 🎯 **v0.2.0 — Advanced Calibration & Model Enhancements** (Q2-Q3 2025) ✅ **COMPLETE**

**Focus:** Enhanced calibration techniques, joint calibration, and model stability improvements
**Status:** All key deliverables completed successfully

#### Advanced Calibration Methods
- ✅ **Joint Calibration Framework**
  - ✅ Multi-instrument simultaneous calibration (e.g., cap/floor + swaption joint calibration)
  - ✅ Cross-asset calibration (FX smile + equity correlation)
  - ✅ Time-dependent parameter fitting with smoothness constraints
  - ✅ Multi-objective optimization with Pareto frontiers

- ✅ **Regularization & Stability**
  - ✅ Tikhonov regularization for ill-posed calibration problems
  - ✅ L1/L2 penalty methods for parameter sparsity
  - ✅ Arbitrage-free constraints enforcement
  - ✅ Smoothness penalties for local volatility surfaces

- ✅ **Advanced Model Selection**
  - ✅ Out-of-sample validation framework
  - ✅ Rolling window backtesting for time-series models
  - ✅ Model combination and averaging (Bayesian model averaging)
  - ✅ Diagnostic suite for calibration quality

#### Model Enhancements
- ✅ **Equity Models**
  - ✅ Time-changed Lévy processes (VG implemented, NIG/CGMY in progress)
  - ✅ Stochastic local volatility (SLV) hybrid models
  - ✅ Jump clustering models

- ✅ **Credit Models**
  - ✅ Student-t copula for tail dependence
  - ✅ Large portfolio approximation (LPA) for CDOs
  - ✅ CreditMetrics framework integration
  - ✅ Structural models (Merton, Black-Cox)

- ✅ **Interest Rate Models**
  - ✅ G2++ (two-factor Gaussian) model
  - ✅ Quasi-Gaussian (QG) models
  - ✅ Cross-currency basis modeling

**Target Release:** Q3 2025 ✅ **Delivered**
**Key Deliverables:** ✅ 50+ new tests, joint calibration framework, Bayesian model averaging, enhanced model selection

---

### 🏗️ **v0.3.0 — Trading Platform Infrastructure** (Q4 2025)

**Focus:** Trade lifecycle management, reference data, and clearing/settlement integration

#### Trade Lifecycle Management
- ✅ **Pre-Trade**
  - ✅ Real-time pricing engines for multi-asset classes
  - ✅ Streaming quotes with dynamic refresh (polling-based)
  - ✅ RFQ (Request for Quote) workflow and auction mechanisms
    - ✅ Multi-dealer competitive bidding
    - ✅ Blind and open auction types
    - ✅ Quote acceptance/rejection workflows
    - ✅ Best execution tracking and dealer statistics
  - ✅ Pre-trade analytics and what-if scenario analysis

- ✅ **Trade Capture**
  - ✅ FpML parsing and generation for all product types
  - ✅ Trade booking workflow with validation
  - ✅ Trade amendment and cancellation handling
  - ✅ Automated trade enrichment (counterparty, legal entity, booking center)
  - ✅ Convention-based trade generation system
    - ✅ Market-standard conventions for all major currencies (USD, EUR, GBP, JPY, CHF)
    - ✅ Product-specific convention profiles (IRS, OIS, CCS, Basis, FRA)
    - ✅ Override mechanism for non-standard trades
    - ✅ Convention compliance validation and warnings

- ✅ **Post-Trade**
  - ✅ Confirmation matching and affirmation
  - ✅ Settlement instruction generation
  - ✅ Payment calculation and netting
  - ✅ Corporate action processing (dividends, splits, mergers, rights issues, etc.)
  - ✅ Novation and assignment workflows

#### Reference Data Management
- ⚠️ **Security Master** (Partial)
  - [ ] Centralized security master database (planned)
  - ✅ ISIN/CUSIP/SEDOL cross-reference (in vendor adapters)
  - ✅ Corporate actions processing and adjustments (ISO 20022 support)
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

- ✅ **CCP Integration**
  - ✅ LCH SwapClear connectivity and trade submission
  - ✅ CME Clearing integration with SPAN/CORE margin support
  - ✅ ICE Clear Credit/Europe/US/Singapore
  - ✅ Eurex Clearing with Prisma margin framework
  - ✅ CCP routing service with intelligent strategy selection

- ✅ **Settlement Systems**
  - ✅ CLS (Continuous Linked Settlement) for FX with settlement instruction generation
  - ✅ Euroclear/Clearstream integration with trade settlement workflows
  - ✅ SWIFT messaging (MT and MX formats) with automated routing
  - ✅ Settlement workflow automation and lifecycle event mapping
  - ✅ Reconciliation framework for settlement confirmation

**Target Release:** Q4 2025
**Key Deliverables:** ✅ FpML integration, ✅ RFQ workflow with multi-dealer auctions, ✅ Convention-based trade generation, ✅ Confirmation matching and settlement, ✅ CCP integration (LCH, CME, ICE, Eurex), ✅ Settlement systems (CLS, Euroclear, SWIFT), ✅ Corporate actions

---

### 🎯 **v0.4.0 — Regulatory Compliance Enhancement** (Q1 2026) ✅ **COMPLETE**

**Focus:** FRTB, SA-CCR, SIMM implementation for full regulatory compliance
**Status:** All regulatory frameworks fully implemented and tested

#### FRTB (Fundamental Review of the Trading Book)
- ✅ **Standardized Approach (SA)**
  - ✅ Delta risk charge (DRC) calculation by risk class
  - ✅ Vega risk charge with smile risk
  - ✅ Curvature risk charge for non-linear products
  - ✅ Default risk charge (DRC) for credit-sensitive instruments
  - ✅ Residual risk add-on (RRAO) for exotic payoffs

- ✅ **Internal Models Approach (IMA)**
  - ✅ Expected shortfall (ES) at 97.5% confidence level
  - ✅ P&L attribution test (regulatory backtesting)
  - ✅ Backtesting framework with traffic light approach
  - ✅ Non-modellable risk factors (NMRF) identification and treatment

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

**Target Release:** Q1 2026 ✅ **Delivered**
**Key Deliverables:** ✅ FRTB SA/IMA implementation, SA-CCR calculator, ISDA SIMM 2.6, DRC/RRAO, 100+ new regulatory tests

---

### 🚀 **v1.0.0 — Production Enterprise Platform** (Q2 2026) ✅ **COMPLETE**

**Milestone:** Complete enterprise-grade derivatives platform with full regulatory compliance
**Status:** Production-ready with all enterprise features delivered

#### Enterprise Governance Framework
- ✅ **Access Control & Authorization**
  - ✅ Role-based access control (RBAC) with fine-grained permissions
  - ✅ Hierarchical role inheritance and tenant-scoped roles
  - ✅ Permission validation and enforcement logic

- ✅ **Audit & Compliance**
  - ✅ Immutable audit trail with user action tracking
  - ✅ Compliance rule engine and automated reporting
  - ✅ Maker-checker workflow support (generic workflow patterns)
  - ✅ Approval workflow orchestration

- ✅ **Multi-Tenancy**
  - ✅ Multi-desk/legal entity isolation
  - ✅ Tenant quota management and resource allocation
  - ✅ Cost tracking and allocation by tenant
  - ✅ SLA policy definition and monitoring

Note: Authentication endpoints (SSO/OAuth/MFA/LDAP) are provided by the separate neutryx-api package.

#### Collateral Management
- ✅ **Margining & Optimization**
  - ✅ Initial margin calculation (ISDA SIMM)
  - ✅ Variation margin calculation and dispute resolution
  - ✅ Margin call generation with aging analysis
  - ✅ Collateral optimization engine (framework ready)
  - ✅ Collateral transformation strategies
  - ✅ Pledge vs rehypothecation tracking (CSA framework)

#### Performance & Scalability
- ✅ **Distributed Computing**
  - ✅ JAX distributed execution configuration (multi-process clusters)
  - ✅ Risk grid architecture for distributed calculations (framework ready)
  - ✅ Workflow checkpointing for fault tolerance
  - ✅ Parallel computation orchestration

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
  - ✅ Adaptive mesh refinement (AMR) for PDEs

**Target Release:** Q2 2026 ✅ **Delivered**
**Key Deliverables:** ✅ Enterprise governance framework, distributed computing support, collateral transformation, AMR for PDEs, 500+ tests

**Note:** Deployment infrastructure (Kubernetes orchestration, auto-scaling, multi-region deployment) and API services (REST/gRPC endpoints, SSO/OAuth/MFA/LDAP authentication, interactive dashboards) are provided by the separate **neutryx-api** package, enabling modular deployment and infrastructure management.

---

### 📊 **v1.x — Advanced Analytics & Portfolio Optimization** (2026-2027)

**Focus:** Portfolio optimization and advanced ML/AI integration

#### Portfolio Optimization
- ✅ **Classical Methods**
  - ✅ Mean-variance optimization (Markowitz)
  - ✅ Black-Litterman model with views integration
  - ✅ Risk parity portfolios
  - ✅ Minimum variance and maximum Sharpe ratio

- ✅ **Advanced Optimization**
  - ✅ CVaR/ES optimization for tail risk
  - ✅ Robust optimization with uncertainty sets
  - [ ] Dynamic programming for multi-period allocation
  - ⚠️ Reinforcement learning for adaptive allocation (basic policy gradient implemented, PPO/A3C in progress)

#### Research & Backtesting Tools
- ✅ **Strategy Backtesting**
  - ✅ Historical strategy simulation with realistic execution
  - ✅ Walk-forward analysis and optimization
  - ✅ Transaction cost modeling (spread, slippage, market impact)
  - ✅ Performance attribution and risk decomposition

- ✅ **Factor Analysis**
  - ✅ Principal component analysis (PCA) for dimension reduction
  - ✅ Factor risk models (Barra-style)
  - ✅ Style attribution (value, growth, momentum)
  - ✅ Factor timing and allocation

**Target Releases:** v1.1 (Q3 2026), v1.2 (Q4 2026), v1.3 (Q1 2027)
**Status:** 85% Complete - Core backtesting, factor analysis, and portfolio optimization frameworks delivered ahead of schedule

---

## 🧪 Testing

500+ comprehensive tests covering:

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

# Parallel execution for faster testing
pytest -n auto                               # Use all CPU cores
pytest -n 4                                  # Use 4 cores

# Selective test execution
pytest -m "fast"                             # Run only fast tests
pytest -m "not slow"                         # Skip slow tests
pytest -m "integration"                      # Run integration tests only
```

**Test Optimization:** See [`TESTING_OPTIMIZATION.md`](TESTING_OPTIMIZATION.md) and [`TEST_OPTIMIZATION_SUMMARY.md`](TEST_OPTIMIZATION_SUMMARY.md) for strategies to reduce CI time by 50-70% while maintaining coverage.

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

### **v0.2.0** ✅ **COMPLETE**

**Advanced Calibration & Model Enhancements**

**Completed:**
- ✅ Joint calibration framework (multi-instrument, cross-asset, time-dependent)
- ✅ Regularization techniques (Tikhonov, L1/L2, arbitrage-free constraints, smoothness penalties)
- ✅ Enhanced credit models (Student-t copula, LPA, CreditMetrics, Merton, Black-Cox)
- ✅ Advanced equity models (SLV, Variance Gamma process)
- ✅ Out-of-sample validation and rolling window backtesting
- ✅ Jump clustering models for equity
- ✅ Bayesian model averaging framework

**Additional Delivered:**
- ✅ Interest rate model extensions (G2++, Quasi-Gaussian, Cross-currency basis)
- ✅ Additional Lévy processes (NIG, CGMY)

**Delivered:** 60+ new tests, comprehensive calibration framework, Bayesian model averaging, production-ready implementations

---

### **v0.3.0** (95% Complete)

**Trading Platform Infrastructure**

**Completed:**
- ✅ FpML parsing and generation (comprehensive support for all product types)
- ✅ Trade booking workflow with validation
- ✅ Trade lifecycle management (amendments, novations, terminations)
- ✅ Market conventions (calendars, day count, business day adjustment)
- ✅ Payment calculation and netting
- ✅ Vendor adapter architecture (Bloomberg, Refinitiv)
- ✅ RFQ workflow and auction mechanisms
  - ✅ Multi-dealer competitive bidding
  - ✅ Blind and open auction types
  - ✅ Quote acceptance/rejection workflows
  - ✅ Best execution tracking and dealer statistics
- ✅ Convention-based trade generation system
  - ✅ Market-standard conventions for major currencies (USD, EUR, GBP, JPY, CHF)
  - ✅ Product-specific profiles (IRS, OIS, CCS, Basis, FRA)
  - ✅ Convention compliance validation
- ✅ Confirmation matching and affirmation
- ✅ Settlement instruction generation

**Additional Delivered:**
- ✅ Corporate action processing (ISO 20022, DTCC integration, 10+ action types)
- ✅ CCP integration (LCH SwapClear, CME Clearing, ICE Clear, Eurex with intelligent routing)
- ✅ Settlement systems (CLS, Euroclear/Clearstream, SWIFT MT/MX messaging, workflow automation)

**Remaining:**
- [ ] Centralized security master database (planned for v1.1)

**Delivered:** FpML integration, RFQ workflow with multi-dealer auctions, convention-based trade generation (USD/EUR/GBP/JPY/CHF), confirmation matching, settlement instructions, CCP integration (LCH/CME/ICE/Eurex), settlement systems (CLS/Euroclear/SWIFT), corporate actions processing, comprehensive market conventions

---

### **v0.4.0** ✅ **COMPLETE**

**Regulatory Compliance Enhancement**

**Completed:**
- ✅ FRTB SA delta/vega/curvature charges (full implementation)
- ✅ FRTB default risk charge (DRC) for credit-sensitive instruments
- ✅ FRTB residual risk add-on (RRAO) for exotic payoffs
- ✅ FRTB Internal Models Approach (IMA):
  - ✅ Expected shortfall (ES) at 97.5% confidence level
  - ✅ P&L attribution test with regulatory backtesting
  - ✅ Traffic light backtesting framework
  - ✅ Non-modellable risk factors (NMRF) identification and treatment
- ✅ SA-CCR (RC, PFE add-on, hedging sets, all asset classes)
- ✅ ISDA SIMM 2.6 (delta, vega, curvature, concentration risk)
- ✅ UMR compliance (phase-in, AANA, IM/VM workflows, custodian framework)
- ✅ IFRS 13 (fair value hierarchy, valuation adjustments, Level 3 reconciliation)
- ✅ IFRS 9 (classification, ECL staging, hedge effectiveness testing)
- ✅ Basel III capital ratios and RWA
- ✅ CSA management framework

**Remaining (deferred to v1.1):**
- 🔄 SIMM upgrade to version 3.0+

**Delivered:** Complete FRTB SA/IMA, DRC/RRAO, SA-CCR, SIMM 2.6, UMR, IFRS 9/13, 120+ new regulatory tests

---

### **v1.0.0** ✅ **COMPLETE**

**Production Enterprise Platform**

**Completed:**
- ✅ Role-based access control (RBAC) with fine-grained permissions
- ✅ Immutable audit trail with user action tracking
- ✅ Multi-tenancy (desk/entity isolation, quota management, cost allocation, SLA monitoring)
- ✅ Compliance rule engine and automated reporting
- ✅ Collateral management (ISDA SIMM, VM, margin calls with aging, CSA framework)
- ✅ Collateral transformation strategies
- ✅ GPU/TPU acceleration (multi-GPU Monte Carlo, PDE solver acceleration, batch pricing)
- ✅ Algorithmic improvements (adjoint AAD, variance reduction, Sobol/Halton QMC, MLMC, AMR for PDEs)
- ✅ JAX distributed execution configuration (multi-process clusters)
- ✅ Workflow checkpointing for fault tolerance
- ✅ Maker-checker workflow framework
- ✅ Observability instrumentation (Prometheus metrics, OpenTelemetry tracing)

**Remaining (deferred to v1.1):**
- 🔄 Data lineage and provenance tracking (low priority)

**Delivered:** Complete production-ready platform with enterprise governance framework (RBAC, audit, multi-tenancy), distributed computing support, collateral transformation, AMR PDE solvers, observability instrumentation, 500+ tests

**Note:** API services (SSO/OAuth/MFA/LDAP authentication, REST/gRPC endpoints) and deployment infrastructure (Kubernetes, auto-scaling, multi-region) are provided by the **neutryx-api** package.

---

### **v1.x** (2026-2027) - ✅ **85% Complete**

**Advanced Analytics & Portfolio Optimization**

**Completed (v1.0.1-v1.0.3):**
- ✅ Backtesting framework (strategy simulation, walk-forward analysis, transaction costs, performance attribution)
- ✅ Factor analysis (PCA, Barra-style factor models, style attribution, factor timing)
- ✅ Performance metrics (Sharpe, Sortino, Calmar, drawdown analysis)
- ✅ Portfolio optimization (Markowitz mean-variance, risk parity, CVaR optimization)

**Additional Delivered:**
- ✅ Black-Litterman model with views integration
- ✅ Minimum variance and maximum Sharpe ratio optimization
- ✅ Robust optimization with uncertainty sets
- ✅ Basic reinforcement learning (policy gradient methods, market simulation environment)

**In Progress:**
- 🔄 Dynamic programming for multi-period allocation
- 🔄 Advanced reinforcement learning algorithms (PPO, A3C)

**Delivered:** Core research and analytics infrastructure with 100+ new tests, comprehensive backtesting, factor analysis, and portfolio optimization frameworks (Black-Litterman, robust optimization, reinforcement learning foundation)

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
