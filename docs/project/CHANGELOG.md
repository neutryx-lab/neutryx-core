# Changelog

All notable changes to Neutryx Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (v1.1 - Q3 2026)
- Black-Litterman model with views integration
- Minimum variance and maximum Sharpe ratio optimization
- Robust optimization with uncertainty sets
- Dynamic programming for multi-period allocation
- Reinforcement learning for adaptive allocation (PPO, A3C)
- Data lineage and provenance tracking
- SIMM upgrade to version 3.0+

### Planned (v0.3.0 completion - Q4 2025)
- RFQ workflow and auction mechanisms
- CCP integration (LCH SwapClear, CME, ICE Clear, Eurex)
- Settlement systems (CLS, Euroclear/Clearstream, SWIFT messaging)
- Additional Lévy processes (NIG, CGMY)
- IR model extensions (G2++, Quasi-Gaussian)

---

## [1.0.3] - 2025-11 (Latest)

### Added - Portfolio Optimization & Research Tools
- **Portfolio Optimization**
  - Mean-variance optimization (Markowitz) with efficient frontier
  - Risk parity portfolios with iterative risk balancing
  - CVaR/ES optimization for tail risk management
  - Portfolio rebalancing and performance tracking

- **Factor Analysis**
  - Principal component analysis (PCA) for dimension reduction
  - Barra-style factor risk models with multi-factor decomposition
  - Style attribution (value, growth, momentum, size, quality)
  - Factor timing and allocation strategies

- **Backtesting Framework**
  - Historical strategy simulation with realistic execution
  - Walk-forward analysis and optimization
  - Transaction cost modeling (spread, slippage, market impact)
  - Performance attribution and risk decomposition
  - Sharpe, Sortino, Calmar ratios and drawdown analysis

### Tests
- 80+ new tests for portfolio optimization and analytics
- Comprehensive backtesting validation
- Factor model accuracy tests

---

## [1.0.0] - 2025-06 (Major Release)

### Added - Production Enterprise Platform

**Enterprise Security & Access Control**
- SSO (Single Sign-On) with OAuth 2.0/OpenID Connect
- Fine-grained role-based access control (RBAC)
- Multi-factor authentication (MFA)
- LDAP/Active Directory integration
- API key and JWT token management

**Audit & Compliance**
- Immutable audit trail with user action tracking
- Maker-checker workflow with 4-eyes principle
- Approval workflows and compliance attestation
- Regulatory reporting framework (EMIR, MiFID II, Basel)

**Multi-Tenancy**
- Multi-desk/legal entity isolation
- Geography-based data segregation and residency
- Compute quota management and cost allocation
- SLA monitoring and reporting by tenant

**Collateral Management**
- ISDA SIMM initial margin calculation
- Variation margin calculation and dispute resolution
- Margin call generation with aging analysis
- Collateral optimization engine
- Collateral transformation strategies
- Pledge vs rehypothecation tracking with CSA framework

**Distributed Computing & Scalability**
- Kubernetes deployment support with auto-scaling
- Risk grid architecture for distributed calculations
- Multi-region deployment with disaster recovery
- Fault tolerance and automatic recovery (workflow checkpointing)
- Health checks and graceful degradation

**GPU/TPU Acceleration**
- Multi-GPU Monte Carlo simulation with pmap/pjit
- PDE solver GPU acceleration
- Batch pricing optimization across devices
- Parallel Greeks calculation on GPU/TPU

**Algorithmic Improvements**
- Adjoint automatic differentiation (AAD) for all product types
- Variance reduction techniques (antithetic, control variates, importance sampling)
- Quasi-random number generation (Sobol, Halton sequences)
- Multilevel Monte Carlo (MLMC) for faster convergence
- Adaptive Mesh Refinement (AMR) for PDE solvers with error-driven refinement

### Changed
- Enhanced API security with JWT and OAuth 2.0
- Improved observability with distributed tracing correlation IDs
- Optimized memory usage in Monte Carlo simulations

### Tests
- Reached 500+ comprehensive tests across all modules
- Added enterprise security integration tests
- K8s deployment tests
- GPU/TPU performance benchmarks

---

## [0.4.0] - 2026-01

### Added - Regulatory Compliance

**FRTB (Fundamental Review of the Trading Book)**
- **Standardized Approach (SA)**
  - Delta risk charge by risk class (IR, FX, Equity, Credit, Commodity)
  - Vega risk charge with smile risk
  - Curvature risk charge for non-linear products
  - Default Risk Charge (DRC) for credit-sensitive instruments
  - Residual Risk Add-On (RRAO) for exotic payoffs and behavioral risks

- **Internal Models Approach (IMA)**
  - Expected Shortfall (ES) at 97.5% confidence level
  - P&L attribution test for regulatory backtesting
  - Traffic light backtesting framework with breach thresholds
  - Non-modellable risk factors (NMRF) identification and treatment
  - Model risk capital add-ons

**SA-CCR (Standardized Approach for Counterparty Credit Risk)**
- Replacement cost (RC) calculation for mark-to-market exposure
- Potential future exposure (PFE) add-on by asset class
- Asset class specific calculations (IR, FX, Credit, Equity, Commodity)
- Margined vs unmargined netting set treatment
- Maturity buckets and supervisory durations
- Hedging set construction with offset recognition
- Basis risk and cross-currency basis handling

**Initial Margin (SIMM & UMR)**
- ISDA SIMM 2.6 implementation
- Risk factor sensitivities (delta, vega, curvature)
- Correlation matrices by product class
- Concentration thresholds and risk weights
- Product class calculations (RatesFX, Credit, Equity, Commodity)
- Uncleared Margin Rules (UMR) compliance
- IM/VM posting and collection workflows
- Custodian integration and pledge tracking
- AANA and MTA threshold monitoring

**Accounting Standards**
- IFRS 9 classification and measurement for derivatives
- Expected Credit Loss (ECL) calculation with staging
- Hedge effectiveness testing (prospective and retrospective)
- IFRS 13 fair value hierarchy (Level 1/2/3) classification
- Valuation adjustments documentation (CVA, DVA, FVA)
- Financial statement disclosure requirements

### Tests
- 120+ new regulatory compliance tests
- FRTB SA/IMA validation tests
- SA-CCR exposure calculation tests
- SIMM sensitivities and IM calculation tests
- IFRS 9/13 accounting tests

---

## [0.2.0] - 2025-04

### Added - Advanced Calibration & Model Enhancements

**Joint Calibration Framework**
- Multi-instrument simultaneous calibration (caps/floors + swaptions)
- Cross-asset calibration (FX smile + equity correlation)
- Time-dependent parameter fitting with smoothness constraints
- Weighted least squares with heteroscedastic error handling

**Regularization & Stability**
- Tikhonov regularization for ill-posed problems
- L1/L2 penalty methods for parameter sparsity
- Arbitrage-free constraint enforcement
- Smoothness penalties for local volatility surfaces
- Positivity constraints for variance parameters

**Advanced Model Selection**
- Out-of-sample validation framework
- Rolling window backtesting for time-series models
- Bayesian model averaging for robust predictions
- Model combination with optimal weights
- Comprehensive diagnostic suite for calibration quality
- Parameter uncertainty quantification

**Enhanced Models**
- Stochastic Local Volatility (SLV) hybrid models
- Variance Gamma process with time-change
- Jump clustering models for equity
- Student-t copula for credit tail dependence
- Large Portfolio Approximation (LPA) for CDOs
- CreditMetrics framework integration
- Merton and Black-Cox structural credit models

### Tests
- 60+ new calibration and model tests
- Joint calibration validation
- Bayesian averaging accuracy tests
- Model selection cross-validation tests

---

## [0.1.0] - 2025-01 (Initial Release)

### Added - Foundation

**Core Infrastructure**
- Monte Carlo simulation engine with JAX JIT compilation
- PDE solvers (Crank-Nicolson with boundary conditions)
- Automatic differentiation framework with Hessian-vector products
- GPU/TPU optimization with pmap/pjit
- Mixed-precision support for efficient computation
- YAML-based configuration system
- Reproducible PRNG seeding across Python, NumPy, and JAX

**Models**
- Black-Scholes analytical pricing and Greeks
- Geometric Brownian Motion (GBM) simulation
- Heston stochastic volatility model
- SABR volatility model
- Jump diffusion models (Merton, Kou)
- Variance Gamma process
- Tree-based pricing methods
- Hull-White interest rate models (1F/2F)
- Black-Karasinski model
- Cheyette, LGM, LMM/BGM, HJM, CIR, Vasicek IR models
- FX models (Garman-Kohlhagen, FX Heston, FX SABR, FX Bates)
- Gaussian copula credit models

**Products**
- Vanilla European options (call/put)
- American options via Longstaff-Schwartz Monte Carlo
- Asian options (arithmetic/geometric average)
- Barrier options (up/down, in/out, window)
- Lookback options (fixed/floating strike)
- Interest rate swaps (IRS, OIS, cross-currency, basis swaps)
- Caps, floors, collars, FRAs
- Swaptions (European, American, Bermudan)
- CMS products with convexity adjustments
- Exotic IR: Range accruals, TARN, snowball notes, autocallable notes
- CDS pricing with ISDA model
- Equity forwards, dividend swaps, variance swaps, TRS
- Commodity forwards with storage and convenience yield
- VIX futures and variance swaps
- Inflation-linked bonds (TIPS)

**Risk & Analytics**
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Higher-order Greeks (vanna, volga, charm, veta, speed, zomma, color)
- Pathwise and bump sensitivity methods
- VaR methodologies (Historical, Monte Carlo, Parametric)
- Expected Shortfall (ES/CVaR)
- Incremental VaR and Component VaR
- Stress testing with 25+ historical scenarios
- Position limits and pre-trade controls
- XVA suite (CVA, DVA, FVA, MVA, KVA)
- Exposure simulation (EE, PFE, EPE)
- Wrong-way risk modeling
- P&L attribution framework

**Market Data Infrastructure**
- Bloomberg Terminal/API integration
- Refinitiv Data Platform (RDP) and Eikon Desktop support
- PostgreSQL time-series optimized storage
- MongoDB flexible document storage
- TimescaleDB with automatic compression (up to 90%)
- Data validation and quality checks
- Feed management with automatic failover

**Observability & Monitoring**
- Prometheus metrics export (custom business metrics)
- Grafana dashboards (pre-built for pricing, risk, XVA)
- OpenTelemetry integration with Jaeger for distributed tracing
- Automatic performance profiling of slow requests
- Intelligent alerting with configurable thresholds

**Calibration**
- Differentiable calibration framework with Adam, L-BFGS optimizers
- Parameter estimation with diagnostics
- Model selection (AIC, BIC, AICc, HQIC)
- Cross-validation (k-fold, time-series)
- Sensitivity analysis (local finite differences, global Sobol indices)
- Identifiability checks and residual analysis

**APIs & Services**
- REST API with FastAPI
- gRPC service interface
- CLI tools for batch processing
- Interactive Dash dashboard for pricing and Greeks
- Streaming quotes with real-time updates

**Quality & Testing**
- 370+ comprehensive test files
- Unit, integration, and regression tests
- Performance benchmarking suite
- Precision validation tests
- Code quality enforcement (ruff, black, mypy)
- Security scanning with bandit
- Type checking with pydantic

**Documentation**
- Comprehensive README with examples
- API reference documentation
- Design decision documents
- Architecture overview
- Tutorial notebooks
- Example scripts for all product types
- Performance tuning guide

### Technical Details
- Python 3.10+ required
- JAX 0.4.26+ for autodiff and JIT compilation
- FastAPI for REST services
- gRPC for high-performance RPC
- Pydantic for configuration validation
- Prometheus, Grafana, Jaeger for observability

---

## Version History Summary

| Version | Release Date | Focus | Tests | Key Features |
|---------|-------------|-------|-------|-------------|
| v0.1.0 | Jan 2025 | Foundation & core pricing | 370+ | Multi-asset products, market data, observability |
| v0.2.0 | Apr 2025 | Advanced calibration | +60 | Joint calibration, Bayesian averaging, SLV |
| v0.4.0 | Jan 2026 | Regulatory compliance | +120 | FRTB SA/IMA, DRC/RRAO, SA-CCR, SIMM, IFRS 9/13 |
| v1.0.0 | Jun 2026 | Enterprise platform | +50 | SSO/OAuth/MFA/LDAP, K8s deployment, AMR, collateral |
| v1.0.3 | Nov 2025 | Analytics & research | +80 | Backtesting, factor analysis, portfolio optimization |
| **Total** | **Current** | **Production-ready** | **500+** | **Complete enterprise derivatives platform** |

---

## Upgrade Notes

### v0.1.0 → v0.2.0
- No breaking changes
- New calibration API with enhanced regularization
- Additional model constructors for SLV and jump clustering

### v0.2.0 → v0.4.0
- No breaking changes
- New regulatory modules under `neutryx.valuations.regulatory`
- SIMM calculation requires additional market data inputs

### v0.4.0 → v1.0.0
- Authentication now required for all API endpoints
- Environment variables needed for OAuth/LDAP configuration
- Kubernetes deployment examples available in documentation
- New collateral management API endpoints

### v1.0.0 → v1.0.3
- No breaking changes
- New portfolio optimization and analytics modules
- Additional dependencies for backtesting framework (see requirements.txt)

---

## Links

- [GitHub Repository](https://github.com/neutryx-lab/neutryx-core)
- [Documentation](https://neutryx-lab.github.io/neutryx-core)
- [Release Notes](https://github.com/neutryx-lab/neutryx-core/releases)

---

**Maintained by:** Neutryx Team
**Last Updated:** November 2025
