# Product Overview

**Neutryx Core** is a JAX-first quantitative finance library for derivatives pricing, risk management, and regulatory compliance. It serves investment banks, hedge funds, and quantitative researchers building enterprise-grade trading systems.

## Core Capabilities

### Multi-Asset Derivatives Platform
Complete derivatives lifecycle management across asset classes (Interest Rate, FX, Equity, Credit, Commodity) with 500+ tests covering vanilla to exotic products. Built on JAX for GPU/TPU acceleration and automatic differentiation.

### Risk & Regulatory Compliance
Production-grade risk management including VaR/ES, stress testing, Greeks calculation, and XVA (CVA/DVA/FVA/MVA/KVA). Full regulatory compliance with FRTB SA/IMA, SA-CCR, ISDA SIMM, and EMIR/MiFID II reporting.

### Enterprise Infrastructure
Real-time market data integration (Bloomberg, Refinitiv), distributed computing support, enterprise governance (RBAC, audit logging, multi-tenancy), and observability instrumentation (Prometheus, OpenTelemetry).

### Advanced Analytics
Differentiable calibration framework with Bayesian model averaging, portfolio optimization (Black-Litterman, robust optimization), backtesting infrastructure, and factor analysis tools.

### Trading Workflow Integration
Trade lifecycle management with FpML support, RFQ workflow, convention-based trade generation, CCP integration (LCH, CME, ICE, Eurex), and settlement systems (CLS, Euroclear, SWIFT).

## Target Use Cases

- **Derivatives Pricing**: Real-time pricing of vanilla and exotic derivatives across all major asset classes with automatic Greeks calculation
- **Risk Management**: Portfolio-level VaR/ES, scenario analysis, limit monitoring, and regulatory capital calculation
- **Regulatory Reporting**: Automated FRTB, SA-CCR, SIMM calculations with XML generation for trade repositories
- **Market Data Processing**: Real-time feeds with validation, storage (TimescaleDB with 90% compression), and quality monitoring
- **Portfolio Optimization**: Multi-period allocation, risk parity, CVaR optimization with backtesting framework
- **Research & Development**: Model calibration, sensitivity analysis, factor analysis with production-ready infrastructure

## Value Proposition

**Unified Computational Platform**: Single JAX-based platform unifying stochastic models, PDE solvers, market data, and regulatory frameworks in one continuous computational graph. Every component from yield curves to XVA is JIT-compiled and GPU-accelerated.

**Production-Ready Enterprise Features**: Unlike academic libraries, Neutryx Core delivers enterprise governance (RBAC, audit, multi-tenancy), distributed computing, real-time market data integration, and comprehensive regulatory compliance out of the box.

**Schema-Driven Development**: Centralized Pydantic schema registry ensures type safety and eliminates schema drift across Python/TypeScript/API boundaries. Automatic code generation for frontend integration accelerates development velocity.

**Differentiable Everything**: Automatic differentiation through entire pricing-to-risk workflow enables efficient calibration, adjoint-based Greeks, and gradient-based optimization impossible with traditional finite difference approaches.

**Performance at Scale**: 10-100x speedup through JIT compilation, multi-GPU Monte Carlo, adaptive mesh refinement for PDEs, and advanced variance reduction techniques (QMC, MLMC, control variates).

---
_Updated: 2025-12-30_
