# Changelog

All notable changes to Neutryx Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Extended model coverage (Kou, Variance Gamma, rough volatility)
- C++/CUDA kernel integration for critical paths
- Advanced distributed execution patterns
- Exotic options (Quanto, Cliquet, Rainbow, baskets)
- Joint calibration framework
- Cloud/HPC orchestration support

## [0.1.0] - 2025-01-XX

### Added
- **Core Infrastructure**
  - Monte Carlo simulation engine with JAX JIT compilation
  - PDE solvers (Crank-Nicolson with boundary conditions)
  - Automatic differentiation framework with Hessian-vector products
  - GPU/TPU optimization with pmap/pjit
  - Mixed-precision support for efficient computation
  - YAML-based configuration system
  - Reproducible PRNG seeding across Python, NumPy, and JAX

- **Models**
  - Black-Scholes analytical pricing and Greeks
  - Geometric Brownian Motion (GBM) simulation
  - Heston stochastic volatility model
  - SABR volatility model
  - Jump diffusion models (Merton)
  - Tree-based pricing methods

- **Products**
  - Vanilla European options (call/put)
  - Asian options (arithmetic/geometric average)
  - Barrier options (up-and-out, down-and-out, up-and-in, down-and-in)
  - Lookback options (fixed/floating strike)
  - American options via Longstaff-Schwartz method

- **Risk & Analytics**
  - Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
  - Pathwise and bump sensitivity methods
  - XVA suite (CVA, FVA, MVA)
  - Exposure simulation framework
  - Stress testing engines

- **Market Data**
  - Yield curve construction and interpolation
  - Volatility surface management
  - Term structure utilities
  - Credit curve support
  - Hazard rate models

- **Calibration**
  - Differentiable calibration framework
  - Parameter estimation with diagnostics
  - Residual analysis tools
  - Identifiability checks

- **APIs & Services**
  - REST API with FastAPI
  - gRPC service interface
  - CLI tools for batch processing
  - Interactive Dash dashboard

- **Quality & Testing**
  - 38+ comprehensive test files
  - Unit, integration, and regression tests
  - Performance benchmarking suite
  - Precision validation tests
  - Code quality enforcement (ruff, black)
  - Type checking with pydantic

- **Documentation**
  - Comprehensive README with examples
  - API reference documentation
  - Design decision documents
  - Tutorial notebooks
  - Example scripts and dashboards

### Technical Details
- Python 3.10+ required
- JAX 0.4.26+ for autodiff and JIT
- FastAPI for REST services
- gRPC for high-performance RPC
- Pydantic for configuration validation

### Known Limitations
- Some advanced exotic options not yet implemented
- Limited support for time-inhomogeneous models
- Cloud orchestration requires manual setup

## [0.9.0] - Planned (Core PDE & GPU Stabilization)

### Planned Features
- Enhanced PDE solver stability
- Optimized GPU kernel implementations
- Expanded benchmarking suite for CPU/GPU/TPU
- Advanced boundary condition handling
- Improved numerical stability for edge cases

## [1.0.0] - Planned (Complete Model Suite)

### Planned Features
- Full rough volatility model suite
- Local volatility with Dupire PDE
- Time-dependent parameter estimation
- Enhanced model selection framework
- Production-ready deployment templates

## [1.1.0] - Planned (Full XVA & Risk Platform)

### Planned Features
- Complete counterparty risk framework
- Real-time market data integration
- Regulatory reporting templates (FRTB, SA-CCR)
- Multi-tenancy support
- Audit logging and compliance features

---

## Version History

- **v0.1.0** - Initial public release with core functionality
- **v0.9.0** - Planned: PDE and GPU stabilization
- **v1.0.0** - Planned: Complete model suite
- **v1.1.0** - Planned: Full XVA and risk platform

[Unreleased]: https://github.com/neutryx-lab/neutryx-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/neutryx-lab/neutryx-core/releases/tag/v0.1.0
