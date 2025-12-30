# Technology Stack

## Architecture

**JAX-First Computational Platform**: Built entirely on JAX for automatic differentiation, JIT compilation, and hardware acceleration (GPU/TPU). Functional programming paradigm with pure functions enabling reproducible computations and seamless distributed execution.

**Modular Multi-Layer Design**: Core pricing engines separated from products, models, market data, and infrastructure layers. Clean boundaries enable independent testing and deployment. API services extracted to separate `neutryx-api` package for modular infrastructure.

## Core Technologies

- **Language**: Python 3.10+ with type hints (typing_extensions)
- **Numerical Computing**: JAX 0.4.26+ (jax, jaxlib) for automatic differentiation and GPU acceleration
- **Scientific Stack**: NumPy 1.26+, SciPy 1.11+, Pandas 2.1+ for data manipulation
- **Optimization**: Optax 0.1.8+ for gradient-based calibration
- **Configuration**: PyYAML 6.0.1+ for declarative config, Pydantic 2.6+ for validation
- **Observability**: Prometheus Client 0.20+, OpenTelemetry 1.25+ for metrics and tracing

## Key Libraries

### Numerical & Computational
- **JAX**: Core computational framework (JIT, autodiff, XLA optimization)
- **Optax**: Gradient-based optimization for model calibration
- **SciPy**: Root finding, interpolation, special functions
- **NumPy/Pandas**: Data structures and array operations

### Optional Integrations
- **QuantLib**: Legacy pricing model validation (optional dependency)
- **Eigen**: High-performance linear algebra via pybind11 FFI (optional)
- **AsyncPG/Motor**: PostgreSQL/MongoDB async connectors for market data
- **Bloomberg/Refinitiv**: Vendor-specific market data adapters

### Infrastructure (Optional - neutryx-api package)
- **FastAPI/Uvicorn**: REST API services (extracted to neutryx-api)
- **gRPC/Protobuf**: High-performance RPC (extracted to neutryx-api)
- **Auth Stack**: python-jose, passlib, pyotp for SSO/OAuth/MFA (neutryx-api)

## Development Standards

### Type Safety
- **Strict typing**: Type hints required for all public APIs
- **Runtime validation**: Pydantic models for config and schemas
- **JAX types**: Use `jax.Array` (not `np.ndarray`) for array types

### Code Quality
- **Formatter**: Black (100-char line length)
- **Linter**: Ruff (select E/F/I/B, ignore E203)
- **Import sorting**: isort with black profile
- **Security**: Bandit static analysis, pip-audit for dependencies

### Testing
- **Framework**: pytest with 500+ tests (unit, integration, regression)
- **Markers**: `@pytest.mark.{unit,integration,slow,fast,regression,performance}`
- **Coverage**: pytest-cov with HTML reports
- **Parallel**: pytest-xdist for multi-core execution (`pytest -n auto`)
- **Async**: pytest-asyncio with `asyncio_mode = "auto"`

### Performance Standards
- **JIT compilation**: All hot paths use `@jax.jit` decorator
- **Vectorization**: Use `jax.vmap` for batch operations
- **Reproducibility**: PRNG seeding via `jax.random.PRNGKey` in configs
- **GPU acceleration**: `pmap`/`pjit` for multi-device parallelism

## Development Environment

### Required Tools
- Python 3.10+ (3.11 recommended)
- JAX 0.4.26+ (with GPU support: `jax[cuda12]` or `jax[cuda11]`)
- Git for version control
- pytest for testing

### Optional Tools
- JupyterLab 4.4.8+ for notebooks (examples/tutorials)
- Docker for containerization (neutryx-api deployment)
- TimescaleDB for market data time-series storage
- Prometheus/Grafana for monitoring (managed by neutryx-api)

### Common Commands
```bash
# Dev environment setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Testing
pytest -v                           # All tests
pytest -n auto                      # Parallel execution
pytest -m "not slow"                # Skip slow tests
pytest --cov=neutryx --cov-report=html

# Code quality
ruff check src/                     # Linting
black src/ tests/                   # Formatting
bandit -r src/                      # Security scan

# Examples
python examples/basic/01_bs_vanilla.py
python examples/applications/dashboard/app.py  # Dash dashboard on :8050
```

## Key Technical Decisions

### JAX Over NumPy/TensorFlow/PyTorch
**Rationale**: JAX combines NumPy familiarity with automatic differentiation, JIT compilation, and hardware acceleration. Unlike TensorFlow/PyTorch (designed for deep learning), JAX excels at scientific computing with pure functional paradigm enabling reproducible finance computations.

### Functional Programming Paradigm
**Rationale**: Pure functions with explicit PRNG state enable reproducible pricing, efficient caching, and seamless distributed execution. No hidden state simplifies testing and debugging.

### Modular Package Architecture
**Rationale**: Core library (`neutryx-core`) focuses on computation; deployment infrastructure extracted to `neutryx-api`. Enables users to integrate Neutryx into existing systems without pulling in API/auth dependencies.

### Multi-Curve Market Data Framework
**Rationale**: Post-2008 OIS discounting and tenor basis spread modeling are industry standard. Multi-curve framework from day one prevents technical debt.

### Differentiable Calibration
**Rationale**: Automatic differentiation enables gradient-based calibration orders of magnitude faster than finite differences. Critical for real-time recalibration and large-scale parameter estimation.

### Vendor-Agnostic Adapters
**Rationale**: Abstract market data interfaces prevent vendor lock-in. Supports Bloomberg, Refinitiv, or custom feeds with consistent API.

---
_Generated: 2025-12-30_
