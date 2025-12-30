# Project Structure

## Organization Philosophy

**Domain-Driven Modular Architecture**: Code organized by financial domain (products, models, market, valuations) rather than technical layers. Each module is self-contained with clear boundaries, enabling independent development and testing.

**Core-Products-Infrastructure Separation**:
- `core/` provides primitives (dates, math, engine)
- `products/` and `models/` implement financial logic
- `market/`, `valuations/`, `regulatory/` provide domain services
- `infrastructure/` handles enterprise concerns (governance, observability)

**Examples Outside Source**: Tutorial code, dashboards, and benchmarks live in `examples/` to keep library focused. Documentation in `docs/`, development tools in `dev/`.

## Directory Patterns

### Core Computational Layer (`/src/neutryx/core/`)
**Purpose**: Fundamental building blocks used across entire codebase
**Contents**: Monte Carlo engine, automatic differentiation, date utilities, calendars, RNG, math solvers
**Example**: `core/engine.py` (simulate_gbm, present_value), `core/dates/calendar.py` (TARGET, US calendars)

### Financial Models (`/src/neutryx/models/`)
**Purpose**: Stochastic processes and pricing models
**Contents**: Black-Scholes, Heston, SABR, Hull-White, LMM, jump-diffusion, rough volatility
**Example**: `models/bs.py` (Black-Scholes analytic), `models/heston.py` (stochastic vol simulation)

### Product Library (`/src/neutryx/products/`)
**Purpose**: Payoff definitions and product-specific pricing logic
**Organization**: Grouped by asset class (linear_rates, fx_complex, energy, etc.)
**Example**: `products/linear_rates/irs.py` (interest rate swaps), `products/swaptions.py` (European/Bermudan swaptions)

### Market Data Infrastructure (`/src/neutryx/market/`)
**Purpose**: Curve construction, market data feeds, storage, validation
**Contents**:
- `adapters/` - Bloomberg, Refinitiv vendor integrations
- `storage/` - TimescaleDB, PostgreSQL, MongoDB connectors
- `feeds/` - Real-time feed orchestration with failover
- `validation/` - Price range checks, anomaly detection
**Example**: `market/adapters/bloomberg.py`, `market/storage/timescale.py`

### Valuations & Risk (`/src/neutryx/valuations/`)
**Purpose**: Risk metrics, Greeks, XVA, margin calculations
**Subdirectories**: `greeks/`, `risk/`, `xva/`, `margin/`, `regulatory/`, `scenarios/`
**Example**: `valuations/risk/var.py` (VaR methods), `valuations/xva/cva.py` (CVA calculation)

### Regulatory Compliance (`/src/neutryx/regulatory/`)
**Purpose**: FRTB, SA-CCR, SIMM, accounting standards, trade reporting
**Contents**: `ima/` (Internal Models Approach), `reporting/` (EMIR, MiFID II), `accounting/` (IFRS 9/13)
**Example**: `regulatory/ima/frtb_ima.py`, `regulatory/reporting/emir.py`

### Calibration Framework (`/src/neutryx/calibration/`)
**Purpose**: Model parameter estimation, diagnostics, regularization
**Contents**: Loss functions, constraints, Bayesian averaging, sensitivity analysis
**Example**: `calibration/heston.py`, `calibration/bayesian_model_averaging.py`

### Portfolio & Trading (`/src/neutryx/portfolio/`, `/src/neutryx/trading/`)
**Purpose**: Trade lifecycle, contracts, RFQ workflow, settlement
**Contents**: Trade generation with conventions, confirmation matching, CSA management
**Example**: `portfolio/trade_generation/conventions.py`, `trading/rfq.py`

### Infrastructure (`/src/neutryx/infrastructure/`)
**Purpose**: Enterprise governance, observability, distributed computing
**Contents**:
- `governance/` - RBAC, audit, multi-tenancy, SLA monitoring
- `observability/` - Prometheus metrics, OpenTelemetry tracing, profiling, alerting
- `config/` - Configuration schemas and defaults
**Example**: `infrastructure/governance/rbac.py`, `infrastructure/observability/metrics.py`

### Integrations (`/src/neutryx/integrations/`)
**Purpose**: External system interfaces (CCP, settlement, FFI)
**Contents**:
- `clearing/` - LCH, CME, ICE, Eurex CCP connectors
- `databases/` - Async database adapters
- `ffi/` - QuantLib, Eigen C++ bridges
- `fpml/` - FpML parsing and generation
**Example**: `integrations/clearing/lch.py`, `integrations/fpml/parser.py`

### Examples & Applications (`/examples/`)
**Purpose**: Tutorials, dashboards, benchmarks (not part of library)
**Organization**:
- `basic/`, `advanced/`, `tutorials/` - Learning materials
- `applications/` - Full apps (fictional_bank, dashboard)
- `benchmarks/`, `demos/` - Performance and feature showcases
**Example**: `examples/applications/fictional_bank/cli.py`, `examples/tutorials/01_vanilla_pricing/`

### Tests (`/tests/`)
**Purpose**: Comprehensive test suite (500+ tests)
**Organization**: Mirrors `src/neutryx/` structure with `test_*.py` files
**Markers**: `@pytest.mark.{unit,integration,slow,fast,regression,performance}`

### Development Tools (`/dev/`)
**Purpose**: CI scripts, monitoring stack, profiling, orchestration
**Contents**:
- `benchmarks/` - Performance testing harness
- `monitoring/` - Prometheus/Grafana/Jaeger deployment configs (managed by neutryx-api)
- `profiling/` - Kernel profiler and analysis tools
- `ci/` - Test reporting, coverage scripts
**Note**: Deployment orchestration (Kubernetes, auto-scaling) managed by neutryx-api package

## Naming Conventions

### Files
- **Modules**: `snake_case.py` (e.g., `hull_white.py`, `european_swaption.py`)
- **Classes**: `PascalCase` matching filename purpose (e.g., `class HullWhiteModel`)
- **Config files**: `lowercase.yaml` or `lowercase.toml`

### Functions & Variables
- **Public API**: `snake_case` (e.g., `def price_european_call()`)
- **Private helpers**: `_leading_underscore` (e.g., `def _validate_params()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `BUSINESS_DAY_CONVENTION`)

### Test Files
- **Pattern**: `test_<module_name>.py` mirroring source structure
- **Test functions**: `test_<what_is_tested>` (e.g., `test_european_call_pricing()`)

## Import Organization

```python
# Standard library
from __future__ import annotations
from typing import Optional, Tuple

# Third-party (grouped by package)
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# Neutryx internal (absolute imports preferred)
from neutryx.core.engine import MCConfig, simulate_gbm
from neutryx.models.bs import price as bs_price
from neutryx.market.curves import YieldCurve

# Local relative (only for same-subpackage imports)
from .utils import validate_inputs
```

**Import Rules**:
- Use absolute imports (`from neutryx.core...`) for cross-module dependencies
- Relative imports (`.utils`) only within same subpackage
- Group imports: stdlib → third-party → neutryx → local
- Follow isort black profile (enforced by CI)

## Code Organization Principles

### Pure Functions Everywhere
JAX requires pure functions for JIT compilation. No global state, explicit PRNG keys passed as arguments.

```python
# ✅ Good: Pure function with explicit randomness
def simulate_gbm(key: PRNGKey, S0: float, ...) -> Array:
    return paths

# ❌ Bad: Implicit global RNG state
def simulate_gbm(S0: float, ...) -> Array:
    np.random.seed(42)  # Global state breaks JAX
```

### Module-Level Entry Points
Each product/model module exposes simple entry point functions (e.g., `price()`, `calibrate()`) alongside classes for advanced usage.

### Configuration Over Code
Use YAML configs for experiments, Pydantic schemas for validation. Avoid hardcoded parameters in library code.

### Test Proximity
Tests mirror source structure (`tests/products/test_swaptions.py` ↔ `src/neutryx/products/swaptions.py`) for easy navigation.

### Documentation Standards
- Docstrings: NumPy style with LaTeX math for formulas
- Type hints: Required for all public APIs
- Examples: Inline code examples in docstrings, full examples in `/examples/`

### Separation of Concerns
- **Library code** (`src/neutryx/`): No CLI, dashboards, or deployment logic
- **API services** (`neutryx-api` package): Authentication, REST/gRPC endpoints, Kubernetes
- **Examples** (`examples/`): Standalone scripts, dashboards, tutorials

---
_Generated: 2025-12-30_
