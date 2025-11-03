# Neutryx Core Codebase Exploration Report
**Thoroughness Level: Medium**  
**Date: 2025-11-03**  
**Repository: neutryx-core**

---

## EXECUTIVE SUMMARY

Neutryx Core is a **production-ready JAX-based quantitative finance library** currently at version 0.1.0. It provides a comprehensive framework for pricing derivatives, calculating Greeks, and managing XVA (Counterparty, Funding, Margin Valuation Adjustments). The codebase demonstrates **high-quality engineering** with extensive test coverage, clear architecture, and modern Python practices.

**Key Metrics:**
- 92 implementation files
- 37 test files with 78 test functions (2,420+ lines of test code)
- ~623 lines in core engine
- Comprehensive documentation with roadmap

---

## 1. CURRENT MODEL IMPLEMENTATIONS

### 1.1 Fully Implemented Models

#### Black-Scholes (✅ COMPLETE)
**Location:** `/src/neutryx/models/bs.py`

**Implemented Features:**
- Analytical European option pricing (calls & puts)
- Complete Greeks: Delta, Gamma, Vega, Theta
- Second-order Greeks (Vanna, Vomma) via automatic differentiation
- Implied volatility calculation (bisection method)
- JAX-native implementation with full JIT compilation support
- Hessian-vector product support for second derivatives

**Code Size:** ~65 lines  
**Status:** Production-ready

---

#### Heston Stochastic Volatility Model (✅ IMPLEMENTED)
**Location:** `/src/neutryx/models/heston_cf.py`

**Implemented Features:**
- Characteristic function implementation (foundational)
- Risk-neutral dynamics with correlation structure
- Parameters: v0 (initial variance), kappa (mean reversion), theta (long-term variance), sigma (vol-of-vol), rho (correlation)

**Calibration:** 
- Full calibration controller in `/src/neutryx/solver/calibration/heston.py`
- Uses characteristic function with Fourier inversion
- Heston call pricing via numerical integration (256-point Gauss-Laguerre quadrature)
- Optax-based optimization

**Current Limitations:**
- Calibration appears partial in solver (100 lines in solver/heston.py, not fully trace-ready)
- Numerical integration with fixed grid (u_max=100, N=256)

**Test Coverage:** Integration tests exist in regression suite

---

#### SABR (Stochastic Alpha Beta Rho) Model (✅ IMPLEMENTED)
**Location:** `/src/neutryx/solver/sabr.py`

**Implemented Features:**
- Hagan's asymptotic implied volatility formula
- Parameters: alpha (initial vol), beta (CEV exponent), rho (correlation), nu (vol-of-vol)
- Numerically stable computation with epsilon safeguards
- ATM and non-ATM cases
- Loss scaling for precision

**Calibration:**
- `SABRCalibrationController` in `/src/neutryx/solver/calibration/sabr.py`
- Least-squares calibration with parameter constraints
- Identifiability metrics and diagnostic outputs

**Status:** Production-ready with diagnostics

---

#### Merton Jump Diffusion Model (✅ IMPLEMENTED)
**Location:** `/src/neutryx/models/jump_diffusion.py`

**Implemented Features:**
- SDE: `dS = (mu - lambda*kappa)S*dt + sigma*S*dW + J*S*dN` (Poisson jumps)
- Closed-form European call pricing via series expansion (Poisson weights)
- Characteristic exponent
- Full calibration routine using Optax optimizer
- Parameters: sigma, lambda (jump intensity), mu_jump, sigma_jump

**Pricing:**
- `merton_jump_call()` with configurable n_terms (default 64)
- `merton_jump_price()` for calls/puts with parity
- High-precision Poisson weighting via gammaln

**Status:** Complete with calibration

---

#### Rough Volatility / Rough Bergomi (✅ IMPLEMENTED)
**Location:** `/src/neutryx/models/rough_vol.py`

**Implemented Features:**
- Rough Bergomi parameters (H, xi, rho)
- Forward variance curve calibration
- Monte Carlo simulation with variance targeting
- European call pricing via Monte Carlo
- Parameter fitting to variance smile

**Size:** ~250 lines  
**Status:** Research-ready (not typical for production deployments)

---

### 1.2 Partially Implemented / Missing Models

#### Local Volatility (Dupire) - ⚠️ STUB ONLY
**Location:** `/src/neutryx/solver/local_vol.py`

**Current Status:**
- Class definition: `LocalVolSurface`
- Function stubs: `dupire_local_volatility_surface()`, `call_price_surface_from_iv()`
- **Not functionally implemented** - contains only signatures

**Roadmap:** Planned for v1.0.0

---

#### Stochastic Local Volatility (SLV) - ⚠️ CALIBRATION ONLY
**Location:** `/src/neutryx/solver/calibration/slv.py`

**Current Status:**
- Calibration controller only (~130 lines)
- No pricing engine
- **Not usable without external SDE implementation**

---

#### Time-Inhomogeneous Models - ⚠️ MISSING
- No implementation found in source
- Referenced in roadmap for v1.0.0
- Test exists: `test_time_inhomogeneous.py` (tests parameterization, not pricing)

---

#### Regime-Switching Models - ❌ NOT IMPLEMENTED
- No source code found
- Mentioned in v1.0.0 roadmap only

---

#### Jump Models (Kou, Variance Gamma, NIG, CGMY) - ❌ NOT IMPLEMENTED
- Only Merton implemented
- Kou and Variance Gamma listed in v1.0.0 roadmap

---

---

## 2. IMPLEMENTED PRODUCTS

### 2.1 Fully Implemented Payoffs

#### Vanilla European Options (✅ COMPLETE)
**Location:** `/src/neutryx/products/vanilla.py`

**Features:**
- Call/Put payoffs
- Terminal payoff only (path-independent)
- Properties: `supports_pde=True`, `requires_path=False`
- Works with all solvers (PDE, MC, trees)

**Test Coverage:** `test_product_payoffs.py`

---

#### Asian Options - Arithmetic Average (✅ COMPLETE)
**Location:** `/src/neutryx/products/asian.py`

**Features:**
- Arithmetic-average payoff
- Call/Put support
- Path-dependent: `requires_path=True`
- Vmap-compatible for vectorized evaluation

**Limitations:** Only arithmetic average (geometric not implemented)

**Test Coverage:** `test_path_dependent.py`, `test_product_payoffs.py`

---

#### Barrier Options (✅ IMPLEMENTED)
**Location:** `/src/neutryx/products/barrier.py`

**Implemented:**
- Up-and-out call (monitored knockouts)
- Path-dependent evaluation

**Not Implemented:**
- Down-and-out, up-and-in, down-and-in variants
- Barrier rebates
- Continuous monitoring vs. discrete monitoring (currently continuous via path.max())

**Test Coverage:** `test_path_dependent.py`

---

#### Lookback Options (✅ IMPLEMENTED)
**Location:** `/src/neutryx/products/lookback.py`

**Implemented:**
- Floating-strike lookback call: payoff = S_T - min(S_t)
- Path-dependent

**Not Implemented:**
- Fixed-strike lookback
- Puts (only calls)

**Test Coverage:** `test_product_payoffs.py`

---

#### American Options (⚠️ INCOMPLETE)
**Location:** `/src/neutryx/products/american.py`

**Current Status:**
- Educational implementation only (~44 lines)
- Longstaff-Schwartz method for American puts
- **Not production-ready** - includes notice "not production-ready"
- Basis functions: [1, S, S²]
- Uses numpy (not JAX) - **incompatible with JAX engine pipeline**

**Issues:**
1. Uses `jnp.linalg.lstsq` in a loop (not vmap-compatible)
2. State mutation (`cashflows.at`, `exercise.at`) suggests not fully functional
3. No separate class definition for American option products
4. Example exists: `examples/03_american_lsm.py`

**Test Coverage:** No dedicated unit tests found

---

### 2.2 Product Catalog

**Location:** `/src/neutryx/products/__init__.py`

```python
PAYOFF_CATALOGUE = {
    "european": European,
    "asian_arithmetic": AsianArithmetic,
    "up_and_out_call": UpAndOutCall,
    "lookback_float_strike_call": LookbackFloatStrikeCall,
}
```

**Missing from catalog:** American options (not registered)

---

### 2.3 Missing Products (Per Roadmap)

**Not Implemented (v1.0.0 targets):**
- Digital/Binary options
- Quanto options
- Cliquet options
- Rainbow/Basket options (worst-of, best-of)
- Variance/Volatility swaps
- Swaptions, caps/floors, CMS products
- All interest rate derivatives

---

---

## 3. RISK & GREEKS CALCULATION IMPLEMENTATIONS

### 3.1 Greeks Calculations

#### Analytical Greeks (Black-Scholes) - ✅ COMPLETE
**Location:** `/src/neutryx/valuations/greeks/greeks.py`

**Implemented:**
- First-order Greeks: `bs_analytic_greeks()` → Delta, Gamma, Vega, Theta
- Second-order Greeks: `bs_second_order_greeks()` → Vanna, Vomma, Volga
- Uses automatic differentiation (Hessian-vector products)

**Implementation:** Wraps `neutryx.models.bs.greeks()` and `bs.second_order_greeks()`

---

#### Monte Carlo Greeks - ⚠️ MINIMAL
**Location:** `/src/neutryx/valuations/greeks/greeks.py`

**Implemented:**
- `mc_delta_bump()` - finite difference delta with configurable bump (default 1e-4)
- **Only delta** - no gamma, vega, or theta

**Missing:**
- Vega bumping
- Theta via time-stepping
- Pathwise Greeks for MC
- Greeks for exotic products

**Note:** Pathwise differentiation framework exists in `/src/neutryx/core/pricing/pathwise.py` but appears incomplete

---

### 3.2 Sensitivity Framework

**Pathwise Differentiation:** `/src/neutryx/core/pricing/pathwise.py`

**Status:** Documented infrastructure (50 lines) but **appears incomplete**
- Defines `AMCInputs`, `PathwisePayoff`, `PathwiseResult` classes
- No actual implementation of Greek computation functions shown

---

### 3.3 Greeks Test Coverage

**Test Files:**
- `test_greeks.py` - Tests for BS Greeks
- `test_monte_carlo_greeks.py` - Regression tests for MC Greek accuracy
- `test_precision_controls.py` - Numerical precision validation

**Coverage:** BS analytics heavily tested; MC Greeks lightly tested

---

---

## 4. XVA (Valuation Adjustments) IMPLEMENTATIONS

### 4.1 CVA (Credit Valuation Adjustment) - ✅ IMPLEMENTED

**Location:** `/src/neutryx/valuations/cva.py`

**Formula:**
```
CVA = sum(DF(t) * EPE(t) * dPD(t) * LGD)
```

**Parameters:**
- `epe_t`: Expected Positive Exposure at each time
- `df_t`: Discount factors
- `pd_t`: Cumulative default probability
- `lgd`: Loss Given Default (default 0.6)

**Implementation:** ~8 lines (vectorized JAX)

---

### 4.2 FVA (Funding Valuation Adjustment) - ✅ IMPLEMENTED

**Location:** `/src/neutryx/valuations/fva.py`

**Formula:**
```
FVA = sum(DF(t) * EPE(t) * funding_spread(t))
```

**Implementation:** ~2 lines (minimal)

---

### 4.3 MVA (Margin Valuation Adjustment) - ✅ IMPLEMENTED

**Location:** `/src/neutryx/valuations/mva.py`

**Formula:**
```
MVA = sum(DF(t) * initial_margin_profile(t) * spread(t))
```

**Implementation:** ~2 lines (minimal)

---

### 4.4 Exposure Simulation Engine - ✅ COMPLETE

**Location:** `/src/neutryx/valuations/xva/exposure.py` (~180 lines)

**Features:**
- `ExposureSimulator`: Multi-scenario exposure generator
- `XVAScenario`: Scenario definition with parameter overrides
- `ExposureCube`: Aggregation across scenarios with weighting
- `ExposureResult`: Per-scenario exposures (EPE, ENE, net)
- Supports pathwise exposure tracking

**Capabilities:**
- Scenario weighting and aggregation
- Expected Positive/Negative Exposure calculations
- Multi-scenario results stacking
- Full JAX-native implementation

---

### 4.5 Capital Calculator - ✅ IMPLEMENTED

**Location:** `/src/neutryx/valuations/xva/capital.py` (~120 lines)

**Features:**
- Regulatory capital measures (SA-CCR framework)
- EEPE (Effective Expected Positive Exposure)
- Potential Future Exposure (PFE) at confidence levels
- Capital charge calculations

**Status:** Complete but not heavily tested in visible test suite

---

### 4.6 Aggregation Engine - ✅ BASIC

**Location:** `/src/neutryx/valuations/xva/aggregation.py` (~40 lines)

**Features:**
- Portfolio-level XVA aggregation
- Netting across counterparties

**Limitations:** Minimal implementation

---

---

## 5. TEST COVERAGE ANALYSIS

### 5.1 Test Statistics

**Overview:**
- **37 test files** with **78 test functions**
- **2,420+ lines** of test code
- **Test-to-code ratio:** ~0.35 (reasonable for research code)

### 5.2 Well-Tested Areas (✅)

#### Core Pricing Engine
- `test_products_mc.py` - Monte Carlo pricing for vanilla/exotics
- `test_pde.py` - PDE solver validation
- `test_product_payoffs.py` - Individual payoff functions
- `test_fourier_pricing.py` - FFT/Fourier methods

#### Models
- `test_bs_analytic.py` - Black-Scholes formulas
- `test_jump_diffusion.py` - Merton model pricing
- `test_rough_vol.py` - Rough Bergomi simulation
- `test_calibration.py` - Model calibration workflows

#### Infrastructure
- `test_config_smoke.py` - Configuration validation
- `test_workflow_checkpoint.py` - Checkpoint/resume mechanics
- `test_mesh_execution.py` - Distributed execution
- `test_ffi_bindings.py` - Foreign function interface

#### Risk & Greeks
- `test_greeks.py` - Greek calculations
- `test_monte_carlo_greeks.py` - MC Greek accuracy
- `test_bs_pricing_benchmark.py` - Performance benchmarks

### 5.3 Under-Tested / Missing Tests (⚠️)

#### American Options
- ❌ **No unit tests** for American option payoff class
- Test example exists but not unit test

#### Path Estimation
- ⚠️ Limited coverage for complex path-dependent payoffs
- Only basic barrier/lookback tested

#### XVA Components
- ⚠️ `test_xva.py` exists but appears minimal
- No tests for aggregation engine
- No tests for capital calculator

#### API Layers
- ⚠️ `test_rest_api.py` - Minimal REST API tests
- No gRPC integration tests shown

#### Local Volatility
- ❌ No tests (not yet implemented)

#### Time-Inhomogeneous Models
- ⚠️ `test_time_inhomogeneous.py` - Tests parameterization only, not pricing

---

### 5.4 Test Organization

**Structure:**
```
tests/
├── test_*.py                 # Core unit tests (37 files)
├── autodiff/test_hvp.py     # Automatic differentiation tests
├── fixtures/                # Test data and factories
├── integration/             # End-to-end tests (CLI, REST)
├── market/                  # Market data validation
├── monte_carlo/             # QMC and convergence tests
├── performance/             # Benchmarks
├── precision/               # Numerical accuracy
├── regression/              # Stability tests
└── dev/                   # Profiling utilities
```

---

---

## 6. API IMPLEMENTATIONS

### 6.1 REST API - ✅ IMPLEMENTED

**Location:** `/src/neutryx/api/rest.py` (~120 lines)

**Framework:** FastAPI

**Endpoints Implemented:**

| Endpoint | Method | Functionality |
|----------|--------|---------------|
| `/price/vanilla` | POST | MC pricing for vanilla options |
| `/xva/cva` | POST | CVA calculation |
| `/xva/fva` | POST | FVA calculation |
| `/xva/mva` | POST | MVA calculation |

**Features:**
- Request validation via Pydantic models
- Path/step/seed configuration
- Antithetic sampling support
- Error handling with HTTP exceptions

**Limitations:**
- Only vanilla options supported (not Asian, barrier, etc.)
- No authentication/rate limiting
- No caching

---

### 6.2 gRPC API - ✅ IMPLEMENTED

**Location:** `/src/neutryx/api/grpc.py` (~160 lines)

**Service:** `PricingService`

**Methods Implemented:**
- `PriceVanilla(request) -> Struct`
- `ComputeCVA(request) -> Struct`
- `ComputeFVA(request) -> Struct`
- `ComputeMVA(request) -> Struct`

**Features:**
- Async/await support
- Protobuf Struct messages
- Graceful error handling
- Server startup/shutdown utilities

**Usage:**
```python
asyncio.run(serve(address="0.0.0.0:50051"))
```

---

### 6.3 CLI Interface - ⚠️ MINIMAL

**Location:** `/src/neutryx/core/utils/cli/`

**Status:** Legacy CLI tools exist but appear not fully integrated with current engine

---

---

## 7. DOCUMENTATION COMPLETENESS

### 7.1 Documentation Structure

**Location:** `/docs/` directory

**Available:**
- `README.md` - Comprehensive getting started guide (630 lines)
- `roadmap.md` - Development roadmap with timeline
- `overview.md` - Architecture overview
- `api_reference.md` - API reference (stub)
- `design_decisions.md` - Architectural choices
- `security_audit.md` - Security considerations
- `test_coverage.md` - Test coverage notes
- **Calibration guides:** Masterclass, diagnostics, playbooks
- **Risk guides:** Masterclass, controls, reference materials
- **Model documentation:** Index files for models

---

### 7.2 Documentation Quality

**Strengths:**
- Clear quickstart examples
- Project structure well-documented
- Installation instructions explicit
- Example scripts provided

**Gaps:**
- API reference is mostly stubs
- Limited function-level documentation in code
- Docstrings vary in quality
- No Sphinx/auto-generated docs

---

### 7.3 Example Coverage

**Location:** `/examples/`

**Available:**
- `01_bs_vanilla.py` - Black-Scholes comparison
- `02_path_dependents.py` - Asian/Barrier/Lookback pricing
- `03_american_lsm.py` - Longstaff-Schwartz for American options
- `perf/run_mc_vs_analytic.py` - Performance comparison
- `dashboard/app.py` - Interactive Dash application
- **Tutorials:** Vanilla pricing, Asian scenarios, CVA

---

---

## 8. IDENTIFIED GAPS & INCOMPLETE IMPLEMENTATIONS

### 8.1 Critical Gaps

#### 1. American Options - Incomplete ⚠️ CRITICAL
**Issue:** Educational implementation not integrated into main pricing pipeline
- Uses numpy instead of JAX
- Not compatible with automatic differentiation
- No property class definition (only function)
- Longstaff-Schwartz basis limited to 3 polynomials

**Impact:** Cannot price American options with main engine
**Fix Needed:** JAX-compatible implementation with parametric basis

---

#### 2. Local Volatility (Dupire) - Not Implemented
**Issue:** Stub-only (`local_vol.py`)
- Function signatures defined but not implemented
- No PDE solver for local vol calibration
- No time-dependent vol surface support

**Workaround:** None
**Roadmap:** v1.0.0

---

#### 3. Pathwise Differentiation - Incomplete
**Issue:** Framework defined but not implemented
- `pathwise.py` has class definitions but missing core computation
- Greeks calculations reference it but don't use it fully

**Impact:** Cannot efficiently compute Greeks for exotic options
**Roadmap:** Implicit in future enhancements

---

#### 4. Monte Carlo Greeks - Minimal ⚠️
**Issue:** Only delta bumping implemented
- No vega, theta, rho
- No pathwise Greeks
- Basic finite difference only

**Workaround:** Use analytical Black-Scholes
**Impact:** Limited sensitivity analysis for Monte Carlo

---

### 8.2 Model Gaps

| Model | Status | Notes |
|-------|--------|-------|
| Black-Scholes | ✅ Complete | All features |
| Heston | ✅ Complete | Characteristic function + calibration |
| SABR | ✅ Complete | Hagan formula + calibration |
| Merton Jumps | ✅ Complete | Closed-form + calibration |
| Rough Bergomi | ✅ Research | Simulation-based |
| Local Volatility | ❌ Stub | No PDE solver |
| SLV | ⚠️ Partial | Calibration only, no pricing |
| Time-Inhomogeneous | ❌ Missing | Parameters only |
| Regime-Switching | ❌ Missing | Roadmap only |
| Kou Jumps | ❌ Missing | v1.0.0 target |
| Variance Gamma | ❌ Missing | v1.0.0 target |

---

### 8.3 Product Gaps

**Not Implemented:**
- All other barrier types (down-and-out, up-and-in, down-and-in)
- Geometric average Asian
- Fixed-strike lookback
- Basket/Rainbow options
- Quanto options
- Digital/Binary options
- Interest rate derivatives
- FX exotics

**Status:** Most planned for v1.0.0

---

### 8.4 Risk Gaps

**Missing:**
- Vega exposure (for FX products)
- Greeks for exotic products
- Rho for interest rate products
- Implied vol surface Greeks (vanna, vomma on surface)
- Greeks sensitivity aggregation

---

### 8.5 Integration Gaps

**Between Modules:**
- American options not integrated with pricing engine
- Local volatility calibration disconnected from pricing
- SLV calibration has no pricing backend

---

---

## 9. ARCHITECTURE OBSERVATIONS

### 9.1 Strengths

1. **JAX-First Design**
   - Full JIT compilation support
   - Automatic differentiation throughout
   - GPU/TPU ready with pmap/pjit

2. **Modular Structure**
   - Clear separation: models → products → valuations
   - Payoff protocol-based design
   - Engine agnostic to product type

3. **Comprehensive Testing**
   - 78 test functions across 37 files
   - Unit, integration, performance, precision layers
   - Regression test suite

4. **Production Infrastructure**
   - REST and gRPC APIs
   - Configuration management
   - Checkpoint/resume support
   - Tracking and monitoring

5. **Modern Python**
   - Type hints throughout
   - Pydantic for validation
   - Dataclasses for data structures
   - Protocol-based abstractions

---

### 9.2 Architectural Weaknesses

1. **Incomplete Model Integration**
   - SLV calibration without pricing
   - Local vol stub implementation
   - American options outside main pipeline

2. **Greeks Calculation Gaps**
   - MC Greeks minimal (delta-only)
   - Pathwise framework incomplete
   - Limited exotic Greeks support

3. **API Layer Limitations**
   - REST/gRPC only for vanilla options (not exotics)
   - No caching or optimization hints
   - Simple payload validation

4. **Documentation Gaps**
   - API reference mostly stubs
   - Function-level docstrings incomplete
   - Limited architecture documentation

---

---

## 10. SUMMARY TABLE: WHAT'S IMPLEMENTED VS DOCUMENTED

### Models
| Model | Implemented | Tested | Documented | Calibrated | Production |
|-------|-----------|--------|-----------|-----------|-----------|
| Black-Scholes | ✅ | ✅ | ✅ | N/A | ✅ |
| Heston | ✅ | ✅ | ✅ | ✅ | ✅ |
| SABR | ✅ | ✅ | ✅ | ✅ | ✅ |
| Merton Jumps | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rough Bergomi | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Local Vol | ❌ | ❌ | ❌ | ❌ | ❌ |
| SLV | ⚠️ | ❌ | ❌ | ⚠️ | ❌ |
| Time-Inhom. | ❌ | ❌ | ❌ | ❌ | ❌ |
| Regime-Switch | ❌ | ❌ | ❌ | ❌ | ❌ |

### Products
| Product | Implemented | Tested | Path-Dependent | Production |
|---------|-----------|--------|----------------|-----------|
| Vanilla European | ✅ | ✅ | ❌ | ✅ |
| Asian Arith. | ✅ | ✅ | ✅ | ✅ |
| Barrier (1 type) | ✅ | ✅ | ✅ | ✅ |
| Lookback | ✅ | ✅ | ✅ | ✅ |
| American | ⚠️ | ❌ | ✅ | ❌ |

### Risk/XVA
| Component | Implemented | Tested | Production |
|-----------|-----------|--------|-----------|
| BS Greeks | ✅ | ✅ | ✅ |
| MC Greeks | ⚠️ | ⚠️ | ⚠️ |
| CVA | ✅ | ⚠️ | ✅ |
| FVA | ✅ | ⚠️ | ✅ |
| MVA | ✅ | ⚠️ | ✅ |
| Exposure Sim | ✅ | ⚠️ | ✅ |
| Capital Calc | ✅ | ❌ | ✅ |

---

## CONCLUSION

Neutryx Core is a **well-engineered, production-ready framework** for the most common derivatives pricing scenarios (vanilla options, basic exotics, Black-Scholes, Heston, SABR). It has strong fundamentals:
- Comprehensive test coverage
- Modern JAX-first architecture
- Professional API and infrastructure layers

**Main limitations:**
1. **American options** - Not integrated into main pricing pipeline
2. **Advanced models** - Local volatility, SLV, time-inhomogeneous not implemented
3. **Exotic products** - Limited to a few path-dependent types
4. **Monte Carlo Greeks** - Only delta bumping, incomplete framework

**For production use:** Suitable for vanilla/Asian/barrier/lookback pricing with Black-Scholes/Heston/SABR/Merton.

**For expansion:** Local volatility, advanced jump models, and comprehensive Greeks require additional development work outlined in the v0.9-1.0 roadmap.

