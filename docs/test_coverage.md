# Test Coverage

Summary of automated test coverage, reporting conventions, and quality gates for Neutryx Core.

## Current Status (November 2025)

**Total Tests:** 500+ comprehensive tests across all modules
**Coverage:** 85%+ line coverage, 90%+ for core modules
**Test Types:** Unit, integration, regression, performance, security

## Test Distribution by Module

| Module | Tests | Focus Areas |
|--------|-------|-------------|
| **Interest Rate Derivatives** | 87 | IRS, OIS, swaptions, CMS, exotic IR |
| **Regulatory Compliance** | 120 | FRTB SA/IMA, DRC/RRAO, SA-CCR, SIMM, IFRS |
| **Calibration** | 60 | Joint calibration, Bayesian averaging, regularization |
| **Risk Management** | 57 | VaR, stress testing, limits, pre-trade controls |
| **Portfolio Analytics** | 80 | Backtesting, factor analysis, optimization |
| **XVA & CCR** | 35 | CVA, DVA, FVA, MVA, KVA, exposure profiles |
| **Market Data** | 25 | Vendor integration, validation, storage |
| **Core Pricing** | 36 | Monte Carlo, PDE solvers, Greeks |
| **Total** | **500+** | **Comprehensive coverage across platform** |

## Test Categories

### 1. Unit Tests (300+)

Test individual functions and components in isolation:

- **Pricing Functions**: Black-Scholes, Heston, SABR analytical pricing
- **Model Implementations**: Parameter validation, constraint checking
- **Greeks Calculation**: Delta, gamma, vega accuracy tests
- **Calibration**: Optimizer convergence, constraint satisfaction
- **Market Data**: Parsing, validation, transformation logic
- **Utilities**: Day count conventions, holiday calendars, date adjustments

**Example:**
```python
def test_black_scholes_call_price():
    """Test BS call option pricing against known values."""
    price = bs_price(S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.2, option_type="call")
    expected = 10.450583572185565
    assert jnp.isclose(price, expected, rtol=1e-6)
```

### 2. Integration Tests (120+)

Test end-to-end workflows and system interactions:

- **Pricing Workflows**: Market data → calibration → pricing → Greeks
- **Risk Workflows**: Position updates → VaR calculation → limit checking
- **XVA Workflows**: Exposure simulation → CVA calculation → reporting
- **API Endpoints**: REST/gRPC request handling and responses
- **Database Operations**: Read/write operations with PostgreSQL, MongoDB, TimescaleDB
- **Regulatory Reports**: FRTB, SA-CCR, SIMM calculation pipelines

**Example:**
```python
def test_heston_calibration_workflow():
    """Test complete Heston calibration workflow."""
    market_data = load_market_data()
    model = HestonModel()
    calibrated = calibrate_model(model, market_data)
    assert calibrated.converged
    assert all(calibrated.params.values() > 0)  # Positive parameters
```

### 3. Regression Tests (50+)

Ensure pricing and risk calculations remain stable across versions:

- **Pricing Stability**: Compare current vs reference prices
- **Greeks Stability**: Verify sensitivities within tolerance
- **Calibration Stability**: Parameter convergence to known values
- **Performance Regression**: Monitor execution times
- **Numerical Precision**: Track floating-point accuracy

**Test Data:**
- Reference prices from QuantLib, Bloomberg
- Historical market data snapshots
- Known analytical solutions

### 4. Performance Tests (20+)

Benchmark critical paths and validate scalability:

- **Monte Carlo**: Paths/second throughput
- **PDE Solvers**: Grid points/second throughput
- **Calibration**: Convergence speed
- **Batch Pricing**: Scaling with batch size
- **GPU Acceleration**: CPU vs GPU speedup
- **Memory Usage**: Peak memory consumption

**Targets:**
- 10-100x speedup with JIT compilation
- Linear scaling for batch pricing
- 100x+ speedup on GPU for Monte Carlo

### 5. Security Tests (10+)

Validate security controls and vulnerabilities:

- **Input Validation**: SQL injection, XSS prevention
- **Authentication**: OAuth/JWT token validation
- **Authorization**: RBAC permission checks
- **Audit Logging**: Event capture and integrity
- **Dependency Scanning**: Known vulnerabilities (bandit, pip-audit)

## Coverage Targets

### Coverage by Module

| Module | Target | Current | Status |
|--------|--------|---------|--------|
| Core Pricing | 95% | 96% | ✅ Exceeds |
| Models | 90% | 92% | ✅ Exceeds |
| Products | 90% | 88% | 🔄 Close |
| Risk Management | 85% | 87% | ✅ Exceeds |
| XVA & Valuations | 85% | 86% | ✅ Exceeds |
| Calibration | 90% | 93% | ✅ Exceeds |
| Market Data | 80% | 82% | ✅ Exceeds |
| Infrastructure | 75% | 78% | ✅ Exceeds |
| **Overall** | **85%** | **87%** | ✅ **Exceeds** |

### Critical Path Coverage

Functions on critical paths require 95%+ coverage:
- Option pricing engines
- Greeks calculation
- VaR methodologies
- XVA calculations
- Regulatory capital calculations

## Running Tests

### All Tests

```bash
pytest -v                              # Verbose output
pytest -q                              # Quiet output (default)
pytest --cov=neutryx --cov-report=html # Coverage report
```

### By Category

```bash
# Unit tests only
pytest src/neutryx/tests/unit/ -v

# Integration tests
pytest src/neutryx/tests/integration/ -v

# Regression tests
pytest -m regression

# Performance tests
pytest -m performance

# Specific module
pytest src/neutryx/tests/products/ -v
pytest src/neutryx/tests/risk/ -v
pytest src/neutryx/tests/valuations/regulatory/ -v
```

### By Feature

```bash
# Interest rate derivatives
pytest src/neutryx/tests/products/test_interest_rate.py -v
pytest src/neutryx/tests/products/test_swaptions.py -v

# Regulatory compliance
pytest src/neutryx/tests/valuations/regulatory/test_frtb.py -v
pytest src/neutryx/tests/valuations/regulatory/test_sa_ccr.py -v
pytest src/neutryx/tests/valuations/regulatory/test_simm.py -v

# Portfolio analytics
pytest src/neutryx/tests/portfolio/test_optimization.py -v
pytest src/neutryx/tests/portfolio/test_backtesting.py -v
pytest src/neutryx/tests/portfolio/test_factor_analysis.py -v
```

## Test Quality Standards

### 1. Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_european_call_option():
    # Arrange: Setup inputs and expected results
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
    expected_price = 10.450583572185565

    # Act: Execute the function under test
    price = european_option_price(S, K, T, r, q, sigma, "call")

    # Assert: Verify results
    assert jnp.isclose(price, expected_price, rtol=1e-6)
```

### 2. Test Isolation

- Each test should be independent
- Use fixtures for shared setup
- Clean up resources in teardown
- No test order dependencies

### 3. Test Documentation

- Docstring explaining what is tested
- Comments for complex test logic
- Clear assertion messages
- Link to requirements/issues

### 4. Test Data

- Use realistic market data
- Include edge cases and boundary conditions
- Test both valid and invalid inputs
- Document data sources

## Continuous Integration

Tests run automatically on:

- **Pull Requests**: All tests must pass
- **Main Branch**: Full test suite + performance benchmarks
- **Nightly Builds**: Extended tests + security scans
- **Release Tags**: Complete validation suite

### Quality Gates

| Gate | Threshold | Current | Status |
|------|-----------|---------|--------|
| All tests pass | 100% | 100% | ✅ |
| Line coverage | >85% | 87% | ✅ |
| Branch coverage | >80% | 82% | ✅ |
| Performance regression | <5% | 2% | ✅ |
| Security vulnerabilities | 0 critical | 0 | ✅ |
| Code quality (ruff) | 0 errors | 0 | ✅ |

## Test Maintenance

### Adding New Tests

1. Identify the feature to test
2. Write test cases covering:
   - Happy path (normal operation)
   - Edge cases (boundary conditions)
   - Error cases (invalid inputs)
3. Ensure tests are deterministic
4. Add to appropriate test module
5. Verify CI passes

### Updating Existing Tests

- Update when requirements change
- Preserve regression test data
- Document breaking changes
- Maintain backward compatibility when possible

### Removing Tests

- Deprecate before removal
- Document reason for removal
- Check for dependent tests
- Update coverage reports

## Test Performance

### Execution Times (Approximate)

| Test Suite | Tests | Time | Per Test |
|------------|-------|------|----------|
| Unit Tests | 300+ | ~60s | ~200ms |
| Integration Tests | 120+ | ~180s | ~1.5s |
| Regression Tests | 50+ | ~120s | ~2.4s |
| Performance Tests | 20+ | ~300s | ~15s |
| **Total** | **500+** | **~10min** | **~1.2s** |

### Optimization Strategies

- Use pytest-xdist for parallel execution
- Cache expensive fixtures
- Mock external dependencies
- Use fast approximations for smoke tests
- Profile slow tests with pytest-profiling

## Test Data Management

### Fixtures

```python
@pytest.fixture
def market_data():
    """Sample market data for testing."""
    return {
        "spot": 100.0,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.02,
        "volatility": 0.20
    }
```

### Reference Data

- Stored in `tests/data/` directory
- Versioned with git
- Documented in README files
- Updated when models change

## Future Enhancements

- [ ] Property-based testing with Hypothesis
- [ ] Mutation testing for test quality
- [ ] Contract testing for APIs
- [ ] Chaos engineering tests
- [ ] Load testing for distributed systems

---

**Last Updated:** November 2025
**Test Count:** 500+
**Coverage:** 87%
**Status:** Production-ready ✅
