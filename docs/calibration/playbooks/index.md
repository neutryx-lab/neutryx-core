# Calibration Playbooks

Production-ready playbooks providing step-by-step workflows for calibrating quantitative models in real-world scenarios. Each playbook includes complete implementation patterns, validation procedures, and operational best practices.

## Overview

Calibration playbooks are structured workflows designed for:

- **Repeatability**: Standardized processes for consistent results
- **Production readiness**: Enterprise-grade error handling and validation
- **Team collaboration**: Clear documentation for knowledge transfer
- **Operational excellence**: Monitoring, alerting, and governance procedures

---

## Playbook Catalog

### 1. Daily SABR Calibration for FX Volatility Surfaces

**Use Case**: Daily recalibration of SABR parameters for FX option books

**Frequency**: Daily, pre-market

**Assets**: Major currency pairs (EUR/USD, USD/JPY, GBP/USD, etc.)

**Workflow:**

```python
# 1. Data Ingestion
from neutryx.market import load_fx_volatility_surface
from neutryx.calibration.sabr import SABRCalibrator
import jax.numpy as jnp

def daily_sabr_calibration(currency_pair: str, trade_date: str):
    """Daily SABR calibration workflow."""

    # Step 1: Load market data
    surface = load_fx_volatility_surface(currency_pair, trade_date)
    forward = surface.forward
    strikes = surface.strikes
    maturities = surface.maturities
    vols = surface.implied_vols

    # Step 2: Apply ATM weighting
    log_moneyness = jnp.log(strikes / forward)
    weights = jnp.exp(-log_moneyness**2 / 0.05)  # Gaussian ATM weight

    market_data = {
        "forward": forward,
        "strikes": strikes,
        "maturities": maturities,
        "target_vols": vols,
        "weights": weights
    }

    # Step 3: Calibrate with validation
    calibrator = SABRCalibrator(max_steps=500, tol=1e-8)
    result = calibrator.calibrate(market_data)

    # Step 4: Validation checks
    if not result.converged:
        raise ValueError(f"Calibration did not converge for {currency_pair}")

    diagnostics = generate_calibration_diagnostics(result, market_data)
    if diagnostics.rmse > 0.005:  # 50 bps threshold
        raise ValueError(f"RMSE too high: {diagnostics.rmse:.4f}")

    # Step 5: Store parameters and diagnostics
    store_calibration_result(currency_pair, trade_date, result, diagnostics)

    return result
```

**Validation Gates:**
- ✓ Convergence achieved within max iterations
- ✓ RMSE < 50 bps across strikes
- ✓ Parameter stability (< 20% change from prior day)
- ✓ Feller condition satisfied (2*kappa*theta > sigma^2 for Heston analog)
- ✓ No arbitrage violations

**Monitoring Metrics:**
- Calibration time (target: < 500ms per pair)
- RMSE distribution across all pairs
- Parameter drift vs historical ranges
- Failure rate (target: < 1%)

---

### 2. Equity Index Heston Calibration

**Use Case**: Weekly calibration of Heston model for equity index options (SPX, NDX, STOXX)

**Frequency**: Weekly, end-of-week

**Workflow:**

```python
from neutryx.calibration.heston import HestonCalibrator
from neutryx.calibration.diagnostics import compute_identifiability_metrics

def weekly_heston_calibration(index: str, date: str):
    """Weekly Heston calibration with identifiability checks."""

    # Step 1: Load option surface
    options = load_equity_option_surface(index, date)

    # Step 2: Filter liquid strikes (open interest > threshold)
    liquid_mask = options.open_interest > 100
    strikes = options.strikes[liquid_mask]
    maturities = options.maturities[liquid_mask]
    prices = options.market_prices[liquid_mask]

    market_data = {
        "spot": options.spot,
        "strikes": strikes,
        "maturities": maturities,
        "target_prices": prices,
        "rate": get_risk_free_rate(date),
        "dividend": get_dividend_yield(index, date)
    }

    # Step 3: Multi-start calibration (avoid local minima)
    best_result = None
    best_loss = float('inf')

    for init_v0 in [0.02, 0.04, 0.06]:
        calibrator = HestonCalibrator()
        calibrator.parameter_specs["v0"].init = init_v0

        result = calibrator.calibrate(market_data)
        final_loss = result.loss_history[-1]

        if result.converged and final_loss < best_loss:
            best_loss = final_loss
            best_result = result

    if best_result is None:
        raise ValueError("All calibration attempts failed")

    # Step 4: Identifiability check
    id_metrics = compute_identifiability_metrics(
        calibrator, best_result.params, market_data
    )

    if id_metrics.condition_number > 200:
        warnings.warn(f"Poor identifiability: CN={id_metrics.condition_number:.1f}")

    # Step 5: Cross-validation
    cv_result = cross_validate(
        calibrator,
        market_data,
        k_fold_split(5)
    )

    if cv_result.std_error / cv_result.mean_error > 0.3:
        warnings.warn("High CV variance, possible overfitting")

    return best_result, id_metrics, cv_result
```

**Validation Gates:**
- ✓ Multi-start convergence agreement (< 5% parameter difference)
- ✓ Condition number < 200
- ✓ Cross-validation std/mean < 30%
- ✓ Feller condition: 2*kappa*theta > sigma^2
- ✓ Price errors < 2% for liquid strikes

---

### 3. Multi-Asset Joint Calibration

**Use Case**: Joint calibration across correlated equity indices with shared correlation parameter

**Frequency**: Monthly, or after major market events

**Workflow:**

```python
from neutryx.calibration.joint_calibration import CrossAssetCalibrator, AssetClassSpec

def multi_asset_joint_calibration(indices: List[str], date: str):
    """Joint calibration across multiple equity indices."""

    # Step 1: Load data for all assets
    asset_data = {}
    for index in indices:
        surface = load_equity_option_surface(index, date)
        asset_data[index] = prepare_heston_data(surface)

    # Step 2: Define cross-asset calibrator with shared rho
    asset_specs = [
        AssetClassSpec(name=idx, weight=1.0, shared_params=["rho"])
        for idx in indices
    ]

    calibrator = CrossAssetCalibrator(
        asset_specs=asset_specs,
        base_calibrator=HestonCalibrator()
    )

    # Step 3: Calibrate jointly
    result = calibrator.calibrate(asset_data)

    # Step 4: Compare to individual calibrations
    individual_results = {}
    for index in indices:
        ind_calibrator = HestonCalibrator()
        individual_results[index] = ind_calibrator.calibrate(asset_data[index])

    # Step 5: Analyze consistency
    consistency_report = analyze_parameter_consistency(
        joint_result=result,
        individual_results=individual_results
    )

    return result, consistency_report
```

**Benefits:**
- Enforces cross-asset consistency for portfolio-level risk
- Reduces parameter uncertainty through pooled estimation
- Shared correlation captures market-wide risk factors
- Better stability during market stress

---

### 4. Time-Dependent Volatility Surface Calibration

**Use Case**: Calibrate piecewise-constant term structure for volatility parameters

**Frequency**: As needed for term structure modeling

**Workflow:**

```python
from neutryx.calibration.joint_calibration import TimeDependentCalibrator, TimeSegment
from neutryx.calibration.regularization import SmoothnessRegularization

def term_structure_calibration(asset: str, date: str):
    """Time-dependent calibration with smoothness regularization."""

    # Step 1: Define time segments
    segments = [
        TimeSegment(start=0.0, end=0.5, params=["theta", "v0"]),    # 0-6m
        TimeSegment(start=0.5, end=1.0, params=["theta", "v0"]),    # 6m-1y
        TimeSegment(start=1.0, end=3.0, params=["theta", "v0"]),    # 1y-3y
        TimeSegment(start=3.0, end=5.0, params=["theta", "v0"]),    # 3y-5y
        TimeSegment(start=5.0, end=10.0, params=["theta", "v0"])    # 5y-10y
    ]

    # Step 2: Load full surface
    surface = load_equity_option_surface(asset, date)
    market_data = prepare_full_surface_data(surface)

    # Step 3: Apply smoothness regularization
    smoothness_penalty = SmoothnessRegularization(
        time_segments=segments,
        weight=0.05  # Balance fit vs smoothness
    )

    # Step 4: Calibrate time-dependent model
    calibrator = TimeDependentCalibrator(
        time_segments=segments,
        base_calibrator=HestonCalibrator(penalty_fn=smoothness_penalty)
    )

    result = calibrator.calibrate(market_data)

    # Step 5: Verify no calendar arbitrage
    check_calendar_arbitrage(result, segments)

    return result
```

**Validation Gates:**
- ✓ No calendar arbitrage (forward variance positive and increasing)
- ✓ Smooth parameter evolution (no jumps > 30% between segments)
- ✓ Feller condition in each segment
- ✓ Out-of-sample validation on recent dates

---

### 5. Model Comparison and Selection

**Use Case**: Compare SABR, Heston, and SLV models to select best fit

**Frequency**: Quarterly model validation

**Workflow:**

```python
from neutryx.calibration.model_selection import compare_models, cross_validate

def quarterly_model_selection(asset: str, date: str):
    """Comprehensive model comparison workflow."""

    # Step 1: Prepare market data
    market_data = load_and_prepare_surface(asset, date)

    # Step 2: Calibrate all candidate models
    sabr_calibrator = SABRCalibrator()
    heston_calibrator = HestonCalibrator()
    slv_calibrator = SLVCalibrator()

    sabr_result = sabr_calibrator.calibrate(market_data)
    heston_result = heston_calibrator.calibrate(market_data)
    slv_result = slv_calibrator.calibrate(market_data)

    # Step 3: Information criteria comparison
    comparison = compare_models([
        ("SABR", sabr_result, market_data),
        ("Heston", heston_result, market_data),
        ("SLV", slv_result, market_data)
    ])

    print(f"Best by AIC: {comparison.best_by_aic}")
    print(f"Best by BIC: {comparison.best_by_bic}")

    # Step 4: Cross-validation
    cv_results = {}
    for name, calibrator in [("SABR", sabr_calibrator),
                             ("Heston", heston_calibrator),
                             ("SLV", slv_calibrator)]:
        cv_results[name] = cross_validate(
            calibrator, market_data, k_fold_split(5)
        )

    # Step 5: Out-of-sample testing
    historical_dates = get_prior_dates(date, n_days=20)
    oos_errors = {}

    for model_name, calibrator, result in [
        ("SABR", sabr_calibrator, sabr_result),
        ("Heston", heston_calibrator, heston_result),
        ("SLV", slv_calibrator, slv_result)
    ]:
        oos_errors[model_name] = evaluate_out_of_sample(
            calibrator, result, historical_dates
        )

    # Step 6: Generate recommendation
    recommendation = generate_model_recommendation(
        comparison, cv_results, oos_errors
    )

    return recommendation
```

**Decision Criteria:**
1. **In-sample fit**: AIC/BIC for parsimony
2. **Stability**: Cross-validation error variance
3. **Generalization**: Out-of-sample predictive accuracy
4. **Operational**: Calibration speed, convergence rate
5. **Risk**: Parameter uncertainty, identifiability

---

### 6. Real-Time Calibration Architecture

**Use Case**: Low-latency calibration for intraday risk updates

**Requirements**:
- Latency: < 100ms per calibration
- Throughput: 1000+ calibrations/sec
- Reliability: 99.9% success rate

**Architecture:**

```python
import jax
from functools import partial

# JIT-compile calibration for speed
@jax.jit
def jitted_calibration_step(params, market_data, optimizer_state):
    """Single optimization step, JIT-compiled."""
    loss, grads = jax.value_and_grad(loss_function)(params, market_data)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, loss, optimizer_state

class RealTimeCalibrationService:
    """Production calibration service with caching and monitoring."""

    def __init__(self):
        self.calibrator = SABRCalibrator()
        self.param_cache = {}  # Cache prior calibration as warm start

    def calibrate_with_warmstart(self, asset: str, market_data: dict):
        """Calibrate using prior parameters as initial guess."""

        # Use cached parameters as initialization
        if asset in self.param_cache:
            for param_name, param_value in self.param_cache[asset].items():
                self.calibrator.parameter_specs[param_name].init = param_value

        # Time the calibration
        start = time.perf_counter()
        result = self.calibrator.calibrate(market_data)
        elapsed = time.perf_counter() - start

        # Update cache
        if result.converged:
            self.param_cache[asset] = result.params

        # Log metrics
        log_calibration_metrics(
            asset=asset,
            elapsed_ms=elapsed * 1000,
            iterations=result.iterations,
            converged=result.converged,
            rmse=compute_rmse(result, market_data)
        )

        return result

# Batch processing with JAX vmap
@partial(jax.vmap, in_axes=(0, 0))
def batch_calibrate(market_data_batch, init_params_batch):
    """Vectorized batch calibration."""
    return single_calibration(market_data_batch, init_params_batch)
```

**Optimizations:**
- JIT compilation for 10-100x speedup
- Warm-start initialization from prior calibration
- Batch processing with `jax.vmap`
- GPU acceleration for large batches
- Parameter caching and versioning

---

### 7. Calibration Monitoring and Alerting

**Use Case**: Production monitoring dashboard and alert system

**Metrics to Track:**

1. **Performance Metrics**:
   - Calibration latency (p50, p95, p99)
   - Success rate by asset class
   - Iterations to convergence

2. **Quality Metrics**:
   - RMSE distribution
   - Parameter stability (drift detection)
   - Identifiability scores

3. **Operational Metrics**:
   - Data ingestion lag
   - Cache hit rate
   - System resource utilization

**Alerting Rules:**

```python
def check_calibration_health(result, diagnostics, historical_stats):
    """Health checks with alerting."""

    alerts = []

    # Check 1: Convergence
    if not result.converged:
        alerts.append(Alert(
            severity="CRITICAL",
            message="Calibration failed to converge"
        ))

    # Check 2: Fit quality
    if diagnostics.rmse > historical_stats.p95_rmse:
        alerts.append(Alert(
            severity="WARNING",
            message=f"RMSE {diagnostics.rmse:.4f} exceeds P95 threshold"
        ))

    # Check 3: Parameter drift
    for param, value in result.params.items():
        historical_mean = historical_stats.params[param].mean
        historical_std = historical_stats.params[param].std

        z_score = abs(value - historical_mean) / historical_std
        if z_score > 3.0:
            alerts.append(Alert(
                severity="WARNING",
                message=f"Parameter {param} drift: z-score={z_score:.2f}"
            ))

    # Check 4: Identifiability
    if diagnostics.identifiability.condition_number > 500:
        alerts.append(Alert(
            severity="WARNING",
            message="Poor parameter identifiability"
        ))

    return alerts
```

---

### 8. Calibration Governance and Audit

**Use Case**: Model risk management and regulatory compliance

**Requirements**:
- Parameter versioning and lineage
- Reproducibility of historical calibrations
- Audit trail for all calibration decisions
- Documentation of model changes

**Implementation:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass
class CalibrationRecord:
    """Immutable calibration record for audit trail."""

    asset: str
    calibration_date: datetime
    model_type: str
    model_version: str
    parameters: Dict[str, float]
    market_data_hash: str
    diagnostics: Dict[str, Any]
    operator: str
    validation_status: str
    approver: Optional[str] = None

def audit_trail_calibration(asset: str, market_data: dict):
    """Calibration with full audit trail."""

    # Step 1: Hash market data for reproducibility
    data_hash = hash_market_data(market_data)

    # Step 2: Calibrate
    calibrator = SABRCalibrator()
    result = calibrator.calibrate(market_data)
    diagnostics = generate_calibration_diagnostics(result, market_data)

    # Step 3: Create audit record
    record = CalibrationRecord(
        asset=asset,
        calibration_date=datetime.utcnow(),
        model_type="SABR",
        model_version=get_model_version(),
        parameters=result.params,
        market_data_hash=data_hash,
        diagnostics={
            "rmse": diagnostics.rmse,
            "converged": result.converged,
            "iterations": result.iterations
        },
        operator=get_current_user(),
        validation_status="PENDING"
    )

    # Step 4: Store immutable record
    store_calibration_record(record)

    # Step 5: Trigger approval workflow if needed
    if requires_approval(asset, result, diagnostics):
        trigger_approval_workflow(record)

    return result, record
```

---

## Complete Playbooks

For comprehensive, end-to-end playbooks with production code, see:

**[Full-Spectrum Calibration Playbook](full_spectrum_calibration.md)**

Includes:
- Complete production implementation patterns
- Error handling and retry logic
- Performance optimization strategies
- Deployment architectures
- Monitoring and observability
- Regulatory compliance procedures

---

## Playbook Templates

### Basic Playbook Structure

```markdown
# Playbook: [Name]

## Objective
[What problem does this solve?]

## Prerequisites
- Data requirements
- System dependencies
- Prior calibrations needed

## Workflow
1. Step 1: [Action]
   - Code example
   - Expected output
   - Validation

2. Step 2: [Action]
   ...

## Validation Gates
- ✓ Check 1
- ✓ Check 2

## Rollback Procedure
[What to do if calibration fails?]

## Monitoring
[What metrics to track?]

## Success Criteria
[How to know it worked?]
```

---

## Best Practices

### Calibration Workflow

1. **Pre-calibration**:
   - Validate input data quality
   - Remove outliers and stale quotes
   - Apply liquidity filters
   - Check for market anomalies

2. **During calibration**:
   - Use sensible initial values
   - Monitor convergence progress
   - Apply appropriate constraints
   - Log all intermediate states

3. **Post-calibration**:
   - Validate convergence
   - Check fit quality (RMSE, residuals)
   - Verify no-arbitrage conditions
   - Compare to prior calibration
   - Archive results

### Production Deployment

1. **Testing**:
   - Unit tests for each component
   - Integration tests for full workflow
   - Regression tests with historical data
   - Stress tests with synthetic data

2. **Monitoring**:
   - Real-time performance metrics
   - Quality metrics trending
   - Alert thresholds and escalation
   - Daily summary reports

3. **Governance**:
   - Version control for models and data
   - Audit trail for all calibrations
   - Approval workflows for material changes
   - Regular model validation reviews

---

## Operational Runbooks

### Calibration Failure Response

**Symptom**: Calibration fails to converge

**Diagnosis**:
1. Check market data quality
2. Inspect loss history for plateau/divergence
3. Review parameter initial values
4. Check for constraint violations

**Resolution**:
1. Re-run with adjusted initial values
2. Increase max_steps or adjust tolerance
3. Try alternative optimizer (Adam → LBFGS)
4. Simplify model (fewer free parameters)
5. Escalate to quant team if unresolved

---

### Parameter Drift Alert Response

**Symptom**: Parameter changes > 3 sigma from mean

**Diagnosis**:
1. Check for market regime change
2. Verify data quality (no errors?)
3. Review recent corporate actions
4. Compare to peer assets

**Resolution**:
1. If justified (regime change): Update historical stats
2. If data error: Re-run with corrected data
3. If spurious: Override and flag for review
4. Document decision in audit log

---

## Next Steps

- **Learn the fundamentals**: Start with [Tutorials](../tutorials/index.md)
- **Deep dive API**: Review [Reference](../reference/index.md) documentation
- **Understand theory**: Study [Calibration Theory](../../theory/calibration_theory.md)

---

## Contributing

Have a playbook to share? [Submit a pull request](https://github.com/neutryx-lab/neutryx-core/pulls) or [open an issue](https://github.com/neutryx-lab/neutryx-core/issues).
