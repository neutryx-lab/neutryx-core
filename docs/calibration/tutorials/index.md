# Calibration Tutorials

Practical, hands-on tutorials that demonstrate calibration workflows from basic to advanced topics. Each tutorial includes complete code examples, explanations, and best practices.

## Learning Path

Follow this recommended learning path to master calibration in Neutryx:

### 1. Fundamentals (Start Here)

**Getting Started with SABR**
Learn the basics of model calibration by fitting a SABR model to FX or interest rate volatility data.

```python
from neutryx.calibration.sabr import SABRCalibrator
from neutryx.models.sabr import SABRParams
import jax.numpy as jnp

# Your first calibration
calibrator = SABRCalibrator()
market_data = {
    "forward": 1.10,
    "strikes": jnp.array([1.00, 1.05, 1.10, 1.15, 1.20]),
    "maturities": jnp.array([0.25, 0.5, 1.0]),
    "target_vols": jnp.array([0.15, 0.14, 0.13, 0.14, 0.16])
}
result = calibrator.calibrate(market_data)
```

**Topics covered:**
- Setting up market data
- Running a basic calibration
- Interpreting calibration results
- Checking convergence

**Estimated time:** 15 minutes

---

**Understanding Parameters and Constraints**
Deep dive into parameter specifications, transformations, and bounds.

```python
from neutryx.calibration.base import ParameterSpec
from neutryx.calibration.constraints import positive, bounded, symmetric

# Define constrained parameters
params = {
    "alpha": ParameterSpec(init=0.2, transform=positive()),        # α > 0
    "beta": ParameterSpec(init=0.5, transform=bounded(0.0, 1.0)),  # 0 ≤ β < 1
    "rho": ParameterSpec(init=-0.3, transform=symmetric(0.99)),    # |ρ| < 0.99
    "nu": ParameterSpec(init=0.4, transform=positive())            # ν > 0
}
```

**Topics covered:**
- Parameter transformation functions
- Enforcing bounds and constraints
- Custom constraint functions
- Initial value selection strategies

**Estimated time:** 20 minutes

---

### 2. Model-Specific Calibration

**Heston Model for Equity Options**
Calibrate the Heston stochastic volatility model to SPX or equity index options.

```python
from neutryx.calibration.heston import HestonCalibrator
from neutryx.models.heston import HestonParams

# Calibrate to equity volatility surface
calibrator = HestonCalibrator()
result = calibrator.calibrate({
    "spot": 4500.0,
    "strikes": strikes,        # Array of strike prices
    "maturities": maturities,  # Array of maturities
    "target_prices": prices,   # Market option prices
    "rate": 0.05,
    "dividend": 0.02
})

print(f"v0={result.params['v0']:.4f}, kappa={result.params['kappa']:.4f}")
```

**Topics covered:**
- Heston model parameters and interpretation
- Fourier-based pricing in calibration
- Handling equity-specific features (dividends)
- Common calibration challenges (local minima)

**Estimated time:** 30 minutes

---

**Stochastic-Local Volatility (SLV) Calibration**
Fit hybrid SLV models for complex payoff valuations.

```python
from neutryx.calibration.slv import SLVCalibrator

calibrator = SLVCalibrator()
result = calibrator.calibrate(market_data)

# SLV provides flexible smile interpolation
print(f"Local slope: {result.params['local_slope']:.4f}")
print(f"Mixing parameter: {result.params['mixing']:.4f}")
```

**Topics covered:**
- SLV model structure and use cases
- Balancing local vs stochastic components
- Calibration to exotic option prices
- Stability across strikes and maturities

**Estimated time:** 30 minutes

---

### 3. Advanced Topics

**Multi-Instrument Joint Calibration**
Calibrate to multiple instrument types simultaneously for consistency.

```python
from neutryx.calibration.joint_calibration import (
    MultiInstrumentCalibrator,
    InstrumentSpec
)

# Calibrate to calls, puts, and variance swaps jointly
calibrator = MultiInstrumentCalibrator([
    InstrumentSpec(name="calls", weight=1.0, loss_fn="rmse"),
    InstrumentSpec(name="puts", weight=1.0, loss_fn="rmse"),
    InstrumentSpec(name="var_swaps", weight=2.0, loss_fn="relative")
])

result = calibrator.calibrate({
    "calls": call_data,
    "puts": put_data,
    "var_swaps": variance_swap_data
})
```

**Topics covered:**
- Multi-instrument data structures
- Loss function weighting strategies
- Resolving conflicting instrument signals
- Put-call parity consistency checks

**Estimated time:** 45 minutes

---

**Time-Dependent Calibration**
Calibrate piecewise-constant parameters across time segments.

```python
from neutryx.calibration.joint_calibration import (
    TimeDependentCalibrator,
    TimeSegment
)

# Piece-wise constant volatility term structure
calibrator = TimeDependentCalibrator([
    TimeSegment(start=0.0, end=1.0, params=["theta"]),    # Short term
    TimeSegment(start=1.0, end=5.0, params=["theta"]),    # Medium term
    TimeSegment(start=5.0, end=10.0, params=["theta"])    # Long term
])

result = calibrator.calibrate(full_surface_data)
```

**Topics covered:**
- Time segment definition
- Smoothness regularization
- Calendar arbitrage constraints
- Parameter evolution analysis

**Estimated time:** 45 minutes

---

**Cross-Asset Calibration**
Joint calibration with shared parameters across multiple assets.

```python
from neutryx.calibration.joint_calibration import (
    CrossAssetCalibrator,
    AssetClassSpec
)

# Calibrate equity indices with shared correlation
calibrator = CrossAssetCalibrator([
    AssetClassSpec(name="SPX", weight=1.0, shared_params=["rho"]),
    AssetClassSpec(name="NDX", weight=0.8, shared_params=["rho"]),
    AssetClassSpec(name="RTY", weight=0.6, shared_params=["rho"])
])

result = calibrator.calibrate(multi_asset_data)
```

**Topics covered:**
- Shared parameter specifications
- Asset-specific vs global parameters
- Cross-asset consistency constraints
- Portfolio-level risk implications

**Estimated time:** 45 minutes

---

### 4. Diagnostics & Validation

**Calibration Diagnostics Deep Dive**
Comprehensive post-calibration analysis and quality checks.

```python
from neutryx.calibration.diagnostics import (
    generate_calibration_diagnostics,
    build_residual_plot_data,
    compute_identifiability_metrics
)

# Generate full diagnostic report
diagnostics = generate_calibration_diagnostics(result, market_data)

print(f"RMSE: {diagnostics.rmse:.4f}")
print(f"Max error: {diagnostics.max_abs_error:.4f}")
print(f"Condition number: {diagnostics.identifiability.condition_number:.2f}")

# Visualize residuals
residual_data = build_residual_plot_data(result, market_data)
# ... plotting code
```

**Topics covered:**
- Residual analysis (strikes, maturities, moneyness)
- Parameter identifiability and covariance
- Convergence diagnostics
- Red flags and warning signs
- Iteration and retry strategies

**Estimated time:** 40 minutes

---

**Model Selection and Comparison**
Statistical comparison of competing models using information criteria.

```python
from neutryx.calibration.model_selection import compare_models

# Compare SABR, Heston, and SLV
comparison = compare_models([
    ("SABR", sabr_result, market_data),
    ("Heston", heston_result, market_data),
    ("SLV", slv_result, market_data)
])

print(f"Best by AIC: {comparison.best_by_aic}")
print(f"Best by BIC: {comparison.best_by_bic}")
print(f"AIC weights: {comparison.aic_weights}")
```

**Topics covered:**
- AIC, BIC, AICc, HQIC information criteria
- Model complexity penalties
- Overfitting detection
- Model selection best practices

**Estimated time:** 35 minutes

---

**Cross-Validation Strategies**
Validate calibration robustness with k-fold and time-series CV.

```python
from neutryx.calibration.model_selection import (
    cross_validate,
    k_fold_split,
    time_series_split
)

# K-fold cross-validation
cv_result = cross_validate(
    calibrator=calibrator,
    data=market_data,
    split_fn=k_fold_split(n_folds=5),
    n_trials=10
)

print(f"Mean CV error: {cv_result.mean_error:.4f}")
print(f"Std CV error: {cv_result.std_error:.4f}")

# Time-series walk-forward validation
ts_result = cross_validate(
    calibrator=calibrator,
    data=time_series_data,
    split_fn=time_series_split(n_splits=10)
)
```

**Topics covered:**
- K-fold cross-validation setup
- Time-series splitting strategies
- Out-of-sample validation metrics
- Overfitting diagnosis
- Production model validation

**Estimated time:** 40 minutes

---

**Sensitivity Analysis**
Analyze parameter sensitivity and uncertainty propagation.

```python
from neutryx.calibration.model_selection import (
    compute_local_sensitivity,
    sobol_indices
)

# Local sensitivity (gradient-based)
local_sens = compute_local_sensitivity(
    calibrator,
    result.params,
    market_data
)

# Global sensitivity (variance-based)
param_ranges = {
    "alpha": (0.1, 0.5),
    "rho": (-0.8, -0.2),
    "nu": (0.2, 0.8)
}

global_sens = sobol_indices(
    calibrator,
    param_ranges,
    market_data,
    n_samples=2048
)

print(f"First-order Sobol indices: {global_sens.first_order}")
print(f"Total Sobol indices: {global_sens.total}")
```

**Topics covered:**
- Local vs global sensitivity
- Finite difference gradients
- Sobol variance decomposition
- Parameter uncertainty quantification
- Risk factor importance ranking

**Estimated time:** 50 minutes

---

### 5. Advanced Techniques

**Bayesian Model Averaging**
Combine multiple models using Bayesian model averaging for robust predictions.

```python
from neutryx.calibration.bayesian_model_averaging import (
    BayesianModelAveraging,
    WeightingScheme
)

# Fit ensemble of models
bma = BayesianModelAveraging(
    models=[sabr_calibrator, heston_calibrator, slv_calibrator],
    weighting_scheme=WeightingScheme.STACKING
)

bma_result = bma.fit(training_data)
predictions = bma.predict(new_market_conditions)

print(f"Model weights: {bma_result.weights}")
print(f"Ensemble predictions: {predictions}")
```

**Topics covered:**
- BMA theoretical foundations
- Weighting schemes (stacking, pseudo-BMA, IC-based)
- Ensemble prediction
- Model uncertainty quantification
- Production deployment considerations

**Estimated time:** 50 minutes

---

**Regularization Techniques**
Apply regularization for stable, smooth parameter evolution.

```python
from neutryx.calibration.regularization import (
    L2Regularization,
    SmoothnessRegularization
)

# L2 penalty on parameter deviations
l2_penalty = L2Regularization(
    reference_params=historical_params,
    weights={"alpha": 0.01, "nu": 0.01}
)

# Smoothness penalty for time-dependent calibration
smoothness_penalty = SmoothnessRegularization(
    time_segments=segments,
    weight=0.05
)

calibrator = MyCalibrator(penalty_fn=l2_penalty)
```

**Topics covered:**
- L1/L2 regularization
- Smoothness constraints
- Historical parameter anchoring
- Regularization strength tuning
- Bias-variance tradeoff

**Estimated time:** 40 minutes

---

**Custom Loss Functions**
Design custom loss functions for specific calibration objectives.

```python
from neutryx.calibration import losses
import jax.numpy as jnp

def custom_weighted_loss(model_vals, target_vals, weights=None):
    """Emphasize ATM options and recent expiries."""
    errors = model_vals - target_vals

    # Weight by moneyness
    atm_weights = jnp.exp(-moneyness**2 / 0.05)

    # Weight by time to maturity
    time_weights = jnp.exp(-maturities / 2.0)

    # Combined weights
    combined = atm_weights * time_weights
    if weights is not None:
        combined = combined * weights

    return jnp.sqrt(jnp.mean(combined * errors**2))

calibrator = MyCalibrator(loss_fn=custom_weighted_loss)
```

**Topics covered:**
- Standard loss functions (RMSE, MAE, MAPE)
- Custom weighting schemes
- Robust loss functions (Huber)
- Multi-objective calibration
- Domain-specific loss design

**Estimated time:** 45 minutes

---

## Complete Masterclass

For a comprehensive, end-to-end calibration workshop covering all topics above in depth, see:

**[Calibration Masterclass and Tuning Workshop](calibration_masterclass.md)**

This advanced curriculum includes:
- Complete theory-to-practice walkthroughs
- Production deployment patterns
- Performance optimization techniques
- Troubleshooting guides
- Real-world case studies

**Estimated time:** 6-8 hours

---

## Quick Reference

### Common Calibration Patterns

**Pattern 1: Basic Single-Model Calibration**
```python
calibrator = SABRCalibrator()
result = calibrator.calibrate(market_data)
diagnostics = generate_calibration_diagnostics(result, market_data)
```

**Pattern 2: Multi-Model Comparison**
```python
models = [SABRCalibrator(), HestonCalibrator(), SLVCalibrator()]
results = [m.calibrate(market_data) for m in models]
comparison = compare_models(zip(["SABR", "Heston", "SLV"], results, [market_data]*3))
```

**Pattern 3: Cross-Validated Calibration**
```python
calibrator = MyCalibrator()
cv_result = cross_validate(calibrator, market_data, k_fold_split(5))
final_result = calibrator.calibrate(market_data) if cv_result.mean_error < threshold else None
```

---

## Troubleshooting Guide

### Calibration Not Converging?

1. **Check initial values**: Use `default_parameter_specs()` for sensible defaults
2. **Inspect loss history**: Look for plateaus or divergence patterns
3. **Adjust optimizer**: Try different learning rates or switch optimizers
4. **Add constraints**: Ensure parameter bounds match model requirements
5. **Simplify the problem**: Start with fewer parameters or data points

### Poor Fit Quality?

1. **Check residual patterns**: Systematic errors indicate model misspecification
2. **Increase weighting**: Weight ATM options or liquid strikes more heavily
3. **Try different models**: SABR vs Heston vs SLV have different smile capabilities
4. **Add data**: More market quotes improve identifiability
5. **Regularize**: Add smoothness or historical anchoring penalties

### Unstable Parameters?

1. **Check identifiability**: High condition numbers indicate collinearity
2. **Cross-validate**: High CV error variance indicates overfitting
3. **Regularize**: Add L2 penalty to stabilize parameter estimates
4. **Simplify model**: Reduce number of free parameters
5. **Add prior information**: Use parameter bounds based on historical ranges

---

## Next Steps

- **API Documentation**: See [Reference](../reference/index.md) for detailed API docs
- **Production Workflows**: Explore [Playbooks](../playbooks/index.md) for enterprise patterns
- **Theory**: Study [Calibration Theory](../../theory/calibration_theory.md) for mathematical foundations

---

## Feedback

Found an issue or have a suggestion? [Open an issue on GitHub](https://github.com/neutryx-lab/neutryx-core/issues)
