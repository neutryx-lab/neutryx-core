# Calibration Hub

The Calibration Hub provides comprehensive tools, workflows, and methodologies for calibrating quantitative finance models to market data. Built on JAX for high-performance automatic differentiation, the calibration framework supports both simple single-instrument calibrations and complex multi-asset, time-dependent calibration campaigns.

## Overview

Model calibration is the process of determining parameter values that minimize the difference between model-predicted prices (or volatilities) and observed market data. Neutryx provides a flexible, gradient-based calibration framework that handles:

- **Multiple volatility models**: SABR, Heston, Stochastic-Local Volatility (SLV)
- **Advanced diagnostics**: Parameter identifiability, residual analysis, convergence monitoring
- **Multi-dimensional calibration**: Joint calibration across instruments, asset classes, and time segments
- **Model comparison**: Information criteria (AIC/BIC), cross-validation, sensitivity analysis
- **Robust optimization**: Automatic differentiation, constraint handling, regularization

## Quick Start

```python
from neutryx.calibration.sabr import SABRCalibrator, generate_sabr_market_data
from neutryx.models.sabr import SABRParams
import jax.numpy as jnp

# Generate sample market data
true_params = SABRParams(alpha=0.25, beta=0.5, rho=-0.3, nu=0.4)
market_data = generate_sabr_market_data(
    forward=100.0,
    strikes=jnp.array([80, 90, 100, 110, 120]),
    maturities=jnp.array([0.25, 0.5, 1.0, 2.0, 5.0]),
    params=true_params
)

# Calibrate the model
calibrator = SABRCalibrator()
result = calibrator.calibrate(market_data)

print(f"Calibrated parameters: {result.params}")
print(f"Converged: {result.converged} in {result.iterations} iterations")
```

## Supported Models

### SABR Model
The Stochastic Alpha Beta Rho (SABR) model is widely used for interest rate and FX volatility surfaces:

- **Parameters**: `alpha` (volatility level), `beta` (backbone), `rho` (correlation), `nu` (vol-of-vol)
- **Use cases**: Interest rate swaptions, FX options, equity volatility smiles
- **Calibration**: Fast semi-analytic Hagan approximation with automatic differentiation

See [`SABRCalibrator`](reference/calibration_reference_compendium.md#sabr-calibrator) for details.

### Heston Model
The Heston stochastic volatility model for equity and index options:

- **Parameters**: `v0` (initial variance), `kappa` (mean reversion), `theta` (long-run variance), `sigma` (vol-of-vol), `rho` (correlation)
- **Use cases**: Equity options, variance swaps, volatility derivatives
- **Calibration**: Fourier-based pricing with gradient-based optimization

See [`HestonCalibrator`](reference/calibration_reference_compendium.md#heston-calibrator) for details.

### Stochastic-Local Volatility (SLV)
A hybrid model combining local and stochastic volatility features:

- **Parameters**: `base_vol`, `local_slope`, `local_curvature`, `mixing`, `time_decay`
- **Use cases**: Exotic derivatives, multi-asset products, model risk management
- **Calibration**: Flexible parametric approach with smooth smile interpolation

See [`SLVCalibrator`](reference/calibration_reference_compendium.md#slv-calibrator) for details.

## Core Components

### CalibrationController
Base class for all calibration workflows providing:

- **Parameter transformations**: Enforce bounds and constraints automatically
- **Gradient-based optimization**: Powered by JAX + Optax optimizers
- **Loss functions**: RMSE, weighted MSE, relative error with custom penalties
- **Convergence monitoring**: Tolerance-based stopping with loss history tracking

```python
from neutryx.calibration.base import CalibrationController, ParameterSpec
from neutryx.calibration.constraints import positive, bounded
import optax

# Define parameter specifications
param_specs = {
    "vol": ParameterSpec(init=0.2, transform=positive()),
    "mean_reversion": ParameterSpec(init=0.5, transform=bounded(0.0, 5.0))
}

# Create calibration controller
controller = CalibrationController(
    parameter_specs=param_specs,
    loss_fn=my_loss_function,
    optimizer=optax.adam(learning_rate=0.01),
    max_steps=500,
    tol=1e-8
)
```

### Diagnostics & Validation

Comprehensive post-calibration analysis:

- **Residual plots**: Visualize fit quality across strikes and maturities
- **Parameter identifiability**: Detect under-determined or correlated parameters
- **Convergence diagnostics**: Loss history, gradient norms, parameter evolution
- **Model comparison**: AIC/BIC/AICc/HQIC information criteria

```python
from neutryx.calibration.diagnostics import generate_calibration_diagnostics

diagnostics = generate_calibration_diagnostics(result, market_data)
print(f"Residual RMSE: {diagnostics.rmse}")
print(f"Max absolute error: {diagnostics.max_abs_error}")
print(f"Identifiability score: {diagnostics.identifiability.condition_number}")
```

### Joint Calibration

Advanced calibration across multiple dimensions:

**Multi-Instrument Calibration**: Calibrate to multiple option types simultaneously
```python
from neutryx.calibration.joint_calibration import MultiInstrumentCalibrator, InstrumentSpec

calibrator = MultiInstrumentCalibrator([
    InstrumentSpec(name="calls", weight=1.0),
    InstrumentSpec(name="puts", weight=1.0),
    InstrumentSpec(name="variance_swaps", weight=2.0)
])
```

**Cross-Asset Calibration**: Joint calibration with shared parameters across assets
```python
from neutryx.calibration.joint_calibration import CrossAssetCalibrator, AssetClassSpec

calibrator = CrossAssetCalibrator([
    AssetClassSpec(name="SPX", weight=1.0, shared_params=["rho"]),
    AssetClassSpec(name="NDX", weight=0.8, shared_params=["rho"])
])
```

**Time-Dependent Calibration**: Piecewise-constant parameters across time segments
```python
from neutryx.calibration.joint_calibration import TimeDependentCalibrator, TimeSegment

calibrator = TimeDependentCalibrator([
    TimeSegment(start=0.0, end=1.0, params=["v0", "theta"]),
    TimeSegment(start=1.0, end=5.0, params=["v0", "theta"]),
    TimeSegment(start=5.0, end=10.0, params=["v0", "theta"])
])
```

### Model Selection & Validation

Statistical tools for model comparison and validation:

**Information Criteria**
```python
from neutryx.calibration.model_selection import compare_models

comparison = compare_models([
    ("SABR", sabr_result, sabr_data),
    ("Heston", heston_result, heston_data),
    ("SLV", slv_result, slv_data)
])

print(f"Best model by AIC: {comparison.best_by_aic}")
print(f"Best model by BIC: {comparison.best_by_bic}")
```

**Cross-Validation**
```python
from neutryx.calibration.model_selection import cross_validate, k_fold_split

cv_result = cross_validate(
    calibrator=my_calibrator,
    data=market_data,
    split_fn=k_fold_split(n_folds=5)
)

print(f"Mean CV error: {cv_result.mean_error}")
print(f"Std CV error: {cv_result.std_error}")
```

**Sensitivity Analysis**
```python
from neutryx.calibration.model_selection import compute_local_sensitivity, sobol_indices

# Local sensitivity (gradients)
local_sens = compute_local_sensitivity(calibrator, result.params, market_data)

# Global sensitivity (Sobol indices)
global_sens = sobol_indices(calibrator, param_ranges, market_data, n_samples=1000)
```

### Bayesian Model Averaging

Combine predictions from multiple models weighted by their evidence:

```python
from neutryx.calibration.bayesian_model_averaging import BayesianModelAveraging, WeightingScheme

bma = BayesianModelAveraging(
    models=[sabr_model, heston_model, slv_model],
    weighting_scheme=WeightingScheme.STACKING
)

bma_result = bma.fit(calibration_data)
predictions = bma.predict(new_market_conditions)

print(f"Model weights: {bma_result.weights}")
```

## Documentation Structure

### [Tutorials](tutorials/index.md)
Step-by-step guides for common calibration workflows:

- Getting started with SABR calibration
- Calibrating Heston to equity options
- Multi-instrument joint calibration
- Model selection and validation workflows
- Advanced diagnostics and troubleshooting

### [Reference](reference/index.md)
Comprehensive API documentation:

- Calibration controller reference
- Model-specific calibrators (SABR, Heston, SLV)
- Loss functions and constraints
- Diagnostics and identifiability metrics
- Joint calibration APIs
- Model selection utilities

### [Playbooks](playbooks/index.md)
Production-ready calibration workflows:

- Full-spectrum calibration campaigns
- Multi-asset calibration playbooks
- Model governance and validation procedures
- Performance optimization strategies
- Real-time calibration architectures

## Performance Considerations

The calibration framework is optimized for production use:

- **JAX acceleration**: GPU/TPU support for large-scale calibrations
- **JIT compilation**: First call compiles, subsequent calls are fast
- **Batched operations**: Vectorized pricing across strikes/maturities
- **Memory efficiency**: Careful management of intermediate arrays

Typical calibration times on CPU:
- SABR (25 quotes): < 100ms
- Heston (50 quotes): 200-500ms
- Multi-instrument joint (100+ quotes): 1-3 seconds

## Best Practices

1. **Parameter bounds**: Always use appropriate constraints to ensure model validity
2. **Initialization**: Start with reasonable initial guesses close to expected values
3. **Loss weighting**: Weight at-the-money options more heavily for better smile quality
4. **Regularization**: Add penalty terms for smoother parameter evolution in time-dependent calibration
5. **Diagnostics**: Always check identifiability and residual patterns post-calibration
6. **Model selection**: Use cross-validation or IC criteria when comparing multiple models
7. **Validation**: Reserve out-of-sample data to test calibration robustness

## Integration with Neutryx

The calibration framework integrates seamlessly with other Neutryx modules:

- **Models**: Directly uses [`neutryx.models`](../models/index.md) for pricing functions
- **Market Data**: Compatible with [`neutryx.market`](../market/index.md) data structures
- **Risk**: Calibrated parameters flow into [`neutryx.risk`](../risk/index.md) calculations
- **Valuations**: Powers mark-to-market in [`neutryx.valuations`](../valuations/index.md)

## Next Steps

- **New users**: Start with [Tutorials](tutorials/index.md) for hands-on examples
- **API reference**: See [Reference](reference/index.md) for detailed API documentation
- **Production deployment**: Check [Playbooks](playbooks/index.md) for enterprise-grade workflows
- **Advanced topics**: Explore [Calibration Theory](../theory/calibration_theory.md) for mathematical foundations
