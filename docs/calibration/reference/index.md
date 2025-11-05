# Calibration Reference

Comprehensive API reference documentation for all calibration modules, classes, and functions in Neutryx. This reference provides detailed specifications, parameters, return types, and usage examples.

## Quick Navigation

- [Core Framework](#core-framework)
- [Model-Specific Calibrators](#model-specific-calibrators)
- [Loss Functions](#loss-functions)
- [Constraints & Transformations](#constraints-transformations)
- [Diagnostics](#diagnostics)
- [Joint Calibration](#joint-calibration)
- [Model Selection](#model-selection)
- [Bayesian Model Averaging](#bayesian-model-averaging)
- [Regularization](#regularization)

---

## Core Framework

### CalibrationController

**Module**: `neutryx.calibration.base`

Base class for all calibration workflows providing gradient-based optimization with JAX.

```python
class CalibrationController:
    """Base class implementing gradient-based calibration workflow."""

    def __init__(
        self,
        parameter_specs: Mapping[str, ParameterSpec],
        loss_fn: Callable[..., Array],
        optimizer: optax.GradientTransformation,
        penalty_fn: Optional[Callable[[Mapping[str, Array], Mapping[str, Array]], Array]] = None,
        max_steps: int = 400,
        tol: float = 1e-8,
        dtype: jnp.dtype = jnp.float64,
    ) -> None
```

**Parameters:**
- `parameter_specs`: Dictionary mapping parameter names to their specifications (initial values, transforms)
- `loss_fn`: Loss function to minimize, signature `(model_vals, target_vals, weights) -> scalar`
- `optimizer`: Optax optimizer (e.g., `optax.adam(0.01)`)
- `penalty_fn`: Optional regularization penalty function
- `max_steps`: Maximum optimization iterations (default: 400)
- `tol`: Convergence tolerance for loss change (default: 1e-8)
- `dtype`: Floating point precision (default: float64)

**Methods:**

`calibrate(market_data: Mapping[str, Any]) -> CalibrationResult`
- Main calibration entry point
- Returns calibrated parameters, loss history, convergence status

**Subclass Hooks:**
- `_prepare_market_data(market_data)`: Convert raw data to JAX arrays
- `_target_observables(market_data)`: Extract target values from market data
- `_model_observables(params, market_data)`: Compute model predictions

**Example:**
```python
from neutryx.calibration.base import CalibrationController, ParameterSpec
from neutryx.calibration.constraints import positive
import optax

class MyCalibrator(CalibrationController):
    def _target_observables(self, market_data):
        return market_data["target_prices"]

    def _model_observables(self, params, market_data):
        return my_pricing_function(params, market_data)

calibrator = MyCalibrator(
    parameter_specs={
        "vol": ParameterSpec(init=0.2, transform=positive())
    },
    loss_fn=losses.rmse,
    optimizer=optax.adam(0.01)
)
```

---

### ParameterSpec

**Module**: `neutryx.calibration.base`

Specification for a calibrated parameter including initial value and transformation.

```python
@dataclass
class ParameterSpec:
    """Specification for a calibrated parameter."""
    init: float                     # Initial value
    transform: ParameterTransform   # Bidirectional transformation
```

**Attributes:**
- `init`: Initial parameter value for optimization
- `transform`: Constraint enforcement via transformation (e.g., log for positivity)

**Usage:**
```python
from neutryx.calibration.base import ParameterSpec
from neutryx.calibration.constraints import positive, bounded

specs = {
    "vol": ParameterSpec(init=0.2, transform=positive()),
    "mean_rev": ParameterSpec(init=1.5, transform=bounded(0.0, 10.0))
}
```

---

### CalibrationResult

**Module**: `neutryx.calibration.base`

Container for calibration output including fitted parameters and diagnostics.

```python
@dataclass
class CalibrationResult:
    """Container for calibration outputs."""
    params: Dict[str, float]         # Calibrated parameter values
    loss_history: Iterable[float]    # Loss at each iteration
    converged: bool                  # Whether optimization converged
    iterations: int                  # Number of iterations performed
```

**Attributes:**
- `params`: Dictionary of calibrated parameter values
- `loss_history`: List of loss values across iterations (for diagnostics)
- `converged`: True if tolerance criterion was met
- `iterations`: Actual number of optimization steps

**Usage:**
```python
result = calibrator.calibrate(market_data)
print(f"Parameters: {result.params}")
print(f"Converged: {result.converged} in {result.iterations} steps")
print(f"Final loss: {result.loss_history[-1]:.6f}")
```

---

### ParameterTransform

**Module**: `neutryx.calibration.base`

Bidirectional transform for enforcing parameter constraints during optimization.

```python
@dataclass
class ParameterTransform:
    """Bidirectional transform used to enforce parameter constraints."""
    forward: Callable[[Array], Array]   # Constrained → unconstrained
    inverse: Callable[[Array], Array]   # Unconstrained → constrained
```

**Usage:**
Typically created via constraint functions (see [Constraints](#constraints-transformations)) rather than directly.

---

## Model-Specific Calibrators

### SABRCalibrator

**Module**: `neutryx.calibration.sabr`

Calibrates SABR (Stochastic Alpha Beta Rho) model to volatility surface data using Hagan's approximation.

```python
class SABRCalibrator(CalibrationController):
    def __init__(
        self,
        parameter_specs: Optional[Mapping[str, ParameterSpec]] = None,
        loss_fn: Callable = losses.rmse,
        optimizer: optax.GradientTransformation = optax.adam(0.01),
        penalty_fn: Optional[Callable] = None,
        max_steps: int = 400,
        tol: float = 1e-8,
    )
```

**SABR Parameters:**
- `alpha`: Initial volatility level (typical: 0.1-0.5)
- `beta`: Backbone parameter, CEV exponent (typical: 0.0-1.0, often 0.5)
- `rho`: Correlation between forward and volatility (typical: -0.9 to 0.9)
- `nu`: Volatility of volatility (typical: 0.1-1.0)

**Market Data Format:**
```python
market_data = {
    "forward": 100.0,                               # Forward price
    "strikes": jnp.array([90, 95, 100, 105, 110]),  # Strike prices
    "maturities": jnp.array([1.0, 2.0, 5.0]),       # Times to maturity
    "target_vols": jnp.array([...]),                # Implied volatilities (flattened)
    "weights": jnp.array([...])                     # Optional weights
}
```

**Default Parameters:**
```python
from neutryx.calibration.sabr import default_parameter_specs
specs = default_parameter_specs()  # Sensible SABR defaults
```

**Example:**
```python
from neutryx.calibration.sabr import SABRCalibrator, generate_sabr_market_data
from neutryx.models.sabr import SABRParams

# Generate synthetic data for testing
true_params = SABRParams(alpha=0.25, beta=0.5, rho=-0.3, nu=0.4)
market_data = generate_sabr_market_data(
    forward=100.0,
    strikes=jnp.linspace(80, 120, 9),
    maturities=jnp.array([0.25, 0.5, 1.0, 2.0, 5.0]),
    params=true_params
)

# Calibrate
calibrator = SABRCalibrator()
result = calibrator.calibrate(market_data)
```

---

### HestonCalibrator

**Module**: `neutryx.calibration.heston`

Calibrates Heston stochastic volatility model to option prices using Fourier inversion.

```python
class HestonCalibrator(CalibrationController):
    def __init__(
        self,
        parameter_specs: Optional[Mapping[str, ParameterSpec]] = None,
        loss_fn: Callable = losses.rmse,
        optimizer: optax.GradientTransformation = optax.adam(0.01),
        penalty_fn: Optional[Callable] = None,
        max_steps: int = 400,
        tol: float = 1e-8,
    )
```

**Heston Parameters:**
- `v0`: Initial variance (typical: 0.01-0.1)
- `kappa`: Mean reversion speed (typical: 0.5-5.0)
- `theta`: Long-run variance (typical: 0.01-0.1)
- `sigma`: Volatility of variance, vol-of-vol (typical: 0.1-1.0)
- `rho`: Correlation between spot and variance (typical: -0.9 to 0.0)

**Market Data Format:**
```python
market_data = {
    "spot": 4500.0,                          # Current spot price
    "strikes": jnp.array([...]),             # Strike prices
    "maturities": jnp.array([...]),          # Times to maturity
    "target_prices": jnp.array([...]),       # Market call prices
    "rate": 0.05,                            # Risk-free rate
    "dividend": 0.02,                        # Dividend yield
    "weights": jnp.array([...])              # Optional weights
}
```

**Default Parameters:**
```python
from neutryx.calibration.heston import default_parameter_specs
specs = default_parameter_specs()  # Sensible Heston defaults
```

**Example:**
```python
from neutryx.calibration.heston import HestonCalibrator, generate_heston_market_data
from neutryx.models.heston import HestonParams

# Generate synthetic data
true_params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
market_data = generate_heston_market_data(
    spot=4500.0,
    strikes=jnp.linspace(4000, 5000, 11),
    maturities=jnp.array([0.25, 0.5, 1.0, 2.0]),
    params=true_params,
    rate=0.05,
    dividend=0.02
)

# Calibrate
calibrator = HestonCalibrator()
result = calibrator.calibrate(market_data)
```

---

### SLVCalibrator

**Module**: `neutryx.calibration.slv`

Calibrates simplified Stochastic-Local Volatility model combining local and stochastic features.

```python
class SLVCalibrator(CalibrationController):
    def __init__(
        self,
        parameter_specs: Optional[Mapping[str, ParameterSpec]] = None,
        loss_fn: Callable = losses.rmse,
        optimizer: optax.GradientTransformation = optax.adam(0.01),
        penalty_fn: Optional[Callable] = None,
        max_steps: int = 400,
        tol: float = 1e-8,
    )
```

**SLV Parameters:**
- `base_vol`: Base volatility level (typical: 0.1-0.5)
- `local_slope`: Slope of local vol wrt log-moneyness (typical: -1.0 to 1.0)
- `local_curvature`: Curvature (smile convexity) (typical: -0.5 to 0.5)
- `mixing`: Mixing between local and stochastic (typical: 0.0-1.0)
- `time_decay`: Time decay rate (typical: 0.0-0.5)

**Market Data Format:**
```python
market_data = {
    "forward": 100.0,
    "strikes": jnp.array([...]),
    "maturities": jnp.array([...]),
    "target_vols": jnp.array([...]),
    "weights": jnp.array([...])  # Optional
}
```

**Example:**
```python
from neutryx.calibration.slv import SLVCalibrator, generate_slv_market_data

# Generate synthetic data
params = {
    "base_vol": 0.2,
    "local_slope": -0.3,
    "local_curvature": 0.1,
    "mixing": 0.5,
    "time_decay": 0.05
}

market_data = generate_slv_market_data(
    forward=100.0,
    strikes=jnp.linspace(70, 130, 13),
    maturities=jnp.array([0.5, 1.0, 2.0, 5.0]),
    params=params
)

# Calibrate
calibrator = SLVCalibrator()
result = calibrator.calibrate(market_data)
```

---

## Loss Functions

**Module**: `neutryx.calibration.losses`

Pre-defined loss functions for calibration optimization.

### rmse
Root mean squared error (default for most calibrators).

```python
def rmse(model_vals: Array, target_vals: Array, weights: Optional[Array] = None) -> Array
```

### mse
Mean squared error.

```python
def mse(model_vals: Array, target_vals: Array, weights: Optional[Array] = None) -> Array
```

### mae
Mean absolute error.

```python
def mae(model_vals: Array, target_vals: Array, weights: Optional[Array] = None) -> Array
```

### relative_error
Relative error (percentage-based).

```python
def relative_error(model_vals: Array, target_vals: Array, weights: Optional[Array] = None) -> Array
```

### log_price_error
Log-space error for price calibration (reduces impact of outliers).

```python
def log_price_error(model_vals: Array, target_vals: Array, weights: Optional[Array] = None) -> Array
```

**Example:**
```python
from neutryx.calibration import losses

# Use custom loss function
calibrator = SABRCalibrator(loss_fn=losses.mae)

# Custom weighted loss
def custom_loss(model_vals, target_vals, weights=None):
    atm_mask = jnp.abs(log_moneyness) < 0.05
    atm_weight = jnp.where(atm_mask, 2.0, 1.0)
    if weights is not None:
        atm_weight = atm_weight * weights
    return losses.rmse(model_vals, target_vals, atm_weight)

calibrator = SABRCalibrator(loss_fn=custom_loss)
```

---

## Constraints & Transformations

**Module**: `neutryx.calibration.constraints`

Functions for creating parameter transformations that enforce constraints.

### positive
Enforces strictly positive parameters via log transform.

```python
def positive(lower_bound: float = 1e-4) -> ParameterTransform
```

**Example:** `"alpha": ParameterSpec(init=0.2, transform=positive())`

---

### positive_with_upper
Positive with upper bound via scaled log-sigmoid transform.

```python
def positive_with_upper(lower_bound: float, upper_bound: float) -> ParameterTransform
```

**Example:** `"alpha": ParameterSpec(init=0.2, transform=positive_with_upper(1e-4, 3.0))`

---

### bounded
General bounded parameter in [lower, upper).

```python
def bounded(lower: float, upper: float) -> ParameterTransform
```

**Example:** `"beta": ParameterSpec(init=0.5, transform=bounded(0.0, 0.999))`

---

### symmetric
Symmetric bound around zero: (-limit, limit).

```python
def symmetric(limit: float) -> ParameterTransform
```

**Example:** `"rho": ParameterSpec(init=-0.3, transform=symmetric(0.999))`

---

### identity
No transformation (unconstrained parameter).

```python
def identity() -> ParameterTransform
```

**Example:** `"drift": ParameterSpec(init=0.05, transform=identity())`

---

## Diagnostics

**Module**: `neutryx.calibration.diagnostics`

Post-calibration diagnostics and quality assessment.

### generate_calibration_diagnostics

```python
def generate_calibration_diagnostics(
    result: CalibrationResult,
    market_data: Mapping[str, Array]
) -> CalibrationDiagnostics
```

**Returns:** `CalibrationDiagnostics` containing:
- `rmse`: Root mean squared error
- `max_abs_error`: Maximum absolute error
- `mean_abs_error`: Mean absolute error
- `residuals`: Array of residuals (model - market)
- `identifiability`: Parameter identifiability metrics

---

### compute_identifiability_metrics

```python
def compute_identifiability_metrics(
    calibrator: CalibrationController,
    params: Mapping[str, float],
    market_data: Mapping[str, Array]
) -> IdentifiabilityMetrics
```

Computes Jacobian-based identifiability analysis.

**Returns:** `IdentifiabilityMetrics` containing:
- `condition_number`: Condition number of Jacobian (< 100 is good)
- `rank`: Effective rank of Jacobian
- `singular_values`: Singular values of Jacobian
- `correlation_matrix`: Parameter correlation matrix

---

### build_residual_plot_data

```python
def build_residual_plot_data(
    result: CalibrationResult,
    market_data: Mapping[str, Array]
) -> ResidualPlotData
```

Prepares data for residual visualization.

**Example:**
```python
from neutryx.calibration.diagnostics import (
    generate_calibration_diagnostics,
    compute_identifiability_metrics
)

# Full diagnostics
diagnostics = generate_calibration_diagnostics(result, market_data)
print(f"RMSE: {diagnostics.rmse:.4f}")
print(f"Max error: {diagnostics.max_abs_error:.4f}")

# Check identifiability
id_metrics = compute_identifiability_metrics(calibrator, result.params, market_data)
if id_metrics.condition_number > 100:
    print("Warning: Poor parameter identifiability!")
```

---

## Joint Calibration

**Module**: `neutryx.calibration.joint_calibration`

Advanced multi-dimensional calibration workflows.

### MultiInstrumentCalibrator

Calibrate to multiple instrument types simultaneously.

```python
class MultiInstrumentCalibrator:
    def __init__(
        self,
        instrument_specs: List[InstrumentSpec],
        base_calibrator: CalibrationController
    )

@dataclass
class InstrumentSpec:
    name: str
    weight: float = 1.0
    loss_fn: str = "rmse"
```

**Example:**
```python
from neutryx.calibration.joint_calibration import MultiInstrumentCalibrator, InstrumentSpec

calibrator = MultiInstrumentCalibrator(
    instrument_specs=[
        InstrumentSpec(name="calls", weight=1.0),
        InstrumentSpec(name="puts", weight=1.0),
        InstrumentSpec(name="var_swaps", weight=2.0)
    ],
    base_calibrator=HestonCalibrator()
)

result = calibrator.calibrate({
    "calls": call_data,
    "puts": put_data,
    "var_swaps": vs_data
})
```

---

### CrossAssetCalibrator

Joint calibration with shared parameters across assets.

```python
class CrossAssetCalibrator:
    def __init__(
        self,
        asset_specs: List[AssetClassSpec],
        base_calibrator: CalibrationController
    )

@dataclass
class AssetClassSpec:
    name: str
    weight: float = 1.0
    shared_params: List[str] = field(default_factory=list)
```

---

### TimeDependentCalibrator

Piecewise-constant parameters across time segments.

```python
class TimeDependentCalibrator:
    def __init__(
        self,
        time_segments: List[TimeSegment],
        base_calibrator: CalibrationController
    )

@dataclass
class TimeSegment:
    start: float
    end: float
    params: List[str]  # Parameters to vary in this segment
```

---

## Model Selection

**Module**: `neutryx.calibration.model_selection`

Model comparison, cross-validation, and sensitivity analysis.

### Information Criteria

```python
def compute_aic(loss: float, n_params: int, n_data: int) -> float
def compute_bic(loss: float, n_params: int, n_data: int) -> float
def compute_aicc(loss: float, n_params: int, n_data: int) -> float
def compute_hqic(loss: float, n_params: int, n_data: int) -> float
```

---

### compare_models

```python
def compare_models(
    models: List[Tuple[str, CalibrationResult, Mapping[str, Array]]]
) -> ModelComparison
```

**Returns:** `ModelComparison` with best models by each criterion.

---

### cross_validate

```python
def cross_validate(
    calibrator: CalibrationController,
    data: Mapping[str, Array],
    split_fn: Callable,
    n_trials: int = 1
) -> CrossValidationResult
```

**Split functions:**
- `k_fold_split(n_folds: int)`
- `time_series_split(n_splits: int)`

---

### Sensitivity Analysis

```python
def compute_local_sensitivity(
    calibrator: CalibrationController,
    params: Mapping[str, float],
    market_data: Mapping[str, Array]
) -> LocalSensitivity

def sobol_indices(
    calibrator: CalibrationController,
    param_ranges: Mapping[str, Tuple[float, float]],
    market_data: Mapping[str, Array],
    n_samples: int = 1024
) -> GlobalSensitivity
```

---

## Bayesian Model Averaging

**Module**: `neutryx.calibration.bayesian_model_averaging`

Ensemble predictions with model uncertainty quantification.

```python
class BayesianModelAveraging:
    def __init__(
        self,
        models: List[CalibrationController],
        weighting_scheme: WeightingScheme = WeightingScheme.STACKING
    )

    def fit(self, data: Mapping[str, Array]) -> BMAResult
    def predict(self, conditions: Mapping[str, Array]) -> Array

class WeightingScheme(Enum):
    STACKING = "stacking"
    PSEUDO_BMA = "pseudo_bma"
    IC_BASED = "ic_based"
```

---

## Regularization

**Module**: `neutryx.calibration.regularization`

Regularization penalties for stable calibration.

### L2Regularization

```python
class L2Regularization:
    def __init__(
        self,
        reference_params: Mapping[str, float],
        weights: Mapping[str, float]
    )

    def __call__(
        self,
        params: Mapping[str, Array],
        transformed_params: Mapping[str, Array]
    ) -> Array
```

---

### SmoothnessRegularization

```python
class SmoothnessRegularization:
    def __init__(
        self,
        time_segments: List[TimeSegment],
        weight: float = 0.01
    )
```

---

## Complete API Documentation

For exhaustive API details including all classes, methods, and parameters, see:

- **[Calibration Reference Compendium](calibration_reference_compendium.md)**: Comprehensive API catalog
- **[Calibration Grandmaster Codex](calibration_grandmaster_codex.md)**: Advanced patterns and recipes

---

## Type Annotations

All APIs use standard Python type hints:

```python
from typing import Callable, Dict, List, Mapping, Optional, Tuple
import jax.numpy as jnp

Array = jnp.ndarray  # JAX array type
```

---

## Next Steps

- **Hands-on learning**: See [Tutorials](../tutorials/index.md)
- **Production workflows**: Explore [Playbooks](../playbooks/index.md)
- **Mathematical foundations**: Study [Calibration Theory](../../theory/calibration_theory.md)
