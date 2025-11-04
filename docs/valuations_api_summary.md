# Neutryx Valuations: API Summary

Quick reference for all public APIs in the valuations module.

## XVA Module

### CVA (Credit Valuation Adjustment)

```python
# Single-currency CVA
from neutryx.valuations.xva.cva import cva, bilateral_cva, MultiCurrencyCVA

cva(epe_t: Array, df_t: Array, pd_t: Array, lgd: float = 0.6) -> float

bilateral_cva(
    epe_t: Array,
    ene_t: Array,
    df_t: Array,
    pd_counterparty_t: Array,
    pd_own_t: Array,
    lgd_counterparty: float = 0.6,
    lgd_own: float = 0.6
) -> tuple[float, float, float]

# Multi-currency
calculator = MultiCurrencyCVA(
    collateral_currency: str,
    counterparty_name: str = "COUNTERPARTY",
    lgd: float = 0.6
)
calculator.calculate(
    market_env: MarketDataEnvironment,
    exposures_by_currency: Dict[str, Array],
    times: Array,
    pd_t: Array
) -> float
```

### FVA (Funding Valuation Adjustment)

```python
from neutryx.valuations.xva.fva import fva

fva(epe_t: Array, funding_spread: float, df_t: Array) -> float
```

### Exposure Calculations

```python
from neutryx.valuations.exposure import epe

epe(paths: Array, K: float, is_call: bool = True) -> float
```

### XVA Exposure Simulation

```python
from neutryx.valuations.xva import (
    ExposureSimulator,
    ExposureResult,
    XVAScenario,
    ExposureCube
)

simulator = ExposureSimulator(
    n_paths: int = 10000,
    n_steps: int = 100,
    horizon: float = 1.0
)

result: ExposureResult = simulator.simulate(
    key: jax.random.KeyArray,
    scenario: XVAScenario,
    product: Product
)

# ExposureResult attributes:
# - epe: Array [n_steps]
# - ene: Array [n_steps]
# - pfe_95: Array [n_steps]
# - pfe_975: Array [n_steps]
# - exposure_cube: ExposureCube
```

### XVA Aggregation

```python
from neutryx.valuations.xva import AggregationEngine

engine = AggregationEngine()
total_xva = engine.aggregate(
    cva: float,
    dva: float,
    fva: float,
    mva: float = 0.0,
    kva: float = 0.0
) -> float
```

### Capital Calculations

```python
from neutryx.valuations.xva import CapitalCalculator

calculator = CapitalCalculator()
capital_requirement = calculator.calculate(
    exposure: Array,
    risk_weight: float,
    capital_ratio: float = 0.08
) -> float
```

---

## Risk Metrics Module

```python
from neutryx.valuations.risk_metrics import (
    # VaR functions
    value_at_risk,
    conditional_value_at_risk,
    expected_shortfall,
    portfolio_var,
    portfolio_cvar,

    # VaR methodologies
    VaRMethod,
    historical_var,
    parametric_var,
    monte_carlo_var,
    cornish_fisher_var,
    calculate_var,

    # Advanced VaR
    incremental_var,
    component_var,
    marginal_var,
    backtest_var,

    # Other risk measures
    downside_deviation,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    compute_all_risk_metrics,
)
```

### Basic VaR

```python
value_at_risk(returns: Array, confidence_level: float = 0.95) -> float

conditional_value_at_risk(returns: Array, confidence_level: float = 0.95) -> float

expected_shortfall(returns: Array, alpha: float = 0.95) -> float
```

### Portfolio VaR

```python
portfolio_var(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95
) -> float

portfolio_cvar(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95
) -> float
```

### VaR Methodologies

```python
class VaRMethod(Enum):
    HISTORICAL = "Historical"
    PARAMETRIC = "Parametric"
    MONTE_CARLO = "MonteCarlo"
    CORNISH_FISHER = "CornishFisher"

calculate_var(
    returns: Array,
    confidence_level: float = 0.95,
    method: VaRMethod = VaRMethod.HISTORICAL,
    **kwargs
) -> float

historical_var(
    returns: Array,
    confidence_level: float = 0.95,
    window: Optional[int] = None
) -> float

parametric_var(
    returns: Array,
    confidence_level: float = 0.95,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> float

monte_carlo_var(
    simulated_returns: Array,
    confidence_level: float = 0.95
) -> float

cornish_fisher_var(
    returns: Array,
    confidence_level: float = 0.95
) -> float
```

### Advanced VaR Analysis

```python
incremental_var(
    portfolio_returns: Array,
    position_returns: Array,
    confidence_level: float = 0.95
) -> float

component_var(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95
) -> Array

marginal_var(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95,
    delta: float = 0.01
) -> Array

backtest_var(
    realized_returns: Array,
    var_forecasts: Array,
    confidence_level: float = 0.95
) -> dict
# Returns: {
#   'violations': int,
#   'violation_rate': float,
#   'expected_violations': float,
#   'kupiec_pvalue': float,
#   'pass_backtest': bool
# }
```

### Other Risk Measures

```python
downside_deviation(returns: Array, threshold: float = 0.0) -> float

maximum_drawdown(cumulative_returns: Array) -> float

sharpe_ratio(returns: Array, risk_free_rate: float = 0.0) -> float

sortino_ratio(
    returns: Array,
    risk_free_rate: float = 0.0,
    threshold: float = 0.0
) -> float

compute_all_risk_metrics(
    returns: Array,
    confidence_levels: list = None,
    risk_free_rate: float = 0.0
) -> dict
```

---

## SIMM Module

```python
from neutryx.valuations.simm import (
    # Calculator
    SIMMCalculator,
    SIMMResult,
    calculate_simm,

    # Risk classes
    RiskClass,
    get_risk_weights,
    get_correlations,

    # Sensitivities
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType,
    bucket_sensitivities,
)
```

### SIMM Calculator

```python
calculator = SIMMCalculator()

result: SIMMResult = calculator.calculate(
    sensitivities: List[RiskFactorSensitivity]
)

# SIMMResult attributes:
# - total_im: float
# - im_by_risk_class: Dict[str, float]
# - im_by_sensitivity_type: Dict[str, float]
# - diversification_benefit: float
```

### Risk Factor Sensitivity

```python
class RiskFactorType(Enum):
    INTEREST_RATE = "InterestRate"
    FX = "FX"
    EQUITY = "Equity"
    CREDIT = "Credit"
    COMMODITY = "Commodity"

class SensitivityType(Enum):
    DELTA = "Delta"
    VEGA = "Vega"
    CURVATURE = "Curvature"

@dataclass
class RiskFactorSensitivity:
    risk_factor: str
    risk_type: RiskFactorType
    sensitivity_type: SensitivityType
    amount: float
    currency: str
    bucket: Optional[str] = None
```

### Risk Weights

```python
class RiskClass(Enum):
    INTEREST_RATE = "InterestRate"
    FX = "FX"
    EQUITY = "Equity"
    CREDIT_QUALIFYING = "CreditQualifying"
    CREDIT_NON_QUALIFYING = "CreditNonQualifying"
    COMMODITY = "Commodity"

get_risk_weights(risk_class: RiskClass) -> Dict[str, float]

get_correlations(
    risk_class: RiskClass,
    bucket_1: str,
    bucket_2: str
) -> float
```

---

## Margin Module

```python
from neutryx.valuations.margin import (
    # Initial Margin
    InitialMarginModel,
    calculate_grid_im,
    calculate_schedule_im,

    # Variation Margin
    calculate_variation_margin,
    calculate_vm_call,
)
```

### Initial Margin

```python
class InitialMarginModel(Enum):
    SIMM = "SIMM"
    GRID = "GRID"
    SCHEDULE = "Schedule"

calculate_grid_im(
    portfolio_value: float,
    asset_class: str,
    maturity_bucket: str
) -> float

calculate_schedule_im(
    notional: float,
    product_type: str,
    remaining_maturity: float
) -> float
```

### Variation Margin

```python
calculate_variation_margin(
    current_mtm: float,
    previous_vm_balance: float,
    threshold: float = 0.0,
    minimum_transfer_amount: float = 0.0
) -> float

calculate_vm_call(
    current_mtm: float,
    collateral_held: float,
    threshold: float = 0.0,
    mta: float = 0.0
) -> float
```

---

## Scenarios Module

```python
from neutryx.valuations.scenarios import (
    # Scenario framework
    Scenario,
    ScenarioSet,
    ScenarioResult,
    Shock,

    # Market bumpers
    BumpType,
    CurveBump,
    CurveBumper,
    MarketScenario,
    SurfaceBumper,
)
```

### Scenario Definition

```python
class BumpType(Enum):
    PARALLEL = "Parallel"
    STICKY_STRIKE = "StickyStrike"
    STICKY_DELTA = "StickyDelta"

@dataclass
class Shock:
    factor: str
    value: float
    shock_type: str = "relative"  # or "absolute"

@dataclass
class Scenario:
    name: str
    description: str
    shocks: List[Shock]
```

### Curve Bumper

```python
bumper = CurveBumper(
    curve_name: str,
    bump_type: BumpType,
    bump_size: float
)

bumped_curve = bumper.apply(original_curve)
```

### Surface Bumper

```python
bumper = SurfaceBumper(
    surface_name: str,
    bump_type: BumpType,
    bump_size: float
)

bumped_surface = bumper.apply(original_surface)
```

---

## Stress Test Module

```python
from neutryx.valuations.stress_test import (
    # Scenarios
    StressScenario,
    HISTORICAL_SCENARIOS,

    # Execution
    run_stress_scenario,
    run_multiple_stress_tests,
    run_historical_stress_tests,
    factor_stress_test,
    reverse_stress_test,

    # Utilities
    apply_shock_to_parameters,
)
```

### Stress Scenario

```python
@dataclass
class StressScenario:
    name: str
    description: str
    shocks: Dict[str, float]

# Pre-defined scenarios
HISTORICAL_SCENARIOS: Dict[str, StressScenario]
# Keys: 'black_monday_1987', 'financial_crisis_2008',
#       'flash_crash_2010', 'covid_crash_2020',
#       'rate_shock_up', 'rate_shock_down', 'volatility_spike'
```

### Stress Test Execution

```python
run_stress_scenario(
    scenario: StressScenario,
    base_params: Dict[str, float],
    valuation_fn: Callable,
    shock_type: str = "relative"
) -> Dict[str, float]

run_multiple_stress_tests(
    scenarios: List[StressScenario],
    base_params: Dict[str, float],
    valuation_fn: Callable,
    shock_type: str = "relative"
) -> List[Dict]

run_historical_stress_tests(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    scenario_names: Optional[List[str]] = None
) -> List[Dict]

factor_stress_test(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    factor: str,
    shock_range: Array
) -> Array

reverse_stress_test(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    factor: str,
    target_loss: float,
    search_range: tuple = (-0.99, 10.0),
    tolerance: float = 1e-4
) -> float
```

---

## Wrong-Way Risk Module

```python
from neutryx.valuations.wrong_way_risk import (
    # Types
    WWRType,
    WWRParameters,

    # CVA with WWR
    cva_with_wwr,
    simulate_correlated_defaults,

    # Models
    GaussianCopulaWWR,
    WWREngine,

    # Adjustments
    wwr_adjustment_factor,
    specific_wwr_multiplier,
)
```

### WWR Parameters

```python
class WWRType(Enum):
    GENERAL = "general"
    SPECIFIC = "specific"

@dataclass
class WWRParameters:
    correlation: float = 0.0  # -1 to 1
    wwr_type: WWRType = WWRType.GENERAL
    recovery_correlation: float = 0.0
    jump_correlation: float = 0.0
```

### CVA with WWR

```python
cva_with_wwr(
    key: jax.random.KeyArray,
    exposure_paths: Array,
    df_t: Array,
    hazard_rate: float,
    lgd: float,
    wwr_params: WWRParameters,
    T: float
) -> Tuple[float, float, Array]
# Returns: (CVA_wwr, CVA_no_wwr, exposure_at_default)

simulate_correlated_defaults(
    key: jax.random.KeyArray,
    exposure_paths: Array,
    hazard_rate: float,
    correlation: float,
    T: float,
    dt: Optional[float] = None
) -> Tuple[Array, Array]
# Returns: (default_times, default_indicators)
```

### Copula Model

```python
copula = GaussianCopulaWWR(
    correlation_matrix: Optional[Array] = None,
    n_factors: int = 2
)

exposure_paths, credit_metrics = copula.simulate_joint(
    key: jax.random.KeyArray,
    n_paths: int,
    n_steps: int,
    exposure_model: Callable,
    credit_model: Callable,
    **model_params
) -> Tuple[Array, Array]
```

### WWR Adjustments

```python
wwr_adjustment_factor(
    correlation: float,
    volatility_exposure: float,
    volatility_spread: float
) -> float

specific_wwr_multiplier(
    exposure_to_reference: float,
    total_exposure: float,
    jump_given_default: float = 1.0
) -> float
```

### WWR Engine

```python
engine = WWREngine(
    general_wwr_params: WWRParameters,
    specific_wwr_exposure: float = 0.0,
    specific_wwr_jump: float = 1.0
)

result: dict = engine.calculate_cva_adjustment(
    key: jax.random.KeyArray,
    exposure_paths: Array,
    df_t: Array,
    hazard_rate: float,
    lgd: float,
    T: float
)
# Returns dict with keys:
# 'cva_base', 'cva_general_wwr', 'cva_total', 'wwr_charge',
# 'general_wwr_impact', 'specific_wwr_impact',
# 'specific_wwr_multiplier', 'avg_exposure_at_default'
```

---

## P&L Attribution Module

```python
from neutryx.valuations.pnl_attribution import (
    # Engine
    PnLAttributionEngine,
    AttributionMethod,

    # Data structures
    MarketState,
    PnLAttribution,
    DailyPnLTracker,

    # Analysis
    analyze_pnl_drivers,
)
```

### Attribution Engine

```python
class AttributionMethod(Enum):
    GREEKS = "greeks"
    REVALUATION = "revaluation"
    HYBRID = "hybrid"

engine = PnLAttributionEngine(
    portfolio_pricer: Callable,
    greeks_calculator: Optional[Callable] = None,
    method: AttributionMethod = AttributionMethod.HYBRID
)

attribution: PnLAttribution = engine.attribute_pnl(
    start_state: MarketState,
    end_state: MarketState,
    start_portfolio_value: Optional[float] = None
)
```

### Market State

```python
@dataclass
class MarketState:
    timestamp: float
    spot_prices: Dict[str, float]
    volatilities: Dict[str, float]
    interest_rates: Dict[str, float]
    fx_rates: Dict[str, float]
    credit_spreads: Dict[str, float]
    dividend_yields: Dict[str, float]

    def get_all_factors(self) -> Dict[str, float]
```

### P&L Attribution

```python
@dataclass
class PnLAttribution:
    total_pnl: float
    theta_pnl: float
    spot_pnl: Dict[str, float]
    vol_pnl: Dict[str, float]
    rate_pnl: Dict[str, float]
    fx_pnl: Dict[str, float]
    spread_pnl: Dict[str, float]
    gamma_pnl: Dict[str, float]
    vega_pnl: Dict[str, float]
    cross_gamma_pnl: Dict[Tuple[str, str], float]
    unexplained_pnl: float

    def total_spot_pnl(self) -> float
    def total_vol_pnl(self) -> float
    def total_rate_pnl(self) -> float
    def explained_pnl(self) -> float
    def explanation_ratio(self) -> float
```

### P&L Tracking

```python
tracker = DailyPnLTracker()

tracker.add_attribution(date: float, attribution: PnLAttribution)

total = tracker.total_pnl(
    start_date: Optional[float] = None,
    end_date: Optional[float] = None
) -> float

cumulative = tracker.cumulative_attribution() -> Dict[str, List[float]]
```

### P&L Analysis

```python
analyze_pnl_drivers(
    attribution: PnLAttribution,
    threshold: float = 0.01
) -> List[Tuple[str, float]]
```

---

## Greeks Module

```python
from neutryx.valuations.greeks import (
    calculate_greeks,
    GreeksCalculator,
    # ... specific greek functions
)
```

### Greeks Calculator

```python
calculator = GreeksCalculator(pricing_function: Callable)

greeks = calculator.compute_all(
    spot: float,
    strike: float,
    time_to_maturity: float,
    volatility: float,
    interest_rate: float,
    **kwargs
) -> GreeksResult

# GreeksResult attributes:
# delta, gamma, vega, theta, rho, vanna, volga, charm, ...
```

---

## Type Aliases

```python
from jax import Array
from neutryx.core.engine import Array  # Alias for JAX arrays

# Common types used throughout
Array: TypeAlias = jax.Array
KeyArray: TypeAlias = jax.random.KeyArray
```

---

## Constants

```python
# Common defaults
DEFAULT_LGD = 0.60  # 60% Loss Given Default
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95% confidence
DEFAULT_N_PATHS = 10000  # Monte Carlo paths
DEFAULT_N_STEPS = 100  # Time steps

# Risk-free rates (examples)
USD_RISK_FREE = 0.04  # 4%
EUR_RISK_FREE = 0.02  # 2%
```

---

For complete documentation, see [Comprehensive Guide](valuations_comprehensive.md).
