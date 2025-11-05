# Neutryx Valuations Framework: Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [XVA Framework](#xva-framework)
4. [Exposure Calculations](#exposure-calculations)
5. [Greeks and Sensitivities](#greeks-and-sensitivities)
6. [Risk Metrics](#risk-metrics)
7. [SIMM and Margin Calculations](#simm-and-margin-calculations)
8. [Scenario Analysis](#scenario-analysis)
9. [Stress Testing](#stress-testing)
10. [Wrong-Way Risk](#wrong-way-risk)
11. [P&L Attribution](#pl-attribution)
12. [API Reference](#api-reference)
13. [Usage Examples](#usage-examples)
14. [Best Practices](#best-practices)

---

## Overview

The Neutryx Valuations module provides a comprehensive suite of tools for derivatives valuation, risk management, and regulatory capital calculations. Built on JAX for high-performance computing, the module implements industry-standard methodologies for:

- **XVA (X-Valuation Adjustments)**: CVA, DVA, FVA, MVA, KVA, and collateral valuation adjustments
- **Exposure Metrics**: Expected Positive/Negative Exposure (EPE/ENE), Potential Future Exposure (PFE)
- **Risk Metrics**: Value at Risk (VaR), Conditional VaR (CVaR/ES), risk-adjusted performance measures
- **Margin Calculations**: Initial Margin (SIMM, GRID), Variation Margin
- **Greeks**: Delta, Gamma, Vega, Theta, Rho, and advanced second-order Greeks
- **Scenario Analysis**: Market shock analysis, historical scenarios, stress testing
- **P&L Attribution**: Factor-based P&L decomposition

### Key Features

- **JAX-Accelerated**: GPU/TPU acceleration for Monte Carlo simulations
- **Production-Ready**: Industry-standard implementations of Basel III, ISDA SIMM, SA-CCR
- **Multi-Currency**: Native support for FX exposure and multi-currency portfolios
- **Regulatory Compliance**: FRTB, SA-CCR, SIMM-compliant calculations
- **Extensible**: Modular design for custom risk measures and valuation adjustments

---

## Architecture

The valuations module is organized into specialized submodules:

```
neutryx.valuations/
├── xva/                    # XVA framework
│   ├── cva.py             # Credit Valuation Adjustment
│   ├── fva.py             # Funding Valuation Adjustment
│   ├── mva.py             # Margin Valuation Adjustment
│   ├── kva.py             # Capital Valuation Adjustment
│   ├── collateral.py      # Collateral modeling
│   ├── exposure.py        # Exposure simulation
│   └── aggregation.py     # XVA aggregation
├── greeks/                 # Greeks calculations
│   ├── greeks.py          # Standard Greeks
│   └── advanced_greeks.py # Higher-order Greeks
├── risk_metrics.py         # VaR, CVaR, risk measures
├── simm/                   # ISDA SIMM implementation
│   ├── calculator.py      # SIMM calculation engine
│   ├── sensitivities.py   # Risk sensitivities
│   └── risk_weights.py    # SIMM calibration parameters
├── margin/                 # Margin calculations
│   ├── initial_margin.py  # Initial Margin (GRID, Schedule)
│   └── variation_margin.py # Variation Margin
├── scenarios/              # Scenario analysis
│   ├── scenario.py        # Scenario framework
│   ├── bumpers.py         # Market data bumpers
│   └── scenario_engine.py # Scenario execution
├── stress_test.py          # Stress testing
├── wrong_way_risk.py       # Wrong-way risk modeling
├── pnl_attribution.py      # P&L attribution
├── exposure.py             # EPE/ENE calculations
└── utils.py                # Utility functions
```

### Design Principles

1. **Functional Design**: Pure functions for deterministic calculations
2. **Type Safety**: Comprehensive type annotations for all APIs
3. **Performance**: JIT compilation and vectorization with JAX
4. **Modularity**: Independent components with clear interfaces
5. **Testability**: Extensive test coverage with numerical validation

---

## XVA Framework

### Overview

XVA (X-Valuation Adjustments) represents the set of adjustments to derivative valuations that account for counterparty credit risk, funding costs, capital requirements, and margin effects.

### Components

#### 1. Credit Valuation Adjustment (CVA)

CVA quantifies the expected loss from counterparty default:

```
CVA = LGD × Σ DF(t) × EPE(t) × ΔPD(t)
```

Where:
- `LGD`: Loss Given Default (typically 40-60%)
- `DF(t)`: Discount factor at time t
- `EPE(t)`: Expected Positive Exposure at time t
- `ΔPD(t)`: Incremental default probability

**Key Functions**: [neutryx.valuations.xva.cva](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/xva/cva.py)

```python
from neutryx.valuations.xva.cva import cva, bilateral_cva, MultiCurrencyCVA

# Single-currency CVA
cva_value = cva(
    epe_t=epe_profile,       # Expected Positive Exposure [n_steps]
    df_t=discount_factors,   # Discount factors [n_steps]
    pd_t=default_probs,      # Cumulative default probabilities [n_steps]
    lgd=0.60                 # Loss Given Default (60%)
)

# Bilateral CVA (CVA - DVA)
cva_val, dva_val, bilateral = bilateral_cva(
    epe_t=epe_profile,
    ene_t=ene_profile,       # Expected Negative Exposure
    df_t=discount_factors,
    pd_counterparty_t=pd_counterparty,
    pd_own_t=pd_own,
    lgd_counterparty=0.60,
    lgd_own=0.40
)
```

**Multi-Currency CVA**:

Handles portfolios with multiple currencies and FX exposure:

```python
from neutryx.market import MarketDataEnvironment
from neutryx.valuations.xva.cva import MultiCurrencyCVA

calculator = MultiCurrencyCVA(
    collateral_currency='USD',
    counterparty_name='BANK_A',
    lgd=0.60
)

# Exposures by currency
exposures = {
    'USD': usd_epe_profile,
    'EUR': eur_epe_profile,
    'GBP': gbp_epe_profile
}

total_cva = calculator.calculate(
    market_env=market_env,
    exposures_by_currency=exposures,
    times=time_grid,
    pd_t=default_probs
)
```

#### 2. Debit Valuation Adjustment (DVA)

DVA represents the benefit from the institution's own default risk:

```
DVA = LGD × Σ DF(t) × ENE(t) × ΔPD_own(t)
```

DVA is controversial as it implies a benefit from worsening creditworthiness, but is required for accounting symmetry.

#### 3. Funding Valuation Adjustment (FVA)

FVA quantifies funding costs for uncollateralized exposure:

```
FVA = Σ DF(t) × EPE(t) × FundingSpread(t)
```

**Key Functions**: [neutryx.valuations.xva.fva](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/xva/fva.py)

```python
from neutryx.valuations.xva.fva import fva

fva_value = fva(
    epe_t=epe_profile,
    funding_spread=0.0050,    # 50bp funding spread
    df_t=discount_factors
)
```

**Components**:
- **FBA** (Funding Benefit Adjustment): Benefit from negative exposure
- **FCA** (Funding Cost Adjustment): Cost of positive exposure

#### 4. Margin Valuation Adjustment (MVA)

MVA represents the cost of posting margin over the life of a trade:

```
MVA = Σ DF(t) × E[IM(t)] × FundingSpread(t)
```

Where `E[IM(t)]` is the expected Initial Margin requirement.

#### 5. Capital Valuation Adjustment (KVA)

KVA quantifies the cost of regulatory capital:

```
KVA = Σ DF(t) × Capital(t) × CostOfCapital(t)
```

### XVA Aggregation

The [AggregationEngine](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/xva/aggregation.py) combines all XVA components:

```python
from neutryx.valuations.xva import AggregationEngine

engine = AggregationEngine()

total_xva = engine.aggregate(
    cva=cva_value,
    dva=dva_value,
    fva=fva_value,
    mva=mva_value,
    kva=kva_value
)

# Total valuation adjustment
adjusted_price = clean_price - total_xva
```

---

## Exposure Calculations

### Expected Positive Exposure (EPE)

EPE represents the average positive mark-to-market value of a position, weighted by probability:

```
EPE(t) = E[max(V(t), 0)]
```

**Implementation**: [neutryx.valuations.exposure](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/exposure.py)

```python
from neutryx.valuations.exposure import epe

epe_value = epe(
    paths=simulation_paths,  # [n_paths, n_steps+1]
    K=strike_price,
    is_call=True
)
```

### Expected Negative Exposure (ENE)

```
ENE(t) = E[max(-V(t), 0)]
```

### Potential Future Exposure (PFE)

PFE is the exposure at a given confidence level:

```
PFE(t, α) = Quantile_α[V(t)]
```

Typically calculated at 95% or 97.5% confidence.

### Exposure Simulation

The [ExposureSimulator](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/xva/exposure.py) provides comprehensive exposure calculation:

```python
from neutryx.valuations.xva import ExposureSimulator, XVAScenario

simulator = ExposureSimulator(
    n_paths=10000,
    n_steps=100,
    horizon=1.0
)

scenario = XVAScenario(
    spot=100.0,
    volatility=0.20,
    interest_rate=0.05
)

result = simulator.simulate(
    key=jax_random_key,
    scenario=scenario,
    product=option_product
)

# Access exposure metrics
epe_profile = result.epe
ene_profile = result.ene
pfe_95 = result.pfe_95
```

### Exposure Cube

For portfolio-level exposure aggregation:

```python
from neutryx.valuations.xva import ExposureCube

cube = ExposureCube(
    n_counterparties=100,
    n_scenarios=10000,
    n_timesteps=50
)

# Store exposures
cube.add_exposure(
    counterparty_id=0,
    scenario_id=i,
    timestep=t,
    exposure=exposure_value
)

# Aggregate metrics
portfolio_epe = cube.calculate_epe()
counterparty_epe = cube.calculate_epe_by_counterparty()
```

---

## Greeks and Sensitivities

Greeks measure the sensitivity of derivatives prices to market parameters.

### First-Order Greeks

#### Delta (Δ)

Sensitivity to underlying price:
```
Δ = ∂V/∂S
```

#### Vega (ν)

Sensitivity to volatility:
```
ν = ∂V/∂σ
```

#### Theta (Θ)

Sensitivity to time decay:
```
Θ = ∂V/∂t
```

#### Rho (ρ)

Sensitivity to interest rate:
```
ρ = ∂V/∂r
```

### Second-Order Greeks

#### Gamma (Γ)

Convexity to underlying price:
```
Γ = ∂²V/∂S²
```

#### Vanna

Cross-sensitivity of delta to volatility:
```
Vanna = ∂²V/∂S∂σ
```

#### Volga (Vomma)

Convexity to volatility:
```
Volga = ∂²V/∂σ²
```

#### Charm

Sensitivity of delta to time:
```
Charm = ∂²V/∂S∂t
```

### Implementation

**Key Module**: [neutryx.valuations.greeks](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/greeks/)

```python
from neutryx.valuations.greeks import calculate_greeks, GreeksCalculator

# Automatic differentiation approach
calculator = GreeksCalculator(pricing_function=price_option)

greeks = calculator.compute_all(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    volatility=0.20,
    interest_rate=0.05
)

print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.4f}")
print(f"Vega: {greeks.vega:.4f}")
print(f"Theta: {greeks.theta:.4f}")
print(f"Rho: {greeks.rho:.4f}")
```

---

## Risk Metrics

### Value at Risk (VaR)

VaR is the maximum loss not exceeded with a given confidence level:

```
VaR_α = -Quantile_{1-α}(Returns)
```

**Key Module**: [neutryx.valuations.risk_metrics](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/risk_metrics.py)

#### VaR Methodologies

##### 1. Historical VaR

Based on empirical return distribution:

```python
from neutryx.valuations.risk_metrics import historical_var

var_95 = historical_var(
    returns=historical_returns,
    confidence_level=0.95,
    window=252  # 1-year rolling window
)
```

##### 2. Parametric VaR

Assumes normal distribution:

```python
from neutryx.valuations.risk_metrics import parametric_var

var_95 = parametric_var(
    returns=returns,
    confidence_level=0.95,
    mean=None,  # Estimated from returns
    std=None    # Estimated from returns
)
```

##### 3. Monte Carlo VaR

From simulated scenarios:

```python
from neutryx.valuations.risk_metrics import monte_carlo_var

var_95 = monte_carlo_var(
    simulated_returns=mc_scenarios,
    confidence_level=0.95
)
```

##### 4. Cornish-Fisher VaR

Parametric with skewness/kurtosis adjustment:

```python
from neutryx.valuations.risk_metrics import cornish_fisher_var

var_95 = cornish_fisher_var(
    returns=returns,
    confidence_level=0.95
)
```

### Conditional VaR (CVaR / Expected Shortfall)

CVaR is the expected loss given that VaR is exceeded:

```
CVaR_α = E[Loss | Loss > VaR_α]
```

```python
from neutryx.valuations.risk_metrics import conditional_value_at_risk

cvar_95 = conditional_value_at_risk(
    returns=returns,
    confidence_level=0.95
)
```

### Portfolio Risk Metrics

```python
from neutryx.valuations.risk_metrics import portfolio_var, portfolio_cvar

# Portfolio VaR
port_var = portfolio_var(
    positions=position_weights,      # [n_assets]
    returns_scenarios=return_matrix, # [n_scenarios, n_assets]
    confidence_level=0.95
)

# Portfolio CVaR
port_cvar = portfolio_cvar(
    positions=position_weights,
    returns_scenarios=return_matrix,
    confidence_level=0.95
)
```

### Advanced Risk Metrics

#### Incremental VaR

VaR impact of adding a position:

```python
from neutryx.valuations.risk_metrics import incremental_var

ivar = incremental_var(
    portfolio_returns=current_portfolio,
    position_returns=new_position,
    confidence_level=0.95
)
```

#### Component VaR

Decomposes portfolio VaR into position contributions:

```python
from neutryx.valuations.risk_metrics import component_var

comp_vars = component_var(
    positions=weights,
    returns_scenarios=scenarios,
    confidence_level=0.95
)
# Returns [n_assets] array where sum equals total portfolio VaR
```

#### Marginal VaR

Sensitivity of VaR to position changes:

```python
from neutryx.valuations.risk_metrics import marginal_var

mvar = marginal_var(
    positions=weights,
    returns_scenarios=scenarios,
    confidence_level=0.95
)
# Returns ∂VaR/∂position_i for each asset
```

### VaR Backtesting

Validate VaR model accuracy:

```python
from neutryx.valuations.risk_metrics import backtest_var

backtest_results = backtest_var(
    realized_returns=actual_returns,
    var_forecasts=predicted_var,
    confidence_level=0.95
)

print(f"Violations: {backtest_results['violations']}")
print(f"Violation Rate: {backtest_results['violation_rate']:.2%}")
print(f"Kupiec Test p-value: {backtest_results['kupiec_pvalue']:.4f}")
print(f"Pass: {backtest_results['pass_backtest']}")
```

### Other Risk Measures

```python
from neutryx.valuations.risk_metrics import (
    downside_deviation,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    compute_all_risk_metrics
)

# Comprehensive risk analysis
risk_metrics = compute_all_risk_metrics(
    returns=portfolio_returns,
    confidence_levels=[0.95, 0.99],
    risk_free_rate=0.03
)

print(risk_metrics)
# {
#   'mean': 0.0012,
#   'std': 0.0215,
#   'skewness': -0.45,
#   'kurtosis': 4.2,
#   'var_95': 0.0351,
#   'cvar_95': 0.0487,
#   'var_99': 0.0521,
#   'cvar_99': 0.0693,
#   'sharpe_ratio': 0.85,
#   'sortino_ratio': 1.12,
#   'downside_deviation': 0.0152,
#   'max_drawdown': 0.187
# }
```

---

## SIMM and Margin Calculations

### ISDA SIMM (Standard Initial Margin Model)

SIMM is the industry-standard methodology for calculating initial margin for uncleared OTC derivatives.

**Key Module**: [neutryx.valuations.simm](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/simm/)

#### Architecture

- **Risk Classes**: Interest Rate, FX, Credit (Qualifying/Non-Qualifying), Equity, Commodity
- **Risk Factors**: Delta, Vega, Curvature
- **Product Classes**: RatesFX, Credit, Equity, Commodity

#### Workflow

1. **Calculate Sensitivities**: Compute delta, vega sensitivities for each risk factor
2. **Bucket Sensitivities**: Group by risk class and bucket
3. **Apply Risk Weights**: Apply SIMM calibrated risk weights
4. **Aggregate**: Combine using correlation matrices
5. **Calculate IM**: Sum across risk classes with diversification

```python
from neutryx.valuations.simm import (
    SIMMCalculator,
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType
)

# Define sensitivities
sensitivities = [
    RiskFactorSensitivity(
        risk_factor="USD_2Y",
        risk_type=RiskFactorType.INTEREST_RATE,
        sensitivity_type=SensitivityType.DELTA,
        amount=100000,
        currency="USD"
    ),
    RiskFactorSensitivity(
        risk_factor="EUR_5Y",
        risk_type=RiskFactorType.INTEREST_RATE,
        sensitivity_type=SensitivityType.DELTA,
        amount=50000,
        currency="EUR"
    ),
    # ... more sensitivities
]

# Calculate SIMM IM
calculator = SIMMCalculator()
result = calculator.calculate(sensitivities)

print(f"Total IM: {result.total_im:,.2f}")
print(f"IM by Risk Class:")
for risk_class, im in result.im_by_risk_class.items():
    print(f"  {risk_class}: {im:,.2f}")
```

#### SIMM Risk Weights

Access calibrated risk weights and correlations:

```python
from neutryx.valuations.simm import get_risk_weights, get_correlations, RiskClass

# Interest rate risk weights
ir_weights = get_risk_weights(RiskClass.INTEREST_RATE)

# Intra-bucket correlations
correlations = get_correlations(
    risk_class=RiskClass.INTEREST_RATE,
    bucket_1="USD_2Y",
    bucket_2="USD_5Y"
)
```

### Initial Margin (Non-SIMM)

#### GRID (Gross Initial Margin)

```python
from neutryx.valuations.margin import calculate_grid_im

grid_im = calculate_grid_im(
    portfolio_value=1000000,
    asset_class="rates",
    maturity_bucket="5-10Y"
)
```

#### Schedule IM

For simpler products:

```python
from neutryx.valuations.margin import calculate_schedule_im

schedule_im = calculate_schedule_im(
    notional=10000000,
    product_type="interest_rate_swap",
    remaining_maturity=5.0
)
```

### Variation Margin

Mark-to-market collateral:

```python
from neutryx.valuations.margin import calculate_variation_margin, calculate_vm_call

# Current VM balance
vm = calculate_variation_margin(
    current_mtm=1500000,
    previous_vm_balance=1400000,
    threshold=100000,
    minimum_transfer_amount=50000
)

# VM call amount
vm_call = calculate_vm_call(
    current_mtm=1500000,
    collateral_held=1200000,
    threshold=100000,
    mta=50000
)
```

---

## Scenario Analysis

Scenario analysis evaluates portfolio performance under specific market conditions.

**Key Module**: [neutryx.valuations.scenarios](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/scenarios/)

### Scenario Framework

```python
from neutryx.valuations.scenarios import Scenario, ScenarioSet, Shock

# Define individual scenario
scenario = Scenario(
    name="Equity Rally",
    description="10% equity rally with vol compression",
    shocks=[
        Shock(factor="equity_spot", value=0.10, shock_type="relative"),
        Shock(factor="implied_vol", value=-0.20, shock_type="relative"),
    ]
)

# Create scenario set
scenario_set = ScenarioSet(
    name="Q4 2024 Scenarios",
    scenarios=[scenario1, scenario2, scenario3]
)

# Run scenarios
results = scenario_set.run(
    portfolio=my_portfolio,
    base_market_data=current_market
)

for result in results:
    print(f"{result.scenario_name}: P&L = {result.pnl:,.2f}")
```

### Market Data Bumpers

Apply systematic shocks to market data:

```python
from neutryx.valuations.scenarios import CurveBumper, SurfaceBumper, BumpType

# Parallel shift of yield curve
curve_bumper = CurveBumper(
    curve_name="USD_CURVE",
    bump_type=BumpType.PARALLEL,
    bump_size=0.0050  # 50bp
)

bumped_curve = curve_bumper.apply(original_curve)

# Volatility surface bump
vol_bumper = SurfaceBumper(
    surface_name="SPX_VOL",
    bump_type=BumpType.STICKY_STRIKE,
    bump_size=0.05  # 5 vol points
)

bumped_surface = vol_bumper.apply(original_surface)
```

### Market Scenario Types

```python
from neutryx.valuations.scenarios import MarketScenario

# Historical scenario
historical = MarketScenario.from_historical(
    date="2020-03-16",  # COVID crash
    reference_date="2020-02-19"
)

# Custom hypothetical
custom = MarketScenario(
    name="Custom Stress",
    equity_shock=-0.25,
    rate_shock=-0.02,
    fx_shock={"EURUSD": 0.05},
    vol_multiplier=2.0,
    credit_spread_shock=0.03
)

# Run scenario
result = custom.apply(portfolio, market_data)
```

---

## Stress Testing

Stress testing evaluates extreme but plausible adverse scenarios.

**Key Module**: [neutryx.valuations.stress_test](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/stress_test.py)

### Historical Stress Scenarios

Pre-defined historical crises:

```python
from neutryx.valuations.stress_test import (
    run_historical_stress_tests,
    HISTORICAL_SCENARIOS
)

# Available scenarios
print(HISTORICAL_SCENARIOS.keys())
# dict_keys(['black_monday_1987', 'financial_crisis_2008',
#            'flash_crash_2010', 'covid_crash_2020',
#            'rate_shock_up', 'rate_shock_down', 'volatility_spike'])

# Run all historical scenarios
results = run_historical_stress_tests(
    base_params={
        "spot": 100,
        "volatility": 0.20,
        "equity": 100,
        "rates": 0.05
    },
    valuation_fn=portfolio_pricer
)

for result in results:
    print(f"{result['scenario_name']}")
    print(f"  Base Value: ${result['base_value']:,.2f}")
    print(f"  Stressed Value: ${result['stressed_value']:,.2f}")
    print(f"  P&L: ${result['pnl']:,.2f} ({result['pnl_percent']:.2f}%)")
```

### Factor Stress Testing

Test sensitivity to individual factors:

```python
from neutryx.valuations.stress_test import factor_stress_test
import jax.numpy as jnp

# Equity shock range from -50% to +50%
shock_range = jnp.linspace(-0.50, 0.50, 101)

pnl_profile = factor_stress_test(
    base_params={"spot": 100, "volatility": 0.20, "rate": 0.05},
    valuation_fn=option_pricer,
    factor="spot",
    shock_range=shock_range
)

# Plot P&L profile
import matplotlib.pyplot as plt
plt.plot(shock_range, pnl_profile)
plt.xlabel("Spot Shock (%)")
plt.ylabel("P&L")
plt.title("P&L Profile vs Equity Shock")
plt.grid(True)
plt.show()
```

### Reverse Stress Testing

Find the shock level that produces a target loss:

```python
from neutryx.valuations.stress_test import reverse_stress_test

# Find equity drop that causes $1M loss
shock_required = reverse_stress_test(
    base_params={"spot": 100, "volatility": 0.20},
    valuation_fn=portfolio_pricer,
    factor="spot",
    target_loss=-1_000_000,
    search_range=(-0.99, 10.0)
)

print(f"Equity must drop by {shock_required*100:.1f}% to lose $1M")
```

### Custom Stress Scenarios

```python
from neutryx.valuations.stress_test import (
    StressScenario,
    run_stress_scenario
)

# Define custom scenario
custom_crisis = StressScenario(
    name="Custom Credit Crisis",
    description="Severe credit event with equity selloff",
    shocks={
        "equity": -0.40,
        "volatility": 3.0,
        "credit_spread": 0.08,
        "rates": -0.025
    }
)

result = run_stress_scenario(
    scenario=custom_crisis,
    base_params=base_market_params,
    valuation_fn=portfolio_valuation,
    shock_type="relative"
)
```

---

## Wrong-Way Risk

Wrong-way risk (WWR) occurs when exposure to a counterparty is adversely correlated with their credit quality.

**Key Module**: [neutryx.valuations.wrong_way_risk](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/wrong_way_risk.py)

### WWR Types

1. **General WWR**: Systemic correlation (macro factors)
2. **Specific WWR**: Direct dependence (e.g., CDS on counterparty's own name)

### CVA with Wrong-Way Risk

```python
from neutryx.valuations.wrong_way_risk import (
    cva_with_wwr,
    WWRParameters,
    WWRType
)
import jax

# Define WWR parameters
wwr_params = WWRParameters(
    correlation=0.50,  # Positive correlation = wrong-way risk
    wwr_type=WWRType.GENERAL,
    recovery_correlation=0.30
)

# Calculate CVA with WWR
key = jax.random.PRNGKey(42)
cva_wwr, cva_base, exposure_at_default = cva_with_wwr(
    key=key,
    exposure_paths=simulated_exposures,
    df_t=discount_factors,
    hazard_rate=0.02,  # 2% hazard rate
    lgd=0.60,
    wwr_params=wwr_params,
    T=5.0
)

wwr_charge = cva_wwr - cva_base
print(f"CVA without WWR: {cva_base:,.2f}")
print(f"CVA with WWR: {cva_wwr:,.2f}")
print(f"WWR Charge: {wwr_charge:,.2f} ({wwr_charge/cva_base*100:.1f}%)")
```

### Simulating Correlated Defaults

```python
from neutryx.valuations.wrong_way_risk import simulate_correlated_defaults

default_times, default_indicators = simulate_correlated_defaults(
    key=jax_key,
    exposure_paths=exposure_paths,
    hazard_rate=0.02,
    correlation=0.40,  # Correlation between exposure and default
    T=5.0
)
```

### Gaussian Copula WWR

Model joint distribution of exposure and default:

```python
from neutryx.valuations.wrong_way_risk import GaussianCopulaWWR
import jax.numpy as jnp

# Correlation matrix
correlation_matrix = jnp.array([
    [1.0, 0.5],
    [0.5, 1.0]
])

copula_model = GaussianCopulaWWR(
    correlation_matrix=correlation_matrix,
    n_factors=2
)

# Simulate joint scenarios
exposure_paths, credit_metrics = copula_model.simulate_joint(
    key=jax_key,
    n_paths=10000,
    n_steps=100,
    exposure_model=exposure_simulator,
    credit_model=credit_simulator
)
```

### WWR Adjustment Factor

First-order approximation:

```python
from neutryx.valuations.wrong_way_risk import wwr_adjustment_factor

adjustment = wwr_adjustment_factor(
    correlation=0.40,
    volatility_exposure=0.25,
    volatility_spread=0.15
)

# Approximate CVA with WWR
cva_wwr_approx = cva_base * adjustment
```

### Specific WWR Multiplier

For cases like CDS on counterparty's own name:

```python
from neutryx.valuations.wrong_way_risk import specific_wwr_multiplier

multiplier = specific_wwr_multiplier(
    exposure_to_reference=5_000_000,  # Exposure tied to counterparty
    total_exposure=10_000_000,
    jump_given_default=15.0  # Jump multiplier on default
)

cva_with_specific_wwr = cva_base * multiplier
```

### Comprehensive WWR Engine

```python
from neutryx.valuations.wrong_way_risk import WWREngine, WWRParameters

engine = WWREngine(
    general_wwr_params=WWRParameters(
        correlation=0.40,
        wwr_type=WWRType.GENERAL
    ),
    specific_wwr_exposure=2_000_000,
    specific_wwr_jump=10.0
)

result = engine.calculate_cva_adjustment(
    key=jax_key,
    exposure_paths=exposure_paths,
    df_t=discount_factors,
    hazard_rate=0.02,
    lgd=0.60,
    T=5.0
)

print(f"CVA Base: {result['cva_base']:,.2f}")
print(f"CVA with General WWR: {result['cva_general_wwr']:,.2f}")
print(f"CVA Total (with Specific WWR): {result['cva_total']:,.2f}")
print(f"Total WWR Charge: {result['wwr_charge']:,.2f}")
```

---

## P&L Attribution

P&L attribution decomposes portfolio P&L into risk factor contributions.

**Key Module**: [neutryx.valuations.pnl_attribution](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/valuations/pnl_attribution.py)

### Attribution Methods

1. **Greeks-Based**: Taylor expansion using first and second-order Greeks
2. **Revaluation**: Full revaluation with sequential factor bumps
3. **Hybrid**: Combines both approaches based on move size

### Basic Usage

```python
from neutryx.valuations.pnl_attribution import (
    PnLAttributionEngine,
    MarketState,
    AttributionMethod
)

# Define market states
start_state = MarketState(
    timestamp=0.0,
    spot_prices={"SPX": 4500, "AAPL": 180},
    volatilities={"SPX": 0.18, "AAPL": 0.25},
    interest_rates={"USD": 0.045},
)

end_state = MarketState(
    timestamp=1.0,
    spot_prices={"SPX": 4600, "AAPL": 185},
    volatilities={"SPX": 0.20, "AAPL": 0.23},
    interest_rates={"USD": 0.048},
)

# Create attribution engine
engine = PnLAttributionEngine(
    portfolio_pricer=portfolio_valuation_function,
    greeks_calculator=calculate_portfolio_greeks,
    method=AttributionMethod.HYBRID
)

# Perform attribution
attribution = engine.attribute_pnl(
    start_state=start_state,
    end_state=end_state
)

# Analyze results
print(f"Total P&L: ${attribution.total_pnl:,.2f}")
print(f"Theta: ${attribution.theta_pnl:,.2f}")
print(f"Spot P&L: ${attribution.total_spot_pnl():,.2f}")
print(f"  SPX: ${attribution.spot_pnl['SPX']:,.2f}")
print(f"  AAPL: ${attribution.spot_pnl['AAPL']:,.2f}")
print(f"Vol P&L: ${attribution.total_vol_pnl():,.2f}")
print(f"Rate P&L: ${attribution.total_rate_pnl():,.2f}")
print(f"Unexplained: ${attribution.unexplained_pnl:,.2f}")
print(f"Explanation Ratio: {attribution.explanation_ratio():.2%}")
```

### Daily P&L Tracking

```python
from neutryx.valuations.pnl_attribution import DailyPnLTracker

tracker = DailyPnLTracker()

# Add daily attributions
for date, start, end in daily_market_states:
    attribution = engine.attribute_pnl(start, end)
    tracker.add_attribution(date, attribution)

# Cumulative attribution
cumulative = tracker.cumulative_attribution()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(cumulative["theta"], label="Theta")
plt.plot(cumulative["spot"], label="Spot")
plt.plot(cumulative["vol"], label="Vol")
plt.plot(cumulative["rate"], label="Rate")
plt.plot(cumulative["unexplained"], label="Unexplained")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Cumulative P&L ($)")
plt.title("Cumulative P&L Attribution")
plt.grid(True)
plt.show()
```

### Analyzing P&L Drivers

```python
from neutryx.valuations.pnl_attribution import analyze_pnl_drivers

drivers = analyze_pnl_drivers(
    attribution=attribution,
    threshold=0.01  # Show contributors > 1% of total
)

print("Main P&L Drivers:")
for name, contribution in drivers:
    pct = contribution / attribution.total_pnl * 100
    print(f"  {name}: ${contribution:,.2f} ({pct:.1f}%)")
```

---

## API Reference

### Core Classes

#### ExposureSimulator

```python
class ExposureSimulator:
    """Simulate exposure profiles for XVA calculations."""

    def __init__(
        self,
        n_paths: int = 10000,
        n_steps: int = 100,
        horizon: float = 1.0
    ):
        """Initialize exposure simulator."""

    def simulate(
        self,
        key: jax.random.KeyArray,
        scenario: XVAScenario,
        product: Product
    ) -> ExposureResult:
        """Simulate exposures."""
```

#### SIMMCalculator

```python
class SIMMCalculator:
    """ISDA SIMM calculator."""

    def calculate(
        self,
        sensitivities: List[RiskFactorSensitivity]
    ) -> SIMMResult:
        """Calculate SIMM initial margin."""
```

#### PnLAttributionEngine

```python
class PnLAttributionEngine:
    """P&L attribution engine."""

    def __init__(
        self,
        portfolio_pricer: Callable,
        greeks_calculator: Optional[Callable] = None,
        method: AttributionMethod = AttributionMethod.HYBRID
    ):
        """Initialize attribution engine."""

    def attribute_pnl(
        self,
        start_state: MarketState,
        end_state: MarketState,
        start_portfolio_value: Optional[float] = None
    ) -> PnLAttribution:
        """Perform P&L attribution."""
```

### Key Functions

#### CVA Functions

```python
def cva(epe_t: Array, df_t: Array, pd_t: Array, lgd: float = 0.6) -> float:
    """Calculate Credit Valuation Adjustment."""

def bilateral_cva(
    epe_t: Array,
    ene_t: Array,
    df_t: Array,
    pd_counterparty_t: Array,
    pd_own_t: Array,
    lgd_counterparty: float = 0.6,
    lgd_own: float = 0.6
) -> tuple[float, float, float]:
    """Calculate bilateral CVA (CVA - DVA)."""
```

#### Risk Metrics Functions

```python
def value_at_risk(returns: Array, confidence_level: float = 0.95) -> float:
    """Compute Value at Risk."""

def conditional_value_at_risk(returns: Array, confidence_level: float = 0.95) -> float:
    """Compute Conditional VaR (Expected Shortfall)."""

def calculate_var(
    returns: Array,
    confidence_level: float = 0.95,
    method: VaRMethod = VaRMethod.HISTORICAL,
    **kwargs
) -> float:
    """Calculate VaR using specified methodology."""

def backtest_var(
    realized_returns: Array,
    var_forecasts: Array,
    confidence_level: float = 0.95
) -> dict:
    """Backtest VaR model."""
```

#### Stress Testing Functions

```python
def run_stress_scenario(
    scenario: StressScenario,
    base_params: Dict[str, float],
    valuation_fn: Callable,
    shock_type: str = "relative"
) -> Dict[str, float]:
    """Run a single stress scenario."""

def factor_stress_test(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    factor: str,
    shock_range: Array
) -> Array:
    """Run stress test across shock range."""

def reverse_stress_test(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    factor: str,
    target_loss: float,
    search_range: tuple = (-0.99, 10.0),
    tolerance: float = 1e-4
) -> float:
    """Find shock producing target loss."""
```

---

## Usage Examples

### Example 1: Complete XVA Calculation

```python
import jax
import jax.numpy as jnp
from neutryx.valuations.xva import ExposureSimulator, XVAScenario
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.xva.fva import fva
from neutryx.products import EuropeanOption

# Setup
key = jax.random.PRNGKey(42)
T = 5.0
n_steps = 100
times = jnp.linspace(0, T, n_steps)

# Market parameters
scenario = XVAScenario(
    spot=100.0,
    volatility=0.25,
    interest_rate=0.03
)

# Option
option = EuropeanOption(
    strike=100.0,
    maturity=T,
    is_call=True
)

# Simulate exposures
simulator = ExposureSimulator(n_paths=10000, n_steps=n_steps, horizon=T)
exposure_result = simulator.simulate(key, scenario, option)

# Discount factors
df_t = jnp.exp(-scenario.interest_rate * times)

# Default probabilities (from credit spread)
credit_spread = 0.0150  # 150bp
hazard_rate = credit_spread / 0.60  # Assuming 60% LGD
pd_t = 1 - jnp.exp(-hazard_rate * times)

# Calculate CVA
cva_value = cva(
    epe_t=exposure_result.epe,
    df_t=df_t,
    pd_t=pd_t,
    lgd=0.60
)

# Calculate FVA
funding_spread = 0.0050  # 50bp
fva_value = fva(
    epe_t=exposure_result.epe,
    funding_spread=funding_spread,
    df_t=df_t
)

# Total XVA
total_xva = cva_value + fva_value

# Clean price
clean_price = option.price(scenario.spot, scenario.volatility, scenario.interest_rate, T)

# Adjusted price
adjusted_price = clean_price - total_xva

print(f"Clean Price: ${clean_price:.2f}")
print(f"CVA: ${cva_value:.2f}")
print(f"FVA: ${fva_value:.2f}")
print(f"Total XVA: ${total_xva:.2f}")
print(f"XVA-Adjusted Price: ${adjusted_price:.2f}")
```

### Example 2: Portfolio Risk Analysis

```python
import jax.numpy as jnp
from neutryx.valuations.risk_metrics import (
    compute_all_risk_metrics,
    portfolio_var,
    portfolio_cvar,
    component_var
)

# Historical returns for 3 assets
returns = jnp.array([
    # 252 days of returns for 3 assets
    # ...
])

# Portfolio weights
weights = jnp.array([0.40, 0.35, 0.25])

# Comprehensive risk metrics
risk_metrics = compute_all_risk_metrics(
    returns=returns @ weights,  # Portfolio returns
    confidence_levels=[0.95, 0.99],
    risk_free_rate=0.03
)

print("Portfolio Risk Metrics:")
for metric, value in risk_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Component VaR
comp_vars = component_var(
    positions=weights,
    returns_scenarios=returns,
    confidence_level=0.95
)

print("\nComponent VaR:")
for i, cvar in enumerate(comp_vars):
    print(f"  Asset {i+1}: {cvar:.4f} ({cvar/sum(comp_vars)*100:.1f}%)")
```

### Example 3: Stress Testing Workflow

```python
from neutryx.valuations.stress_test import (
    run_historical_stress_tests,
    factor_stress_test,
    reverse_stress_test
)
import jax.numpy as jnp

# Portfolio valuation function
def portfolio_value(spot, volatility, rates):
    # ... calculate portfolio value
    return value

# Base parameters
base_params = {
    "spot": 100.0,
    "volatility": 0.20,
    "rates": 0.05
}

# 1. Historical stress tests
historical_results = run_historical_stress_tests(
    base_params=base_params,
    valuation_fn=portfolio_value
)

print("Historical Stress Test Results:")
for result in historical_results:
    print(f"{result['scenario_name']}: {result['pnl_percent']:.2f}%")

# 2. Factor stress test
shock_range = jnp.linspace(-0.50, 0.50, 101)
pnl_profile = factor_stress_test(
    base_params=base_params,
    valuation_fn=portfolio_value,
    factor="spot",
    shock_range=shock_range
)

# 3. Reverse stress test
shock_for_1m_loss = reverse_stress_test(
    base_params=base_params,
    valuation_fn=portfolio_value,
    factor="spot",
    target_loss=-1_000_000
)

print(f"\nShock required for $1M loss: {shock_for_1m_loss*100:.2f}%")
```

### Example 4: SIMM Calculation

```python
from neutryx.valuations.simm import (
    SIMMCalculator,
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType
)

# Define portfolio sensitivities
sensitivities = []

# Interest rate deltas
for tenor in ["2Y", "5Y", "10Y"]:
    sensitivities.append(
        RiskFactorSensitivity(
            risk_factor=f"USD_{tenor}",
            risk_type=RiskFactorType.INTEREST_RATE,
            sensitivity_type=SensitivityType.DELTA,
            amount=100_000 * (1 + hash(tenor) % 3),
            currency="USD"
        )
    )

# FX deltas
for ccy_pair in ["EURUSD", "GBPUSD"]:
    sensitivities.append(
        RiskFactorSensitivity(
            risk_factor=ccy_pair,
            risk_type=RiskFactorType.FX,
            sensitivity_type=SensitivityType.DELTA,
            amount=50_000,
            currency="USD"
        )
    )

# Equity deltas
for index in ["SPX", "SX5E"]:
    sensitivities.append(
        RiskFactorSensitivity(
            risk_factor=index,
            risk_type=RiskFactorType.EQUITY,
            sensitivity_type=SensitivityType.DELTA,
            amount=200_000,
            currency="USD"
        )
    )

# Calculate SIMM IM
calculator = SIMMCalculator()
result = calculator.calculate(sensitivities)

print(f"Total Initial Margin: ${result.total_im:,.2f}")
print("\nBreakdown by Risk Class:")
for risk_class, im in result.im_by_risk_class.items():
    pct = im / result.total_im * 100
    print(f"  {risk_class}: ${im:,.2f} ({pct:.1f}%)")
```

---

## Best Practices

### 1. Performance Optimization

**Use JAX JIT Compilation**:

```python
import jax
from functools import partial

@jax.jit
def calculate_portfolio_var(returns, weights, confidence):
    portfolio_returns = returns @ weights
    return jnp.quantile(-portfolio_returns, confidence)

# Compile once, use many times
var_95 = calculate_portfolio_var(returns, weights, 0.95)
```

**Vectorize Operations**:

```python
# Bad: Loop over scenarios
results = []
for i in range(n_scenarios):
    result = calculate_cva(scenarios[i])
    results.append(result)

# Good: Vectorized
results = jax.vmap(calculate_cva)(scenarios)
```

### 2. Numerical Stability

**Avoid Division by Zero**:

```python
# Add small epsilon
sharpe = (mean_return - rf) / (std_return + 1e-10)
```

**Use Log-Space for Probabilities**:

```python
# For small probabilities, use log-space
log_prob = jnp.log(prob + 1e-300)
```

### 3. Model Validation

**Always Backtest VaR**:

```python
from neutryx.valuations.risk_metrics import backtest_var

backtest_results = backtest_var(
    realized_returns=actual_returns,
    var_forecasts=var_predictions,
    confidence_level=0.95
)

assert backtest_results['pass_backtest'], "VaR model failed backtesting"
```

**Validate Greeks with Finite Differences**:

```python
def validate_delta(greeks_delta, spot, epsilon=0.01):
    price_up = price(spot + epsilon)
    price_down = price(spot - epsilon)
    fd_delta = (price_up - price_down) / (2 * epsilon)

    assert jnp.abs(greeks_delta - fd_delta) < 0.01, "Delta validation failed"
```

### 4. Risk Management

**Implement Risk Limits**:

```python
# Portfolio VaR limit
MAX_VAR = 1_000_000

portfolio_var = calculate_var(portfolio_returns, 0.95)
assert portfolio_var <= MAX_VAR, f"VaR limit breach: {portfolio_var:,.2f}"
```

**Monitor Concentration Risk**:

```python
# Component VaR to identify concentrations
comp_vars = component_var(weights, returns, 0.95)
max_contribution = jnp.max(comp_vars) / jnp.sum(comp_vars)

if max_contribution > 0.40:
    print(f"Warning: High concentration risk ({max_contribution:.1%})")
```

### 5. Documentation

**Document Assumptions**:

```python
def cva(epe_t, df_t, pd_t, lgd=0.60):
    """Calculate CVA.

    Assumptions:
    - Independence between exposure and default
    - No wrong-way risk
    - Constant LGD
    - No collateral
    """
    ...
```

### 6. Testing

**Unit Tests for All Functions**:

```python
import pytest

def test_cva_calculation():
    """Test CVA calculation."""
    epe = jnp.array([100, 110, 105, 95])
    df = jnp.array([0.99, 0.98, 0.97, 0.96])
    pd = jnp.array([0.01, 0.02, 0.03, 0.04])

    result = cva(epe, df, pd, lgd=0.60)

    assert result > 0, "CVA should be positive"
    assert result < 100, "CVA should be less than max exposure"
```

### 7. Code Organization

**Separate Configuration**:

```python
# config.py
XVA_CONFIG = {
    "lgd_default": 0.60,
    "funding_spread": 0.0050,
    "n_paths": 10000,
    "n_steps": 100
}

# usage
from config import XVA_CONFIG

cva_value = cva(epe, df, pd, lgd=XVA_CONFIG["lgd_default"])
```

### 8. Monitoring and Logging

**Log Key Metrics**:

```python
import logging

logger = logging.getLogger(__name__)

def calculate_portfolio_xva(portfolio):
    logger.info(f"Calculating XVA for portfolio {portfolio.id}")

    cva_value = calculate_cva(portfolio)
    logger.info(f"CVA: {cva_value:,.2f}")

    if cva_value > 1_000_000:
        logger.warning(f"High CVA detected: {cva_value:,.2f}")

    return cva_value
```

---

## Conclusion

The Neutryx Valuations framework provides enterprise-grade tools for derivatives valuation and risk management. Key takeaways:

1. **Comprehensive Coverage**: XVA, Greeks, risk metrics, SIMM, scenarios, and attribution
2. **High Performance**: JAX-accelerated for GPU/TPU computation
3. **Production-Ready**: Industry-standard implementations
4. **Extensible**: Modular design for customization
5. **Well-Tested**: Extensive validation and backtesting

For additional support:
- **API Documentation**: See individual module docstrings
- **Examples**: Check `/examples` directory
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions on Discord

---

**Last Updated**: 2024-11-04
**Version**: 1.0.0
**Module**: neutryx.valuations
