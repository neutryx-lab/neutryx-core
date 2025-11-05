# Risk Management Hub

> **Enterprise-grade risk analytics: From real-time VaR to regulatory capital**

Comprehensive risk management framework for derivatives trading, portfolio analytics, and regulatory compliance. Built on JAX for high-performance computation with GPU acceleration.

---

## Overview

Neutryx Core's Risk Management framework provides a complete suite of tools for measuring, monitoring, and managing market risk across multi-asset portfolios. From real-time pre-trade controls to overnight batch VaR calculations, our JAX-powered engine delivers **10-100x faster** performance than legacy systems.

**Key Capabilities:**
- **VaR & Expected Shortfall**: Historical, Monte Carlo, parametric methodologies
- **Stress Testing**: 25+ historical scenarios, hypothetical shocks, reverse stress testing
- **Position Limits & Controls**: Real-time pre-trade checks with hierarchical breach thresholds
- **Greeks & Sensitivities**: DV01, CS01, vega bucketing, higher-order Greeks
- **P&L Attribution**: Daily explain (carry, delta, gamma, vega, theta, rho)
- **Regulatory Reporting**: FRTB SA/IMA, SA-CCR, SIMM, Basel III/IV compliance

---

## Core Components

### 1. Risk Engine

**Central risk calculation engine** with support for multiple methodologies and portfolio-level aggregation.

**Features:**
- Multi-threaded computation with JAX parallelization
- Incremental risk calculation for large portfolios
- Historical data management with TimescaleDB integration
- Real-time market data feeds from Bloomberg/Refinitiv
- Configurable confidence levels and time horizons

**API:**
```python
from neutryx.risk import RiskEngine, VaRMethod

# Initialize risk engine
engine = RiskEngine(
    var_method=VaRMethod.HISTORICAL,
    confidence_level=0.99,
    holding_period_days=10
)

# Calculate portfolio VaR
portfolio_var = engine.calculate_var(portfolio, market_data)

# Calculate Expected Shortfall (CVaR)
portfolio_es = engine.calculate_expected_shortfall(portfolio, market_data)
```

### 2. VaR Methodologies

Comprehensive Value-at-Risk calculation with multiple approaches.

#### Historical Simulation VaR

**Non-parametric approach** using historical price movements.

**Advantages:**
- No distribution assumptions
- Captures fat tails and skewness
- Easy to explain and implement

**Code Example:**
```python
from neutryx.risk import compute_var, VaRMethod

# Calculate historical VaR at 99% confidence
var_99 = compute_var(
    returns=historical_returns,
    confidence_level=0.99,
    method=VaRMethod.HISTORICAL
)

print(f"1-day 99% VaR: ${var_99:,.2f}")
```

**Options:**
- Weighted historical simulation (exponential decay)
- Filtered historical simulation (GARCH volatility adjustment)
- Hybrid approaches combining historical and parametric methods

#### Monte Carlo VaR

**Simulation-based VaR** using stochastic models.

**Advantages:**
- Captures complex dependencies
- Handles non-linear products (options, exotics)
- Flexible scenario generation

**Code Example:**
```python
from neutryx.risk import generate_monte_carlo_scenarios, scenario_expected_shortfall

# Generate 10,000 Monte Carlo scenarios
scenarios = generate_monte_carlo_scenarios(
    portfolio=portfolio,
    model=heston_model,
    num_scenarios=10_000,
    horizon_days=10
)

# Calculate VaR and ES from scenarios
var_mc = compute_var(scenarios.returns, confidence_level=0.99)
es_mc = scenario_expected_shortfall(scenarios, confidence_level=0.99)

print(f"10-day 99% VaR (MC): ${var_mc:,.2f}")
print(f"10-day 99% ES (MC): ${es_mc:,.2f}")
```

**Variance Reduction:**
- Antithetic variates
- Control variates
- Importance sampling
- Quasi-random numbers (Sobol sequences)

#### Parametric VaR

**Closed-form VaR** assuming normal distribution or Cornish-Fisher expansion.

**Advantages:**
- Fastest computation
- Suitable for linear portfolios
- Easy to decompose by risk factors

**Code Example:**
```python
# Parametric VaR with normal distribution
var_param = compute_var(
    returns=historical_returns,
    confidence_level=0.99,
    method=VaRMethod.PARAMETRIC
)

# Parametric VaR with Cornish-Fisher correction for skewness/kurtosis
var_cf = compute_var(
    returns=historical_returns,
    confidence_level=0.99,
    method=VaRMethod.CORNISH_FISHER
)
```

#### Expected Shortfall (CVaR)

**Conditional VaR** measuring average loss beyond VaR threshold.

**Advantages:**
- Coherent risk measure (sub-additive)
- Captures tail risk better than VaR
- Preferred by regulators (Basel III, FRTB IMA)

**Code Example:**
```python
from neutryx.risk import compute_expected_shortfall

# Calculate Expected Shortfall (CVaR)
es = compute_expected_shortfall(returns, confidence_level=0.975)

# ES is always greater than or equal to VaR
var = compute_var(returns, confidence_level=0.975)
assert es >= var
```

#### Component VaR & Incremental VaR

**Portfolio decomposition** to understand risk contributions.

**Code Example:**
```python
from neutryx.risk import risk_factor_attribution, RiskFactorAttributionMethod

# Component VaR: contribution of each position
component_var = risk_factor_attribution(
    portfolio=portfolio,
    market_data=market_data,
    method=RiskFactorAttributionMethod.COMPONENT
)

# Incremental VaR: marginal impact of adding/removing a position
incremental_var = risk_factor_attribution(
    portfolio=portfolio,
    market_data=market_data,
    method=RiskFactorAttributionMethod.INCREMENTAL,
    trade_id="SWAP_12345"
)

print(f"Incremental VaR of SWAP_12345: ${incremental_var:,.2f}")
```

### 3. Stress Testing

**Scenario-based risk analysis** with 25+ historical scenarios and custom shocks.

#### Historical Stress Scenarios

**Pre-built scenarios** from major market events:
- **2008 Global Financial Crisis**: Lehman collapse, credit crunch
- **2011 European Debt Crisis**: Greek sovereign default fears
- **2013 Taper Tantrum**: Fed tapering announcement
- **2015 Yuan Devaluation**: Chinese currency shock
- **2016 Brexit**: UK referendum result
- **2018 Q4 Selloff**: Equity market correction
- **2020 COVID-19 Pandemic**: Global lockdowns, volatility spike
- **2022 Russia-Ukraine**: Commodity shock, inflation surge
- **2023 Banking Crisis**: SVB collapse, Credit Suisse rescue

**Code Example:**
```python
from neutryx.risk import run_stress_tests, HISTORICAL_SCENARIOS

# Run all 25+ historical scenarios
results = run_stress_tests(
    portfolio=portfolio,
    scenarios=HISTORICAL_SCENARIOS,
    valuation_fn=portfolio_pricer
)

# Display scenario impacts
for scenario_name, pnl_impact in results.items():
    print(f"{scenario_name}: ${pnl_impact:,.2f}")

# Identify worst-case scenario
worst_scenario = min(results, key=results.get)
worst_loss = results[worst_scenario]
print(f"\nWorst scenario: {worst_scenario} with loss of ${worst_loss:,.2f}")
```

#### Custom Stress Scenarios

**Define your own scenarios** with market factor shocks.

**Code Example:**
```python
from neutryx.risk import StressScenario, run_stress_test

# Define custom stress scenario
custom_scenario = StressScenario(
    name="Rates +200bp, Equity -30%, Vol +50%",
    shocks={
        "interest_rates": 0.02,  # +200 basis points
        "equity_spot": -0.30,    # -30%
        "volatility": 0.50       # +50%
    }
)

# Run custom scenario
pnl_impact = run_stress_test(
    scenario=custom_scenario,
    base_params=current_market_data,
    valuation_fn=portfolio_pricer
)

print(f"Custom scenario P&L: ${pnl_impact:,.2f}")
```

#### Reverse Stress Testing

**Identify scenarios** that breach risk limits or cause unacceptable losses.

**Code Example:**
```python
from neutryx.risk import reverse_stress_test

# Find scenarios that breach VaR limit
breach_scenarios = reverse_stress_test(
    portfolio=portfolio,
    loss_threshold=10_000_000,  # $10M loss
    market_factors=["interest_rates", "fx_rates", "equity_spot"],
    num_scenarios=1000
)

print(f"Found {len(breach_scenarios)} scenarios exceeding $10M loss")
```

### 4. Position Limits & Pre-Trade Controls

**Real-time limit checking** with hierarchical breach thresholds.

#### Limit Types

**Comprehensive limit framework** covering all risk dimensions:

1. **Notional Limits**: Absolute exposure by product/desk/trader
2. **VaR Limits**: Value-at-Risk limits with utilization tracking
3. **Concentration Limits**: Single-name, sector, geography limits
4. **Issuer Exposure Limits**: Credit exposure to counterparties
5. **Greek Limits**: Delta, vega, gamma exposure limits
6. **Tenor Limits**: Exposure by maturity bucket

**Code Example:**
```python
from neutryx.risk import (
    LimitManager, NotionalLimit, VaRLimit,
    ConcentrationLimit, LimitType
)

# Define limits
limits = LimitManager()

# Notional limit: $500M for rates desk
limits.add_limit(NotionalLimit(
    name="Rates Desk Notional",
    hard_limit=500_000_000,
    soft_limit=400_000_000,
    warning_threshold=0.8,
    scope={"desk": "rates"}
))

# VaR limit: $10M daily VaR at 99%
limits.add_limit(VaRLimit(
    name="Rates Desk VaR",
    hard_limit=10_000_000,
    confidence_level=0.99,
    holding_period_days=1,
    scope={"desk": "rates"}
))

# Concentration limit: max 10% in single issuer
limits.add_limit(ConcentrationLimit(
    name="Single Name Concentration",
    hard_limit=0.10,  # 10% of portfolio
    limit_type=LimitType.CONCENTRATION,
    scope={"portfolio": "credit"}
))
```

#### Pre-Trade Controls

**Real-time limit checking** before trade execution.

**Code Example:**
```python
from neutryx.risk import pre_trade_control, PreTradeCheck

# Define proposed trade
proposed_trade = Trade(
    instrument="USD_IRS_10Y",
    notional=50_000_000,
    direction="receive_fixed"
)

# Run pre-trade check
check_result = pre_trade_control(
    trade=proposed_trade,
    portfolio=current_portfolio,
    limits=limits,
    market_data=market_data
)

if check_result.approved:
    print("Trade approved for execution")
else:
    print(f"Trade rejected: {check_result.breach_reasons}")
    for breach in check_result.breaches:
        print(f"  - {breach.limit_name}: {breach.severity}")
```

**Breach Thresholds:**
- **OK**: < 80% of hard limit (green)
- **WARNING**: 80-100% of hard limit (yellow)
- **SOFT BREACH**: 100-120% of hard limit (orange, requires approval)
- **HARD BREACH**: > 120% of hard limit (red, not tradeable)

#### What-If Analysis

**Scenario-based impact assessment** for proposed trades.

**Code Example:**
```python
from neutryx.risk import what_if_analysis, WhatIfScenario

# Define what-if scenarios
scenarios = [
    WhatIfScenario(
        name="Add $100M 10Y Swap",
        trade=Trade("USD_IRS_10Y", notional=100_000_000)
    ),
    WhatIfScenario(
        name="Add $50M EUR/USD FX Forward",
        trade=Trade("EURUSD_FWD_1Y", notional=50_000_000)
    )
]

# Analyze impact on portfolio metrics
results = what_if_analysis(
    scenarios=scenarios,
    portfolio=current_portfolio,
    metrics=["var", "expected_shortfall", "dv01", "cs01"]
)

# Display results
for scenario_name, metrics in results.items():
    print(f"\n{scenario_name}:")
    print(f"  Current VaR: ${metrics['var_before']:,.2f}")
    print(f"  New VaR: ${metrics['var_after']:,.2f}")
    print(f"  Incremental VaR: ${metrics['var_delta']:,.2f}")
```

### 5. Greeks & Sensitivity Analysis

**Comprehensive Greek calculation** with automatic differentiation.

#### First-Order Greeks

**Delta, DV01, CS01, FX Delta, Vega**

**Code Example:**
```python
from neutryx.risk import (
    calculate_dv01, calculate_cs01,
    calculate_vega_surface, calculate_fx_greeks
)

# DV01: Interest rate sensitivity (PV change per 1bp shift)
dv01 = calculate_dv01(
    portfolio=portfolio,
    market_data=market_data,
    tenor_points=["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
)

print("DV01 by tenor:")
for tenor, sensitivity in dv01.items():
    print(f"  {tenor}: ${sensitivity:,.2f}")

# CS01: Credit spread sensitivity
cs01 = calculate_cs01(
    portfolio=credit_portfolio,
    market_data=market_data,
    issuers=["AAPL", "MSFT", "JPM"]
)

# Vega surface: volatility sensitivity by strike and maturity
vega_surface = calculate_vega_surface(
    portfolio=options_portfolio,
    market_data=market_data,
    strikes=[90, 95, 100, 105, 110],
    maturities=["1M", "3M", "6M", "1Y"]
)

# FX Greeks: delta, gamma, vega for FX options
fx_greeks = calculate_fx_greeks(
    portfolio=fx_portfolio,
    market_data=market_data,
    currency_pairs=["EURUSD", "GBPUSD", "USDJPY"]
)
```

#### Higher-Order Greeks

**Gamma, vanna, volga, charm, veta, speed, zomma, color**

**Code Example:**
```python
from neutryx.risk import calculate_higher_order_greeks

# Calculate all higher-order Greeks
greeks = calculate_higher_order_greeks(
    portfolio=options_portfolio,
    market_data=market_data,
    include=[
        "gamma",   # Convexity (second derivative wrt spot)
        "vanna",   # Cross-derivative (spot vs volatility)
        "volga",   # Volatility convexity
        "charm",   # Delta decay (delta vs time)
        "veta",    # Vega decay (vega vs time)
        "speed",   # Gamma of gamma
        "zomma",   # Gamma wrt volatility
        "color"    # Gamma decay (gamma vs time)
    ]
)

print(f"Portfolio gamma: {greeks['gamma']:,.2f}")
print(f"Portfolio vanna: {greeks['vanna']:,.2f}")
print(f"Portfolio volga: {greeks['volga']:,.2f}")
```

**Use Cases:**
- **Gamma**: Hedging convexity risk, managing P&L volatility
- **Vanna**: Managing spot-vol correlation risk
- **Volga**: Volatility convexity hedging
- **Charm**: Time decay of delta hedge
- **Veta**: Vega decay for long option portfolios

#### Portfolio-Level Aggregation

**Aggregate sensitivities** across positions with netting.

**Code Example:**
```python
from neutryx.risk import aggregate_portfolio_sensitivities

# Calculate portfolio-level Greeks with netting
portfolio_greeks = aggregate_portfolio_sensitivities(
    portfolio=portfolio,
    market_data=market_data,
    sensitivity_types=["delta", "gamma", "vega", "theta", "rho"],
    netting_sets=["desk", "asset_class"]
)

# Format report
from neutryx.risk import format_sensitivity_report

report = format_sensitivity_report(
    sensitivities=portfolio_greeks,
    format="html"  # or "pdf", "excel", "json"
)
```

### 6. P&L Attribution

**Daily P&L explain** decomposed into risk factors.

**Code Example:**
```python
from neutryx.risk import explain_pnl, AttributionMethod

# P&L attribution for previous day
attribution = explain_pnl(
    portfolio=portfolio,
    market_state_t0=yesterday_market_data,
    market_state_t1=today_market_data,
    method=AttributionMethod.TAYLOR_EXPANSION
)

# Display attribution breakdown
print(f"Total P&L: ${attribution.total_pnl:,.2f}")
print(f"\nBreakdown:")
print(f"  Carry: ${attribution.carry:,.2f}")
print(f"  Delta: ${attribution.delta_pnl:,.2f}")
print(f"  Gamma: ${attribution.gamma_pnl:,.2f}")
print(f"  Vega: ${attribution.vega_pnl:,.2f}")
print(f"  Theta: ${attribution.theta_pnl:,.2f}")
print(f"  Rho: ${attribution.rho_pnl:,.2f}")
print(f"  Unexplained: ${attribution.residual:,.2f}")

# FRTB P&L attribution test
frtb_test_result = attribution.frtb_test(threshold=0.10)  # 10% threshold
print(f"\nFRTB Test: {'PASS' if frtb_test_result.passed else 'FAIL'}")
```

**Attribution Methods:**
- **Taylor Expansion**: Linear + quadratic terms (delta + 0.5 * gamma)
- **Finite Differences**: Exact revaluation with bumped market data
- **Risk Factor Attribution**: Attribution to specific risk factors (curves, surfaces)

### 7. Regulatory Risk Reporting

**Automated generation** of regulatory risk reports.

#### FRTB (Fundamental Review of the Trading Book)

**Standardized Approach (SA):**
- Delta risk charge by risk class (IR, FX, EQ, CR, CM)
- Vega risk charge with smile risk
- Curvature risk charge for non-linear products
- Default Risk Charge (DRC)
- Residual Risk Add-On (RRAO)

**Internal Models Approach (IMA):**
- Expected Shortfall at 97.5% confidence
- P&L attribution test (Spearman correlation > 0.8)
- Backtesting with traffic light approach
- Non-Modellable Risk Factors (NMRF)

**See:** [FRTB Documentation](../src/neutryx/regulatory/)

#### SA-CCR (Counterparty Credit Risk)

**Standardized approach** for calculating counterparty exposure:
- Replacement Cost (RC)
- Potential Future Exposure (PFE) add-on
- Hedging set construction
- Asset class-specific calculations

**See:** [SA-CCR Implementation](../src/neutryx/regulatory/)

#### SIMM (Standard Initial Margin Model)

**ISDA SIMM 2.6** implementation:
- Risk factor sensitivities (delta, vega, curvature)
- Correlation matrices by product class
- Concentration thresholds
- Initial margin calculation

**See:** [SIMM Documentation](../src/neutryx/valuations/simm/)

---

## Quick Start

### Installation

```bash
pip install neutryx-core
```

### Basic VaR Calculation

```python
import jax.numpy as jnp
from neutryx.risk import compute_var, VaRMethod

# Sample portfolio returns (1000 days)
returns = jnp.array([...])  # Your historical returns

# Calculate 99% 1-day VaR
var_99 = compute_var(
    returns=returns,
    confidence_level=0.99,
    method=VaRMethod.HISTORICAL
)

print(f"1-day 99% VaR: ${var_99:,.2f}")
```

### Stress Testing

```python
from neutryx.risk import run_stress_tests, HISTORICAL_SCENARIOS

# Run all historical scenarios
results = run_stress_tests(
    portfolio=my_portfolio,
    scenarios=HISTORICAL_SCENARIOS,
    valuation_fn=my_pricer
)

# Display worst scenario
worst_scenario = min(results, key=results.get)
print(f"Worst scenario: {worst_scenario}")
print(f"Loss: ${results[worst_scenario]:,.2f}")
```

### Pre-Trade Control

```python
from neutryx.risk import pre_trade_control, LimitManager

# Setup limits
limits = LimitManager()
limits.add_limit(NotionalLimit("Desk Limit", hard_limit=500_000_000))

# Check proposed trade
result = pre_trade_control(
    trade=proposed_trade,
    portfolio=current_portfolio,
    limits=limits
)

if result.approved:
    execute_trade(proposed_trade)
else:
    print(f"Trade rejected: {result.breach_reasons}")
```

---

## Performance Benchmarks

**VaR Calculation (10,000 scenarios, 500-position portfolio):**
- NumPy: ~2,000ms
- JAX (CPU): ~200ms (10x faster)
- JAX (GPU): ~20ms (100x faster)

**Greek Calculation (1,000-position portfolio):**
- Finite Differences: ~5,000ms
- Automatic Differentiation: ~500ms (10x faster)

**Stress Testing (25 scenarios, 500-position portfolio):**
- Sequential: ~25,000ms
- Parallel (JAX pmap): ~2,500ms (10x faster)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Risk Management Layer                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  VaR Engine  │  │ Stress Test  │  │    Limits    │     │
│  │  (HS/MC/P)   │  │   Engine     │  │   Manager    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Greeks     │  │     P&L      │  │  Regulatory  │     │
│  │ Calculator   │  │ Attribution  │  │   Reports    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│                   Valuation & Pricing Layer                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Portfolio   │  │  Scenario    │  │   Market     │     │
│  │   Pricer     │  │   Engine     │  │    Data      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│                      JAX Computation Layer                  │
├─────────────────────────────────────────────────────────────┤
│  JIT Compilation │ Auto Differentiation │ GPU Acceleration │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Examples

### Integration with Trading Systems

```python
from neutryx.risk import RiskEngine, pre_trade_control

# Initialize risk engine
risk_engine = RiskEngine()

# Hook into order management system
@trading_system.on_order_received
def check_risk_before_execution(order):
    result = pre_trade_control(
        trade=order,
        portfolio=get_current_portfolio(),
        limits=get_active_limits()
    )

    if result.approved:
        return trading_system.execute(order)
    else:
        return trading_system.reject(order, reason=result.breach_reasons)
```

### Integration with Risk Dashboards

```python
from neutryx.risk import RiskEngine
import dash
import plotly.graph_objs as go

app = dash.Dash(__name__)

@app.callback(Output('var-chart', 'figure'))
def update_var_chart():
    # Calculate VaR for last 250 days
    var_history = []
    for date in last_250_days:
        portfolio = get_portfolio_snapshot(date)
        var = risk_engine.calculate_var(portfolio)
        var_history.append({'date': date, 'var': var})

    # Create chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d['date'] for d in var_history],
        y=[d['var'] for d in var_history],
        name='99% VaR'
    ))
    return fig
```

### Integration with Regulatory Reporting

```python
from neutryx.risk import RiskEngine
from neutryx.regulatory import FRTBReporter

# Generate daily FRTB report
risk_engine = RiskEngine()
frtb_reporter = FRTBReporter()

def generate_daily_frtb_report():
    # Calculate risk metrics
    var = risk_engine.calculate_var(portfolio)
    es = risk_engine.calculate_expected_shortfall(portfolio)

    # Generate FRTB report
    report = frtb_reporter.generate_report(
        portfolio=portfolio,
        market_data=market_data,
        var=var,
        expected_shortfall=es
    )

    # Submit to regulator
    submit_to_regulator(report.to_xml())
```

---

## Best Practices

### 1. VaR Calculation

**Recommendations:**
- Use **historical simulation** for fat-tailed distributions
- Use **Monte Carlo** for portfolios with non-linear products
- Use **parametric VaR** only for linear portfolios
- Always calculate **Expected Shortfall** alongside VaR
- Use **appropriate lookback window**: 250 days (1 year) or 500 days (2 years)
- **Scale VaR** appropriately: $\text{VaR}_T = \text{VaR}_1 \times \sqrt{T}$ (only valid for i.i.d. returns)

### 2. Stress Testing

**Recommendations:**
- Run stress tests **at least daily**, more frequently for volatile markets
- Include **both historical and hypothetical** scenarios
- Perform **reverse stress testing** quarterly to identify vulnerabilities
- Document all stress scenarios and review annually
- Consider **tail scenarios** (1-in-100 year events)

### 3. Position Limits

**Recommendations:**
- Set **hard limits** at 120% of target exposure
- Set **soft limits** at 100% of target exposure
- Set **warning thresholds** at 80% of target exposure
- Review limits **quarterly** and adjust for market conditions
- Implement **tiered approval workflows**: trader → desk head → CRO
- Monitor limit utilization in **real-time** dashboards

### 4. Greeks Calculation

**Recommendations:**
- Use **automatic differentiation** for accurate Greeks (avoid finite differences)
- Calculate **portfolio-level Greeks** with proper netting
- Monitor **higher-order Greeks** (gamma, vanna, volga) for option portfolios
- Implement **Greek hedging** strategies to reduce P&L volatility
- Validate Greeks against **finite difference** calculations periodically

### 5. P&L Attribution

**Recommendations:**
- Perform **daily P&L attribution** to identify unexplained P&L
- Set **tolerance threshold** at 10% for FRTB P&L test
- Investigate **large residuals** (> 5% of total P&L)
- Track **P&L attribution quality** over time (Spearman correlation)
- Document model changes and their impact on attribution

---

## Resources

### Documentation
- [Risk Masterclass](tutorials/risk_masterclass.md) - Comprehensive risk management tutorial
- [Risk Controls Atlas](controls/risk_controls_atlas.md) - Pre-trade controls and limit management
- [Enterprise Risk Reference](reference/enterprise_risk_reference.md) - Detailed API reference
- [Risk Grandmaster Codex](reference/risk_grandmaster_codex.md) - Advanced risk techniques

### Code Examples
- [VaR Calculation Examples](../../examples/var_calculation_example.py)
- [Stress Testing Examples](../../examples/stress_testing_example.py)
- [Greek Calculation Examples](../../examples/greeks_example.py)
- [Pre-Trade Controls Demo](../../examples/pre_trade_controls_demo.py)

### Regulatory Guidance
- [FRTB Implementation](../src/neutryx/regulatory/frtb/)
- [SA-CCR Calculator](../src/neutryx/regulatory/sa_ccr/)
- [SIMM 2.6 Documentation](../src/neutryx/valuations/simm/)

### Research Papers
- [Mathematical Foundations](../theory/mathematical_foundations.md)
- [Numerical Methods](../theory/numerical_methods.md)
- [Bibliography & References](../references.md)

---

## FAQ

### Q: How does JAX improve risk calculation performance?

**A:** JAX provides three key benefits:
1. **JIT Compilation**: 10-100x speedup through XLA optimization
2. **Automatic Differentiation**: Accurate Greeks without finite differences
3. **GPU/TPU Acceleration**: Parallel computation for Monte Carlo and scenario analysis

### Q: Can I use my own VaR methodology?

**A:** Yes, the risk engine supports custom VaR implementations:

```python
from neutryx.risk import RiskEngine

def my_custom_var(returns, confidence_level, **kwargs):
    # Your custom VaR logic
    return var_value

# Register custom method
risk_engine = RiskEngine()
risk_engine.register_var_method("custom", my_custom_var)

# Use custom method
var = risk_engine.calculate_var(portfolio, method="custom")
```

### Q: How do I integrate with my existing risk system?

**A:** Neutryx Core provides several integration options:
1. **Python API**: Direct integration via `neutryx.risk` module
2. **REST API**: HTTP endpoints for language-agnostic integration
3. **gRPC**: Low-latency binary protocol for real-time systems
4. **Batch Processing**: Scheduled jobs for overnight risk runs

See [Deployment Guide](../deployment.md) for details.

### Q: What confidence levels are supported for VaR?

**A:** Any confidence level between 0 and 1:
- **95%**: Common for internal risk management
- **99%**: Regulatory standard (Basel III)
- **97.5%**: FRTB IMA Expected Shortfall
- **99.9%**: Extreme tail risk

### Q: How do I handle missing market data?

**A:** The risk engine provides several options:
1. **Forward fill**: Use last available value
2. **Interpolation**: Linear or cubic interpolation
3. **Model-based**: Proxy using correlated instruments
4. **Exclusion**: Exclude positions with missing data

```python
risk_engine = RiskEngine(
    missing_data_policy="forward_fill",
    max_fill_days=5  # Maximum days to forward fill
)
```

### Q: Can I run risk calculations on GPU?

**A:** Yes, all risk calculations are JAX-native and automatically leverage GPU:

```python
import jax

# Set default device to GPU
jax.config.update('jax_default_device', jax.devices('gpu')[0])

# All subsequent calculations will use GPU
var = compute_var(returns, confidence_level=0.99)
```

---

## Support & Community

### Get Help
- **Documentation**: [https://neutryx-lab.github.io/neutryx-core](https://neutryx-lab.github.io/neutryx-core)
- **GitHub Issues**: [Report bugs and request features](https://github.com/neutryx-lab/neutryx-core/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/neutryx-lab/neutryx-core/discussions)
- **Email**: dev@neutryx.tech

### Contributing
We welcome contributions to the risk management framework! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

**Priority areas:**
- Additional VaR methodologies (filtered historical simulation, GARCH-based)
- More stress scenarios (recent market events)
- Machine learning-based risk models
- Integration with commercial risk systems

---

## Conclusion

Neutryx Core's Risk Management framework provides a **complete, production-ready** solution for measuring and managing market risk across multi-asset portfolios. With **10-100x performance improvements** through JAX, comprehensive regulatory compliance, and enterprise-grade features, it's the ideal foundation for modern risk management systems.

**Ready to get started?** Check out the [Risk Masterclass](tutorials/risk_masterclass.md) for hands-on tutorials and examples.

---

**Related Pages:**
- [Product Strategy](../products.md) - Target personas and value propositions
- [Valuations Hub](../valuations_index.md) - XVA and exposure calculations
- [Regulatory Compliance](../src/neutryx/regulatory/) - FRTB, SA-CCR, SIMM
- [Getting Started](../getting_started.md) - Installation and quick start
