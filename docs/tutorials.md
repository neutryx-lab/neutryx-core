# Neutryx Core Tutorials

Hands-on tutorials for mastering Neutryx Core. These tutorials progress from beginner to advanced topics, covering pricing, risk management, calibration, and production deployment.

## Getting Started Tutorials

### Tutorial 1: Your First Option Price

**Level**: Beginner | **Time**: 15 minutes

Learn the basics of option pricing with Black-Scholes.

```python
import jax.numpy as jnp
from neutryx.models.bs import price, greeks

# Market setup
S = 100.0  # Spot price
K = 100.0  # Strike
T = 1.0    # Maturity (years)
r = 0.05   # Risk-free rate
q = 0.02   # Dividend yield
σ = 0.20   # Volatility

# Price call and put
call = price(S, K, T, r, q, σ, "call")
put = price(S, K, T, r, q, σ, "put")

print(f"Call: ${call:.4f}, Put: ${put:.4f}")

# Calculate Greeks
delta, gamma, vega, theta, rho = greeks(S, K, T, r, q, σ, "call")
print(f"Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}")
```

**Key Concepts**:
- Analytic pricing vs. numerical methods
- Understanding Greeks
- Put-call parity

**Exercises**:
1. Verify put-call parity holds
2. Calculate at-the-money (ATM) option value
3. Compare in-the-money (ITM) vs out-of-the-money (OTM) Greeks

---

### Tutorial 2: Monte Carlo Simulation

**Level**: Beginner | **Time**: 20 minutes

Master Monte Carlo simulation for option pricing.

```python
import jax
import jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value

# Setup
key = jax.random.PRNGKey(42)
S0, K, T = 100.0, 100.0, 1.0
r, q, σ = 0.05, 0.02, 0.20

# Configure simulation
config = MCConfig(
    steps=252,      # Daily steps
    paths=100_000,  # Number of simulations
    seed=42
)

# Simulate GBM paths
paths = simulate_gbm(key, S0, r - q, σ, T, config)

# European call payoff
ST = paths[:, -1]
payoff = jnp.maximum(ST - K, 0.0)
call_price = present_value(payoff, jnp.array(T), r)

print(f"MC Price: ${float(call_price):.4f}")

# Confidence interval
std_error = jnp.std(payoff * jnp.exp(-r * T)) / jnp.sqrt(config.paths)
ci_95 = 1.96 * std_error
print(f"95% CI: [{call_price - ci_95:.4f}, {call_price + ci_95:.4f}]")
```

**Key Concepts**:
- Geometric Brownian Motion (GBM)
- Path simulation
- Confidence intervals
- Variance estimation

**Exercises**:
1. Increase paths and observe convergence
2. Calculate convergence rate (should be √N)
3. Implement antithetic variates for variance reduction

---

### Tutorial 3: Path-Dependent Options

**Level**: Intermediate | **Time**: 30 minutes

Price exotic options that depend on the entire price path.

#### Asian Option (Arithmetic Average)

```python
import jax
import jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value

key = jax.random.PRNGKey(42)
S0, K, T = 100.0, 100.0, 1.0
r, q, σ = 0.05, 0.02, 0.20

config = MCConfig(steps=252, paths=100_000)
paths = simulate_gbm(key, S0, r - q, σ, T, config)

# Arithmetic average
avg_price = jnp.mean(paths, axis=1)
payoff = jnp.maximum(avg_price - K, 0.0)
asian_price = present_value(payoff, jnp.array(T), r)

print(f"Asian Call: ${float(asian_price):.4f}")
```

#### Barrier Option (Up-and-Out)

```python
# Up-and-out barrier call
barrier = 120.0
knocked_out = jnp.any(paths >= barrier, axis=1)
ST = paths[:, -1]
payoff = jnp.where(knocked_out, 0.0, jnp.maximum(ST - K, 0.0))
barrier_price = present_value(payoff, jnp.array(T), r)

print(f"Barrier Call: ${float(barrier_price):.4f}")
```

#### Lookback Option

```python
# Lookback call (floating strike)
min_price = jnp.min(paths, axis=1)
ST = paths[:, -1]
payoff = ST - min_price
lookback_price = present_value(payoff, jnp.array(T), r)

print(f"Lookback Call: ${float(lookback_price):.4f}")
```

**Exercises**:
1. Implement down-and-in barrier option
2. Calculate geometric average Asian option
3. Compare lookback option with different monitoring frequencies

---

## Risk Management Tutorials

### Tutorial 4: Value at Risk (VaR)

**Level**: Intermediate | **Time**: 30 minutes

Calculate portfolio VaR using multiple methodologies.

#### Historical Simulation VaR

```python
import jax.numpy as jnp
from neutryx.risk.var import historical_var, expected_shortfall

# Historical returns (daily)
returns = jnp.array([...])  # Load your returns data

# Calculate VaR at 95% and 99% confidence
var_95 = historical_var(returns, confidence=0.95)
var_99 = historical_var(returns, confidence=0.99)

print(f"VaR(95%): ${var_95:,.2f}")
print(f"VaR(99%): ${var_99:,.2f}")

# Expected Shortfall (CVaR)
es_95 = expected_shortfall(returns, confidence=0.95)
es_99 = expected_shortfall(returns, confidence=0.99)

print(f"ES(95%): ${es_95:,.2f}")
print(f"ES(99%): ${es_99:,.2f}")
```

#### Monte Carlo VaR

```python
from neutryx.risk.var import monte_carlo_var

# Portfolio positions
positions = jnp.array([1_000_000, 2_000_000, 1_500_000])

# Simulate returns
key = jax.random.PRNGKey(42)
mean_returns = jnp.array([0.0001, 0.0002, 0.00015])
cov_matrix = jnp.array([
    [0.0004, 0.0001, 0.0002],
    [0.0001, 0.0009, 0.0003],
    [0.0002, 0.0003, 0.0006]
])

var_mc = monte_carlo_var(
    positions, mean_returns, cov_matrix,
    confidence=0.95, num_simulations=10_000, key=key
)

print(f"Monte Carlo VaR(95%): ${var_mc:,.2f}")
```

#### Parametric VaR

```python
from neutryx.risk.var import parametric_var

# Calculate using variance-covariance method
portfolio_value = jnp.sum(positions)
weights = positions / portfolio_value

# Portfolio statistics
portfolio_return = jnp.dot(weights, mean_returns)
portfolio_vol = jnp.sqrt(jnp.dot(weights, jnp.dot(cov_matrix, weights)))

var_param = parametric_var(
    portfolio_value, portfolio_return, portfolio_vol,
    confidence=0.95
)

print(f"Parametric VaR(95%): ${var_param:,.2f}")
```

**Key Concepts**:
- Historical vs. parametric vs. Monte Carlo VaR
- Expected Shortfall (CVaR)
- Portfolio correlation and diversification
- VaR backtesting

**Exercises**:
1. Compare all three VaR methods
2. Calculate component VaR for each position
3. Implement VaR backtesting framework

---

### Tutorial 5: Position Limits and Pre-Trade Controls

**Level**: Intermediate | **Time**: 25 minutes

Implement risk limits and pre-trade checking.

```python
from neutryx.risk.limits import (
    PositionLimits, LimitChecker, LimitBreachLevel
)

# Define hierarchical limits
limits = PositionLimits(
    notional_limit=10_000_000,
    var_limit_hard=500_000,
    var_limit_soft=400_000,
    var_limit_warning=300_000,
    concentration_limit=0.25,
    issuer_exposure_limit=2_000_000
)

# Create limit checker
checker = LimitChecker(limits)

# Current portfolio state
current_state = {
    "total_notional": 8_000_000,
    "portfolio_var": 350_000,
    "positions": {
        "AAPL": {"notional": 2_000_000, "var": 100_000},
        "MSFT": {"notional": 1_500_000, "var": 80_000},
        "GOOGL": {"notional": 1_000_000, "var": 70_000}
    }
}

# Proposed trade
proposed_trade = {
    "ticker": "AAPL",
    "notional": 500_000,
    "var": 25_000,
    "type": "buy"
}

# Pre-trade check
result = checker.check_trade(proposed_trade, current_state)

if result.approved:
    print("✓ Trade approved")
else:
    print(f"✗ Trade rejected: {result.reason}")
    print(f"  Breach level: {result.breach_level}")
    print(f"  Limit utilization: {result.utilization:.1%}")

# What-if analysis
what_if_var = checker.calculate_what_if_var(
    proposed_trade, current_state
)
print(f"Post-trade VaR: ${what_if_var:,.2f}")
```

**Key Concepts**:
- Hierarchical limit structures
- Pre-trade impact analysis
- Limit breach severity levels
- Approval workflows

**Exercises**:
1. Implement desk-level limits
2. Add concentration limits by sector
3. Create limit breach alerting system

---

## Model Calibration Tutorials

### Tutorial 6: Heston Model Calibration

**Level**: Advanced | **Time**: 45 minutes

Calibrate the Heston stochastic volatility model to market data.

```python
import jax
import jax.numpy as jnp
from neutryx.calibration.heston import calibrate_heston
from neutryx.calibration.diagnostics import calibration_diagnostics
from neutryx.calibration.model_selection import select_model_aic

# Market data: implied volatility surface
strikes = jnp.array([90, 95, 100, 105, 110])
maturities = jnp.array([0.25, 0.5, 1.0, 2.0])
spot = 100.0

# Observed market implied vols (example data)
market_vols = jnp.array([
    [0.22, 0.20, 0.19, 0.21, 0.24],  # T=0.25
    [0.21, 0.19, 0.18, 0.20, 0.23],  # T=0.5
    [0.20, 0.18, 0.17, 0.19, 0.22],  # T=1.0
    [0.19, 0.17, 0.16, 0.18, 0.21],  # T=2.0
])

# Convert to prices for calibration
from neutryx.models.bs import price as bs_price
market_prices = jnp.array([
    [bs_price(spot, K, T, 0.05, 0.02, vol, "call")
     for K, vol in zip(strikes, vols)]
    for T, vols in zip(maturities, market_vols)
])

# Initial parameter guess
initial_params = {
    "kappa": 2.0,    # Mean reversion speed
    "theta": 0.04,   # Long-term variance
    "sigma": 0.3,    # Vol of vol
    "rho": -0.7,     # Correlation
    "v0": 0.04       # Initial variance
}

# Calibrate
calibrated_params, diagnostics = calibrate_heston(
    spot=spot,
    strikes=strikes,
    maturities=maturities,
    market_prices=market_prices,
    initial_params=initial_params,
    r=0.05,
    q=0.02
)

print("Calibrated Parameters:")
for param, value in calibrated_params.items():
    print(f"  {param}: {value:.6f}")

print(f"\nCalibration RMSE: {diagnostics['rmse']:.6f}")
print(f"Max absolute error: {diagnostics['max_error']:.6f}")
print(f"Iterations: {diagnostics['iterations']}")

# Model selection using AIC
aic = select_model_aic(
    calibrated_params, market_prices, num_params=5
)
print(f"AIC Score: {aic:.2f}")
```

**Key Concepts**:
- Implied volatility surface
- Characteristic function pricing
- Calibration objectives (RMSE, weighted least squares)
- Parameter constraints and stability
- Model selection criteria

**Exercises**:
1. Add Tikhonov regularization
2. Implement joint calibration with volatility smile
3. Compare Heston vs. SABR calibration quality

---

### Tutorial 7: Model Selection and Validation

**Level**: Advanced | **Time**: 40 minutes

Select the best model using statistical criteria and cross-validation.

```python
from neutryx.calibration.model_selection import (
    select_model_aic, select_model_bic, select_model_aicc,
    cross_validate_model, sensitivity_analysis
)

# Calibrate multiple models
models = {
    "Black-Scholes": {"params": {...}, "num_params": 1},
    "Heston": {"params": {...}, "num_params": 5},
    "SABR": {"params": {...}, "num_params": 4}
}

# Information criteria comparison
for model_name, model_data in models.items():
    aic = select_model_aic(model_data["params"], market_prices,
                           model_data["num_params"])
    bic = select_model_bic(model_data["params"], market_prices,
                           model_data["num_params"])
    aicc = select_model_aicc(model_data["params"], market_prices,
                             model_data["num_params"])

    print(f"{model_name}:")
    print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}, AICc: {aicc:.2f}")

# Cross-validation
cv_results = cross_validate_model(
    model="heston",
    data=market_prices,
    k_folds=5,
    metric="rmse"
)

print(f"\nCross-Validation RMSE: {cv_results['mean']:.6f} ± {cv_results['std']:.6f}")

# Sensitivity analysis
sensitivity = sensitivity_analysis(
    calibrated_params,
    market_prices,
    parameter="kappa",
    range_pct=0.20
)

print(f"\nParameter Sensitivity:")
for param, sens in sensitivity.items():
    print(f"  {param}: {sens:.4f}")
```

**Key Concepts**:
- Information criteria (AIC, BIC, AICc, HQIC)
- Cross-validation strategies
- Parameter sensitivity and identifiability
- Model comparison and selection

**Exercises**:
1. Implement time-series cross-validation
2. Calculate global sensitivity using Sobol indices
3. Test model stability under different market regimes

---

## Advanced Topics

### Tutorial 8: XVA Calculations

**Level**: Advanced | **Time**: 60 minutes

Calculate Credit Valuation Adjustment (CVA) and other XVA metrics.

```python
from neutryx.valuations.cva import calculate_cva, calculate_dva
from neutryx.valuations.exposure import exposure_profile

# Simulate exposure over time
key = jax.random.PRNGKey(42)
num_paths = 10_000
time_grid = jnp.linspace(0, 5, 21)  # Quarterly for 5 years

# Simulate underlying (swap rates, equity prices, etc.)
underlying_paths = simulate_gbm(key, S0=100, mu=0.05, sigma=0.20,
                                T=5.0, steps=20, paths=num_paths)

# Calculate exposure for interest rate swap (example)
notional = 10_000_000
fixed_rate = 0.05
discount_curve = jnp.array([...])  # Term structure

# Expected Exposure (EE)
ee = exposure_profile(underlying_paths, notional, time_grid)

# Potential Future Exposure (PFE) at 95%
pfe_95 = jnp.percentile(ee, 95, axis=0)

# Credit spreads and survival probabilities
counterparty_spreads = jnp.array([0.01, 0.012, 0.014, ...])
survival_probs = jnp.exp(-jnp.cumsum(counterparty_spreads * time_grid))

# Calculate CVA
recovery_rate = 0.40
cva = calculate_cva(
    exposure=ee,
    survival_probs=survival_probs,
    recovery_rate=recovery_rate,
    discount_factors=discount_curve
)

print(f"CVA: ${cva:,.2f}")
print(f"CVA as % of notional: {(cva/notional)*100:.4f}%")

# Calculate DVA (own credit risk)
own_spreads = jnp.array([0.005, 0.006, 0.007, ...])
dva = calculate_dva(
    exposure=-ee,  # Negative exposure from counterparty perspective
    survival_probs=jnp.exp(-jnp.cumsum(own_spreads * time_grid)),
    recovery_rate=recovery_rate,
    discount_factors=discount_curve
)

print(f"DVA: ${dva:,.2f}")
print(f"Net CVA: ${cva - dva:,.2f}")
```

**Key Concepts**:
- Expected Exposure (EE) and Potential Future Exposure (PFE)
- CVA, DVA, FVA calculations
- Wrong-way risk
- Collateral modeling

**Exercises**:
1. Implement FVA (Funding Valuation Adjustment)
2. Model collateral with CSA terms
3. Calculate MVA (Margin Valuation Adjustment)

---

### Tutorial 9: Real-Time Market Data Pipeline

**Level**: Advanced | **Time**: 50 minutes

Build a production-grade market data pipeline.

```python
import asyncio
from neutryx.market.adapters import BloombergAdapter, BloombergConfig
from neutryx.market.storage import TimescaleDBStorage, TimescaleDBConfig
from neutryx.market.validation import (
    ValidationPipeline, PriceRangeValidator,
    SpreadValidator, VolumeValidator
)
from neutryx.market.feeds import FeedManager

# Configure Bloomberg adapter
bloomberg_config = BloombergConfig(
    adapter_name="bloomberg",
    host="localhost",
    port=8194,
    auth_token="your_token"
)

# Configure TimescaleDB with compression
storage_config = TimescaleDBConfig(
    host="localhost",
    port=5432,
    database="market_data",
    user="trader",
    password="secure_password",
    compression_enabled=True,
    compression_age_days=7,
    retention_policy_days=90
)

# Setup validation pipeline
validation = ValidationPipeline()
validation.add_validator(
    PriceRangeValidator(max_jump_pct=0.20)
)
validation.add_validator(
    SpreadValidator(max_bid_ask_spread_pct=0.05)
)
validation.add_validator(
    VolumeValidator(min_volume=1000, spike_threshold=10.0)
)

# Create feed manager
async def run_market_data_pipeline():
    adapter = BloombergAdapter(bloomberg_config)
    storage = TimescaleDBStorage(storage_config)

    manager = FeedManager(
        adapters=[adapter],
        storage=storage,
        validation_pipeline=validation,
        buffer_size=1000,
        flush_interval=1.0
    )

    # Start feed
    await manager.start()

    # Subscribe to instruments
    await manager.subscribe("equity", ["AAPL", "MSFT", "GOOGL", "AMZN"])
    await manager.subscribe("fx", ["EURUSD", "GBPUSD", "USDJPY"])

    # Monitor feed
    while True:
        stats = await manager.get_statistics()
        print(f"Messages received: {stats['total_messages']}")
        print(f"Quality score: {stats['average_quality']:.2f}")
        print(f"Failed validations: {stats['failed_validations']}")

        await asyncio.sleep(10)

# Run pipeline
asyncio.run(run_market_data_pipeline())
```

**Key Concepts**:
- Real-time data ingestion
- Data validation and quality scoring
- TimescaleDB hypertables and compression
- Automatic failover and buffering

**Exercises**:
1. Add Refinitiv as secondary data source
2. Implement automatic failover between vendors
3. Create data quality dashboard

---

### Tutorial 10: Production Deployment with Monitoring

**Level**: Advanced | **Time**: 60 minutes

Deploy Neutryx with full observability stack.

```python
# Setup Prometheus metrics
from neutryx.infrastructure.observability.metrics import (
    MetricsCollector, register_pricing_metrics
)

metrics = MetricsCollector()
register_pricing_metrics(metrics)

# Instrument pricing function
from functools import wraps
import time

def track_pricing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            metrics.pricing_duration.observe(time.time() - start)
            metrics.pricing_success.inc()
            return result
        except Exception as e:
            metrics.pricing_errors.inc()
            raise
    return wrapper

@track_pricing
def price_option(S, K, T, r, sigma):
    from neutryx.models.bs import price
    return price(S, K, T, r, 0.0, sigma, "call")

# Setup distributed tracing
from neutryx.infrastructure.observability.tracing import (
    init_tracer, trace_operation
)

tracer = init_tracer("neutryx-core", "jaeger", "localhost:6831")

@trace_operation(tracer, "price_portfolio")
def price_portfolio(portfolio):
    with tracer.start_span("load_market_data"):
        market_data = load_market_data()

    with tracer.start_span("calibrate_models"):
        models = calibrate_models(market_data)

    with tracer.start_span("calculate_prices"):
        prices = [price_instrument(inst, models) for inst in portfolio]

    return prices

# Setup alerting
from neutryx.infrastructure.observability.alerting import AlertManager

alerts = AlertManager()

alerts.add_rule(
    name="high_pricing_latency",
    condition="pricing_duration > 1.0",
    severity="warning",
    notification_channels=["email", "slack"]
)

alerts.add_rule(
    name="pricing_error_rate",
    condition="pricing_errors / pricing_total > 0.05",
    severity="critical",
    notification_channels=["pagerduty", "email"]
)

# Start metrics server
from prometheus_client import start_http_server
start_http_server(8000)

print("Metrics available at http://localhost:8000/metrics")
print("Jaeger UI at http://localhost:16686")
```

**Key Concepts**:
- Prometheus metrics and instrumentation
- Distributed tracing with Jaeger
- Grafana dashboards
- Alerting and notification

**Exercises**:
1. Create custom Grafana dashboard
2. Implement SLA monitoring
3. Setup multi-region deployment

---

## Interactive Examples

All tutorials are available as Jupyter notebooks in the `examples/tutorials/` directory:

- `01_vanilla_pricing/` - Basic option pricing
- `02_asian_scenario/` - Path-dependent options
- `03_counterparty_cva/` - XVA calculations
- `04_heston_calibration/` (coming soon)
- `05_production_deployment/` (coming soon)

## Additional Resources

- **[Getting Started Guide](getting_started.md)** - Basic setup and first steps
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Model Reference](models/reference/index.md)** - Detailed model documentation
- **[Risk Controls Atlas](risk/controls/risk_controls_atlas.md)** - Risk management guide
- **[Performance Tuning](performance_tuning.md)** - Optimization techniques

## Next Steps

After completing these tutorials, you'll be ready to:

1. Build production pricing systems
2. Implement risk management frameworks
3. Calibrate models to live market data
4. Deploy scalable quantitative finance infrastructure

Continue learning with our advanced documentation and examples!
