# Neutryx Valuations: Quick Start Guide

## Installation

```bash
pip install neutryx-core
```

## 5-Minute Tutorial

### 1. Calculate CVA for a European Option

```python
import jax
import jax.numpy as jnp
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.exposure import epe
from neutryx.core.models import BlackScholes

# Setup
key = jax.random.PRNGKey(42)
T = 5.0
n_steps = 100
n_paths = 10000

# Market parameters
S0 = 100.0
K = 100.0
r = 0.03
sigma = 0.25

# Simulate paths
model = BlackScholes(r=r, sigma=sigma)
times = jnp.linspace(0, T, n_steps + 1)
dt = T / n_steps
paths = model.simulate(key, S0, T, n_steps, n_paths)

# Calculate EPE
epe_profile = jnp.array([
    epe(paths[:, :i+1], K, is_call=True)
    for i in range(n_steps)
])

# Discount factors
df_t = jnp.exp(-r * times[1:])

# Default probabilities (from 150bp credit spread)
credit_spread = 0.0150
hazard_rate = credit_spread / 0.60
pd_t = 1 - jnp.exp(-hazard_rate * times[1:])

# Calculate CVA
cva_value = cva(
    epe_t=epe_profile,
    df_t=df_t,
    pd_t=pd_t,
    lgd=0.60
)

print(f"CVA: ${cva_value:.2f}")
```

### 2. Calculate Portfolio VaR

```python
import jax.numpy as jnp
from neutryx.valuations.risk_metrics import portfolio_var, portfolio_cvar

# Historical returns (252 days, 3 assets)
returns = jnp.array([
    # ... your returns data
])

# Portfolio weights
weights = jnp.array([0.40, 0.35, 0.25])

# Calculate VaR
var_95 = portfolio_var(
    positions=weights,
    returns_scenarios=returns,
    confidence_level=0.95
)

# Calculate CVaR
cvar_95 = portfolio_cvar(
    positions=weights,
    returns_scenarios=returns,
    confidence_level=0.95
)

print(f"95% VaR: ${var_95:,.2f}")
print(f"95% CVaR: ${cvar_95:,.2f}")
```

### 3. Run Stress Tests

```python
from neutryx.valuations.stress_test import run_historical_stress_tests

def portfolio_value(spot, volatility, rates):
    # Your portfolio valuation logic
    return value

base_params = {
    "spot": 100.0,
    "volatility": 0.20,
    "rates": 0.05
}

results = run_historical_stress_tests(
    base_params=base_params,
    valuation_fn=portfolio_value,
    scenario_names=["financial_crisis_2008", "covid_crash_2020"]
)

for result in results:
    print(f"{result['scenario_name']}: {result['pnl_percent']:.2f}%")
```

### 4. Calculate SIMM Initial Margin

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
        risk_factor="USD_5Y",
        risk_type=RiskFactorType.INTEREST_RATE,
        sensitivity_type=SensitivityType.DELTA,
        amount=100000,
        currency="USD"
    ),
    RiskFactorSensitivity(
        risk_factor="SPX",
        risk_type=RiskFactorType.EQUITY,
        sensitivity_type=SensitivityType.DELTA,
        amount=200000,
        currency="USD"
    )
]

# Calculate SIMM
calculator = SIMMCalculator()
result = calculator.calculate(sensitivities)

print(f"Total IM: ${result.total_im:,.2f}")
```

### 5. Analyze Wrong-Way Risk

```python
from neutryx.valuations.wrong_way_risk import (
    cva_with_wwr,
    WWRParameters,
    WWRType
)

wwr_params = WWRParameters(
    correlation=0.40,  # Positive = wrong-way risk
    wwr_type=WWRType.GENERAL
)

cva_wwr, cva_base, _ = cva_with_wwr(
    key=jax.random.PRNGKey(42),
    exposure_paths=exposure_paths,
    df_t=discount_factors,
    hazard_rate=0.02,
    lgd=0.60,
    wwr_params=wwr_params,
    T=5.0
)

wwr_charge = cva_wwr - cva_base
print(f"WWR Charge: ${wwr_charge:.2f} ({wwr_charge/cva_base*100:.1f}%)")
```

## Next Steps

- Read the [Comprehensive Documentation](valuations_comprehensive.md)
- Check out [Examples Notebook](../examples/valuations_examples.ipynb)
- Explore the [API Reference](api_reference.md)

## Common Patterns

### Pattern 1: Full XVA Stack

```python
# Calculate all XVA components
cva_value = cva(epe_t, df_t, pd_counterparty, lgd=0.60)
dva_value = dva(ene_t, df_t, pd_own, lgd=0.40)
fva_value = fva(epe_t, funding_spread, df_t)

# Total XVA adjustment
total_xva = cva_value - dva_value + fva_value
adjusted_price = clean_price - total_xva
```

### Pattern 2: Risk Reporting

```python
from neutryx.valuations.risk_metrics import compute_all_risk_metrics

metrics = compute_all_risk_metrics(
    returns=portfolio_returns,
    confidence_levels=[0.95, 0.99],
    risk_free_rate=0.03
)

# Generate report
print(f"Mean Return: {metrics['mean']:.4f}")
print(f"Volatility: {metrics['std']:.4f}")
print(f"95% VaR: {metrics['var_95']:.4f}")
print(f"99% VaR: {metrics['var_99']:.4f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Pattern 3: P&L Attribution

```python
from neutryx.valuations.pnl_attribution import (
    PnLAttributionEngine,
    MarketState
)

engine = PnLAttributionEngine(
    portfolio_pricer=price_portfolio,
    greeks_calculator=calculate_greeks
)

attribution = engine.attribute_pnl(start_state, end_state)

print(f"Total P&L: ${attribution.total_pnl:,.2f}")
print(f"  Theta: ${attribution.theta_pnl:,.2f}")
print(f"  Spot: ${attribution.total_spot_pnl():,.2f}")
print(f"  Vol: ${attribution.total_vol_pnl():,.2f}")
```

## Tips and Tricks

### Tip 1: Use JIT for Performance

```python
import jax

@jax.jit
def calculate_cva_jit(epe, df, pd, lgd):
    return cva(epe, df, pd, lgd)

# First call compiles, subsequent calls are fast
result = calculate_cva_jit(epe_t, df_t, pd_t, 0.60)
```

### Tip 2: Vectorize Over Multiple Portfolios

```python
# Calculate VaR for multiple portfolios at once
vars = jax.vmap(
    lambda w: portfolio_var(w, returns, 0.95)
)(portfolio_weights)  # [n_portfolios, n_assets]
```

### Tip 3: Cache Market Data

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_discount_factors(maturity, rate):
    times = jnp.linspace(0, maturity, 100)
    return jnp.exp(-rate * times)
```

## Troubleshooting

### Issue: NaN in CVA calculation

**Solution**: Check for zero discount factors or probabilities

```python
# Add small epsilon
pd_t = jnp.maximum(pd_t, 1e-10)
df_t = jnp.maximum(df_t, 1e-10)
```

### Issue: VaR backtest failing

**Solution**: Increase sample size or adjust confidence level

```python
# Use larger historical window
var_95 = historical_var(returns, 0.95, window=500)
```

### Issue: Slow SIMM calculation

**Solution**: Use JAX JIT compilation

```python
@jax.jit
def calculate_simm_jit(sensitivities):
    return calculator.calculate(sensitivities)
```

## FAQ

**Q: How do I handle multi-currency portfolios?**

A: Use `MultiCurrencyCVA` with `MarketDataEnvironment`:

```python
from neutryx.valuations.xva.cva import MultiCurrencyCVA
from neutryx.market import MarketDataEnvironment

calculator = MultiCurrencyCVA(
    collateral_currency='USD',
    lgd=0.60
)

cva_value = calculator.calculate(
    market_env=env,
    exposures_by_currency={'USD': usd_epe, 'EUR': eur_epe},
    times=times,
    pd_t=pd_t
)
```

**Q: How do I calibrate VaR models?**

A: Use backtesting to validate:

```python
from neutryx.valuations.risk_metrics import backtest_var

results = backtest_var(realized, forecasts, 0.95)
if not results['pass_backtest']:
    print("Recalibrate VaR model")
```

**Q: Can I use custom stress scenarios?**

A: Yes, create custom `StressScenario` objects:

```python
from neutryx.valuations.stress_test import StressScenario

custom = StressScenario(
    name="Custom Crisis",
    description="...",
    shocks={"equity": -0.40, "volatility": 3.0}
)
```

## Resources

- [Full Documentation](valuations_comprehensive.md)
- [API Reference](api_reference.md)
- [GitHub Repository](https://github.com/neutryx/neutryx-core)
- [Examples](../examples/)

---

**Getting Help**: Open an issue on GitHub or join our community Discord.
