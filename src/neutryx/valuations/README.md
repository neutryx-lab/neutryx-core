# Neutryx Valuations Module

Enterprise-grade derivatives valuation, risk management, and regulatory calculations powered by JAX.

## Overview

The Neutryx Valuations module provides comprehensive tools for:

- **XVA Calculations**: CVA, DVA, FVA, MVA, KVA
- **Risk Metrics**: VaR, CVaR, Greeks, risk-adjusted performance
- **Regulatory Compliance**: SIMM, SA-CCR, FRTB
- **Scenario Analysis**: Stress testing, what-if analysis
- **P&L Attribution**: Factor-based P&L decomposition

## Quick Start

```python
# CVA Calculation
from neutryx.valuations.xva.cva import cva

cva_value = cva(
    epe_t=exposure_profile,
    df_t=discount_factors,
    pd_t=default_probs,
    lgd=0.60
)

# Portfolio VaR
from neutryx.valuations.risk_metrics import portfolio_var

var_95 = portfolio_var(
    positions=weights,
    returns_scenarios=returns,
    confidence_level=0.95
)

# SIMM Initial Margin
from neutryx.valuations.simm import SIMMCalculator

calculator = SIMMCalculator()
result = calculator.calculate(sensitivities)
print(f"Total IM: ${result.total_im:,.2f}")

# Stress Testing
from neutryx.valuations.stress_test import run_historical_stress_tests

results = run_historical_stress_tests(
    base_params=market_params,
    valuation_fn=portfolio_valuation
)
```

## Key Features

### XVA Framework

- **CVA/DVA**: Bilateral credit valuation adjustments
- **Multi-Currency**: Native FX exposure handling
- **FVA**: Funding valuation adjustments
- **MVA**: Margin valuation adjustments
- **KVA**: Capital valuation adjustments
- **Wrong-Way Risk**: Correlation between exposure and credit

### Risk Management

- **Multiple VaR Methods**: Historical, Parametric, Monte Carlo, Cornish-Fisher
- **CVaR / Expected Shortfall**: Tail risk measures
- **Component VaR**: Risk decomposition
- **VaR Backtesting**: Model validation
- **Greeks**: Delta, Gamma, Vega, Theta, Rho, and higher-order

### Regulatory Compliance

- **ISDA SIMM**: Standard Initial Margin Model
- **SA-CCR**: Standardized Approach for Counterparty Credit Risk
- **FRTB**: Fundamental Review of the Trading Book
- **Initial/Variation Margin**: Complete margin workflow

### Analysis Tools

- **Scenario Analysis**: Market shock modeling
- **Stress Testing**: Historical and hypothetical scenarios
- **P&L Attribution**: Factor-based decomposition
- **Reverse Stress Tests**: Find break-even scenarios

## Architecture

```
neutryx.valuations/
├── xva/                    # XVA framework
├── greeks/                 # Greeks calculations
├── risk_metrics.py         # VaR, CVaR, risk measures
├── simm/                   # ISDA SIMM
├── margin/                 # Margin calculations
├── scenarios/              # Scenario analysis
├── stress_test.py          # Stress testing
├── wrong_way_risk.py       # WWR modeling
├── pnl_attribution.py      # P&L attribution
└── exposure.py             # EPE/ENE calculations
```

## Documentation

- **[Quick Start Guide](../../../docs/valuations_quickstart.md)**: Get started in 5 minutes
- **[Comprehensive Guide](../../../docs/valuations_comprehensive.md)**: Complete documentation
- **[API Reference](../../../docs/valuations_api_summary.md)**: Full API listing
- **[Documentation Index](../../../docs/valuations_index.md)**: Navigate all docs

## Examples

### Complete XVA Calculation

```python
import jax
import jax.numpy as jnp
from neutryx.valuations.xva import ExposureSimulator, XVAScenario
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.xva.fva import fva

# Setup
key = jax.random.PRNGKey(42)
scenario = XVAScenario(spot=100, volatility=0.25, interest_rate=0.03)

# Simulate exposures
simulator = ExposureSimulator(n_paths=10000, n_steps=100, horizon=5.0)
exposure_result = simulator.simulate(key, scenario, option)

# Calculate CVA
cva_value = cva(
    epe_t=exposure_result.epe,
    df_t=discount_factors,
    pd_t=default_probs,
    lgd=0.60
)

# Calculate FVA
fva_value = fva(
    epe_t=exposure_result.epe,
    funding_spread=0.0050,
    df_t=discount_factors
)

# Total XVA
total_xva = cva_value + fva_value
```

### Risk Analysis

```python
from neutryx.valuations.risk_metrics import (
    compute_all_risk_metrics,
    component_var,
    backtest_var
)

# Comprehensive risk metrics
metrics = compute_all_risk_metrics(
    returns=portfolio_returns,
    confidence_levels=[0.95, 0.99],
    risk_free_rate=0.03
)

# Component VaR
comp_vars = component_var(weights, returns, 0.95)

# Backtest VaR model
backtest_results = backtest_var(realized, forecasts, 0.95)
```

### Stress Testing

```python
from neutryx.valuations.stress_test import (
    run_historical_stress_tests,
    factor_stress_test,
    reverse_stress_test
)

# Historical scenarios
historical_results = run_historical_stress_tests(
    base_params=market_params,
    valuation_fn=portfolio_value
)

# Factor stress test
shock_range = jnp.linspace(-0.50, 0.50, 101)
pnl_profile = factor_stress_test(
    base_params=market_params,
    valuation_fn=portfolio_value,
    factor="spot",
    shock_range=shock_range
)

# Reverse stress test
shock_for_1m_loss = reverse_stress_test(
    base_params=market_params,
    valuation_fn=portfolio_value,
    factor="spot",
    target_loss=-1_000_000
)
```

## Performance

Built on JAX for high-performance computing:

```python
import jax

# JIT compilation for speed
@jax.jit
def calculate_portfolio_var(returns, weights, confidence):
    portfolio_returns = returns @ weights
    return jnp.quantile(-portfolio_returns, confidence)

# Vectorize over multiple portfolios
vars = jax.vmap(
    lambda w: portfolio_var(w, returns, 0.95)
)(portfolio_weights)
```

## Testing

Comprehensive test coverage:

```bash
# Run all valuations tests
pytest src/neutryx/tests/valuations/

# Run specific module tests
pytest src/neutryx/tests/valuations/test_risk_metrics.py
pytest src/neutryx/tests/valuations/test_xva.py
pytest src/neutryx/tests/valuations/test_simm.py
```

## Requirements

- Python 3.9+
- JAX 0.4.0+
- NumPy
- SciPy

## Installation

```bash
pip install neutryx-core
```

## Use Cases

### Trading Desks
- Daily P&L attribution
- Greeks calculation
- Scenario analysis

### Risk Management
- Portfolio VaR monitoring
- Stress testing
- Risk limit validation

### Regulatory Reporting
- SIMM initial margin
- SA-CCR exposure
- FRTB capital

### Credit Risk
- CVA calculation
- Wrong-way risk analysis
- Exposure profiles

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

## Support

- **Documentation**: [Full Docs](../../../docs/valuations_index.md)
- **Issues**: [GitHub Issues](https://github.com/neutryx-lab/neutryx-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neutryx-lab/neutryx-core/discussions)

## License

Copyright © 2024 Neutryx. All rights reserved.

## References

- Basel Committee on Banking Supervision: Basel III Framework
- ISDA: SIMM Methodology
- BIS: FRTB Standardized Approach
- Gregory, Jon: "The xVA Challenge"
- Hull, John: "Options, Futures, and Other Derivatives"

---

**Version**: 1.0.0
**Last Updated**: 2024-11-04
**Module**: neutryx.valuations
