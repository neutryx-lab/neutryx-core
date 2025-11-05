# Neutryx Valuations Module Documentation

Complete documentation for the Neutryx Valuations framework - a comprehensive suite for derivatives valuation, risk management, and regulatory calculations.

## Documentation Structure

### 📚 Getting Started

1. **[Quick Start Guide](valuations_quickstart.md)**
   - 5-minute tutorial
   - Common patterns
   - Tips and tricks
   - Troubleshooting
   - FAQ

2. **[Comprehensive Documentation](valuations_comprehensive.md)**
   - Complete module overview
   - Detailed API descriptions
   - Theory and mathematics
   - Extended examples
   - Best practices

3. **[API Summary](valuations_api_summary.md)**
   - Quick API reference
   - Function signatures
   - Parameter descriptions
   - Return types

### 📖 Core Topics

#### XVA Framework
- [XVA Overview](valuations_comprehensive.md#xva-framework)
- [CVA Calculation](valuations_comprehensive.md#1-credit-valuation-adjustment-cva)
- [FVA, MVA, KVA](valuations_comprehensive.md#3-funding-valuation-adjustment-fva)
- [Exposure Calculations](valuations_comprehensive.md#exposure-calculations)

#### Risk Management
- [Risk Metrics Overview](valuations_comprehensive.md#risk-metrics)
- [Value at Risk (VaR)](valuations_comprehensive.md#value-at-risk-var)
- [VaR Methodologies](valuations_comprehensive.md#var-methodologies)
- [Conditional VaR / Expected Shortfall](valuations_comprehensive.md#conditional-var-cvar-expected-shortfall)
- [Portfolio Risk](valuations_comprehensive.md#portfolio-risk-metrics)
- [VaR Backtesting](valuations_comprehensive.md#var-backtesting)

#### Greeks and Sensitivities
- [Greeks Overview](valuations_comprehensive.md#greeks-and-sensitivities)
- [First-Order Greeks](valuations_comprehensive.md#first-order-greeks)
- [Second-Order Greeks](valuations_comprehensive.md#second-order-greeks)
- [Implementation](valuations_comprehensive.md#implementation)

#### SIMM and Margin
- [SIMM Overview](valuations_comprehensive.md#isda-simm-standard-initial-margin-model)
- [SIMM Workflow](valuations_comprehensive.md#workflow)
- [SIMM Calculation](valuations_comprehensive.md#workflow)
- [Initial Margin](valuations_comprehensive.md#initial-margin-non-simm)
- [Variation Margin](valuations_comprehensive.md#variation-margin)

#### Scenario Analysis
- [Scenario Framework](valuations_comprehensive.md#scenario-analysis)
- [Market Data Bumpers](valuations_comprehensive.md#market-data-bumpers)
- [Scenario Types](valuations_comprehensive.md#market-scenario-types)

#### Stress Testing
- [Stress Testing Overview](valuations_comprehensive.md#stress-testing)
- [Historical Scenarios](valuations_comprehensive.md#historical-stress-scenarios)
- [Factor Stress Tests](valuations_comprehensive.md#factor-stress-testing)
- [Reverse Stress Testing](valuations_comprehensive.md#reverse-stress-testing)

#### Wrong-Way Risk
- [WWR Overview](valuations_comprehensive.md#wrong-way-risk)
- [CVA with WWR](valuations_comprehensive.md#cva-with-wrong-way-risk)
- [Copula Models](valuations_comprehensive.md#gaussian-copula-wwr)
- [WWR Engine](valuations_comprehensive.md#comprehensive-wwr-engine)

#### P&L Attribution
- [P&L Attribution Overview](valuations_comprehensive.md#pl-attribution)
- [Attribution Methods](valuations_comprehensive.md#attribution-methods)
- [Daily Tracking](valuations_comprehensive.md#daily-pl-tracking)
- [Driver Analysis](valuations_comprehensive.md#analyzing-pl-drivers)

### 🔧 Practical Guides

#### Examples
- [Example 1: Complete XVA Calculation](valuations_comprehensive.md#example-1-complete-xva-calculation)
- [Example 2: Portfolio Risk Analysis](valuations_comprehensive.md#example-2-portfolio-risk-analysis)
- [Example 3: Stress Testing Workflow](valuations_comprehensive.md#example-3-stress-testing-workflow)
- [Example 4: SIMM Calculation](valuations_comprehensive.md#example-4-simm-calculation)

#### Best Practices
- [Performance Optimization](valuations_comprehensive.md#1-performance-optimization)
- [Numerical Stability](valuations_comprehensive.md#2-numerical-stability)
- [Model Validation](valuations_comprehensive.md#3-model-validation)
- [Risk Management](valuations_comprehensive.md#4-risk-management)
- [Testing](valuations_comprehensive.md#6-testing)

### 📋 Reference

#### API Documentation
- [XVA Module](valuations_api_summary.md#xva-module)
- [Risk Metrics Module](valuations_api_summary.md#risk-metrics-module)
- [SIMM Module](valuations_api_summary.md#simm-module)
- [Margin Module](valuations_api_summary.md#margin-module)
- [Scenarios Module](valuations_api_summary.md#scenarios-module)
- [Stress Test Module](valuations_api_summary.md#stress-test-module)
- [Wrong-Way Risk Module](valuations_api_summary.md#wrong-way-risk-module)
- [P&L Attribution Module](valuations_api_summary.md#pl-attribution-module)
- [Greeks Module](valuations_api_summary.md#greeks-module)

#### Module Structure
```
neutryx.valuations/
├── xva/                    # XVA framework (CVA, DVA, FVA, MVA, KVA)
│   ├── cva.py
│   ├── fva.py
│   ├── mva.py
│   ├── kva.py
│   ├── collateral.py
│   ├── exposure.py
│   ├── aggregation.py
│   └── capital.py
├── greeks/                 # Greeks calculations
│   ├── greeks.py
│   └── advanced_greeks.py
├── risk_metrics.py         # VaR, CVaR, risk measures
├── simm/                   # ISDA SIMM implementation
│   ├── calculator.py
│   ├── sensitivities.py
│   └── risk_weights.py
├── margin/                 # Margin calculations
│   ├── initial_margin.py
│   └── variation_margin.py
├── scenarios/              # Scenario analysis
│   ├── scenario.py
│   ├── bumpers.py
│   └── scenario_engine.py
├── stress_test.py          # Stress testing
├── wrong_way_risk.py       # Wrong-way risk modeling
├── pnl_attribution.py      # P&L attribution
├── exposure.py             # EPE/ENE calculations
└── utils.py                # Utility functions
```

## Quick Navigation

### By Use Case

#### Derivatives Pricing
- [Clean Price Calculation](valuations_comprehensive.md#overview)
- [XVA Adjustments](valuations_comprehensive.md#xva-framework)
- [XVA Framework](valuations_comprehensive.md#xva-framework)

#### Risk Management
- [Portfolio VaR](valuations_quickstart.md#2-calculate-portfolio-var)
- [Stress Testing](valuations_quickstart.md#3-run-stress-tests)
- [Risk Management Best Practices](valuations_comprehensive.md#4-risk-management)

#### Regulatory Compliance
- [SIMM Initial Margin](valuations_quickstart.md#4-calculate-simm-initial-margin)
- [SA-CCR](valuations_comprehensive.md#xva-framework)
- [FRTB](valuations_comprehensive.md#risk-metrics)

#### Trading Desk Operations
- [Daily P&L Attribution](valuations_comprehensive.md#daily-pl-tracking)
- [Greeks Calculation](valuations_comprehensive.md#greeks-and-sensitivities)
- [What-If Analysis](valuations_comprehensive.md#scenario-analysis)

#### Credit Risk
- [CVA Calculation](valuations_quickstart.md#1-calculate-cva-for-a-european-option)
- [Wrong-Way Risk](valuations_quickstart.md#5-analyze-wrong-way-risk)
- [Exposure Profiles](valuations_comprehensive.md#exposure-calculations)

### By Asset Class

#### Rates
- Interest Rate Swaps XVA
- Swaption Greeks
- SIMM Interest Rate Risk

#### Equity
- Equity Option CVA
- Equity Greeks
- SIMM Equity Risk

#### FX
- FX Option Pricing
- Multi-Currency CVA
- SIMM FX Risk

#### Credit
- CDS CVA
- Wrong-Way Risk
- SIMM Credit Risk

## Learning Path

### Beginner
1. Start with [Quick Start Guide](valuations_quickstart.md)
2. Work through the 5-minute tutorial
3. Try the [common patterns](valuations_quickstart.md#common-patterns)
4. Read [CVA basics](valuations_comprehensive.md#1-credit-valuation-adjustment-cva)

### Intermediate
1. Read [Comprehensive Documentation](valuations_comprehensive.md)
2. Study [XVA Framework](valuations_comprehensive.md#xva-framework)
3. Learn [Risk Metrics](valuations_comprehensive.md#risk-metrics)
4. Explore [SIMM](valuations_comprehensive.md#isda-simm-standard-initial-margin-model)
5. Practice with [examples](valuations_comprehensive.md#usage-examples)

### Advanced
1. Deep dive into [Wrong-Way Risk](valuations_comprehensive.md#wrong-way-risk)
2. Master [P&L Attribution](valuations_comprehensive.md#pl-attribution)
3. Study [Best Practices](valuations_comprehensive.md#best-practices)
4. Implement custom scenarios
5. Optimize performance with JAX

## Additional Resources

### Code Examples
- [GitHub Repository](https://github.com/neutryx-lab/neutryx-core)
- [Examples Directory](https://github.com/neutryx-lab/neutryx-core/tree/main/examples)
- [Jupyter Notebooks](https://github.com/neutryx-lab/neutryx-core/tree/main/examples/notebooks)

### Theory and Background
- Basel III Framework
- ISDA SIMM Methodology
- FRTB Standardized Approach
- CVA Capital Requirements

### Community
- [GitHub Issues](https://github.com/neutryx/neutryx-core/issues)
- [Discussions](https://github.com/neutryx/neutryx-core/discussions)
- Discord Community

### Related Documentation
- [Core Models](models/index.md)
- [Products](products.md)
- [Market Data](https://github.com/neutryx-lab/neutryx-core/tree/main/src/neutryx/market)
- [Calibration](calibration/index.md)

## Quick Reference Cards

### CVA Calculation
```python
from neutryx.valuations.xva.cva import cva

cva_value = cva(epe_t, df_t, pd_t, lgd=0.60)
```

### Portfolio VaR
```python
from neutryx.valuations.risk_metrics import portfolio_var

var_95 = portfolio_var(weights, returns, 0.95)
```

### SIMM IM
```python
from neutryx.valuations.simm import SIMMCalculator

calculator = SIMMCalculator()
result = calculator.calculate(sensitivities)
```

### Stress Test
```python
from neutryx.valuations.stress_test import run_historical_stress_tests

results = run_historical_stress_tests(base_params, valuation_fn)
```

## Frequently Asked Questions

### General
- [How do I handle multi-currency portfolios?](valuations_quickstart.md#faq)
- [How do I calibrate VaR models?](valuations_quickstart.md#faq)
- [Can I use custom stress scenarios?](valuations_quickstart.md#faq)

### Performance
- [How do I optimize for GPU?](valuations_comprehensive.md#1-performance-optimization)
- [What about memory usage?](valuations_comprehensive.md#1-performance-optimization)
- [Can I parallelize calculations?](valuations_comprehensive.md#1-performance-optimization)

### Integration
- How do I integrate with existing systems?
- What data formats are supported?
- Can I export results to Excel/CSV?

### Troubleshooting
- [NaN in CVA calculation](valuations_quickstart.md#issue-nan-in-cva-calculation)
- [VaR backtest failing](valuations_quickstart.md#issue-var-backtest-failing)
- [Slow SIMM calculation](valuations_quickstart.md#issue-slow-simm-calculation)

## Version History

- **v1.0.0** (2024-11-04): Initial comprehensive documentation
  - Complete XVA framework
  - Full risk metrics suite
  - SIMM implementation
  - Scenario analysis and stress testing
  - Wrong-way risk modeling
  - P&L attribution

## Contributing

We welcome contributions! Please see:
- [Contributing Guidelines](https://github.com/neutryx-lab/neutryx-core/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/neutryx-lab/neutryx-core/blob/main/CODE_OF_CONDUCT.md)
- [Development Setup](https://github.com/neutryx-lab/neutryx-core#development)

## Support

- **Documentation Issues**: Open an issue on GitHub
- **Bug Reports**: Use the issue tracker
- **Feature Requests**: Submit via discussions
- **Security Issues**: Email security@neutryx.com

---

**Last Updated**: 2024-11-04
**Module Version**: 1.0.0
**Python**: 3.9+
**JAX**: 0.4.0+

[Return to Main Documentation](index.md)
