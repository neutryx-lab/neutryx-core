# Neutryx Core Documentation

Welcome to the comprehensive documentation for **Neutryx Core** — the JAX-powered quantitative finance platform for derivatives pricing, risk management, and regulatory compliance.

## Quick Navigation

### 🚀 Getting Started

Start your journey with Neutryx Core:

- **[Getting Started Guide](getting_started.md)** - Installation, first examples, and quick wins
- **[Overview](overview.md)** - Vision, architecture, and core capabilities
- **[Tutorials](tutorials.md)** - Hands-on tutorials from beginner to advanced

### 📚 Core Documentation

#### Fundamentals
- **[Architecture Guide](architecture.md)** - System design, patterns, and components
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Products Guide](products.md)** - Multi-asset class derivatives

#### Operational Guides
- **[Performance Tuning](performance_tuning.md)** - Optimization strategies and best practices
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Configuration](configuration.md)** - System configuration reference
- **[Deployment](deployment.md)** - Production deployment guide

#### Infrastructure
- **[Market Data](market_data.md)** - Real-time data feeds, storage, and validation
- **[Monitoring & Observability](monitoring.md)** - Metrics, tracing, and alerting
- **[CI/CD Pipeline](ci_pipeline.md)** - Continuous integration and deployment

### 💼 Use Case Guides

#### Pricing & Valuation
- **[Valuations Index](valuations_index.md)** - XVA framework overview
- **[Valuations Quick Start](valuations_quickstart.md)** - Get started in 5 minutes
- **[Comprehensive Valuations Guide](valuations_comprehensive.md)** - Full XVA documentation
- **[Valuations API](valuations_api_summary.md)** - API reference

#### Risk Management
- **[Risk Analytics](risk/index.md)** - VaR, stress testing, and limits
- **[Risk Tutorials](risk/tutorials/index.md)** - Hands-on risk management
- **[Risk Reference](risk/reference/index.md)** - Comprehensive risk documentation
- **[Risk Controls](risk/controls/index.md)** - Position limits and pre-trade controls

#### Model Calibration
- **[Calibration Overview](calibration/index.md)** - Calibration framework
- **[Calibration Tutorials](calibration/tutorials/index.md)** - Step-by-step guides
- **[Calibration Reference](calibration/reference/index.md)** - Advanced techniques
- **[Model Selection](calibration/playbooks/index.md)** - Best practices

#### Models & Pricing
- **[Models Overview](models/index.md)** - Available models (BS, Heston, SABR)
- **[Model Tutorials](models/tutorials/index.md)** - Model implementation guides
- **[Model Reference](models/reference/index.md)** - Detailed model documentation

### 🔬 Advanced Topics

#### Theory & Mathematics
- **[Mathematical Foundations](theory/mathematical_foundations.md)** - Core mathematics
- **[Pricing Models](theory/pricing_models.md)** - Model theory and derivations
- **[Numerical Methods](theory/numerical_methods.md)** - PDE solvers, MC methods
- **[Calibration Theory](theory/calibration_theory.md)** - Parameter estimation theory

#### Integration & Extensions
- **[FpML Integration](fpml_integration.md)** - Financial product markup language
- **[Trade Management](trade_management.md)** - Trade lifecycle management
- **[Orchestration](orchestration.md)** - Workflow orchestration

### 👨‍💻 For Developers

- **[Developer Guide](developer_guide.md)** - Contributing, coding standards, testing
- **[Design Decisions](design_decisions.md)** - Architecture and design rationale
- **[Roadmap](roadmap.md)** - Development roadmap and milestones
- **[References](references.md)** - Academic papers and resources

### 📊 Project Information

#### Documentation
- **[Documentation Index](notebooks/index.md)** - Jupyter notebooks and examples
- **[Test Coverage](test_coverage.md)** - Code coverage reports
- **[Security Audit](security_audit.md)** - Security analysis and best practices

#### Project Management
- **[Changelog](project/CHANGELOG.md)** - Version history and changes
- **[Release Checklist](project/RELEASE_CHECKLIST.md)** - Release process
- **[Codebase Exploration](project/CODEBASE_EXPLORATION_REPORT.md)** - Code analysis

## Featured Documentation

### Valuations & XVA Framework
Comprehensive derivatives valuation, XVA calculations, risk management, and regulatory compliance:

- **XVA Components**: CVA, DVA, FVA, MVA, KVA
- **Risk Metrics**: VaR, Expected Shortfall (CVaR), Greeks
- **Exposure Analytics**: EE, PFE, EPE profiles
- **Initial Margin**: ISDA SIMM methodology
- **Stress Testing**: Scenario analysis and historical scenarios
- **Wrong-Way Risk**: Advanced correlation modeling
- **P&L Attribution**: Daily explain and factor attribution

**Quick Links**:
- [Valuations Quick Start](valuations_quickstart.md) - 5-minute setup
- [Comprehensive Guide](valuations_comprehensive.md) - Full documentation
- [API Reference](valuations_api_summary.md) - Complete API

### Risk Management Framework
Enterprise risk analytics with multiple VaR methodologies and position controls:

- **VaR Methodologies**: Historical, Monte Carlo, parametric VaR, ES/CVaR
- **Component Risk**: Incremental VaR, component VaR, marginal VaR
- **Position Limits**: Notional, VaR, concentration, issuer exposure limits
- **Pre-Trade Controls**: Real-time limit checking, what-if analysis, approval workflows
- **Stress Testing**: Historical scenarios, hypothetical scenarios, reverse stress testing

**Quick Links**:
- [Risk Overview](risk/index.md) - Framework introduction
- [Risk Controls Atlas](risk/controls/risk_controls_atlas.md) - Comprehensive guide
- [Risk Tutorials](risk/tutorials/risk_masterclass.md) - Hands-on learning

### Market Data Infrastructure
Production-grade real-time market data pipeline:

- **Vendor Integration**: Bloomberg Terminal/API, Refinitiv RDP/Eikon
- **Storage Solutions**: PostgreSQL, MongoDB, TimescaleDB with 90% compression
- **Data Quality**: Price validation, spread checks, volume spike detection, quality scoring
- **Feed Management**: Real-time orchestration, automatic failover, buffering

**Quick Links**:
- [Market Data Guide](market_data.md) - Complete documentation

### Observability & Monitoring
Enterprise observability stack for production deployments:

- **Metrics**: Prometheus with custom business metrics
- **Dashboards**: Pre-built Grafana dashboards (overview, performance, risk)
- **Tracing**: OpenTelemetry + Jaeger distributed tracing
- **Profiling**: Automatic performance profiling with cProfile
- **Alerting**: Intelligent alerts with configurable thresholds

**Quick Links**:
- [Monitoring Guide](monitoring.md) - Setup and configuration
- [Observability Guide](observability.md) - Best practices

## Learning Paths

### For Quantitative Analysts
1. [Getting Started](getting_started.md) - Setup and basics
2. [Tutorials](tutorials.md) - Pricing and risk examples
3. [Valuations Guide](valuations_comprehensive.md) - XVA framework
4. [Model Reference](models/reference/index.md) - Model documentation

### For Risk Managers
1. [Getting Started](getting_started.md) - Installation
2. [Risk Overview](risk/index.md) - Risk framework
3. [Risk Controls](risk/controls/risk_controls_atlas.md) - Limits and controls
4. [Risk Tutorials](risk/tutorials/risk_masterclass.md) - Practical examples

### For Developers
1. [Developer Guide](developer_guide.md) - Setup and standards
2. [Architecture Guide](architecture.md) - System design
3. [Performance Tuning](performance_tuning.md) - Optimization
4. [Troubleshooting](troubleshooting.md) - Common issues

### For System Administrators
1. [Deployment Guide](deployment.md) - Production setup
2. [Configuration](configuration.md) - System configuration
3. [Monitoring](monitoring.md) - Observability stack
4. [Troubleshooting](troubleshooting.md) - Operations guide

## Quick Start Example

Get pricing in 60 seconds:

```python
import jax.numpy as jnp
from neutryx.models.bs import price, greeks

# Price a European call option
call_price = price(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    risk_free=0.05,
    dividend=0.02,
    volatility=0.20,
    option_type="call"
)

# Calculate Greeks
delta, gamma, vega, theta, rho = greeks(
    100.0, 100.0, 1.0, 0.05, 0.02, 0.20, "call"
)

print(f"Call Price: ${call_price:.4f}")
print(f"Delta: {delta:.4f}")
```

**Next Steps**: [Getting Started Guide](getting_started.md)

## Support & Community

- **Documentation**: You're here!
- **GitHub Repository**: [neutryx-lab/neutryx-core](https://github.com/neutryx-lab/neutryx-core)
- **Issues**: [Report bugs or request features](https://github.com/neutryx-lab/neutryx-core/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/neutryx-lab/neutryx-core/discussions)
- **Website**: [neutryx.tech](https://neutryx.tech)

## License

Neutryx Core is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

**Built for Investment Banks, Hedge Funds, and Quantitative Researchers**

*Accelerating quantitative finance with differentiable computing and enterprise-grade infrastructure*
