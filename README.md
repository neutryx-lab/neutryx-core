# 🚀 Neutryx Core — The JAX-Driven Frontier of Quantitative Finance

> **Lightning-fast. Differentiable. Reproducible.**
> Neutryx Core fuses modern computational science with financial engineering — bringing together
> **JAX**, **automatic differentiation**, and **high-performance computing** to redefine how we price derivatives,
> measure risk, and model complex markets.

<p align="center">
  <img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge"/>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge"/>
  <img alt="JAX Version" src="https://img.shields.io/badge/jax-0.4.26+-orange?style=for-the-badge"/>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge"/>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue?style=for-the-badge"/>
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-features">Features</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🌌 Why Neutryx?

Neutryx Core is a **next-generation JAX-first quantitative finance library** —
designed for **researchers**, **banks**, and **AI-for-science teams** building real-time pricing, calibration,
and risk engines at HPC scale.

It unifies stochastic models, PDE solvers, and calibration workflows into a single, differentiable framework.
Every component — from yield curves to Greeks — is **JIT-compiled**, **vectorised**, and **autodiff-compatible**.

> *From PDEs to Monte Carlo, from calibration to XVA — all within one continuous computational graph.*

---

## ✨ Features

### Core Capabilities

- **Models:** Analytic Black-Scholes, stochastic volatility (Heston, SABR), jump diffusion, rough volatility
- **Products:** Comprehensive multi-asset class coverage including vanilla, exotic, and structured products
  - **Derivatives:** European, Asian, Barrier, Lookback, American (Longstaff-Schwartz)
  - **Equity:** Forwards, dividend swaps, variance swaps, TRS, equity-linked notes
  - **Commodities:** Forwards with convenience yield, options, swaps, spread options
  - **Fixed Income:** Bonds (zero-coupon, coupon, FRN), inflation-linked securities, swaptions
  - **Credit:** CDS pricing, hazard models, structural models
  - **Volatility:** VIX futures/options, variance swaps, corridor swaps, gamma swaps
  - **Convertibles:** Convertible bonds, mandatory convertibles, exchangeable bonds
- **Risk:** Pathwise & bump Greeks, stress testing, and adjoint-based sensitivity analysis
- **Market:** Multi-curve framework with OIS discounting, tenor basis, FX volatility surfaces
- **XVA:** Exposure models (CVA, FVA, MVA) for counterparty risk prototyping
- **Calibration:** Differentiable calibration framework with diagnostics and identifiability checks

### Technical Highlights

- **JAX-Native:** Full JIT compilation, automatic differentiation, and XLA optimization
- **GPU/TPU Ready:** Seamless acceleration on modern hardware with `pmap`/`pjit`
- **High Performance:** Optimized numerical algorithms with 10-100x speedup for repeated calculations
- **Reproducible:** Unified configuration via YAML, consistent PRNG seeding
- **Production-Ready:** FastAPI/gRPC APIs, comprehensive test suite (100+ tests), quality tooling (ruff, bandit)
- **Extensible:** FFI bridges to QuantLib/Eigen, plugin architecture for custom models

---

## ⚡ Quickstart

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest -q
```

### Optional Dependencies

```bash
# For QuantLib integration
pip install -e ".[quantlib]"

# For Eigen bindings
pip install -e ".[eigen]"

# Install all optional dependencies
pip install -e ".[native]"
```

### Your First Pricing Example

```python
import jax
import jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value
from neutryx.models.bs import price as bs_price

# Setup
key = jax.random.PRNGKey(42)
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.01, 0.00, 0.2

# Monte Carlo pricing
cfg = MCConfig(steps=252, paths=100_000)
paths = simulate_gbm(key, S, r - q, sigma, T, cfg)
ST = paths[:, -1]
call_mc = present_value(jnp.maximum(ST - K, 0.0), jnp.array(T), r)

# Analytical pricing
call_an = bs_price(S, K, T, r, q, sigma, "call")

print(f"Call (MC): {float(call_mc):.4f}")
print(f"Call (BS): {float(call_an):.4f}")
```

### Multi-Asset Class Pricing

```python
from neutryx.products.equity import equity_forward_price, variance_swap_value
from neutryx.products.commodity import commodity_forward_price
from neutryx.products.inflation import inflation_linked_bond_price
from neutryx.products.volatility import vix_futures_price

# Equity forward with dividends
eq_forward = equity_forward_price(
    spot=100.0, maturity=1.0, risk_free_rate=0.05, dividend_yield=0.02
)

# Commodity forward with convenience yield
commodity_forward = commodity_forward_price(
    spot=50.0, maturity=1.0, risk_free_rate=0.05,
    storage_cost=0.02, convenience_yield=0.03
)

# VIX futures with mean reversion
vix_future = vix_futures_price(
    vix_spot=20.0, maturity=0.25, mean_reversion=1.5, long_term_vol=18.0
)

# Inflation-linked bond (TIPS)
tips_price = inflation_linked_bond_price(
    face_value=1000.0, real_coupon_rate=0.005, real_yield=0.003,
    maturity=10.0, index_ratio=1.25, frequency=2
)

print(f"Equity Forward: ${eq_forward:.2f}")
print(f"Commodity Forward: ${commodity_forward:.2f}")
print(f"VIX Future: {vix_future:.2f}")
print(f"TIPS Price: ${tips_price:.2f}")
```

### Configuration & Reproducibility

```python
from neutryx.config import get_config, init_environment
from neutryx.core.rng import KeySeq

# Load configuration
config = get_config({
    "seed": 2024,
    "logging": {"level": "INFO"},
})

# Initialize runtime environment
runtime = init_environment(config)
key_seq = KeySeq.from_config(runtime)
```

The configuration system seeds Python, NumPy, and JAX, configures logging, and provides
a runtime-aware configuration that carries a shared JAX PRNG key for downstream components.

### Greeks and Implied Volatility

```python
from neutryx.models import bs

# Price and Greeks
price = bs.price(S=100, K=100, T=1, r=0.01, q=0, sigma=0.2, kind="call")
delta, gamma, vega, theta = bs.greeks(100, 100, 1, 0.01, 0, 0.2)

# Second-order Greeks with automatic differentiation
greeks_dict = bs.second_order_greeks(
    S=100, K=100, T=1, r=0.01, q=0, sigma=0.2, kind="call"
)

# Implied volatility
implied = bs.implied_vol(100, 100, 1, 0.01, 0, price, kind="call")

print(f"Price: {price:.4f}, Delta: {delta:.4f}")
print(f"Gamma: {greeks_dict['gamma']:.6f}, Vanna: {greeks_dict['vanna']:.6f}")
print(f"IV: {implied:.4f}")
```

---

## 🧭 Project Structure

```text
neutryx-core/
├── .github/              # Continuous integration workflows and repository automation
├── config/               # YAML configuration presets for different environments
├── docs/                 # Lightweight design notes and reference documents
├── demos/                # Executable examples, dashboards, tutorials, and notebooks
├── src/                  # Source code for the Neutryx Python package
└── dev/                  # Developer tooling for profiling, reproducibility, and automation

demos/
├── asset_class_showcase.py  # Multi-asset class pricing demonstration
├── 01_bs_vanilla.py         # Vanilla Black–Scholes comparison
├── 02_path_dependents.py    # Asian, lookback, and barrier payoffs
├── 03_american_lsm.py       # Longstaff–Schwartz pricing workflow
├── dashboard/               # Dash application for interactive exploration
├── data/                    # Lightweight datasets consumed by examples
├── notebooks/               # Jupyter notebooks for exploratory analysis
├── perf/                    # Performance benchmark scripts
└── tutorials/               # Step-by-step guided walkthroughs

src/neutryx/
├── api/                  # REST and gRPC service surfaces
├── bridge/               # Integration layers for external runtimes (QuantLib, pandas)
├── core/                 # Simulation engines, infrastructure, grids, RNG, and utilities
│   ├── autodiff/         # Automatic differentiation helpers (Hessian-vector products)
│   ├── config/           # Runtime configuration management
│   ├── pricing/          # Core pricing engines and primitives
│   └── utils/            # Numerical solvers, time utilities, math helpers
├── market/               # Market data abstractions (curves, surfaces, conventions)
│   ├── credit/           # CDS pricing, hazard rates, structural models
│   └── fx.py             # FX volatility surfaces and conventions
├── models/               # Stochastic models (BS, Heston, SABR, jump diffusion, rough vol)
├── portfolio/            # Portfolio aggregation and allocation utilities
├── products/             # Complete product library across all asset classes
│   ├── vanilla.py        # European vanilla options
│   ├── asian.py          # Asian options
│   ├── barrier.py        # Barrier options
│   ├── american.py       # American options (Longstaff-Schwartz)
│   ├── equity.py         # Equity forwards, swaps, variance swaps, TRS, ELN
│   ├── commodity.py      # Commodity forwards/options/swaps with convenience yield
│   ├── bonds.py          # Zero-coupon, coupon, floating rate notes
│   ├── inflation.py      # TIPS, inflation swaps, caps/floors, breakeven analysis
│   ├── volatility.py     # VIX futures/options, variance swaps, VVIX
│   ├── convertible.py    # Convertible bonds, mandatory convertibles
│   ├── fx_options.py     # FX vanilla and barrier options
│   └── swaptions.py      # Interest rate swaptions
├── solver/               # Calibration routines and numerical solvers
│   └── calibration/      # Model calibration framework with diagnostics
├── tests/                # Comprehensive test suite (100+ tests)
│   ├── products/         # Product pricing tests (48 tests for new asset classes)
│   ├── market/           # Market data and curve tests
│   ├── monte_carlo/      # Monte Carlo accuracy tests
│   ├── performance/      # Benchmark tests
│   └── regression/       # Numerical stability tests
└── valuations/           # Exposure analytics and valuation adjustments (CVA, XVA)

dev/
├── benchmarks/           # Stand-alone benchmarking harnesses
├── profiling/            # Performance profiling tools and analysis
├── repro/                # Environment capture for reproducibility
└── scripts/              # Automation scripts (benchmarks, profiling)
```

---

## 📚 Documentation

The repository ships with a concise, Markdown-based knowledge base:

- [docs/overview.md](docs/overview.md) — High-level introduction and guiding principles.
- [docs/api_reference.md](docs/api_reference.md) — Public Python API surface and expected usage patterns.
- [docs/design_decisions.md](docs/design_decisions.md) — Architectural trade-offs, conventions, and rationale.
- [docs/roadmap.md](docs/roadmap.md) — Short- and mid-term development priorities.

You can extend these documents and publish them with MkDocs:

```bash
mkdocs serve  # run a local preview at http://127.0.0.1:8000/
mkdocs build  # produce a static site in the "site/" directory
```

When adding new guides, keep them alongside the existing files under `docs/` and link them from `mkdocs.yml` to surface them in the generated site.

---

## 💻 Examples

### Multi-Asset Class Showcase

Explore comprehensive pricing across all asset classes:

```bash
# Run the interactive multi-asset class demonstration
python demos/asset_class_showcase.py
```

This showcase demonstrates:
- **Equity:** Forwards, dividend swaps, variance swaps, total return swaps
- **Commodities:** Forwards with storage costs and convenience yield, options, swaps
- **Inflation:** TIPS pricing, zero-coupon inflation swaps, breakeven analysis
- **Volatility:** VIX futures, variance swaps, realized variance calculation
- **Convertibles:** Convertible bond analysis with conversion premiums

### Basic Examples

Run the included examples to explore different features:

```bash
# Vanilla option pricing (Monte Carlo vs. analytical)
python demos/01_bs_vanilla.py

# Path-dependent options (Asian, Lookback, Barrier)
python demos/02_path_dependents.py

# American options with Longstaff–Schwartz
python demos/03_american_lsm.py
```

### Performance Benchmarks

```bash
# Compare Monte Carlo vs. analytical pricing performance
python demos/perf/run_mc_vs_analytic.py

# Run benchmark suite
dev/scripts/bench.sh
```

### Dash Dashboard

```bash
cd demos/dashboard
python app.py
```

Visit `http://localhost:8050` to explore interactive pricing, Greeks, and scenario analysis views.

### Notebooks & Tutorials

- The `demos/notebooks/` directory hosts exploratory Jupyter notebooks such as `benchmark.ipynb` and `calibration_heston.ipynb`.
- Guided walkthrough material lives under `demos/tutorials/`, grouped by numbered subfolders.
- Place any lightweight CSV or JSON assets required by the scripts under `demos/data/`.

---

## 🔧 Development

### Code Quality Tools

```bash
# Run linter
ruff check src/

# Format code
ruff format src/

# Type checking (if using mypy)
mypy src/

# Run all tests
pytest

# Run with coverage
pytest --cov=neutryx --cov-report=html

# Run specific test categories
pytest src/neutryx/tests/products/      # Product tests
pytest src/neutryx/tests/regression/    # Regression tests
pytest src/neutryx/tests/performance/   # Performance benchmarks

# Security scanning
pip-audit                                # Dependency vulnerabilities
bandit -r src -ll                        # Static security analysis
```

### Configuration Files

Neutryx uses YAML configuration files for environment management:

- `src/neutryx/config/default.yaml` - Production defaults
- `src/neutryx/config/dev.yaml` - Development overrides

Load configurations programmatically:

```python
from neutryx.config import get_config

config = get_config("src/neutryx/config/dev.yaml")
```

### API Services

Start the REST API server:

```bash
uvicorn neutryx.api.rest:app --reload
```

Start the gRPC server:

```bash
python -m neutryx.api.grpc_server
```

---

## 🧪 Testing

The test suite includes multiple layers of validation with **100+ comprehensive tests**:

- **Unit tests:** Core functionality and model correctness
- **Integration tests:** End-to-end workflows
- **Product tests:** 48 tests for multi-asset class coverage
- **Regression tests:** Numerical stability checks
- **Performance tests:** Benchmarking and profiling
- **Precision tests:** Numerical accuracy validation

Run the full suite:

```bash
pytest -v
```

Run specific test suites:

```bash
# Test new asset class products
pytest src/neutryx/tests/products/test_equity.py -v
pytest src/neutryx/tests/products/test_commodity.py -v
pytest src/neutryx/tests/products/test_inflation.py -v
pytest src/neutryx/tests/products/test_volatility.py -v
pytest src/neutryx/tests/products/test_convertible.py -v
```

Generate coverage report:

```bash
pytest --cov=neutryx --cov-report=html
open htmlcov/index.html
```

---

## 🗺️ Development Roadmap

### ✅ Completed Features (v0.1.0)

Neutryx Core has successfully implemented a comprehensive quantitative finance framework with:

#### Core Infrastructure
- **Numerical Engines:** PDE solvers (Crank-Nicolson, boundary conditions), GPU optimization (pmap/pjit)
- **Differentiation:** AAD with Hessian-vector products, second-order Greeks
- **Performance:** JIT compilation for 10-100x speedup, mixed-precision support
- **Configuration:** YAML-based runtime configuration, reproducible PRNG seeding

#### Advanced Pricing
- **Monte Carlo:** Adjoint Monte Carlo, QMC/MLMC engines, pathwise Greeks
- **Models:** Black-Scholes, Heston, SABR, jump diffusion, rough volatility
- **Methods:** FFT/COS pricing, tree-based methods (binomial/trinomial)
- **Products:** Full exotic options suite (American, Asian, Barrier, Lookback)

#### Multi-Asset Class Products (NEW)
- **Equity Derivatives:**
  - Forwards with dividend adjustments
  - Dividend swaps and total return swaps (TRS)
  - Variance and volatility swaps with Carr-Madan replication
  - Equity-linked notes (ELN) with caps and floors
- **Commodity Products:**
  - Forwards with convenience yield and storage costs
  - Options with commodity-specific cost-of-carry
  - Commodity swaps and spread options (Kirk's approximation)
  - Asian commodity options
- **Fixed Income:**
  - Zero-coupon, coupon, and floating rate notes
  - Duration, convexity, and yield calculations
  - Pricing from yield curves
- **Inflation-Linked:**
  - Treasury Inflation-Protected Securities (TIPS)
  - Zero-coupon and year-on-year inflation swaps
  - Inflation caps and floors using Black's model
  - Breakeven inflation analysis
- **Volatility Products:**
  - VIX futures with mean reversion models
  - VIX options using Black's model
  - Variance swaps with market replication
  - Corridor variance swaps and gamma swaps
  - VVIX (volatility of VIX) calculation
- **Convertible Bonds:**
  - Convertible bond pricing and analytics
  - Mandatory convertibles with variable conversion ratios
  - Exchangeable bonds
  - Conversion parity, premiums, and delta calculations

#### Calibration & Risk
- **Calibration:** Differentiable framework for SABR/SLV/Heston with diagnostics
- **XVA Suite:** Full implementation (CVA, FVA, MVA) with exposure simulation
- **Risk:** Stress testing engines, scenario generation, capital metrics
- **Credit:** CDS pricing, hazard rate models, structural credit models

#### Market Data
- **Multi-Curve Framework:** OIS discounting, tenor basis, currency basis
- **FX Markets:** Advanced volatility surfaces (10Δ/15Δ, smile interpolation)
- **Term Structures:** Yield curve bootstrapping from market instruments

#### Infrastructure
- **APIs:** CLI/REST/gRPC endpoints for production integration
- **Dashboards:** Interactive Dash applications for pricing and risk
- **CI/CD:** Comprehensive automation with security scanning (pip-audit, bandit)
- **Quality:** 100+ tests, code quality enforcement (ruff/black), SBOM generation

### 🚀 Future Enhancements

#### Extended Model Coverage
- Kou and Variance Gamma jump models
- Local volatility (Dupire PDE) with advanced boundary treatments
- Rough volatility models (fractional Heston, rBergomi)
- Time-inhomogeneous and regime-switching models

#### Performance Optimization
- Expand benchmarking suite for CPU/GPU/TPU comparison
- C++/CUDA kernel integration via FFI for critical paths
- Advanced distributed execution patterns across multi-GPU clusters

#### Product Expansion
- Exotic options: Quanto, Cliquet, Rainbow, Worst-of/Best-of baskets
- Interest rate derivatives: Bermudan swaptions, CMS products
- Structured products: Principal-protected notes, autocallables

#### Advanced Calibration
- Joint calibration across multiple instruments and underlyings
- Time-dependent parameter estimation
- Model selection and identifiability analysis
- Regularization techniques for ill-posed problems

#### Production Readiness
- Cloud/HPC orchestration (Slurm, Ray, Kubernetes)
- Checkpoint and resume capabilities for long-running jobs
- Real-time market data integration
- Production monitoring and alerting

#### Developer Experience
- Enhanced documentation with video tutorials
- Interactive Jupyter notebooks for all examples
- Plugin system for custom models and products
- Performance profiling and visualization tools

#### Research Integration
- Deep learning-based model-free pricing
- Generative models for scenario generation
- Causal inference for risk attribution
- Quantum computing experiments (variational pricing)

#### Enterprise Features
- Multi-tenancy and access control
- Audit logging and compliance reporting
- Integration with commercial data providers (Bloomberg, Refinitiv)
- Regulatory reporting templates (FRTB, SA-CCR)

#### Community & Ecosystem
- Plugin marketplace for community models
- Reproducibility tracking with Weights & Biases/MLflow
- Certified training programs
- Academic partnerships and benchmarking initiatives

See [docs/roadmap.md](docs/roadmap.md) for detailed timelines and milestones.

---

## 🤝 Contributing

We welcome contributions from the community! Here's how to get started:

### Contribution Workflow

1. **Fork the repository** and create a feature branch

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** following our coding standards:
   - Use `ruff` for linting and formatting
   - Add tests for new functionality (aim for >90% coverage)
   - Update documentation as needed
   - Include docstrings (Google style)

3. **Run the test suite** before submitting

   ```bash
   pytest -q
   ruff check src/
   ruff format src/
   pip-audit  # Check for security vulnerabilities
   ```

4. **Commit your changes** with clear, concise messages

   ```bash
   git commit -m "feat: Add feature description"
   ```

5. **Push to your fork** and open a pull request

   ```bash
   git push origin feature/my-feature
   ```

### Guidelines

- **Code Style:** Follow PEP 8, enforced by `ruff` and `black`
- **Testing:** Include unit tests for new features and bug fixes
- **Documentation:** Update relevant docs and add comprehensive docstrings
- **Model Additions:** Supply analytical reference tests and convergence studies
- **Commit Messages:** Use conventional commits format (feat:, fix:, docs:, etc.)

### Areas for Contribution

- New pricing models or products
- Performance optimizations
- Documentation improvements
- Bug fixes and issue reports
- Examples and tutorials
- Test coverage expansion
- Integration with external libraries

---

## 💬 Community & Support

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/neutryx-lab/neutryx-core/issues)
- **Discussions:** Ask questions and share ideas in [GitHub Discussions](https://github.com/neutryx-lab/neutryx-core/discussions)
- **Email:** For private inquiries, contact `dev@neutryx.tech`

### Code of Conduct

Be kind. Be curious. No harassment or discrimination. We are committed to providing a welcoming and inclusive environment for all contributors.

---

## 📋 Changelog

**v0.1.0** - Initial public release with comprehensive multi-asset class support

Core features:

- Black-Scholes analytical pricing and Greeks with JIT compilation
- Advanced Monte Carlo engine with GBM simulation
- Path-dependent products (Asian, Barrier, Lookback, American)
- **Multi-Asset Class Products:**
  - Equity: Forwards, swaps, variance products, TRS, ELN
  - Commodities: Forwards, options, swaps with convenience yield
  - Fixed Income: Bonds, TIPS, inflation swaps
  - Volatility: VIX futures/options, variance swaps
  - Convertibles: Convertible bonds and analytics
- Configuration and reproducibility system
- REST/gRPC API endpoints
- Comprehensive test suite (100+ tests)
- Interactive demo showcase

### Upcoming Releases

- **v0.2** – Enhanced calibration framework and model extensions
- **v0.3** – Advanced risk metrics and scenario analysis
- **v1.0** – Production-ready release with full API stability

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## 🔒 Security

Neutryx Core is research-oriented software intended for development and prototyping.

- **Vulnerability Reports:** Please report security issues privately to `dev@neutryx.tech`
- **Security Audits:** See [docs/security_audit.md](docs/security_audit.md) for procedures
- **Dependencies:** Regularly updated via Dependabot and monitored with pip-audit
- **Static Analysis:** Continuous security scanning with bandit
- **SBOM:** Software Bill of Materials generated in CI pipeline

For production deployments, conduct thorough security reviews and testing.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2025 Neutryx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

Neutryx Core builds upon the incredible work of the scientific computing community:

- **JAX Team** at Google Research for the foundational JAX framework
- **QuantLib** contributors for quantitative finance reference implementations
- **SciPy/NumPy** communities for scientific computing infrastructure
- All open-source contributors who make projects like this possible

Special thanks to the quantitative finance community for sharing knowledge and best practices.

---

## 🔗 Links

- **Documentation:** [https://neutryx-lab.github.io/neutryx-core](https://neutryx-lab.github.io/neutryx-core)
- **Repository:** [https://github.com/neutryx-lab/neutryx-core](https://github.com/neutryx-lab/neutryx-core)
- **Issues:** [https://github.com/neutryx-lab/neutryx-core/issues](https://github.com/neutryx-lab/neutryx-core/issues)
- **Website:** [https://neutryx.tech](https://neutryx.tech)

---

<p align="center">
  Built with ❤️ by the Neutryx team
</p>

<p align="center">
  <sub>Accelerating quantitative finance with differentiable computing</sub>
</p>
