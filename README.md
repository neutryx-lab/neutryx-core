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

- **Models:** Analytic Black-Scholes, stochastic volatility (Heston, SABR), Monte Carlo engines
- **Products:** European, Asian, Barrier, Lookback, and American (Longstaff-Schwartz) payoffs
- **Risk:** Pathwise & bump Greeks, stress testing, and adjoint-based sensitivity analysis
- **Market:** Yield curve & volatility surface management for calibration or scenario replay
- **XVA:** Exposure models (CVA, FVA, MVA) for counterparty risk prototyping
- **Calibration:** Differentiable calibration framework with diagnostics and identifiability checks

### Technical Highlights

- **JAX-Native:** Full JIT compilation, automatic differentiation, and XLA optimization
- **GPU/TPU Ready:** Seamless acceleration on modern hardware with `pmap`/`pjit`
- **Reproducible:** Unified configuration via YAML, consistent PRNG seeding
- **Production-Ready:** FastAPI/gRPC APIs, comprehensive test suite (pytest), quality tooling (ruff, black)
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

# Install in editable mode
pip install -e .

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

# Implied volatility
implied = bs.implied_vol(100, 100, 1, 0.01, 0, price, kind="call")

print(f"Price: {price:.4f}, Delta: {delta:.4f}, IV: {implied:.4f}")
```

---

## 🧭 Project Structure

```text
neutryx-Core/
├── .github/              # Continuous integration workflows and repository automation
├── config/               # YAML configuration presets for different environments
├── docs/                 # Lightweight design notes and reference documents
├── demos/             # Executable examples, dashboards, tutorials, and notebooks
├── src/                  # Source code for the Neutryx Python package
└── dev/                # Developer tooling for profiling, reproducibility, and automation

demos/
├── 01_bs_vanilla.py      # Vanilla Black–Scholes comparison
├── 02_path_dependents.py # Asian, lookback, and barrier payoffs
├── 03_american_lsm.py    # Longstaff–Schwartz pricing workflow
├── dashboard/            # Dash application for interactive exploration
├── data/                 # Lightweight datasets consumed by examples
├── notebooks/            # Jupyter notebooks for exploratory analysis
├── perf/                 # Performance benchmark scripts
└── tutorials/            # Step-by-step guided walkthroughs

src/neutryx/
├── api/                  # REST and gRPC service surfaces
├── bridge/               # Integration layers for external runtimes (QuantLib, pandas, native extensions)
├── core/                 # Simulation engines, infrastructure, grids, RNG, and shared utilities
├── market/               # Yield curves, term structures, and credit market helpers
├── models/               # Stochastic models and pricing formulas
├── portfolio/            # Aggregation and allocation utilities
├── products/             # Payoff definitions for vanilla and exotic options
├── solver/               # Calibration routines and solver implementations
├── tests/                # In-package regression, integration, and precision tests
└── valuations/           # Exposure analytics and valuation adjustments (CVA, XVA scenarios)
```

## 🗂️ Directory Purpose Reference (for AI-assisted edits)

To keep automated tooling from scattering files in the wrong locations, refer to the following map before creating new code or assets:

| Path | Purpose |
| --- | --- |
| `.github/` | GitHub-specific configuration such as workflows and repository automation. |
| `.github/workflows/` | CI pipelines (e.g., `ci.yml`) executed on pull requests and pushes. |
| `.vscode/` | Editor settings and recommended workspace configuration for VS Code. |
| `config/` | Base and development YAML settings consumed by the runtime configuration loader. |
| `docs/` | High-level documentation delivered with the repository (API notes, architecture decisions, roadmap). |
| `demos/` | Scripted examples demonstrating pricing flows and benchmarking scenarios. |
| `demos/dashboard/` | Dash app for interactive pricing and risk exploration. |
| `demos/dashboard/assets/` | Static assets (CSS/images) loaded by the dashboard. |
| `demos/data/` | Auxiliary datasets referenced by example scripts (keep lightweight sample data here). |
| `demos/notebooks/` | Jupyter notebooks for exploratory analyses and tutorials. |
| `demos/perf/` | Performance benchmark scripts focusing on runtime comparisons. |
| `demos/tutorials/` | Guided walkthroughs broken into topical subfolders. |
| `demos/tutorials/01_vanilla_pricing/` | Step-by-step vanilla option pricing tutorial resources. |
| `demos/tutorials/02_asian_scenario/` | Scenario analysis tutorial for Asian options. |
| `demos/tutorials/03_counterparty_cva/` | Counterparty credit valuation adjustment tutorial assets. |
| `src/` | Python package source tree (installable via `pip`). |
| `src/neutryx/` | Root package namespace. New library modules should live underneath this path. |
| `src/neutryx/api/` | REST and gRPC endpoints exposing pricing services. |
| `src/neutryx/bridge/` | Adapters that connect Neutryx with external systems (e.g., pandas, QuantLib, native code). |
| `src/neutryx/bridge/cpp/` | Documentation and stubs related to prospective C++ bindings. |
| `src/neutryx/bridge/cuda/` | Documentation and scaffolding for CUDA extensions. |
| `src/neutryx/bridge/ffi/` | Shared FFI glue and build notes. |
| `src/neutryx/core/` | Core simulation utilities (engines, grids, RNG, infrastructure). |
| `src/neutryx/core/autodiff/` | Automatic differentiation helpers and Jacobian/Hessian utilities. |
| `src/neutryx/core/config/` | Runtime configuration management and environment bootstrapping. |
| `src/neutryx/core/infrastructure/` | Execution backends, batching, and distributed runtime helpers. |
| `src/neutryx/core/pricing/` | Pricing engines and reusable primitives. |
| `src/neutryx/core/utils/` | Low-level math helpers, numerical schemes, and shared utilities. |
| `src/neutryx/core/utils/cli/` | Command-line tooling for orchestrating runs. |
| `src/neutryx/market/` | Market data abstractions such as curves and volatility surfaces. |
| `src/neutryx/market/credit/` | Credit curve and hazard rate utilities. |
| `src/neutryx/models/` | Mathematical models and stochastic dynamics implementations. |
| `src/neutryx/portfolio/` | Portfolio aggregation logic and capital allocation helpers. |
| `src/neutryx/products/` | Payoff definitions for American, Asian, barrier, and vanilla products. |
| `src/neutryx/solver/` | Calibration and solver routines (e.g., Heston, SABR). |
| `src/neutryx/solver/calibration/` | Shared calibration workflows and diagnostics. |
| `src/neutryx/tests/` | In-package pytest suite mirroring package modules. |
| `src/neutryx/tests/autodiff/` | Automatic differentiation regression tests. |
| `src/neutryx/tests/fixtures/` | Fixture data and helper factories for tests. |
| `src/neutryx/tests/integration/` | Integration and workflow validation tests. |
| `src/neutryx/tests/market/` | Market data and curve validation tests. |
| `src/neutryx/tests/monte_carlo/` | Monte Carlo accuracy and convergence tests. |
| `src/neutryx/tests/performance/` | Performance and benchmarking-focused tests. |
| `src/neutryx/tests/precision/` | High-precision numerical validation tests. |
| `src/neutryx/tests/regression/` | Runtime and stability regression tests. |
| `src/neutryx/tests/tools/` | Utilities leveraged by the in-package test suite. |
| `src/neutryx/valuations/` | Valuation adjustment engines (CVA/FVA/MVA) and supporting helpers. |
| `src/neutryx/valuations/greeks/` | Sensitivity and Greek calculators tied to valuations. |
| `src/neutryx/valuations/scenarios/` | Scenario generation utilities for valuation analytics. |
| `src/neutryx/valuations/xva/` | Counterparty valuation adjustment components. |
| `dev/` | Developer utilities for profiling, reproducibility, and scripted workflows. |
| `dev/benchmarks/` | Stand-alone benchmarking harnesses and reports. |
| `dev/ci/` | Helper scripts used in continuous integration pipelines. |
| `dev/orchestration/` | Deployment and orchestration helpers (e.g., batch scripts). |
| `dev/profiling/` | Artefacts and configuration for performance profiling. |
| `dev/profiling/flamegraphs/` | Placeholder directory for generated flamegraph SVGs. |
| `dev/profiling/notebooks/` | Profiling-focused notebooks and analysis aids. |
| `dev/profiling/xla_hlo_dumps/` | Storage for XLA HLO dumps produced during JAX profiling runs. |
| `dev/repro/` | Environment capture files (pip freeze, backend info) supporting reproducibility. |
| `dev/scripts/` | Shell and Python automation scripts (benchmarks, profiling helpers). |

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
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type checking (if using mypy)
mypy src/

# Run all tests
pytest

# Run with coverage
pytest --cov=neutryx --cov-report=html

# Run specific test categories
pytest -m regression  # Regression tests
pytest -m performance  # Performance benchmarks
```

### Configuration Files

Neutryx uses YAML configuration files for environment management:

- `config/default.yaml` - Production defaults
- `config/dev.yaml` - Development overrides

Load configurations programmatically:

```python
from neutryx.config import get_config

config = get_config("config/dev.yaml")
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

The test suite includes multiple layers of validation:

- **Unit tests:** Core functionality and model correctness
- **Integration tests:** End-to-end workflows
- **Regression tests:** Numerical stability checks
- **Performance tests:** Benchmarking and profiling
- **Precision tests:** Numerical accuracy validation

Run the full suite:

```bash
pytest -v
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

- **Core Infrastructure:** PDE solvers (Crank-Nicolson, boundary conditions), GPU optimization (pmap/pjit), AAD with Hessian-vector products, mixed-precision support, YAML configuration system
- **Advanced Pricing:** Adjoint Monte Carlo, QMC/MLMC engines, FFT/COS methods, jump diffusion models, tree-based methods
- **Calibration:** Differentiable framework for SABR/SLV/Heston with diagnostics and residual analysis
- **Risk & XVA:** Full XVA suite (CVA, FVA, MVA), stress testing engines, exposure simulation, capital metrics
- **Credit:** CDS pricing, hazard rate models, structural models
- **APIs & Tools:** CLI/REST/gRPC endpoints, interactive dashboards, CI/CD automation with security scanning
- **Quality:** 38+ test files, code quality enforcement (ruff/black/pydantic), comprehensive documentation

### 🚀 Future Enhancements

- **Extended Model Coverage**
  - Kou and Variance Gamma jump models
  - Local volatility (Dupire PDE) with advanced boundary treatments
  - Rough volatility models (fractional Heston, rBergomi)
  - Time-inhomogeneous and regime-switching models

- **Performance Optimization**
  - Expand benchmarking suite for CPU/GPU/TPU comparison
  - C++/CUDA kernel integration via FFI for critical paths
  - Advanced distributed execution patterns across multi-GPU clusters

- **Product Expansion**
  - Exotic options: Quanto, Cliquet, Rainbow, Worst-of/Best-of baskets
  - FX products: Variance swaps, volatility swaps
  - Interest rate derivatives: Swaptions, caps/floors, Bermudan options

- **Advanced Calibration**
  - Joint calibration across multiple instruments
  - Time-dependent parameter estimation
  - Model selection and identifiability analysis
  - Regularization techniques for ill-posed problems

- **Production Readiness**
  - Cloud/HPC orchestration (Slurm, Ray, Kubernetes)
  - Checkpoint and resume capabilities for long-running jobs
  - Real-time market data integration
  - Production monitoring and alerting

- **Developer Experience**
  - Enhanced documentation with video tutorials
  - Interactive Jupyter notebooks for all examples
  - Plugin system for custom models and products
  - Performance profiling and visualization tools

- **Research Integration**
  - Deep learning-based model-free pricing
  - Generative models for scenario generation
  - Causal inference for risk attribution
  - Quantum computing experiments (variational pricing)

- **Enterprise Features**
  - Multi-tenancy and access control
  - Audit logging and compliance reporting
  - Integration with commercial data providers (Bloomberg, Refinitiv)
  - Regulatory reporting templates (FRTB, SA-CCR)

- **Community & Ecosystem**
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
   - Add tests for new functionality
   - Update documentation as needed
   - Include docstrings (Google style)

3. **Run the test suite** before submitting

   ```bash
   pytest -q
   ruff check src/ tests/
   ruff format src/ tests/
   ```

4. **Commit your changes** with clear, concise messages

   ```bash
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork** and open a pull request

   ```bash
   git push origin feature/my-feature
   ```

### Guidelines

- **Code Style:** Follow PEP 8, enforced by `ruff` and `black`
- **Testing:** Include unit tests for new features and bug fixes
- **Documentation:** Update relevant docs and add docstrings
- **Model Additions:** Supply analytical reference tests and convergence studies
- **Commit Messages:** Use clear, descriptive messages explaining the "why"

### Areas for Contribution

- New pricing models or products
- Performance optimizations
- Documentation improvements
- Bug fixes and issue reports
- Examples and tutorials
- Test coverage expansion

---

## 💬 Community & Support

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/neutryx-lab/neutryx-core/issues)
- **Discussions:** Ask questions and share ideas in [GitHub Discussions](https://github.com/neutryx-lab/neutryx-core/discussions)
- **Email:** For private inquiries, contact `dev@neutryx.tech`

### Code of Conduct

Be kind. Be curious. No harassment or discrimination. We are committed to providing a welcoming and inclusive environment for all contributors.

---

## 📋 Changelog

**v0.1.0** - Initial public release

Core features:

- Black-Scholes analytical pricing and Greeks
- Monte Carlo engine with GBM simulation
- Path-dependent products (Asian, Barrier, Lookback)
- American options via Longstaff-Schwartz
- Configuration and reproducibility system
- REST/gRPC API endpoints
- Comprehensive test suite

### Upcoming Releases

- **v0.9-core** – Core PDE and GPU engine stabilization
- **v1.0-release** – Complete model suite and calibration framework
- **v1.1-risk** – Full XVA and scenario platform

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## 🔒 Security

Neutryx Core is research-oriented software intended for development and prototyping.

- **Vulnerability Reports:** Please report security issues privately to `dev@neutryx.tech`
- **Security Audits:** See [docs/security_audit.md](docs/security_audit.md) for procedures
- **Dependencies:** Regularly updated via Dependabot
- **SBOM:** Software Bill of Materials available upon request

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
