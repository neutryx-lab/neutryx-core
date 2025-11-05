# Developer Guide

Welcome to the Neutryx Core developer guide! This document covers everything you need to know to contribute to the project.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Style and Standards](#code-style-and-standards)
3. [Testing](#testing)
4. [Contributing Workflow](#contributing-workflow)
5. [Adding New Features](#adding-new-features)
6. [Documentation](#documentation)
7. [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- GPU (optional, but recommended for performance testing)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/neutryx-lab/neutryx-core.git
cd neutryx-core

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest -q
```

### IDE Setup

#### VS Code

Recommended extensions:
- Python
- Pylance
- Ruff
- Test Explorer

`.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

#### PyCharm

1. Open project in PyCharm
2. Configure interpreter: Settings → Project → Python Interpreter
3. Enable pytest: Settings → Tools → Python Integrated Tools → Testing
4. Install Ruff plugin

## Code Style and Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- Line length: 100 characters (not 79)
- Use double quotes for strings
- Use type hints for all function signatures

### Code Formatting

We use Ruff for linting and Black for formatting:

```bash
# Format code
black src/neutryx tests

# Lint code
ruff check src/neutryx tests

# Auto-fix linting issues
ruff check --fix src/neutryx tests
```

### Type Hints

Always use type hints:

```python
from typing import Tuple, Optional
import jax.numpy as jnp

def price_option(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str = "call"
) -> float:
    """
    Price a European option using Black-Scholes.

    Args:
        spot: Current spot price
        strike: Strike price
        maturity: Time to maturity in years
        rate: Risk-free rate
        volatility: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Option price

    Raises:
        ValueError: If option_type is invalid
    """
    if option_type not in ["call", "put"]:
        raise ValueError(f"Invalid option_type: {option_type}")

    # Implementation...
    return price
```

### Documentation Standards

Every public function must have a docstring:

```python
def calculate_var(
    returns: jnp.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk using historical simulation.

    This function computes VaR by sorting historical returns and
    finding the appropriate quantile.

    Args:
        returns: Array of historical returns
        confidence: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR as a positive number

    Examples:
        >>> returns = jnp.array([-0.02, -0.01, 0.01, 0.02, 0.03])
        >>> var_95 = calculate_var(returns, confidence=0.95)
        >>> print(f"VaR(95%): {var_95:.4f}")
        VaR(95%): 0.0180

    References:
        - Jorion, P. (2006). Value at Risk. McGraw-Hill.
    """
    sorted_returns = jnp.sort(returns)
    index = int((1 - confidence) * len(returns))
    return -sorted_returns[index]
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest src/neutryx/tests/test_models.py

# Run tests with coverage
pytest --cov=neutryx --cov-report=html

# Run tests in parallel
pytest -n auto

# Run only fast tests (skip slow Monte Carlo tests)
pytest -m "not slow"
```

### Writing Tests

#### Unit Tests

```python
# src/neutryx/tests/test_pricing.py
import pytest
import jax.numpy as jnp
from neutryx.models.bs import price

class TestBlackScholes:
    """Test Black-Scholes pricing"""

    def test_call_option_atm(self):
        """Test at-the-money call option"""
        S, K, T = 100.0, 100.0, 1.0
        r, q, sigma = 0.05, 0.02, 0.20
        price_call = price(S, K, T, r, q, sigma, "call")

        # Check price is reasonable
        assert 0 < price_call < S
        assert price_call == pytest.approx(9.625, rel=0.01)

    def test_put_call_parity(self):
        """Test put-call parity holds"""
        S, K, T = 100.0, 100.0, 1.0
        r, q, sigma = 0.05, 0.02, 0.20

        call = price(S, K, T, r, q, sigma, "call")
        put = price(S, K, T, r, q, sigma, "put")

        # C - P = S*e^(-qT) - K*e^(-rT)
        lhs = call - put
        rhs = S * jnp.exp(-q * T) - K * jnp.exp(-r * T)

        assert lhs == pytest.approx(rhs, abs=1e-10)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_positive_prices(self, option_type):
        """Test that prices are always positive"""
        S, K, T = 100.0, 100.0, 1.0
        r, q, sigma = 0.05, 0.02, 0.20
        price_val = price(S, K, T, r, q, sigma, option_type)

        assert price_val > 0

    def test_invalid_option_type(self):
        """Test error handling for invalid option type"""
        with pytest.raises(ValueError, match="Invalid option_type"):
            price(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, "invalid")
```

#### Integration Tests

```python
# src/neutryx/tests/integration/test_pricing_workflow.py
import pytest
from neutryx.models.bs import BlackScholesModel
from neutryx.products.options import VanillaOption
from neutryx.market.data import MarketData

class TestPricingWorkflow:
    """Test end-to-end pricing workflow"""

    def test_complete_pricing_pipeline(self):
        """Test complete pricing from market data to result"""
        # Setup market data
        market = MarketData(
            spot=100.0,
            rate=0.05,
            dividend=0.02,
            volatility=0.20
        )

        # Create product
        option = VanillaOption(
            strike=100.0,
            maturity=1.0,
            option_type="call"
        )

        # Create model and price
        model = BlackScholesModel()
        price = model.price(option, market)

        # Verify result
        assert price > 0
        assert price < market.spot

        # Calculate Greeks
        greeks = model.greeks(option, market)
        assert 0 < greeks['delta'] < 1
        assert greeks['gamma'] > 0
        assert greeks['vega'] > 0
```

#### Performance Tests

```python
# src/neutryx/tests/test_performance.py
import pytest
import time
import jax
import jax.numpy as jnp
from neutryx.models.bs import price

@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""

    def test_batch_pricing_speed(self, benchmark):
        """Benchmark batch pricing performance"""
        spots = jnp.ones(10000) * 100.0
        strikes = jnp.ones(10000) * 100.0
        maturities = jnp.ones(10000) * 1.0

        @jax.jit
        def price_batch(S, K, T):
            return jax.vmap(
                lambda s, k, t: price(s, k, t, 0.05, 0.02, 0.20, "call")
            )(S, K, T)

        # Warmup
        _ = price_batch(spots, strikes, maturities).block_until_ready()

        # Benchmark
        result = benchmark(
            lambda: price_batch(spots, strikes, maturities).block_until_ready()
        )

        # Should price 10k options in < 100ms
        assert benchmark.stats['mean'] < 0.1

    @pytest.mark.slow
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo convergence rate"""
        from neutryx.core.engine import monte_carlo_price

        path_counts = [1000, 5000, 10000, 50000]
        analytical = 9.625

        for n_paths in path_counts:
            mc_price = monte_carlo_price(
                S0=100, K=100, T=1.0, r=0.05, sigma=0.20,
                n_paths=n_paths
            )
            error = abs(mc_price - analytical)
            expected_error = 5.0 / np.sqrt(n_paths)  # ~5σ error bound

            assert error < expected_error
```

### Test Fixtures

```python
# src/neutryx/tests/conftest.py
import pytest
import jax.numpy as jnp

@pytest.fixture
def market_data():
    """Standard market data fixture"""
    return {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.05,
        "dividend": 0.02,
        "volatility": 0.20
    }

@pytest.fixture
def sample_returns():
    """Sample return data for testing"""
    return jnp.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])

@pytest.fixture(scope="session")
def test_database():
    """Database fixture for integration tests"""
    # Setup
    db = create_test_database()
    yield db
    # Teardown
    db.drop_all()
```

## Contributing Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

### Contribution Process

1. **Fork and Clone**
```bash
git clone https://github.com/YOUR_USERNAME/neutryx-core.git
cd neutryx-core
git remote add upstream https://github.com/neutryx-lab/neutryx-core.git
```

2. **Create Feature Branch**
```bash
git checkout -b feature/my-new-feature develop
```

3. **Make Changes**
```bash
# Edit files
vim src/neutryx/models/new_model.py

# Add tests
vim src/neutryx/tests/test_new_model.py

# Format code
black src/neutryx tests
ruff check --fix src/neutryx tests
```

4. **Commit Changes**
```bash
git add src/neutryx/models/new_model.py
git add src/neutryx/tests/test_new_model.py
git commit -m "feat: add new pricing model

- Implement XYZ model
- Add comprehensive tests
- Update documentation

Closes #123"
```

**Commit Message Convention**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

5. **Push and Create PR**
```bash
git push origin feature/my-new-feature
```

Then create Pull Request on GitHub.

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass (`pytest`)
- [ ] New code has tests (>80% coverage)
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts with `develop`
- [ ] PR description explains changes
- [ ] Links to related issues

## Adding New Features

### Adding a New Model

1. **Create Model Class**
```python
# src/neutryx/models/my_model.py
from neutryx.models.base import Model
import jax.numpy as jnp

class MyModel(Model):
    """
    My custom pricing model.

    This model implements...

    References:
        - Author (Year). Paper Title. Journal.
    """

    def __init__(self, **params):
        self.params = params

    @jax.jit
    def price(self, spot, strike, maturity, **kwargs):
        """Price an option"""
        # Implementation
        return price

    def calibrate(self, market_data, **kwargs):
        """Calibrate model to market data"""
        # Implementation
        return calibrated_params
```

2. **Add Tests**
```python
# src/neutryx/tests/test_my_model.py
import pytest
from neutryx.models.my_model import MyModel

class TestMyModel:
    def test_pricing(self):
        model = MyModel(param1=1.0, param2=2.0)
        price = model.price(100, 100, 1.0)
        assert price > 0

    def test_calibration(self):
        model = MyModel()
        params = model.calibrate(market_data)
        assert params is not None
```

3. **Add Documentation**
```python
# docs/models/my_model.md
```

4. **Update Exports**
```python
# src/neutryx/models/__init__.py
from .my_model import MyModel

__all__ = ["MyModel", ...]
```

### Adding a New Product

```python
# src/neutryx/products/my_product.py
from neutryx.products.base import Product
import jax.numpy as jnp

class MyProduct(Product):
    """
    Custom derivative product.

    Args:
        param1: Description
        param2: Description
    """

    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def payoff(self, paths):
        """Calculate payoff given price paths"""
        # Implementation
        return payoff

    def price(self, model, market_data):
        """Price product using model"""
        paths = model.simulate(market_data)
        payoff = self.payoff(paths)
        return self.present_value(payoff, market_data.rate)
```

## Documentation

### Building Documentation

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Build documentation
mkdocs build

# Serve locally
mkdocs serve
# Visit http://localhost:8000

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Documentation Structure

```
docs/
├── index.md                 # Home page
├── getting_started.md       # Quick start guide
├── overview.md              # Project overview
├── tutorials.md             # Tutorials
├── architecture.md          # Architecture guide
├── performance_tuning.md    # Performance guide
├── troubleshooting.md       # Troubleshooting
├── developer_guide.md       # This file
├── api_reference.md         # API docs
├── models/                  # Model documentation
│   ├── index.md
│   └── black_scholes.md
└── products/                # Product documentation
    ├── index.md
    └── vanilla_options.md
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. **Update Version**
```python
# src/neutryx/__init__.py
__version__ = "0.2.0"
```

2. **Update Changelog**
```markdown
# docs/project/CHANGELOG.md

## [0.2.0] - 2025-01-15

### Added
- New Heston model implementation
- Real-time market data integration

### Fixed
- Monte Carlo convergence issue (#123)

### Changed
- Improved calibration performance
```

3. **Run Full Test Suite**
```bash
pytest --cov=neutryx --cov-report=html
pytest --benchmark-only
```

4. **Build and Test Package**
```bash
python -m build
pip install dist/neutryx_core-0.2.0-py3-none-any.whl
python -c "import neutryx; print(neutryx.__version__)"
```

5. **Create Git Tag**
```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

6. **Create GitHub Release**
- Go to GitHub Releases
- Create new release from tag
- Add release notes
- Attach built packages

7. **Publish to PyPI** (if applicable)
```bash
python -m twine upload dist/*
```

## Code Review Guidelines

### For Authors

- Keep PRs focused and small (<500 lines)
- Write clear PR descriptions
- Respond to feedback promptly
- Update PR based on comments

### For Reviewers

Check for:
- [ ] Code correctness
- [ ] Test coverage
- [ ] Performance implications
- [ ] Security concerns
- [ ] Documentation quality
- [ ] Code style compliance

## Community

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs and request features
- **Discord** (coming soon): Chat with developers

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [QuantLib](https://www.quantlib.org/)
- [Python Packaging Guide](https://packaging.python.org/)

Thank you for contributing to Neutryx Core!
