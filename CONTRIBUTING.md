# Contributing to Neutryx Core

Thank you for your interest in contributing to Neutryx Core! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Guidelines](#contribution-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

### Expected Behavior

- Be respectful and considerate of differing viewpoints
- Provide constructive feedback
- Focus on what is best for the community and the project
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling, insulting, or derogatory remarks
- Personal or political attacks
- Publishing others' private information without permission

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/neutryx-lab/neutryx-core.git
cd neutryx-core

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Run tests to verify setup
pytest -q
```

---

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/neutryx-core.git
cd neutryx-core

# Add upstream remote
git remote add upstream https://github.com/neutryx-lab/neutryx-core.git
```

### 2. Create a Branch

```bash
# Create a feature branch from main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 3. Make Changes

- Write clean, maintainable code
- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation as needed

### 4. Commit Changes

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: Add support for SOFR forward curve construction"

# Or for fixes
git commit -m "fix: Correct OIS daily compounding calculation"
```

#### Commit Message Guidelines

Follow conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### 5. Keep Your Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch
git rebase upstream/main
```

### 6. Push Changes

```bash
# Push to your fork
git push origin feature/your-feature-name
```

---

## Contribution Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use type hints for function signatures
- Keep functions focused and concise
- Use descriptive variable names
- Maximum line length: 100 characters

#### Example

```python
from typing import Array
import jax.numpy as jnp


def calculate_forward_rate(
    spot_rate: float,
    dividend_yield: float,
    time_to_maturity: float,
    risk_free_rate: float
) -> float:
    """Calculate forward rate for equity derivatives.

    Args:
        spot_rate: Current spot price
        dividend_yield: Continuous dividend yield
        time_to_maturity: Time to maturity in years
        risk_free_rate: Risk-free interest rate

    Returns:
        Forward rate
    """
    return spot_rate * jnp.exp((risk_free_rate - dividend_yield) * time_to_maturity)
```

### JAX Best Practices

- Use JAX numpy (jnp) instead of numpy
- Write pure functions for JIT compilation
- Avoid in-place operations
- Use functional programming patterns
- Ensure functions are differentiable when needed

### Code Quality Tools

Run these tools before submitting:

```bash
# Format code with black
black src/neutryx tests

# Sort imports
isort src/neutryx tests

# Lint with ruff
ruff check src/neutryx tests

# Type checking (optional but recommended)
mypy src/neutryx

# Security scanning
bandit -r src/neutryx
```

---

## Testing Requirements

### Writing Tests

- All new features must include tests
- Tests should be in `src/neutryx/tests/`
- Use pytest framework
- Aim for high code coverage (>80%)
- Include edge cases and error conditions

#### Test Structure

```python
import pytest
import jax.numpy as jnp
from neutryx.products.equity import equity_forward_price


class TestEquityForward:
    """Test suite for equity forward pricing."""

    def test_basic_forward_pricing(self):
        """Test basic forward rate calculation."""
        spot = 100.0
        maturity = 1.0
        rate = 0.05
        dividend = 0.02

        forward = equity_forward_price(spot, maturity, rate, dividend)

        expected = spot * jnp.exp((rate - dividend) * maturity)
        assert jnp.allclose(forward, expected, rtol=1e-6)

    def test_zero_dividend(self):
        """Test forward pricing with zero dividend."""
        spot = 100.0
        maturity = 1.0
        rate = 0.05
        dividend = 0.0

        forward = equity_forward_price(spot, maturity, rate, dividend)

        expected = spot * jnp.exp(rate * maturity)
        assert jnp.allclose(forward, expected, rtol=1e-6)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            equity_forward_price(-100.0, 1.0, 0.05, 0.02)
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest src/neutryx/tests/products/test_equity.py -v

# Run with coverage
pytest --cov=neutryx --cov-report=html

# Run parallel tests
pytest -n auto

# Run only fast tests
pytest -m "fast"
```

### Test Markers

Use pytest markers to categorize tests:

- `@pytest.mark.fast` - Quick tests (<0.1s)
- `@pytest.mark.slow` - Long-running tests (>1s)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.performance` - Performance benchmarks

---

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def price_european_option(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    option_type: str = "call"
) -> float:
    """Price a European option using Black-Scholes formula.

    This function implements the analytical Black-Scholes formula for
    European call and put options. The formula assumes constant volatility,
    risk-free rate, and no dividends.

    Args:
        spot: Current underlying asset price
        strike: Option strike price
        maturity: Time to expiration in years
        volatility: Annualized volatility (implied or historical)
        rate: Risk-free interest rate (continuous compounding)
        option_type: Type of option, either "call" or "put"

    Returns:
        Option price in the same currency as spot and strike

    Raises:
        ValueError: If spot, strike, or volatility are negative
        ValueError: If option_type is not "call" or "put"

    Examples:
        >>> price = price_european_option(100.0, 100.0, 1.0, 0.2, 0.05, "call")
        >>> print(f"Call price: {price:.4f}")
        Call price: 10.4506

    References:
        Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate
        Liabilities. Journal of Political Economy, 81(3), 637-654.
    """
    # Implementation here
```

### Documentation Files

- Update README.md for user-facing changes
- Add examples in `examples/` directory
- Update API reference in `docs/api_reference.md`
- Create tutorials for complex features

---

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Add tests for new functionality
3. Update documentation
4. Run code quality tools
5. Rebase on latest main branch
6. Write clear commit messages

### Submitting a Pull Request

1. Push your branch to your fork
2. Go to the original repository on GitHub
3. Click "New Pull Request"
4. Select your branch
5. Fill out the PR template

#### PR Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing

Describe the tests you ran to verify your changes

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Coverage maintained or improved

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing

## Related Issues

Closes #(issue number)
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, maintainers will merge your PR
4. Your contribution will be included in the next release

---

## Community

### Communication Channels

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: dev@neutryx.tech for strategic discussions

### Reporting Bugs

Use GitHub Issues to report bugs:

1. Check if the issue already exists
2. Use the bug report template
3. Include:
   - Minimal reproducible example
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Error messages and stack traces

#### Bug Report Template

```markdown
## Bug Description

Clear description of the bug

## To Reproduce

Steps to reproduce the behavior:
1. Import module '...'
2. Call function '...'
3. See error

## Expected Behavior

What you expected to happen

## Actual Behavior

What actually happened

## Minimal Reproducible Example

```python
import jax.numpy as jnp
from neutryx.models.bs import price

# Code that reproduces the bug
result = price(...)
```

## Environment

- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- JAX version: [e.g., 0.4.26]
- Neutryx version: [e.g., 1.0.3]

## Additional Context

Any other relevant information
```

### Feature Requests

Use GitHub Issues for feature requests:

1. Check if the feature has been requested
2. Describe the use case and benefits
3. Provide examples if possible
4. Tag with "enhancement" label

---

## Contribution Areas

### High Priority

- Interest rate derivatives (swaps, swaptions)
- FX exotics and structured products
- Credit derivatives and CDOs
- FRTB and regulatory capital calculations
- Performance optimizations (GPU/TPU)
- Trading infrastructure (v0.3.0 completion)

### Medium Priority

- Additional stochastic models
- Market microstructure tools
- Portfolio optimization algorithms
- ML/AI pricing methods
- Advanced calibration techniques

### Always Welcome

- Documentation improvements
- Bug fixes and tests
- Examples and tutorials
- Performance benchmarks
- Code quality improvements

---

## Recognition

Contributors will be:

- Listed in the project's contributors file
- Mentioned in release notes for significant contributions
- Acknowledged in documentation for major features
- Invited to join the core team for sustained contributions

---

## Questions?

If you have questions about contributing:

- Check existing documentation
- Search GitHub Discussions
- Open a discussion thread
- Email: dev@neutryx.tech

---

**Thank you for contributing to Neutryx Core!**

Together, we're building the future of quantitative finance with JAX.
