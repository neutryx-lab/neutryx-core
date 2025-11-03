# Contributing to Neutryx Core

Thank you for your interest in contributing to Neutryx Core! We welcome contributions from the community and appreciate your help in making this project better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)
- [Getting Help](#getting-help)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

**In short: Be kind. Be curious. No harassment or discrimination.**

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/neutryx-core.git
   cd neutryx-core
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/neutryx-lab/neutryx-core.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (includes dev tools)
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

## Contribution Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Run the test suite**:
   ```bash
   pytest -v
   ```

5. **Run code quality checks**:
   ```bash
   # Linting
   ruff check src/ tests/

   # Format code
   ruff format src/ tests/

   # Type checking (if applicable)
   mypy src/
   ```

6. **Commit your changes** with clear messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

7. **Keep your fork updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

8. **Push to your fork**:
   ```bash
   git push origin feature/my-feature-name
   ```

9. **Open a Pull Request** on GitHub

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **Google-style docstrings**
- Maximum line length: **100 characters**
- Use **type hints** where appropriate
- Code is automatically formatted with **ruff** and **black**

### Code Organization

- Place new modules in the appropriate directory (see README.md directory structure)
- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Avoid premature optimization

### Example Docstring

```python
def price_option(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate European call option price using Black-Scholes formula.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility

    Returns:
        Option price as a float

    Raises:
        ValueError: If any parameter is negative or invalid

    Examples:
        >>> price_option(100.0, 100.0, 1.0, 0.05, 0.2)
        10.450583572185565
    """
    # Implementation here
    pass
```

## Testing Guidelines

### Test Organization

- Place tests in `src/neutryx/tests/`
- Mirror the package structure
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`

### Test Categories

Mark tests with appropriate pytest markers:

```python
import pytest

@pytest.mark.regression
def test_bs_price_regression():
    """Regression test for Black-Scholes pricing."""
    pass

@pytest.mark.performance
def test_mc_performance():
    """Performance benchmark for Monte Carlo."""
    pass
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest src/neutryx/tests/test_bs_analytic.py

# Run tests with specific marker
pytest -m regression

# Run with coverage
pytest --cov=neutryx --cov-report=html
```

### Test Requirements

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows
- **Regression tests**: Ensure numerical stability
- **Performance tests**: Benchmark critical paths
- Aim for **>80% code coverage** for new code

## Documentation

### Updating Documentation

- Update docstrings for all public APIs
- Add examples to demonstrate usage
- Update README.md if adding major features
- Create or update relevant docs in `docs/` directory
- Add tutorial notebooks for complex features

### Documentation Style

- Use **Markdown** for documentation files
- Include **code examples** where helpful
- Explain the **"why"** not just the "what"
- Link to related functions and concepts

## Commit Message Guidelines

### Format

```
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Build process or auxiliary tool changes

### Example

```
feat: Add SABR model calibration

Implement differentiable SABR calibration using JAX autodiff.
Includes parameter bounds, diagnostics, and convergence checks.

Closes #123
```

## Pull Request Process

1. **Ensure all tests pass** and code quality checks succeed
2. **Update documentation** as needed
3. **Add a clear PR description**:
   - What changes were made?
   - Why were they made?
   - How were they tested?
4. **Link related issues** using "Closes #123" or "Fixes #456"
5. **Request review** from maintainers
6. **Address review comments** promptly
7. **Squash commits** if requested before merging

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines (ruff/black)
- [ ] Docstrings added/updated
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] No breaking changes (or clearly documented)

## Areas for Contribution

We welcome contributions in the following areas:

### High Priority

- **Bug fixes**: Help us fix reported issues
- **Test coverage**: Expand test suite
- **Documentation**: Improve guides and examples
- **Performance**: Optimize critical code paths

### Feature Additions

- **New pricing models**: Kou, Variance Gamma, rough volatility
- **Exotic options**: Quanto, Cliquet, Rainbow, baskets
- **Advanced calibration**: Joint calibration, regularization
- **Integration**: Data provider connections, cloud deployment

### Research & Experiments

- **Deep learning pricing**: Model-free approaches
- **Quantum computing**: Variational pricing experiments
- **Causal inference**: Risk attribution methods

## Getting Help

### Questions & Discussion

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs or request features
- **Email**: Contact `dev@neutryx.tech` for private inquiries

### Resources

- [README.md](README.md) - Project overview and quickstart
- [docs/](docs/) - Detailed documentation
- [examples/](examples/) - Code examples and tutorials
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Recognition

Contributors will be recognized in:

- Git commit history
- Release notes
- Project acknowledgments

Thank you for contributing to Neutryx Core! 🚀

---

**Questions?** Open an issue or start a discussion on GitHub.

Built with ❤️ by the Neutryx community
