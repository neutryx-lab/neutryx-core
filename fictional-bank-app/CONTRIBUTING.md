# Contributing to Fictional Bank Example

Thank you for your interest in contributing! This guide will help you extend and improve the Fictional Bank portfolio example.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Adding Features](#adding-features)
7. [Testing Guidelines](#testing-guidelines)
8. [Documentation](#documentation)
9. [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git for version control
- Familiarity with financial derivatives (helpful but not required)
- Understanding of XVA concepts (helpful but not required)

### First Steps

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/neutryx-core.git
   cd neutryx-core/examples/applications/fictional_bank
   ```

3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

## Development Setup

### Virtual Environment

Always use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### IDE Configuration

**VS Code** (recommended settings):
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

**PyCharm**:
- Set Python interpreter to virtual environment
- Enable PEP 8 code style
- Configure Black as external tool

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix errors in existing code
2. **New Features**: Add new analytics or capabilities
3. **Documentation**: Improve or add documentation
4. **Tests**: Add test coverage
5. **Examples**: Add new use cases or scenarios
6. **Performance**: Optimize existing code

### Finding Issues

- Look for issues labeled `good first issue`
- Check `help wanted` issues
- Review TODO comments in code
- Suggest new features via issues

## Coding Standards

### Python Style Guide

Follow **PEP 8** with these specifics:

**Line Length**: 100 characters (not 79)

**Imports**:
```python
# Standard library
import sys
from pathlib import Path
from typing import Any, Dict, List

# Third-party
import pandas as pd
import numpy as np

# Local
from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio
```

**Naming Conventions**:
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names, avoid abbreviations

**Type Hints**:
```python
def calculate_xva(
    portfolio_id: str,
    netting_set_id: Optional[str] = None
) -> Dict[str, float]:
    """Calculate XVA metrics."""
    pass
```

**Docstrings**:
```python
def my_function(param1: str, param2: int) -> bool:
    """Short one-line description.

    Longer description if needed, explaining the function's
    behavior in more detail.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> my_function("test", 5)
        True
    """
    pass
```

### Code Quality Tools

**Black** (code formatter):
```bash
black *.py
```

**Flake8** (linter):
```bash
flake8 *.py --max-line-length=100
```

**MyPy** (type checker):
```bash
mypy *.py --ignore-missing-imports
```

### Git Commit Messages

Format:
```
type: Short description (50 chars max)

Longer explanation if needed (wrap at 72 characters).
Explain what and why, not how.

Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `style`: Formatting changes
- `chore`: Maintenance tasks

**Examples**:
```
feat: Add PFE analysis script

Implement potential future exposure calculation including:
- Expected Exposure (EE)
- Expected Positive Exposure (EPE)
- PFE at multiple confidence levels
- Visualization of exposure profiles

Closes #45

---

fix: Correct MTM aggregation in portfolio summary

MTM was being double-counted for netting sets with multiple trades.
Now correctly sums net MTM per netting set.

Fixes #67

---

docs: Update USER_GUIDE with stress testing examples

Add detailed walkthrough of stress testing workflow including:
- Scenario selection
- Result interpretation
- Custom scenario creation
```

## Adding Features

### Adding a New Analytics Script

1. **Create the script**:
   ```bash
   touch my_new_analysis.py
   chmod +x my_new_analysis.py
   ```

2. **Use the standard template**:
   ```python
   #!/usr/bin/env python3
   """Description of the new analysis.

   This script demonstrates/calculates...
   """
   import sys
   from pathlib import Path
   from typing import Any, Dict

   # Add project root to path
   project_root = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(project_root / "src"))

   from neutryx.tests.fixtures.fictional_portfolio import (
       create_fictional_portfolio,
       get_portfolio_summary,
   )


   class MyAnalyzer:
       """Analyzer for my new analysis."""

       def __init__(self, output_dir: Path):
           """Initialize analyzer.

           Args:
               output_dir: Directory to save results
           """
           self.output_dir = Path(output_dir)
           self.output_dir.mkdir(parents=True, exist_ok=True)

       def analyze(self, portfolio: Any) -> Dict:
           """Perform analysis.

           Args:
               portfolio: Portfolio to analyze

           Returns:
               Analysis results
           """
           results = {}
           # Your analysis logic here
           return results


   def main():
       """Main entry point."""
       print("=" * 80)
       print("My New Analysis")
       print("=" * 80)
       print()

       # Load portfolio
       portfolio, book_hierarchy = create_fictional_portfolio()
       print(f"Portfolio loaded: {portfolio.name}")

       # Run analysis
       analyzer = MyAnalyzer(Path("reports"))
       results = analyzer.analyze(portfolio)

       # Display results
       print(results)


   if __name__ == "__main__":
       main()
   ```

3. **Add CLI command** in `cli.py`:
   ```python
   @cli.command()
   def my_analysis():
       """Run my new analysis."""
       import subprocess
       script_path = Path(__file__).parent / "my_new_analysis.py"
       subprocess.run([sys.executable, str(script_path)])
   ```

4. **Update documentation**:
   - Add to README.md features list
   - Document in USER_GUIDE.md
   - Update ARCHITECTURE.md if needed

### Adding a New Stress Scenario

Edit `stress_testing.py`:

```python
def _define_scenarios(self) -> List[StressScenario]:
    scenarios = []

    # ... existing scenarios ...

    # Add your new scenario
    scenarios.append(
        StressScenario(
            name="MY_CustomScenario",
            description="Description of what this scenario tests",
            category="combined",  # or "rates", "fx", "equity", etc.
            shocks={
                "USD_curve": "+150bps",
                "SPX": "-25%",
                "credit_spreads": "+200bps",
            },
            severity="severe",  # "mild", "moderate", "severe", "extreme"
        )
    )

    return scenarios
```

### Adding a New Visualization

Edit `visualization.py`:

```python
def plot_my_new_chart(self, data: Dict) -> Path:
    """Create my new chart.

    Args:
        data: Data to visualize

    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Your plotting code here
    # ...

    ax.set_xlabel("X Label", fontweight="bold")
    ax.set_ylabel("Y Label", fontweight="bold")
    ax.set_title("Chart Title", fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = self.output_dir / "my_new_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file
```

Then add to `create_all_charts()`:

```python
def create_all_charts(self, portfolio, book_hierarchy, xva_results=None):
    # ... existing charts ...

    chart = self.plot_my_new_chart(data)
    charts["my_new_chart"] = chart
    print(f"✓ My new chart")

    return charts
```

## Testing Guidelines

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
from my_feature import MyFeature


class TestMyFeature:
    """Test suite for MyFeature."""

    def test_basic_functionality(self):
        """Test basic feature works."""
        feature = MyFeature()
        result = feature.process(input_data)
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        feature = MyFeature()
        with pytest.raises(ValueError):
            feature.process(invalid_data)

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test multiple input cases."""
        feature = MyFeature()
        assert feature.double(input) == expected
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_portfolio_loading.py

# Run with coverage
pytest --cov=. tests/

# Run with verbose output
pytest -v tests/
```

### Test Coverage

Aim for:
- **New features**: 80%+ coverage
- **Bug fixes**: Add test for the bug
- **Critical paths**: 100% coverage

## Documentation

### Docstring Requirements

Every public function/class needs:
- One-line summary
- Detailed description (if complex)
- Args with types and descriptions
- Returns description
- Raises (if applicable)
- Example usage (for complex functions)

### Updating Documentation

When adding features, update:
1. **README.md**: Overview and feature list
2. **USER_GUIDE.md**: Usage instructions
3. **API_EXAMPLES.md**: If API-related
4. **ARCHITECTURE.md**: If architectural changes
5. **Inline comments**: For complex logic

### Documentation Style

- Use clear, simple language
- Include examples
- Explain "why" not just "what"
- Keep examples up to date
- Use proper markdown formatting

## Pull Request Process

### Before Submitting

1. **Run tests**: `pytest tests/`
2. **Format code**: `black *.py`
3. **Check linting**: `flake8 *.py`
4. **Update documentation**
5. **Test manually**: Run the script/feature
6. **Check all files**: No debug code, print statements, etc.

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Test coverage
- [ ] Performance improvement

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing completed

## Documentation
- [ ] README updated
- [ ] Docstrings added/updated
- [ ] USER_GUIDE updated (if needed)

## Screenshots (if applicable)
Add screenshots of new features/charts

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] No console.log/debug statements
- [ ] Documentation is clear
```

### Review Process

1. Submit PR with clear description
2. Address reviewer feedback
3. Keep PR focused (one feature/fix per PR)
4. Be responsive to comments
5. Update PR based on feedback

### After Merge

- Delete your branch
- Update your fork
- Celebrate! 🎉

## Questions?

- Open an issue for questions
- Check existing documentation first
- Be specific about your question
- Include relevant code/error messages

---

Thank you for contributing to make this example better! 🙏
