# Fictional Bank Portfolio - User Guide

A comprehensive guide to using the Fictional Bank portfolio example for learning, testing, and demonstrating Neutryx capabilities.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Portfolio](#understanding-the-portfolio)
3. [Using the CLI](#using-the-cli)
4. [Running Individual Scripts](#running-individual-scripts)
5. [Understanding Outputs](#understanding-outputs)
6. [Common Workflows](#common-workflows)
7. [Customization Guide](#customization-guide)
8. [Best Practices](#best-practices)

## Getting Started

### Installation

1. **Navigate to the directory:**
   ```bash
   cd examples/applications/fictional_bank
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   ./cli.py status --check-deps
   ```

### First Steps

The easiest way to get started is using the CLI:

```bash
# View portfolio information
./cli.py info

# Check system status
./cli.py status

# Run the complete demo
./cli.py demo
```

## Understanding the Portfolio

### Portfolio Composition

The fictional portfolio represents a realistic trading desk with:

**Organizational Structure:**
- 1 Legal Entity: Global Investment Bank Ltd
- 1 Business Unit: Global Trading
- 3 Trading Desks
- 7 Books
- 6 Traders

**Counterparty Diversity:**
- 6 Counterparties across different sectors
- Mix of credit ratings (AAA to BBB)
- 67% CSA coverage (4 of 6 counterparties)

**Trade Mix:**
- 11 trades across 3 asset classes
- Interest Rates: IRS, Swaptions
- FX: Options on major and EM pairs
- Equity: Vanilla and exotic options, variance swaps

### Key Concepts

#### Netting Sets
Each counterparty has one netting set that combines:
- ISDA Master Agreement (legal framework)
- CSA (collateral terms, if applicable)
- All trades with that counterparty

#### CSA Impact
Counterparties with CSA agreements:
- Require collateral posting (bilateral)
- Reduce counterparty exposure
- Lower CVA/FVA charges

#### Book Hierarchy
Trades are organized by:
- **Desk**: Rates, FX, or Equity
- **Book**: Sub-desk trading book
- **Trader**: Individual responsible

## Using the CLI

### Basic Commands

#### View Information
```bash
# Portfolio overview
./cli.py info

# System status
./cli.py status

# Check API connectivity
./cli.py status --check-api

# Check dependencies
./cli.py status --check-deps
```

#### Load Portfolio
```bash
# Load and display summary
./cli.py load

# Save to custom directory
./cli.py load --output-dir my_snapshots
```

#### Generate Reports
```bash
# Generate all report formats
./cli.py report

# Specify output directory
./cli.py report --output-dir my_reports

# Generate specific format (not yet implemented)
./cli.py report --format html
```

#### Create Visualizations
```bash
# Generate all charts
./cli.py visualize

# Custom output directory
./cli.py visualize --output-dir my_charts
```

#### Run Stress Tests
```bash
# Run all stress scenarios
./cli.py stress

# Run specific category (future enhancement)
./cli.py stress --category rates
```

#### Compute Sensitivities
```bash
# Calculate Greeks and sensitivities
./cli.py sensitivity

# Custom output
./cli.py sensitivity --output-dir my_analysis
```

#### Compute XVA
```bash
# Requires API to be running
./cli.py xva

# Custom API URL
./cli.py xva --api-url http://custom-api:8000
```

#### Run Complete Demo
```bash
# Execute entire workflow
./cli.py demo
```

### CLI Help

```bash
# Show all commands
./cli.py --help

# Show command-specific help
./cli.py load --help
./cli.py report --help
```

## Running Individual Scripts

### Load Portfolio

```bash
./load_portfolio.py
```

**Output:**
- Console: Portfolio summary, counterparty breakdown, desk breakdown
- File: `snapshots/portfolio_snapshot.json`

**What it does:**
- Creates the fictional portfolio
- Displays comprehensive statistics
- Saves portfolio snapshot as JSON

### Compute XVA

```bash
# Start API first
uvicorn neutryx.api.rest:create_app --factory --reload

# In another terminal
./compute_xva.py
```

**Output:**
- Console: Portfolio and netting set XVA metrics
- File: `reports/xva_results.json`

**What it does:**
- Registers portfolio with Neutryx API
- Computes CVA, DVA, FVA, MVA
- Analyzes CSA impact on XVA

### Generate Reports

```bash
./portfolio_report.py
```

**Output:**
- `reports/portfolio_report_*.html` - Interactive HTML report
- `reports/portfolio_report_*.xlsx` - Multi-sheet Excel workbook
- `reports/portfolio_report_*.json` - Structured JSON data
- `reports/counterparties_*.csv` - Counterparty details
- `reports/books_*.csv` - Book breakdown
- `reports/desks_*.csv` - Desk summary

**What it does:**
- Generates comprehensive reports in multiple formats
- Includes XVA results if available
- Creates formatted HTML with styling

### Create Visualizations

```bash
./visualization.py
```

**Output:**
- `sample_outputs/charts/*.png` - All visualization charts

**Charts created:**
1. Counterparty exposure (bar chart)
2. Desk breakdown (pie + bar)
3. MTM distribution (histogram)
4. CSA comparison (bar + pie)
5. Rating distribution (bar chart)
6. XVA waterfall (if XVA available)
7. XVA by counterparty (stacked bar)
8. CSA impact (box plot)

### Run Stress Tests

```bash
./stress_testing.py
```

**Output:**
- `reports/stress_test_results_*.json` - Detailed results
- `reports/stress_test_results_*.csv` - Tabular data
- `reports/stress_test_results_*.xlsx` - Excel with multiple sheets

**Scenarios tested:**
- Interest Rate: +100bp, -50bp, steepener, flattener
- FX: USD strength/weakness, EM crisis
- Equity: Market crash, rally, sector crash
- Volatility: Spikes and crushes
- Credit: Spread widening, downgrades
- Combined: 2008 Crisis, COVID-19, Perfect Storm, etc.

### Compute Sensitivities

```bash
./sensitivity_analysis.py
```

**Output:**
- `reports/sensitivity_analysis_*.json` - Complete results
- `reports/greeks_*.xlsx` - Greeks by trade
- `reports/sensitivity_analysis_*.xlsx` - Multi-sheet analysis

**What it calculates:**
- Option Greeks (Delta, Gamma, Vega, Theta, Rho)
- IR sensitivities (PV01, DV01)
- FX sensitivities (Delta by pair)
- Equity sensitivities (Delta, Gamma by underlying)
- Vega by asset class
- Bucketed IR sensitivities by tenor

## Understanding Outputs

### Report Files

#### HTML Reports
- Professional formatting with CSS
- Embedded tables with hover effects
- Color-coded MTM (green=positive, red=negative)
- CSA status badges
- Responsive design

#### Excel Reports
- Multiple sheets for different views
- Formatted headers and data
- Formulas preserved for calculations
- Easy to analyze in Excel/Google Sheets

#### CSV Reports
- Raw data for custom analysis
- Import into databases or analytics tools
- Compatible with pandas, R, etc.

#### JSON Reports
- Structured data for APIs
- Programmatic access
- Version control friendly

### Visualization Charts

All charts are saved as high-resolution PNG files (300 DPI) suitable for:
- Presentations
- Reports
- Documentation
- Publications

**Chart Types:**
- **Bar Charts**: Comparing values across categories
- **Pie Charts**: Showing proportions
- **Histograms**: Distributions
- **Box Plots**: Statistical distributions
- **Stacked Charts**: Multi-component breakdowns
- **Waterfall**: Cumulative effects

### Snapshot Files

Portfolio snapshots (JSON) contain:
- Complete trade details
- Counterparty information
- Book hierarchy
- CSA terms
- Market data (if embedded)

Use for:
- Version control
- Backup
- Sharing portfolios
- Regression testing

## Common Workflows

### Workflow 1: Quick Portfolio Overview

```bash
# 1. Load and view portfolio
./cli.py load

# 2. Generate visual summary
./cli.py visualize

# 3. Review charts
open sample_outputs/charts/
```

### Workflow 2: Complete Risk Analysis

```bash
# 1. Run complete demo
./cli.py demo

# 2. Review HTML report
open reports/portfolio_report_*.html

# 3. Analyze stress tests
open reports/stress_test_results_*.xlsx

# 4. Review sensitivities
open reports/sensitivity_analysis_*.xlsx
```

### Workflow 3: XVA Analysis

```bash
# 1. Start API
uvicorn neutryx.api.rest:create_app --factory --reload

# 2. In another terminal, compute XVA
./cli.py xva

# 3. Generate report with XVA
./cli.py report

# 4. View XVA charts
./cli.py visualize

# 5. Review HTML report
open reports/portfolio_report_*.html
```

### Workflow 4: Custom Analysis

```bash
# 1. Load portfolio
./cli.py load

# 2. Run specific analyses
./stress_testing.py
./sensitivity_analysis.py

# 3. Generate custom reports
./portfolio_report.py

# 4. Analyze in Excel
open reports/*.xlsx
```

## Customization Guide

### Modifying Market Data

Edit `config.yaml`:

```yaml
market_data:
  rates:
    USD:
      curve_date: "2024-01-15"
      spot_rates:
        - tenor: "1Y"
          rate: 0.0485  # Modify rate here
```

### Adding Stress Scenarios

In `stress_testing.py`, add to `_define_scenarios()`:

```python
StressScenario(
    name="My_Custom_Scenario",
    description="Description of the scenario",
    category="combined",
    shocks={
        "USD_curve": "+200bps",
        "SPX": "-25%",
        # ... more shocks
    },
    severity="severe",
)
```

### Customizing Visualizations

In `visualization.py`, modify chart properties:

```python
# Change figure size
fig, ax = plt.subplots(figsize=(16, 10))  # Larger chart

# Change colors
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # Custom palette

# Add annotations
ax.annotate("Important point", xy=(x, y), ...)
```

### Adding Report Sections

In `portfolio_report.py`, add to HTML generation:

```python
html += """
<h2>My Custom Section</h2>
<p>Custom analysis content here...</p>
"""
```

## Best Practices

### Performance

1. **Large Portfolios**: Use batch processing for many trades
2. **API Calls**: Cache results when possible
3. **Visualizations**: Generate only needed charts
4. **Reports**: Use appropriate formats (CSV for large data)

### Organization

1. **Use descriptive filenames** with timestamps
2. **Organize outputs by date/project**
3. **Keep snapshots** for reproducibility
4. **Version control config** changes

### Workflow

1. **Start with CLI** for exploration
2. **Use individual scripts** for specific tasks
3. **Automate with run_all_demos.py** for testing
4. **Review outputs** systematically

### Development

1. **Test changes** on small portfolios first
2. **Validate outputs** against known results
3. **Document modifications** in comments
4. **Follow existing code style**

## Tips & Tricks

### Rapid Prototyping

```bash
# Quick check without full demo
./cli.py load && ./cli.py visualize
```

### Batch Processing

```bash
# Run multiple analyses in background
./stress_testing.py > stress.log 2>&1 &
./sensitivity_analysis.py > sensitivity.log 2>&1 &
```

### Data Export

```bash
# Extract specific data with jq
cat reports/xva_results.json | jq '.netting_set_xva[] | {counterparty, total_xva}'
```

### Quick Comparisons

```python
# Load multiple snapshots and compare
import json
with open('snapshots/snapshot1.json') as f:
    port1 = json.load(f)
with open('snapshots/snapshot2.json') as f:
    port2 = json.load(f)

# Compare MTMs, etc.
```

## Next Steps

After mastering this guide:

1. **Explore the code**: Review implementation details
2. **Read ARCHITECTURE.md**: Understand system design
3. **Check API_EXAMPLES.md**: Learn API integration
4. **Try TROUBLESHOOTING.md**: Solve common issues
5. **Contribute**: Add features or improvements

## Support

For additional help:
- Review inline code comments
- Check TROUBLESHOOTING.md
- Consult main Neutryx documentation
- Experiment with different configurations

---

**Happy analyzing!** 📊📈🎯
