# Advanced Stress Testing Guide

## 📖 Overview

This guide covers the enhanced stress testing capabilities of the Fictional Bank portfolio, including:

- **Custom scenario creation** - Build your own market shock scenarios
- **Scenario persistence** - Save and load scenarios from YAML/JSON files
- **Sensitivity analysis** - Analyze portfolio response to shock ranges
- **Advanced visualizations** - Heatmaps, tornado charts, interactive dashboards
- **Comprehensive reporting** - Detailed impact analysis across scenarios

## 🚀 Quick Start

### Running the Demo

The fastest way to see all features in action:

```bash
cd examples/applications/fictional_bank
python advanced_stress_testing_demo.py
```

This will:
1. Load the portfolio
2. Run predefined scenarios
3. Create custom scenarios
4. Load scenarios from YAML files
5. Perform sensitivity analysis
6. Generate comprehensive reports and visualizations

### Basic Usage

```python
from pathlib import Path
from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio
from stress_testing import StressTester

# Load portfolio
portfolio, book_hierarchy = create_fictional_portfolio()

# Initialize stress tester
output_dir = Path("reports")
tester = StressTester(output_dir)

# Run all predefined scenarios
results = tester.run_all_stress_tests(portfolio, book_hierarchy)

# Generate reports
tester.generate_stress_report(results)
```

## 📝 Creating Custom Scenarios

### Method 1: Programmatically

Create scenarios directly in Python:

```python
# Create a custom crisis scenario
tester.create_custom_scenario(
    name="My_Custom_Crisis",
    description="Custom market crisis scenario",
    shocks={
        "USD_curve": "+150bps",
        "EUR_curve": "+100bps",
        "SPX": "-25%",
        "EURUSD": "+10%",
        "FX_vols": "+75%"
    },
    category="custom",
    severity="severe",
    save=True  # Automatically saves to YAML
)
```

### Method 2: YAML Files

Create a YAML file in `data/scenarios/`:

```yaml
name: My_Custom_Scenario
description: Description of what this scenario tests
category: custom  # or rates, fx, equity, volatility, credit, combined
severity: moderate  # mild, moderate, severe, extreme
shocks:
  USD_curve: "+100bps"
  SPX: "-15%"
  EURUSD: "+5%"
  FX_vols: "+50%"
```

Load scenarios from file:

```python
# Load specific file
tester.load_custom_scenarios(Path("data/scenarios/my_scenarios.yaml"))

# Or load all scenarios from directory
tester.load_custom_scenarios()  # Loads all .yaml/.yml/.json files
```

### Method 3: JSON Files

Create a JSON file with one or more scenarios:

```json
[
  {
    "name": "Custom_Scenario_1",
    "description": "First custom scenario",
    "category": "custom",
    "severity": "moderate",
    "shocks": {
      "USD_curve": "+100bps",
      "SPX": "-20%"
    }
  },
  {
    "name": "Custom_Scenario_2",
    "description": "Second custom scenario",
    "category": "custom",
    "severity": "severe",
    "shocks": {
      "EURUSD": "+15%",
      "credit_spreads": "+200bps"
    }
  }
]
```

## 🎯 Shock Syntax Reference

### Interest Rate Shocks
```yaml
shocks:
  USD_curve: "+100bps"    # Parallel shift up 100 basis points
  EUR_curve: "-50bps"     # Parallel shift down 50 basis points
  USD_short: "+50bps"     # Short end up 50bps
  USD_long: "-25bps"      # Long end down 25bps
```

### Equity Shocks
```yaml
shocks:
  SPX: "-20%"     # S&P 500 down 20%
  AAPL: "+30%"    # Apple up 30%
  TSLA: "-45%"    # Tesla down 45%
```

### FX Shocks
```yaml
shocks:
  EURUSD: "+10%"   # Euro strengthens 10% vs USD
  USDJPY: "-15%"   # Yen strengthens 15% vs USD
  USDBRL: "+25%"   # Dollar strengthens 25% vs BRL
```

### Volatility Shocks
```yaml
shocks:
  FX_vols: "+50%"   # FX volatility up 50%
  EQ_vols: "+100%"  # Equity volatility up 100%
  IR_vols: "-30%"   # IR volatility down 30%
```

### Credit Shocks
```yaml
shocks:
  credit_spreads: "+150bps"  # Credit spreads widen 150bps
  rating_migration: "-1"     # One notch downgrade
```

## 📊 Scenario Analysis

### Running Specific Scenarios

```python
# Run a single scenario
scenario = tester.scenarios[0]  # Get first scenario
result = tester.run_scenario(scenario, portfolio, book_hierarchy)

# Run only custom scenarios
custom_results = tester.run_custom_stress_tests(portfolio, book_hierarchy)

# Run both predefined and custom
all_results = tester.run_custom_stress_tests(
    portfolio, book_hierarchy, include_predefined=True
)
```

### Comparing Scenarios

```python
# Compare specific scenarios
comparison = tester.compare_scenarios(
    portfolio,
    book_hierarchy,
    scenario_names=[
        "IR_ParallelShift_Up100",
        "EQ_MarketCrash_20",
        "COMB_2008Crisis",
        "My_Custom_Scenario"
    ]
)

print(comparison)
```

## 🔬 Sensitivity Analysis

Analyze portfolio response to a range of shock values:

```python
# Equity sensitivity
equity_sens = tester.analyze_shock_sensitivity(
    portfolio,
    book_hierarchy,
    shock_type="SPX",
    shock_range=["-30%", "-20%", "-10%", "0", "+10%", "+20%", "+30%"]
)

# Rate sensitivity
rate_sens = tester.analyze_shock_sensitivity(
    portfolio,
    book_hierarchy,
    shock_type="USD_curve",
    shock_range=["-100bps", "-50bps", "0", "+50bps", "+100bps", "+200bps"]
)

# FX sensitivity
fx_sens = tester.analyze_shock_sensitivity(
    portfolio,
    book_hierarchy,
    shock_type="EURUSD",
    shock_range=["-15%", "-10%", "-5%", "0", "+5%", "+10%", "+15%"]
)

# Results are returned as pandas DataFrame
print(equity_sens)
```

## 📈 Visualizations

### Static Charts (Matplotlib/Seaborn)

Create comprehensive static visualizations:

```python
from stress_test_visualization import StressTestVisualizer

visualizer = StressTestVisualizer(output_dir=Path("sample_outputs/charts"))

# Create all visualizations
charts = visualizer.create_all_visualizations(results)

# Create specific visualizations
heatmap = visualizer.plot_impact_heatmap(results)
tornado = visualizer.plot_tornado_chart(results)
comparison = visualizer.plot_scenario_comparison(results)
severity = visualizer.plot_severity_distribution(results)
category = visualizer.plot_category_impact(results)

# Sensitivity analysis charts
sensitivity_chart = visualizer.plot_sensitivity_analysis(
    sensitivity_df, shock_type="SPX"
)
```

#### Available Static Charts

1. **Impact Heatmap** - Color-coded view of all scenarios
2. **Tornado Chart** - Top N scenarios by absolute impact
3. **Scenario Comparison** - Grouped by category with severity colors
4. **Severity Distribution** - Pie chart and box plots by severity level
5. **Category Impact** - Average, min, max impact by category
6. **Sensitivity Analysis** - Line and bar charts for shock ranges

### Interactive Charts (Plotly)

Create interactive HTML dashboards:

```python
# Create interactive dashboard
dashboard = visualizer.create_interactive_dashboard(results)
# Opens in browser: stress_interactive_dashboard.html

# Create 3D risk surface
surface_3d = visualizer.create_3d_risk_surface(results)
# Opens in browser: stress_3d_risk_surface.html
```

#### Interactive Features

- **Hover details** - View exact values by hovering over charts
- **Zoom and pan** - Interactive exploration of data
- **Export** - Save charts as PNG images
- **Filtering** - Click legend items to show/hide data series

## 📄 Reports

### Standard Report

```python
# Generate comprehensive report (JSON, CSV, Excel)
report_file = tester.generate_stress_report(results)
```

Generated reports include:
- **JSON**: Structured data with all details
- **CSV**: Flat file for analysis in Excel/Python
- **Excel**: Multi-sheet workbook with:
  - Summary statistics
  - All scenarios
  - Breakdowns by category

### Report Contents

- Total scenarios executed
- Category and severity breakdowns
- Worst and best scenarios
- Average impact metrics
- Detailed results for each scenario:
  - Baseline MTM
  - Stressed MTM
  - P&L impact (absolute and %)
  - Shock parameters

## 🎨 Example Scenarios

### Market Crash Scenarios

```yaml
# Tech sector crash
name: Tech_Sector_Crash
shocks:
  AAPL: "-40%"
  TSLA: "-50%"
  SPX: "-15%"
  EQ_vols: "+80%"

# Flash crash
name: Flash_Crash
shocks:
  SPX: "-25%"
  EQ_vols: "+150%"
  credit_spreads: "+200bps"
```

### Currency Crisis Scenarios

```yaml
# Emerging market crisis
name: EM_Crisis
shocks:
  USDBRL: "+40%"
  FX_vols: "+120%"
  credit_spreads: "+250bps"

# Dollar strength
name: Strong_Dollar
shocks:
  EURUSD: "-12%"
  USDJPY: "+15%"
  GBPUSD: "-10%"
```

### Interest Rate Scenarios

```yaml
# Hawkish Fed
name: Hawkish_Fed
shocks:
  USD_curve: "+200bps"
  SPX: "-12%"
  credit_spreads: "+100bps"

# Yield curve inversion
name: Curve_Inversion
shocks:
  USD_short: "+150bps"
  USD_long: "-50bps"
```

### Combined Scenarios

```yaml
# Stagflation
name: Stagflation
shocks:
  USD_curve: "+200bps"
  EUR_curve: "+150bps"
  SPX: "-15%"
  credit_spreads: "+150bps"
  FX_vols: "+60%"

# Perfect storm
name: Perfect_Storm
shocks:
  USD_curve: "+150bps"
  SPX: "-30%"
  EURUSD: "+15%"
  credit_spreads: "+300bps"
  FX_vols: "+100%"
  EQ_vols: "+120%"
```

## 🔧 Advanced Usage

### Custom Impact Calculation

Override the default impact simulation:

```python
# Extend StressTester class
class CustomStressTester(StressTester):
    def _simulate_stress_impact(self, baseline_mtm, scenario, summary):
        # Your custom logic here
        # Could integrate with real pricing models
        # Could use historical correlations
        # Could call external APIs
        return stressed_mtm
```

### Batch Scenario Generation

Generate multiple related scenarios:

```python
# Generate rate scenarios programmatically
for shock in ["+50bps", "+100bps", "+150bps", "+200bps"]:
    tester.create_custom_scenario(
        name=f"Rate_Shock_{shock}",
        description=f"Parallel rate shift {shock}",
        shocks={"USD_curve": shock, "EUR_curve": shock},
        category="rates",
        severity="moderate"
    )
```

### Scenario Libraries

Organize scenarios by theme:

```
data/scenarios/
├── crisis_scenarios.yaml      # Historical crisis replays
├── regulatory_scenarios.yaml  # CCAR/regulatory scenarios
├── market_risk_scenarios.yaml # Standard market shocks
└── custom_scenarios.yaml      # Your custom scenarios
```

## 📊 Best Practices

### Scenario Design

1. **Start Simple** - Begin with single-factor shocks
2. **Build Complexity** - Combine factors based on correlations
3. **Use Historical Events** - Model real crises (2008, COVID, etc.)
4. **Consider Extremes** - Test tail risk scenarios
5. **Document Assumptions** - Clear descriptions in YAML files

### Testing Strategy

1. **Baseline Tests** - Standard market shocks
2. **Sensitivity Analysis** - Map response curves
3. **Scenario Comparison** - Understand relative impacts
4. **Custom Scenarios** - Test specific concerns
5. **Regular Updates** - Refresh with current market data

### Reporting

1. **Executive Summary** - Top 5 worst scenarios
2. **Category Analysis** - Impact by risk type
3. **Drill-Down** - Detailed scenario results
4. **Visualizations** - Charts for presentations
5. **Action Items** - Risk mitigation recommendations

## 🎯 Use Cases

### Risk Management
- Daily risk limits monitoring
- VaR backtesting scenarios
- Stress capital buffer calculation
- Risk appetite framework validation

### Regulatory
- CCAR/DFAST scenarios
- Reverse stress testing
- Recovery and resolution planning
- Internal capital adequacy assessment

### Trading
- Position sizing decisions
- Hedge effectiveness analysis
- Strategy backtesting
- Portfolio optimization

### Research
- Market regime analysis
- Factor sensitivity studies
- Model validation
- Historical event studies

## 🔍 Troubleshooting

### Common Issues

**Scenarios not loading**
- Check YAML syntax (use YAML validator)
- Verify file is in `data/scenarios/` directory
- Ensure shock syntax is correct

**Visualizations failing**
- Install plotly: `pip install plotly`
- Check matplotlib backend
- Verify output directory exists

**Results seem incorrect**
- Note: This demo uses simplified impact models
- For production, integrate with proper pricing models
- Consider using Neutryx API for accurate calculations

## 📚 Additional Resources

- [Main README](README.md) - Overview of fictional_bank
- [USER_GUIDE.md](USER_GUIDE.md) - General usage guide
- [API_EXAMPLES.md](API_EXAMPLES.md) - Neutryx API integration
- Example scenarios: `data/scenarios/example_custom_scenarios.yaml`
- Demo script: `advanced_stress_testing_demo.py`

## 💡 Tips

1. **Version Control** - Track your custom scenarios in git
2. **Naming Conventions** - Use descriptive, consistent names
3. **Severity Levels** - Be consistent in severity classification
4. **Documentation** - Write clear scenario descriptions
5. **Validation** - Compare results against known benchmarks
6. **Automation** - Schedule regular stress test runs
7. **Alerts** - Set up notifications for breach scenarios
8. **Review** - Regularly update scenario library

---

**Questions or Issues?**

- Review the troubleshooting section
- Check example files for reference
- Consult the main Neutryx documentation
