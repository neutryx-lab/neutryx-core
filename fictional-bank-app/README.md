# Fictional Bank Portfolio - Comprehensive Neutryx Showcase

A production-quality demonstration of Neutryx's XVA and risk analytics capabilities, featuring a realistic multi-desk trading portfolio with comprehensive analysis tools.

## 🌟 Overview

This example showcases a complete enterprise-grade risk management system for a fictional global investment bank's trading portfolio. It demonstrates Neutryx's capabilities across:

- **XVA Calculations** (CVA, DVA, FVA, MVA)
- **Portfolio Risk Analytics**
- **Stress Testing**
- **Sensitivity Analysis (Greeks)**
- **Comprehensive Reporting**
- **Interactive Visualizations**
- **Collateral Management**

## 📊 Portfolio Structure

### Organization Hierarchy
- **Legal Entity**: Global Investment Bank Ltd (GB)
- **Business Unit**: Global Trading
- **Desks**:
  - Interest Rates Desk (3 books)
  - Foreign Exchange Desk (2 books)
  - Equity Derivatives Desk (2 books)
- **Traders**: 6 professionals across all desks
- **Total Books**: 7 trading books

### Counterparty Profile

| Counterparty | Rating | Type | CSA | Products |
|-------------|--------|------|-----|----------|
| AAA Global Bank | AAA | Financial | ✓ | USD IRS |
| Tech Corporation A | A | Corporate | ✓ | FX Options |
| Industrial Group BBB | BBB | Corporate | ✗ | EUR IRS |
| Alpha Strategies Fund | A- | Hedge Fund | ✗ | FX EM, EQ Exotics |
| Republic Investment Authority | AA+ | Sovereign | ✓ | EQ Vanilla Options |
| Global Insurance Group | AA | Financial | ✓ | Swaptions, Var Swaps |

### Trade Statistics
- **Total Trades**: 11 across multiple asset classes
- **Product Mix**:
  - Interest Rate Swaps: 3
  - Swaptions: 1
  - FX Options: 3
  - Equity Options: 5
  - Variance Swaps: 1
- **Total Notional**: ~USD 152M
- **CSA Coverage**: 4 of 6 counterparties (67%)

## 🏗️ Architecture

```
fictional_bank/
├── Core Scripts
│   ├── load_portfolio.py                 # Portfolio loader with summary
│   ├── compute_xva.py                    # XVA calculation via API
│   ├── standalone_xva_demo.py            # Standalone XVA calculator (NEW!)
│   ├── xva_dashboard.py                  # Interactive XVA dashboard (NEW!)
│   ├── portfolio_report.py               # Multi-format reporting
│   ├── visualization.py                  # Chart generation
│   ├── stress_testing.py                 # Enhanced stress testing framework
│   ├── stress_test_visualization.py      # Advanced stress test charts
│   ├── advanced_stress_testing_demo.py   # Comprehensive stress test demo
│   ├── sensitivity_analysis.py           # Greeks & sensitivities
│   ├── compliance_monitoring.py          # Compliance & limit monitoring
│   ├── compliance_dashboard.py           # Compliance dashboards
│   ├── compliance_demo.py                # Compliance monitoring demo
│   ├── timeseries_analysis.py            # Time series analysis & forecasting (NEW!)
│   ├── timeseries_visualization.py       # Time series visualizations (NEW!)
│   └── timeseries_demo.py                # Time series analysis demo (NEW!)
│
├── CLI & Automation
│   ├── cli.py                   # Interactive command-line interface
│   └── run_all_demos.py         # Master demo orchestrator
│
├── Configuration
│   ├── config.yaml              # Market data & XVA parameters
│   └── requirements.txt         # Python dependencies
│
├── Output Directories
│   ├── reports/                 # Generated reports (JSON, CSV, Excel, HTML)
│   ├── snapshots/               # Portfolio snapshots
│   ├── sample_outputs/charts/   # Visualization outputs
│   ├── data/scenarios/          # Custom stress test scenarios (YAML/JSON)
│   ├── data/compliance/         # Compliance limit configurations (YAML) (NEW!)
│   └── templates/               # HTML report templates
│
└── Documentation
    ├── README.md                   # This file
    ├── USER_GUIDE.md               # Detailed user guide
    ├── STRESS_TESTING_GUIDE.md     # Advanced stress testing guide
    ├── API_EXAMPLES.md             # API usage examples
    ├── ARCHITECTURE.md             # System architecture
    └── TROUBLESHOOTING.md          # Common issues & solutions
```

## 🚀 Quick Start

### Prerequisites

1. **Python Environment** (3.10+)
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Install Dependencies**
   ```bash
   cd examples/applications/fictional_bank
   pip install -r requirements.txt
   ```

3. **Start Neutryx API** (Optional for XVA calculations)
   ```bash
   uvicorn neutryx.api.rest:create_app --factory --reload
   ```

### Basic Usage

#### Option 1: Use the CLI (Recommended)

```bash
# View portfolio information
./cli.py info

# Load and display portfolio
./cli.py load

# Generate all reports
./cli.py report

# Create visualizations
./cli.py visualize

# Run stress tests
./cli.py stress

# Compute sensitivities
./cli.py sensitivity

# Run complete demo workflow
./cli.py demo

# Check system status
./cli.py status --check-api --check-deps
```

#### Option 2: Run Individual Scripts

```bash
# 1. Load portfolio
./load_portfolio.py

# 2. Compute XVA (requires API)
./compute_xva.py

# 3. Generate reports
./portfolio_report.py

# 4. Create visualizations
./visualization.py

# 5. Run stress tests
./stress_testing.py

# 6. Run advanced stress testing (NEW! - Includes custom scenarios & visualizations)
./advanced_stress_testing_demo.py

# 7. Run compliance monitoring (NEW! - Limit checks & dashboards)
./compliance_demo.py

# 8. Compute sensitivities
./sensitivity_analysis.py

# 9. Run standalone XVA calculator (NEW! - No API required)
./standalone_xva_demo.py

# 10. Generate interactive XVA dashboard (NEW!)
./xva_dashboard.py
```

#### Option 3: Execute Complete Demo

```bash
# Run all analyses in sequence
./run_all_demos.py
```

## 💰 NEW: Standalone XVA Calculator

Comprehensive XVA analytics without requiring the API! Calculate CVA, DVA, FVA, and MVA using Monte Carlo simulation:

### Key Features

**XVA Components:**
- **CVA** (Credit Valuation Adjustment): Counterparty credit risk charge
- **DVA** (Debit Valuation Adjustment): Own-credit adjustment
- **FVA** (Funding Valuation Adjustment): Funding cost of uncollateralized exposure
- **MVA** (Margin Valuation Adjustment): Cost of posting initial margin

**Advanced Analytics:**
- Monte Carlo simulation (1000+ paths) for exposure profiles
- Expected Positive Exposure (EPE) calculation
- Peak Exposure estimation
- CSA impact analysis (70% exposure reduction)
- Credit rating-based default probabilities
- Time-dependent exposure profiles over 5-year horizon

**Visualizations:**
- **XVA Waterfall**: Component breakdown visualization
- **Exposure Profiles**: Time series of expected positive exposure by counterparty
- **XVA by Counterparty**: Stacked bar charts showing component breakdown
- **CSA Impact**: Comparison of collateralized vs. uncollateralized positions

**Interactive Dashboard:**
- Plotly-based HTML dashboard with drill-down capabilities
- Real-time metrics indicators
- Interactive heatmaps and charts
- Export-ready for presentations

**Reporting:**
- Comprehensive Excel workbooks with multiple sheets
- JSON reports for API integration
- Summary metrics and detailed breakdowns
- Netting set-level analysis

### Quick Start

```bash
# Run standalone XVA calculator
./standalone_xva_demo.py

# Generate interactive dashboard
./xva_dashboard.py

# Open the dashboard in your browser
# File: sample_outputs/xva_dashboard.html
```

### Sample Output

```
Portfolio XVA Summary:
  Total CVA:  $      4,397.14
  Total DVA:  $          0.00  (Benefit)
  Total FVA:  $     35,739.29
  Total MVA:  $      3,573.93
  ────────────────────────────
  Total XVA:  $     43,710.37
```

## 🔥 NEW: Advanced Stress Testing Features

We've significantly enhanced the stress testing framework! Key new capabilities:

### Custom Scenario Creation
Create your own stress scenarios using YAML/JSON files or programmatically:

```python
# Create custom scenario
tester.create_custom_scenario(
    name="My_Custom_Crisis",
    description="Custom market crisis scenario",
    shocks={
        "USD_curve": "+150bps",
        "SPX": "-25%",
        "FX_vols": "+75%"
    },
    severity="severe"
)
```

### Sensitivity Analysis
Analyze portfolio response across shock ranges:

```python
# Equity sensitivity analysis
sensitivity = tester.analyze_shock_sensitivity(
    portfolio, book_hierarchy,
    shock_type="SPX",
    shock_range=["-30%", "-20%", "-10%", "0", "+10%", "+20%"]
)
```

### Advanced Visualizations
- **Heatmaps**: Color-coded impact across all scenarios
- **Tornado Charts**: Ranked impact visualization
- **Interactive Dashboards**: Plotly-based HTML dashboards with zoom/pan
- **3D Risk Surfaces**: Multi-dimensional risk visualization
- **Sensitivity Curves**: Visual representation of shock sensitivity

### Try It Now!

```bash
# Run comprehensive stress testing demo
./advanced_stress_testing_demo.py

# See STRESS_TESTING_GUIDE.md for detailed documentation
```

## 🚨 NEW: Compliance Monitoring & Limit Management

Enterprise-grade compliance monitoring system for real-time limit checking and regulatory reporting:

### Configurable Limit Framework
Define limits at multiple levels via YAML configuration:

```yaml
# Portfolio-level limits
- name: "Portfolio Total Notional"
  type: notional
  value: 200000000  # $200M
  scope: portfolio

# Counterparty limits
- name: "AAA Global Bank - Notional"
  type: notional
  scope: counterparty
  entity_id: "CP001"
  value: 75000000  # $75M

# Desk limits, concentration limits, Greek limits, etc.
```

### Real-Time Monitoring
- **Multi-level alerts**: 🟢 OK (<70%) | 🟡 Warning (70-90%) | 🔴 Critical (90-100%) | 🚨 Breach (>100%)
- **Automatic breach detection**: Immediate identification of limit violations
- **Health scoring**: Portfolio-wide compliance health metrics
- **Utilization tracking**: Real-time usage of allocated limits

### Interactive Dashboards
- **Compliance overview**: Health score, alert distribution, key metrics
- **Limit utilization charts**: Visual representation of all limits
- **Gauge dashboards**: Real-time utilization indicators
- **Breach analysis**: Detailed analysis of violations and warnings
- **Scope breakdown**: Analysis by counterparty, desk, product type

### Regulatory Reporting
- **Comprehensive reports**: JSON, CSV, Excel formats
- **Breach documentation**: Complete audit trail
- **Remediation tracking**: Action items and suggestions
- **Export capabilities**: Integration-ready formats

### Try It Now!

```bash
# Run compliance monitoring demo
./compliance_demo.py

# Uses configuration from data/compliance/compliance_limits.yaml
```

## 📊 NEW: Time Series Analysis & Forecasting

Advanced time series analysis for portfolio monitoring and forecasting:

### Historical Data Generation
Generate synthetic historical data with realistic market dynamics:

```python
analyzer = TimeSeriesAnalyzer()

# Generate 1 year of historical MTM data
mtm_history = analyzer.generate_historical_data(
    portfolio, book_hierarchy,
    num_days=252,  # 1 year of trading days
    volatility=0.02,  # 2% daily volatility
    drift=0.0001  # Small positive drift
)

# Generate counterparty exposure history
exposure_history = analyzer.generate_exposure_history(
    portfolio, book_hierarchy,
    num_days=252
)
```

### Statistical Analysis
- **Descriptive Statistics**: Mean, std, min, max
- **Return Analysis**: Mean return, volatility, Sharpe ratio
- **Risk Metrics**: Maximum drawdown, Value at Risk (VaR), Conditional VaR
- **Technical Indicators**: Moving averages, rolling volatility

### Monte Carlo Forecasting
Simulate future portfolio values with confidence intervals:

```python
# Run Monte Carlo simulation
forecast = analyzer.forecast_montecarlo(
    mtm_history,
    num_days=60,  # 3 months
    num_scenarios=1000,
    seed=42
)

# Access forecast statistics
mean_forecast = forecast.mean
p95_upper = forecast.percentile_95
p5_lower = forecast.percentile_5
```

### Advanced Visualizations
- **Trend Charts**: Historical values with moving averages (5, 20, 50-day)
- **Volatility Charts**: Rolling volatility analysis
- **Forecast Charts**: Monte Carlo projections with confidence intervals
- **Exposure Evolution**: Counterparty exposure over time
- **Interactive Dashboard**: Plotly-based HTML dashboard

### Try It Now!

```bash
# Run time series analysis demo
./timeseries_demo.py

# Generates historical data, forecasts, and visualizations
```

## 📈 Features & Capabilities

### 1. Portfolio Management
- **Hierarchical Organization**: Multi-level book structure (Entity → Business Unit → Desk → Book)
- **Counterparty Management**: Credit ratings, entity types, jurisdictions
- **Netting Sets**: ISDA agreements with CSA modeling
- **Trade Lifecycle**: Comprehensive trade attributes and lifecycle management

### 2. XVA Calculations
- **CVA** (Credit Valuation Adjustment)
- **DVA** (Debit Valuation Adjustment)
- **FVA** (Funding Valuation Adjustment)
- **MVA** (Margin Valuation Adjustment)
- **Portfolio-level** and **netting set-level** calculations
- **CSA impact analysis**

### 3. Risk Analytics

#### Stress Testing (Enhanced!)
- **Predefined Scenarios**: 18+ standard market shock scenarios
  - Interest Rate: Parallel shifts, steepeners, flatteners
  - FX: Currency strength/weakness, EM crisis
  - Equity: Market crashes, sector-specific shocks
  - Volatility: Spikes and crushes across asset classes
  - Credit: Spread widening, rating migrations
  - Combined: Multi-factor stress (2008 Crisis, COVID-19, etc.)

- **Custom Scenario Framework**:
  - Create custom scenarios programmatically or from YAML/JSON files
  - Save and load scenario libraries
  - Build scenario templates for recurring analyses

- **Advanced Analytics**:
  - Sensitivity analysis across shock ranges
  - Scenario comparison and ranking
  - Impact attribution by risk factor
  - Category and severity-based analysis

- **Enhanced Visualizations**:
  - Static charts: Heatmaps, tornado charts, category breakdowns
  - Interactive dashboards: Plotly-based HTML dashboards
  - 3D risk surfaces
  - Sensitivity curve plotting

#### Sensitivity Analysis
- **Option Greeks**:
  - Delta: First-order price sensitivity
  - Gamma: Convexity measure
  - Vega: Volatility sensitivity
  - Theta: Time decay
  - Rho: Interest rate sensitivity
- **Rate Sensitivities**: PV01, DV01, CS01
- **FX Sensitivities**: Delta exposures by currency pair
- **Equity Sensitivities**: Delta and Gamma by underlying
- **Bucketed Analysis**: Tenor-based risk ladders

### 4. Reporting & Visualization

#### Report Formats
- **HTML**: Interactive reports with embedded charts
- **Excel**: Multi-sheet workbooks with formatting
- **CSV**: Raw data for analysis
- **JSON**: Structured data for APIs

#### Visualizations
- Counterparty exposure bar charts
- Desk breakdown pie charts
- MTM distribution histograms
- CSA comparison charts
- Credit rating distribution
- XVA waterfall charts
- XVA by counterparty (stacked)
- CSA impact box plots

### 5. Interactive CLI
- **User-friendly** commands with rich terminal output
- **Progress indicators** for long-running tasks
- **Colored output** for better readability
- **Status checking** for dependencies and API connectivity
- **Flexible options** for customization

## 📁 Output Files

### Reports Directory
```
reports/
├── portfolio_report_YYYYMMDD_HHMMSS.json
├── portfolio_report_YYYYMMDD_HHMMSS.html
├── portfolio_report_YYYYMMDD_HHMMSS.xlsx
├── counterparties_YYYYMMDD_HHMMSS.csv
├── books_YYYYMMDD_HHMMSS.csv
├── xva_results.json
├── stress_test_results_YYYYMMDD_HHMMSS.json
├── stress_test_results_YYYYMMDD_HHMMSS.xlsx
├── sensitivity_analysis_YYYYMMDD_HHMMSS.json
└── greeks_YYYYMMDD_HHMMSS.xlsx
```

### Visualizations
```
sample_outputs/charts/
├── counterparty_exposure.png
├── desk_breakdown.png
├── mtm_distribution.png
├── csa_comparison.png
├── rating_distribution.png
├── xva_waterfall.png
├── xva_by_counterparty.png
└── csa_impact.png
```

## 🔧 Configuration

### Market Data
The `config.yaml` file contains:
- Interest rate curves (USD, EUR)
- FX spot rates and volatilities
- Equity spots, dividends, and vol surfaces
- Swaption volatility matrices

### XVA Parameters
- Credit parameters (LGD, default probabilities)
- Funding spreads (collateralized/uncollateralized)
- Initial margin haircuts
- Monte Carlo simulation settings
- Exposure grid configuration

## 📚 Additional Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Detailed tutorials and walkthroughs
- **[STRESS_TESTING_GUIDE.md](STRESS_TESTING_GUIDE.md)** - Advanced stress testing guide (NEW!)
- **[API_EXAMPLES.md](API_EXAMPLES.md)** - Neutryx API integration examples
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and architecture
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## 🎯 Use Cases

This example is perfect for:

1. **Product Demonstrations**: Showcase Neutryx capabilities to clients
2. **Training & Education**: Learn XVA and risk analytics concepts
3. **Testing & Validation**: Verify system functionality
4. **Development**: Template for building custom analytics
5. **Integration Testing**: End-to-end workflow validation
6. **Benchmarking**: Performance testing reference

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Adding new scenarios
- Extending analytics
- Improving visualizations
- Enhancing documentation

## 📝 Notes

### Important Considerations

1. **Simplified Calculations**: Some analytics use simplified models for demonstration. Production systems should use proper pricing models via the Neutryx API.

2. **Market Data**: Static market data is used in config.yaml. Real systems should integrate with market data providers.

3. **Performance**: The demo runs serially for clarity. Production systems can parallelize calculations.

4. **API Dependency**: XVA calculations require the Neutryx API to be running. Other features work standalone.

### Test Data Disclaimer

All counterparty names, ratings, and trade data are fictional and created solely for demonstration purposes.

## 📞 Support

For questions or issues:
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Review [USER_GUIDE.md](USER_GUIDE.md)
- Consult the main Neutryx documentation

## 🎓 Learning Path

**Beginners:**
1. Start with `./cli.py info` to understand the portfolio
2. Run `./cli.py demo` for a complete walkthrough
3. Review generated HTML reports

**Intermediate:**
1. Experiment with individual scripts
2. Modify config.yaml parameters
3. Analyze CSV outputs in Excel

**Advanced:**
1. Review source code for implementation details
2. Extend with custom analytics
3. Integrate with real market data sources
4. Build custom scenarios

## 🏆 Key Highlights

- ✅ **Production-Quality**: Professional code structure and documentation
- ✅ **Comprehensive**: Covers all major risk analytics workflows
- ✅ **Interactive**: CLI and automated demo modes
- ✅ **Well-Documented**: Extensive inline comments and guides
- ✅ **Extensible**: Easy to customize and extend
- ✅ **Educational**: Great learning resource for XVA concepts

---

**Version**: 1.0.0
**Last Updated**: 2024-11-07
**Powered by**: [Neutryx Risk Analytics Platform](../../..)
