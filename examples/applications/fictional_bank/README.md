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
- **Total Trades**: 13 across multiple asset classes
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
│   ├── load_portfolio.py        # Portfolio loader with summary
│   ├── compute_xva.py           # XVA calculation via API
│   ├── portfolio_report.py      # Multi-format reporting
│   ├── visualization.py         # Chart generation
│   ├── stress_testing.py        # Market stress scenarios
│   └── sensitivity_analysis.py  # Greeks & sensitivities
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
│   ├── data/scenarios/          # Stress test scenarios
│   └── templates/               # HTML report templates
│
└── Documentation
    ├── README.md                # This file
    ├── USER_GUIDE.md            # Detailed user guide
    ├── API_EXAMPLES.md          # API usage examples
    ├── ARCHITECTURE.md          # System architecture
    └── TROUBLESHOOTING.md       # Common issues & solutions
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

# 6. Compute sensitivities
./sensitivity_analysis.py
```

#### Option 3: Execute Complete Demo

```bash
# Run all analyses in sequence
./run_all_demos.py
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

#### Stress Testing
- **Interest Rate Scenarios**: Parallel shifts, steepeners, flatteners
- **FX Stress**: Currency strength/weakness scenarios
- **Equity Stress**: Market crashes, sector-specific shocks
- **Volatility Shocks**: Spikes and crushes
- **Credit Scenarios**: Spread widening, rating migrations
- **Combined Scenarios**: Multi-factor stress tests
- **Historical Replays**: 2008 Crisis, COVID-19, etc.

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
