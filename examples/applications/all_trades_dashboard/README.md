# Fictional Bank - All Trades Dashboard

This interactive dashboard provides comprehensive real-time monitoring of all trades in the Fictional Bank portfolio, with full PV (Present Value), risk analytics (Greeks), and XVA calculations. Built with [Gradio](https://www.gradio.app/), it demonstrates the power of `neutryx` for enterprise-grade portfolio risk management.

## Features

### 📊 Real-time Trade Monitoring
- **All Trades View**: Comprehensive table showing every trade with live PV and Greeks
- **Smart Filtering**: Filter by counterparty, product type, or trading desk
- **Auto-refresh**: Dashboard updates every 5 seconds to simulate real-time market movements
- **Greeks Display**: Delta, Gamma, Vega for every position

### 🎯 Key Performance Indicators
- Total number of trades and counterparties
- Aggregate notional exposure
- Total MTM (Mark-to-Market) and PV
- Net portfolio Greeks (Delta, Gamma, Vega)

### 🔥 Risk Heatmap
- **Concentration Analysis**: Visual matrix of exposure by counterparty and product type
- **Risk Identification**: Quickly identify concentrated positions
- **Multi-dimensional View**: See where your largest exposures lie

### ⚡ Stress Testing
Pre-configured stress scenarios to assess portfolio resilience:
- **Interest Rate Shocks**: ±100bps parallel shifts
- **Volatility Shocks**: ±20% volatility changes
- **FX Scenarios**: ±10% spot rate movements
- **Equity Stress**: -20% equity market crash

Each scenario shows:
- PV change from base case
- Delta change
- Total portfolio impact

### 💰 XVA Analysis
Comprehensive valuation adjustments by counterparty:
- **CVA (Credit Valuation Adjustment)**: Credit risk pricing
- **DVA (Debit Valuation Adjustment)**: Own credit risk benefit
- **FVA (Funding Valuation Adjustment)**: Funding cost impact
- **MVA (Margin Valuation Adjustment)**: Margin posting costs
- **Total XVA**: Net adjustment to fair value

### 🚨 Real-time Alerts
Automated monitoring of risk limits:
- Net Delta limit breaches
- Gamma concentration alerts
- Counterparty concentration warnings (>30% of portfolio)
- Visual indicators for risk status

### 📥 Export Functionality
- Export current portfolio state to CSV
- Timestamped filenames for audit trail
- Full trade details with PV and Greeks

## Architecture

The dashboard leverages several core `neutryx` components:

- **Portfolio Management**: `neutryx.portfolio.portfolio.Portfolio`
- **Test Fixtures**: `neutryx.tests.fixtures.fictional_portfolio.create_fictional_portfolio()`
- **Pricing Engines**: `neutryx.engines.fourier` (COS method, FFT)
- **Risk Calculations**: Custom Greeks calculation based on trade characteristics

## Prerequisites

Install the project dependencies together with Gradio:

```bash
pip install -r requirements.txt
pip install gradio>=4.0 pandas pyyaml
```

If you already have the repository's development dependencies installed, you only need to add `gradio` because `pandas` and `pyyaml` are already part of the base requirements.

## Configuration

The dashboard is configured via `config.yaml`, which allows you to customize:

- **Refresh interval**: How often the dashboard auto-updates (default: 5 seconds)
- **Risk limits**: Thresholds for alerts (Net Delta, Gamma, concentration)
- **Stress scenarios**: Market shock parameters
- **XVA parameters**: Loss Given Default (LGD), funding spreads, margin periods
- **Display settings**: Formatting, decimal places, row limits

Example configuration:

```yaml
# Risk limits and alerts
risk_limits:
  max_net_delta: 1000000
  max_counterparty_exposure: 5000000
  max_gamma: 50000
  concentration_threshold: 0.3  # 30%

# XVA calculation parameters
xva_params:
  default_lgd: 0.6
  funding_spread_bps: 50
  margin_period_of_risk_days: 10
```

## Run Locally

```bash
python examples/applications/all_trades_dashboard/app.py
```

Gradio will print a `http://127.0.0.1:7860` link in the console. Open it in a browser to interact with the dashboard.

The dashboard will automatically:
1. Load the Fictional Bank portfolio with 13 trades across 6 counterparties
2. Calculate initial PV and Greeks for all positions
3. Display key metrics and risk indicators
4. Begin auto-refreshing every 5 seconds

## Usage Examples

### Monitor a Specific Counterparty
1. Navigate to the "All Trades" tab
2. Select a counterparty from the dropdown (e.g., "CP_BANK_A")
3. View all trades and aggregate exposure for that counterparty

### Run Stress Tests
1. Go to the "Stress Tests" tab
2. Review pre-calculated scenarios
3. Identify which scenarios have the largest impact on your portfolio

### Analyze XVA Costs
1. Open the "XVA Analysis" tab
2. Compare CVA, DVA, FVA, and MVA across counterparties
3. Identify counterparties with the highest credit/funding costs

### Check Risk Alerts
1. Alerts are displayed at the top of the dashboard
2. Green checkmark = all limits OK
3. Warning symbols = limit breaches requiring attention

### Export Data
1. Configure your filters on the "All Trades" tab
2. Click "Export to CSV"
3. Download timestamped CSV file for offline analysis

## Fictional Portfolio Details

The dashboard uses a comprehensive test portfolio created by `create_fictional_portfolio()` with:

- **13 trades** across multiple asset classes
- **6 counterparties** with diverse credit ratings
- **3 trading desks**: Rates, FX, Equity
- **Product types**:
  - Interest Rate Swaps (IRS)
  - Swaptions
  - FX Options
  - Equity Options
  - Variance Swaps
- **Multiple netting sets** with and without CSA (Credit Support Annex)

This provides a realistic example of a mid-sized trading portfolio for demonstration purposes.

## Technical Notes

### PV Calculation
The current implementation uses simplified PV calculations based on MTM with random variations to simulate real-time market movements. In a production environment, you would integrate with:
- Live pricing engines
- Market data feeds (Bloomberg, Refinitiv)
- Internal valuation models

### Greeks Calculation
Greeks are calculated using simplified approximations based on trade characteristics. For production use, you should implement:
- Full derivative pricing models
- Bump-and-revalue for precise sensitivities
- Cross-gamma and other higher-order Greeks

### XVA Calculations
XVA calculations use simplified formulas for demonstration. Production implementations should include:
- Full exposure simulation (Monte Carlo)
- Collateral modeling
- Dynamic initial margin (SIMM)
- Bilateral CVA/DVA
- Regulatory capital costs (KVA)

## Extending the Dashboard

This dashboard serves as a template that can be extended with:

1. **Historical Analysis**: Add time-series charts of PV/Greeks evolution
2. **VaR Calculation**: Implement Value at Risk metrics
3. **Scenario Manager**: Allow users to create custom stress scenarios
4. **Limit Management**: Integrate with firm-wide limit framework
5. **Regulatory Reporting**: Generate SA-CCR, FRTB reports
6. **Trade Entry**: Add ability to create/modify trades
7. **Real Market Data**: Connect to live market data providers
8. **Multi-Currency**: Enhanced FX risk analysis
9. **Liquidity Risk**: Add liquidity horizon analysis
10. **What-If Analysis**: Interactive trade idea simulation

## Screenshot

*Screenshot will be added after running the dashboard*

## Related Examples

- **Pricing Dashboard** (`examples/applications/dashboard/`): Black-Scholes option pricing with FFT/COS methods
- **Fictional Bank Portfolio** (`examples/applications/fictional_bank/`): XVA calculation examples
- **Market Data Feeds** (`src/neutryx/market/feeds/`): Real-time market data integration

## Support

For questions or issues with this dashboard, please refer to the main `neutryx` documentation or open an issue on the project repository.
