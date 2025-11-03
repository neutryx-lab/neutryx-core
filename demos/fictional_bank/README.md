# Fictional Bank Portfolio

This directory contains a comprehensive fictional portfolio for testing Neutryx's XVA and risk analytics capabilities.

## Portfolio Structure

### Organization Hierarchy
- **Legal Entity**: Global Investment Bank Ltd (GB)
- **Business Unit**: Global Trading
- **Desks**:
  - Interest Rates Desk (3 books)
  - Foreign Exchange Desk (2 books)
  - Equity Derivatives Desk (2 books)
- **Traders**: 6 traders across all desks

### Counterparties
1. **AAA Global Bank** (AAA, Financial Institution)
   - CSA: Bilateral, USD 1M threshold
   - Products: USD IRS

2. **Tech Corporation A** (A, Corporate)
   - CSA: Bilateral, USD 1M threshold
   - Products: FX Options (EUR/USD, USD/JPY)

3. **Industrial Group BBB** (BBB, Corporate)
   - No CSA
   - Products: EUR IRS

4. **Alpha Strategies Fund** (A-, Hedge Fund)
   - No CSA
   - Products: FX Emerging Markets, Equity Exotics

5. **Republic Investment Authority** (AA+, Sovereign)
   - CSA: Bilateral, USD 1M threshold
   - Products: Equity Vanilla Options

6. **Global Insurance Group** (AA, Financial Institution)
   - CSA: Bilateral, USD 1M threshold
   - Products: Swaptions, Variance Swaps

### Trade Summary
- **Total Trades**: 13
- **Product Types**:
  - Interest Rate Swaps: 3
  - Swaptions: 1
  - FX Options: 3
  - Equity Options: 5
  - Variance Swaps: 1
- **Total Notional**: ~USD 152M
- **Net MTM**: Variable based on market conditions

## Files

- `config.yaml` - Portfolio configuration and market data settings
- `load_portfolio.py` - Script to load and register the portfolio
- `compute_xva.py` - Script to compute XVA metrics via API
- `portfolio_report.py` - Generate comprehensive portfolio reports

## Usage

### Load Portfolio
```bash
python demos/fictional_bank/load_portfolio.py
```

### Compute Portfolio XVA
```bash
python demos/fictional_bank/compute_xva.py
```

### Generate Reports
```bash
python demos/fictional_bank/portfolio_report.py
```

## API Integration

The portfolio can be loaded into the Neutryx API and analyzed:

```python
from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio

portfolio, book_hierarchy = create_fictional_portfolio()

# Register with API
import requests
response = requests.post(
    "http://localhost:8000/portfolio/register",
    json=portfolio.model_dump()
)

# Compute XVA
xva_response = requests.post(
    "http://localhost:8000/portfolio/xva",
    json={
        "portfolio_id": "Global Trading Portfolio",
        "valuation_date": "2024-01-15",
        "compute_cva": True,
        "compute_dva": True,
        "compute_fva": True,
        "compute_mva": True,
    }
)
```

## Testing Scenarios

This portfolio is designed to test:
1. **Multi-desk risk aggregation**
2. **CSA vs non-CSA netting sets**
3. **Multiple counterparty credit ratings**
4. **Diverse product types and maturities**
5. **Book hierarchy reporting**
6. **XVA calculations at portfolio and netting set levels**
