# FpML Integration Examples

This directory contains examples demonstrating Neutryx's FpML (Financial products Markup Language) integration capabilities.

## Overview

Neutryx provides comprehensive FpML 5.x support, enabling seamless integration with standard financial messaging systems. The integration supports:

- **Equity Options** (European and American)
- **FX Options**
- **Interest Rate Swaps** (vanilla fixed-floating)

## Files in This Directory

- `equity_call_option.xml` - Sample FpML document for an equity call option on AAPL
- `fx_call_option.xml` - Sample FpML document for an EUR/USD FX call option
- `fpml_pricing_example.py` - Comprehensive Python examples demonstrating FpML workflows

## Quick Start

### 1. Parse an FpML Document

```python
from neutryx.bridge import fpml

# Load and parse FpML XML
with open("equity_call_option.xml") as f:
    fpml_xml = f.read()

fpml_doc = fpml.parse_fpml(fpml_xml)

# Access trade information
trade = fpml_doc.primary_trade
if trade.equityOption:
    print(f"Strike: {trade.equityOption.strike.strikePrice}")
    print(f"Underlyer: {trade.equityOption.underlyer.instrumentId}")
```

### 2. Price an FpML Trade

```python
from neutryx.bridge.fpml_adapter import quick_price_fpml

# Define market data
market_data = {
    "spot": 155.0,
    "volatility": 0.25,
    "rate": 0.05,
    "dividend": 0.01,
}

# Price the trade
price = quick_price_fpml(fpml_xml, market_data)
print(f"Option Price: ${price:.2f}")
```

### 3. Export to FpML

```python
from datetime import date
from neutryx.api.rest import VanillaOptionRequest
from neutryx.bridge import fpml

# Create a pricing request
request = VanillaOptionRequest(
    spot=100.0,
    strike=105.0,
    maturity=1.0,
    volatility=0.25,
    call=True,
)

# Convert to FpML
fpml_doc = fpml.neutryx_to_fpml(
    request,
    instrument_id="US0378331005",
    trade_date=date(2024, 1, 15)
)

# Serialize to XML
xml_output = fpml.serialize_fpml(fpml_doc)
print(xml_output)
```

## REST API Integration

Neutryx also exposes FpML functionality through REST endpoints:

### Price an FpML Document

```bash
curl -X POST http://localhost:8000/fpml/price \
  -H "Content-Type: application/json" \
  -d '{
    "fpml_xml": "<dataDocument>...</dataDocument>",
    "market_data": {
      "spot": 155.0,
      "volatility": 0.25,
      "rate": 0.05
    }
  }'
```

### Parse an FpML Document

```bash
curl -X POST http://localhost:8000/fpml/parse \
  -H "Content-Type: application/json" \
  -d '{
    "fpml_xml": "<dataDocument>...</dataDocument>"
  }'
```

### Validate an FpML Document

```bash
curl -X POST http://localhost:8000/fpml/validate \
  -H "Content-Type: application/json" \
  -d '{
    "fpml_xml": "<dataDocument>...</dataDocument>"
  }'
```

## Running the Examples

```bash
# Navigate to the examples directory
cd examples/fpml

# Run the comprehensive example
python fpml_pricing_example.py
```

## Advanced Usage

### Custom Monte Carlo Configuration

```python
from neutryx.bridge.fpml_adapter import FpMLPricingAdapter
from neutryx.core.engine import MCConfig

adapter = FpMLPricingAdapter(
    default_mc_config=MCConfig(steps=500, paths=500_000),
    seed=12345
)

result = adapter.price_from_xml(fpml_xml, market_data)
```

### Batch Pricing

```python
from neutryx.bridge.fpml_adapter import FpMLBatchPricer

batch_pricer = FpMLBatchPricer(seed=42)

xml_documents = [fpml_xml1, fpml_xml2, fpml_xml3]
market_data_list = [market1, market2, market3]

results = batch_pricer.price_multiple_xml(xml_documents, market_data_list)
```

### Custom Reference Date

```python
from datetime import date
from neutryx.bridge.fpml import FpMLToNeutryxMapper

mapper = FpMLToNeutryxMapper(reference_date=date(2024, 6, 1))
request = mapper.map_trade(fpml_doc.primary_trade, market_data)
```

## FpML Schema Support

### Supported Elements

- ✅ Equity Options (European/American)
- ✅ FX Options
- ✅ Interest Rate Swaps (vanilla)
- ✅ Party references
- ✅ Trade headers
- ✅ Adjustable dates
- ✅ Money/currency amounts
- ✅ Strike prices
- ✅ Exercise terms

### Product-Specific Features

#### Equity Options
- Put/Call specification
- Instrument identifiers (ISIN, RIC)
- Strike price
- Expiration date
- Number of options
- Settlement type (Cash/Physical)

#### FX Options
- Currency pair quotation
- Put/Call currency amounts
- Strike rate
- Spot rate
- Expiry date and time
- Cut name (timezone)

#### Interest Rate Swaps
- Fixed/Floating legs
- Notional schedules
- Payment frequencies
- Day count conventions
- Floating rate indices (LIBOR, SOFR, EURIBOR)

## Error Handling

The FpML integration provides clear error messages:

```python
from neutryx.bridge import fpml

try:
    fpml_doc = fpml.parse_fpml(invalid_xml)
except fpml.FpMLParseError as e:
    print(f"Parse error: {e}")

try:
    request = fpml.fpml_to_neutryx(fpml_doc, {})
except fpml.FpMLMappingError as e:
    print(f"Mapping error: {e}")
```

## Testing

Run the FpML integration tests:

```bash
pytest src/neutryx/tests/test_fpml.py -v
```

## References

- [FpML Official Website](https://www.fpml.org/)
- [FpML 5.x Specification](https://www.fpml.org/spec/)
- [Neutryx Documentation](../../docs/)

## Support

For questions or issues with FpML integration:
- Open an issue on GitHub
- Check the main Neutryx documentation
- Review the example code in this directory

## License

This code is part of the Neutryx project and is licensed under the MIT License.
