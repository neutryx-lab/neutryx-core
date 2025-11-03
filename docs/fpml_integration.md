# FpML Integration Guide

Neutryx provides comprehensive support for FpML (Financial products Markup Language), the industry-standard XML format for representing derivatives trades and market data. For a high-level introduction to the platform, refer to the [project README](https://github.com/neutryx-lab/neutryx-core/blob/main/README.md).

## Overview

The FpML integration enables:

- **Parsing** FpML 5.x XML documents
- **Mapping** FpML trades to Neutryx pricing models
- **Pricing** FpML trades using JAX-accelerated Monte Carlo engines
- **Serializing** Neutryx results back to FpML format
- **Validation** of FpML document structure

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FpML Integration Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
│  │   Parser   │──▶│  Mappings  │──▶│  Neutryx   │            │
│  │  (XML→Py)  │   │ (FpML→Req) │   │   Engine   │            │
│  └────────────┘   └────────────┘   └────────────┘            │
│                                           │                     │
│                                           ▼                     │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
│  │ Serializer │◀──│  Mappings  │◀──│   Results  │            │
│  │  (Py→XML)  │   │ (Req→FpML) │   │            │            │
│  └────────────┘   └────────────┘   └────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/neutryx/bridge/fpml/
├── __init__.py          # Public API exports
├── schemas.py           # Pydantic models for FpML elements
├── parser.py            # XML → Pydantic conversion
├── serializer.py        # Pydantic → XML conversion
└── mappings.py          # FpML ⇔ Neutryx conversion

src/neutryx/bridge/
└── fpml_adapter.py      # High-level workflow API
```

## Quick Start

### 1. Parse FpML Document

```python
from neutryx.bridge import fpml

# Load FpML XML
with open("trade.xml") as f:
    fpml_xml = f.read()

# Parse to structured format
fpml_doc = fpml.parse_fpml(fpml_xml)

# Access trade details
trade = fpml_doc.primary_trade
print(f"Trade date: {trade.tradeHeader.tradeDate}")

if trade.equityOption:
    print(f"Strike: {trade.equityOption.strike.strikePrice}")
```

### 2. Price FpML Trade

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
print(f"Price: ${price:.2f}")
```

### 3. Convert to FpML

```python
from datetime import date
from neutryx.api.rest import VanillaOptionRequest
from neutryx.bridge import fpml

# Create pricing request
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
    trade_date=date.today()
)

# Serialize to XML
xml_output = fpml.serialize_fpml(fpml_doc)
```

## Supported Products

### Equity Options

**Supported Features:**
- Put/Call specification
- European and American exercise
- Strike price
- Underlyer identification (ISIN, RIC, etc.)
- Expiration dates
- Number of options
- Settlement type

**Example FpML:**
```xml
<equityOption>
    <buyerPartyReference href="party1"/>
    <sellerPartyReference href="party2"/>
    <optionType>Call</optionType>
    <underlyer>
        <instrumentId>US0378331005</instrumentId>
        <description>Apple Inc.</description>
    </underlyer>
    <strike>
        <strikePrice>150.00</strikePrice>
    </strike>
    <equityExercise>
        <equityEuropeanExercise>
            <expirationDate>
                <adjustableDate>
                    <unadjustedDate>2025-01-15</unadjustedDate>
                </adjustableDate>
            </expirationDate>
        </equityEuropeanExercise>
    </equityExercise>
</equityOption>
```

### FX Options

**Supported Features:**
- Currency pair specification
- Put/Call currency amounts
- Strike rate
- Spot rate
- European and American exercise
- Expiry date and time
- Cut name (timezone)

**Example FpML:**
```xml
<fxOption>
    <putCurrencyAmount>
        <currency>USD</currency>
        <amount>1100000.00</amount>
    </putCurrencyAmount>
    <callCurrencyAmount>
        <currency>EUR</currency>
        <amount>1000000.00</amount>
    </callCurrencyAmount>
    <strike>
        <rate>1.10</rate>
    </strike>
    <europeanExercise>
        <expiryDate>2025-01-15</expiryDate>
    </europeanExercise>
</fxOption>
```

### Interest Rate Swaps

**Supported Features:**
- ✅ Fixed and floating legs
- ✅ Notional schedules  
- ✅ Payment frequencies (Monthly, Quarterly, Semiannual, Annual)
- ✅ Day count conventions (ACT/360, ACT/365, 30/360, ACT/ACT)
- ✅ Floating rate indices (LIBOR, SOFR, EURIBOR, ESTR)
- ✅ Calculation period dates with business day adjustments
- ✅ Present value calculation
- ✅ DV01 and risk metrics
- ✅ Cash flow schedule generation

**Valuation:**
Full swap valuation is now implemented with JAX-accelerated pricing. The swap pricer supports:
- Fixed vs. floating leg present value calculations
- Discount factor curves
- Payment schedule generation with business day conventions
- Risk metrics including DV01 (dollar value of 1 basis point)

**Example:**
```python
from neutryx.products.swap import price_vanilla_swap

# Price a 5-year interest rate swap
value = price_vanilla_swap(
    notional=10_000_000,  # $10M notional
    fixed_rate=0.05,       # 5% fixed rate
    floating_rate=0.045,   # 4.5% current floating rate
    maturity=5.0,          # 5 years
    payment_frequency=2,   # Semiannual payments
    discount_rate=0.05,    # 5% discount rate
    pay_fixed=True         # Pay fixed, receive floating
)
print(f"Swap Value: ${value:,.2f}")
```

## API Reference

### Core Functions

#### `parse_fpml(xml_content: str) -> FpMLDocument`

Parse FpML XML to structured Pydantic model.

**Parameters:**
- `xml_content`: FpML XML document as string

**Returns:**
- `FpMLDocument`: Parsed document with parties and trades

**Raises:**
- `FpMLParseError`: If XML is invalid or cannot be parsed

**Example:**
```python
from neutryx.bridge import fpml

doc = fpml.parse_fpml(xml_string)
print(f"Parties: {len(doc.party)}")
print(f"Trades: {len(doc.trade)}")
```

#### `fpml_to_neutryx(fpml_doc: FpMLDocument, market_data: dict) -> VanillaOptionRequest`

Convert FpML document to Neutryx pricing request.

**Parameters:**
- `fpml_doc`: Parsed FpML document
- `market_data`: Dictionary with market data:
  - For equity options: `spot`, `volatility`, `rate`, `dividend`
  - For FX options: `spot_rate`, `volatility`, `domestic_rate`, `foreign_rate`

**Returns:**
- `VanillaOptionRequest`: Neutryx pricing request

**Raises:**
- `FpMLMappingError`: If required data missing or conversion fails

**Example:**
```python
from neutryx.bridge import fpml

doc = fpml.parse_fpml(xml_string)
market_data = {"spot": 100.0, "volatility": 0.25, "rate": 0.05}
request = fpml.fpml_to_neutryx(doc, market_data)
```

#### `neutryx_to_fpml(request: VanillaOptionRequest, instrument_id: str, trade_date: date) -> FpMLDocument`

Convert Neutryx request to FpML document.

**Parameters:**
- `request`: Neutryx vanilla option request
- `instrument_id`: Underlying instrument identifier
- `trade_date`: Trade date

**Returns:**
- `FpMLDocument`: FpML document ready for serialization

**Example:**
```python
from datetime import date
from neutryx.api.rest import VanillaOptionRequest
from neutryx.bridge import fpml

request = VanillaOptionRequest(
    spot=100.0, strike=105.0, maturity=1.0, volatility=0.25, call=True
)
doc = fpml.neutryx_to_fpml(request, "US0378331005", date.today())
```

#### `serialize_fpml(fpml_doc: FpMLDocument, pretty_print: bool = True) -> str`

Serialize FpML document to XML string.

**Parameters:**
- `fpml_doc`: FpML document
- `pretty_print`: Whether to format with indentation

**Returns:**
- XML string representation

**Example:**
```python
from neutryx.bridge import fpml

xml_string = fpml.serialize_fpml(doc, pretty_print=True)
print(xml_string)
```

### High-Level Adapter

#### `FpMLPricingAdapter`

Comprehensive adapter for end-to-end FpML workflows.

**Methods:**
- `parse_xml(xml_content)`: Parse FpML XML
- `price_from_xml(xml_content, market_data, mc_config)`: Price from XML
- `price_from_document(fpml_doc, market_data, mc_config)`: Price from parsed doc
- `export_to_fpml(...)`: Export parameters to FpML

**Example:**
```python
from neutryx.bridge.fpml_adapter import FpMLPricingAdapter
from neutryx.core.engine import MCConfig

adapter = FpMLPricingAdapter(
    default_mc_config=MCConfig(steps=252, paths=100_000),
    seed=42
)

result = adapter.price_from_xml(fpml_xml, market_data)
print(f"Price: {result['price']}")
print(f"Trade info: {result['trade_info']}")
```

#### `FpMLBatchPricer`

Batch pricer for multiple trades.

**Example:**
```python
from neutryx.bridge.fpml_adapter import FpMLBatchPricer

pricer = FpMLBatchPricer(seed=42)
results = pricer.price_multiple_xml(xml_list, market_data_list)

for i, result in enumerate(results):
    print(f"Trade {i}: ${result['price']:.2f}")
```

## REST API Endpoints

### POST /fpml/price

Price an FpML trade document.

**Request:**
```json
{
  "fpml_xml": "<dataDocument>...</dataDocument>",
  "market_data": {
    "spot": 155.0,
    "volatility": 0.25,
    "rate": 0.05,
    "dividend": 0.01
  },
  "steps": 252,
  "paths": 100000,
  "seed": 42
}
```

**Response:**
```json
{
  "price": 12.34,
  "trade_date": "2024-01-15",
  "trade_info": {
    "product_type": "EquityOption",
    "option_type": "Call",
    "strike": 150.0,
    "underlyer": "US0378331005"
  }
}
```

### POST /fpml/parse

Parse and inspect FpML document structure.

**Request:**
```json
{
  "fpml_xml": "<dataDocument>...</dataDocument>"
}
```

**Response:**
```json
{
  "trade_date": "2024-01-15",
  "parties": [
    {"id": "party1", "name": "Bank ABC"},
    {"id": "party2", "name": "Client XYZ"}
  ],
  "product": {
    "type": "EquityOption",
    "option_type": "Call",
    "strike": 150.0,
    "underlyer": "US0378331005",
    "expiration": "2025-01-15"
  }
}
```

### POST /fpml/validate

Validate FpML document structure.

**Request:**
```json
{
  "fpml_xml": "<dataDocument>...</dataDocument>"
}
```

**Response:**
```json
{
  "valid": true,
  "message": "FpML document is valid"
}
```

## Error Handling

### Parse Errors

```python
from neutryx.bridge import fpml

try:
    doc = fpml.parse_fpml(invalid_xml)
except fpml.FpMLParseError as e:
    print(f"Parse error: {e}")
    # Handle: invalid XML, missing elements, etc.
```

### Mapping Errors

```python
from neutryx.bridge import fpml

try:
    request = fpml.fpml_to_neutryx(doc, market_data)
except fpml.FpMLMappingError as e:
    print(f"Mapping error: {e}")
    # Handle: missing market data, expired options, unsupported products
```

## Advanced Usage

### Custom Reference Date

By default, maturity is calculated from the trade date. You can specify a custom reference date:

```python
from datetime import date
from neutryx.bridge.fpml import FpMLToNeutryxMapper

mapper = FpMLToNeutryxMapper(reference_date=date(2024, 6, 1))
request = mapper.map_trade(fpml_doc.primary_trade, market_data)
```

### Custom Monte Carlo Configuration

```python
from neutryx.bridge.fpml_adapter import FpMLPricingAdapter
from neutryx.core.engine import MCConfig

# High-accuracy configuration
mc_config = MCConfig(
    steps=500,
    paths=1_000_000,
    antithetic=True
)

adapter = FpMLPricingAdapter(default_mc_config=mc_config)
result = adapter.price_from_xml(fpml_xml, market_data, mc_config)
```

### Accessing Detailed Trade Information

```python
from neutryx.bridge import fpml

doc = fpml.parse_fpml(fpml_xml)
trade = doc.primary_trade

if trade.equityOption:
    opt = trade.equityOption
    print(f"Buyer: {opt.buyerPartyReference.href}")
    print(f"Seller: {opt.sellerPartyReference.href}")
    print(f"Underlyer: {opt.underlyer.instrumentId}")
    print(f"Description: {opt.underlyer.description}")
    print(f"Strike: {opt.strike.strikePrice}")
    print(f"Expiration: {opt.equityExercise.expirationDate.unadjustedDate}")
```

## Performance Considerations

### XML Parsing

The default implementation uses Python's built-in `xml.etree.ElementTree` which is sufficient for most use cases. For high-volume processing, consider:

1. **lxml**: Faster XML parsing (install with `pip install lxml`)
2. **Batch processing**: Use `FpMLBatchPricer` for multiple trades
3. **Caching**: Parse FpML documents once and reuse

### Pricing Performance

Monte Carlo pricing is GPU-accelerated via JAX:

```python
# Standard configuration (~10ms per option)
MCConfig(steps=252, paths=100_000)

# Fast configuration (~2ms per option, lower accuracy)
MCConfig(steps=64, paths=10_000)

# High-accuracy configuration (~100ms per option)
MCConfig(steps=500, paths=1_000_000)
```

## Testing

Run FpML integration tests:

```bash
pytest src/neutryx/tests/test_fpml.py -v
```

Test coverage includes:
- Parsing equity and FX options
- Round-trip conversions
- Error handling
- API endpoints
- Batch pricing

## Examples

Complete examples are available in `examples/fpml/`:

- `equity_call_option.xml` - Sample equity option
- `fx_call_option.xml` - Sample FX option
- `fpml_pricing_example.py` - Comprehensive Python examples
- `README.md` - Quick start guide

Run examples:
```bash
cd examples/fpml
python fpml_pricing_example.py
```

## Limitations and Future Work

### Current Limitations

1. **Swaps**: Basic structure parsing only; full valuation requires yield curve module
2. **Exotic Options**: Barrier, Asian, Lookback require custom mapping
3. **Schema Validation**: Optional (requires xmlschema package)
4. **Namespace Handling**: Supports FpML 5.x default namespace

### Roadmap

- [ ] Full interest rate swap valuation
- [ ] Barrier and exotic option support
- [ ] Multi-leg structures
- [ ] Credit derivatives (CDS)
- [ ] Real-time market data integration
- [ ] FpML 6.x support

## Best Practices

1. **Always validate market data**: Ensure all required fields are present
2. **Handle errors gracefully**: Use try/except blocks for parsing and mapping
3. **Cache parsed documents**: Avoid re-parsing the same XML
4. **Use batch pricing**: More efficient for multiple trades
5. **Monitor performance**: Adjust MC configuration based on accuracy needs

## Support

For issues or questions:
- GitHub Issues: [neutryx-lab/neutryx-core/issues](https://github.com/neutryx-lab/neutryx-core/issues)
- Documentation: [docs.neutryx.tech](https://docs.neutryx.tech)
- Examples: `examples/fpml/`

## References

- [FpML Official Website](https://www.fpml.org/)
- [FpML 5.x Specification](https://www.fpml.org/spec/fpml-5-0-0/)
- [Neutryx Core Documentation](https://github.com/neutryx-lab/neutryx-core/blob/main/README.md)
- [REST API Reference](api_reference.md)
