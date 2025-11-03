"""FpML Pricing Example with Neutryx.

This example demonstrates how to:
1. Load an FpML XML document
2. Parse it with Neutryx
3. Price the trade using Monte Carlo simulation
4. Export results back to FpML
"""
from pathlib import Path

import jax

from neutryx.bridge import fpml
from neutryx.bridge.fpml_adapter import FpMLPricingAdapter, quick_price_fpml
from neutryx.core.engine import MCConfig


def main():
    print("=" * 70)
    print("FpML Pricing Example with Neutryx")
    print("=" * 70)
    print()

    # Example 1: Quick pricing
    print("Example 1: Quick Pricing of Equity Call Option")
    print("-" * 70)

    equity_option_xml = Path(__file__).parent / "equity_call_option.xml"
    with open(equity_option_xml) as f:
        fpml_xml = f.read()

    market_data = {
        "spot": 155.0,  # Current AAPL price
        "volatility": 0.25,  # 25% volatility
        "rate": 0.05,  # 5% risk-free rate
        "dividend": 0.01,  # 1% dividend yield
    }

    price = quick_price_fpml(fpml_xml, market_data)
    print(f"Option Price: ${price:.2f}")
    print()

    # Example 2: Detailed pricing with adapter
    print("Example 2: Detailed Pricing with Full Information")
    print("-" * 70)

    adapter = FpMLPricingAdapter(
        default_mc_config=MCConfig(steps=252, paths=100_000), seed=42
    )

    result = adapter.price_from_xml(fpml_xml, market_data)

    print(f"Price: ${result['price']:.2f}")
    print(f"Trade Date: {result['trade'].tradeHeader.tradeDate}")

    trade_info = result["trade_info"]
    print(f"Product Type: {trade_info['product_type']}")
    print(f"Option Type: {trade_info['option_type']}")
    print(f"Strike: ${trade_info['strike']:.2f}")
    print(f"Underlyer: {trade_info['underlyer']}")
    print()

    # Example 3: Parse and inspect FpML structure
    print("Example 3: Parse and Inspect FpML Structure")
    print("-" * 70)

    fpml_doc = fpml.parse_fpml(fpml_xml)

    print(f"Number of parties: {len(fpml_doc.party)}")
    for party in fpml_doc.party:
        print(f"  - {party.name} (id: {party.id})")

    trade = fpml_doc.primary_trade
    if trade.equityOption:
        opt = trade.equityOption
        print(f"\nEquity Option Details:")
        print(f"  Option Type: {opt.optionType.value}")
        print(f"  Exercise Type: {opt.equityExercise.optionType.value}")
        print(f"  Underlyer: {opt.underlyer.instrumentId}")
        print(f"  Description: {opt.underlyer.description}")
        print(f"  Strike: {opt.strike.strikePrice}")
        print(f"  Number of Options: {opt.numberOfOptions}")
        print(f"  Expiration: {opt.equityExercise.expirationDate.unadjustedDate}")
    print()

    # Example 4: Convert Neutryx request to FpML
    print("Example 4: Export Neutryx Pricing Request to FpML")
    print("-" * 70)

    from datetime import date

    from neutryx.api.rest import VanillaOptionRequest

    # Create a pricing request
    request = VanillaOptionRequest(
        spot=100.0,
        strike=105.0,
        maturity=0.5,
        rate=0.04,
        dividend=0.01,
        volatility=0.22,
        call=False,  # Put option
    )

    # Convert to FpML
    fpml_doc = fpml.neutryx_to_fpml(
        request, instrument_id="EXAMPLE_STOCK", trade_date=date(2024, 1, 15)
    )

    # Serialize to XML
    xml_output = fpml.serialize_fpml(fpml_doc)

    print("Generated FpML XML (first 500 characters):")
    print(xml_output[:500] + "...")
    print()

    # Example 5: FX Option pricing
    print("Example 5: FX Option Pricing")
    print("-" * 70)

    fx_option_xml = Path(__file__).parent / "fx_call_option.xml"
    with open(fx_option_xml) as f:
        fx_fpml = f.read()

    fx_market_data = {
        "spot_rate": 1.08,  # Current EUR/USD rate
        "volatility": 0.15,  # 15% FX volatility
        "domestic_rate": 0.05,  # USD interest rate
        "foreign_rate": 0.03,  # EUR interest rate
    }

    fx_price = quick_price_fpml(fx_fpml, fx_market_data)
    print(f"FX Option Price: ${fx_price:.2f}")
    print()

    # Example 6: Validate FpML documents
    print("Example 6: FpML Validation")
    print("-" * 70)

    from neutryx.bridge.fpml_adapter import validate_fpml

    valid_xml = fpml_xml
    invalid_xml = "<invalid>xml</invalid>"

    print(f"Valid FpML: {validate_fpml(valid_xml)}")
    print(f"Invalid XML: {validate_fpml(invalid_xml)}")
    print()

    # Example 7: Batch pricing
    print("Example 7: Batch Pricing Multiple Trades")
    print("-" * 70)

    from neutryx.bridge.fpml_adapter import FpMLBatchPricer

    batch_pricer = FpMLBatchPricer(seed=42)

    xml_documents = [fpml_xml, fx_fpml]
    market_data_list = [market_data, fx_market_data]

    results = batch_pricer.price_multiple_xml(xml_documents, market_data_list)

    for i, result in enumerate(results, 1):
        if "error" not in result:
            print(f"Trade {i} Price: ${result['price']:.2f}")
        else:
            print(f"Trade {i} Error: {result['error']}")
    print()

    print("=" * 70)
    print("FpML Integration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
