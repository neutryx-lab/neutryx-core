"""Tests for FpML integration."""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from neutryx.integrations import fpml


# Sample FpML documents for testing

EQUITY_OPTION_FPML = """<?xml version="1.0" encoding="UTF-8"?>
<dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation" fpmlVersion="5-0">
    <party id="party1">
        <partyName>Bank ABC</partyName>
    </party>
    <party id="party2">
        <partyName>Counterparty XYZ</partyName>
    </party>
    <trade>
        <tradeHeader>
            <tradeDate>2024-01-15</tradeDate>
        </tradeHeader>
        <equityOption>
            <buyerPartyReference href="party1"/>
            <sellerPartyReference href="party2"/>
            <optionType>Call</optionType>
            <underlyer>
                <instrumentId>US0378331005</instrumentId>
                <description>Apple Inc.</description>
            </underlyer>
            <numberOfOptions>100</numberOfOptions>
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
    </trade>
</dataDocument>
"""

FX_OPTION_FPML = """<?xml version="1.0" encoding="UTF-8"?>
<dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation" fpmlVersion="5-0">
    <party id="party1">
        <partyName>Bank ABC</partyName>
    </party>
    <party id="party2">
        <partyName>Counterparty XYZ</partyName>
    </party>
    <trade>
        <tradeHeader>
            <tradeDate>2024-01-15</tradeDate>
        </tradeHeader>
        <fxOption>
            <buyerPartyReference href="party1"/>
            <sellerPartyReference href="party2"/>
            <putCurrencyAmount>
                <currency>EUR</currency>
                <amount>1000000</amount>
            </putCurrencyAmount>
            <callCurrencyAmount>
                <currency>USD</currency>
                <amount>1100000</amount>
            </callCurrencyAmount>
            <strike>
                <rate>1.10</rate>
            </strike>
            <europeanExercise>
                <expiryDate>2025-01-15</expiryDate>
            </europeanExercise>
        </fxOption>
    </trade>
</dataDocument>
"""


class TestFpMLParser:
    """Test FpML parsing functionality."""

    def test_parse_equity_option(self):
        """Test parsing equity option FpML."""
        doc = fpml.parse_fpml(EQUITY_OPTION_FPML)

        assert len(doc.party) == 2
        assert doc.party[0].id == "party1"
        assert doc.party[0].name == "Bank ABC"

        trade = doc.primary_trade
        assert trade.tradeHeader.tradeDate == date(2024, 1, 15)

        assert trade.equityOption is not None
        option = trade.equityOption
        assert option.optionType == fpml.PutCallEnum.CALL
        assert option.underlyer.instrumentId == "US0378331005"
        assert option.strike.strikePrice == Decimal("150.00")
        assert option.numberOfOptions == Decimal("100")

    def test_parse_fx_option(self):
        """Test parsing FX option FpML."""
        doc = fpml.parse_fpml(FX_OPTION_FPML)

        trade = doc.primary_trade
        assert trade.fxOption is not None
        option = trade.fxOption

        assert option.putCurrencyAmount.currency == fpml.CurrencyCode.EUR
        assert option.putCurrencyAmount.amount == Decimal("1000000")
        assert option.callCurrencyAmount.currency == fpml.CurrencyCode.USD
        assert option.strike.rate == Decimal("1.10")
        assert option.fxExercise.expiryDate == date(2025, 1, 15)

    def test_parse_invalid_xml(self):
        """Test parsing invalid XML."""
        with pytest.raises(fpml.FpMLParseError):
            fpml.parse_fpml("<invalid>xml")

    def test_parse_empty_document(self):
        """Test parsing document without trades."""
        empty_doc = """<?xml version="1.0" encoding="UTF-8"?>
        <dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation">
            <party id="party1"/>
        </dataDocument>
        """
        with pytest.raises(fpml.FpMLParseError, match="No valid trades"):
            fpml.parse_fpml(empty_doc)


class TestFpMLMapping:
    """Test FpML to Neutryx mapping."""

    def test_map_equity_option_to_neutryx(self):
        """Test mapping equity option to Neutryx parameters."""
        doc = fpml.parse_fpml(EQUITY_OPTION_FPML)
        market_data = {
            "spot": 145.0,
            "volatility": 0.25,
            "rate": 0.05,
            "dividend": 0.01,
        }

        params = fpml.fpml_to_neutryx(doc, market_data)

        assert isinstance(params, dict)
        assert params["spot"] == 145.0
        assert params["strike"] == 150.0
        assert params["volatility"] == 0.25
        assert params["rate"] == 0.05
        assert params["dividend"] == 0.01
        assert params["call"] is True
        # Maturity should be ~1 year (from 2024-01-15 to 2025-01-15)
        assert 0.99 < params["maturity"] < 1.01

    def test_map_fx_option_to_neutryx(self):
        """Test mapping FX option to Neutryx parameters."""
        doc = fpml.parse_fpml(FX_OPTION_FPML)
        market_data = {
            "spot_rate": 1.08,
            "volatility": 0.15,
            "domestic_rate": 0.04,
            "foreign_rate": 0.02,
        }

        params = fpml.fpml_to_neutryx(doc, market_data)

        assert isinstance(params, dict)
        assert params["spot"] == 1.08
        assert params["strike"] == 1.10
        assert params["volatility"] == 0.15

    def test_map_without_required_market_data(self):
        """Test mapping fails without required market data."""
        doc = fpml.parse_fpml(EQUITY_OPTION_FPML)
        market_data = {}  # Missing spot and volatility

        with pytest.raises(fpml.FpMLMappingError, match="Spot price is required"):
            fpml.fpml_to_neutryx(doc, market_data)

    def test_map_expired_option(self):
        """Test mapping expired option raises error."""
        doc = fpml.parse_fpml(EQUITY_OPTION_FPML)
        mapper = fpml.FpMLToNeutryxMapper(reference_date=date(2026, 1, 1))

        with pytest.raises(fpml.FpMLMappingError, match="expired"):
            mapper.map_trade(
                doc.primary_trade, {"spot": 100.0, "volatility": 0.2}
            )


class TestFpMLSerializer:
    """Test FpML serialization."""

    def test_serialize_equity_option(self):
        """Test serializing equity option to FpML."""
        params = {
            "spot": 100.0,
            "strike": 105.0,
            "maturity": 1.0,
            "rate": 0.05,
            "dividend": 0.01,
            "volatility": 0.25,
            "call": True,
            "steps": 252,
            "paths": 100000,
        }

        fpml_doc = fpml.neutryx_to_fpml(
            params, instrument_id="US0378331005", trade_date=date(2024, 1, 15)
        )

        assert len(fpml_doc.party) == 2
        assert len(fpml_doc.trade) == 1

        trade = fpml_doc.primary_trade
        assert trade.equityOption is not None
        assert trade.equityOption.optionType == fpml.PutCallEnum.CALL
        assert trade.equityOption.strike.strikePrice == Decimal("105.0")

    def test_serialize_to_xml(self):
        """Test full serialization to XML string."""
        params = {
            "spot": 100.0,
            "strike": 105.0,
            "maturity": 1.0,
            "volatility": 0.25,
            "rate": 0.05,
            "dividend": 0.0,
            "call": False,  # Put option
            "steps": 252,
            "paths": 100000,
        }

        fpml_doc = fpml.neutryx_to_fpml(params, "TEST_INSTRUMENT")
        xml_string = fpml.serialize_fpml(fpml_doc)

        # Verify it's valid XML and can be parsed back
        assert "<?xml" in xml_string
        assert "equityOption" in xml_string
        assert "Put" in xml_string

        # Round-trip test
        reparsed = fpml.parse_fpml(xml_string)
        assert reparsed.primary_trade.equityOption is not None
        assert reparsed.primary_trade.equityOption.optionType == fpml.PutCallEnum.PUT


class TestFpMLRoundTrip:
    """Test round-trip conversions."""

    def test_roundtrip_equity_option(self):
        """Test Neutryx -> FpML -> Neutryx round trip."""
        original_params = {
            "spot": 100.0,
            "strike": 110.0,
            "maturity": 0.5,
            "rate": 0.03,
            "dividend": 0.02,
            "volatility": 0.30,
            "call": True,
            "steps": 252,
            "paths": 100000,
        }

        # Convert to FpML
        trade_date = date.today()
        fpml_doc = fpml.neutryx_to_fpml(original_params, "INST123", trade_date)
        xml_string = fpml.serialize_fpml(fpml_doc)

        # Parse back
        reparsed_doc = fpml.parse_fpml(xml_string)

        # Convert back to Neutryx
        market_data = {
            "spot": 100.0,
            "volatility": 0.30,
            "rate": 0.03,
            "dividend": 0.02,
        }
        mapper = fpml.FpMLToNeutryxMapper(reference_date=trade_date)
        converted_params = mapper.map_trade(reparsed_doc.primary_trade, market_data)

        # Verify key parameters match
        assert converted_params["strike"] == original_params["strike"]
        assert converted_params["spot"] == original_params["spot"]
        assert converted_params["volatility"] == original_params["volatility"]
        assert converted_params["call"] == original_params["call"]
        # Maturity might differ slightly due to date arithmetic
        assert abs(converted_params["maturity"] - original_params["maturity"]) < 0.01


class TestFpMLAdapter:
    """Test high-level FpML adapter."""

    def test_adapter_price_from_xml(self):
        """Test pricing directly from XML."""
        from neutryx.integrations.adapters.fpml_adapter import FpMLPricingAdapter

        adapter = FpMLPricingAdapter(seed=42)
        market_data = {
            "spot": 145.0,
            "volatility": 0.25,
            "rate": 0.05,
            "dividend": 0.01,
        }

        result = adapter.price_from_xml(EQUITY_OPTION_FPML, market_data)

        assert "price" in result
        assert result["price"] > 0
        assert "trade" in result
        assert "request" in result

    def test_adapter_export_to_fpml(self):
        """Test exporting parameters to FpML."""
        from neutryx.integrations.adapters.fpml_adapter import FpMLPricingAdapter

        adapter = FpMLPricingAdapter()
        xml_string = adapter.export_to_fpml(
            spot=100.0,
            strike=105.0,
            maturity=1.0,
            volatility=0.25,
            is_call=True,
            instrument_id="TEST123",
        )

        assert "<?xml" in xml_string
        assert "equityOption" in xml_string
        assert "105" in xml_string

    def test_quick_price_fpml(self):
        """Test quick pricing convenience function."""
        from neutryx.integrations.adapters.fpml_adapter import quick_price_fpml

        market_data = {
            "spot": 145.0,
            "volatility": 0.25,
            "rate": 0.05,
        }

        price = quick_price_fpml(EQUITY_OPTION_FPML, market_data)
        assert price > 0

    def test_validate_fpml(self):
        """Test FpML validation."""
        from neutryx.integrations.adapters.fpml_adapter import validate_fpml

        assert validate_fpml(EQUITY_OPTION_FPML) is True
        assert validate_fpml("<invalid>xml") is False


class TestFpMLSwap:
    """Test FpML swap integration."""

    def test_map_swap(self):
        """Test mapping FpML swap to pricing result."""
        # Create a simple swap FpML document
        swap_fpml = """<?xml version="1.0" encoding="UTF-8"?>
        <dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation" fpmlVersion="5-0">
            <party id="party1">
                <partyName>Bank A</partyName>
            </party>
            <party id="party2">
                <partyName>Bank B</partyName>
            </party>
            <trade>
                <tradeHeader>
                    <tradeDate>2024-01-15</tradeDate>
                </tradeHeader>
                <swap>
                    <swapStream>
                        <payerPartyReference href="party1"/>
                        <receiverPartyReference href="party2"/>
                        <calculationPeriodDates>
                            <effectiveDate><unadjustedDate>2024-01-15</unadjustedDate></effectiveDate>
                            <terminationDate><unadjustedDate>2029-01-15</unadjustedDate></terminationDate>
                            <calculationPeriodFrequency>
                                <periodMultiplier>6</periodMultiplier>
                                <period>M</period>
                            </calculationPeriodFrequency>
                        </calculationPeriodDates>
                        <paymentDates>
                            <paymentFrequency>
                                <periodMultiplier>6</periodMultiplier>
                                <period>M</period>
                            </paymentFrequency>
                        </paymentDates>
                        <calculationPeriodAmount>
                            <calculation>
                                <notionalSchedule>
                                    <notionalStepSchedule>
                                        <currency>USD</currency>
                                        <initialValue>10000000</initialValue>
                                    </notionalStepSchedule>
                                </notionalSchedule>
                                <fixedRateSchedule>
                                    <initialValue>0.05</initialValue>
                                </fixedRateSchedule>
                                <dayCountFraction>ACT/360</dayCountFraction>
                            </calculation>
                        </calculationPeriodAmount>
                    </swapStream>
                    <swapStream>
                        <payerPartyReference href="party2"/>
                        <receiverPartyReference href="party1"/>
                        <calculationPeriodDates>
                            <effectiveDate><unadjustedDate>2024-01-15</unadjustedDate></effectiveDate>
                            <terminationDate><unadjustedDate>2029-01-15</unadjustedDate></terminationDate>
                            <calculationPeriodFrequency>
                                <periodMultiplier>6</periodMultiplier>
                                <period>M</period>
                            </calculationPeriodFrequency>
                        </calculationPeriodDates>
                        <paymentDates>
                            <paymentFrequency>
                                <periodMultiplier>6</periodMultiplier>
                                <period>M</period>
                            </paymentFrequency>
                        </paymentDates>
                        <calculationPeriodAmount>
                            <calculation>
                                <notionalSchedule>
                                    <notionalStepSchedule>
                                        <currency>USD</currency>
                                        <initialValue>10000000</initialValue>
                                    </notionalStepSchedule>
                                </notionalSchedule>
                                <floatingRateCalculation>
                                    <floatingRateIndex>USD-LIBOR</floatingRateIndex>
                                    <dayCountFraction>ACT/360</dayCountFraction>
                                </floatingRateCalculation>
                                <dayCountFraction>ACT/360</dayCountFraction>
                            </calculation>
                        </calculationPeriodAmount>
                    </swapStream>
                </swap>
            </trade>
        </dataDocument>
        """

        doc = fpml.parse_fpml(swap_fpml)
        market_data = {"discount_rate": 0.05}

        result = fpml.fpml_to_neutryx(doc, market_data)

        # Result should be a dict for swaps
        assert isinstance(result, dict)
        assert "value" in result
        assert "notional" in result
        assert "fixed_rate" in result
        assert result["notional"] == 10_000_000
        assert result["fixed_rate"] == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
