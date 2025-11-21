"""Tests for FpML Adapter"""

import pytest
from datetime import date
from pathlib import Path
import tempfile

from neutryx.portfolio.trade_generation.fpml_adapter import (
    FpMLAdapter,
    import_fpml_file,
    export_to_fpml_file,
)


# Sample FpML for IRS
SAMPLE_IRS_FPML = """<?xml version="1.0" encoding="utf-8"?>
<dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation"
              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
              fpmlVersion="5-10">
  <trade>
    <tradeHeader>
      <partyTradeIdentifier>
        <partyReference href="#PARTY1"/>
        <tradeId tradeIdScheme="http://www.example.com/trade-id">IRS-001</tradeId>
      </partyTradeIdentifier>
      <tradeDate>2024-01-15</tradeDate>
    </tradeHeader>
    <swap>
      <swapStream id="fixedLeg">
        <calculationPeriodDates id="fixedLegCalcPeriodDates">
          <effectiveDate>
            <unadjustedDate>2024-01-17</unadjustedDate>
          </effectiveDate>
          <terminationDate>
            <unadjustedDate>2029-01-17</unadjustedDate>
          </terminationDate>
        </calculationPeriodDates>
        <notionalStepSchedule>
          <currency>USD</currency>
          <initialValue>100000000</initialValue>
        </notionalStepSchedule>
        <fixedRateSchedule>
          <initialValue>0.05</initialValue>
        </fixedRateSchedule>
      </swapStream>
      <swapStream id="floatingLeg">
        <calculationPeriodDates id="floatingLegCalcPeriodDates">
          <effectiveDate>
            <unadjustedDate>2024-01-17</unadjustedDate>
          </effectiveDate>
          <terminationDate>
            <unadjustedDate>2029-01-17</unadjustedDate>
          </terminationDate>
        </calculationPeriodDates>
        <notionalStepSchedule>
          <currency>USD</currency>
          <initialValue>100000000</initialValue>
        </notionalStepSchedule>
        <floatingRateCalculation>
          <floatingRateIndex>USD-LIBOR-BBA</floatingRateIndex>
        </floatingRateCalculation>
      </swapStream>
    </swap>
  </trade>
</dataDocument>
"""

# Sample FpML for OIS
SAMPLE_OIS_FPML = """<?xml version="1.0" encoding="utf-8"?>
<dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation"
              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
              fpmlVersion="5-10">
  <trade>
    <tradeHeader>
      <partyTradeIdentifier>
        <partyReference href="#PARTY1"/>
        <tradeId>OIS-001</tradeId>
      </partyTradeIdentifier>
      <tradeDate>2024-01-15</tradeDate>
    </tradeHeader>
    <swap>
      <swapStream id="fixedLeg">
        <calculationPeriodDates>
          <effectiveDate>
            <unadjustedDate>2024-01-17</unadjustedDate>
          </effectiveDate>
          <terminationDate>
            <unadjustedDate>2029-01-17</unadjustedDate>
          </terminationDate>
        </calculationPeriodDates>
        <notionalStepSchedule>
          <currency>USD</currency>
          <initialValue>50000000</initialValue>
        </notionalStepSchedule>
        <fixedRateSchedule>
          <initialValue>0.045</initialValue>
        </fixedRateSchedule>
      </swapStream>
      <swapStream id="floatingLeg">
        <calculationPeriodDates>
          <effectiveDate>
            <unadjustedDate>2024-01-17</unadjustedDate>
          </effectiveDate>
          <terminationDate>
            <unadjustedDate>2029-01-17</unadjustedDate>
          </terminationDate>
        </calculationPeriodDates>
        <notionalStepSchedule>
          <currency>USD</currency>
          <initialValue>50000000</initialValue>
        </notionalStepSchedule>
        <floatingRateCalculation>
          <floatingRateIndex>USD-SOFR</floatingRateIndex>
        </floatingRateCalculation>
      </swapStream>
    </swap>
  </trade>
</dataDocument>
"""


class TestFpMLAdapter:
    """Test FpML Adapter"""

    def test_adapter_creation(self):
        """Test creating FpML adapter"""
        adapter = FpMLAdapter()
        assert adapter is not None
        assert adapter.irs_generator is not None
        assert adapter.ois_generator is not None
        assert adapter.fra_generator is not None

    def test_import_irs_from_fpml_string(self):
        """Test importing IRS from FpML string"""
        adapter = FpMLAdapter()

        results = adapter.import_from_fpml_string(SAMPLE_IRS_FPML)

        assert len(results) == 1
        result = results[0]

        assert result.success is True
        assert result.product_type == "IRS"
        assert result.trade_result is not None
        assert result.trade_result.trade.currency == "USD"
        assert result.trade_result.trade.notional == 100_000_000

    def test_import_ois_from_fpml_string(self):
        """Test importing OIS from FpML string"""
        adapter = FpMLAdapter()

        results = adapter.import_from_fpml_string(SAMPLE_OIS_FPML)

        assert len(results) == 1
        result = results[0]

        assert result.success is True
        assert result.product_type == "OIS"
        assert result.trade_result is not None
        assert result.trade_result.trade.currency == "USD"
        assert result.trade_result.trade.notional == 50_000_000

    def test_import_from_fpml_file(self, tmp_path):
        """Test importing from FpML file"""
        # Create temporary FpML file
        fpml_file = tmp_path / "test_irs.xml"
        with open(fpml_file, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_IRS_FPML)

        adapter = FpMLAdapter()
        results = adapter.import_from_fpml_file(str(fpml_file))

        assert len(results) >= 1
        assert results[0].success is True

    def test_convenience_import_function(self, tmp_path):
        """Test convenience import function"""
        fpml_file = tmp_path / "test_irs.xml"
        with open(fpml_file, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_IRS_FPML)

        results = import_fpml_file(str(fpml_file))

        assert len(results) >= 1
        assert results[0].success is True

    def test_export_irs_to_fpml(self):
        """Test exporting IRS to FpML"""
        # First import a trade
        adapter = FpMLAdapter()
        results = adapter.import_from_fpml_string(SAMPLE_IRS_FPML)
        assert results[0].success is True

        trade = results[0].trade_result.trade

        # Export to FpML
        fpml_xml = adapter.export_to_fpml(trade)

        # Verify basic structure
        assert '<?xml version' in fpml_xml or 'dataDocument' in fpml_xml
        assert 'trade' in fpml_xml
        assert 'swap' in fpml_xml or 'tradeHeader' in fpml_xml

    def test_export_to_fpml_file(self, tmp_path):
        """Test exporting trade to FpML file"""
        # Import a trade
        adapter = FpMLAdapter()
        results = adapter.import_from_fpml_string(SAMPLE_IRS_FPML)
        trade = results[0].trade_result.trade

        # Export to file
        output_file = tmp_path / "exported_trade.xml"
        export_to_fpml_file(trade, str(output_file))

        # Verify file was created
        assert output_file.exists()

        # Verify content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'trade' in content

    def test_roundtrip_import_export(self, tmp_path):
        """Test importing and re-exporting maintains structure"""
        adapter = FpMLAdapter()

        # Import
        results = adapter.import_from_fpml_string(SAMPLE_IRS_FPML)
        assert results[0].success is True
        trade = results[0].trade_result.trade

        # Export
        fpml_xml = adapter.export_to_fpml(trade)

        # Verify export contains key elements
        assert trade.currency in fpml_xml
        assert str(int(trade.notional)) in fpml_xml

    def test_detect_product_type(self):
        """Test product type detection"""
        adapter = FpMLAdapter()

        # Test IRS detection
        irs_results = adapter.import_from_fpml_string(SAMPLE_IRS_FPML)
        assert irs_results[0].product_type == "IRS"

        # Test OIS detection
        ois_results = adapter.import_from_fpml_string(SAMPLE_OIS_FPML)
        assert ois_results[0].product_type == "OIS"

    def test_parse_date(self):
        """Test FpML date parsing"""
        adapter = FpMLAdapter()

        parsed = adapter._parse_date("2024-01-15")
        assert parsed == date(2024, 1, 15)

    def test_invalid_fpml(self):
        """Test handling of invalid FpML"""
        adapter = FpMLAdapter()

        invalid_fpml = """<?xml version="1.0"?>
        <invalid>
            <structure>test</structure>
        </invalid>
        """

        results = adapter.import_from_fpml_string(invalid_fpml)

        # Should return empty list or failed result
        assert len(results) == 0 or not results[0].success

    def test_multiple_trades_in_file(self, tmp_path):
        """Test importing file with single trade (multiple trades would need proper XML structure)"""
        # For now, test with single trade
        fpml_file = tmp_path / "single_trade.xml"
        with open(fpml_file, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_IRS_FPML)

        adapter = FpMLAdapter()
        results = adapter.import_from_fpml_file(str(fpml_file))

        # Should find 1 trade
        assert len(results) == 1
        assert results[0].success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
