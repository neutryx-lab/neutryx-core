"""Tests for MiFID II/MiFIR transaction reporting."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from neutryx.compliance.mifid import (
    BestExecutionAnalyzer,
    CapacityType,
    InstrumentClassification,
    MiFIDReferenceDataReport,
    MiFIDTransactionReport,
    MiFIDTransactionReporter,
    TransactionType,
    VenueType,
)


class TestMiFIDTransactionReport:
    """Test MiFID II transaction report functionality."""

    def test_create_transaction_report(self):
        """Test creating a basic MiFID transaction report."""
        report = MiFIDTransactionReport(
            transaction_reference_number="TXN123456",
            trading_date_time=datetime.utcnow(),
            instrument_id="US0378331005",  # Apple Inc. ISIN
            buyer_lei="12345678901234567890",
            seller_lei="09876543210987654321",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
            buy_sell_indicator=TransactionType.BUY,
        )

        assert report.transaction_reference_number == "TXN123456"
        assert report.buy_sell_indicator == TransactionType.BUY
        assert report.quantity == Decimal("1000")

    def test_validate_transaction_report_success(self):
        """Test validation of a valid transaction report."""
        report = MiFIDTransactionReport(
            transaction_reference_number="TXN123456",
            trading_date_time=datetime.utcnow(),
            instrument_id="US0378331005",
            buyer_lei="12345678901234567890",
            seller_lei="09876543210987654321",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
        )

        assert report.validate()
        assert len(report.errors) == 0

    def test_validate_missing_transaction_ref(self):
        """Test validation fails for missing transaction reference."""
        report = MiFIDTransactionReport(
            transaction_reference_number="",
            trading_date_time=datetime.utcnow(),
            instrument_id="US0378331005",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
        )

        assert not report.validate()
        assert any("reference" in error for error in report.errors)

    def test_validate_invalid_lei(self):
        """Test validation fails for invalid LEI."""
        report = MiFIDTransactionReport(
            transaction_reference_number="TXN123456",
            trading_date_time=datetime.utcnow(),
            instrument_id="US0378331005",
            buyer_lei="INVALID",  # Not 20 characters
            seller_lei="09876543210987654321",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
        )

        assert not report.validate()
        assert any("LEI" in error for error in report.errors)

    def test_validate_negative_quantity(self):
        """Test validation fails for negative quantity."""
        report = MiFIDTransactionReport(
            transaction_reference_number="TXN123456",
            trading_date_time=datetime.utcnow(),
            instrument_id="US0378331005",
            buyer_lei="12345678901234567890",
            seller_lei="09876543210987654321",
            quantity=Decimal("-1000"),
            price=Decimal("150.50"),
        )

        assert not report.validate()
        assert any("Quantity" in error for error in report.errors)

    def test_validate_cleared_derivative_requires_ccp(self):
        """Test cleared derivative requires CCP LEI."""
        report = MiFIDTransactionReport(
            transaction_reference_number="TXN123456",
            trading_date_time=datetime.utcnow(),
            instrument_id="US0378331005",
            buyer_lei="12345678901234567890",
            seller_lei="09876543210987654321",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
            derivative_cleared=True,
            ccp_lei=None,
        )

        assert not report.validate()
        assert any("CCP" in error for error in report.errors)

    def test_to_xml(self):
        """Test XML generation."""
        report = MiFIDTransactionReport(
            transaction_reference_number="TXN123456",
            trading_date_time=datetime.utcnow(),
            instrument_id="US0378331005",
            buyer_lei="12345678901234567890",
            seller_lei="09876543210987654321",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
        )

        xml = report.to_xml()
        assert "<?xml version" in xml
        assert "TXN123456" in xml
        assert "US0378331005" in xml
        assert "1000" in xml


class TestMiFIDReferenceDataReport:
    """Test MiFID II reference data reporting."""

    def test_create_reference_data_report(self):
        """Test creating a reference data report."""
        report = MiFIDReferenceDataReport(
            isin="US0378331005",
            instrument_full_name="Apple Inc. Common Stock",
            instrument_classification=InstrumentClassification.EQUITY,
            trading_venue_mic="XNAS",  # NASDAQ
            issuer_lei="HWUPKR0MPOU8FGXBT394",
        )

        assert report.isin == "US0378331005"
        assert report.instrument_classification == InstrumentClassification.EQUITY
        assert report.trading_venue_mic == "XNAS"

    def test_validate_reference_data(self):
        """Test validation of reference data report."""
        report = MiFIDReferenceDataReport(
            isin="US0378331005",
            instrument_full_name="Apple Inc. Common Stock",
            trading_venue_mic="XNAS",
            issuer_lei="HWUPKR0MPOU8FGXBT394",
        )

        assert report.validate()
        assert len(report.errors) == 0

    def test_validate_invalid_isin(self):
        """Test validation fails for invalid ISIN."""
        report = MiFIDReferenceDataReport(
            isin="INVALID",  # Not 12 characters
            instrument_full_name="Test Instrument",
            trading_venue_mic="XNAS",
        )

        assert not report.validate()
        assert any("ISIN" in error for error in report.errors)

    def test_validate_invalid_mic(self):
        """Test validation fails for invalid MIC."""
        report = MiFIDReferenceDataReport(
            isin="US0378331005",
            instrument_full_name="Test Instrument",
            trading_venue_mic="TOOLONG",  # Not 4 characters
        )

        assert not report.validate()
        assert any("MIC" in error for error in report.errors)


class TestBestExecutionAnalyzer:
    """Test best execution analysis."""

    def test_create_analyzer(self):
        """Test creating best execution analyzer."""
        analyzer = BestExecutionAnalyzer(firm_lei="12345678901234567890")
        assert analyzer.firm_lei == "12345678901234567890"
        assert len(analyzer.execution_data) == 0

    def test_add_execution(self):
        """Test adding execution data."""
        analyzer = BestExecutionAnalyzer(firm_lei="12345678901234567890")

        analyzer.add_execution(
            venue="XNAS",
            instrument_class=InstrumentClassification.EQUITY,
            execution_time=5.0,
            spread_bps=2.5,
            price_improvement_bps=0.5,
        )

        assert len(analyzer.execution_data) == 1
        assert analyzer.execution_data[0]["venue"] == "XNAS"
        assert analyzer.execution_data[0]["execution_time"] == 5.0

    def test_generate_report(self):
        """Test generating best execution report."""
        analyzer = BestExecutionAnalyzer(firm_lei="12345678901234567890")

        # Add sample executions
        start_date = datetime.utcnow() - timedelta(days=30)
        for i in range(10):
            analyzer.add_execution(
                venue="XNAS" if i % 2 == 0 else "XLON",
                instrument_class=InstrumentClassification.EQUITY,
                execution_time=5.0 + i * 0.5,
                spread_bps=2.0 + i * 0.1,
                price_improvement_bps=0.5 if i % 3 == 0 else 0.0,
            )

        end_date = datetime.utcnow()
        report = analyzer.generate_report(start_date, end_date)

        assert report.total_orders == 10
        assert report.executed_orders == 10
        assert report.average_execution_time_seconds > 0
        assert report.average_spread_bps > 0
        assert "XNAS" in report.venue_concentration
        assert "XLON" in report.venue_concentration

    def test_execution_rate(self):
        """Test execution rate calculation."""
        analyzer = BestExecutionAnalyzer(firm_lei="12345678901234567890")

        start_date = datetime.utcnow() - timedelta(days=1)
        for _ in range(10):
            analyzer.add_execution(
                venue="XNAS",
                instrument_class=InstrumentClassification.EQUITY,
                execution_time=5.0,
                spread_bps=2.0,
            )

        end_date = datetime.utcnow()
        report = analyzer.generate_report(start_date, end_date)

        assert report.execution_rate() == 1.0  # All executed

    def test_best_execution_criteria(self):
        """Test best execution criteria evaluation."""
        analyzer = BestExecutionAnalyzer(firm_lei="12345678901234567890")

        start_date = datetime.utcnow() - timedelta(days=1)
        # Add high-quality executions
        for _ in range(20):
            analyzer.add_execution(
                venue="XNAS",
                instrument_class=InstrumentClassification.EQUITY,
                execution_time=5.0,  # Fast execution
                spread_bps=3.0,  # Tight spreads
                price_improvement_bps=1.0,
            )

        end_date = datetime.utcnow()
        report = analyzer.generate_report(start_date, end_date)

        assert report.meets_best_execution_criteria()


class TestMiFIDTransactionReporter:
    """Test MiFID transaction reporter engine."""

    def test_create_reporter(self):
        """Test creating MiFID transaction reporter."""
        reporter = MiFIDTransactionReporter(
            firm_lei="12345678901234567890",
            competent_authority="FCA",
        )

        assert reporter.firm_lei == "12345678901234567890"
        assert reporter.competent_authority == "FCA"

    def test_create_transaction_report(self):
        """Test creating transaction report via reporter."""
        reporter = MiFIDTransactionReporter(
            firm_lei="12345678901234567890",
        )

        report = reporter.create_transaction_report(
            transaction_ref="TXN123",
            instrument_isin="US0378331005",
            buyer_lei="12345678901234567890",
            seller_lei="09876543210987654321",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
        )

        assert report.transaction_reference_number == "TXN123"
        assert "TXN123" in reporter.reports

    def test_batch_submit_all_valid(self):
        """Test batch submission with all valid reports."""
        reporter = MiFIDTransactionReporter(
            firm_lei="12345678901234567890",
        )

        reports = [
            reporter.create_transaction_report(
                transaction_ref=f"TXN{i}",
                instrument_isin="US0378331005",
                buyer_lei="12345678901234567890",
                seller_lei="09876543210987654321",
                quantity=Decimal("1000"),
                price=Decimal("150.50"),
            )
            for i in range(5)
        ]

        results = reporter.batch_submit(reports)

        assert results["total"] == 5
        assert results["validated"] == 5
        assert results["failed"] == 0

    def test_batch_submit_with_failures(self):
        """Test batch submission with some invalid reports."""
        reporter = MiFIDTransactionReporter(
            firm_lei="12345678901234567890",
        )

        valid_report = reporter.create_transaction_report(
            transaction_ref="TXN_VALID",
            instrument_isin="US0378331005",
            buyer_lei="12345678901234567890",
            seller_lei="09876543210987654321",
            quantity=Decimal("1000"),
            price=Decimal("150.50"),
        )

        invalid_report = reporter.create_transaction_report(
            transaction_ref="",  # Missing
            instrument_isin="",  # Missing
            quantity=Decimal("-100"),  # Negative
            price=Decimal("-50"),  # Negative
        )

        results = reporter.batch_submit([valid_report, invalid_report])

        assert results["total"] == 2
        assert results["validated"] == 1
        assert results["failed"] == 1
        assert len(results["errors"]) == 1
