"""Tests for EMIR/Dodd-Frank trade reporting."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from neutryx.regulatory.reporting.emir import (
    ClearingStatus,
    EMIRLifecycleEvent,
    EMIRReconciliation,
    EMIRTradeReport,
    EMIRTradeReporter,
    EMIRValuationReport,
    LifecycleEventType,
    ProductClassification,
)


class TestEMIRTradeReport:
    """Test EMIR trade report functionality."""

    def test_create_trade_report(self):
        """Test creating a basic EMIR trade report."""
        report = EMIRTradeReport(
            unique_trade_identifier="UTI123456789",
            reporting_counterparty_lei="12345678901234567890",
            other_counterparty_lei="09876543210987654321",
            product_classification=ProductClassification.INTEREST_RATE,
            notional_amount=Decimal("1000000"),
            notional_currency="USD",
            execution_timestamp=datetime.utcnow(),
        )

        assert report.unique_trade_identifier == "UTI123456789"
        assert report.product_classification == ProductClassification.INTEREST_RATE
        assert report.notional_amount == Decimal("1000000")

    def test_validate_trade_report_success(self):
        """Test validation of a valid trade report."""
        report = EMIRTradeReport(
            unique_trade_identifier="UTI123456789",
            reporting_counterparty_lei="12345678901234567890",
            other_counterparty_lei="09876543210987654321",
            product_classification=ProductClassification.INTEREST_RATE,
            notional_amount=Decimal("1000000"),
            notional_currency="USD",
            execution_timestamp=datetime.utcnow(),
            value_date=datetime.utcnow(),
            maturity_date=datetime.utcnow() + timedelta(days=365),
        )

        assert report.validate()
        assert len(report.errors) == 0

    def test_validate_missing_uti(self):
        """Test validation fails for missing UTI."""
        report = EMIRTradeReport(
            unique_trade_identifier="",
            reporting_counterparty_lei="12345678901234567890",
            other_counterparty_lei="09876543210987654321",
        )

        assert not report.validate()
        assert any("UTI" in error for error in report.errors)

    def test_validate_invalid_lei(self):
        """Test validation fails for invalid LEI."""
        report = EMIRTradeReport(
            unique_trade_identifier="UTI123456789",
            reporting_counterparty_lei="INVALID_LEI",  # Not 20 characters
            other_counterparty_lei="09876543210987654321",
        )

        assert not report.validate()
        assert any("LEI" in error for error in report.errors)

    def test_validate_maturity_before_value_date(self):
        """Test validation fails when maturity is before value date."""
        value_date = datetime.utcnow()
        maturity_date = value_date - timedelta(days=365)

        report = EMIRTradeReport(
            unique_trade_identifier="UTI123456789",
            reporting_counterparty_lei="12345678901234567890",
            other_counterparty_lei="09876543210987654321",
            value_date=value_date,
            maturity_date=maturity_date,
            notional_amount=Decimal("1000000"),
        )

        assert not report.validate()
        assert any("Maturity date" in error for error in report.errors)

    def test_validate_cleared_trade_requires_ccp(self):
        """Test validation fails for cleared trade without CCP."""
        report = EMIRTradeReport(
            unique_trade_identifier="UTI123456789",
            reporting_counterparty_lei="12345678901234567890",
            other_counterparty_lei="09876543210987654321",
            clearing_status=ClearingStatus.CLEARED,
            ccp_lei=None,
            notional_amount=Decimal("1000000"),
        )

        assert not report.validate()
        assert any("CCP" in error for error in report.errors)

    def test_to_xml(self):
        """Test XML generation."""
        report = EMIRTradeReport(
            unique_trade_identifier="UTI123456789",
            reporting_counterparty_lei="12345678901234567890",
            other_counterparty_lei="09876543210987654321",
            product_classification=ProductClassification.INTEREST_RATE,
            notional_amount=Decimal("1000000"),
            notional_currency="USD",
        )

        xml = report.to_xml()
        assert "<?xml version" in xml
        assert "UTI123456789" in xml
        assert "12345678901234567890" in xml
        assert "1000000" in xml


class TestEMIRLifecycleEvent:
    """Test EMIR lifecycle event reporting."""

    def test_create_modification_event(self):
        """Test creating a modification lifecycle event."""
        event = EMIRLifecycleEvent(
            unique_trade_identifier="UTI123456789",
            event_type=LifecycleEventType.MODIFICATION,
            event_timestamp=datetime.utcnow(),
            modified_notional=Decimal("2000000"),
        )

        assert event.event_type == LifecycleEventType.MODIFICATION
        assert event.modified_notional == Decimal("2000000")

    def test_create_termination_event(self):
        """Test creating a termination lifecycle event."""
        event = EMIRLifecycleEvent(
            unique_trade_identifier="UTI123456789",
            event_type=LifecycleEventType.TERMINATION,
            event_timestamp=datetime.utcnow(),
            termination_amount=Decimal("50000"),
            termination_currency="USD",
        )

        assert event.event_type == LifecycleEventType.TERMINATION
        assert event.termination_amount == Decimal("50000")

    def test_validate_novation_requires_new_uti(self):
        """Test novation validation requires new UTI."""
        event = EMIRLifecycleEvent(
            unique_trade_identifier="UTI123456789",
            event_type=LifecycleEventType.NOVATION,
            event_timestamp=datetime.utcnow(),
            new_uti=None,
        )

        assert not event.validate()
        assert any("New UTI" in error for error in event.errors)


class TestEMIRValuationReport:
    """Test EMIR valuation reporting."""

    def test_create_valuation_report(self):
        """Test creating a valuation report."""
        report = EMIRValuationReport(
            unique_trade_identifier="UTI123456789",
            valuation_date=datetime.utcnow(),
            valuation_amount=Decimal("150000"),
            valuation_currency="USD",
            valuation_type="MTOM",
        )

        assert report.valuation_amount == Decimal("150000")
        assert report.valuation_type == "MTOM"

    def test_validate_valuation_report(self):
        """Test validation of valuation report."""
        report = EMIRValuationReport(
            unique_trade_identifier="UTI123456789",
            valuation_date=datetime.utcnow(),
            valuation_amount=Decimal("150000"),
            valuation_currency="USD",
            valuation_type="MTOM",
        )

        assert report.validate()
        assert len(report.errors) == 0

    def test_validate_invalid_valuation_type(self):
        """Test validation fails for invalid valuation type."""
        report = EMIRValuationReport(
            unique_trade_identifier="UTI123456789",
            valuation_date=datetime.utcnow(),
            valuation_amount=Decimal("150000"),
            valuation_type="INVALID",
        )

        assert not report.validate()
        assert any("valuation type" in error for error in report.errors)


class TestEMIRReconciliation:
    """Test EMIR reconciliation functionality."""

    def test_match_rate_calculation(self):
        """Test reconciliation match rate calculation."""
        recon = EMIRReconciliation(
            portfolio_id="PORT123",
            reconciliation_date=datetime.utcnow(),
            counterparty_lei="12345678901234567890",
            total_trades=100,
            matched_trades=95,
            unmatched_trades=5,
        )

        assert recon.match_rate() == 0.95

    def test_requires_dispute_resolution_low_match_rate(self):
        """Test dispute resolution required for low match rate."""
        recon = EMIRReconciliation(
            portfolio_id="PORT123",
            reconciliation_date=datetime.utcnow(),
            counterparty_lei="12345678901234567890",
            total_trades=100,
            matched_trades=90,  # 90% < 95% threshold
            unmatched_trades=10,
        )

        assert recon.requires_dispute_resolution()

    def test_no_dispute_resolution_high_match_rate(self):
        """Test no dispute resolution needed for high match rate."""
        recon = EMIRReconciliation(
            portfolio_id="PORT123",
            reconciliation_date=datetime.utcnow(),
            counterparty_lei="12345678901234567890",
            total_trades=100,
            matched_trades=98,  # 98% >= 95% threshold
            unmatched_trades=2,
        )

        assert not recon.requires_dispute_resolution()


class TestEMIRTradeReporter:
    """Test EMIR trade reporter engine."""

    def test_create_reporter(self):
        """Test creating EMIR trade reporter."""
        reporter = EMIRTradeReporter(
            reporting_lei="12345678901234567890",
            trade_repository="DTCC-GTR",
        )

        assert reporter.reporting_lei == "12345678901234567890"
        assert reporter.trade_repository == "DTCC-GTR"

    def test_create_trade_report(self):
        """Test creating trade report via reporter."""
        reporter = EMIRTradeReporter(
            reporting_lei="12345678901234567890",
        )

        report = reporter.create_trade_report(
            uti="UTI123456789",
            counterparty_lei="09876543210987654321",
            product_classification=ProductClassification.INTEREST_RATE,
            notional_amount=Decimal("1000000"),
        )

        assert report.unique_trade_identifier == "UTI123456789"
        assert report.reporting_counterparty_lei == "12345678901234567890"
        assert "UTI123456789" in reporter.reports

    def test_batch_report_all_valid(self):
        """Test batch reporting with all valid reports."""
        reporter = EMIRTradeReporter(
            reporting_lei="12345678901234567890",
        )

        reports = [
            reporter.create_trade_report(
                uti=f"UTI{i}",
                counterparty_lei="09876543210987654321",
                product_classification=ProductClassification.INTEREST_RATE,
                notional_amount=Decimal("1000000"),
                execution_timestamp=datetime.utcnow(),
            )
            for i in range(5)
        ]

        results = reporter.batch_report(reports)

        assert results["total"] == 5
        assert results["validated"] == 5
        assert results["failed"] == 0

    def test_batch_report_with_failures(self):
        """Test batch reporting with some invalid reports."""
        reporter = EMIRTradeReporter(
            reporting_lei="12345678901234567890",
        )

        # Create mix of valid and invalid reports
        valid_report = reporter.create_trade_report(
            uti="UTI_VALID",
            counterparty_lei="09876543210987654321",
            product_classification=ProductClassification.INTEREST_RATE,
            notional_amount=Decimal("1000000"),
            execution_timestamp=datetime.utcnow(),
        )

        invalid_report = reporter.create_trade_report(
            uti="",  # Missing UTI
            counterparty_lei="INVALID_LEI",
            notional_amount=Decimal("0"),  # Invalid notional
        )

        results = reporter.batch_report([valid_report, invalid_report])

        assert results["total"] == 2
        assert results["validated"] == 1
        assert results["failed"] == 1
        assert len(results["errors"]) == 1
