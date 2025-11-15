"""Tests for Basel III/IV capital reporting."""

from datetime import datetime
from decimal import Decimal

import pytest

from neutryx.regulatory.reporting.basel_reporting import (
    BaselCapitalReport,
    BaselCapitalReporter,
    CVACapitalReport,
    FRTBCapitalReport,
    LeverageRatioReport,
    OperationalRiskReport,
    RiskWeightApproach,
)


class TestCVACapitalReport:
    """Test CVA capital report functionality."""

    def test_create_cva_report(self):
        """Test creating a CVA capital report."""
        report = CVACapitalReport(
            portfolio_id="CVA_PORT_001",
            cva_capital_charge=Decimal("5000000"),
            total_epe=Decimal("20000000"),
            number_of_counterparties=15,
            average_counterparty_pd=0.02,
        )

        assert report.portfolio_id == "CVA_PORT_001"
        assert report.cva_capital_charge == Decimal("5000000")
        assert report.number_of_counterparties == 15

    def test_validate_cva_report_success(self):
        """Test validation of a valid CVA report."""
        report = CVACapitalReport(
            portfolio_id="CVA_PORT_001",
            cva_capital_charge=Decimal("5000000"),
            total_epe=Decimal("20000000"),
            average_counterparty_pd=0.02,
            average_lgd=0.45,
            hedge_effectiveness=0.75,
        )

        assert report.validate()
        assert len(report.errors) == 0

    def test_validate_missing_portfolio_id(self):
        """Test validation fails for missing portfolio ID."""
        report = CVACapitalReport(
            portfolio_id="",
            cva_capital_charge=Decimal("5000000"),
        )

        assert not report.validate()
        assert any("Portfolio ID" in error for error in report.errors)

    def test_validate_negative_capital_charge(self):
        """Test validation fails for negative capital charge."""
        report = CVACapitalReport(
            portfolio_id="CVA_PORT_001",
            cva_capital_charge=Decimal("-5000000"),
        )

        assert not report.validate()
        assert any("cannot be negative" in error for error in report.errors)

    def test_validate_invalid_pd(self):
        """Test validation fails for invalid PD."""
        report = CVACapitalReport(
            portfolio_id="CVA_PORT_001",
            cva_capital_charge=Decimal("5000000"),
            average_counterparty_pd=1.5,  # > 1.0
        )

        assert not report.validate()
        assert any("PD" in error for error in report.errors)

    def test_to_xml(self):
        """Test XML generation."""
        report = CVACapitalReport(
            portfolio_id="CVA_PORT_001",
            cva_capital_charge=Decimal("5000000"),
            total_epe=Decimal("20000000"),
        )

        xml = report.to_xml()
        assert "<?xml version" in xml
        assert "CVA_PORT_001" in xml
        assert "5000000" in xml


class TestFRTBCapitalReport:
    """Test FRTB market risk capital report."""

    def test_create_frtb_report_standardized(self):
        """Test creating FRTB report with Standardized Approach."""
        report = FRTBCapitalReport(
            trading_desk_id="DESK_001",
            approach=RiskWeightApproach.STANDARDIZED,
            delta_charge=Decimal("10000000"),
            vega_charge=Decimal("2000000"),
            curvature_charge=Decimal("3000000"),
            default_risk_charge=Decimal("5000000"),
            residual_risk_add_on=Decimal("1000000"),
            diversification_benefit=Decimal("2000000"),
        )

        assert report.trading_desk_id == "DESK_001"
        assert report.approach == RiskWeightApproach.STANDARDIZED
        # Total = 10M + 2M + 3M + 5M + 1M - 2M = 19M
        assert report.total_market_risk_capital == Decimal("19000000")

    def test_create_frtb_report_ima(self):
        """Test creating FRTB report with Internal Models Approach."""
        report = FRTBCapitalReport(
            trading_desk_id="DESK_002",
            approach=RiskWeightApproach.INTERNAL_MODELS,
            var=Decimal("8000000"),
            stressed_var=Decimal("12000000"),
            incremental_risk_charge=Decimal("3000000"),
        )

        assert report.approach == RiskWeightApproach.INTERNAL_MODELS
        # Total = max(3*VaR, (VaR + sVaR)/2) + IRC
        # = max(24M, 10M) + 3M = 27M
        assert report.total_market_risk_capital == Decimal("27000000")

    def test_validate_frtb_report(self):
        """Test validation of FRTB report."""
        report = FRTBCapitalReport(
            trading_desk_id="DESK_001",
            delta_charge=Decimal("10000000"),
        )

        assert report.validate()
        assert len(report.errors) == 0

    def test_validate_ima_requires_var(self):
        """Test IMA approach requires VaR metrics."""
        report = FRTBCapitalReport(
            trading_desk_id="DESK_002",
            approach=RiskWeightApproach.INTERNAL_MODELS,
            var=None,
            stressed_var=None,
        )

        assert not report.validate()
        assert any("VaR" in error for error in report.errors)


class TestOperationalRiskReport:
    """Test operational risk capital report."""

    def test_create_operational_risk_report(self):
        """Test creating operational risk report."""
        report = OperationalRiskReport(
            interest_income=Decimal("100000000"),
            interest_expense=Decimal("60000000"),
            services_income=Decimal("50000000"),
            financial_income=Decimal("30000000"),
            financial_expense=Decimal("20000000"),
        )

        assert report.business_indicator > 0
        assert report.operational_risk_capital > 0

    def test_business_indicator_calculation(self):
        """Test Business Indicator calculation."""
        report = OperationalRiskReport(
            interest_income=Decimal("100000000"),
            interest_expense=Decimal("60000000"),
            services_income=Decimal("50000000"),
            financial_income=Decimal("30000000"),
            financial_expense=Decimal("20000000"),
        )

        # ILDC = |100M - 60M| = 40M
        # SC = 50M
        # FC = |30M - 20M| = 10M
        # BI = 40M + 50M + 10M = 100M
        assert report.business_indicator == Decimal("100000000")

    def test_operational_capital_calculation_tier1(self):
        """Test operational capital calculation for Tier 1 (BI <= €1bn)."""
        report = OperationalRiskReport(
            interest_income=Decimal("500000000"),
            interest_expense=Decimal("400000000"),
            services_income=Decimal("200000000"),
            internal_loss_multiplier=1.0,
        )

        # BI = 100M + 200M = 300M < 1B
        # BIC = 300M * 0.12 = 36M
        # Capital = 36M * 1.0 = 36M
        expected_capital = Decimal("300000000") * Decimal("0.12")
        assert report.operational_risk_capital == expected_capital

    def test_validate_operational_risk_report(self):
        """Test validation of operational risk report."""
        report = OperationalRiskReport(
            interest_income=Decimal("100000000"),
            interest_expense=Decimal("60000000"),
            services_income=Decimal("50000000"),
        )

        assert report.validate()
        assert len(report.errors) == 0


class TestLeverageRatioReport:
    """Test leverage ratio report."""

    def test_create_leverage_ratio_report(self):
        """Test creating leverage ratio report."""
        report = LeverageRatioReport(
            tier1_capital=Decimal("50000000000"),  # $50B
            on_balance_sheet_exposures=Decimal("800000000000"),  # $800B
            derivative_exposures=Decimal("100000000000"),  # $100B
            securities_financing_exposures=Decimal("50000000000"),  # $50B
            off_balance_sheet_exposures=Decimal("50000000000"),  # $50B
        )

        # Total exposure = 800B + 100B + 50B + 50B = 1000B
        assert report.total_exposure_measure == Decimal("1000000000000")
        # Leverage ratio = 50B / 1000B = 0.05 = 5%
        assert report.leverage_ratio == 0.05

    def test_leverage_ratio_compliant(self):
        """Test compliant leverage ratio."""
        report = LeverageRatioReport(
            tier1_capital=Decimal("50000000000"),
            on_balance_sheet_exposures=Decimal("900000000000"),
            derivative_exposures=Decimal("100000000000"),
        )

        # Ratio = 50B / 1000B = 5% >= 3% minimum
        assert report.is_compliant()

    def test_leverage_ratio_non_compliant(self):
        """Test non-compliant leverage ratio."""
        report = LeverageRatioReport(
            tier1_capital=Decimal("25000000000"),  # $25B
            on_balance_sheet_exposures=Decimal("900000000000"),
            derivative_exposures=Decimal("100000000000"),
        )

        # Ratio = 25B / 1000B = 2.5% < 3% minimum
        assert not report.is_compliant()

    def test_validate_leverage_ratio(self):
        """Test validation of leverage ratio report."""
        report = LeverageRatioReport(
            tier1_capital=Decimal("50000000000"),
            on_balance_sheet_exposures=Decimal("800000000000"),
        )

        assert report.validate()

    def test_validate_non_compliant_generates_warning(self):
        """Test non-compliant ratio generates warning."""
        report = LeverageRatioReport(
            tier1_capital=Decimal("25000000000"),
            on_balance_sheet_exposures=Decimal("900000000000"),
            derivative_exposures=Decimal("100000000000"),
        )

        report.validate()
        assert len(report.warnings) > 0
        assert any("below minimum" in warning for warning in report.warnings)


class TestBaselCapitalReport:
    """Test comprehensive Basel capital report."""

    def test_create_basel_capital_report(self):
        """Test creating comprehensive Basel capital report."""
        report = BaselCapitalReport(
            bank_lei="12345678901234567890",
            cet1_capital=Decimal("100000000000"),  # $100B
            at1_capital=Decimal("20000000000"),  # $20B
            tier2_capital=Decimal("30000000000"),  # $30B
            credit_rwa=Decimal("800000000000"),  # $800B
            market_rwa=Decimal("100000000000"),  # $100B
            operational_rwa=Decimal("50000000000"),  # $50B
            cva_rwa=Decimal("50000000000"),  # $50B
        )

        # Total capital = 100B + 20B + 30B = 150B
        assert report.total_capital == Decimal("150000000000")
        # Total RWA = 800B + 100B + 50B + 50B = 1000B
        assert report.total_rwa == Decimal("1000000000000")
        # CET1 ratio = 100B / 1000B = 10%
        assert report.cet1_ratio == 0.10
        # Tier 1 ratio = 120B / 1000B = 12%
        assert report.tier1_ratio == 0.12
        # Total capital ratio = 150B / 1000B = 15%
        assert report.total_capital_ratio == 0.15

    def test_minimum_requirements(self):
        """Test minimum capital requirements calculation."""
        report = BaselCapitalReport(
            bank_lei="12345678901234567890",
            cet1_capital=Decimal("100000000000"),
            credit_rwa=Decimal("1000000000000"),
            capital_conservation_buffer=0.025,
            countercyclical_buffer=0.01,
            systemic_buffer=0.02,  # G-SIB
        )

        # CET1 min = 4.5% + 2.5% + 1% + 2% = 10%
        assert report.minimum_cet1_requirement() == pytest.approx(0.10)
        # Tier 1 min = 6% + 2.5% + 1% + 2% = 11.5%
        assert report.minimum_tier1_requirement() == pytest.approx(0.115)
        # Total min = 8% + 2.5% + 1% + 2% = 13.5%
        assert report.minimum_total_capital_requirement() == pytest.approx(0.135)

    def test_well_capitalized(self):
        """Test well-capitalized determination."""
        report = BaselCapitalReport(
            bank_lei="12345678901234567890",
            cet1_capital=Decimal("100000000000"),  # 10%
            at1_capital=Decimal("25000000000"),  # 2.5%
            tier2_capital=Decimal("35000000000"),  # 3.5%
            credit_rwa=Decimal("1000000000000"),
            capital_conservation_buffer=0.025,
        )

        # CET1: 10% >= 7% (4.5% + 2.5%)
        # Tier 1: 12.5% >= 8.5% (6% + 2.5%)
        # Total: 16% >= 10.5% (8% + 2.5%)
        assert report.is_well_capitalized()

    def test_not_well_capitalized(self):
        """Test not well-capitalized determination."""
        report = BaselCapitalReport(
            bank_lei="12345678901234567890",
            cet1_capital=Decimal("50000000000"),  # 5%
            at1_capital=Decimal("10000000000"),  # 1%
            tier2_capital=Decimal("20000000000"),  # 2%
            credit_rwa=Decimal("1000000000000"),
            capital_conservation_buffer=0.025,
        )

        # CET1: 5% < 7% (4.5% + 2.5%)
        assert not report.is_well_capitalized()

    def test_validate_basel_report(self):
        """Test validation of Basel capital report."""
        report = BaselCapitalReport(
            bank_lei="12345678901234567890",
            cet1_capital=Decimal("100000000000"),
            credit_rwa=Decimal("1000000000000"),
        )

        assert report.validate()

    def test_validate_invalid_lei(self):
        """Test validation fails for invalid LEI."""
        report = BaselCapitalReport(
            bank_lei="INVALID",  # Not 20 characters
            cet1_capital=Decimal("100000000000"),
        )

        assert not report.validate()
        assert any("LEI" in error for error in report.errors)

    def test_validate_under_capitalized_warning(self):
        """Test under-capitalized generates warning."""
        report = BaselCapitalReport(
            bank_lei="12345678901234567890",
            cet1_capital=Decimal("50000000000"),
            credit_rwa=Decimal("1000000000000"),
        )

        report.validate()
        assert len(report.warnings) > 0
        assert any("capital requirements" in warning for warning in report.warnings)

    def test_to_xml(self):
        """Test Pillar 3 XML generation."""
        report = BaselCapitalReport(
            bank_lei="12345678901234567890",
            cet1_capital=Decimal("100000000000"),
            credit_rwa=Decimal("1000000000000"),
        )

        xml = report.to_xml()
        assert "<?xml version" in xml
        assert "12345678901234567890" in xml
        assert "100000000000" in xml


class TestBaselCapitalReporter:
    """Test Basel capital reporter engine."""

    def test_create_reporter(self):
        """Test creating Basel capital reporter."""
        reporter = BaselCapitalReporter(bank_lei="12345678901234567890")
        assert reporter.bank_lei == "12345678901234567890"
        assert len(reporter.reports) == 0

    def test_create_comprehensive_report(self):
        """Test creating comprehensive report via reporter."""
        reporter = BaselCapitalReporter(bank_lei="12345678901234567890")

        report = reporter.create_comprehensive_report(
            reporting_date=datetime.utcnow(),
            cet1_capital=Decimal("100000000000"),
            credit_rwa=Decimal("1000000000000"),
        )

        assert report.bank_lei == "12345678901234567890"
        assert len(reporter.reports) == 1

    def test_generate_pillar3_disclosure(self):
        """Test generating Pillar 3 disclosure."""
        reporter = BaselCapitalReporter(bank_lei="12345678901234567890")

        report = reporter.create_comprehensive_report(
            reporting_date=datetime.utcnow(),
            cet1_capital=Decimal("100000000000"),
            at1_capital=Decimal("20000000000"),
            tier2_capital=Decimal("30000000000"),
            credit_rwa=Decimal("800000000000"),
            market_rwa=Decimal("100000000000"),
            operational_rwa=Decimal("50000000000"),
            cva_rwa=Decimal("50000000000"),
        )

        disclosure = reporter.generate_pillar3_disclosure(report)

        assert "reporting_date" in disclosure
        assert "bank_lei" in disclosure
        assert "capital_structure" in disclosure
        assert "risk_weighted_assets" in disclosure
        assert "capital_ratios" in disclosure
        assert disclosure["well_capitalized"] is True
