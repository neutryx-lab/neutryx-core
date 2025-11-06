"""Basel III/IV capital reporting.

This module implements capital reporting requirements for:
- CVA capital charge (SA-CVA)
- Market risk capital (FRTB - Fundamental Review of the Trading Book)
- Operational risk capital (Standardized Approach)
- Leverage ratio reporting
- Pillar 3 disclosures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from .report_engine import RegulatoryReport, ReportStatus, ReportType


class CapitalComponent(str, Enum):
    """Basel capital components."""

    CET1 = "CET1"  # Common Equity Tier 1
    AT1 = "AT1"  # Additional Tier 1
    TIER2 = "T2"  # Tier 2


class RiskWeightApproach(str, Enum):
    """Risk weight calculation approaches."""

    STANDARDIZED = "SA"
    INTERNAL_MODELS = "IMA"
    FOUNDATION_IRB = "F-IRB"
    ADVANCED_IRB = "A-IRB"


@dataclass
class CVACapitalReport(RegulatoryReport):
    """CVA (Credit Valuation Adjustment) capital report.

    Implements SA-CVA (Standardized Approach for CVA capital charge).
    """

    # Portfolio identification
    portfolio_id: str = ""
    reporting_date: datetime = field(default_factory=datetime.utcnow)

    # CVA metrics
    cva_capital_charge: Decimal = Decimal("0")
    reduced_cva_capital: Decimal = Decimal("0")
    cva_var: Decimal = Decimal("0")  # CVA VaR if using IMA
    cva_stressed_var: Decimal = Decimal("0")

    # Counterparty exposures
    total_epe: Decimal = Decimal("0")  # Expected Positive Exposure
    total_ene: Decimal = Decimal("0")  # Expected Negative Exposure
    number_of_counterparties: int = 0

    # Risk weights
    average_counterparty_pd: float = 0.0  # Probability of Default
    average_lgd: float = 0.45  # Loss Given Default (45% per Basel)
    maturity_adjustment: float = 1.0

    # Hedging
    cva_hedges: Decimal = Decimal("0")
    hedge_effectiveness: float = 0.0

    def __post_init__(self):
        """Initialize CVA capital report."""
        self.report_type = ReportType.BASEL_CVA
        if not self.data:
            self.data = self._build_data_dict()

    def _build_data_dict(self) -> Dict[str, Any]:
        """Build data dictionary."""
        return {
            "portfolio_id": self.portfolio_id,
            "reporting_date": self.reporting_date.isoformat(),
            "cva_capital_charge": str(self.cva_capital_charge),
            "total_epe": str(self.total_epe),
            "number_of_counterparties": self.number_of_counterparties,
            "average_pd": self.average_counterparty_pd,
            "average_lgd": self.average_lgd,
            "cva_hedges": str(self.cva_hedges),
        }

    def validate(self) -> bool:
        """Validate CVA capital report."""
        super().validate()

        if not self.portfolio_id:
            self.errors.append("Portfolio ID is required")

        if self.cva_capital_charge < 0:
            self.errors.append("CVA capital charge cannot be negative")

        if not (0 <= self.average_counterparty_pd <= 1):
            self.errors.append("Average PD must be between 0 and 1")

        if not (0 <= self.average_lgd <= 1):
            self.errors.append("Average LGD must be between 0 and 1")

        if not (0 <= self.hedge_effectiveness <= 1):
            self.errors.append("Hedge effectiveness must be between 0 and 1")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to XML for regulatory reporting."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<CVACapitalReport>
  <PortfolioID>{self.portfolio_id}</PortfolioID>
  <ReportingDate>{self.reporting_date.date().isoformat()}</ReportingDate>
  <CVACapitalCharge>{self.cva_capital_charge}</CVACapitalCharge>
  <TotalEPE>{self.total_epe}</TotalEPE>
  <NumberOfCounterparties>{self.number_of_counterparties}</NumberOfCounterparties>
  <AveragePD>{self.average_counterparty_pd}</AveragePD>
  <AverageLGD>{self.average_lgd}</AverageLGD>
  <CVAHedges>{self.cva_hedges}</CVAHedges>
</CVACapitalReport>"""


@dataclass
class FRTBCapitalReport(RegulatoryReport):
    """FRTB (Fundamental Review of the Trading Book) market risk capital report."""

    # Portfolio identification
    trading_desk_id: str = ""
    reporting_date: datetime = field(default_factory=datetime.utcnow)
    approach: RiskWeightApproach = RiskWeightApproach.STANDARDIZED

    # Standardized Approach (SA) components
    delta_charge: Decimal = Decimal("0")
    vega_charge: Decimal = Decimal("0")
    curvature_charge: Decimal = Decimal("0")
    default_risk_charge: Decimal = Decimal("0")
    residual_risk_add_on: Decimal = Decimal("0")

    # Internal Models Approach (IMA) - if applicable
    var: Optional[Decimal] = None
    stressed_var: Optional[Decimal] = None
    incremental_risk_charge: Optional[Decimal] = None

    # Total market risk capital
    total_market_risk_capital: Decimal = Decimal("0")

    # Risk factor sensitivities
    interest_rate_risk: Decimal = Decimal("0")
    fx_risk: Decimal = Decimal("0")
    equity_risk: Decimal = Decimal("0")
    commodity_risk: Decimal = Decimal("0")
    credit_spread_risk: Decimal = Decimal("0")

    # Diversification benefit
    diversification_benefit: Decimal = Decimal("0")

    def __post_init__(self):
        """Initialize FRTB report."""
        self.report_type = ReportType.BASEL_FRTB
        if not self.data:
            self.data = self._build_data_dict()
        # Calculate total if not provided
        if self.total_market_risk_capital == Decimal("0"):
            self.total_market_risk_capital = self._calculate_total()

    def _calculate_total(self) -> Decimal:
        """Calculate total market risk capital."""
        if self.approach == RiskWeightApproach.STANDARDIZED:
            # SA total = sum of all charges
            return (
                self.delta_charge +
                self.vega_charge +
                self.curvature_charge +
                self.default_risk_charge +
                self.residual_risk_add_on -
                self.diversification_benefit
            )
        elif self.approach == RiskWeightApproach.INTERNAL_MODELS:
            # IMA total
            if self.var and self.stressed_var:
                return max(
                    self.var * Decimal("3"),  # VaR multiplier
                    (self.var + self.stressed_var) / Decimal("2"),
                ) + (self.incremental_risk_charge or Decimal("0"))
        return Decimal("0")

    def _build_data_dict(self) -> Dict[str, Any]:
        """Build data dictionary."""
        return {
            "trading_desk_id": self.trading_desk_id,
            "reporting_date": self.reporting_date.isoformat(),
            "approach": self.approach.value,
            "delta_charge": str(self.delta_charge),
            "vega_charge": str(self.vega_charge),
            "curvature_charge": str(self.curvature_charge),
            "default_risk_charge": str(self.default_risk_charge),
            "total_market_risk_capital": str(self.total_market_risk_capital),
        }

    def validate(self) -> bool:
        """Validate FRTB report."""
        super().validate()

        if not self.trading_desk_id:
            self.errors.append("Trading desk ID is required")

        if self.total_market_risk_capital < 0:
            self.errors.append("Total market risk capital cannot be negative")

        if self.approach == RiskWeightApproach.INTERNAL_MODELS:
            if self.var is None or self.stressed_var is None:
                self.errors.append("VaR and Stressed VaR required for IMA approach")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to XML for regulatory reporting."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<FRTBCapitalReport>
  <TradingDeskID>{self.trading_desk_id}</TradingDeskID>
  <ReportingDate>{self.reporting_date.date().isoformat()}</ReportingDate>
  <Approach>{self.approach.value}</Approach>
  <DeltaCharge>{self.delta_charge}</DeltaCharge>
  <VegaCharge>{self.vega_charge}</VegaCharge>
  <CurvatureCharge>{self.curvature_charge}</CurvatureCharge>
  <DefaultRiskCharge>{self.default_risk_charge}</DefaultRiskCharge>
  <TotalMarketRiskCapital>{self.total_market_risk_capital}</TotalMarketRiskCapital>
</FRTBCapitalReport>"""


@dataclass
class OperationalRiskReport(RegulatoryReport):
    """Operational risk capital report (Standardized Approach)."""

    reporting_date: datetime = field(default_factory=datetime.utcnow)

    # Business Indicator (BI) components
    interest_income: Decimal = Decimal("0")
    interest_expense: Decimal = Decimal("0")
    services_income: Decimal = Decimal("0")
    financial_income: Decimal = Decimal("0")
    financial_expense: Decimal = Decimal("0")
    net_pnl_trading_book: Decimal = Decimal("0")
    net_pnl_banking_book: Decimal = Decimal("0")

    # Calculated values
    business_indicator: Decimal = Decimal("0")
    business_indicator_component: Decimal = Decimal("0")
    internal_loss_multiplier: float = 1.0

    # Operational risk capital
    operational_risk_capital: Decimal = Decimal("0")

    def __post_init__(self):
        """Initialize operational risk report."""
        self.report_type = ReportType.BASEL_CAPITAL
        if self.business_indicator == Decimal("0"):
            self.business_indicator = self._calculate_bi()
        if self.operational_risk_capital == Decimal("0"):
            self.operational_risk_capital = self._calculate_capital()

    def _calculate_bi(self) -> Decimal:
        """Calculate Business Indicator per Basel III."""
        # Simplified BI calculation
        # BI = ILDC + SC + FC
        # ILDC = |Interest Income - Interest Expense|
        ildc = abs(self.interest_income - self.interest_expense)
        # SC = Services Income
        sc = self.services_income
        # FC = |Financial Income - Financial Expense + Net P&L|
        fc = abs(
            self.financial_income -
            self.financial_expense +
            self.net_pnl_trading_book +
            self.net_pnl_banking_book
        )
        return ildc + sc + fc

    def _calculate_capital(self) -> Decimal:
        """Calculate operational risk capital charge."""
        bi = self.business_indicator

        # Business Indicator Component (BIC) - marginal coefficients
        if bi <= Decimal("1_000_000_000"):  # €1bn
            bic = bi * Decimal("0.12")
        elif bi <= Decimal("30_000_000_000"):  # €30bn
            bic = Decimal("120_000_000") + (bi - Decimal("1_000_000_000")) * Decimal("0.15")
        else:  # > €30bn
            bic = Decimal("4_470_000_000") + (bi - Decimal("30_000_000_000")) * Decimal("0.18")

        # Operational Risk Capital = BIC × ILM (Internal Loss Multiplier)
        return bic * Decimal(str(self.internal_loss_multiplier))

    def validate(self) -> bool:
        """Validate operational risk report."""
        super().validate()

        if self.business_indicator < 0:
            self.errors.append("Business Indicator cannot be negative")

        if self.internal_loss_multiplier < 1.0:
            self.errors.append("Internal Loss Multiplier cannot be less than 1.0")

        if self.operational_risk_capital < 0:
            self.errors.append("Operational risk capital cannot be negative")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to XML for regulatory reporting."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<OperationalRiskReport>
  <ReportingDate>{self.reporting_date.date().isoformat()}</ReportingDate>
  <BusinessIndicator>{self.business_indicator}</BusinessIndicator>
  <InternalLossMultiplier>{self.internal_loss_multiplier}</InternalLossMultiplier>
  <OperationalRiskCapital>{self.operational_risk_capital}</OperationalRiskCapital>
</OperationalRiskReport>"""


@dataclass
class LeverageRatioReport(RegulatoryReport):
    """Basel III leverage ratio report."""

    reporting_date: datetime = field(default_factory=datetime.utcnow)

    # Tier 1 capital
    tier1_capital: Decimal = Decimal("0")

    # Total exposure measure components
    on_balance_sheet_exposures: Decimal = Decimal("0")
    derivative_exposures: Decimal = Decimal("0")  # SA-CCR exposure
    securities_financing_exposures: Decimal = Decimal("0")  # SFTs
    off_balance_sheet_exposures: Decimal = Decimal("0")

    # Total exposure measure
    total_exposure_measure: Decimal = Decimal("0")

    # Leverage ratio
    leverage_ratio: float = 0.0

    # Minimum regulatory requirement
    minimum_leverage_ratio: float = 0.03  # 3%

    def __post_init__(self):
        """Initialize leverage ratio report."""
        self.report_type = ReportType.BASEL_LEVERAGE
        if self.total_exposure_measure == Decimal("0"):
            self.total_exposure_measure = self._calculate_total_exposure()
        if self.leverage_ratio == 0.0:
            self.leverage_ratio = self._calculate_leverage_ratio()

    def _calculate_total_exposure(self) -> Decimal:
        """Calculate total exposure measure."""
        return (
            self.on_balance_sheet_exposures +
            self.derivative_exposures +
            self.securities_financing_exposures +
            self.off_balance_sheet_exposures
        )

    def _calculate_leverage_ratio(self) -> float:
        """Calculate leverage ratio."""
        if self.total_exposure_measure == 0:
            return 0.0
        return float(self.tier1_capital / self.total_exposure_measure)

    def is_compliant(self) -> bool:
        """Check if leverage ratio meets minimum requirement."""
        return self.leverage_ratio >= self.minimum_leverage_ratio

    def validate(self) -> bool:
        """Validate leverage ratio report."""
        super().validate()

        if self.tier1_capital < 0:
            self.errors.append("Tier 1 capital cannot be negative")

        if self.total_exposure_measure < 0:
            self.errors.append("Total exposure measure cannot be negative")

        if not self.is_compliant():
            self.warnings.append(
                f"Leverage ratio {self.leverage_ratio:.2%} below minimum {self.minimum_leverage_ratio:.2%}"
            )

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to XML for regulatory reporting."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<LeverageRatioReport>
  <ReportingDate>{self.reporting_date.date().isoformat()}</ReportingDate>
  <Tier1Capital>{self.tier1_capital}</Tier1Capital>
  <TotalExposureMeasure>{self.total_exposure_measure}</TotalExposureMeasure>
  <LeverageRatio>{self.leverage_ratio:.4f}</LeverageRatio>
  <MinimumRequirement>{self.minimum_leverage_ratio:.4f}</MinimumRequirement>
  <Compliant>{str(self.is_compliant()).lower()}</Compliant>
</LeverageRatioReport>"""


@dataclass
class BaselCapitalReport(RegulatoryReport):
    """Comprehensive Basel III/IV capital adequacy report."""

    reporting_date: datetime = field(default_factory=datetime.utcnow)
    bank_lei: str = ""

    # Capital components
    cet1_capital: Decimal = Decimal("0")
    at1_capital: Decimal = Decimal("0")
    tier2_capital: Decimal = Decimal("0")

    # Risk-weighted assets
    credit_rwa: Decimal = Decimal("0")
    market_rwa: Decimal = Decimal("0")
    operational_rwa: Decimal = Decimal("0")
    cva_rwa: Decimal = Decimal("0")

    # Total
    total_capital: Decimal = Decimal("0")
    total_rwa: Decimal = Decimal("0")

    # Capital ratios
    cet1_ratio: float = 0.0
    tier1_ratio: float = 0.0
    total_capital_ratio: float = 0.0

    # Buffers
    capital_conservation_buffer: float = 0.025  # 2.5%
    countercyclical_buffer: float = 0.0  # Varies by jurisdiction
    systemic_buffer: float = 0.0  # For G-SIBs

    # Sub-reports
    cva_report: Optional[CVACapitalReport] = None
    frtb_report: Optional[FRTBCapitalReport] = None
    operational_report: Optional[OperationalRiskReport] = None
    leverage_report: Optional[LeverageRatioReport] = None

    def __post_init__(self):
        """Initialize Basel capital report."""
        self.report_type = ReportType.BASEL_CAPITAL
        if self.total_capital == Decimal("0"):
            self.total_capital = self.cet1_capital + self.at1_capital + self.tier2_capital
        if self.total_rwa == Decimal("0"):
            self.total_rwa = (
                self.credit_rwa + self.market_rwa +
                self.operational_rwa + self.cva_rwa
            )
        self._calculate_ratios()

    def _calculate_ratios(self) -> None:
        """Calculate capital ratios."""
        if self.total_rwa > 0:
            self.cet1_ratio = float(self.cet1_capital / self.total_rwa)
            tier1_capital = self.cet1_capital + self.at1_capital
            self.tier1_ratio = float(tier1_capital / self.total_rwa)
            self.total_capital_ratio = float(self.total_capital / self.total_rwa)

    def minimum_cet1_requirement(self) -> float:
        """Calculate minimum CET1 requirement including buffers."""
        base = 0.045  # 4.5% base
        return base + self.capital_conservation_buffer + self.countercyclical_buffer + self.systemic_buffer

    def minimum_tier1_requirement(self) -> float:
        """Calculate minimum Tier 1 requirement including buffers."""
        base = 0.06  # 6% base
        return base + self.capital_conservation_buffer + self.countercyclical_buffer + self.systemic_buffer

    def minimum_total_capital_requirement(self) -> float:
        """Calculate minimum total capital requirement including buffers."""
        base = 0.08  # 8% base
        return base + self.capital_conservation_buffer + self.countercyclical_buffer + self.systemic_buffer

    def is_well_capitalized(self) -> bool:
        """Check if bank is well-capitalized."""
        return (
            self.cet1_ratio >= self.minimum_cet1_requirement() and
            self.tier1_ratio >= self.minimum_tier1_requirement() and
            self.total_capital_ratio >= self.minimum_total_capital_requirement()
        )

    def validate(self) -> bool:
        """Validate Basel capital report."""
        super().validate()

        if not self.bank_lei or len(self.bank_lei) != 20:
            self.errors.append("Valid bank LEI (20 characters) is required")

        if self.total_capital < 0:
            self.errors.append("Total capital cannot be negative")

        if self.total_rwa < 0:
            self.errors.append("Total RWA cannot be negative")

        if not self.is_well_capitalized():
            self.warnings.append("Bank does not meet minimum capital requirements")
            self.warnings.append(f"CET1 ratio: {self.cet1_ratio:.2%} (min: {self.minimum_cet1_requirement():.2%})")
            self.warnings.append(f"Tier 1 ratio: {self.tier1_ratio:.2%} (min: {self.minimum_tier1_requirement():.2%})")
            self.warnings.append(f"Total capital ratio: {self.total_capital_ratio:.2%} (min: {self.minimum_total_capital_requirement():.2%})")

        return len(self.errors) == 0

    def to_xml(self) -> str:
        """Convert to XML for Pillar 3 disclosure."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<BaselCapitalReport>
  <BankLEI>{self.bank_lei}</BankLEI>
  <ReportingDate>{self.reporting_date.date().isoformat()}</ReportingDate>
  <Capital>
    <CET1>{self.cet1_capital}</CET1>
    <AT1>{self.at1_capital}</AT1>
    <Tier2>{self.tier2_capital}</Tier2>
    <Total>{self.total_capital}</Total>
  </Capital>
  <RWA>
    <Credit>{self.credit_rwa}</Credit>
    <Market>{self.market_rwa}</Market>
    <Operational>{self.operational_rwa}</Operational>
    <CVA>{self.cva_rwa}</CVA>
    <Total>{self.total_rwa}</Total>
  </RWA>
  <CapitalRatios>
    <CET1Ratio>{self.cet1_ratio:.4f}</CET1Ratio>
    <Tier1Ratio>{self.tier1_ratio:.4f}</Tier1Ratio>
    <TotalCapitalRatio>{self.total_capital_ratio:.4f}</TotalCapitalRatio>
  </CapitalRatios>
  <WellCapitalized>{str(self.is_well_capitalized()).lower()}</WellCapitalized>
</BaselCapitalReport>"""


class BaselCapitalReporter:
    """Engine for generating Basel III/IV capital reports."""

    def __init__(self, bank_lei: str):
        """Initialize Basel capital reporter.

        Parameters
        ----------
        bank_lei : str
            Bank's Legal Entity Identifier
        """
        self.bank_lei = bank_lei
        self.reports: Dict[datetime, BaselCapitalReport] = {}

    def create_comprehensive_report(
        self,
        reporting_date: datetime,
        **kwargs,
    ) -> BaselCapitalReport:
        """Create comprehensive Basel capital report.

        Parameters
        ----------
        reporting_date : datetime
            Reporting date
        **kwargs
            Capital and RWA components

        Returns
        -------
        BaselCapitalReport
            Created capital report
        """
        report = BaselCapitalReport(
            reporting_date=reporting_date,
            bank_lei=self.bank_lei,
            **kwargs,
        )
        self.reports[reporting_date] = report
        return report

    def generate_pillar3_disclosure(
        self,
        report: BaselCapitalReport,
    ) -> Dict[str, Any]:
        """Generate Pillar 3 disclosure document.

        Parameters
        ----------
        report : BaselCapitalReport
            Basel capital report

        Returns
        -------
        dict
            Pillar 3 disclosure data
        """
        return {
            "reporting_date": report.reporting_date.isoformat(),
            "bank_lei": report.bank_lei,
            "capital_structure": {
                "cet1": str(report.cet1_capital),
                "at1": str(report.at1_capital),
                "tier2": str(report.tier2_capital),
                "total": str(report.total_capital),
            },
            "risk_weighted_assets": {
                "credit": str(report.credit_rwa),
                "market": str(report.market_rwa),
                "operational": str(report.operational_rwa),
                "cva": str(report.cva_rwa),
                "total": str(report.total_rwa),
            },
            "capital_ratios": {
                "cet1_ratio": f"{report.cet1_ratio:.2%}",
                "tier1_ratio": f"{report.tier1_ratio:.2%}",
                "total_capital_ratio": f"{report.total_capital_ratio:.2%}",
            },
            "minimum_requirements": {
                "cet1": f"{report.minimum_cet1_requirement():.2%}",
                "tier1": f"{report.minimum_tier1_requirement():.2%}",
                "total_capital": f"{report.minimum_total_capital_requirement():.2%}",
            },
            "well_capitalized": report.is_well_capitalized(),
        }
