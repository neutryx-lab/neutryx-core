"""XVA: Valuation Adjustments Framework.

This module implements X-Value Adjustments required for IFRS 13 fair value measurement:
- CVA: Credit Valuation Adjustment (counterparty default risk)
- DVA: Debit Valuation Adjustment (own credit risk)
- FVA: Funding Valuation Adjustment (funding costs/benefits)
- MVA: Margin Valuation Adjustment (margin/collateral costs)
- KVA: Capital Valuation Adjustment (regulatory capital costs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import jax.numpy as jnp


@dataclass
class ValuationAdjustment:
    """Base class for valuation adjustments."""

    instrument_id: str
    calculation_date: datetime
    adjustment_amount: Decimal = Decimal("0")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CVACalculator:
    """Credit Valuation Adjustment calculator.

    CVA reflects the market value of counterparty credit risk.
    CVA = LGD × Σ(EE(ti) × PD(ti-1, ti) × DF(ti))
    """

    counterparty_id: str
    loss_given_default: float = 0.40  # Typical senior unsecured
    recovery_rate: Optional[float] = None

    # Exposure profile
    expected_exposure: List[Decimal] = field(default_factory=list)  # EE(t)
    time_points: List[float] = field(default_factory=list)  # in years

    # Credit curve
    survival_probabilities: List[float] = field(default_factory=list)  # S(t)
    hazard_rates: Optional[List[float]] = None  # λ(t)

    # Discounting
    discount_factors: List[Decimal] = field(default_factory=list)  # DF(t)

    def __post_init__(self):
        """Initialize recovery rate if not provided."""
        if self.recovery_rate is None:
            self.recovery_rate = 1.0 - self.loss_given_default

    def calculate_cva(self) -> Decimal:
        """Calculate CVA using expected exposure profile.

        Returns
        -------
        Decimal
            Credit Valuation Adjustment

        Notes
        -----
        CVA = LGD × Σ(EE(ti) × PD(ti-1, ti) × DF(ti))

        Where:
        - EE(ti) = Expected Exposure at time ti
        - PD(ti-1, ti) = Marginal probability of default in interval
        - DF(ti) = Discount factor to time ti
        - LGD = Loss Given Default
        """
        if not self.expected_exposure or not self.survival_probabilities:
            return Decimal("0")

        cva = Decimal("0")
        n_points = min(
            len(self.expected_exposure),
            len(self.survival_probabilities) - 1,  # Need t and t-1
            len(self.discount_factors),
        )

        for i in range(n_points):
            # Expected exposure
            ee = self.expected_exposure[i]

            # Marginal PD = S(t-1) - S(t)
            marginal_pd = (
                self.survival_probabilities[i] -
                self.survival_probabilities[i + 1]
            )

            # Discount factor
            df = self.discount_factors[i]

            # CVA contribution
            cva += ee * Decimal(str(marginal_pd)) * df

        # Apply LGD
        cva *= Decimal(str(self.loss_given_default))

        return cva

    def calculate_cva_from_cds_spread(
        self,
        cds_spread_bps: float,
        maturity_years: float,
        expected_positive_exposure: Decimal,
    ) -> Decimal:
        """Calculate CVA from CDS spread (simplified).

        Parameters
        ----------
        cds_spread_bps : float
            CDS spread in basis points
        maturity_years : float
            Maturity in years
        expected_positive_exposure : Decimal
            Average expected positive exposure

        Returns
        -------
        Decimal
            Approximate CVA

        Notes
        -----
        Simplified formula: CVA ≈ EPE × CDS_spread × Duration × LGD
        """
        cds_spread_decimal = cds_spread_bps / 10000  # Convert bps to decimal

        # Approximate duration
        duration = maturity_years * 0.75  # Simplified

        cva = (
            expected_positive_exposure *
            Decimal(str(cds_spread_decimal)) *
            Decimal(str(duration)) *
            Decimal(str(self.loss_given_default))
        )

        return cva


@dataclass
class DVACalculator:
    """Debit Valuation Adjustment calculator.

    DVA reflects the market value of own credit risk.
    DVA = LGD_own × Σ(ENE(ti) × PD_own(ti-1, ti) × DF(ti))
    """

    own_entity_id: str
    loss_given_default: float = 0.40

    # Exposure profile (from counterparty perspective)
    expected_negative_exposure: List[Decimal] = field(default_factory=list)  # ENE(t)
    time_points: List[float] = field(default_factory=list)

    # Own credit curve
    own_survival_probabilities: List[float] = field(default_factory=list)

    # Discounting
    discount_factors: List[Decimal] = field(default_factory=list)

    def calculate_dva(self) -> Decimal:
        """Calculate DVA using expected negative exposure.

        Returns
        -------
        Decimal
            Debit Valuation Adjustment

        Notes
        -----
        DVA = LGD_own × Σ(ENE(ti) × PD_own(ti-1, ti) × DF(ti))

        DVA represents a gain from own credit deterioration.
        """
        if not self.expected_negative_exposure or not self.own_survival_probabilities:
            return Decimal("0")

        dva = Decimal("0")
        n_points = min(
            len(self.expected_negative_exposure),
            len(self.own_survival_probabilities) - 1,
            len(self.discount_factors),
        )

        for i in range(n_points):
            # Expected negative exposure
            ene = self.expected_negative_exposure[i]

            # Marginal PD (own)
            marginal_pd = (
                self.own_survival_probabilities[i] -
                self.own_survival_probabilities[i + 1]
            )

            # Discount factor
            df = self.discount_factors[i]

            # DVA contribution
            dva += ene * Decimal(str(marginal_pd)) * df

        # Apply LGD
        dva *= Decimal(str(self.loss_given_default))

        return dva


@dataclass
class FVACalculator:
    """Funding Valuation Adjustment calculator.

    FVA reflects funding costs/benefits for uncollateralized exposure.
    FVA = FCA - FBA
    Where FCA = Funding Cost Adjustment, FBA = Funding Benefit Adjustment
    """

    funding_spread: float  # Over risk-free rate (bps)

    # Exposure profiles
    expected_positive_exposure: List[Decimal] = field(default_factory=list)
    expected_negative_exposure: List[Decimal] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)

    # Discounting (at risk-free + funding spread)
    discount_factors: List[Decimal] = field(default_factory=list)

    def calculate_fva(self) -> Decimal:
        """Calculate total FVA (FCA - FBA).

        Returns
        -------
        Decimal
            Funding Valuation Adjustment

        Notes
        -----
        FVA = FCA - FBA
        - FCA = Cost of funding expected positive exposure
        - FBA = Benefit of investing expected negative exposure
        """
        fca = self.calculate_fca()
        fba = self.calculate_fba()

        return fca - fba

    def calculate_fca(self) -> Decimal:
        """Calculate Funding Cost Adjustment.

        Returns
        -------
        Decimal
            Funding cost for positive exposures
        """
        if not self.expected_positive_exposure:
            return Decimal("0")

        funding_spread_decimal = self.funding_spread / 10000

        fca = Decimal("0")
        n_points = min(
            len(self.expected_positive_exposure),
            len(self.time_points),
            len(self.discount_factors),
        )

        for i in range(n_points):
            epe = self.expected_positive_exposure[i]
            dt = self.time_points[i] - (self.time_points[i - 1] if i > 0 else 0)
            df = self.discount_factors[i]

            # Funding cost for period
            fca += epe * Decimal(str(funding_spread_decimal * dt)) * df

        return fca

    def calculate_fba(self) -> Decimal:
        """Calculate Funding Benefit Adjustment.

        Returns
        -------
        Decimal
            Funding benefit from negative exposures
        """
        if not self.expected_negative_exposure:
            return Decimal("0")

        funding_spread_decimal = self.funding_spread / 10000

        fba = Decimal("0")
        n_points = min(
            len(self.expected_negative_exposure),
            len(self.time_points),
            len(self.discount_factors),
        )

        for i in range(n_points):
            ene = self.expected_negative_exposure[i]
            dt = self.time_points[i] - (self.time_points[i - 1] if i > 0 else 0)
            df = self.discount_factors[i]

            # Funding benefit for period
            fba += ene * Decimal(str(funding_spread_decimal * dt)) * df

        return fba


@dataclass
class MVACalculator:
    """Margin Valuation Adjustment calculator.

    MVA reflects the cost of posting initial margin and meeting margin calls.
    """

    initial_margin: Decimal = Decimal("0")
    funding_spread: float = 0.0  # Cost of funding margin (bps)
    maturity_years: float = 1.0

    # For variation margin
    expected_margin_calls: List[Decimal] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    discount_factors: List[Decimal] = field(default_factory=list)

    def calculate_mva(self) -> Decimal:
        """Calculate Margin Valuation Adjustment.

        Returns
        -------
        Decimal
            Cost of margin funding

        Notes
        -----
        MVA = Cost of funding initial margin + Cost of variation margin
        """
        # Cost of funding initial margin over life of trade
        im_cost = (
            self.initial_margin *
            Decimal(str(self.funding_spread / 10000)) *
            Decimal(str(self.maturity_years))
        )

        # Cost of variation margin
        vm_cost = Decimal("0")
        if self.expected_margin_calls and self.time_points and self.discount_factors:
            n_points = min(
                len(self.expected_margin_calls),
                len(self.time_points),
                len(self.discount_factors),
            )

            for i in range(n_points):
                margin = self.expected_margin_calls[i]
                dt = self.time_points[i] - (self.time_points[i - 1] if i > 0 else 0)
                df = self.discount_factors[i]

                vm_cost += margin * Decimal(str(self.funding_spread / 10000 * dt)) * df

        return im_cost + vm_cost


@dataclass
class KVACalculator:
    """Capital Valuation Adjustment calculator.

    KVA reflects the cost of regulatory capital required to support the exposure.
    """

    capital_requirement: Decimal  # Regulatory capital required
    cost_of_capital: float = 0.12  # Hurdle rate (12% typical)
    maturity_years: float = 1.0

    # Time-varying capital requirements
    capital_profile: List[Decimal] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    discount_factors: List[Decimal] = field(default_factory=list)

    def calculate_kva(self) -> Decimal:
        """Calculate Capital Valuation Adjustment.

        Returns
        -------
        Decimal
            Cost of regulatory capital

        Notes
        -----
        KVA = Present value of cost of capital over trade life
        """
        if self.capital_profile and self.time_points and self.discount_factors:
            # Time-varying capital
            kva = Decimal("0")
            n_points = min(
                len(self.capital_profile),
                len(self.time_points),
                len(self.discount_factors),
            )

            for i in range(n_points):
                capital = self.capital_profile[i]
                dt = self.time_points[i] - (self.time_points[i - 1] if i > 0 else 0)
                df = self.discount_factors[i]

                kva += capital * Decimal(str(self.cost_of_capital * dt)) * df

            return kva
        else:
            # Simplified: constant capital over maturity
            kva = (
                self.capital_requirement *
                Decimal(str(self.cost_of_capital)) *
                Decimal(str(self.maturity_years))
            )
            return kva


@dataclass
class XVAEngine:
    """Comprehensive XVA calculation engine."""

    instrument_id: str
    counterparty_id: str
    calculation_date: datetime = field(default_factory=datetime.utcnow)

    # Component calculators
    cva_calculator: Optional[CVACalculator] = None
    dva_calculator: Optional[DVACalculator] = None
    fva_calculator: Optional[FVACalculator] = None
    mva_calculator: Optional[MVACalculator] = None
    kva_calculator: Optional[KVACalculator] = None

    # Results
    cva: Decimal = Decimal("0")
    dva: Decimal = Decimal("0")
    fva: Decimal = Decimal("0")
    mva: Decimal = Decimal("0")
    kva: Decimal = Decimal("0")

    total_xva: Decimal = Decimal("0")

    def calculate_all_xva(self) -> Dict[str, Decimal]:
        """Calculate all XVA components.

        Returns
        -------
        dict
            All XVA components and total
        """
        # CVA
        if self.cva_calculator:
            self.cva = self.cva_calculator.calculate_cva()

        # DVA
        if self.dva_calculator:
            self.dva = self.dva_calculator.calculate_dva()

        # FVA
        if self.fva_calculator:
            self.fva = self.fva_calculator.calculate_fva()

        # MVA
        if self.mva_calculator:
            self.mva = self.mva_calculator.calculate_mva()

        # KVA
        if self.kva_calculator:
            self.kva = self.kva_calculator.calculate_kva()

        # Total XVA
        # Note: CVA is a cost (negative), DVA is a benefit (positive)
        self.total_xva = -self.cva + self.dva - self.fva - self.mva - self.kva

        return {
            "cva": self.cva,
            "dva": self.dva,
            "fva": self.fva,
            "mva": self.mva,
            "kva": self.kva,
            "total_xva": self.total_xva,
            "adjusted_fair_value_impact": -self.total_xva,  # Impact on FV
        }

    def get_ifrs13_credit_adjustment(self) -> Decimal:
        """Get credit adjustment for IFRS 13 fair value.

        Returns
        -------
        Decimal
            Credit risk adjustment (CVA - DVA)
        """
        # IFRS 13 requires adjustment for counterparty credit risk and own credit risk
        return self.cva - self.dva

    def get_ifrs13_total_adjustments(self) -> Dict[str, Decimal]:
        """Get all IFRS 13 fair value adjustments.

        Returns
        -------
        dict
            Breakdown of fair value adjustments
        """
        return {
            "credit_adjustment": self.get_ifrs13_credit_adjustment(),
            "funding_adjustment": self.fva,
            "margin_adjustment": self.mva,
            "capital_adjustment": self.kva,
            "total_adjustments": -self.total_xva,
        }

    def xva_sensitivity(
        self,
        parameter: str,
        shock_bps: int = 10,
    ) -> Dict[str, Decimal]:
        """Calculate XVA sensitivity to parameter changes.

        Parameters
        ----------
        parameter : str
            Parameter to shock (e.g., "credit_spread", "funding_spread")
        shock_bps : int
            Shock size in basis points

        Returns
        -------
        dict
            Sensitivity results
        """
        base_xva = self.total_xva

        # Apply shock and recalculate
        # (Implementation would modify the relevant calculator parameters)

        return {
            "parameter": parameter,
            "base_xva": base_xva,
            "shock_bps": shock_bps,
            "shocked_xva": base_xva,  # Placeholder
            "sensitivity": Decimal("0"),  # Placeholder
        }
