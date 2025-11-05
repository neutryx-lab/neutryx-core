"""IFRS 9: Financial Instruments.

This module implements IFRS 9 requirements for:
- Classification and measurement (FVPL, FVOCI, Amortized Cost)
- Impairment: Expected Credit Loss (ECL) model
- Hedge accounting (fair value, cash flow, net investment hedges)
- Hedge effectiveness testing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.scipy.stats import norm


class FinancialInstrumentCategory(str, Enum):
    """IFRS 9 classification categories."""

    FVPL = "FVPL"  # Fair Value through Profit or Loss
    FVOCI = "FVOCI"  # Fair Value through Other Comprehensive Income
    AMORTIZED_COST = "Amortized Cost"


class BusinessModel(str, Enum):
    """Business model assessment per IFRS 9."""

    HOLD_TO_COLLECT = "Hold to Collect"  # Amortized Cost eligible
    HOLD_AND_SELL = "Hold and Sell"  # FVOCI eligible
    OTHER = "Other"  # FVPL


class ECLStage(str, Enum):
    """IFRS 9 ECL staging."""

    STAGE_1 = "Stage 1"  # 12-month ECL
    STAGE_2 = "Stage 2"  # Lifetime ECL (no credit impairment)
    STAGE_3 = "Stage 3"  # Lifetime ECL (credit-impaired)


class HedgeType(str, Enum):
    """Types of hedge relationships per IFRS 9."""

    FAIR_VALUE = "Fair Value Hedge"
    CASH_FLOW = "Cash Flow Hedge"
    NET_INVESTMENT = "Net Investment Hedge"


@dataclass
class IFRS9Classifier:
    """Classifier for IFRS 9 financial instrument categories."""

    @staticmethod
    def classify(
        business_model: BusinessModel,
        cash_flows_solely_payments_principal_interest: bool,
        equity_instrument: bool = False,
        elected_fvoci: bool = False,
        elected_fvpl: bool = False,
    ) -> FinancialInstrumentCategory:
        """Classify financial instrument per IFRS 9.

        Parameters
        ----------
        business_model : BusinessModel
            Entity's business model for managing the asset
        cash_flows_solely_payments_principal_interest : bool
            Whether cash flows are SPPI (solely payments of principal and interest)
        equity_instrument : bool
            Whether the instrument is an equity investment
        elected_fvoci : bool
            Whether FVOCI election made for equity (irrevocable)
        elected_fvpl : bool
            Whether Fair Value Option exercised

        Returns
        -------
        FinancialInstrumentCategory
            IFRS 9 classification

        Notes
        -----
        Classification decision tree per IFRS 9:
        1. Check if Fair Value Option elected → FVPL
        2. If equity: Check FVOCI election → FVOCI or FVPL
        3. If debt: Check SPPI + Business Model
        """
        # Fair Value Option overrides
        if elected_fvpl:
            return FinancialInstrumentCategory.FVPL

        # Equity instruments
        if equity_instrument:
            if elected_fvoci:
                return FinancialInstrumentCategory.FVOCI
            return FinancialInstrumentCategory.FVPL

        # Debt instruments - SPPI test
        if not cash_flows_solely_payments_principal_interest:
            return FinancialInstrumentCategory.FVPL

        # Business model assessment
        if business_model == BusinessModel.HOLD_TO_COLLECT:
            return FinancialInstrumentCategory.AMORTIZED_COST

        if business_model == BusinessModel.HOLD_AND_SELL:
            return FinancialInstrumentCategory.FVOCI

        # Default to FVPL
        return FinancialInstrumentCategory.FVPL


@dataclass
class ECLModel:
    """Expected Credit Loss model per IFRS 9."""

    exposure_at_default: Decimal
    probability_of_default: float  # 1-year or lifetime
    loss_given_default: float = 0.45  # Typically 45% for unsecured
    discount_rate: float = 0.05  # Effective interest rate
    time_horizon_years: float = 1.0  # 12 months for Stage 1, lifetime for Stage 2/3

    # For lifetime ECL
    forward_looking_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    scenario_weights: List[float] = field(default_factory=list)

    def calculate_12_month_ecl(self) -> Decimal:
        """Calculate 12-month ECL for Stage 1.

        Returns
        -------
        Decimal
            12-month expected credit loss
        """
        ecl = (
            self.exposure_at_default *
            Decimal(str(self.probability_of_default)) *
            Decimal(str(self.loss_given_default))
        )
        return ecl

    def calculate_lifetime_ecl(
        self,
        maturity_years: float,
        pd_curve: Optional[List[float]] = None,
    ) -> Decimal:
        """Calculate lifetime ECL for Stage 2/3.

        Parameters
        ----------
        maturity_years : float
            Remaining maturity in years
        pd_curve : list of float, optional
            Probability of default curve over lifetime

        Returns
        -------
        Decimal
            Lifetime expected credit loss
        """
        if pd_curve is None:
            # Simplified: assume constant PD
            pd_curve = [self.probability_of_default] * int(maturity_years)

        total_ecl = Decimal("0")
        exposure = self.exposure_at_default

        for year, pd in enumerate(pd_curve, start=1):
            # PV of ECL for this period
            ecl_period = (
                exposure *
                Decimal(str(pd)) *
                Decimal(str(self.loss_given_default))
            )

            # Discount to present value
            discount_factor = Decimal(str(1 / ((1 + self.discount_rate) ** year)))
            total_ecl += ecl_period * discount_factor

        return total_ecl

    def calculate_probability_weighted_ecl(self) -> Decimal:
        """Calculate ECL using multiple forward-looking scenarios.

        Returns
        -------
        Decimal
            Probability-weighted ECL across scenarios
        """
        if not self.forward_looking_scenarios or not self.scenario_weights:
            return self.calculate_12_month_ecl()

        weighted_ecl = Decimal("0")

        for scenario, weight in zip(self.forward_looking_scenarios, self.scenario_weights):
            scenario_pd = scenario.get("probability_of_default", self.probability_of_default)
            scenario_lgd = scenario.get("loss_given_default", self.loss_given_default)

            scenario_ecl = (
                self.exposure_at_default *
                Decimal(str(scenario_pd)) *
                Decimal(str(scenario_lgd))
            )

            weighted_ecl += scenario_ecl * Decimal(str(weight))

        return weighted_ecl


@dataclass
class ECLResult:
    """Expected Credit Loss calculation result."""

    instrument_id: str
    calculation_date: datetime
    ecl_stage: ECLStage
    exposure_at_default: Decimal
    probability_of_default: float
    loss_given_default: float

    # ECL amounts
    ecl_12_month: Decimal = Decimal("0")
    ecl_lifetime: Decimal = Decimal("0")
    ecl_provision: Decimal = Decimal("0")  # Amount to be recognized

    # Stage determination
    days_past_due: int = 0
    credit_impaired: bool = False
    significant_increase_in_credit_risk: bool = False

    # Movement
    opening_provision: Decimal = Decimal("0")
    provision_charge: Decimal = Decimal("0")
    write_offs: Decimal = Decimal("0")
    recoveries: Decimal = Decimal("0")
    closing_provision: Decimal = Decimal("0")

    def determine_stage(self) -> ECLStage:
        """Determine ECL stage per IFRS 9.

        Returns
        -------
        ECLStage
            Appropriate ECL stage

        Notes
        -----
        Staging criteria:
        - Stage 1: No significant increase in credit risk since initial recognition
        - Stage 2: Significant increase in credit risk (but not impaired)
        - Stage 3: Credit-impaired (objective evidence of impairment)
        """
        # Stage 3: Credit-impaired
        if self.credit_impaired or self.days_past_due > 90:
            self.ecl_stage = ECLStage.STAGE_3
            self.ecl_provision = self.ecl_lifetime
            return ECLStage.STAGE_3

        # Stage 2: Significant increase in credit risk
        if self.significant_increase_in_credit_risk or self.days_past_due > 30:
            self.ecl_stage = ECLStage.STAGE_2
            self.ecl_provision = self.ecl_lifetime
            return ECLStage.STAGE_2

        # Stage 1: Performing
        self.ecl_stage = ECLStage.STAGE_1
        self.ecl_provision = self.ecl_12_month
        return ECLStage.STAGE_1

    def calculate_provision_movement(self) -> None:
        """Calculate ECL provision movement."""
        self.provision_charge = (
            self.ecl_provision - self.opening_provision +
            self.write_offs - self.recoveries
        )
        self.closing_provision = self.ecl_provision


@dataclass
class HedgeRelationship:
    """IFRS 9 hedge accounting relationship."""

    hedge_id: str
    hedge_type: HedgeType
    designation_date: datetime

    # Hedged item
    hedged_item_id: str
    hedged_item_description: str
    hedged_risk: str  # e.g., "Interest Rate Risk", "FX Risk"

    # Hedging instrument
    hedging_instrument_id: str
    hedging_instrument_type: str  # e.g., "Interest Rate Swap", "FX Forward"
    hedging_instrument_notional: Decimal = Decimal("0")

    # Hedge ratio
    hedge_ratio: Decimal = Decimal("1.0")  # Hedging instrument : Hedged item

    # Effectiveness testing
    prospective_effective: bool = True
    retrospective_effective: bool = True
    ineffectiveness_amount: Decimal = Decimal("0")

    # For cash flow hedges
    cash_flow_hedge_reserve: Decimal = Decimal("0")  # In OCI
    reclassified_to_pnl: Decimal = Decimal("0")

    # Discontinuation
    discontinued: bool = False
    discontinuation_date: Optional[datetime] = None
    discontinuation_reason: str = ""

    def is_eligible_for_hedge_accounting(self) -> bool:
        """Check if relationship qualifies for hedge accounting.

        Returns
        -------
        bool
            True if eligible per IFRS 9 requirements
        """
        # IFRS 9 hedge accounting criteria:
        # 1. Formal designation and documentation
        # 2. Hedge expected to be highly effective
        # 3. Effectiveness can be reliably measured
        # 4. Hedge assessed on ongoing basis

        if self.discontinued:
            return False

        if not self.prospective_effective:
            return False

        # Hedge ratio should be reasonable
        if self.hedge_ratio <= 0 or self.hedge_ratio > Decimal("2.0"):
            return False

        return True


@dataclass
class HedgeEffectivenessTest:
    """IFRS 9 hedge effectiveness testing."""

    hedge_relationship: HedgeRelationship
    test_date: datetime

    # Changes in fair value or cash flows
    hedged_item_change: Decimal = Decimal("0")
    hedging_instrument_change: Decimal = Decimal("0")

    # Effectiveness thresholds (IFRS 9 allows judgment)
    # Common practice: 80%-125% effectiveness ratio
    min_effectiveness_ratio: Decimal = Decimal("0.80")
    max_effectiveness_ratio: Decimal = Decimal("1.25")

    def calculate_effectiveness_ratio(self) -> Decimal:
        """Calculate hedge effectiveness ratio.

        Returns
        -------
        Decimal
            Effectiveness ratio (hedging instrument change / hedged item change)
        """
        if self.hedged_item_change == 0:
            return Decimal("1.0")

        # For fair value and cash flow hedges
        # Ratio = Change in hedging instrument / Change in hedged item
        ratio = abs(self.hedging_instrument_change / self.hedged_item_change)

        return ratio

    def test_prospective_effectiveness(
        self,
        expected_hedged_item_change: Decimal,
        expected_hedging_instrument_change: Decimal,
    ) -> bool:
        """Test prospective (forward-looking) effectiveness.

        Parameters
        ----------
        expected_hedged_item_change : Decimal
            Expected future change in hedged item
        expected_hedging_instrument_change : Decimal
            Expected future change in hedging instrument

        Returns
        -------
        bool
            True if hedge is expected to be highly effective
        """
        if expected_hedged_item_change == 0:
            return True

        expected_ratio = abs(
            expected_hedging_instrument_change / expected_hedged_item_change
        )

        is_effective = (
            self.min_effectiveness_ratio <= expected_ratio <= self.max_effectiveness_ratio
        )

        self.hedge_relationship.prospective_effective = is_effective
        return is_effective

    def test_retrospective_effectiveness(self) -> bool:
        """Test retrospective (backward-looking) effectiveness.

        Returns
        -------
        bool
            True if hedge was highly effective in the period
        """
        actual_ratio = self.calculate_effectiveness_ratio()

        is_effective = (
            self.min_effectiveness_ratio <= actual_ratio <= self.max_effectiveness_ratio
        )

        self.hedge_relationship.retrospective_effective = is_effective

        # Calculate ineffectiveness
        if is_effective:
            # Ineffectiveness = excess change in hedging instrument
            self.hedge_relationship.ineffectiveness_amount = (
                self.hedging_instrument_change +
                self.hedged_item_change  # Note: typically opposite signs
            )
        else:
            # Fully ineffective - all goes to P&L
            self.hedge_relationship.ineffectiveness_amount = self.hedging_instrument_change

        return is_effective

    def dollar_offset_test(self) -> float:
        """Perform dollar offset test.

        Returns
        -------
        float
            Dollar offset ratio (should be close to -1.0 for perfect hedge)
        """
        if self.hedged_item_change == 0:
            return 0.0

        # Dollar offset = Change in HI / Change in HI
        # For perfect hedge, should be -1.0 (offsetting)
        offset = float(self.hedging_instrument_change / self.hedged_item_change)

        return offset

    def regression_analysis_test(
        self,
        historical_hedged_item_changes: List[Decimal],
        historical_hedging_instrument_changes: List[Decimal],
    ) -> Dict[str, float]:
        """Perform statistical regression analysis.

        Parameters
        ----------
        historical_hedged_item_changes : list of Decimal
            Historical changes in hedged item
        historical_hedging_instrument_changes : list of Decimal
            Historical changes in hedging instrument

        Returns
        -------
        dict
            Regression results (slope, R², etc.)
        """
        if len(historical_hedged_item_changes) < 2:
            return {}

        # Convert to numpy arrays
        x = jnp.array([float(v) for v in historical_hedged_item_changes])
        y = jnp.array([float(v) for v in historical_hedging_instrument_changes])

        # Simple linear regression: y = slope * x + intercept
        n = len(x)
        x_mean = jnp.mean(x)
        y_mean = jnp.mean(y)

        # Slope
        numerator = jnp.sum((x - x_mean) * (y - y_mean))
        denominator = jnp.sum((x - x_mean) ** 2)
        slope = numerator / denominator if denominator != 0 else 0.0

        # R-squared
        ss_tot = jnp.sum((y - y_mean) ** 2)
        y_pred = slope * x
        ss_res = jnp.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "slope": float(slope),
            "r_squared": float(r_squared),
            "n_observations": n,
            "effective": float(r_squared) > 0.80 and -1.25 <= float(slope) <= -0.80,
        }


def sppi_test(
    contractual_cash_flows: List[Decimal],
    principal_amount: Decimal,
    interest_rate: float,
    time_value_of_money: bool = True,
    credit_risk: bool = True,
    other_basic_lending_risks: bool = True,
) -> bool:
    """SPPI test: Solely Payments of Principal and Interest.

    Parameters
    ----------
    contractual_cash_flows : list of Decimal
        Expected contractual cash flows
    principal_amount : Decimal
        Principal amount of the instrument
    interest_rate : float
        Stated interest rate
    time_value_of_money : bool
        Whether interest represents time value of money
    credit_risk : bool
        Whether interest includes consideration for credit risk
    other_basic_lending_risks : bool
        Whether interest includes only basic lending risks

    Returns
    -------
    bool
        True if SPPI test passed (eligible for AC or FVOCI)

    Notes
    -----
    IFRS 9 SPPI test evaluates whether contractual cash flows are
    solely payments of principal and interest on the principal outstanding.

    Interest = consideration for time value of money and credit risk.
    """
    # Check if interest components are appropriate
    if not time_value_of_money:
        return False  # Must include time value of money

    if not credit_risk:
        return False  # Must include credit risk

    # Check for leverage, commodity prices, equity returns, etc.
    if not other_basic_lending_risks:
        return False  # Contains non-basic lending risks

    # Additional checks on cash flows
    # In practice, would analyze the contractual terms more deeply

    return True
