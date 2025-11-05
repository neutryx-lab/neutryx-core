"""IFRS 13: Fair Value Measurement.

This module implements IFRS 13 requirements for:
- Fair value hierarchy (Level 1, 2, 3)
- Valuation techniques (market, income, cost approaches)
- Observable vs unobservable inputs
- Fair value disclosures
- Day 1 gain/loss
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class FairValueHierarchy(str, Enum):
    """IFRS 13 fair value hierarchy levels."""

    LEVEL_1 = "Level 1"  # Quoted prices in active markets
    LEVEL_2 = "Level 2"  # Observable inputs other than Level 1
    LEVEL_3 = "Level 3"  # Unobservable inputs


class ValuationTechnique(str, Enum):
    """IFRS 13 valuation techniques."""

    MARKET_APPROACH = "Market"  # Comparable market transactions
    INCOME_APPROACH = "Income"  # PV of future cash flows
    COST_APPROACH = "Cost"  # Replacement cost


class InputObservability(str, Enum):
    """Input observability classification."""

    OBSERVABLE = "Observable"
    UNOBSERVABLE = "Unobservable"
    PARTIALLY_OBSERVABLE = "Partially Observable"


@dataclass
class FairValueInput:
    """Fair value measurement input."""

    name: str
    value: Decimal
    observability: InputObservability
    source: str = ""  # E.g., "Bloomberg", "Internal Model"
    as_of_date: datetime = field(default_factory=datetime.utcnow)

    # For Level 2/3 inputs
    adjustment: Decimal = Decimal("0")  # Model adjustment
    uncertainty: Optional[Decimal] = None  # Input uncertainty/range


@dataclass
class FairValueMeasurement:
    """IFRS 13 fair value measurement."""

    instrument_id: str
    measurement_date: datetime
    fair_value: Decimal
    hierarchy_level: FairValueHierarchy
    valuation_technique: ValuationTechnique

    # Inputs used in valuation
    inputs: List[FairValueInput] = field(default_factory=list)

    # Day 1 P&L
    transaction_price: Optional[Decimal] = None
    day1_gain_loss: Optional[Decimal] = None

    # Valuation adjustments
    bid_ask_adjustment: Decimal = Decimal("0")
    liquidity_adjustment: Decimal = Decimal("0")
    credit_adjustment: Decimal = Decimal("0")  # CVA/DVA
    model_adjustment: Decimal = Decimal("0")
    other_adjustments: Decimal = Decimal("0")

    # For Level 3
    unobservable_inputs_sensitivity: Dict[str, Decimal] = field(default_factory=dict)
    reasonably_possible_alternatives: Dict[str, Decimal] = field(default_factory=dict)

    # Movement reconciliation (for Level 3)
    beginning_balance: Optional[Decimal] = None
    purchases: Decimal = Decimal("0")
    sales: Decimal = Decimal("0")
    issuances: Decimal = Decimal("0")
    settlements: Decimal = Decimal("0")
    transfers_in: Decimal = Decimal("0")
    transfers_out: Decimal = Decimal("0")
    realized_gains_losses: Decimal = Decimal("0")
    unrealized_gains_losses_pnl: Decimal = Decimal("0")
    unrealized_gains_losses_oci: Decimal = Decimal("0")  # Other Comprehensive Income

    def total_adjustments(self) -> Decimal:
        """Calculate total valuation adjustments."""
        return (
            self.bid_ask_adjustment +
            self.liquidity_adjustment +
            self.credit_adjustment +
            self.model_adjustment +
            self.other_adjustments
        )

    def unadjusted_fair_value(self) -> Decimal:
        """Get fair value before adjustments."""
        return self.fair_value - self.total_adjustments()

    def calculate_day1_gain_loss(self) -> Decimal:
        """Calculate Day 1 gain or loss.

        Returns
        -------
        Decimal
            Day 1 gain (positive) or loss (negative)

        Notes
        -----
        IFRS 13 requires Day 1 gain/loss recognition based on
        hierarchy level and transaction circumstances.
        """
        if self.transaction_price is None:
            return Decimal("0")

        day1_diff = self.fair_value - self.transaction_price

        # Level 1: Immediate recognition
        if self.hierarchy_level == FairValueHierarchy.LEVEL_1:
            self.day1_gain_loss = day1_diff
            return day1_diff

        # Level 2/3: May need to defer recognition
        # This is entity-specific accounting policy
        # Default: recognize if supported by observable market data
        if self.hierarchy_level == FairValueHierarchy.LEVEL_2:
            # Check if observable inputs support the difference
            self.day1_gain_loss = day1_diff  # Simplified
            return day1_diff

        # Level 3: Usually deferred and amortized
        self.day1_gain_loss = Decimal("0")  # Defer recognition
        return Decimal("0")

    def level3_reconciliation(self) -> Decimal:
        """Reconcile Level 3 fair value movements.

        Returns
        -------
        Decimal
            Ending balance
        """
        if self.beginning_balance is None:
            return self.fair_value

        ending = (
            self.beginning_balance +
            self.purchases -
            self.sales +
            self.issuances -
            self.settlements +
            self.transfers_in -
            self.transfers_out +
            self.realized_gains_losses +
            self.unrealized_gains_losses_pnl +
            self.unrealized_gains_losses_oci
        )

        return ending

    def sensitivity_analysis(
        self,
        input_name: str,
        shock_bps: int = 100,
    ) -> Dict[str, Decimal]:
        """Perform sensitivity analysis on unobservable input.

        Parameters
        ----------
        input_name : str
            Name of the unobservable input
        shock_bps : int
            Shock size in basis points

        Returns
        -------
        dict
            Sensitivity results with up/down shocks
        """
        # Find the input
        input_value = None
        for inp in self.inputs:
            if inp.name == input_name and inp.observability == InputObservability.UNOBSERVABLE:
                input_value = inp.value
                break

        if input_value is None:
            return {}

        # Simplified sensitivity (would use actual revaluation in production)
        shock_decimal = Decimal(str(shock_bps / 10000))
        up_shock = input_value * (Decimal("1") + shock_decimal)
        down_shock = input_value * (Decimal("1") - shock_decimal)

        # Estimate fair value impact (simplified linear approximation)
        # In practice, would revalue with shocked inputs
        impact_per_unit = self.fair_value / (input_value if input_value != 0 else Decimal("1"))

        return {
            "input": input_name,
            "base_value": input_value,
            "shock_bps": shock_bps,
            "up_shock_value": up_shock,
            "down_shock_value": down_shock,
            "up_shock_fv_impact": (up_shock - input_value) * impact_per_unit,
            "down_shock_fv_impact": (down_shock - input_value) * impact_per_unit,
        }


@dataclass
class IFRS13Disclosure:
    """IFRS 13 disclosure requirements."""

    reporting_date: datetime
    entity_name: str

    # Fair value by hierarchy level
    level1_assets: Decimal = Decimal("0")
    level1_liabilities: Decimal = Decimal("0")
    level2_assets: Decimal = Decimal("0")
    level2_liabilities: Decimal = Decimal("0")
    level3_assets: Decimal = Decimal("0")
    level3_liabilities: Decimal = Decimal("0")

    # Level 3 movements
    level3_measurements: List[FairValueMeasurement] = field(default_factory=list)

    # Valuation techniques by class
    valuation_techniques: Dict[str, ValuationTechnique] = field(default_factory=dict)

    # Transfers between levels
    transfers_level1_to_level2: Decimal = Decimal("0")
    transfers_level2_to_level1: Decimal = Decimal("0")
    transfers_level2_to_level3: Decimal = Decimal("0")
    transfers_level3_to_level2: Decimal = Decimal("0")

    # Unrealized gains/losses on Level 3
    level3_unrealized_gains_losses: Decimal = Decimal("0")

    def total_assets_at_fair_value(self) -> Decimal:
        """Total assets measured at fair value."""
        return self.level1_assets + self.level2_assets + self.level3_assets

    def total_liabilities_at_fair_value(self) -> Decimal:
        """Total liabilities measured at fair value."""
        return self.level1_liabilities + self.level2_liabilities + self.level3_liabilities

    def fair_value_hierarchy_table(self) -> Dict[str, Any]:
        """Generate fair value hierarchy disclosure table.

        Returns
        -------
        dict
            Formatted hierarchy table for disclosure
        """
        return {
            "reporting_date": self.reporting_date.date().isoformat(),
            "fair_value_hierarchy": {
                "Level 1": {
                    "assets": str(self.level1_assets),
                    "liabilities": str(self.level1_liabilities),
                    "description": "Quoted prices in active markets",
                },
                "Level 2": {
                    "assets": str(self.level2_assets),
                    "liabilities": str(self.level2_liabilities),
                    "description": "Observable inputs other than quoted prices",
                },
                "Level 3": {
                    "assets": str(self.level3_assets),
                    "liabilities": str(self.level3_liabilities),
                    "description": "Unobservable inputs",
                },
            },
            "total": {
                "assets": str(self.total_assets_at_fair_value()),
                "liabilities": str(self.total_liabilities_at_fair_value()),
            },
        }

    def level3_reconciliation_table(self) -> List[Dict[str, Any]]:
        """Generate Level 3 reconciliation disclosure.

        Returns
        -------
        list
            Level 3 movement reconciliation for each measurement
        """
        reconciliations = []

        for measurement in self.level3_measurements:
            if measurement.beginning_balance is not None:
                recon = {
                    "instrument_id": measurement.instrument_id,
                    "beginning_balance": str(measurement.beginning_balance),
                    "purchases": str(measurement.purchases),
                    "sales": str(measurement.sales),
                    "issuances": str(measurement.issuances),
                    "settlements": str(measurement.settlements),
                    "transfers_in": str(measurement.transfers_in),
                    "transfers_out": str(measurement.transfers_out),
                    "realized_gains_losses": str(measurement.realized_gains_losses),
                    "unrealized_gains_losses_pnl": str(measurement.unrealized_gains_losses_pnl),
                    "unrealized_gains_losses_oci": str(measurement.unrealized_gains_losses_oci),
                    "ending_balance": str(measurement.level3_reconciliation()),
                }
                reconciliations.append(recon)

        return reconciliations

    def unobservable_inputs_disclosure(self) -> List[Dict[str, Any]]:
        """Generate disclosure of unobservable inputs for Level 3.

        Returns
        -------
        list
            Unobservable inputs and sensitivities
        """
        disclosures = []

        for measurement in self.level3_measurements:
            for inp in measurement.inputs:
                if inp.observability == InputObservability.UNOBSERVABLE:
                    disclosure = {
                        "instrument_id": measurement.instrument_id,
                        "input_name": inp.name,
                        "value": str(inp.value),
                        "valuation_technique": measurement.valuation_technique.value,
                        "adjustment": str(inp.adjustment),
                    }

                    # Add sensitivity if available
                    if inp.name in measurement.unobservable_inputs_sensitivity:
                        disclosure["sensitivity"] = str(
                            measurement.unobservable_inputs_sensitivity[inp.name]
                        )

                    # Add alternative values if available
                    if inp.name in measurement.reasonably_possible_alternatives:
                        disclosure["alternative_value"] = str(
                            measurement.reasonably_possible_alternatives[inp.name]
                        )

                    disclosures.append(disclosure)

        return disclosures

    def transfers_between_levels_disclosure(self) -> Dict[str, str]:
        """Disclose transfers between hierarchy levels.

        Returns
        -------
        dict
            Transfers between levels with reasons
        """
        return {
            "level_1_to_level_2": {
                "amount": str(self.transfers_level1_to_level2),
                "reason": "Market became less active" if self.transfers_level1_to_level2 > 0 else "N/A",
            },
            "level_2_to_level_1": {
                "amount": str(self.transfers_level2_to_level1),
                "reason": "Market became active" if self.transfers_level2_to_level1 > 0 else "N/A",
            },
            "level_2_to_level_3": {
                "amount": str(self.transfers_level2_to_level3),
                "reason": "Observable inputs no longer available" if self.transfers_level2_to_level3 > 0 else "N/A",
            },
            "level_3_to_level_2": {
                "amount": str(self.transfers_level3_to_level2),
                "reason": "Observable inputs became available" if self.transfers_level3_to_level2 > 0 else "N/A",
            },
        }


def determine_hierarchy_level(
    has_active_market_quotes: bool,
    has_observable_inputs: bool,
    observable_inputs_adjustments_significant: bool = False,
) -> FairValueHierarchy:
    """Determine appropriate fair value hierarchy level.

    Parameters
    ----------
    has_active_market_quotes : bool
        Quoted prices available in active markets
    has_observable_inputs : bool
        Observable market inputs available
    observable_inputs_adjustments_significant : bool
        Whether adjustments to observable inputs are significant

    Returns
    -------
    FairValueHierarchy
        Appropriate hierarchy level

    Notes
    -----
    IFRS 13 hierarchy:
    - Level 1: Quoted prices in active markets for identical assets/liabilities
    - Level 2: Observable inputs other than Level 1 quotes
    - Level 3: Unobservable inputs
    """
    if has_active_market_quotes:
        return FairValueHierarchy.LEVEL_1

    if has_observable_inputs and not observable_inputs_adjustments_significant:
        return FairValueHierarchy.LEVEL_2

    return FairValueHierarchy.LEVEL_3


def classify_input_observability(
    input_source: str,
    input_type: str,
) -> InputObservability:
    """Classify input observability per IFRS 13.

    Parameters
    ----------
    input_source : str
        Source of the input (e.g., "Bloomberg", "Internal Model")
    input_type : str
        Type of input (e.g., "Interest Rate", "Volatility")

    Returns
    -------
    InputObservability
        Classification of input observability
    """
    observable_sources = ["Bloomberg", "Reuters", "Exchange", "Broker Quote"]
    observable_types = ["Interest Rate", "FX Rate", "Listed Equity Price"]

    # Check source
    if any(src in input_source for src in observable_sources):
        # Check type
        if any(typ in input_type for typ in observable_types):
            return InputObservability.OBSERVABLE
        return InputObservability.PARTIALLY_OBSERVABLE

    # Internal models typically use unobservable inputs
    if "Internal" in input_source or "Model" in input_source:
        return InputObservability.UNOBSERVABLE

    # Default to partially observable
    return InputObservability.PARTIALLY_OBSERVABLE
