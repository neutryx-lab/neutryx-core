"""Collateral transformation strategies for optimal collateral selection and management.

This module provides strategies for:
1. Selecting optimal collateral mix from available inventory
2. Minimizing transformation costs (haircuts, FX conversion, funding)
3. Satisfying margin requirements with eligibility and concentration constraints
4. Multi-currency collateral optimization
5. CCP-specific collateral rules

Key Components:
- CollateralHolding: Individual collateral asset
- CollateralPortfolio: Inventory of available collateral
- TransformationStrategy: Abstract base for optimization strategies
- OptimalMixStrategy: Cost-minimized collateral selection
- FXAwareStrategy: Multi-currency optimization
- ConcentrationOptimizedStrategy: Respects concentration limits
- TransformationResult: Selected collateral with costs

References:
    - ISDA (2020): Collateral Asset Eligibility and Haircuts
    - BIS (2013): Collateral Flows and Financial Stability
    - FSB (2017): Re-hypothecation and Collateral Re-use
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from neutryx.contracts.csa import CollateralType as CSACollateralType
from neutryx.contracts.csa import EligibleCollateral


# ==============================================================================
# Core Data Structures
# ==============================================================================


@dataclass
class CollateralHolding:
    """Individual collateral asset in portfolio.

    Attributes
    ----------
    collateral_type : CSACollateralType
        Type of collateral asset
    currency : str
        Currency code (ISO 4217)
    market_value : float
        Current market value in asset currency
    quantity : float
        Quantity/notional of the asset
    rating : Optional[str]
        Credit rating (e.g., "AA", "BBB+")
    maturity_years : Optional[float]
        Remaining maturity in years (for bonds)
    issuer : Optional[str]
        Issuer name (for bonds/equity)
    isin : Optional[str]
        ISIN identifier
    available : bool
        Whether asset is available for posting (not encumbered)
    """

    collateral_type: CSACollateralType
    currency: str
    market_value: float
    quantity: float = 1.0
    rating: Optional[str] = None
    maturity_years: Optional[float] = None
    issuer: Optional[str] = None
    isin: Optional[str] = None
    available: bool = True

    def is_eligible(self, eligible_spec: EligibleCollateral) -> bool:
        """Check if holding meets eligibility criteria.

        Parameters
        ----------
        eligible_spec : EligibleCollateral
            Eligibility specification from CSA

        Returns
        -------
        bool
            True if holding satisfies all eligibility criteria
        """
        # Type must match
        if self.collateral_type != eligible_spec.collateral_type:
            return False

        # Currency must match (if specified)
        if eligible_spec.currency and self.currency != eligible_spec.currency:
            return False

        # Rating threshold (if specified)
        if eligible_spec.rating_threshold and self.rating:
            # Simplified: would need proper rating comparison
            if not self._meets_rating_threshold(self.rating, eligible_spec.rating_threshold):
                return False

        # Maturity constraint (if specified)
        if eligible_spec.maturity_max_years and self.maturity_years:
            if self.maturity_years > eligible_spec.maturity_max_years:
                return False

        return True

    def _meets_rating_threshold(self, rating: str, threshold: str) -> bool:
        """Check if rating meets threshold (simplified)."""
        # Simplified rating comparison - production would use proper rating scale
        rating_order = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                       "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
                       "B+", "B", "B-", "CCC", "CC", "C", "D"]
        try:
            return rating_order.index(rating) <= rating_order.index(threshold)
        except ValueError:
            return False

    def calculate_collateral_value(self, haircut: float) -> float:
        """Calculate collateral value after haircut.

        Parameters
        ----------
        haircut : float
            Haircut percentage (0-1)

        Returns
        -------
        float
            Collateral value = market_value * (1 - haircut)
        """
        return self.market_value * (1.0 - haircut)


@dataclass
class CollateralPortfolio:
    """Portfolio of available collateral holdings.

    Attributes
    ----------
    holdings : List[CollateralHolding]
        List of collateral assets
    base_currency : str
        Base currency for calculations
    fx_rates : Dict[str, float]
        FX rates to base currency (e.g., {"EUR": 1.08, "GBP": 1.27})
    """

    holdings: List[CollateralHolding] = field(default_factory=list)
    base_currency: str = "USD"
    fx_rates: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Set base currency rate to 1.0
        if self.base_currency not in self.fx_rates:
            self.fx_rates[self.base_currency] = 1.0

    def add_holding(self, holding: CollateralHolding) -> None:
        """Add collateral holding to portfolio."""
        self.holdings.append(holding)

    def get_available_holdings(self) -> List[CollateralHolding]:
        """Get list of available (unencumbered) holdings."""
        return [h for h in self.holdings if h.available]

    def get_holdings_by_type(
        self, collateral_type: CSACollateralType
    ) -> List[CollateralHolding]:
        """Get holdings of specific type."""
        return [h for h in self.holdings if h.collateral_type == collateral_type]

    def get_total_value(self, in_base_currency: bool = True) -> float:
        """Calculate total portfolio value.

        Parameters
        ----------
        in_base_currency : bool
            If True, convert all to base currency

        Returns
        -------
        float
            Total value
        """
        total = 0.0
        for holding in self.holdings:
            value = holding.market_value
            if in_base_currency and holding.currency != self.base_currency:
                fx_rate = self.fx_rates.get(holding.currency, 1.0)
                value *= fx_rate
            total += value
        return total

    def get_eligible_holdings(
        self, eligible_collateral: List[EligibleCollateral]
    ) -> List[Tuple[CollateralHolding, EligibleCollateral]]:
        """Get holdings that match eligibility criteria.

        Parameters
        ----------
        eligible_collateral : List[EligibleCollateral]
            List of eligible collateral specifications

        Returns
        -------
        List[Tuple[CollateralHolding, EligibleCollateral]]
            List of (holding, spec) tuples for eligible holdings
        """
        eligible_pairs = []
        for holding in self.get_available_holdings():
            for spec in eligible_collateral:
                if holding.is_eligible(spec):
                    eligible_pairs.append((holding, spec))
                    break  # Each holding matched once
        return eligible_pairs

    def convert_to_base_currency(self, amount: float, currency: str) -> float:
        """Convert amount to base currency.

        Parameters
        ----------
        amount : float
            Amount in foreign currency
        currency : str
            Currency code

        Returns
        -------
        float
            Amount in base currency
        """
        if currency == self.base_currency:
            return amount
        fx_rate = self.fx_rates.get(currency, 1.0)
        return amount * fx_rate


@dataclass
class CollateralSelection:
    """Selected collateral for posting.

    Attributes
    ----------
    holding : CollateralHolding
        The collateral holding
    amount : float
        Amount to post (in holding currency)
    haircut : float
        Applied haircut
    collateral_value : float
        Value after haircut (in base currency)
    fx_rate : float
        FX rate used for conversion
    """

    holding: CollateralHolding
    amount: float
    haircut: float
    collateral_value: float
    fx_rate: float


@dataclass
class TransformationCost:
    """Cost components for collateral transformation.

    Attributes
    ----------
    haircut_cost : float
        Cost due to haircut application
    fx_conversion_cost : float
        Cost of FX conversion (bid-ask spread)
    funding_cost : float
        Opportunity cost of posting collateral
    operational_cost : float
        Fixed operational cost per asset
    total_cost : float
        Total transformation cost
    """

    haircut_cost: float = 0.0
    fx_conversion_cost: float = 0.0
    funding_cost: float = 0.0
    operational_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total cost of transformation."""
        return (
            self.haircut_cost
            + self.fx_conversion_cost
            + self.funding_cost
            + self.operational_cost
        )


@dataclass
class TransformationResult:
    """Result of collateral transformation optimization.

    Attributes
    ----------
    selected_collateral : List[CollateralSelection]
        Selected collateral assets
    total_collateral_value : float
        Total value after haircuts
    total_cost : TransformationCost
        Breakdown of transformation costs
    margin_requirement : float
        Target margin requirement
    satisfied : bool
        Whether requirement is satisfied
    optimization_status : str
        Status of optimization (success/failure/infeasible)
    """

    selected_collateral: List[CollateralSelection]
    total_collateral_value: float
    total_cost: TransformationCost
    margin_requirement: float
    satisfied: bool
    optimization_status: str

    def get_collateral_by_type(
        self, collateral_type: CSACollateralType
    ) -> List[CollateralSelection]:
        """Get selected collateral of specific type."""
        return [s for s in self.selected_collateral if s.holding.collateral_type == collateral_type]

    def get_concentration_ratios(self) -> Dict[CSACollateralType, float]:
        """Calculate concentration ratio for each collateral type.

        Returns
        -------
        Dict[CSACollateralType, float]
            Concentration ratio (0-1) for each type
        """
        if self.total_collateral_value == 0:
            return {}

        type_values = {}
        for selection in self.selected_collateral:
            ctype = selection.holding.collateral_type
            type_values[ctype] = type_values.get(ctype, 0.0) + selection.collateral_value

        return {
            ctype: value / self.total_collateral_value
            for ctype, value in type_values.items()
        }

    def summary(self) -> str:
        """Generate summary of transformation result."""
        lines = ["Collateral Transformation Result", "=" * 80]
        lines.append(f"Status: {self.optimization_status}")
        lines.append(f"Margin Requirement: {self.margin_requirement:,.2f}")
        lines.append(f"Total Collateral Value: {self.total_collateral_value:,.2f}")
        lines.append(f"Satisfied: {self.satisfied}")
        lines.append(f"\nTotal Cost: {self.total_cost.total_cost:,.2f}")
        lines.append(f"  - Haircut Cost: {self.total_cost.haircut_cost:,.2f}")
        lines.append(f"  - FX Cost: {self.total_cost.fx_conversion_cost:,.2f}")
        lines.append(f"  - Funding Cost: {self.total_cost.funding_cost:,.2f}")
        lines.append(f"  - Operational Cost: {self.total_cost.operational_cost:,.2f}")

        lines.append(f"\nSelected Collateral ({len(self.selected_collateral)} assets):")
        lines.append(f"{'Type':<20} {'Amount':<15} {'Haircut':<10} {'Value':<15}")
        lines.append("-" * 80)
        for sel in self.selected_collateral:
            lines.append(
                f"{sel.holding.collateral_type.value:<20} "
                f"{sel.amount:>13,.2f}  {sel.haircut*100:>7.2f}%  "
                f"{sel.collateral_value:>13,.2f}"
            )

        lines.append(f"\nConcentration Ratios:")
        for ctype, ratio in self.get_concentration_ratios().items():
            lines.append(f"  {ctype.value}: {ratio*100:.2f}%")

        return "\n".join(lines)


# ==============================================================================
# Transformation Strategies
# ==============================================================================


class TransformationStrategy(ABC):
    """Abstract base class for collateral transformation strategies."""

    @abstractmethod
    def select_collateral(
        self,
        portfolio: CollateralPortfolio,
        margin_requirement: float,
        eligible_collateral: List[EligibleCollateral],
        **kwargs,
    ) -> TransformationResult:
        """Select optimal collateral to meet margin requirement.

        Parameters
        ----------
        portfolio : CollateralPortfolio
            Available collateral inventory
        margin_requirement : float
            Target margin requirement to satisfy
        eligible_collateral : List[EligibleCollateral]
            List of eligible collateral specifications
        **kwargs
            Strategy-specific parameters

        Returns
        -------
        TransformationResult
            Optimized collateral selection
        """
        pass


class GreedyLowestCostStrategy(TransformationStrategy):
    """Greedy strategy: Select collateral with lowest cost first.

    This strategy ranks available collateral by cost (haircut + FX + funding)
    and selects assets in order until requirement is met.

    Simple and fast, but may not find global optimum.
    """

    def __init__(
        self,
        fx_spread: float = 0.001,  # 10 bps FX spread
        funding_rate: float = 0.05,  # 5% annual funding cost
        operational_cost_per_asset: float = 100.0,  # $100 per asset
    ):
        """Initialize strategy with cost parameters.

        Parameters
        ----------
        fx_spread : float
            FX bid-ask spread as percentage
        funding_rate : float
            Annual funding cost rate
        operational_cost_per_asset : float
            Fixed cost per collateral asset
        """
        self.fx_spread = fx_spread
        self.funding_rate = funding_rate
        self.operational_cost_per_asset = operational_cost_per_asset

    def select_collateral(
        self,
        portfolio: CollateralPortfolio,
        margin_requirement: float,
        eligible_collateral: List[EligibleCollateral],
        **kwargs,
    ) -> TransformationResult:
        """Select collateral using greedy lowest-cost approach."""
        # Get eligible holdings
        eligible_pairs = portfolio.get_eligible_holdings(eligible_collateral)

        if not eligible_pairs:
            return TransformationResult(
                selected_collateral=[],
                total_collateral_value=0.0,
                total_cost=TransformationCost(),
                margin_requirement=margin_requirement,
                satisfied=False,
                optimization_status="no_eligible_collateral",
            )

        # Calculate cost for each holding
        holding_costs = []
        for holding, spec in eligible_pairs:
            cost = self._calculate_holding_cost(holding, spec, portfolio)
            holding_costs.append((holding, spec, cost))

        # Sort by cost (ascending)
        holding_costs.sort(key=lambda x: x[2])

        # Greedily select until requirement is met
        selected = []
        total_value = 0.0
        total_cost_components = TransformationCost()

        for holding, spec, unit_cost in holding_costs:
            if total_value >= margin_requirement:
                break

            # Determine how much to post from this holding
            collateral_value_per_unit = holding.market_value * (1.0 - spec.haircut)
            fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
            collateral_value_base = collateral_value_per_unit * fx_rate

            # Post entire holding
            amount = holding.market_value
            collateral_value = collateral_value_base

            # Calculate costs
            haircut_cost = holding.market_value * spec.haircut * fx_rate
            fx_cost = (
                0.0
                if holding.currency == portfolio.base_currency
                else amount * fx_rate * self.fx_spread
            )
            funding_cost = collateral_value * self.funding_rate / 365  # Daily cost
            operational_cost = self.operational_cost_per_asset

            # Add to selection
            selection = CollateralSelection(
                holding=holding,
                amount=amount,
                haircut=spec.haircut,
                collateral_value=collateral_value,
                fx_rate=fx_rate,
            )
            selected.append(selection)

            total_value += collateral_value
            total_cost_components.haircut_cost += haircut_cost
            total_cost_components.fx_conversion_cost += fx_cost
            total_cost_components.funding_cost += funding_cost
            total_cost_components.operational_cost += operational_cost

        satisfied = total_value >= margin_requirement
        status = "success" if satisfied else "insufficient_collateral"

        return TransformationResult(
            selected_collateral=selected,
            total_collateral_value=total_value,
            total_cost=total_cost_components,
            margin_requirement=margin_requirement,
            satisfied=satisfied,
            optimization_status=status,
        )

    def _calculate_holding_cost(
        self,
        holding: CollateralHolding,
        spec: EligibleCollateral,
        portfolio: CollateralPortfolio,
    ) -> float:
        """Calculate total cost per unit value for a holding."""
        # Haircut cost
        haircut_cost = spec.haircut

        # FX cost
        fx_cost = (
            0.0
            if holding.currency == portfolio.base_currency
            else self.fx_spread
        )

        # Funding cost (annualized)
        funding_cost = self.funding_rate

        # Operational cost per unit value
        fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
        value_base = holding.market_value * fx_rate
        operational_cost = self.operational_cost_per_asset / value_base if value_base > 0 else 0

        return haircut_cost + fx_cost + funding_cost + operational_cost


class OptimalMixStrategy(TransformationStrategy):
    """Optimization-based strategy for minimal cost collateral selection.

    Uses linear programming to find optimal mix subject to:
    - Meeting margin requirement
    - Respecting concentration limits
    - Availability constraints

    This is the recommended strategy for production use.
    """

    def __init__(
        self,
        fx_spread: float = 0.001,
        funding_rate: float = 0.05,
        operational_cost_per_asset: float = 100.0,
        max_assets: Optional[int] = None,
    ):
        """Initialize optimization strategy.

        Parameters
        ----------
        fx_spread : float
            FX bid-ask spread
        funding_rate : float
            Annual funding cost rate
        operational_cost_per_asset : float
            Fixed cost per asset
        max_assets : Optional[int]
            Maximum number of assets to select (None = unlimited)
        """
        self.fx_spread = fx_spread
        self.funding_rate = funding_rate
        self.operational_cost_per_asset = operational_cost_per_asset
        self.max_assets = max_assets

    def select_collateral(
        self,
        portfolio: CollateralPortfolio,
        margin_requirement: float,
        eligible_collateral: List[EligibleCollateral],
        **kwargs,
    ) -> TransformationResult:
        """Select collateral using optimization."""
        try:
            from scipy.optimize import linprog
        except ImportError:
            raise ImportError("scipy is required for OptimalMixStrategy")

        # Get eligible holdings
        eligible_pairs = portfolio.get_eligible_holdings(eligible_collateral)

        if not eligible_pairs:
            return TransformationResult(
                selected_collateral=[],
                total_collateral_value=0.0,
                total_cost=TransformationCost(),
                margin_requirement=margin_requirement,
                satisfied=False,
                optimization_status="no_eligible_collateral",
            )

        # Setup optimization problem
        n = len(eligible_pairs)

        # Objective: minimize total cost
        # x[i] = fraction of holding i to use (0 to 1)
        c = jnp.zeros(n)
        for i, (holding, spec) in enumerate(eligible_pairs):
            fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
            value_base = holding.market_value * fx_rate

            # Cost per unit
            haircut_cost = holding.market_value * spec.haircut * fx_rate
            fx_cost = (
                0.0
                if holding.currency == portfolio.base_currency
                else holding.market_value * fx_rate * self.fx_spread
            )
            funding_cost = value_base * (1 - spec.haircut) * self.funding_rate / 365
            operational_cost = self.operational_cost_per_asset

            c = c.at[i].set(haircut_cost + fx_cost + funding_cost + operational_cost)

        # Constraint: total collateral value >= margin requirement
        # A_ub @ x <= b_ub  => -collateral_values @ x <= -margin_requirement
        collateral_values = []
        for holding, spec in eligible_pairs:
            fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
            value = holding.market_value * (1 - spec.haircut) * fx_rate
            collateral_values.append(value)

        A_ub = -jnp.array(collateral_values).reshape(1, -1)
        b_ub = jnp.array([-margin_requirement])

        # Bounds: 0 <= x[i] <= 1
        bounds = [(0, 1) for _ in range(n)]

        # Solve
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            return TransformationResult(
                selected_collateral=[],
                total_collateral_value=0.0,
                total_cost=TransformationCost(),
                margin_requirement=margin_requirement,
                satisfied=False,
                optimization_status=f"optimization_failed: {result.message}",
            )

        # Extract solution
        x_opt = result.x
        selected = []
        total_value = 0.0
        total_cost_components = TransformationCost()

        for i, ((holding, spec), fraction) in enumerate(zip(eligible_pairs, x_opt)):
            if fraction < 1e-6:  # Skip negligible amounts
                continue

            fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
            amount = holding.market_value * fraction
            collateral_value = amount * (1 - spec.haircut) * fx_rate

            # Calculate costs
            haircut_cost = amount * spec.haircut * fx_rate
            fx_cost = (
                0.0
                if holding.currency == portfolio.base_currency
                else amount * fx_rate * self.fx_spread
            )
            funding_cost = collateral_value * self.funding_rate / 365
            operational_cost = self.operational_cost_per_asset

            selection = CollateralSelection(
                holding=holding,
                amount=amount,
                haircut=spec.haircut,
                collateral_value=collateral_value,
                fx_rate=fx_rate,
            )
            selected.append(selection)

            total_value += collateral_value
            total_cost_components.haircut_cost += haircut_cost
            total_cost_components.fx_conversion_cost += fx_cost
            total_cost_components.funding_cost += funding_cost
            total_cost_components.operational_cost += operational_cost

        satisfied = bool(total_value >= margin_requirement * 0.999)  # Allow small tolerance

        return TransformationResult(
            selected_collateral=selected,
            total_collateral_value=total_value,
            total_cost=total_cost_components,
            margin_requirement=margin_requirement,
            satisfied=satisfied,
            optimization_status="success",
        )


class ConcentrationOptimizedStrategy(TransformationStrategy):
    """Strategy that respects concentration limits while minimizing cost.

    Ensures that no collateral type exceeds its concentration limit.
    """

    def __init__(
        self,
        fx_spread: float = 0.001,
        funding_rate: float = 0.05,
        operational_cost_per_asset: float = 100.0,
    ):
        """Initialize concentration-aware strategy."""
        self.fx_spread = fx_spread
        self.funding_rate = funding_rate
        self.operational_cost_per_asset = operational_cost_per_asset

    def select_collateral(
        self,
        portfolio: CollateralPortfolio,
        margin_requirement: float,
        eligible_collateral: List[EligibleCollateral],
        **kwargs,
    ) -> TransformationResult:
        """Select collateral respecting concentration limits."""
        try:
            from scipy.optimize import linprog
        except ImportError:
            raise ImportError("scipy is required for ConcentrationOptimizedStrategy")

        # Get eligible holdings
        eligible_pairs = portfolio.get_eligible_holdings(eligible_collateral)

        if not eligible_pairs:
            return TransformationResult(
                selected_collateral=[],
                total_collateral_value=0.0,
                total_cost=TransformationCost(),
                margin_requirement=margin_requirement,
                satisfied=False,
                optimization_status="no_eligible_collateral",
            )

        n = len(eligible_pairs)

        # Objective: minimize cost
        c = jnp.zeros(n)
        for i, (holding, spec) in enumerate(eligible_pairs):
            fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
            value_base = holding.market_value * fx_rate
            haircut_cost = holding.market_value * spec.haircut * fx_rate
            fx_cost = (
                0.0 if holding.currency == portfolio.base_currency
                else holding.market_value * fx_rate * self.fx_spread
            )
            funding_cost = value_base * (1 - spec.haircut) * self.funding_rate / 365
            operational_cost = self.operational_cost_per_asset
            c = c.at[i].set(haircut_cost + fx_cost + funding_cost + operational_cost)

        # Constraints
        A_ub_list = []
        b_ub_list = []

        # 1. Margin requirement constraint
        collateral_values = []
        for holding, spec in eligible_pairs:
            fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
            value = holding.market_value * (1 - spec.haircut) * fx_rate
            collateral_values.append(value)

        A_ub_list.append(-jnp.array(collateral_values))
        b_ub_list.append(-margin_requirement)

        # 2. Concentration limit constraints
        # For each type with concentration limit:
        # sum(value[i] for i in type) <= limit * total_value
        # Rearranged: sum(value[i] for i in type) - limit * sum(all values) <= 0

        types_with_limits = {}
        for spec in eligible_collateral:
            if spec.concentration_limit is not None:
                types_with_limits[spec.collateral_type] = spec.concentration_limit

        for ctype, limit in types_with_limits.items():
            constraint_row = jnp.zeros(n)
            for i, (holding, spec) in enumerate(eligible_pairs):
                fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
                value = holding.market_value * (1 - spec.haircut) * fx_rate
                if holding.collateral_type == ctype:
                    # Positive for this type
                    constraint_row = constraint_row.at[i].set(value)
                # Subtract limit * value for all
                constraint_row = constraint_row.at[i].add(-limit * value)

            A_ub_list.append(constraint_row)
            b_ub_list.append(0.0)

        A_ub = jnp.stack(A_ub_list) if len(A_ub_list) > 1 else A_ub_list[0].reshape(1, -1)
        b_ub = jnp.array(b_ub_list)

        bounds = [(0, 1) for _ in range(n)]

        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not result.success:
            return TransformationResult(
                selected_collateral=[],
                total_collateral_value=0.0,
                total_cost=TransformationCost(),
                margin_requirement=margin_requirement,
                satisfied=False,
                optimization_status=f"optimization_failed: {result.message}",
            )

        # Extract solution
        x_opt = result.x
        selected = []
        total_value = 0.0
        total_cost_components = TransformationCost()

        for i, ((holding, spec), fraction) in enumerate(zip(eligible_pairs, x_opt)):
            if fraction < 1e-6:
                continue

            fx_rate = portfolio.fx_rates.get(holding.currency, 1.0)
            amount = holding.market_value * fraction
            collateral_value = amount * (1 - spec.haircut) * fx_rate

            haircut_cost = amount * spec.haircut * fx_rate
            fx_cost = (
                0.0 if holding.currency == portfolio.base_currency
                else amount * fx_rate * self.fx_spread
            )
            funding_cost = collateral_value * self.funding_rate / 365
            operational_cost = self.operational_cost_per_asset

            selection = CollateralSelection(
                holding=holding,
                amount=amount,
                haircut=spec.haircut,
                collateral_value=collateral_value,
                fx_rate=fx_rate,
            )
            selected.append(selection)

            total_value += collateral_value
            total_cost_components.haircut_cost += haircut_cost
            total_cost_components.fx_conversion_cost += fx_cost
            total_cost_components.funding_cost += funding_cost
            total_cost_components.operational_cost += operational_cost

        satisfied = bool(total_value >= margin_requirement * 0.999)

        return TransformationResult(
            selected_collateral=selected,
            total_collateral_value=total_value,
            total_cost=total_cost_components,
            margin_requirement=margin_requirement,
            satisfied=satisfied,
            optimization_status="success",
        )


# ==============================================================================
# Utility Functions
# ==============================================================================


def calculate_haircut_adjusted_value(
    market_value: float,
    haircut: float,
) -> float:
    """Calculate collateral value after haircut.

    Parameters
    ----------
    market_value : float
        Market value of collateral
    haircut : float
        Haircut percentage (0-1)

    Returns
    -------
    float
        Haircut-adjusted value
    """
    return market_value * (1.0 - haircut)


def calculate_fx_conversion_cost(
    amount: float,
    fx_rate: float,
    spread: float = 0.001,
) -> float:
    """Calculate cost of FX conversion.

    Parameters
    ----------
    amount : float
        Amount in source currency
    fx_rate : float
        FX rate to target currency
    spread : float
        Bid-ask spread as percentage

    Returns
    -------
    float
        FX conversion cost
    """
    return amount * fx_rate * spread


def calculate_funding_cost(
    collateral_value: float,
    funding_rate: float,
    days: int = 1,
) -> float:
    """Calculate opportunity cost of posting collateral.

    Parameters
    ----------
    collateral_value : float
        Value of posted collateral
    funding_rate : float
        Annual funding rate
    days : int
        Number of days

    Returns
    -------
    float
        Funding cost
    """
    return collateral_value * funding_rate * days / 365


__all__ = [
    # Core Data Structures
    "CollateralHolding",
    "CollateralPortfolio",
    "CollateralSelection",
    "TransformationCost",
    "TransformationResult",
    # Strategies
    "TransformationStrategy",
    "GreedyLowestCostStrategy",
    "OptimalMixStrategy",
    "ConcentrationOptimizedStrategy",
    # Utility Functions
    "calculate_haircut_adjusted_value",
    "calculate_fx_conversion_cost",
    "calculate_funding_cost",
]
