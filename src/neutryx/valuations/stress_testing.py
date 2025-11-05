"""Enhanced stress testing framework with concentration risk and advanced analytics.

This module extends the basic stress testing capabilities with:
- Comprehensive historical scenarios
- Hypothetical scenario builder
- Advanced reverse stress testing with optimization
- Concentration risk metrics (Herfindahl, top-N, sector)
- Portfolio-level stress analytics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

# Import existing stress testing infrastructure
from neutryx.valuations.stress_test import StressScenario, HISTORICAL_SCENARIOS


class ConcentrationMetric(Enum):
    """Types of concentration risk metrics."""

    HERFINDAHL = "herfindahl"  # Herfindahl-Hirschman Index
    GINI = "gini"  # Gini coefficient
    TOP_N = "top_n"  # Top-N concentration
    ENTROPY = "entropy"  # Shannon entropy


@dataclass
class HistoricalScenarioLibrary:
    """Extended library of historical stress scenarios."""

    # All historical scenarios from original plus new ones
    scenarios: Dict[str, StressScenario] = field(default_factory=lambda: {
        **HISTORICAL_SCENARIOS,
        # Emerging market crises
        "asian_crisis_1997": StressScenario(
            name="Asian Financial Crisis 1997",
            description="Thai baht collapse and contagion across Asia",
            shocks={"equity": -0.35, "fx_em": -0.30, "rates": 0.05, "credit_spread": 0.08},
        ),
        "russian_default_1998": StressScenario(
            name="Russian Default 1998",
            description="Russian ruble collapse and bond default",
            shocks={"equity": -0.15, "fx_em": -0.40, "credit_spread": 0.12, "volatility": 2.5},
        ),
        "argentina_crisis_2001": StressScenario(
            name="Argentina Crisis 2001",
            description="Argentine peso crisis and default",
            shocks={"equity": -0.25, "fx_em": -0.35, "credit_spread": 0.10},
        ),
        "taper_tantrum_2013": StressScenario(
            name="Taper Tantrum 2013",
            description="Fed taper announcement roiling EM markets",
            shocks={"rates": 0.01, "fx_em": -0.15, "equity": -0.08, "credit_spread": 0.03},
        ),
        # European crises
        "eurozone_crisis_2011": StressScenario(
            name="Eurozone Sovereign Debt Crisis 2011",
            description="Greek debt crisis and eurozone contagion",
            shocks={"equity": -0.20, "credit_spread": 0.08, "fx_eur": -0.10, "volatility": 2.0},
        ),
        "brexit_2016": StressScenario(
            name="Brexit Vote 2016",
            description="UK referendum to leave the European Union",
            shocks={"equity": -0.12, "fx_gbp": -0.15, "volatility": 1.5, "credit_spread": 0.02},
        ),
        # Technology bubbles
        "dotcom_bust_2000": StressScenario(
            name="Dot-com Bubble Burst 2000-2002",
            description="Tech stock crash following internet bubble",
            shocks={"equity": -0.49, "volatility": 2.2, "rates": -0.015},
        ),
        # Energy and commodity
        "oil_crash_2014": StressScenario(
            name="Oil Price Crash 2014",
            description="Oil price collapse from $110 to $40",
            shocks={"commodity_energy": -0.65, "fx_em": -0.12, "equity": -0.10},
        ),
        "oil_negative_2020": StressScenario(
            name="Negative Oil Prices 2020",
            description="WTI crude briefly trades at negative prices",
            shocks={"commodity_energy": -1.20, "equity": -0.15, "volatility": 3.0},
        ),
        # Credit events
        "ltcm_1998": StressScenario(
            name="LTCM Collapse 1998",
            description="Long-Term Capital Management hedge fund failure",
            shocks={"credit_spread": 0.05, "volatility": 2.0, "equity": -0.12},
        ),
        "credit_crisis_2002": StressScenario(
            name="Corporate Credit Crisis 2002",
            description="Worldcom, Enron accounting scandals",
            shocks={"equity": -0.22, "credit_spread": 0.06, "volatility": 1.5},
        ),
        # Inflation/stagflation
        "inflation_shock_2022": StressScenario(
            name="Inflation Shock 2022",
            description="Post-COVID inflation surge and Fed tightening",
            shocks={"rates": 0.025, "equity": -0.18, "volatility": 1.3, "credit_spread": 0.02},
        ),
        # Regional crises
        "china_devaluation_2015": StressScenario(
            name="China Currency Devaluation 2015",
            description="PBoC unexpected yuan devaluation",
            shocks={"fx_cny": -0.04, "equity": -0.08, "commodity": -0.10, "volatility": 1.5},
        ),
        "snb_peg_removal_2015": StressScenario(
            name="Swiss Franc Shock 2015",
            description="SNB removes EUR/CHF floor",
            shocks={"fx_chf": 0.20, "equity": -0.06, "volatility": 1.8},
        ),
    })

    def get_scenario(self, name: str) -> StressScenario:
        """Get a scenario by name."""
        if name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {name}")
        return self.scenarios[name]

    def list_scenarios(self) -> List[str]:
        """List all available scenario names."""
        return list(self.scenarios.keys())

    def filter_by_crisis_type(self, crisis_type: str) -> List[StressScenario]:
        """Filter scenarios by type (equity, credit, fx, etc.)."""
        # Simple keyword matching
        keywords = {
            "equity": ["crash", "bubble", "black monday"],
            "credit": ["credit", "default", "ltcm"],
            "fx": ["currency", "devaluation", "franc", "baht", "peso"],
            "rates": ["rate shock", "taper", "inflation"],
            "emerging": ["asian", "russian", "argentina", "taper", "china"],
            "commodity": ["oil", "commodity"],
        }

        if crisis_type not in keywords:
            return []

        matching = []
        for scenario in self.scenarios.values():
            if any(kw in scenario.name.lower() or kw in scenario.description.lower()
                   for kw in keywords[crisis_type]):
                matching.append(scenario)

        return matching


@dataclass
class HypotheticalScenarioBuilder:
    """Builder for creating custom hypothetical stress scenarios."""

    name: str = "Custom Scenario"
    description: str = ""
    _shocks: Dict[str, float] = field(default_factory=dict)

    def add_shock(self, risk_factor: str, shock_value: float) -> 'HypotheticalScenarioBuilder':
        """Add a shock to the scenario.

        Args:
            risk_factor: Name of the risk factor
            shock_value: Shock magnitude (relative, e.g., -0.30 for -30%)

        Returns:
            Self for method chaining
        """
        self._shocks[risk_factor] = shock_value
        return self

    def parallel_shift(self, risk_factor: str, shift: float) -> 'HypotheticalScenarioBuilder':
        """Add a parallel shift across a risk factor."""
        return self.add_shock(risk_factor, shift)

    def twist(self, short_factor: str, long_factor: str,
              short_shift: float, long_shift: float) -> 'HypotheticalScenarioBuilder':
        """Add a twist (steepening/flattening) across a curve."""
        self.add_shock(short_factor, short_shift)
        self.add_shock(long_factor, long_shift)
        return self

    def correlation_shock(self, factor1: str, factor2: str,
                          shock: float) -> 'HypotheticalScenarioBuilder':
        """Add a correlation shock between two factors."""
        corr_key = f"corr_{factor1}_{factor2}"
        self.add_shock(corr_key, shock)
        return self

    def build(self) -> StressScenario:
        """Build the final stress scenario."""
        if not self.description:
            self.description = f"Hypothetical scenario with {len(self._shocks)} shocks"

        return StressScenario(
            name=self.name,
            description=self.description,
            shocks=self._shocks.copy(),
        )


@dataclass
class ConcentrationRiskMetrics:
    """Concentration risk metrics for portfolios.

    Measures how concentrated a portfolio is across various dimensions
    (counterparties, sectors, geographies, products, etc.).
    """

    @staticmethod
    def herfindahl_index(exposures: Array) -> float:
        """Compute Herfindahl-Hirschman Index (HHI).

        HHI = sum(s_i^2) where s_i is the share of exposure i.

        Range: [1/N, 1] where N is number of exposures
        - 1/N: perfectly diversified
        - 1: fully concentrated

        Args:
            exposures: Array of exposures (positive values)

        Returns:
            HHI value
        """
        exposures = jnp.abs(exposures)  # Handle negative exposures
        total = jnp.sum(exposures)

        # Avoid division by zero
        total = jnp.where(total == 0, 1.0, total)

        shares = exposures / total
        hhi = jnp.sum(shares ** 2)

        return float(hhi)

    @staticmethod
    def gini_coefficient(exposures: Array) -> float:
        """Compute Gini coefficient.

        Measures inequality in the distribution of exposures.

        Range: [0, 1]
        - 0: perfect equality
        - 1: maximum inequality

        Args:
            exposures: Array of exposures

        Returns:
            Gini coefficient
        """
        exposures = jnp.abs(jnp.sort(exposures))
        n = len(exposures)

        if n == 0:
            return 0.0

        cumsum = jnp.cumsum(exposures)
        total = jnp.sum(exposures)

        # Avoid division by zero
        total = jnp.where(total == 0, 1.0, total)

        gini = (2.0 * jnp.sum((jnp.arange(n) + 1) * exposures)) / (n * total) - (n + 1) / n

        return float(gini)

    @staticmethod
    def top_n_concentration(exposures: Array, n: int = 5) -> float:
        """Compute top-N concentration ratio.

        Fraction of total exposure in the top N largest exposures.

        Args:
            exposures: Array of exposures
            n: Number of top exposures to consider

        Returns:
            Concentration ratio (0 to 1)
        """
        exposures = jnp.abs(exposures)
        total = jnp.sum(exposures)

        # Avoid division by zero
        total = jnp.where(total == 0, 1.0, total)

        # Sort and take top N
        sorted_exposures = jnp.sort(exposures)[::-1]
        top_n_sum = jnp.sum(sorted_exposures[:n])

        concentration = top_n_sum / total

        return float(concentration)

    @staticmethod
    def entropy(exposures: Array) -> float:
        """Compute Shannon entropy.

        Higher entropy indicates more diversification.

        Args:
            exposures: Array of exposures

        Returns:
            Entropy value
        """
        exposures = jnp.abs(exposures)
        total = jnp.sum(exposures)

        # Avoid division by zero
        total = jnp.where(total == 0, 1.0, total)

        probabilities = exposures / total

        # Avoid log(0)
        probabilities = jnp.where(probabilities > 0, probabilities, 1e-10)

        entropy_val = -jnp.sum(probabilities * jnp.log(probabilities))

        return float(entropy_val)

    @classmethod
    def compute_all_metrics(cls, exposures: Array, top_n: int = 5) -> Dict[str, float]:
        """Compute all concentration metrics.

        Args:
            exposures: Array of exposures
            top_n: Number for top-N concentration

        Returns:
            Dictionary of metrics
        """
        return {
            "herfindahl_index": cls.herfindahl_index(exposures),
            "gini_coefficient": cls.gini_coefficient(exposures),
            f"top_{top_n}_concentration": cls.top_n_concentration(exposures, n=top_n),
            "entropy": cls.entropy(exposures),
        }


@dataclass
class ReverseStressTestResult:
    """Result from reverse stress testing."""

    scenario_name: str
    target_loss: float
    required_shocks: Dict[str, float]
    implied_scenario: StressScenario
    plausibility_score: Optional[float] = None


class AdvancedReverseStressTest:
    """Advanced reverse stress testing using optimization.

    Finds the minimal shock magnitude that produces a target loss,
    or finds realistic scenarios that could cause specified losses.
    """

    def __init__(
        self,
        valuation_fn: Callable,
        base_params: Dict[str, float],
        risk_factors: List[str],
        optimizer: Optional[optax.GradientTransformation] = None,
    ):
        """Initialize reverse stress tester.

        Args:
            valuation_fn: Portfolio valuation function
            base_params: Base market parameters
            risk_factors: List of risk factors to stress
            optimizer: Optax optimizer (default: Adam)
        """
        self.valuation_fn = valuation_fn
        self.base_params = base_params
        self.risk_factors = risk_factors
        self.base_value = valuation_fn(**base_params)

        if optimizer is None:
            optimizer = optax.adam(learning_rate=0.01)
        self.optimizer = optimizer

    def find_minimal_shock(
        self,
        target_loss: float,
        factor: str,
        max_steps: int = 500,
        tolerance: float = 1e-4,
    ) -> float:
        """Find minimal shock to a single factor that produces target loss.

        Uses gradient-based optimization for smooth valuation functions.

        Args:
            target_loss: Target loss amount (negative for losses)
            factor: Risk factor to shock
            max_steps: Maximum optimization steps
            tolerance: Convergence tolerance

        Returns:
            Required shock magnitude
        """
        if factor not in self.base_params:
            raise ValueError(f"Factor {factor} not in base params")

        target_value = self.base_value + target_loss

        # Initialize shock parameter
        shock = jnp.array(0.0)
        opt_state = self.optimizer.init(shock)

        @jax.jit
        def loss_fn(shock_val):
            """Squared distance from target."""
            stressed_params = self.base_params.copy()
            stressed_params[factor] = self.base_params[factor] * (1 + shock_val)
            stressed_value = self.valuation_fn(**stressed_params)
            return (stressed_value - target_value) ** 2

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        for _ in range(max_steps):
            loss, grad = loss_and_grad(shock)

            if loss < tolerance:
                break

            updates, opt_state = self.optimizer.update(grad, opt_state)
            shock = optax.apply_updates(shock, updates)

        return float(shock)

    def find_realistic_scenario(
        self,
        target_loss: float,
        historical_scenarios: Optional[List[StressScenario]] = None,
        max_steps: int = 500,
    ) -> ReverseStressTestResult:
        """Find a realistic scenario that produces target loss.

        Searches for a combination of shocks across multiple risk factors
        that produces the target loss while staying plausible (similar to
        historical scenarios).

        Args:
            target_loss: Target loss amount
            historical_scenarios: Reference scenarios for plausibility
            max_steps: Maximum optimization steps

        Returns:
            ReverseStressTestResult with optimal scenario
        """
        n_factors = len(self.risk_factors)

        # Initialize shocks
        shocks = jnp.zeros(n_factors)
        opt_state = self.optimizer.init(shocks)

        target_value = self.base_value + target_loss

        @jax.jit
        def loss_fn(shock_vec):
            """Objective: hit target loss with minimal total shock magnitude."""
            stressed_params = self.base_params.copy()

            for i, factor in enumerate(self.risk_factors):
                if factor in stressed_params:
                    stressed_params[factor] = self.base_params[factor] * (1 + shock_vec[i])

            stressed_value = self.valuation_fn(**stressed_params)

            # Primary objective: hit target
            target_error = (stressed_value - target_value) ** 2

            # Secondary objective: minimize shock magnitude (for plausibility)
            shock_magnitude = jnp.sum(shock_vec ** 2)

            return target_error + 0.01 * shock_magnitude

        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        for _ in range(max_steps):
            loss, grad = loss_and_grad(shocks)
            updates, opt_state = self.optimizer.update(grad, opt_state)
            shocks = optax.apply_updates(shocks, updates)

        # Build result
        shock_dict = {factor: float(shocks[i])
                      for i, factor in enumerate(self.risk_factors)}

        scenario = StressScenario(
            name="Reverse Stress Test Result",
            description=f"Scenario producing {target_loss:.2f} loss",
            shocks=shock_dict,
        )

        # Compute plausibility score if historical scenarios provided
        plausibility = None
        if historical_scenarios:
            plausibility = self._compute_plausibility(shock_dict, historical_scenarios)

        return ReverseStressTestResult(
            scenario_name="Reverse Test",
            target_loss=target_loss,
            required_shocks=shock_dict,
            implied_scenario=scenario,
            plausibility_score=plausibility,
        )

    def _compute_plausibility(
        self,
        shocks: Dict[str, float],
        historical_scenarios: List[StressScenario],
    ) -> float:
        """Compute plausibility score by comparing to historical scenarios.

        Returns a score between 0 and 1, where 1 means the scenario is
        similar to historical ones.
        """
        if not historical_scenarios:
            return 0.5  # Neutral score

        # Compute distance to each historical scenario
        distances = []
        for hist_scenario in historical_scenarios:
            # Euclidean distance in shock space
            dist = 0.0
            common_factors = set(shocks.keys()) & set(hist_scenario.shocks.keys())

            for factor in common_factors:
                dist += (shocks[factor] - hist_scenario.shocks[factor]) ** 2

            if common_factors:
                dist = (dist / len(common_factors)) ** 0.5
                distances.append(dist)

        if not distances:
            return 0.5

        # Plausibility is inverse of minimum distance
        min_distance = min(distances)
        plausibility = 1.0 / (1.0 + min_distance)

        return plausibility


__all__ = [
    "ConcentrationMetric",
    "HistoricalScenarioLibrary",
    "HypotheticalScenarioBuilder",
    "ConcentrationRiskMetrics",
    "ReverseStressTestResult",
    "AdvancedReverseStressTest",
]
