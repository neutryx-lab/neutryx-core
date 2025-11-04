"""Scenario analysis engine for risk management and what-if analysis.

This module provides comprehensive scenario analysis capabilities including:
- Market risk scenarios (spot, vol, rates, correlations)
- Historical scenarios
- Hypothetical scenarios
- Sensitivity analysis
- Scenario generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


class ScenarioType(Enum):
    """Type of scenario."""
    HISTORICAL = "historical"  # Based on historical market moves
    HYPOTHETICAL = "hypothetical"  # User-defined scenarios
    MONTE_CARLO = "monte_carlo"  # Simulated scenarios
    SENSITIVITY = "sensitivity"  # Parameter sensitivity analysis
    STRESS = "stress"  # Stress test scenarios


@dataclass
class MarketScenario:
    """Definition of a market scenario.

    Attributes:
        name: Scenario name
        scenario_type: Type of scenario
        spot_shocks: Relative spot price changes by asset
        vol_shocks: Absolute volatility changes by asset
        rate_shocks: Absolute interest rate changes by currency
        correlation_adjustments: Correlation matrix adjustments
        spread_shocks: Credit spread shocks
        fx_shocks: FX rate shocks
        probability: Scenario probability (for expected shortfall)
    """

    name: str
    scenario_type: ScenarioType
    spot_shocks: Dict[str, float] = field(default_factory=dict)
    vol_shocks: Dict[str, float] = field(default_factory=dict)
    rate_shocks: Dict[str, float] = field(default_factory=dict)
    correlation_adjustments: Dict[Tuple[str, str], float] = field(default_factory=dict)
    spread_shocks: Dict[str, float] = field(default_factory=dict)
    fx_shocks: Dict[str, float] = field(default_factory=dict)
    probability: float = 1.0

    def apply_to_market_data(self, base_data: Dict[str, float]) -> Dict[str, float]:
        """Apply scenario shocks to base market data.

        Args:
            base_data: Base market data dictionary

        Returns:
            Shocked market data
        """
        shocked_data = base_data.copy()

        # Apply spot shocks (multiplicative)
        for asset, shock in self.spot_shocks.items():
            key = f"spot_{asset}"
            if key in shocked_data:
                shocked_data[key] = shocked_data[key] * (1 + shock)

        # Apply vol shocks (additive)
        for asset, shock in self.vol_shocks.items():
            key = f"vol_{asset}"
            if key in shocked_data:
                shocked_data[key] = shocked_data[key] + shock

        # Apply rate shocks (additive)
        for currency, shock in self.rate_shocks.items():
            key = f"rate_{currency}"
            if key in shocked_data:
                shocked_data[key] = shocked_data[key] + shock

        # Apply FX shocks (multiplicative)
        for pair, shock in self.fx_shocks.items():
            key = f"fx_{pair}"
            if key in shocked_data:
                shocked_data[key] = shocked_data[key] * (1 + shock)

        return shocked_data


@dataclass
class ScenarioResult:
    """Result of scenario analysis.

    Attributes:
        scenario_name: Name of scenario
        base_value: Portfolio value in base scenario
        scenario_value: Portfolio value in stressed scenario
        pnl: P&L impact (scenario_value - base_value)
        pnl_pct: P&L as percentage of base value
        greeks_impact: Impact on Greeks
        risk_metrics: Additional risk metrics
    """

    scenario_name: str
    base_value: float
    scenario_value: float
    pnl: float
    pnl_pct: float
    greeks_impact: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict[str, float]] = None


class ScenarioEngine:
    """Scenario analysis engine for portfolio valuation.

    Performs comprehensive scenario analysis including historical scenarios,
    hypothetical stresses, and Monte Carlo simulation.
    """

    def __init__(
        self,
        portfolio_pricer: Callable,
        base_market_data: Dict[str, float]
    ):
        """Initialize scenario engine.

        Args:
            portfolio_pricer: Function to price portfolio given market data
            base_market_data: Base market data dictionary
        """
        self.portfolio_pricer = portfolio_pricer
        self.base_market_data = base_market_data
        self.scenarios: List[MarketScenario] = []

    def add_scenario(self, scenario: MarketScenario):
        """Add a scenario to the engine.

        Args:
            scenario: Market scenario to add
        """
        self.scenarios.append(scenario)

    def run_scenario(self, scenario: MarketScenario) -> ScenarioResult:
        """Run a single scenario.

        Args:
            scenario: Scenario to run

        Returns:
            Scenario result
        """
        # Base portfolio value
        base_value = self.portfolio_pricer(self.base_market_data)

        # Apply scenario shocks
        shocked_data = scenario.apply_to_market_data(self.base_market_data)

        # Scenario portfolio value
        scenario_value = self.portfolio_pricer(shocked_data)

        # P&L
        pnl = scenario_value - base_value
        pnl_pct = (pnl / base_value) * 100 if base_value != 0 else 0.0

        return ScenarioResult(
            scenario_name=scenario.name,
            base_value=base_value,
            scenario_value=scenario_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

    def run_all_scenarios(self) -> List[ScenarioResult]:
        """Run all scenarios.

        Returns:
            List of scenario results
        """
        return [self.run_scenario(scenario) for scenario in self.scenarios]

    def worst_case_scenario(self) -> ScenarioResult:
        """Find worst-case scenario (maximum loss).

        Returns:
            Worst-case scenario result
        """
        results = self.run_all_scenarios()
        return min(results, key=lambda r: r.pnl)

    def best_case_scenario(self) -> ScenarioResult:
        """Find best-case scenario (maximum gain).

        Returns:
            Best-case scenario result
        """
        results = self.run_all_scenarios()
        return max(results, key=lambda r: r.pnl)

    def expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Compute expected shortfall (CVaR).

        Args:
            confidence_level: Confidence level (e.g., 0.95 = 95%)

        Returns:
            Expected shortfall (average loss beyond VaR)
        """
        results = self.run_all_scenarios()

        # Sort by P&L
        sorted_pnls = sorted([r.pnl for r in results])

        # VaR threshold
        var_index = int((1 - confidence_level) * len(sorted_pnls))

        # Expected shortfall: average of losses beyond VaR
        tail_losses = sorted_pnls[:var_index]

        if tail_losses:
            return float(jnp.mean(jnp.array(tail_losses)))
        else:
            return 0.0


@dataclass
class SensitivityAnalysis:
    """Sensitivity analysis for portfolio parameters.

    Performs one-dimensional sensitivity analysis by varying each
    parameter independently.
    """

    portfolio_pricer: Callable
    base_params: Dict[str, float]

    def compute_sensitivity(
        self,
        param_name: str,
        shock_sizes: List[float]
    ) -> Dict[float, float]:
        """Compute sensitivity to a single parameter.

        Args:
            param_name: Parameter to shock
            shock_sizes: Relative shock sizes (e.g., [-0.1, -0.05, 0, 0.05, 0.1])

        Returns:
            Dictionary mapping shock size to portfolio value
        """
        sensitivities = {}

        base_value = self.base_params[param_name]

        for shock in shock_sizes:
            # Apply shock
            shocked_params = self.base_params.copy()
            shocked_params[param_name] = base_value * (1 + shock)

            # Revalue
            value = self.portfolio_pricer(shocked_params)
            sensitivities[shock] = value

        return sensitivities

    def compute_all_sensitivities(
        self,
        shock_sizes: List[float] = None
    ) -> Dict[str, Dict[float, float]]:
        """Compute sensitivity to all parameters.

        Args:
            shock_sizes: Shock sizes to apply (default: [-10%, -5%, 0, +5%, +10%])

        Returns:
            Dictionary mapping parameter name to sensitivities
        """
        if shock_sizes is None:
            shock_sizes = [-0.10, -0.05, 0.0, 0.05, 0.10]

        all_sensitivities = {}

        for param_name in self.base_params.keys():
            all_sensitivities[param_name] = self.compute_sensitivity(
                param_name, shock_sizes
            )

        return all_sensitivities


def create_historical_scenarios(
    historical_returns: Dict[str, Array],
    lookback_period: int = 252
) -> List[MarketScenario]:
    """Create scenarios from historical market moves.

    Args:
        historical_returns: Historical returns by asset
        lookback_period: Number of historical days to consider

    Returns:
        List of historical scenarios
    """
    scenarios = []

    # Get worst historical days for each asset
    for asset, returns in historical_returns.items():
        # Take last N days
        recent_returns = returns[-lookback_period:]

        # Find worst days
        sorted_returns = jnp.sort(recent_returns)
        worst_returns = sorted_returns[:10]  # Top 10 worst days

        for i, ret in enumerate(worst_returns):
            scenario = MarketScenario(
                name=f"Historical_{asset}_Worst_{i+1}",
                scenario_type=ScenarioType.HISTORICAL,
                spot_shocks={asset: float(ret)},
            )
            scenarios.append(scenario)

    return scenarios


def create_stress_scenarios() -> List[MarketScenario]:
    """Create standard stress test scenarios.

    Returns:
        List of predefined stress scenarios
    """
    scenarios = [
        # Equity crash scenarios
        MarketScenario(
            name="Equity_Crash_20%",
            scenario_type=ScenarioType.STRESS,
            spot_shocks={"equity": -0.20},
            vol_shocks={"equity": 0.15},
        ),
        MarketScenario(
            name="Equity_Crash_30%",
            scenario_type=ScenarioType.STRESS,
            spot_shocks={"equity": -0.30},
            vol_shocks={"equity": 0.25},
        ),
        # Interest rate scenarios
        MarketScenario(
            name="Rates_Up_100bp",
            scenario_type=ScenarioType.STRESS,
            rate_shocks={"USD": 0.01, "EUR": 0.01},
        ),
        MarketScenario(
            name="Rates_Down_50bp",
            scenario_type=ScenarioType.STRESS,
            rate_shocks={"USD": -0.005, "EUR": -0.005},
        ),
        # Volatility scenarios
        MarketScenario(
            name="Vol_Spike_10pts",
            scenario_type=ScenarioType.STRESS,
            vol_shocks={"equity": 0.10, "fx": 0.05},
        ),
        # Credit scenarios
        MarketScenario(
            name="Credit_Widening_100bp",
            scenario_type=ScenarioType.STRESS,
            spread_shocks={"IG": 0.01, "HY": 0.02},
        ),
        # FX scenarios
        MarketScenario(
            name="USD_Strengthen_10%",
            scenario_type=ScenarioType.STRESS,
            fx_shocks={"EURUSD": -0.10, "GBPUSD": -0.10, "USDJPY": 0.10},
        ),
        # Combined crisis scenario
        MarketScenario(
            name="Financial_Crisis",
            scenario_type=ScenarioType.STRESS,
            spot_shocks={"equity": -0.40},
            vol_shocks={"equity": 0.30},
            rate_shocks={"USD": -0.02},
            spread_shocks={"IG": 0.02, "HY": 0.05},
            correlation_adjustments={("equity", "equity"): 0.20},  # Correlations increase in crisis
        ),
    ]

    return scenarios


def monte_carlo_scenarios(
    key: jax.random.KeyArray,
    n_scenarios: int,
    asset_volatilities: Dict[str, float],
    time_horizon: float = 1.0 / 252.0  # 1 day
) -> List[MarketScenario]:
    """Generate Monte Carlo scenarios.

    Args:
        key: JAX random key
        n_scenarios: Number of scenarios to generate
        asset_volatilities: Volatility for each asset
        time_horizon: Time horizon for scenarios (default: 1 day)

    Returns:
        List of Monte Carlo scenarios
    """
    scenarios = []

    assets = list(asset_volatilities.keys())
    n_assets = len(assets)

    # Generate correlated random shocks
    # Assume independent for simplicity; can add correlation
    for i in range(n_scenarios):
        key, subkey = jax.random.split(key)
        normals = jax.random.normal(subkey, (n_assets,))

        spot_shocks = {}
        for j, asset in enumerate(assets):
            vol = asset_volatilities[asset]
            shock = float(normals[j] * vol * jnp.sqrt(time_horizon))
            spot_shocks[asset] = shock

        scenario = MarketScenario(
            name=f"MC_Scenario_{i+1}",
            scenario_type=ScenarioType.MONTE_CARLO,
            spot_shocks=spot_shocks,
            probability=1.0 / n_scenarios,
        )
        scenarios.append(scenario)

    return scenarios
