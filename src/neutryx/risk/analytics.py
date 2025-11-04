"""High-level risk analytics toolkit built on valuation components.

This module provides a lightweight facade that bundles together the existing
valuation primitives (risk metrics, stress testing, scenario analysis, and
P&L attribution) into a cohesive API for risk management workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import jax.numpy as jnp
from jax import Array

from neutryx.valuations.pnl_attribution import (
    AttributionMethod,
    MarketState,
    PnLAttribution,
    PnLAttributionEngine,
)
from neutryx.valuations.risk_metrics import (
    VaRMethod,
    calculate_var,
    component_var,
    conditional_value_at_risk,
    expected_shortfall,
    incremental_var,
    marginal_var,
)
from neutryx.valuations.scenarios.scenario_engine import (
    MarketScenario,
    ScenarioEngine,
    ScenarioResult,
    create_historical_scenarios,
    create_stress_scenarios,
    monte_carlo_scenarios,
)
from neutryx.valuations.stress_test import (
    HISTORICAL_SCENARIOS,
    StressScenario,
    run_multiple_stress_tests,
    run_stress_scenario,
)


class RiskFactorAttributionMethod(str, Enum):
    """Supported VaR-based risk factor attribution methods."""

    COMPONENT = "component"
    MARGINAL = "marginal"
    INCREMENTAL = "incremental"


def _coerce_var_method(method: VaRMethod | str) -> VaRMethod:
    """Normalize VaR method input."""
    if isinstance(method, VaRMethod):
        return method
    try:
        return VaRMethod(method)
    except ValueError as err:
        # Try enum member names (case-insensitive)
        normalized = str(method).upper()
        if normalized in VaRMethod.__members__:
            return VaRMethod[normalized]
        # Fall back to matching enum values ignoring case
        for candidate in VaRMethod:
            if candidate.value.lower() == str(method).lower():
                return candidate
        raise ValueError(f"Unsupported VaR method '{method}'") from err


def compute_var(
    returns: Array,
    confidence_level: float = 0.95,
    method: VaRMethod | str = VaRMethod.HISTORICAL,
    **kwargs,
) -> float:
    """Compute Value at Risk using the selected methodology."""
    var_method = _coerce_var_method(method)
    return float(calculate_var(returns, confidence_level, var_method, **kwargs))


def compute_cvar(returns: Array, confidence_level: float = 0.95) -> float:
    """Compute Conditional VaR (a.k.a. Expected Shortfall)."""
    return float(conditional_value_at_risk(returns, confidence_level))


def compute_expected_shortfall(returns: Array, confidence_level: float = 0.95) -> float:
    """Alias for Expected Shortfall (identical to CVaR)."""
    return float(expected_shortfall(returns, confidence_level))


def run_stress_test(
    scenario: StressScenario | str,
    base_params: Dict[str, float],
    valuation_fn: Callable[..., float],
    *,
    shock_type: str = "relative",
) -> Dict[str, float]:
    """Run a single stress scenario and return the P&L breakdown."""
    stress = (
        scenario
        if isinstance(scenario, StressScenario)
        else _resolve_stress_scenario(scenario)
    )
    return run_stress_scenario(stress, base_params, valuation_fn, shock_type)


def run_stress_tests(
    base_params: Dict[str, float],
    valuation_fn: Callable[..., float],
    scenarios: Optional[Sequence[StressScenario | str]] = None,
    *,
    shock_type: str = "relative",
) -> List[Dict[str, float]]:
    """Run multiple stress scenarios."""
    if scenarios is None:
        stress_scenarios: List[StressScenario] = list(HISTORICAL_SCENARIOS.values())
    else:
        stress_scenarios = [
            _resolve_stress_scenario(s) if not isinstance(s, StressScenario) else s
            for s in scenarios
        ]

    return run_multiple_stress_tests(stress_scenarios, base_params, valuation_fn, shock_type)


def _resolve_stress_scenario(name: str) -> StressScenario:
    """Resolve a named stress scenario from the registry."""
    try:
        return HISTORICAL_SCENARIOS[name]
    except KeyError as err:
        raise ValueError(f"Unknown stress scenario '{name}'") from err


def generate_historical_scenarios(
    historical_returns: Dict[str, Array],
    lookback_period: int = 252,
) -> List[MarketScenario]:
    """Create scenarios from historical return data."""
    return create_historical_scenarios(historical_returns, lookback_period)


def generate_standard_stress_scenarios() -> List[MarketScenario]:
    """Return the library of standard stress scenarios."""
    return create_stress_scenarios()


def generate_monte_carlo_scenarios(
    key,
    n_scenarios: int,
    asset_volatilities: Dict[str, float],
    *,
    time_horizon: float = 1.0 / 252.0,
) -> List[MarketScenario]:
    """Generate Monte Carlo scenarios for the supplied assets."""
    return monte_carlo_scenarios(key, n_scenarios, asset_volatilities, time_horizon)


def run_scenario_analysis(
    portfolio_pricer: Callable[[Dict[str, float]], float],
    base_market_data: Dict[str, float],
    scenarios: Sequence[MarketScenario],
) -> List[ScenarioResult]:
    """Run scenario analysis for the provided market scenarios."""
    engine = ScenarioEngine(portfolio_pricer, base_market_data)
    for scenario in scenarios:
        engine.add_scenario(scenario)
    return engine.run_all_scenarios()


def scenario_expected_shortfall(
    portfolio_pricer: Callable[[Dict[str, float]], float],
    base_market_data: Dict[str, float],
    scenarios: Sequence[MarketScenario],
    confidence_level: float = 0.95,
) -> float:
    """Compute Expected Shortfall from scenario analysis results."""
    engine = ScenarioEngine(portfolio_pricer, base_market_data)
    for scenario in scenarios:
        engine.add_scenario(scenario)
    return float(engine.expected_shortfall(confidence_level))


def risk_factor_attribution(
    positions: Array,
    returns_scenarios: Array,
    *,
    confidence_level: float = 0.95,
    method: RiskFactorAttributionMethod | str = RiskFactorAttributionMethod.COMPONENT,
    target_index: Optional[int] = None,
    delta: float = 0.01,
) -> Array | float:
    """Decompose portfolio VaR into risk factor contributions."""
    method_enum = (
        method
        if isinstance(method, RiskFactorAttributionMethod)
        else RiskFactorAttributionMethod(method)
    )

    positions = jnp.asarray(positions)
    returns_scenarios = jnp.asarray(returns_scenarios)

    if method_enum == RiskFactorAttributionMethod.COMPONENT:
        return component_var(positions, returns_scenarios, confidence_level)

    if method_enum == RiskFactorAttributionMethod.MARGINAL:
        return marginal_var(positions, returns_scenarios, confidence_level, delta)

    if target_index is None:
        raise ValueError("target_index must be provided for incremental VaR attribution")

    portfolio_returns = jnp.dot(returns_scenarios, positions)
    position_returns = returns_scenarios[:, target_index] * positions[target_index]
    return float(incremental_var(portfolio_returns, position_returns, confidence_level))


def explain_pnl(
    start_state: MarketState,
    end_state: MarketState,
    portfolio_pricer: Callable[[MarketState], float],
    *,
    greeks_calculator: Optional[Callable[[MarketState], Dict[str, float]]] = None,
    method: AttributionMethod | str | None = None,
    start_portfolio_value: Optional[float] = None,
) -> PnLAttribution:
    """Explain P&L between two market states."""
    attribution_method = _coerce_attribution_method(method, greeks_calculator)
    engine = PnLAttributionEngine(
        portfolio_pricer,
        greeks_calculator,
        attribution_method,
    )
    return engine.attribute_pnl(start_state, end_state, start_portfolio_value)


def _coerce_attribution_method(
    method: AttributionMethod | str | None,
    greeks_calculator: Optional[Callable[[MarketState], Dict[str, float]]],
) -> AttributionMethod:
    """Determine the attribution method to use."""
    if method is None:
        return (
            AttributionMethod.HYBRID
            if greeks_calculator is not None
            else AttributionMethod.REVALUATION
        )

    if isinstance(method, AttributionMethod):
        if method == AttributionMethod.HYBRID and greeks_calculator is None:
            raise ValueError("HYBRID attribution requires a greeks_calculator.")
        return method

    coerced = AttributionMethod(method)
    if coerced == AttributionMethod.HYBRID and greeks_calculator is None:
        raise ValueError("HYBRID attribution requires a greeks_calculator.")
    return coerced


@dataclass
class RiskEngine:
    """Convenience wrapper that wires together risk analytics workflows."""

    portfolio_pricer: Callable[[Dict[str, float]], float]
    base_market_data: Dict[str, float]
    greeks_calculator: Optional[Callable[[MarketState], Dict[str, float]]] = None

    def __post_init__(self):
        self._scenario_engine = ScenarioEngine(self.portfolio_pricer, self.base_market_data)

    # ---- VaR/CVaR utilities -------------------------------------------------
    def var(
        self,
        returns: Array,
        confidence_level: float = 0.95,
        method: VaRMethod | str = VaRMethod.HISTORICAL,
        **kwargs,
    ) -> float:
        """Value at Risk helper."""
        return compute_var(returns, confidence_level, method, **kwargs)

    def cvar(self, returns: Array, confidence_level: float = 0.95) -> float:
        """Conditional Value at Risk helper."""
        return compute_cvar(returns, confidence_level)

    def expected_shortfall(self, returns: Array, confidence_level: float = 0.95) -> float:
        """Expected Shortfall helper."""
        return compute_expected_shortfall(returns, confidence_level)

    # ---- Scenario analysis --------------------------------------------------
    @property
    def scenario_engine(self) -> ScenarioEngine:
        """Access the underlying scenario engine."""
        return self._scenario_engine

    def add_scenarios(self, scenarios: Iterable[MarketScenario]):
        """Register scenarios for subsequent analysis."""
        for scenario in scenarios:
            self._scenario_engine.add_scenario(scenario)

    def analyze_scenarios(self) -> List[ScenarioResult]:
        """Run all registered scenarios."""
        return self._scenario_engine.run_all_scenarios()

    def scenario_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Expected Shortfall based on registered scenarios."""
        return float(self._scenario_engine.expected_shortfall(confidence_level))

    def generate_and_add_mc_scenarios(
        self,
        key,
        n_scenarios: int,
        asset_volatilities: Dict[str, float],
        *,
        time_horizon: float = 1.0 / 252.0,
    ) -> List[MarketScenario]:
        """Generate Monte Carlo scenarios and register them."""
        scenarios = generate_monte_carlo_scenarios(
            key,
            n_scenarios,
            asset_volatilities,
            time_horizon=time_horizon,
        )
        self.add_scenarios(scenarios)
        return scenarios

    # ---- Stress testing -----------------------------------------------------
    def run_stress_tests(
        self,
        base_params: Dict[str, float],
        valuation_fn: Callable[..., float],
        scenarios: Optional[Sequence[StressScenario | str]] = None,
        *,
        shock_type: str = "relative",
    ) -> List[Dict[str, float]]:
        """Run stress tests via the helper function."""
        return run_stress_tests(base_params, valuation_fn, scenarios, shock_type=shock_type)

    # ---- Risk factor attribution -------------------------------------------
    def attribute_risk(
        self,
        positions: Array,
        returns_scenarios: Array,
        *,
        confidence_level: float = 0.95,
        method: RiskFactorAttributionMethod | str = RiskFactorAttributionMethod.COMPONENT,
        target_index: Optional[int] = None,
        delta: float = 0.01,
    ) -> Array | float:
        """Perform risk factor attribution."""
        return risk_factor_attribution(
            positions,
            returns_scenarios,
            confidence_level=confidence_level,
            method=method,
            target_index=target_index,
            delta=delta,
        )

    # ---- P&L explain --------------------------------------------------------
    def explain_pnl(
        self,
        start_state: MarketState,
        end_state: MarketState,
        *,
        greeks_calculator: Optional[Callable[[MarketState], Dict[str, float]]] = None,
        method: AttributionMethod | str | None = None,
        start_portfolio_value: Optional[float] = None,
    ) -> PnLAttribution:
        """Explain P&L using the stored portfolio pricer."""
        calculator = greeks_calculator or self.greeks_calculator
        return explain_pnl(
            start_state,
            end_state,
            self._wrap_market_pricer(),
            greeks_calculator=calculator,
            method=method,
            start_portfolio_value=start_portfolio_value,
        )

    def _wrap_market_pricer(self) -> Callable[[MarketState], float]:
        """Adapt the stored portfolio pricer to accept MarketState objects."""
        def pricer_from_state(state: MarketState) -> float:
            market_data = self.base_market_data.copy()
            market_data.update(state.get_all_factors())
            return self.portfolio_pricer(market_data)

        return pricer_from_state
