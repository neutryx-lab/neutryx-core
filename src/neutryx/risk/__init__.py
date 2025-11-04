"""Risk analytics interface built on Neutryx valuation components."""

from __future__ import annotations

from .analytics import (
    RiskEngine,
    RiskFactorAttributionMethod,
    compute_cvar,
    compute_expected_shortfall,
    compute_var,
    explain_pnl,
    generate_historical_scenarios,
    generate_monte_carlo_scenarios,
    generate_standard_stress_scenarios,
    risk_factor_attribution,
    run_scenario_analysis,
    run_stress_test,
    run_stress_tests,
    scenario_expected_shortfall,
)
from neutryx.valuations.pnl_attribution import (
    AttributionMethod,
    MarketState,
    PnLAttribution,
)
from neutryx.valuations.risk_metrics import VaRMethod
from neutryx.valuations.scenarios.scenario_engine import (
    MarketScenario,
    ScenarioResult,
    ScenarioType,
)
from neutryx.valuations.stress_test import HISTORICAL_SCENARIOS, StressScenario

__all__ = [
    "RiskEngine",
    "RiskFactorAttributionMethod",
    "AttributionMethod",
    "VaRMethod",
    "MarketScenario",
    "ScenarioResult",
    "ScenarioType",
    "StressScenario",
    "HISTORICAL_SCENARIOS",
    "MarketState",
    "PnLAttribution",
    "compute_var",
    "compute_cvar",
    "compute_expected_shortfall",
    "run_stress_test",
    "run_stress_tests",
    "generate_historical_scenarios",
    "generate_standard_stress_scenarios",
    "generate_monte_carlo_scenarios",
    "run_scenario_analysis",
    "scenario_expected_shortfall",
    "risk_factor_attribution",
    "explain_pnl",
]
