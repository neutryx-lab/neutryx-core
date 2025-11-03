"""Stress testing framework for portfolio and option risk management.

This module provides tools for conducting stress tests on portfolios,
including historical scenarios, hypothetical shocks, and factor-based stress testing.
"""
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from neutryx.core.engine import Array


@dataclass
class StressScenario:
    """Definition of a stress testing scenario.

    Attributes
    ----------
    name : str
        Name of the scenario (e.g., "2008 Financial Crisis")
    description : str
        Description of the scenario
    shocks : Dict[str, float]
        Dictionary mapping risk factors to their shock values
        (e.g., {"equity": -0.40, "volatility": 2.0, "rates": -0.02})
    """

    name: str
    description: str
    shocks: Dict[str, float]


# Pre-defined historical stress scenarios
HISTORICAL_SCENARIOS = {
    "black_monday_1987": StressScenario(
        name="Black Monday 1987",
        description="October 19, 1987 stock market crash",
        shocks={"equity": -0.227, "volatility": 2.5, "rates": -0.005},
    ),
    "financial_crisis_2008": StressScenario(
        name="Financial Crisis 2008",
        description="Lehman Brothers collapse and subsequent crisis",
        shocks={"equity": -0.45, "volatility": 3.0, "rates": -0.03, "credit_spread": 0.05},
    ),
    "flash_crash_2010": StressScenario(
        name="Flash Crash 2010",
        description="May 6, 2010 flash crash",
        shocks={"equity": -0.09, "volatility": 1.8},
    ),
    "covid_crash_2020": StressScenario(
        name="COVID-19 Crash 2020",
        description="March 2020 pandemic-induced market crash",
        shocks={"equity": -0.34, "volatility": 4.0, "rates": -0.015},
    ),
    "rate_shock_up": StressScenario(
        name="Interest Rate Shock (Up)",
        description="Sudden 200bp increase in interest rates",
        shocks={"rates": 0.02, "equity": -0.10, "volatility": 0.5},
    ),
    "rate_shock_down": StressScenario(
        name="Interest Rate Shock (Down)",
        description="Sudden 200bp decrease in interest rates",
        shocks={"rates": -0.02, "equity": 0.05, "volatility": 0.3},
    ),
    "volatility_spike": StressScenario(
        name="Volatility Spike",
        description="Sudden doubling of implied volatility",
        shocks={"volatility": 1.0, "equity": -0.15},
    ),
}


def apply_shock_to_parameters(
    base_params: Dict[str, float],
    shock: Dict[str, float],
    shock_type: str = "absolute",
) -> Dict[str, float]:
    """Apply stress shock to base parameters.

    Parameters
    ----------
    base_params : dict
        Base parameter values
    shock : dict
        Shock values to apply
    shock_type : str
        Type of shock: "absolute" (additive) or "relative" (multiplicative)

    Returns
    -------
    dict
        Stressed parameter values
    """
    stressed_params = base_params.copy()

    for param, shock_value in shock.items():
        if param not in base_params:
            continue

        if shock_type == "absolute":
            stressed_params[param] = base_params[param] + shock_value
        elif shock_type == "relative":
            stressed_params[param] = base_params[param] * (1 + shock_value)
        else:
            raise ValueError(f"Unknown shock type: {shock_type}")

    return stressed_params


def run_stress_scenario(
    scenario: StressScenario,
    base_params: Dict[str, float],
    valuation_fn: Callable,
    shock_type: str = "relative",
) -> Dict[str, float]:
    """Run a single stress scenario.

    Parameters
    ----------
    scenario : StressScenario
        Stress scenario definition
    base_params : dict
        Base parameter values
    valuation_fn : callable
        Function that takes parameters and returns valuation
    shock_type : str
        Type of shock application

    Returns
    -------
    dict
        Results including base value, stressed value, and P&L
    """
    # Compute base valuation
    base_value = valuation_fn(**base_params)

    # Apply shocks
    stressed_params = apply_shock_to_parameters(base_params, scenario.shocks, shock_type)

    # Compute stressed valuation
    stressed_value = valuation_fn(**stressed_params)

    # Compute P&L
    pnl = stressed_value - base_value
    pnl_pct = (pnl / base_value) * 100 if base_value != 0 else 0.0

    return {
        "scenario_name": scenario.name,
        "base_value": float(base_value),
        "stressed_value": float(stressed_value),
        "pnl": float(pnl),
        "pnl_percent": float(pnl_pct),
        "shocks_applied": scenario.shocks,
    }


def run_multiple_stress_tests(
    scenarios: List[StressScenario],
    base_params: Dict[str, float],
    valuation_fn: Callable,
    shock_type: str = "relative",
) -> List[Dict]:
    """Run multiple stress scenarios.

    Parameters
    ----------
    scenarios : list
        List of stress scenarios
    base_params : dict
        Base parameter values
    valuation_fn : callable
        Valuation function
    shock_type : str
        Type of shock application

    Returns
    -------
    list
        List of results for each scenario
    """
    results = []

    for scenario in scenarios:
        result = run_stress_scenario(scenario, base_params, valuation_fn, shock_type)
        results.append(result)

    return results


def run_historical_stress_tests(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    scenario_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Run pre-defined historical stress scenarios.

    Parameters
    ----------
    base_params : dict
        Base parameter values
    valuation_fn : callable
        Valuation function
    scenario_names : list, optional
        List of scenario names to run (if None, runs all)

    Returns
    -------
    list
        List of results for each scenario
    """
    if scenario_names is None:
        scenarios = list(HISTORICAL_SCENARIOS.values())
    else:
        scenarios = [HISTORICAL_SCENARIOS[name] for name in scenario_names]

    return run_multiple_stress_tests(scenarios, base_params, valuation_fn)


def factor_stress_test(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    factor: str,
    shock_range: Array,
) -> Array:
    """Run stress test across a range of shocks for a single factor.

    Useful for generating stress test P&L profiles and identifying
    concentration risk.

    Parameters
    ----------
    base_params : dict
        Base parameter values
    valuation_fn : callable
        Valuation function
    factor : str
        Name of the factor to stress (must be in base_params)
    shock_range : Array
        Array of shock values to apply (relative shocks)

    Returns
    -------
    Array
        P&L for each shock value
    """
    if factor not in base_params:
        raise ValueError(f"Factor {factor} not found in base_params")

    base_value = valuation_fn(**base_params)

    pnls = []
    for shock in shock_range:
        stressed_params = base_params.copy()
        stressed_params[factor] = base_params[factor] * (1 + shock)

        stressed_value = valuation_fn(**stressed_params)
        pnl = stressed_value - base_value
        pnls.append(pnl)

    return jnp.array(pnls)


def reverse_stress_test(
    base_params: Dict[str, float],
    valuation_fn: Callable,
    factor: str,
    target_loss: float,
    search_range: tuple = (-0.99, 10.0),
    tolerance: float = 1e-4,
) -> float:
    """Find the shock level that produces a target loss (reverse stress test).

    Parameters
    ----------
    base_params : dict
        Base parameter values
    valuation_fn : callable
        Valuation function
    factor : str
        Factor to stress
    target_loss : float
        Target loss amount (negative for losses)
    search_range : tuple
        Range to search for shock (min_shock, max_shock)
    tolerance : float
        Tolerance for convergence

    Returns
    -------
    float
        Shock level that produces target loss
    """
    base_value = valuation_fn(**base_params)
    target_value = base_value + target_loss

    def objective(shock):
        stressed_params = base_params.copy()
        stressed_params[factor] = base_params[factor] * (1 + shock)
        stressed_value = valuation_fn(**stressed_params)
        return stressed_value - target_value

    # Binary search
    low, high = search_range
    while high - low > tolerance:
        mid = (low + high) / 2
        if objective(mid) < 0:
            low = mid
        else:
            high = mid

    return (low + high) / 2


__all__ = [
    "StressScenario",
    "HISTORICAL_SCENARIOS",
    "apply_shock_to_parameters",
    "run_stress_scenario",
    "run_multiple_stress_tests",
    "run_historical_stress_tests",
    "factor_stress_test",
    "reverse_stress_test",
]
