"""Tests for the high-level risk toolkit facade."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from neutryx.valuations.risk import (
    MarketScenario,
    MarketState,
    RiskEngine,
    RiskFactorAttributionMethod,
    ScenarioType,
    compute_cvar,
    compute_expected_shortfall,
    compute_var,
    explain_pnl,
    risk_factor_attribution,
    run_scenario_analysis,
    run_stress_test,
    run_stress_tests,
    scenario_expected_shortfall,
)


def test_var_cvar_expected_shortfall_basic():
    """VaR/CVaR/ES helpers should agree with core implementations."""
    returns = jnp.array([-0.05, -0.02, -0.01, 0.0, 0.01, 0.03, 0.05])

    var_95 = compute_var(returns, 0.95, method="historical")
    cvar_95 = compute_cvar(returns, 0.95)
    es_95 = compute_expected_shortfall(returns, 0.95)

    assert isinstance(var_95, float)
    assert isinstance(cvar_95, float)
    assert math.isclose(cvar_95, es_95)
    assert cvar_95 >= var_95


def test_stress_testing_apis():
    """Stress testing helpers should execute scenarios and report P&L."""
    base_params = {"equity": 1.0, "volatility": 0.2, "rates": 0.01}

    def valuation_fn(equity=0.0, volatility=0.0, rates=0.0):
        return 100.0 * (1.0 + equity) - 5.0 * volatility - 500.0 * rates

    crisis = run_stress_test("financial_crisis_2008", base_params, valuation_fn)
    assert crisis["scenario_name"] == "Financial Crisis 2008"
    assert crisis["pnl"] < 0.0

    batch = run_stress_tests(
        base_params,
        valuation_fn,
        scenarios=["financial_crisis_2008", "rate_shock_up"],
    )
    assert len(batch) == 2
    assert batch[0]["scenario_name"]


def test_scenario_analysis_and_es():
    """Scenario analysis wrappers should return results and expected shortfall."""
    base_market_data = {"spot_EQ": 100.0}

    def simple_pricer(market_data):
        return market_data["spot_EQ"]

    scenarios = [
        MarketScenario(
            name="Down10",
            scenario_type=ScenarioType.HYPOTHETICAL,
            spot_shocks={"EQ": -0.10},
        ),
        MarketScenario(
            name="Up5",
            scenario_type=ScenarioType.HYPOTHETICAL,
            spot_shocks={"EQ": 0.05},
        ),
    ]

    results = run_scenario_analysis(simple_pricer, base_market_data, scenarios)
    assert len(results) == 2
    down = next(r for r in results if r.scenario_name == "Down10")
    assert pytest.approx(down.pnl, rel=1e-6) == -10.0

    es = scenario_expected_shortfall(simple_pricer, base_market_data, scenarios, 0.95)
    assert es <= 0.0

    engine = RiskEngine(simple_pricer, base_market_data)
    engine.add_scenarios(scenarios)
    engine_results = engine.analyze_scenarios()
    assert len(engine_results) == 2
    assert pytest.approx(engine.scenario_expected_shortfall(0.95), rel=1e-6) == es


def test_risk_factor_attribution_methods():
    """Risk factor attribution facade should cover component/marginal/incremental VaR."""
    positions = jnp.array([100.0, 50.0])
    returns = jnp.array(
        [
            [0.01, 0.005],
            [-0.02, -0.01],
            [0.015, 0.007],
            [-0.03, -0.015],
            [0.005, 0.002],
        ]
    )

    component = risk_factor_attribution(
        positions, returns, confidence_level=0.95, method=RiskFactorAttributionMethod.COMPONENT
    )
    assert component.shape == (2,)

    total_var = compute_var(jnp.dot(returns, positions), 0.95)
    assert pytest.approx(float(jnp.sum(component)), rel=1e-6) == total_var

    marginal = risk_factor_attribution(
        positions, returns, confidence_level=0.95, method="marginal", delta=0.05
    )
    assert marginal.shape == (2,)

    incremental = risk_factor_attribution(
        positions,
        returns,
        confidence_level=0.95,
        method="incremental",
        target_index=1,
    )
    assert isinstance(incremental, float)


def test_pnl_explain_helpers():
    """PnL explain helper should reconcile PnL using provided Greeks."""
    base_market_data = {"spot_EQ": 100.0}

    def dict_pricer(market_data):
        return market_data["spot_EQ"]

    def greeks_calc(state: MarketState):
        return {
            "theta": 0.0,
            "delta_EQ": 1.0,
            "gamma_EQ": 0.0,
        }

    start_state = MarketState(timestamp=0.0, spot_prices={"EQ": 100.0})
    end_state = MarketState(timestamp=1.0, spot_prices={"EQ": 110.0})

    engine = RiskEngine(dict_pricer, base_market_data, greeks_calculator=greeks_calc)
    attribution = engine.explain_pnl(start_state, end_state)

    assert pytest.approx(attribution.total_pnl, rel=1e-6) == 10.0
    assert pytest.approx(attribution.total_spot_pnl(), rel=1e-6) == 10.0

    # The functional interface should provide the same result.
    def state_pricer(state: MarketState) -> float:
        return dict_pricer({"spot_EQ": state.spot_prices["EQ"]})

    attribution_direct = explain_pnl(
        start_state,
        end_state,
        state_pricer,
        greeks_calculator=greeks_calc,
        method="greeks",
    )

    assert pytest.approx(attribution_direct.total_pnl, rel=1e-6) == 10.0
