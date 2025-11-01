import pytest

from neutryx.portfolio.aggregation import (
    aggregate_sensitivities,
    format_sensitivity_report,
)
from neutryx.valuations.scenarios.scenario import Scenario, ScenarioSet, Shock

from tests.fixtures.sample_portfolios import SAMPLE_PORTFOLIO


@pytest.fixture
def aggregated_portfolio():
    return aggregate_sensitivities(SAMPLE_PORTFOLIO)


def test_portfolio_sensitivity_report_regression(aggregated_portfolio):
    report = format_sensitivity_report(aggregated_portfolio)
    expected = """Risk Factor        Measure               Value
----------------------------------------------
EQ_SPX             delta             7250.0000
EQ_SPX             vega               620.0000
FX_EURUSD          delta           -42000.0000
IR_USD_1Y          delta           107500.0000
IR_USD_1Y          vega               750.0000
IR_USD_5Y          delta           -29500.0000"""
    assert report == expected


def test_portfolio_scenario_regression(aggregated_portfolio):
    parallel_rates = Scenario(
        "Parallel USD rates",
        shocks=(Shock("IR_USD_1Y", 0.01), Shock("IR_USD_5Y", 0.005)),
    )
    fx_shock = Scenario("EURUSD depreciation", shocks=(Shock("FX_EURUSD", -0.02),))

    report = ScenarioSet((parallel_rates, fx_shock)).evaluate(aggregated_portfolio)

    totals = report.totals()
    assert totals["Parallel USD rates"] == pytest.approx(927.5)
    assert totals["EURUSD depreciation"] == pytest.approx(840.0)

    contributions = report.results[0].contributions
    assert contributions["IR_USD_1Y"] == pytest.approx(1075.0)
    assert contributions["IR_USD_5Y"] == pytest.approx(-147.5)
