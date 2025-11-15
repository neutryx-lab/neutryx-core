"""Tests for P&L attribution framework."""

import jax.numpy as jnp
import pytest

from neutryx.valuations.attribution.pnl_attribution import (
    DailyPLExplain,
    GreekPLCalculator,
    ModelRiskCalculator,
    ModelRiskMetrics,
    PLComponent,
    RiskFactorAttribution,
    RiskFactorPLCalculator,
)


# ===== PLComponent Tests =====


def test_pl_component_creation():
    """Test creating P&L component."""
    component = PLComponent(
        name="delta",
        value=1000.0,
        risk_factor="AAPL",
        description="Delta P&L from equity moves",
    )

    assert component.name == "delta"
    assert component.value == 1000.0
    assert component.risk_factor == "AAPL"
    assert "Delta P&L" in component.description


# ===== GreekPLCalculator Tests =====


def test_delta_pnl():
    """Test delta P&L calculation."""
    calc = GreekPLCalculator()

    delta = 1000.0  # Delta of portfolio
    price_change = 2.5  # $2.5 move in underlying

    delta_pnl = calc.compute_delta_pnl(delta, price_change)

    # Delta P&L = Delta × ΔS = 1000 × 2.5 = 2500
    assert delta_pnl == pytest.approx(2500.0, abs=1e-6)


def test_gamma_pnl():
    """Test gamma P&L calculation."""
    calc = GreekPLCalculator()

    gamma = 50.0  # Gamma of portfolio
    price_change = 3.0  # $3 move in underlying

    gamma_pnl = calc.compute_gamma_pnl(gamma, price_change)

    # Gamma P&L = 0.5 × Gamma × (ΔS)² = 0.5 × 50 × 9 = 225
    assert gamma_pnl == pytest.approx(225.0, abs=1e-6)


def test_vega_pnl():
    """Test vega P&L calculation."""
    calc = GreekPLCalculator()

    vega = 2000.0  # Vega (per 1% vol move)
    vol_change = 0.02  # 2% increase in volatility

    vega_pnl = calc.compute_vega_pnl(vega, vol_change)

    # Vega P&L = Vega × Δσ = 2000 × 0.02 = 40
    assert vega_pnl == pytest.approx(40.0, abs=1e-6)


def test_theta_pnl():
    """Test theta P&L calculation."""
    calc = GreekPLCalculator()

    theta = -50.0  # Theta (daily time decay)
    time_elapsed = 1.0  # 1 day elapsed

    theta_pnl = calc.compute_theta_pnl(theta, time_elapsed)

    # Theta P&L = Theta × Δt = -50 × 1 = -50
    assert theta_pnl == pytest.approx(-50.0, abs=1e-6)


def test_rho_pnl():
    """Test rho P&L calculation."""
    calc = GreekPLCalculator()

    rho = 500.0  # Rho (per 1% rate move)
    rate_change = 0.0025  # 25bp rate increase

    rho_pnl = calc.compute_rho_pnl(rho, rate_change)

    # Rho P&L = Rho × Δr = 500 × 0.0025 = 1.25
    assert rho_pnl == pytest.approx(1.25, abs=1e-6)


def test_carry_pnl():
    """Test carry P&L calculation."""
    calc = GreekPLCalculator()

    position_value_t0 = 1_000_000.0
    position_value_t1 = 1_000_100.0  # Accrued 100
    funding_cost = 50.0

    carry_pnl = calc.compute_carry_pnl(position_value_t0, position_value_t1, funding_cost)

    # Carry = (t1 - t0) - funding = 100 - 50 = 50
    assert carry_pnl == pytest.approx(50.0, abs=1e-6)


def test_daily_pnl_explain_integration():
    """Test complete daily P&L explain calculation."""
    calc = GreekPLCalculator()

    explain = calc.compute_daily_pnl_explain(
        date="2024-01-15",
        total_pnl=3000.0,
        delta=1000.0,
        gamma=50.0,
        vega=2000.0,
        theta=-50.0,
        rho=500.0,
        price_change=2.5,
        vol_change=0.02,
        rate_change=0.0025,
        carry_pnl=100.0,
    )

    # Check that explain has all components
    assert explain.date == "2024-01-15"
    assert explain.total_pnl == 3000.0
    assert explain.carry_pnl == 100.0
    assert explain.delta_pnl == pytest.approx(2500.0, abs=1e-6)  # 1000 × 2.5
    assert explain.gamma_pnl == pytest.approx(156.25, abs=1e-6)  # 0.5 × 50 × 6.25
    assert explain.vega_pnl == pytest.approx(40.0, abs=1e-6)  # 2000 × 0.02
    assert explain.theta_pnl == pytest.approx(-50.0, abs=1e-6)  # -50 × 1
    assert explain.rho_pnl == pytest.approx(1.25, abs=1e-6)  # 500 × 0.0025

    # Check explanation ratio
    assert explain.passes_attribution_test(threshold=0.80)
    assert len(explain.components) == 7


def test_daily_pl_explain_explanation_ratio():
    """Test explanation ratio calculation."""
    calc = GreekPLCalculator()

    explain = calc.compute_daily_pnl_explain(
        date="2024-01-15",
        total_pnl=10000.0,
        delta=800.0,
        gamma=30.0,
        vega=2000.0,
        theta=-50.0,
        rho=0.0,
        price_change=10.0,
        vol_change=0.0,
        rate_change=0.0,
        carry_pnl=200.0,
    )

    # Delta: 800 × 10 = 8000
    # Gamma: 0.5 × 30 × 100 = 1500
    # Carry: 200
    # Total explained: 9700
    # Ratio: 9700/10000 = 0.97

    assert explain.explanation_ratio == pytest.approx(0.97, abs=0.01)
    assert explain.passes_attribution_test(threshold=0.9)


def test_daily_pl_explain_basel_test():
    """Test Basel P&L attribution test."""
    calc = GreekPLCalculator()

    # Good explanation (>90%)
    good_explain = calc.compute_daily_pnl_explain(
        date="2024-01-15",
        total_pnl=10000.0,
        delta=900.0,
        gamma=50.0,
        vega=2000.0,
        theta=-50.0,
        rho=0.0,
        price_change=10.0,
        vol_change=0.0,
        rate_change=0.0,
        carry_pnl=100.0,
    )

    assert good_explain.passes_attribution_test(threshold=0.9)

    # Poor explanation (<90%)
    poor_explain = calc.compute_daily_pnl_explain(
        date="2024-01-15",
        total_pnl=10000.0,
        delta=500.0,
        gamma=20.0,
        vega=1000.0,
        theta=-50.0,
        rho=0.0,
        price_change=10.0,
        vol_change=0.0,
        rate_change=0.0,
        carry_pnl=0.0,
    )

    # Delta: 5000, Gamma: 1000 = 6000/10000 = 60%
    assert not poor_explain.passes_attribution_test(threshold=0.9)


# ===== RiskFactorPLCalculator Tests =====


def test_ir_attribution():
    """Test interest rate risk attribution."""
    calc = RiskFactorPLCalculator()

    rate_sensitivities = {
        "USD.3M": 1000.0,
        "USD.1Y": 5000.0,
        "USD.5Y": 10000.0,
        "USD.10Y": 8000.0,
    }

    rate_changes = {
        "USD.3M": 0.0005,  # 5bp
        "USD.1Y": 0.0010,  # 10bp
        "USD.5Y": 0.0015,  # 15bp
        "USD.10Y": 0.0008,  # 8bp
    }

    attributions = calc.compute_ir_attribution(rate_sensitivities, rate_changes)

    assert len(attributions) == 4

    # Check total IR P&L
    total_ir = sum(attr.delta_pnl for attr in attributions)

    # Expected = 1000×0.0005 + 5000×0.0010 + 10000×0.0015 + 8000×0.0008
    #          = 0.5 + 5 + 15 + 6.4 = 26.9
    expected = 26.9
    assert total_ir == pytest.approx(expected, abs=1e-6)

    # Check first attribution
    assert attributions[0].risk_factor == "USD.3M"
    assert attributions[0].asset_class == "IR"
    assert attributions[0].delta_pnl == pytest.approx(0.5, abs=1e-6)


def test_credit_attribution():
    """Test credit spread risk attribution."""
    calc = RiskFactorPLCalculator()

    credit_sensitivities = {
        "CORP.AAA": 2000.0,
        "CORP.AA": 3000.0,
        "CORP.A": 5000.0,
        "CORP.BBB": 4000.0,
    }

    spread_changes = {
        "CORP.AAA": 0.0002,  # 2bp widening
        "CORP.AA": 0.0005,  # 5bp widening
        "CORP.A": -0.0003,  # 3bp tightening
        "CORP.BBB": 0.0010,  # 10bp widening
    }

    attributions = calc.compute_credit_attribution(credit_sensitivities, spread_changes)

    assert len(attributions) == 4

    total_credit = sum(attr.delta_pnl for attr in attributions)

    # Expected = 2000×0.0002 + 3000×0.0005 + 5000×(-0.0003) + 4000×0.0010
    #          = 0.4 + 1.5 - 1.5 + 4 = 4.4
    expected = 4.4
    assert total_credit == pytest.approx(expected, abs=1e-6)


def test_fx_attribution():
    """Test FX risk attribution."""
    calc = RiskFactorPLCalculator()

    fx_deltas = {
        "EURUSD": 10000.0,
        "GBPUSD": 5000.0,
        "USDJPY": -8000.0,
    }

    fx_changes = {
        "EURUSD": 0.02,  # 2% EUR appreciation
        "GBPUSD": -0.01,  # 1% GBP depreciation
        "USDJPY": 0.015,  # 1.5% JPY depreciation
    }

    attributions = calc.compute_fx_attribution(fx_deltas, fx_changes)

    assert len(attributions) == 3

    total_fx = sum(attr.delta_pnl for attr in attributions)

    # Expected = 10000×0.02 + 5000×(-0.01) + (-8000)×0.015
    #          = 200 - 50 - 120 = 30
    expected = 30.0
    assert total_fx == pytest.approx(expected, abs=1e-6)


def test_equity_attribution():
    """Test equity risk attribution with gamma."""
    calc = RiskFactorPLCalculator()

    equity_deltas = {
        "AAPL": 1000.0,
        "MSFT": 800.0,
        "GOOGL": 500.0,
    }

    equity_gammas = {
        "AAPL": 50.0,
        "MSFT": 30.0,
        "GOOGL": 20.0,
    }

    equity_changes = {
        "AAPL": 5.0,  # $5 move
        "MSFT": -3.0,  # $3 move down
        "GOOGL": 2.0,  # $2 move
    }

    attributions = calc.compute_equity_attribution(
        equity_deltas, equity_changes, equity_gammas
    )

    assert len(attributions) == 3

    # Delta P&L = 1000×5 + 800×(-3) + 500×2 = 5000 - 2400 + 1000 = 3600
    # Gamma P&L = 0.5×(50×25 + 30×9 + 20×4) = 0.5×(1250 + 270 + 80) = 800
    # Total = 4400

    total_delta = sum(attr.delta_pnl for attr in attributions)
    total_gamma = sum(attr.gamma_pnl for attr in attributions)
    total_eq = sum(attr.total_pnl for attr in attributions)

    assert total_delta == pytest.approx(3600.0, abs=1e-6)
    assert total_gamma == pytest.approx(800.0, abs=1e-6)
    assert total_eq == pytest.approx(4400.0, abs=1e-6)


# ===== ModelRiskCalculator Tests =====


def test_mark_to_model_reserve():
    """Test mark-to-model reserve calculation."""
    calc = ModelRiskCalculator()

    model_price = 102.5
    market_price = 100.0
    bid_ask_spread = 0.5

    reserve = calc.compute_mark_to_model_reserve(
        model_price, market_price, bid_ask_spread
    )

    # Reserve should be based on model-market difference
    assert reserve == pytest.approx(2.5, abs=1e-6)


def test_mark_to_model_reserve_no_market():
    """Test mark-to-model reserve without market price."""
    calc = ModelRiskCalculator()

    model_price = 102.5
    bid_ask_spread = 1.0

    reserve = calc.compute_mark_to_model_reserve(
        model_price, market_price=None, bid_ask_spread=bid_ask_spread
    )

    # Reserve should be half of bid-ask spread
    assert reserve == pytest.approx(0.5, abs=1e-6)


def test_parameter_uncertainty():
    """Test parameter uncertainty quantification."""
    calc = ModelRiskCalculator()

    parameter_std = jnp.array([0.02, 0.01])  # 2% and 1% std
    sensitivity_to_param = jnp.array([50000.0, 30000.0])  # Sensitivities

    uncertainty = calc.compute_parameter_uncertainty(parameter_std, sensitivity_to_param)

    # Uncertainty = sqrt((50000×0.02)² + (30000×0.01)²)
    #             = sqrt(1000² + 300²) = sqrt(1090000) ≈ 1044.03
    expected = jnp.sqrt((50000 * 0.02) ** 2 + (30000 * 0.01) ** 2)
    assert uncertainty == pytest.approx(float(expected), abs=0.1)


def test_model_replacement_impact():
    """Test model replacement impact."""
    calc = ModelRiskCalculator()

    current_model = 100.0
    alternative_model = 102.5

    impact = calc.compute_model_replacement_impact(current_model, alternative_model)

    # Impact = alternative - current = 2.5
    assert impact == pytest.approx(2.5, abs=1e-6)


def test_unexplained_pnl_analysis():
    """Test unexplained P&L analysis."""
    calc = ModelRiskCalculator()

    unexplained_history = jnp.array([100.0, -50.0, 200.0, -100.0, 150.0])

    mean, std = calc.analyze_unexplained_pnl(unexplained_history)

    # Mean should be close to 60
    assert mean == pytest.approx(60.0, abs=1e-6)

    # Std should be reasonable
    assert std > 0


def test_model_risk_metrics():
    """Test complete model risk metrics."""
    calc = ModelRiskCalculator()

    metrics = calc.compute_model_risk_metrics(
        model_price=102.5,
        market_price=100.0,
        bid_ask_spread=0.5,
        parameter_std=jnp.array([0.02]),
        sensitivity_to_param=jnp.array([50000.0]),
        alternative_model_value=103.0,
        unexplained_pnl_history=jnp.array([100.0, -50.0, 200.0]),
    )

    assert isinstance(metrics, ModelRiskMetrics)
    assert metrics.mark_to_model_reserve > 0
    assert metrics.parameter_uncertainty > 0
    assert metrics.model_replacement_impact != 0
    assert metrics.total_model_risk > 0


# ===== Integration Tests =====


def test_complete_pnl_attribution_workflow():
    """Test complete P&L attribution workflow."""
    # Setup
    greek_calc = GreekPLCalculator()
    risk_calc = RiskFactorPLCalculator()
    model_calc = ModelRiskCalculator()

    # Compute Greek-based P&L explain
    explain = greek_calc.compute_daily_pnl_explain(
        date="2024-01-15",
        total_pnl=2700.0,
        delta=1000.0,
        gamma=50.0,
        vega=2000.0,
        theta=-50.0,
        rho=500.0,
        price_change=2.0,
        vol_change=0.01,
        rate_change=0.001,
        carry_pnl=100.0,
    )

    # Check attribution quality
    assert explain.passes_attribution_test(threshold=0.80)
    assert len(explain.components) == 7

    # Compute risk factor attribution
    ir_attributions = risk_calc.compute_ir_attribution(
        rate_sensitivities={"USD.5Y": 10000.0},
        rate_changes={"USD.5Y": 0.001},
    )

    credit_attributions = risk_calc.compute_credit_attribution(
        credit_sensitivities={"CORP.BBB": 5000.0},
        spread_changes={"CORP.BBB": 0.0005},
    )

    total_ir = sum(attr.total_pnl for attr in ir_attributions)
    total_credit = sum(attr.total_pnl for attr in credit_attributions)

    assert total_ir > 0
    assert total_credit > 0

    # Compute model risk
    model_risk = model_calc.compute_model_risk_metrics(
        model_price=100.5,
        market_price=100.0,
        bid_ask_spread=0.25,
        parameter_std=jnp.array([0.01]),
        sensitivity_to_param=jnp.array([10000.0]),
        alternative_model_value=100.8,
        unexplained_pnl_history=jnp.array([50.0, -30.0, 40.0]),
    )

    assert model_risk.mark_to_model_reserve >= 0
    assert model_risk.parameter_uncertainty > 0


def test_negative_pnl_attribution():
    """Test P&L attribution with negative P&L."""
    calc = GreekPLCalculator()

    explain = calc.compute_daily_pnl_explain(
        date="2024-01-15",
        total_pnl=-3500.0,
        delta=1000.0,
        gamma=50.0,
        vega=2000.0,
        theta=-50.0,
        rho=500.0,
        price_change=-3.0,  # Negative move
        vol_change=-0.02,  # Vol decrease
        rate_change=-0.001,  # Rate decrease
        carry_pnl=100.0,
    )

    # Total P&L should match
    assert explain.total_pnl == -3500.0

    # Delta P&L should be negative (positive delta, negative move)
    assert explain.delta_pnl < 0

    # Gamma P&L should still be positive (always convex)
    assert explain.gamma_pnl > 0

    # Vega P&L should be negative (positive vega, vol decrease)
    assert explain.vega_pnl < 0


def test_risk_factor_attribution():
    """Test RiskFactorAttribution dataclass."""
    attribution = RiskFactorAttribution(
        risk_factor="USD.5Y",
        asset_class="IR",
        delta_pnl=100.0,
        gamma_pnl=20.0,
        vega_pnl=10.0,
        basis_pnl=5.0,
    )

    assert attribution.total_pnl == pytest.approx(135.0, abs=1e-6)
    assert attribution.risk_factor == "USD.5Y"
    assert attribution.asset_class == "IR"
