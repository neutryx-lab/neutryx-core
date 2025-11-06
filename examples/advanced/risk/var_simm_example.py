"""Example demonstrating VaR and SIMM calculations.

This example shows how to:
1. Calculate VaR using different methodologies (Historical, Parametric, Monte Carlo)
2. Compute SIMM initial margin for derivatives portfolios
3. Backtest VaR models
4. Analyze portfolio risk metrics
"""
import jax.numpy as jnp
import jax.random as random
import numpy as np

from neutryx.valuations.risk_metrics import (
    VaRMethod,
    backtest_var,
    calculate_var,
    component_var,
    compute_all_risk_metrics,
    marginal_var,
    portfolio_var,
)
from neutryx.valuations.simm import (
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType,
    calculate_simm,
)


def example_var_calculations():
    """Demonstrate different VaR calculation methods."""
    print("=" * 80)
    print("VaR Calculation Examples")
    print("=" * 80)

    # Generate sample returns (250 trading days)
    np.random.seed(42)
    returns = jnp.array(np.random.normal(0.0005, 0.015, 250))

    print(f"\nSample returns: {len(returns)} days")
    print(f"Mean return: {float(jnp.mean(returns)):.4%}")
    print(f"Std deviation: {float(jnp.std(returns)):.4%}")

    # Calculate VaR using different methods
    confidence_level = 0.95

    var_hist = calculate_var(returns, confidence_level, VaRMethod.HISTORICAL)
    var_param = calculate_var(returns, confidence_level, VaRMethod.PARAMETRIC)
    var_mc = calculate_var(returns, confidence_level, VaRMethod.MONTE_CARLO)
    var_cf = calculate_var(returns, confidence_level, VaRMethod.CORNISH_FISHER)

    print(f"\n95% VaR Results:")
    print(f"  Historical VaR:      {var_hist:.4%}")
    print(f"  Parametric VaR:      {var_param:.4%}")
    print(f"  Monte Carlo VaR:     {var_mc:.4%}")
    print(f"  Cornish-Fisher VaR:  {var_cf:.4%}")

    # 99% VaR
    var_99 = calculate_var(returns, 0.99, VaRMethod.HISTORICAL)
    print(f"\n99% Historical VaR:    {var_99:.4%}")

    # Comprehensive risk metrics
    print("\nComprehensive Risk Metrics:")
    metrics = compute_all_risk_metrics(returns, confidence_levels=[0.95, 0.99])
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.6f}")


def example_portfolio_var():
    """Demonstrate portfolio VaR analysis."""
    print("\n" + "=" * 80)
    print("Portfolio VaR Analysis")
    print("=" * 80)

    # Create portfolio with 3 assets
    np.random.seed(42)
    n_scenarios = 1000
    n_assets = 3

    # Generate correlated returns
    cov_matrix = np.array([
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0009, 0.0003],
        [0.0001, 0.0003, 0.0016],
    ])

    mean_returns = np.array([0.0008, 0.0012, 0.0015])
    returns_scenarios = jnp.array(
        np.random.multivariate_normal(mean_returns, cov_matrix, n_scenarios)
    )

    # Portfolio positions (in dollars)
    positions = jnp.array([1_000_000, 500_000, 750_000])

    print(f"\nPortfolio Positions:")
    for i, pos in enumerate(positions):
        print(f"  Asset {i+1}: ${pos:,.0f}")
    print(f"  Total:   ${float(jnp.sum(positions)):,.0f}")

    # Calculate portfolio VaR
    pvar_95 = portfolio_var(positions, returns_scenarios, 0.95)
    pvar_99 = portfolio_var(positions, returns_scenarios, 0.99)

    print(f"\nPortfolio VaR:")
    print(f"  95% VaR: ${pvar_95:,.0f}")
    print(f"  99% VaR: ${pvar_99:,.0f}")

    # Component VaR
    comp_vars = component_var(positions, returns_scenarios, 0.95)
    print(f"\nComponent VaR (95%):")
    for i, cv in enumerate(comp_vars):
        print(f"  Asset {i+1}: ${float(cv):,.0f}")

    # Marginal VaR
    marg_vars = marginal_var(positions, returns_scenarios, 0.95)
    print(f"\nMarginal VaR (95%):")
    for i, mv in enumerate(marg_vars):
        print(f"  Asset {i+1}: ${float(mv):.2f} per $1 increase")


def example_var_backtest():
    """Demonstrate VaR backtesting."""
    print("\n" + "=" * 80)
    print("VaR Backtesting Example")
    print("=" * 80)

    # Generate realized returns and VaR forecasts
    np.random.seed(42)
    n_days = 250

    realized_returns = jnp.array(np.random.normal(0.0005, 0.015, n_days))

    # Good model: VaR forecasts match distribution
    var_good = jnp.full(n_days, 0.025)  # ~95th percentile

    # Bad model: underestimated VaR
    var_bad = jnp.full(n_days, 0.010)  # Too low

    print(f"\nBacktesting over {n_days} days")

    # Backtest good model
    result_good = backtest_var(realized_returns, var_good, 0.95)
    print(f"\nGood Model Results:")
    print(f"  Violations:         {result_good['violations']} / {n_days}")
    print(f"  Violation Rate:     {result_good['violation_rate']:.2%}")
    print(f"  Expected Rate:      {result_good['expected_rate']:.2%}")
    print(f"  Kupiec p-value:     {result_good['kupiec_pvalue']:.4f}")
    print(f"  Pass backtest:      {result_good['pass_backtest']}")

    # Backtest bad model
    result_bad = backtest_var(realized_returns, var_bad, 0.95)
    print(f"\nBad Model Results:")
    print(f"  Violations:         {result_bad['violations']} / {n_days}")
    print(f"  Violation Rate:     {result_bad['violation_rate']:.2%}")
    print(f"  Expected Rate:      {result_bad['expected_rate']:.2%}")
    print(f"  Kupiec p-value:     {result_bad['kupiec_pvalue']:.4f}")
    print(f"  Pass backtest:      {result_bad['pass_backtest']}")


def example_simm_ir_portfolio():
    """Demonstrate SIMM calculation for IR derivatives."""
    print("\n" + "=" * 80)
    print("SIMM Calculation - Interest Rate Derivatives")
    print("=" * 80)

    # Create IR sensitivities for an interest rate swap portfolio
    sensitivities = [
        # USD sensitivities at different tenors
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-LIBOR",
            sensitivity=500_000,
            tenor="2Y",
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-LIBOR",
            sensitivity=-300_000,
            tenor="5Y",
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-LIBOR",
            sensitivity=200_000,
            tenor="10Y",
        ),
        # EUR sensitivities
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="EUR",
            risk_factor="EUR-EURIBOR",
            sensitivity=400_000,
            tenor="5Y",
        ),
    ]

    print(f"\nPortfolio IR Sensitivities:")
    for sens in sensitivities:
        sign = "+" if sens.sensitivity > 0 else ""
        print(f"  {sens.bucket} {sens.tenor}: {sign}{sens.sensitivity:,.0f}")

    # Calculate SIMM
    result = calculate_simm(sensitivities)

    print(f"\nSIMM Results:")
    print(f"  Total Initial Margin:  ${result.total_im:,.0f}")
    print(f"  Delta Margin:          ${result.delta_im:,.0f}")
    print(f"  Vega Margin:           ${result.vega_im:,.0f}")

    print(f"\nBy Risk Class:")
    for risk_class, im in result.im_by_risk_class.items():
        print(f"  {risk_class.value:20s}: ${im:,.0f}")


def example_simm_multi_asset():
    """Demonstrate SIMM with multiple asset classes."""
    print("\n" + "=" * 80)
    print("SIMM Calculation - Multi-Asset Portfolio")
    print("=" * 80)

    # Portfolio with IR, FX, and Equity sensitivities
    sensitivities = [
        # IR sensitivities
        RiskFactorSensitivity(
            RiskFactorType.IR,
            SensitivityType.DELTA,
            "USD",
            "USD-LIBOR",
            300_000,
            "5Y",
        ),
        RiskFactorSensitivity(
            RiskFactorType.IR,
            SensitivityType.DELTA,
            "EUR",
            "EUR-EURIBOR",
            250_000,
            "5Y",
        ),
        # FX sensitivities
        RiskFactorSensitivity(
            RiskFactorType.FX, SensitivityType.DELTA, "1", "EURUSD", 150_000
        ),
        RiskFactorSensitivity(
            RiskFactorType.FX, SensitivityType.DELTA, "1", "GBPUSD", 100_000
        ),
        # Equity sensitivities
        RiskFactorSensitivity(
            RiskFactorType.EQUITY, SensitivityType.DELTA, "1", "AAPL", 80_000
        ),
        RiskFactorSensitivity(
            RiskFactorType.EQUITY, SensitivityType.DELTA, "1", "MSFT", 60_000
        ),
        # Vega sensitivities
        RiskFactorSensitivity(
            RiskFactorType.IR,
            SensitivityType.VEGA,
            "USD",
            "USD-SWAPTION",
            20_000,
            "5Y",
        ),
        RiskFactorSensitivity(
            RiskFactorType.EQUITY, SensitivityType.VEGA, "1", "SPX-VOL", 15_000
        ),
    ]

    print(f"\nMulti-Asset Portfolio:")
    print(f"  Total risk factors: {len(sensitivities)}")

    # Group by type
    delta_count = sum(1 for s in sensitivities if s.sensitivity_type == SensitivityType.DELTA)
    vega_count = sum(1 for s in sensitivities if s.sensitivity_type == SensitivityType.VEGA)

    print(f"  Delta sensitivities: {delta_count}")
    print(f"  Vega sensitivities:  {vega_count}")

    # Calculate SIMM
    result = calculate_simm(sensitivities, product_class_multiplier=1.0)

    print(f"\nSIMM Results:")
    print(f"  Total Initial Margin:  ${result.total_im:,.0f}")
    print(f"  Delta Margin:          ${result.delta_im:,.0f}")
    print(f"  Vega Margin:           ${result.vega_im:,.0f}")

    print(f"\nMargin by Risk Class:")
    for risk_class, im in result.im_by_risk_class.items():
        pct = (im / result.total_im) * 100
        print(f"  {risk_class.value:20s}: ${im:10,.0f}  ({pct:5.1f}%)")

    # Compare with higher product class multiplier
    result_pcm = calculate_simm(sensitivities, product_class_multiplier=1.3)
    print(f"\nWith Product Class Multiplier 1.3:")
    print(f"  Total Initial Margin:  ${result_pcm.total_im:,.0f}")
    print(f"  Increase:              ${result_pcm.total_im - result.total_im:,.0f} "
          f"({(result_pcm.total_im / result.total_im - 1) * 100:.0f}%)")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print(" Neutryx Risk Metrics: VaR and SIMM Examples")
    print("*" * 80)

    # VaR examples
    example_var_calculations()
    example_portfolio_var()
    example_var_backtest()

    # SIMM examples
    example_simm_ir_portfolio()
    example_simm_multi_asset()

    print("\n" + "*" * 80)
    print(" Examples Complete")
    print("*" * 80)
    print()


if __name__ == "__main__":
    main()
