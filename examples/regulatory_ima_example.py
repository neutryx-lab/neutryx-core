"""Example: Internal Models Approach (IMA) for Market Risk Capital.

This example demonstrates the Basel III/FRTB IMA framework including:
1. Expected Shortfall (ES) calculation at 97.5% confidence
2. P&L Attribution (PLA) testing for model validation
3. VaR/ES backtesting with traffic light approach
4. Non-Modellable Risk Factors (NMRF) identification and treatment

Run this example:
    python examples/regulatory_ima_example.py
"""

from datetime import date, timedelta

import jax
import jax.numpy as jnp

from neutryx.regulatory.ima import (
    # Expected Shortfall
    calculate_expected_shortfall,
    calculate_stressed_es,
    get_liquidity_horizon,
    # P&L Attribution
    calculate_pla_metrics,
    PLATestResult,
    # Backtesting
    backtest_var,
    backtest_expected_shortfall,
    TrafficLightZone,
    # NMRF
    identify_nmrfs,
    StressScenarioCalibrator,
    NMRFCapitalCalculator,
)


def example_1_expected_shortfall():
    """Example 1: Calculate Expected Shortfall at 97.5% confidence."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Expected Shortfall (ES) Calculation")
    print("="*80)

    # Generate P&L scenarios (e.g., from Monte Carlo simulation)
    key = jax.random.PRNGKey(42)
    pnl_scenarios = jax.random.normal(key, (10000,)) * 100000.0  # $100k std dev

    # Calculate ES at 97.5% confidence (Basel III requirement)
    es, var, diagnostics = calculate_expected_shortfall(
        pnl_scenarios,
        confidence_level=0.975
    )

    print(f"\nP&L Scenarios: {len(pnl_scenarios)} simulations")
    print(f"VaR (97.5%): ${abs(var):,.2f}")
    print(f"ES (97.5%): ${abs(es):,.2f}")
    print(f"ES/VaR Ratio: {abs(es/var):.2f}")
    print(f"\nTail statistics:")
    print(f"  - Tail observations: {diagnostics['tail_observations']}")
    print(f"  - Max loss: ${diagnostics['max_loss']:,.2f}")
    print(f"  - Mean excess loss: ${diagnostics['mean_excess_loss']:,.2f}")

    # Demonstrate liquidity horizon adjustments
    print(f"\nLiquidity Horizons (Basel III):")
    print(f"  - Large Cap Equity: {get_liquidity_horizon('equity', 'large_cap').value}")
    print(f"  - Illiquid Credit: {get_liquidity_horizon('credit', 'hy_single').value}")
    print(f"  - Commodity: {get_liquidity_horizon('commodity', 'electricity').value}")

    return es, var


def example_2_pla_testing():
    """Example 2: P&L Attribution Testing."""
    print("\n" + "="*80)
    print("EXAMPLE 2: P&L Attribution (PLA) Testing")
    print("="*80)

    # Simulate 250 days of P&L data
    key = jax.random.PRNGKey(123)
    n_days = 250

    # Hypothetical P&L (from front-office pricing models)
    hypothetical_pnl = jax.random.normal(key, (n_days,)) * 50000.0

    # Risk-Theoretical P&L (from risk management models)
    # Good model: RTPL tracks HPL closely with small noise
    noise = jax.random.normal(jax.random.PRNGKey(456), (n_days,)) * 5000.0
    risk_theoretical_pnl = hypothetical_pnl + noise

    # Calculate PLA metrics
    pla_metrics = calculate_pla_metrics(
        hypothetical_pnl,
        risk_theoretical_pnl,
        spearman_threshold=0.85,
        ks_threshold=0.09
    )

    print(f"\nPLA Test Results ({n_days} days):")
    print(f"  - Spearman Correlation: {pla_metrics.spearman_correlation:.4f} "
          f"(threshold: {pla_metrics.spearman_threshold})")
    print(f"  - Kolmogorov-Smirnov: {pla_metrics.kolmogorov_smirnov_statistic:.4f} "
          f"(threshold: {pla_metrics.ks_threshold})")
    print(f"  - Test Result: {pla_metrics.test_result.value.upper()}")
    print(f"  - Passes Test: {'YES' if pla_metrics.passes_test else 'NO'}")
    print(f"\nAttribution Quality:")
    print(f"  - Mean Unexplained P&L: ${pla_metrics.mean_unexplained_pnl:,.2f}")
    print(f"  - RMSE: ${pla_metrics.root_mean_squared_error:,.2f}")

    if pla_metrics.test_result == PLATestResult.GREEN:
        print(f"\n✓ Trading desk QUALIFIES for IMA")
    else:
        print(f"\n✗ Trading desk requires additional monitoring or standardized approach")

    return pla_metrics


def example_3_var_backtesting():
    """Example 3: VaR Backtesting with Traffic Light Approach."""
    print("\n" + "="*80)
    print("EXAMPLE 3: VaR Backtesting with Traffic Light Approach")
    print("="*80)

    # Simulate 250 days of actual P&L and VaR forecasts
    key = jax.random.PRNGKey(789)
    n_days = 250

    actual_pnl = jax.random.normal(key, (n_days,)) * 10000.0
    var_forecasts = jnp.ones(n_days) * -25000.0  # 99% VaR forecast

    # Backtest VaR
    result = backtest_var(
        actual_pnl,
        var_forecasts,
        coverage_level=0.99
    )

    print(f"\nVaR Backtest Results ({n_days} days, 99% confidence):")
    print(f"  - Exceptions: {result.num_exceptions} "
          f"(expected: {result.expected_exceptions:.1f})")
    print(f"  - Exception Rate: {result.exception_rate:.2%}")
    print(f"  - Traffic Light Zone: {result.traffic_light_zone.value.upper()}")
    print(f"  - Capital Multiplier: {result.capital_multiplier:.2f}x")

    print(f"\nStatistical Tests:")
    if result.kupiec_pof_pvalue is not None:
        print(f"  - Kupiec POF p-value: {result.kupiec_pof_pvalue:.4f}")
    if result.christoffersen_test_pvalue is not None:
        print(f"  - Christoffersen p-value: {result.christoffersen_test_pvalue:.4f}")

    # Interpret results
    if result.traffic_light_zone == TrafficLightZone.GREEN:
        print(f"\n✓ GREEN ZONE: Model performs well, no capital add-on required")
    elif result.traffic_light_zone == TrafficLightZone.AMBER:
        print(f"\n⚠ AMBER ZONE: Model acceptable but requires monitoring")
    else:
        print(f"\n✗ RED ZONE: Model inadequate, must recalibrate or use standardized approach")

    return result


def example_4_nmrf_identification():
    """Example 4: Non-Modellable Risk Factors (NMRF) Identification."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Non-Modellable Risk Factors (NMRF) Identification")
    print("="*80)

    # Create observation dates for different risk factors
    start_date = date(2023, 1, 1)

    risk_factors = {
        # Modellable: Liquid G10 FX pair with frequent observations
        "USD.EUR": [start_date + timedelta(days=i * 15) for i in range(25)],

        # Modellable: Major equity index
        "SPX.Index": [start_date + timedelta(days=i * 14) for i in range(27)],

        # Non-modellable: Exotic option with insufficient data
        "EXOTIC.OPTION.1": [start_date + timedelta(days=i * 30) for i in range(10)],

        # Non-modellable: Illiquid bond with large gaps
        "ILLIQUID.BOND": [
            start_date,
            start_date + timedelta(days=60),  # Large gap
            start_date + timedelta(days=120),
        ] + [start_date + timedelta(days=120 + i * 10) for i in range(20)],
    }

    # Identify NMRFs
    modellable, non_modellable, test_results = identify_nmrfs(
        risk_factors,
        min_observations=24,
        observation_period_days=365,
        max_gap_days=30
    )

    print(f"\nRisk Factor Modellability Assessment:")
    print(f"\nModellable Risk Factors ({len(modellable)}):")
    for rf in modellable:
        result = test_results[rf]
        print(f"  ✓ {rf}: {result.observation_count} observations over "
              f"{result.observation_period_days} days")

    print(f"\nNon-Modellable Risk Factors ({len(non_modellable)}):")
    for rf in non_modellable:
        result = test_results[rf]
        print(f"  ✗ {rf}: {result.reason}")

    print(f"\nRegulatory Treatment:")
    print(f"  - Modellable RFs: Use internal models (ES at 97.5%)")
    print(f"  - Non-Modellable RFs: Use stressed scenarios")

    return modellable, non_modellable


def example_5_nmrf_capital():
    """Example 5: NMRF Capital Calculation with Stressed Scenarios."""
    print("\n" + "="*80)
    print("EXAMPLE 5: NMRF Capital Calculation")
    print("="*80)

    # Generate historical returns for stress period identification
    key = jax.random.PRNGKey(999)
    returns = jnp.concatenate([
        jax.random.normal(jax.random.PRNGKey(1), (200,)) * 0.02,  # Normal period
        jax.random.normal(jax.random.PRNGKey(2), (200,)) * 0.08 - 0.03,  # Stress period
        jax.random.normal(jax.random.PRNGKey(3), (200,)) * 0.02,  # Recovery
    ])

    dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(len(returns))]

    # Identify stress period
    calibrator = StressScenarioCalibrator(stress_period_days=200)
    stress_period = calibrator.identify_stress_period(returns, dates)

    print(f"\nStress Period Identification:")
    print(f"  - Period: {stress_period.start_date} to {stress_period.end_date}")
    print(f"  - Duration: {stress_period.duration_days} days")
    print(f"  - Stressed ES: {stress_period.stressed_es:.4f}")
    print(f"  - Severity Score: {stress_period.severity_score:.2f}")

    # Generate stress scenarios
    stress_scenarios = calibrator.calibrate_stress_scenarios(
        stress_period,
        num_scenarios=1000
    )

    # Calculate NMRF capital
    calculator = NMRFCapitalCalculator(
        confidence_level=0.975,
        liquidity_horizon_days=40  # Illiquid asset
    )

    capital = calculator.calculate_capital(
        risk_factor="ILLIQUID.BOND",
        stress_period=stress_period,
        stress_scenarios=stress_scenarios,
        position_size=10_000_000.0  # $10M position
    )

    print(f"\nNMRF Capital Calculation:")
    print(f"  - Risk Factor: {capital.risk_factor}")
    print(f"  - Position Size: ${capital.diagnostics['position_size']:,.0f}")
    print(f"  - Liquidity Horizon: {capital.liquidity_horizon_days} days")
    print(f"  - Scaling Factor: {capital.scaling_factor:.2f}x")
    print(f"  - Stressed ES (unscaled): ${abs(capital.stressed_es)*100:.2f}%")
    print(f"  - Capital Charge: ${capital.capital_charge:,.2f}")

    print(f"\nStress Scenario Statistics:")
    print(f"  - Number of Scenarios: {capital.diagnostics['num_scenarios']}")
    print(f"  - Mean Scenario Loss: {capital.diagnostics['mean_scenario_loss']:.4f}")
    print(f"  - Max Scenario Loss: {capital.diagnostics['max_scenario_loss']:.4f}")

    return capital


def main():
    """Run all IMA examples."""
    print("\n" + "="*80)
    print("BASEL III INTERNAL MODELS APPROACH (IMA) - EXAMPLES")
    print("="*80)
    print("\nDemonstrating market risk capital calculation under Basel III/FRTB")

    # Run all examples
    example_1_expected_shortfall()
    example_2_pla_testing()
    example_3_var_backtesting()
    example_4_nmrf_identification()
    example_5_nmrf_capital()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
