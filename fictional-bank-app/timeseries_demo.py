#!/usr/bin/env python3
"""Comprehensive Time Series Analysis Demo

This script demonstrates the complete time series analysis framework:
1. Historical data generation (MTM and exposures)
2. Statistical analysis and risk metrics
3. Monte Carlo forecasting
4. Comprehensive visualizations
5. Interactive dashboards

Run this to see the full time series capabilities!
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio
from timeseries_analysis import TimeSeriesAnalyzer
from timeseries_visualization import TimeSeriesVisualizer


def main():
    """Run comprehensive time series analysis demo."""
    print("=" * 80)
    print("Time Series Analysis Demo - Fictional Bank Portfolio")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: Portfolio Setup")
    print("=" * 80)
    print()

    # Load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(output_dir=Path(__file__).parent / "reports")
    print("✓ Time series analyzer initialized")
    print()

    # -------------------------------------------------------------------------
    # 2. Generate Historical Data
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: Generating Historical Data")
    print("=" * 80)
    print()

    print("Generating historical MTM data (252 trading days / 1 year)...")
    mtm_history = analyzer.generate_historical_data(
        portfolio,
        book_hierarchy,
        num_days=252,
        volatility=0.02,  # 2% daily volatility
        drift=0.0001,     # Small positive drift
        seed=42
    )
    print(f"✓ Generated {len(mtm_history.dates)} days of MTM history")
    print(f"  Start: {mtm_history.dates[0].date()}")
    print(f"  End: {mtm_history.dates[-1].date()}")
    print(f"  Initial MTM: ${mtm_history.values[0]:,.2f}")
    print(f"  Final MTM: ${mtm_history.values[-1]:,.2f}")
    print()

    print("Generating counterparty exposure history...")
    exposure_history = analyzer.generate_exposure_history(
        portfolio,
        book_hierarchy,
        num_days=252,
        volatility=0.015,
        seed=42
    )
    print(f"✓ Generated exposure history for {len(exposure_history)} counterparties")
    print()

    # -------------------------------------------------------------------------
    # 3. Statistical Analysis
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: Statistical Analysis")
    print("=" * 80)
    print()

    print("Calculating historical statistics...")
    stats = analyzer.calculate_statistics(mtm_history)
    print("✓ Statistics calculated")
    print()

    print("Calculating Value at Risk (VaR)...")
    var_95 = analyzer.calculate_var(mtm_history, confidence_level=0.95, time_horizon=1)
    var_99 = analyzer.calculate_var(mtm_history, confidence_level=0.99, time_horizon=1)
    print("✓ VaR calculated")
    print()

    # Print summary
    analyzer.print_summary(stats, var_95)

    print("Additional VaR (99% confidence):")
    print("-" * 80)
    print(f"  99% VaR:      ${var_99['var_absolute']:>15,.2f}  ({var_99['var_relative']:>6.2f}%)")
    print(f"  99% CVaR:     ${var_99['cvar_absolute']:>15,.2f}  ({var_99['cvar_relative']:>6.2f}%)")
    print()

    # -------------------------------------------------------------------------
    # 4. Technical Analysis
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: Technical Analysis")
    print("=" * 80)
    print()

    print("Calculating moving averages (5, 20, 50 days)...")
    moving_averages = analyzer.calculate_moving_averages(
        mtm_history,
        windows=[5, 20, 50]
    )
    print("✓ Moving averages calculated")
    print()

    print("Calculating rolling volatility (20-day window)...")
    rolling_vol = analyzer.calculate_rolling_volatility(mtm_history, window=20)
    print("✓ Rolling volatility calculated")
    print()

    # -------------------------------------------------------------------------
    # 5. Forecasting
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: Monte Carlo Forecasting")
    print("=" * 80)
    print()

    print("Running Monte Carlo simulation...")
    print("  Forecast horizon: 60 days (~3 months)")
    print("  Number of scenarios: 1000")
    print()

    forecast = analyzer.forecast_montecarlo(
        mtm_history,
        num_days=60,
        num_scenarios=1000,
        seed=42
    )

    print("✓ Forecast complete")
    print()
    print("Forecast Summary:")
    print("-" * 80)
    print(f"  Forecast Start:   {forecast.dates[0].date()}")
    print(f"  Forecast End:     {forecast.dates[-1].date()}")
    print(f"  Current MTM:      ${mtm_history.values[-1]:>15,.2f}")
    print(f"  Mean Forecast:    ${forecast.mean[-1]:>15,.2f}")
    print(f"  Std Dev:          ${forecast.std[-1]:>15,.2f}")
    print(f"  95% Upper Bound:  ${forecast.percentile_95[-1]:>15,.2f}")
    print(f"  5% Lower Bound:   ${forecast.percentile_5[-1]:>15,.2f}")
    print(f"  Range:            ${forecast.percentile_95[-1] - forecast.percentile_5[-1]:>15,.2f}")
    print()

    # Calculate expected change
    expected_change = forecast.mean[-1] - mtm_history.values[-1]
    expected_pct = (expected_change / mtm_history.values[-1]) * 100

    print(f"  Expected Change:  ${expected_change:>15,.2f}  ({expected_pct:>+6.2f}%)")
    print()

    # -------------------------------------------------------------------------
    # 6. Generate Reports
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 6: Generating Reports")
    print("=" * 80)
    print()

    report_file = analyzer.generate_report(stats, forecast, var_95)
    print()

    # -------------------------------------------------------------------------
    # 7. Create Visualizations
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 7: Creating Visualizations")
    print("=" * 80)
    print()

    viz_dir = Path(__file__).parent / "sample_outputs" / "charts"
    visualizer = TimeSeriesVisualizer(viz_dir)

    charts = visualizer.create_all_charts(
        mtm_history=mtm_history,
        forecast=forecast,
        moving_averages=moving_averages,
        rolling_vol=rolling_vol,
        exposure_history=exposure_history,
        create_interactive=True
    )
    print()

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("TIME SERIES ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    print("Data Generated:")
    print(f"  Historical days: {len(mtm_history.dates)}")
    print(f"  Counterparties tracked: {len(exposure_history)}")
    print(f"  Forecast days: {len(forecast.dates)}")
    print(f"  Monte Carlo scenarios: 1000")
    print()

    print("Key Findings:")
    health_indicator = "🟢" if stats['sharpe_ratio'] > 1 else "🟡" if stats['sharpe_ratio'] > 0 else "🔴"
    print(f"  {health_indicator} Sharpe Ratio: {stats['sharpe_ratio']:.2f}")

    vol_indicator = "🟢" if stats['volatility_annual'] < 0.15 else "🟡" if stats['volatility_annual'] < 0.25 else "🔴"
    print(f"  {vol_indicator} Annual Volatility: {stats['volatility_annual']*100:.2f}%")

    dd_indicator = "🟢" if stats['max_drawdown'] > -10 else "🟡" if stats['max_drawdown'] > -20 else "🔴"
    print(f"  {dd_indicator} Max Drawdown: {stats['max_drawdown']:.2f}%")

    var_indicator = "🟢" if abs(var_95['var_relative']) < 5 else "🟡" if abs(var_95['var_relative']) < 10 else "🔴"
    print(f"  {var_indicator} 1-day 95% VaR: {var_95['var_relative']:.2f}%")
    print()

    print("Generated Outputs:")
    print(f"  Reports: {analyzer.output_dir}")
    print(f"    - {report_file.name}")
    print()

    print(f"  Charts ({len(charts)}): {viz_dir}")
    for name, path in sorted(charts.items()):
        icon = "🌐" if path.suffix == ".html" else "📊"
        print(f"    {icon} {name}: {path.name}")
    print()

    # -------------------------------------------------------------------------
    # 9. Next Steps
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    print("1. View Interactive Dashboard:")
    if "interactive_dashboard" in charts:
        print(f"   - Open: {charts['interactive_dashboard']}")
    print()

    print("2. Analyze Trends:")
    print("   - Review moving average crossovers")
    print("   - Check volatility spikes")
    print("   - Examine exposure evolution")
    print()

    print("3. Review Forecast:")
    print("   - Assess confidence intervals")
    print("   - Plan for worst-case scenarios")
    print("   - Monitor actual vs forecast")
    print()

    print("4. Risk Management:")
    print(f"   - 95% VaR: ${var_95['var_absolute']:,.2f}")
    print(f"   - 99% VaR: ${var_99['var_absolute']:,.2f}")
    print("   - Consider hedging strategies")
    print()

    print("5. Customize Analysis:")
    print("   - Adjust forecast horizon")
    print("   - Change number of scenarios")
    print("   - Modify volatility assumptions")
    print("   - Add custom stress scenarios")
    print()

    print("=" * 80)
    print("Time Series Analysis Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
