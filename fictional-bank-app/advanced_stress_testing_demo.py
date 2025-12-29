#!/usr/bin/env python3
"""Advanced Stress Testing Demo - Comprehensive Example

This script demonstrates all the advanced stress testing features:
1. Loading and running predefined scenarios
2. Creating custom scenarios programmatically
3. Loading custom scenarios from YAML files
4. Sensitivity analysis
5. Scenario comparison
6. Advanced visualizations (heatmaps, tornado charts, interactive dashboards)
7. Comprehensive reporting

Run this to see the full capabilities of the enhanced stress testing framework!
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio
from stress_testing import StressTester
from stress_test_visualization import StressTestVisualizer


def main():
    """Run comprehensive stress testing demo."""
    print("=" * 80)
    print("Advanced Stress Testing Demo - Fictional Bank Portfolio")
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

    # Initialize stress tester
    output_dir = Path(__file__).parent / "reports"
    tester = StressTester(output_dir)
    print(f"✓ Stress tester initialized")
    print(f"  Output directory: {output_dir}")
    print(f"  Scenario directory: {tester.scenario_dir}")
    print(f"  Predefined scenarios: {len(tester.scenarios)}")
    print()

    # -------------------------------------------------------------------------
    # 2. Run Standard Stress Tests
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: Running Standard Stress Tests")
    print("=" * 80)
    print()

    # Run a subset of predefined scenarios for demo
    print("Running selected predefined scenarios...")
    selected_scenarios = [
        "IR_ParallelShift_Up100",
        "FX_USDStrength",
        "EQ_MarketCrash_20",
        "VOL_VolatilitySpike",
        "COMB_2008Crisis",
    ]

    standard_results = tester.compare_scenarios(
        portfolio, book_hierarchy, selected_scenarios
    )
    print()
    print("Standard Scenario Results:")
    print(standard_results[["scenario", "category", "pnl_impact", "pnl_impact_pct"]].to_string())
    print()

    # -------------------------------------------------------------------------
    # 3. Create Custom Scenarios Programmatically
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: Creating Custom Scenarios Programmatically")
    print("=" * 80)
    print()

    # Example 1: Custom scenario
    print("Creating custom scenario: Japanese Yen Crisis...")
    tester.create_custom_scenario(
        name="Custom_JPY_Crisis",
        description="Japanese Yen crisis with carry trade unwind",
        shocks={
            "USDJPY": "-25%",  # Yen strengthens significantly
            "FX_vols": "+100%",
            "SPX": "-15%",
            "credit_spreads": "+120bps",
        },
        category="custom",
        severity="severe",
        save=True,
    )
    print()

    # Example 2: Another custom scenario
    print("Creating custom scenario: Green Energy Transition...")
    tester.create_custom_scenario(
        name="Custom_Green_Transition",
        description="Accelerated green energy transition impacts traditional sectors",
        shocks={
            "SPX": "+5%",
            "TSLA": "+40%",  # Green stocks rally
            "USD_curve": "+25bps",
            "EQ_vols": "-15%",
        },
        category="custom",
        severity="mild",
        save=True,
    )
    print()

    print(f"✓ Created {len(tester.custom_scenarios)} custom scenario(s)")
    print()

    # -------------------------------------------------------------------------
    # 4. Load Custom Scenarios from Files
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: Loading Custom Scenarios from YAML Files")
    print("=" * 80)
    print()

    # Load example scenarios
    example_file = tester.scenario_dir / "example_custom_scenarios.yaml"
    if example_file.exists():
        count = tester.load_custom_scenarios(example_file)
        print(f"✓ Loaded {count} scenario(s) from {example_file.name}")
    else:
        print(f"Note: Example file not found at {example_file}")
        print("      Copy example_custom_scenarios.yaml to the scenarios directory")
    print()

    print(f"Total custom scenarios available: {len(tester.custom_scenarios)}")
    print()

    # -------------------------------------------------------------------------
    # 5. Run Custom Scenarios
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: Running Custom Scenarios")
    print("=" * 80)
    print()

    if tester.custom_scenarios:
        custom_results = tester.run_custom_stress_tests(portfolio, book_hierarchy)

        # Combine with standard results for comparison
        all_results_dict = {}

        # Convert DataFrame to dict for standard results
        for _, row in standard_results.iterrows():
            all_results_dict[row["scenario"]] = row.to_dict()

        # Add custom results
        all_results_dict.update(custom_results)
    else:
        print("No custom scenarios to run")
        all_results_dict = {}
        for _, row in standard_results.iterrows():
            all_results_dict[row["scenario"]] = row.to_dict()

    print()

    # -------------------------------------------------------------------------
    # 6. Sensitivity Analysis
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 6: Sensitivity Analysis")
    print("=" * 80)
    print()

    # Analyze sensitivity to equity shocks
    print("Analyzing equity market sensitivity (SPX)...")
    equity_sensitivity = tester.analyze_shock_sensitivity(
        portfolio,
        book_hierarchy,
        shock_type="SPX",
        shock_range=["-30%", "-20%", "-10%", "0", "+10%", "+20%", "+30%"],
    )
    print()

    # Analyze sensitivity to rate shocks
    print("Analyzing interest rate sensitivity (USD curve)...")
    rate_sensitivity = tester.analyze_shock_sensitivity(
        portfolio,
        book_hierarchy,
        shock_type="USD_curve",
        shock_range=["-100bps", "-50bps", "0", "+50bps", "+100bps", "+200bps"],
    )
    print()

    # -------------------------------------------------------------------------
    # 7. Generate Reports
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 7: Generating Reports")
    print("=" * 80)
    print()

    report_file = tester.generate_stress_report(all_results_dict)
    print()

    # -------------------------------------------------------------------------
    # 8. Create Visualizations
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 8: Creating Advanced Visualizations")
    print("=" * 80)
    print()

    viz_dir = Path(__file__).parent / "sample_outputs" / "charts"
    visualizer = StressTestVisualizer(viz_dir)

    # Create all visualizations
    charts = visualizer.create_all_visualizations(all_results_dict, create_interactive=True)
    print()

    # Create sensitivity analysis charts
    print("Creating sensitivity analysis charts...")
    viz_file = visualizer.plot_sensitivity_analysis(equity_sensitivity, "SPX")
    charts["equity_sensitivity"] = viz_file
    print(f"✓ Equity sensitivity chart: {viz_file.name}")

    viz_file = visualizer.plot_sensitivity_analysis(rate_sensitivity, "USD_curve")
    charts["rate_sensitivity"] = viz_file
    print(f"✓ Rate sensitivity chart: {viz_file.name}")
    print()

    # -------------------------------------------------------------------------
    # 9. Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print()

    print("Scenarios Executed:")
    print(f"  Standard scenarios: {len(selected_scenarios)}")
    print(f"  Custom scenarios: {len(tester.custom_scenarios)}")
    print(f"  Total scenarios: {len(all_results_dict)}")
    print()

    print("Sensitivity Analyses:")
    print(f"  Equity (SPX): {len(equity_sensitivity)} data points")
    print(f"  Rates (USD): {len(rate_sensitivity)} data points")
    print()

    print("Generated Outputs:")
    print(f"  Reports: {output_dir}")
    print(f"  Visualizations ({len(charts)} charts): {viz_dir}")
    print()

    print("Key Charts Created:")
    for chart_name, file_path in sorted(charts.items()):
        print(f"  - {chart_name}: {file_path.name}")
    print()

    # Print worst scenarios
    import pandas as pd

    df = pd.DataFrame.from_dict(all_results_dict, orient="index")
    df_sorted = df.sort_values("pnl_impact")

    print("Top 5 Worst Case Scenarios:")
    print("-" * 80)
    for i, (idx, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f"{i}. {row['scenario']:<35} ${row['pnl_impact']:>12,.0f} ({row['pnl_impact_pct']:>+6.2f}%)")
    print()

    print("Top 5 Best Case Scenarios:")
    print("-" * 80)
    for i, (idx, row) in enumerate(df_sorted.tail(5)[::-1].iterrows(), 1):
        print(f"{i}. {row['scenario']:<35} ${row['pnl_impact']:>12,.0f} ({row['pnl_impact_pct']:>+6.2f}%)")
    print()

    # -------------------------------------------------------------------------
    # 10. Next Steps
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review the generated reports in:", output_dir)
    print("2. Open the interactive dashboard:", viz_dir / "stress_interactive_dashboard.html")
    print("3. Create your own custom scenarios in:", tester.scenario_dir)
    print("4. Modify example_custom_scenarios.yaml as a template")
    print("5. Run specific scenarios using the StressTester API")
    print()
    print("For more information, see the updated README and documentation.")
    print()

    print("=" * 80)
    print("Advanced Stress Testing Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
