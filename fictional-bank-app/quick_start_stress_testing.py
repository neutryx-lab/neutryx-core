#!/usr/bin/env python3
"""Quick Start - Advanced Stress Testing

This is a minimal example to get you started with the enhanced stress testing features.
Run this script to see custom scenarios, sensitivity analysis, and visualizations in action.

Usage:
    python quick_start_stress_testing.py
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
    print("=" * 80)
    print("Quick Start: Advanced Stress Testing")
    print("=" * 80)
    print()

    # 1. Load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # 2. Initialize stress tester
    output_dir = Path(__file__).parent / "reports"
    tester = StressTester(output_dir)
    print(f"✓ Stress tester initialized")
    print()

    # 3. Create a custom scenario
    print("Creating custom scenario...")
    tester.create_custom_scenario(
        name="QuickStart_MarketShock",
        description="Quick example: Combined market shock",
        shocks={
            "USD_curve": "+100bps",
            "SPX": "-15%",
            "FX_vols": "+60%"
        },
        category="custom",
        severity="moderate",
        save=True
    )
    print("✓ Custom scenario created and saved")
    print()

    # 4. Run a few scenarios
    print("Running stress scenarios...")
    selected_scenarios = [
        "IR_ParallelShift_Up100",
        "EQ_MarketCrash_20",
        "QuickStart_MarketShock"
    ]

    results_df = tester.compare_scenarios(
        portfolio, book_hierarchy, selected_scenarios
    )

    # Convert to dict for visualization
    results_dict = {}
    for _, row in results_df.iterrows():
        results_dict[row["scenario"]] = row.to_dict()

    print()
    print("Results:")
    print(results_df[["scenario", "pnl_impact", "pnl_impact_pct"]].to_string())
    print()

    # 5. Quick sensitivity analysis
    print("Running sensitivity analysis...")
    sensitivity = tester.analyze_shock_sensitivity(
        portfolio,
        book_hierarchy,
        shock_type="SPX",
        shock_range=["-20%", "-10%", "0", "+10%", "+20%"]
    )
    print()

    # 6. Generate visualizations
    print("Creating visualizations...")
    viz_dir = Path(__file__).parent / "sample_outputs" / "charts"
    visualizer = StressTestVisualizer(viz_dir)

    # Create key charts
    heatmap = visualizer.plot_impact_heatmap(results_dict)
    print(f"✓ Heatmap: {heatmap.name}")

    tornado = visualizer.plot_tornado_chart(results_dict)
    print(f"✓ Tornado chart: {tornado.name}")

    sens_chart = visualizer.plot_sensitivity_analysis(sensitivity, "SPX")
    print(f"✓ Sensitivity chart: {sens_chart.name}")

    # Create interactive dashboard if Plotly available
    try:
        dashboard = visualizer.create_interactive_dashboard(results_dict)
        print(f"✓ Interactive dashboard: {dashboard.name}")
        print()
        print(f"🌐 Open in browser: {dashboard}")
    except Exception as e:
        print(f"Note: Interactive dashboard not created ({e})")

    print()

    # 7. Summary
    print("=" * 80)
    print("QUICK START COMPLETE!")
    print("=" * 80)
    print()
    print("What you just did:")
    print("  ✓ Created a custom stress scenario")
    print("  ✓ Ran multiple stress tests")
    print("  ✓ Performed sensitivity analysis")
    print("  ✓ Generated advanced visualizations")
    print()
    print("Next steps:")
    print("  1. Check the generated charts in:", viz_dir)
    print("  2. View your custom scenario in:", tester.scenario_dir)
    print("  3. Run the full demo: ./advanced_stress_testing_demo.py")
    print("  4. Read STRESS_TESTING_GUIDE.md for detailed documentation")
    print()
    print("Try creating more custom scenarios in:", tester.scenario_dir)
    print("Use example_custom_scenarios.yaml as a template!")
    print()


if __name__ == "__main__":
    main()
