#!/usr/bin/env python3
"""Stress testing framework for portfolio analysis.

This script demonstrates comprehensive stress testing including:
- Interest rate shocks (parallel and non-parallel shifts)
- FX rate stress scenarios
- Equity market crashes
- Volatility shocks
- Credit spread widening
- Combined multi-factor stress tests
- Historical scenario replays
"""
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


@dataclass
class StressScenario:
    """Definition of a stress scenario."""

    name: str
    description: str
    category: str  # 'rates', 'fx', 'equity', 'volatility', 'credit', 'combined'
    shocks: Dict[str, Any]
    severity: str  # 'mild', 'moderate', 'severe', 'extreme'


class StressTester:
    """Comprehensive stress testing framework."""

    def __init__(self, output_dir: Path, scenario_dir: Optional[Path] = None):
        """Initialize the stress tester.

        Args:
            output_dir: Directory to save stress test results
            scenario_dir: Optional directory to load/save custom scenarios
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scenario_dir = Path(scenario_dir) if scenario_dir else self.output_dir.parent / "data" / "scenarios"
        self.scenario_dir.mkdir(parents=True, exist_ok=True)

        self.scenarios = self._define_scenarios()
        self.custom_scenarios = []

    def _define_scenarios(self) -> List[StressScenario]:
        """Define stress test scenarios."""
        scenarios = []

        # Interest rate scenarios
        scenarios.extend(
            [
                StressScenario(
                    name="IR_ParallelShift_Up100",
                    description="Parallel shift up 100 bps across all maturities",
                    category="rates",
                    shocks={"USD_curve": "+100bps", "EUR_curve": "+100bps"},
                    severity="moderate",
                ),
                StressScenario(
                    name="IR_ParallelShift_Down50",
                    description="Parallel shift down 50 bps across all maturities",
                    category="rates",
                    shocks={"USD_curve": "-50bps", "EUR_curve": "-50bps"},
                    severity="mild",
                ),
                StressScenario(
                    name="IR_Steepener",
                    description="Yield curve steepening (short -25bps, long +75bps)",
                    category="rates",
                    shocks={
                        "USD_short": "-25bps",
                        "USD_long": "+75bps",
                        "EUR_short": "-25bps",
                        "EUR_long": "+75bps",
                    },
                    severity="moderate",
                ),
                StressScenario(
                    name="IR_Flattener",
                    description="Yield curve flattening (short +50bps, long -25bps)",
                    category="rates",
                    shocks={
                        "USD_short": "+50bps",
                        "USD_long": "-25bps",
                        "EUR_short": "+50bps",
                        "EUR_long": "-25bps",
                    },
                    severity="moderate",
                ),
            ]
        )

        # FX scenarios
        scenarios.extend(
            [
                StressScenario(
                    name="FX_USDStrength",
                    description="USD strengthens 10% against all currencies",
                    category="fx",
                    shocks={"EURUSD": "-10%", "USDJPY": "+10%", "GBPUSD": "-10%", "USDBRL": "+10%"},
                    severity="moderate",
                ),
                StressScenario(
                    name="FX_USDWeakness",
                    description="USD weakens 10% against all currencies",
                    category="fx",
                    shocks={"EURUSD": "+10%", "USDJPY": "-10%", "GBPUSD": "+10%", "USDBRL": "-10%"},
                    severity="moderate",
                ),
                StressScenario(
                    name="FX_EMCrisis",
                    description="Emerging market currency crisis (BRL -30%)",
                    category="fx",
                    shocks={"USDBRL": "+30%"},
                    severity="severe",
                ),
            ]
        )

        # Equity scenarios
        scenarios.extend(
            [
                StressScenario(
                    name="EQ_MarketCrash_20",
                    description="Equity market crash: 20% decline across all equities",
                    category="equity",
                    shocks={"SPX": "-20%", "AAPL": "-20%", "TSLA": "-20%"},
                    severity="severe",
                ),
                StressScenario(
                    name="EQ_MarketRally_15",
                    description="Equity market rally: 15% rise across all equities",
                    category="equity",
                    shocks={"SPX": "+15%", "AAPL": "+15%", "TSLA": "+15%"},
                    severity="moderate",
                ),
                StressScenario(
                    name="EQ_TechCrash",
                    description="Technology sector crash (tech stocks -35%)",
                    category="equity",
                    shocks={"AAPL": "-35%", "TSLA": "-35%"},
                    severity="severe",
                ),
            ]
        )

        # Volatility scenarios
        scenarios.extend(
            [
                StressScenario(
                    name="VOL_VolatilitySpike",
                    description="Volatility spike: +50% across all asset classes",
                    category="volatility",
                    shocks={
                        "FX_vols": "+50%",
                        "EQ_vols": "+50%",
                        "IR_vols": "+50%",
                    },
                    severity="severe",
                ),
                StressScenario(
                    name="VOL_VolatilityCrush",
                    description="Volatility crush: -30% across all asset classes",
                    category="volatility",
                    shocks={
                        "FX_vols": "-30%",
                        "EQ_vols": "-30%",
                        "IR_vols": "-30%",
                    },
                    severity="moderate",
                ),
            ]
        )

        # Credit scenarios
        scenarios.extend(
            [
                StressScenario(
                    name="CR_SpreadWidening_Mild",
                    description="Credit spread widening: +50 bps",
                    category="credit",
                    shocks={"credit_spreads": "+50bps"},
                    severity="mild",
                ),
                StressScenario(
                    name="CR_SpreadWidening_Severe",
                    description="Credit spread widening: +200 bps",
                    category="credit",
                    shocks={"credit_spreads": "+200bps"},
                    severity="severe",
                ),
                StressScenario(
                    name="CR_RatingDowngrade",
                    description="Mass rating downgrades (1 notch for all counterparties)",
                    category="credit",
                    shocks={"rating_migration": "-1"},
                    severity="moderate",
                ),
            ]
        )

        # Combined scenarios (multi-factor stress)
        scenarios.extend(
            [
                StressScenario(
                    name="COMB_2008Crisis",
                    description="2008 Financial Crisis scenario",
                    category="combined",
                    shocks={
                        "USD_curve": "-150bps",
                        "EUR_curve": "-100bps",
                        "SPX": "-40%",
                        "credit_spreads": "+400bps",
                        "FX_vols": "+100%",
                        "EQ_vols": "+150%",
                    },
                    severity="extreme",
                ),
                StressScenario(
                    name="COMB_2020COVID",
                    description="2020 COVID-19 pandemic scenario",
                    category="combined",
                    shocks={
                        "USD_curve": "-100bps",
                        "SPX": "-35%",
                        "USDBRL": "+25%",
                        "FX_vols": "+80%",
                        "EQ_vols": "+120%",
                    },
                    severity="extreme",
                ),
                StressScenario(
                    name="COMB_Stagflation",
                    description="Stagflation scenario (high rates, weak equities)",
                    category="combined",
                    shocks={
                        "USD_curve": "+200bps",
                        "EUR_curve": "+150bps",
                        "SPX": "-15%",
                        "credit_spreads": "+150bps",
                    },
                    severity="severe",
                ),
                StressScenario(
                    name="COMB_PerfectStorm",
                    description="Perfect storm: all risk factors move adversely",
                    category="combined",
                    shocks={
                        "USD_curve": "+150bps",
                        "EUR_curve": "+150bps",
                        "SPX": "-30%",
                        "AAPL": "-40%",
                        "TSLA": "-50%",
                        "EURUSD": "+15%",
                        "USDBRL": "+35%",
                        "credit_spreads": "+300bps",
                        "FX_vols": "+100%",
                        "EQ_vols": "+100%",
                    },
                    severity="extreme",
                ),
            ]
        )

        return scenarios

    def run_all_stress_tests(
        self, portfolio: Any, book_hierarchy: Any
    ) -> Dict[str, Dict]:
        """Run all stress scenarios.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object

        Returns:
            Dictionary of stress test results by scenario name
        """
        print("Running comprehensive stress tests...")
        print(f"Total scenarios to execute: {len(self.scenarios)}")
        print()

        results = {}
        baseline_summary = get_portfolio_summary(portfolio, book_hierarchy)
        baseline_mtm = baseline_summary["total_mtm"]

        for i, scenario in enumerate(self.scenarios, 1):
            print(f"[{i}/{len(self.scenarios)}] Running: {scenario.name}")
            print(f"  Category: {scenario.category.upper()}")
            print(f"  Severity: {scenario.severity.upper()}")
            print(f"  Description: {scenario.description}")

            # Simulate stress test (in real implementation, this would call API)
            stressed_mtm = self._simulate_stress_impact(
                baseline_mtm, scenario, baseline_summary
            )

            pnl_impact = stressed_mtm - baseline_mtm
            pnl_pct = (pnl_impact / baseline_mtm * 100) if baseline_mtm != 0 else 0

            results[scenario.name] = {
                "scenario": scenario.name,
                "category": scenario.category,
                "severity": scenario.severity,
                "description": scenario.description,
                "baseline_mtm": baseline_mtm,
                "stressed_mtm": stressed_mtm,
                "pnl_impact": pnl_impact,
                "pnl_impact_pct": pnl_pct,
                "shocks": scenario.shocks,
            }

            print(f"  Baseline MTM: ${baseline_mtm:,.2f}")
            print(f"  Stressed MTM: ${stressed_mtm:,.2f}")
            print(f"  P&L Impact: ${pnl_impact:,.2f} ({pnl_pct:+.2f}%)")
            print()

        return results

    def _simulate_stress_impact(
        self, baseline_mtm: float, scenario: StressScenario, summary: Dict
    ) -> float:
        """Simulate stress impact on MTM.

        This is a simplified simulation for demonstration.
        In a real implementation, this would recalculate portfolio MTM
        with stressed market data via the Neutryx API.

        Args:
            baseline_mtm: Baseline portfolio MTM
            scenario: Stress scenario
            summary: Portfolio summary

        Returns:
            Stressed portfolio MTM
        """
        # Simplified impact estimation based on scenario type
        impact_factor = 1.0

        if scenario.category == "rates":
            # Rate shock impact (approximate duration impact)
            if "100bps" in str(scenario.shocks):
                impact_factor = 0.95  # -5% for 100bps up
            elif "-50bps" in str(scenario.shocks):
                impact_factor = 1.025  # +2.5% for 50bps down

        elif scenario.category == "fx":
            # FX shock impact
            if "10%" in str(scenario.shocks):
                impact_factor = 0.97  # -3% impact

        elif scenario.category == "equity":
            # Equity shock impact
            if "-20%" in str(scenario.shocks):
                impact_factor = 0.85  # -15% impact
            elif "+15%" in str(scenario.shocks):
                impact_factor = 1.12  # +12% impact
            elif "-35%" in str(scenario.shocks):
                impact_factor = 0.75  # -25% impact

        elif scenario.category == "volatility":
            # Volatility impact (options portfolio)
            if "+50%" in str(scenario.shocks):
                impact_factor = 1.08  # +8% for vol spike
            elif "-30%" in str(scenario.shocks):
                impact_factor = 0.94  # -6% for vol crush

        elif scenario.category == "credit":
            # Credit spread impact
            if "+200bps" in str(scenario.shocks):
                impact_factor = 0.92  # -8% for severe widening

        elif scenario.category == "combined":
            # Extreme scenarios
            if "2008" in scenario.name:
                impact_factor = 0.65  # -35% for 2008 crisis
            elif "COVID" in scenario.name:
                impact_factor = 0.70  # -30% for COVID
            elif "PerfectStorm" in scenario.name:
                impact_factor = 0.55  # -45% for perfect storm
            elif "Stagflation" in scenario.name:
                impact_factor = 0.80  # -20% for stagflation

        return baseline_mtm * impact_factor

    def load_custom_scenarios(self, file_path: Optional[Path] = None) -> int:
        """Load custom scenarios from YAML or JSON file.

        Args:
            file_path: Path to scenario file. If None, loads all scenarios from scenario_dir

        Returns:
            Number of scenarios loaded
        """
        loaded_count = 0

        if file_path:
            files_to_load = [Path(file_path)]
        else:
            # Load all YAML and JSON files from scenario directory
            files_to_load = list(self.scenario_dir.glob("*.yaml")) + list(
                self.scenario_dir.glob("*.yml")
            ) + list(self.scenario_dir.glob("*.json"))

        for file in files_to_load:
            try:
                with open(file, "r") as f:
                    if file.suffix in [".yaml", ".yml"]:
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)

                # Handle both single scenario and multiple scenarios
                scenarios_data = data if isinstance(data, list) else [data]

                for scenario_data in scenarios_data:
                    scenario = StressScenario(
                        name=scenario_data["name"],
                        description=scenario_data["description"],
                        category=scenario_data.get("category", "custom"),
                        shocks=scenario_data["shocks"],
                        severity=scenario_data.get("severity", "moderate"),
                    )
                    self.custom_scenarios.append(scenario)
                    loaded_count += 1

                print(f"✓ Loaded {len(scenarios_data)} scenario(s) from {file.name}")

            except Exception as e:
                print(f"✗ Failed to load scenarios from {file.name}: {e}")

        return loaded_count

    def save_custom_scenario(self, scenario: StressScenario, file_name: Optional[str] = None):
        """Save a custom scenario to file.

        Args:
            scenario: Scenario to save
            file_name: Optional file name. If None, uses scenario name
        """
        if not file_name:
            file_name = f"{scenario.name.lower().replace(' ', '_')}.yaml"

        file_path = self.scenario_dir / file_name
        scenario_dict = asdict(scenario)

        with open(file_path, "w") as f:
            yaml.dump(scenario_dict, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Scenario saved to {file_path}")

    def create_custom_scenario(
        self,
        name: str,
        description: str,
        shocks: Dict[str, Any],
        category: str = "custom",
        severity: str = "moderate",
        save: bool = True,
    ) -> StressScenario:
        """Create a custom stress scenario.

        Args:
            name: Scenario name
            description: Scenario description
            shocks: Dictionary of market shocks
            category: Scenario category
            severity: Severity level (mild/moderate/severe/extreme)
            save: Whether to save the scenario to file

        Returns:
            Created StressScenario

        Example:
            >>> tester.create_custom_scenario(
            ...     name="Custom_Crisis",
            ...     description="My custom crisis scenario",
            ...     shocks={
            ...         "USD_curve": "+200bps",
            ...         "EUR_curve": "+150bps",
            ...         "SPX": "-25%",
            ...         "FX_vols": "+75%"
            ...     },
            ...     severity="severe"
            ... )
        """
        scenario = StressScenario(
            name=name,
            description=description,
            category=category,
            shocks=shocks,
            severity=severity,
        )

        self.custom_scenarios.append(scenario)

        if save:
            self.save_custom_scenario(scenario)

        return scenario

    def get_all_scenarios(self) -> List[StressScenario]:
        """Get all scenarios (predefined + custom).

        Returns:
            Combined list of all scenarios
        """
        return self.scenarios + self.custom_scenarios

    def run_scenario(self, scenario: StressScenario, portfolio: Any, book_hierarchy: Any) -> Dict:
        """Run a single stress scenario.

        Args:
            scenario: Scenario to run
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object

        Returns:
            Scenario result dictionary
        """
        baseline_summary = get_portfolio_summary(portfolio, book_hierarchy)
        baseline_mtm = baseline_summary["total_mtm"]

        stressed_mtm = self._simulate_stress_impact(baseline_mtm, scenario, baseline_summary)
        pnl_impact = stressed_mtm - baseline_mtm
        pnl_pct = (pnl_impact / baseline_mtm * 100) if baseline_mtm != 0 else 0

        return {
            "scenario": scenario.name,
            "category": scenario.category,
            "severity": scenario.severity,
            "description": scenario.description,
            "baseline_mtm": baseline_mtm,
            "stressed_mtm": stressed_mtm,
            "pnl_impact": pnl_impact,
            "pnl_impact_pct": pnl_pct,
            "shocks": scenario.shocks,
        }

    def run_custom_stress_tests(
        self, portfolio: Any, book_hierarchy: Any, include_predefined: bool = False
    ) -> Dict[str, Dict]:
        """Run only custom stress scenarios.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object
            include_predefined: Whether to also include predefined scenarios

        Returns:
            Dictionary of stress test results
        """
        scenarios_to_run = (
            self.get_all_scenarios() if include_predefined else self.custom_scenarios
        )

        if not scenarios_to_run:
            print("No custom scenarios to run. Create some first!")
            return {}

        print(f"Running {len(scenarios_to_run)} stress scenario(s)...")
        print()

        results = {}
        baseline_summary = get_portfolio_summary(portfolio, book_hierarchy)
        baseline_mtm = baseline_summary["total_mtm"]

        for i, scenario in enumerate(scenarios_to_run, 1):
            print(f"[{i}/{len(scenarios_to_run)}] Running: {scenario.name}")
            print(f"  Category: {scenario.category.upper()}")
            print(f"  Severity: {scenario.severity.upper()}")

            result = self.run_scenario(scenario, portfolio, book_hierarchy)
            results[scenario.name] = result

            print(f"  P&L Impact: ${result['pnl_impact']:,.2f} ({result['pnl_impact_pct']:+.2f}%)")
            print()

        return results

    def analyze_shock_sensitivity(
        self, portfolio: Any, book_hierarchy: Any, shock_type: str, shock_range: List[str]
    ) -> pd.DataFrame:
        """Analyze portfolio sensitivity to a range of shock values.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object
            shock_type: Type of shock (e.g., 'USD_curve', 'SPX', 'EURUSD')
            shock_range: List of shock values (e.g., ['-100bps', '0', '+100bps', '+200bps'])

        Returns:
            DataFrame with sensitivity analysis results
        """
        print(f"Analyzing sensitivity to {shock_type} shocks...")
        print()

        results = []
        baseline_summary = get_portfolio_summary(portfolio, book_hierarchy)
        baseline_mtm = baseline_summary["total_mtm"]

        for shock_value in shock_range:
            scenario = StressScenario(
                name=f"Sensitivity_{shock_type}_{shock_value}",
                description=f"{shock_type} shock: {shock_value}",
                category="sensitivity",
                shocks={shock_type: shock_value},
                severity="mild",
            )

            stressed_mtm = self._simulate_stress_impact(baseline_mtm, scenario, baseline_summary)
            pnl_impact = stressed_mtm - baseline_mtm
            pnl_pct = (pnl_impact / baseline_mtm * 100) if baseline_mtm != 0 else 0

            results.append({
                "shock_type": shock_type,
                "shock_value": shock_value,
                "baseline_mtm": baseline_mtm,
                "stressed_mtm": stressed_mtm,
                "pnl_impact": pnl_impact,
                "pnl_impact_pct": pnl_pct,
            })

            print(f"  {shock_value:>10}: P&L Impact = ${pnl_impact:>12,.2f} ({pnl_pct:>+6.2f}%)")

        print()
        return pd.DataFrame(results)

    def compare_scenarios(
        self, portfolio: Any, book_hierarchy: Any, scenario_names: List[str]
    ) -> pd.DataFrame:
        """Compare multiple scenarios side-by-side.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object
            scenario_names: List of scenario names to compare

        Returns:
            DataFrame with comparison results
        """
        all_scenarios = {s.name: s for s in self.get_all_scenarios()}
        results = []

        for name in scenario_names:
            if name not in all_scenarios:
                print(f"Warning: Scenario '{name}' not found. Skipping.")
                continue

            scenario = all_scenarios[name]
            result = self.run_scenario(scenario, portfolio, book_hierarchy)
            results.append(result)

        return pd.DataFrame(results)

    def generate_stress_report(self, results: Dict[str, Dict]) -> Path:
        """Generate comprehensive stress test report.

        Args:
            results: Stress test results

        Returns:
            Path to generated report
        """
        print("Generating stress test report...")

        # Create DataFrame
        df = pd.DataFrame.from_dict(results, orient="index")
        df = df.sort_values("pnl_impact")

        # Generate summary statistics
        summary_stats = {
            "total_scenarios": len(results),
            "categories": df["category"].nunique(),
            "severity_breakdown": df["severity"].value_counts().to_dict(),
            "worst_scenario": df.iloc[0]["scenario"],
            "worst_impact": df.iloc[0]["pnl_impact"],
            "worst_impact_pct": df.iloc[0]["pnl_impact_pct"],
            "best_scenario": df.iloc[-1]["scenario"],
            "best_impact": df.iloc[-1]["pnl_impact"],
            "best_impact_pct": df.iloc[-1]["pnl_impact_pct"],
            "average_impact": df["pnl_impact"].mean(),
            "average_impact_pct": df["pnl_impact_pct"].mean(),
        }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON report
        json_file = self.output_dir / f"stress_test_results_{timestamp}.json"
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "stress_testing",
            },
            "summary_statistics": summary_stats,
            "scenario_results": results,
        }

        with open(json_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # CSV report
        csv_file = self.output_dir / f"stress_test_results_{timestamp}.csv"
        df.to_csv(csv_file)

        # Excel report with multiple sheets
        excel_file = self.output_dir / f"stress_test_results_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            # Summary sheet
            summary_df = pd.DataFrame.from_dict(summary_stats, orient="index", columns=["Value"])
            summary_df.to_excel(writer, sheet_name="Summary")

            # All results
            df.to_excel(writer, sheet_name="All Scenarios")

            # By category
            for category in df["category"].unique():
                cat_df = df[df["category"] == category]
                cat_df.to_excel(writer, sheet_name=f"{category.upper()}")

        print(f"✓ Stress test reports generated:")
        print(f"  JSON: {json_file.name}")
        print(f"  CSV: {csv_file.name}")
        print(f"  Excel: {excel_file.name}")

        return json_file

    def print_summary(self, results: Dict[str, Dict]):
        """Print stress test summary to console.

        Args:
            results: Stress test results
        """
        print("=" * 80)
        print("STRESS TEST SUMMARY")
        print("=" * 80)
        print()

        df = pd.DataFrame.from_dict(results, orient="index")
        df = df.sort_values("pnl_impact")

        # Top 5 worst scenarios
        print("Top 5 Worst Scenarios:")
        print("-" * 80)
        print(f"{'Scenario':<35} {'Category':<12} {'P&L Impact':<15} {'Impact %':<10}")
        print("-" * 80)

        for idx, row in df.head(5).iterrows():
            print(
                f"{row['scenario']:<35} {row['category']:<12} ${row['pnl_impact']:>13,.0f} {row['pnl_impact_pct']:>8.2f}%"
            )
        print()

        # By category
        print("Impact by Category:")
        print("-" * 80)
        category_summary = df.groupby("category")["pnl_impact"].agg(
            ["mean", "min", "max", "count"]
        )
        print(category_summary)
        print()

        # By severity
        print("Impact by Severity:")
        print("-" * 80)
        severity_summary = df.groupby("severity")["pnl_impact"].agg(
            ["mean", "min", "max", "count"]
        )
        print(severity_summary)
        print()


def main():
    """Main entry point."""
    print("=" * 80)
    print("Fictional Bank Portfolio - Stress Testing Framework")
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

    # Run stress tests
    results = tester.run_all_stress_tests(portfolio, book_hierarchy)

    # Generate reports
    report_file = tester.generate_stress_report(results)
    print()

    # Print summary
    tester.print_summary(results)

    print("=" * 80)
    print("Stress Testing Complete!")
    print("=" * 80)
    print()
    print("Note: This is a simplified demonstration using simulated stress impacts.")
    print("In production, stress tests would recalculate portfolio MTM using")
    print("stressed market data via the Neutryx API for accurate results.")
    print()


if __name__ == "__main__":
    main()
