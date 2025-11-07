#!/usr/bin/env python3
"""Generate comprehensive visualizations for portfolio analysis.

This script creates various charts and plots:
- XVA waterfall charts
- Counterparty risk heatmaps
- Exposure profiles over time
- Notional breakdown by desk/asset class
- CSA vs non-CSA comparison
- Interactive Plotly dashboards
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


class PortfolioVisualizer:
    """Generate visualizations for portfolio analysis."""

    def __init__(self, output_dir: Path):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save chart files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_all_charts(
        self,
        portfolio: Any,
        book_hierarchy: Any,
        xva_results: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """Create all visualization charts.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object
            xva_results: Optional XVA calculation results

        Returns:
            Dictionary mapping chart name to file path
        """
        print("Generating portfolio visualizations...")
        print()

        summary = get_portfolio_summary(portfolio, book_hierarchy)
        charts = {}

        # Portfolio composition charts
        chart = self.plot_counterparty_exposure(summary)
        charts["counterparty_exposure"] = chart
        print(f"✓ Counterparty exposure chart")

        chart = self.plot_desk_breakdown(summary)
        charts["desk_breakdown"] = chart
        print(f"✓ Desk breakdown chart")

        chart = self.plot_mtm_distribution(summary)
        charts["mtm_distribution"] = chart
        print(f"✓ MTM distribution chart")

        chart = self.plot_csa_comparison(summary)
        charts["csa_comparison"] = chart
        print(f"✓ CSA comparison chart")

        chart = self.plot_rating_distribution(summary)
        charts["rating_distribution"] = chart
        print(f"✓ Rating distribution chart")

        # XVA charts (if available)
        if xva_results and "netting_set_xva" in xva_results:
            chart = self.plot_xva_waterfall(xva_results)
            charts["xva_waterfall"] = chart
            print(f"✓ XVA waterfall chart")

            chart = self.plot_xva_by_counterparty(xva_results)
            charts["xva_by_counterparty"] = chart
            print(f"✓ XVA by counterparty chart")

            chart = self.plot_csa_impact(xva_results)
            charts["csa_impact"] = chart
            print(f"✓ CSA impact chart")

        print()
        print(f"All charts saved to: {self.output_dir}")
        return charts

    def plot_counterparty_exposure(self, summary: Dict) -> Path:
        """Create counterparty exposure bar chart."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Prepare data
        counterparties = []
        exposures = []
        colors = []

        for cp_id, cp_info in summary["counterparties"].items():
            counterparties.append(cp_info["name"][:30])
            exposures.append(cp_info["net_mtm"])
            # Color by MTM sign
            colors.append("#2ecc71" if cp_info["net_mtm"] >= 0 else "#e74c3c")

        # Create bar chart
        y_pos = np.arange(len(counterparties))
        bars = ax.barh(y_pos, exposures, color=colors, alpha=0.7)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(counterparties)
        ax.set_xlabel("Net MTM (USD)", fontweight="bold", fontsize=12)
        ax.set_title("Net Exposure by Counterparty", fontweight="bold", fontsize=14, pad=20)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, exposures)):
            label = f"${val:,.0f}"
            x_pos = val + (max(exposures) - min(exposures)) * 0.02
            if val < 0:
                x_pos = val - (max(exposures) - min(exposures)) * 0.02
                ha = "right"
            else:
                ha = "left"
            ax.text(x_pos, i, label, va="center", ha=ha, fontsize=9)

        plt.tight_layout()
        output_file = self.output_dir / "counterparty_exposure.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_desk_breakdown(self, summary: Dict) -> Path:
        """Create desk breakdown pie chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Pie chart by number of trades
        desks = []
        trades = []
        mtms = []

        for desk_id, desk_info in summary["desks"].items():
            desks.append(desk_info["name"])
            trades.append(desk_info["num_trades"])
            mtms.append(abs(desk_info["total_mtm"]))

        # Trades pie
        colors = sns.color_palette("husl", len(desks))
        ax1.pie(
            trades,
            labels=desks,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 10},
        )
        ax1.set_title("Trade Distribution by Desk", fontweight="bold", fontsize=12, pad=20)

        # MTM bar chart
        y_pos = np.arange(len(desks))
        bars = ax2.barh(y_pos, mtms, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(desks)
        ax2.set_xlabel("Absolute MTM (USD)", fontweight="bold", fontsize=11)
        ax2.set_title("MTM by Desk", fontweight="bold", fontsize=12, pad=20)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, mtms)):
            ax2.text(val + max(mtms) * 0.02, i, f"${val:,.0f}", va="center", fontsize=9)

        plt.tight_layout()
        output_file = self.output_dir / "desk_breakdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_mtm_distribution(self, summary: Dict) -> Path:
        """Create MTM distribution histogram."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Collect MTMs
        mtms = [cp_info["net_mtm"] for cp_info in summary["counterparties"].values()]

        # Create histogram
        n, bins, patches = ax.hist(mtms, bins=15, alpha=0.7, color="#3498db", edgecolor="black")

        # Color bars by value
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor("#e74c3c")
            else:
                patch.set_facecolor("#2ecc71")

        ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Zero MTM")
        ax.set_xlabel("Net MTM (USD)", fontweight="bold", fontsize=12)
        ax.set_ylabel("Frequency", fontweight="bold", fontsize=12)
        ax.set_title("Distribution of Net MTM by Counterparty", fontweight="bold", fontsize=14, pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "mtm_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_csa_comparison(self, summary: Dict) -> Path:
        """Create CSA vs non-CSA comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Separate CSA and non-CSA counterparties
        csa_count = sum(1 for cp in summary["counterparties"].values() if cp["has_csa"])
        non_csa_count = len(summary["counterparties"]) - csa_count

        csa_trades = sum(
            cp["num_trades"] for cp in summary["counterparties"].values() if cp["has_csa"]
        )
        non_csa_trades = sum(
            cp["num_trades"] for cp in summary["counterparties"].values() if not cp["has_csa"]
        )

        csa_notional = sum(
            cp["gross_notional"] for cp in summary["counterparties"].values() if cp["has_csa"]
        )
        non_csa_notional = sum(
            cp["gross_notional"] for cp in summary["counterparties"].values() if not cp["has_csa"]
        )

        # Bar chart comparing counts
        categories = ["Counterparties", "Trades", "Notional ($M)"]
        csa_values = [csa_count, csa_trades, csa_notional / 1_000_000]
        non_csa_values = [non_csa_count, non_csa_trades, non_csa_notional / 1_000_000]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, csa_values, width, label="With CSA", color="#2ecc71", alpha=0.8)
        bars2 = ax1.bar(
            x + width / 2, non_csa_values, width, label="Without CSA", color="#e74c3c", alpha=0.8
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.set_ylabel("Count / Value", fontweight="bold", fontsize=11)
        ax1.set_title("CSA vs Non-CSA Comparison", fontweight="bold", fontsize=12, pad=20)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Pie chart for CSA coverage
        ax2.pie(
            [csa_count, non_csa_count],
            labels=["With CSA", "Without CSA"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#2ecc71", "#e74c3c"],
            textprops={"fontsize": 11, "fontweight": "bold"},
        )
        ax2.set_title(
            "CSA Coverage by Counterparty Count", fontweight="bold", fontsize=12, pad=20
        )

        plt.tight_layout()
        output_file = self.output_dir / "csa_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_rating_distribution(self, summary: Dict) -> Path:
        """Create rating distribution chart."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Collect ratings
        ratings = {}
        for cp_info in summary["counterparties"].values():
            rating = cp_info["rating"]
            ratings[rating] = ratings.get(rating, 0) + 1

        # Sort ratings
        rating_order = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]
        sorted_ratings = sorted(
            ratings.items(), key=lambda x: rating_order.index(x[0]) if x[0] in rating_order else 99
        )

        labels, counts = zip(*sorted_ratings) if sorted_ratings else ([], [])

        # Create bar chart
        colors_map = {
            "AAA": "#27ae60",
            "AA+": "#2ecc71",
            "AA": "#3498db",
            "AA-": "#5dade2",
            "A+": "#f39c12",
            "A": "#f1c40f",
            "A-": "#e67e22",
            "BBB+": "#e74c3c",
            "BBB": "#c0392b",
            "BBB-": "#922b21",
        }
        bar_colors = [colors_map.get(r, "#95a5a6") for r in labels]

        bars = ax.bar(range(len(labels)), counts, color=bar_colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Credit Rating", fontweight="bold", fontsize=12)
        ax.set_ylabel("Number of Counterparties", fontweight="bold", fontsize=12)
        ax.set_title("Counterparty Credit Rating Distribution", fontweight="bold", fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        output_file = self.output_dir / "rating_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_xva_waterfall(self, xva_results: Dict) -> Path:
        """Create XVA waterfall chart."""
        fig, ax = plt.subplots(figsize=(12, 8))

        if "portfolio_xva" in xva_results:
            portfolio_xva = xva_results["portfolio_xva"]
            components = ["CVA", "DVA", "FVA", "MVA", "Total XVA"]
            values = [
                portfolio_xva.get("cva", 0),
                portfolio_xva.get("dva", 0),
                portfolio_xva.get("fva", 0),
                portfolio_xva.get("mva", 0),
                portfolio_xva.get("total_xva", 0),
            ]
        else:
            # Aggregate from netting sets
            ns_xva = xva_results["netting_set_xva"]
            components = ["CVA", "DVA", "FVA", "MVA", "Total XVA"]
            values = [
                sum(ns.get("cva", 0) for ns in ns_xva),
                sum(ns.get("dva", 0) for ns in ns_xva),
                sum(ns.get("fva", 0) for ns in ns_xva),
                sum(ns.get("mva", 0) for ns in ns_xva),
                sum(ns.get("total_xva", 0) for ns in ns_xva),
            ]

        # Create waterfall
        x = np.arange(len(components))
        colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#2ecc71"]
        colors[-1] = "#1abc9c"  # Different color for total

        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_ylabel("Amount (USD)", fontweight="bold", fontsize=12)
        ax.set_title("XVA Components Breakdown", fontweight="bold", fontsize=14, pad=20)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            label = f"${val:,.0f}"
            y_pos = val + (max(values) - min(values)) * 0.02
            ax.text(i, y_pos, label, ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        output_file = self.output_dir / "xva_waterfall.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_xva_by_counterparty(self, xva_results: Dict) -> Path:
        """Create XVA breakdown by counterparty."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Prepare data
        ns_xva = xva_results["netting_set_xva"]
        counterparties = [ns["counterparty"][:30] for ns in ns_xva]
        cva = [ns.get("cva", 0) for ns in ns_xva]
        dva = [ns.get("dva", 0) for ns in ns_xva]
        fva = [ns.get("fva", 0) for ns in ns_xva]
        mva = [ns.get("mva", 0) for ns in ns_xva]

        # Create stacked bar chart
        x = np.arange(len(counterparties))
        width = 0.6

        p1 = ax.bar(x, cva, width, label="CVA", color="#3498db", alpha=0.8)
        p2 = ax.bar(x, dva, width, bottom=cva, label="DVA", color="#e74c3c", alpha=0.8)
        p3 = ax.bar(x, fva, width, bottom=np.array(cva) + np.array(dva), label="FVA", color="#f39c12", alpha=0.8)
        p4 = ax.bar(
            x,
            mva,
            width,
            bottom=np.array(cva) + np.array(dva) + np.array(fva),
            label="MVA",
            color="#9b59b6",
            alpha=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(counterparties, rotation=45, ha="right")
        ax.set_ylabel("XVA Amount (USD)", fontweight="bold", fontsize=12)
        ax.set_title("XVA Breakdown by Counterparty", fontweight="bold", fontsize=14, pad=20)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_file = self.output_dir / "xva_by_counterparty.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_csa_impact(self, xva_results: Dict) -> Path:
        """Create CSA impact comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Separate CSA and non-CSA
        ns_xva = xva_results["netting_set_xva"]
        csa_xva = [ns.get("total_xva", 0) for ns in ns_xva if ns.get("has_csa", False)]
        non_csa_xva = [ns.get("total_xva", 0) for ns in ns_xva if not ns.get("has_csa", False)]

        # Create box plot
        data_to_plot = [csa_xva, non_csa_xva]
        labels = ["With CSA", "Without CSA"]
        colors = ["#2ecc71", "#e74c3c"]

        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="yellow", markersize=8),
        )

        # Color boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Total XVA (USD)", fontweight="bold", fontsize=12)
        ax.set_title("CSA Impact on XVA", fontweight="bold", fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics
        if csa_xva and non_csa_xva:
            avg_csa = np.mean(csa_xva)
            avg_non_csa = np.mean(non_csa_xva)
            reduction = (avg_non_csa - avg_csa) / avg_non_csa * 100 if avg_non_csa > 0 else 0

            textstr = f"Average XVA:\n  With CSA: ${avg_csa:,.0f}\n  Without CSA: ${avg_non_csa:,.0f}\n  Reduction: {reduction:.1f}%"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(
                0.02,
                0.98,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

        plt.tight_layout()
        output_file = self.output_dir / "csa_impact.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file


def main():
    """Main entry point."""
    print("=" * 80)
    print("Fictional Bank Portfolio - Visualization Generator")
    print("=" * 80)
    print()

    # Load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # Try to load XVA results if available
    xva_results = None
    xva_file = Path(__file__).parent / "reports" / "xva_results.json"

    if xva_file.exists():
        print(f"Loading XVA results from {xva_file}...")
        with open(xva_file, "r") as f:
            xva_results = json.load(f)
        print("✓ XVA results loaded")
        print()
    else:
        print("Note: XVA results not found. Some charts will be skipped.")
        print("      Run compute_xva.py first to generate XVA visualizations.")
        print()

    # Generate visualizations
    output_dir = Path(__file__).parent / "sample_outputs" / "charts"
    visualizer = PortfolioVisualizer(output_dir)

    charts = visualizer.create_all_charts(portfolio, book_hierarchy, xva_results)

    print()
    print("=" * 80)
    print("Visualization Generation Complete!")
    print("=" * 80)
    print()
    print(f"Generated {len(charts)} charts:")
    for chart_name, file_path in charts.items():
        print(f"  {chart_name}: {file_path.name}")
    print()


if __name__ == "__main__":
    main()
