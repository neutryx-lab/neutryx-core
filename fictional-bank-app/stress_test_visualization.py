#!/usr/bin/env python3
"""Advanced stress test visualizations.

This module provides comprehensive visualization tools for stress testing results:
- Heatmaps showing impact across scenarios and risk factors
- Tornado charts for sensitivity analysis
- Scenario comparison charts
- Interactive dashboards
- Risk factor correlation analysis
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive charts will be disabled.")

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10


class StressTestVisualizer:
    """Create advanced visualizations for stress test results."""

    def __init__(self, output_dir: Path):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_all_visualizations(
        self,
        results: Dict[str, Dict],
        create_interactive: bool = True,
    ) -> Dict[str, Path]:
        """Create all stress test visualizations.

        Args:
            results: Stress test results dictionary
            create_interactive: Whether to create interactive Plotly charts

        Returns:
            Dictionary mapping chart name to file path
        """
        print("Generating stress test visualizations...")
        print()

        charts = {}

        # Static charts (matplotlib)
        chart = self.plot_impact_heatmap(results)
        charts["impact_heatmap"] = chart
        print("✓ Impact heatmap")

        chart = self.plot_tornado_chart(results)
        charts["tornado_chart"] = chart
        print("✓ Tornado chart")

        chart = self.plot_scenario_comparison(results)
        charts["scenario_comparison"] = chart
        print("✓ Scenario comparison")

        chart = self.plot_severity_distribution(results)
        charts["severity_distribution"] = chart
        print("✓ Severity distribution")

        chart = self.plot_category_impact(results)
        charts["category_impact"] = chart
        print("✓ Category impact")

        # Interactive charts (Plotly)
        if create_interactive and PLOTLY_AVAILABLE:
            chart = self.create_interactive_dashboard(results)
            charts["interactive_dashboard"] = chart
            print("✓ Interactive dashboard (Plotly)")

            chart = self.create_3d_risk_surface(results)
            charts["3d_risk_surface"] = chart
            print("✓ 3D risk surface")

        print()
        print(f"All stress test visualizations saved to: {self.output_dir}")
        return charts

    def plot_impact_heatmap(self, results: Dict[str, Dict]) -> Path:
        """Create a heatmap showing P&L impact across scenarios.

        Args:
            results: Stress test results

        Returns:
            Path to saved chart
        """
        df = pd.DataFrame.from_dict(results, orient="index")

        # Create pivot table for heatmap
        # Group by category and scenario
        df_sorted = df.sort_values(["category", "pnl_impact"])

        fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.4)))

        # Create heatmap data
        heatmap_data = df_sorted[["pnl_impact_pct"]].T

        # Create heatmap
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(df_sorted)))
        ax.set_xticklabels(df_sorted["scenario"], rotation=45, ha="right")
        ax.set_yticks([0])
        ax.set_yticklabels(["P&L Impact %"])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
        cbar.set_label("P&L Impact (%)", fontweight="bold")

        # Add values on heatmap
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            val = row["pnl_impact_pct"]
            text_color = "white" if abs(val) > 15 else "black"
            ax.text(i, 0, f"{val:.1f}%", ha="center", va="center",
                   color=text_color, fontweight="bold", fontsize=9)

        # Add category separators
        categories = df_sorted["category"].values
        for i in range(1, len(categories)):
            if categories[i] != categories[i-1]:
                ax.axvline(x=i-0.5, color="black", linewidth=2)

        ax.set_title("Stress Test Impact Heatmap", fontweight="bold", fontsize=14, pad=20)

        plt.tight_layout()
        output_file = self.output_dir / "stress_impact_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_tornado_chart(self, results: Dict[str, Dict], top_n: int = 15) -> Path:
        """Create a tornado chart showing top scenarios by impact.

        Args:
            results: Stress test results
            top_n: Number of top scenarios to display

        Returns:
            Path to saved chart
        """
        df = pd.DataFrame.from_dict(results, orient="index")
        df = df.sort_values("pnl_impact", key=abs, ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        # Separate positive and negative impacts
        colors = ["#27ae60" if x >= 0 else "#e74c3c" for x in df["pnl_impact"]]

        # Create horizontal bar chart
        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df["pnl_impact"], color=colors, alpha=0.7, edgecolor="black")

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["scenario"])
        ax.set_xlabel("P&L Impact (USD)", fontweight="bold", fontsize=12)
        ax.set_title(f"Top {top_n} Stress Scenarios by Impact (Tornado Chart)",
                    fontweight="bold", fontsize=14, pad=20)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df["pnl_impact"])):
            pct = df["pnl_impact_pct"].iloc[i]
            label = f"${val:,.0f} ({pct:+.1f}%)"
            x_pos = val + (df["pnl_impact"].abs().max() * 0.02)
            if val < 0:
                x_pos = val - (df["pnl_impact"].abs().max() * 0.02)
                ha = "right"
            else:
                ha = "left"
            ax.text(x_pos, i, label, va="center", ha=ha, fontsize=8, fontweight="bold")

        plt.tight_layout()
        output_file = self.output_dir / "stress_tornado_chart.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_scenario_comparison(self, results: Dict[str, Dict]) -> Path:
        """Create a grouped bar chart comparing scenarios by category.

        Args:
            results: Stress test results

        Returns:
            Path to saved chart
        """
        df = pd.DataFrame.from_dict(results, orient="index")

        # Group by category
        categories = df["category"].unique()
        fig, axes = plt.subplots(
            len(categories), 1, figsize=(14, 5 * len(categories)), squeeze=False
        )

        for idx, category in enumerate(categories):
            ax = axes[idx, 0]
            cat_df = df[df["category"] == category].sort_values("pnl_impact")

            # Color by severity
            severity_colors = {
                "mild": "#3498db",
                "moderate": "#f39c12",
                "severe": "#e74c3c",
                "extreme": "#c0392b",
            }
            colors = [severity_colors.get(s, "#95a5a6") for s in cat_df["severity"]]

            # Create bar chart
            y_pos = np.arange(len(cat_df))
            bars = ax.barh(y_pos, cat_df["pnl_impact"], color=colors, alpha=0.7, edgecolor="black")

            # Customize
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cat_df["scenario"])
            ax.set_xlabel("P&L Impact (USD)", fontweight="bold", fontsize=11)
            ax.set_title(f"{category.upper()} Scenarios", fontweight="bold", fontsize=12, pad=15)
            ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
            ax.grid(True, alpha=0.3, axis="x")

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, cat_df["pnl_impact"])):
                pct = cat_df["pnl_impact_pct"].iloc[i]
                label = f"${val:,.0f}\n({pct:+.1f}%)"
                x_pos = val + (cat_df["pnl_impact"].abs().max() * 0.02 if val >= 0 else -cat_df["pnl_impact"].abs().max() * 0.02)
                ha = "left" if val >= 0 else "right"
                ax.text(x_pos, i, label, va="center", ha=ha, fontsize=8)

        plt.tight_layout()
        output_file = self.output_dir / "stress_scenario_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_severity_distribution(self, results: Dict[str, Dict]) -> Path:
        """Create charts showing distribution by severity.

        Args:
            results: Stress test results

        Returns:
            Path to saved chart
        """
        df = pd.DataFrame.from_dict(results, orient="index")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Pie chart - count by severity
        severity_counts = df["severity"].value_counts()
        severity_colors = {
            "mild": "#3498db",
            "moderate": "#f39c12",
            "severe": "#e74c3c",
            "extreme": "#c0392b",
        }
        colors = [severity_colors.get(s, "#95a5a6") for s in severity_counts.index]

        ax1.pie(
            severity_counts.values,
            labels=severity_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 11, "fontweight": "bold"},
        )
        ax1.set_title("Scenario Count by Severity", fontweight="bold", fontsize=12, pad=20)

        # Box plot - impact by severity
        severity_order = ["mild", "moderate", "severe", "extreme"]
        severity_data = [
            df[df["severity"] == s]["pnl_impact"].values
            for s in severity_order
            if s in df["severity"].values
        ]
        severity_labels = [s for s in severity_order if s in df["severity"].values]
        colors_for_box = [severity_colors[s] for s in severity_labels]

        bp = ax2.boxplot(
            severity_data,
            labels=severity_labels,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="yellow", markersize=8),
        )

        # Color boxes
        for patch, color in zip(bp["boxes"], colors_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel("P&L Impact (USD)", fontweight="bold", fontsize=11)
        ax2.set_xlabel("Severity Level", fontweight="bold", fontsize=11)
        ax2.set_title("P&L Impact Distribution by Severity", fontweight="bold", fontsize=12, pad=20)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)

        plt.tight_layout()
        output_file = self.output_dir / "stress_severity_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_category_impact(self, results: Dict[str, Dict]) -> Path:
        """Create summary chart showing average impact by category.

        Args:
            results: Stress test results

        Returns:
            Path to saved chart
        """
        df = pd.DataFrame.from_dict(results, orient="index")

        # Calculate statistics by category
        category_stats = df.groupby("category")["pnl_impact"].agg(["mean", "min", "max", "count"])
        category_stats = category_stats.sort_values("mean")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Bar chart with error bars
        categories = category_stats.index
        means = category_stats["mean"]
        mins = category_stats["min"]
        maxs = category_stats["max"]

        y_pos = np.arange(len(categories))
        colors_list = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))

        bars = ax1.barh(y_pos, means, color=colors_list, alpha=0.7, edgecolor="black")

        # Add min/max range markers
        for i, (cat, mean, min_val, max_val) in enumerate(zip(categories, means, mins, maxs)):
            ax1.plot([min_val, max_val], [i, i], "k-", linewidth=2, alpha=0.5)
            ax1.plot([min_val], [i], "ko", markersize=6)
            ax1.plot([max_val], [i], "ko", markersize=6)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(categories)
        ax1.set_xlabel("P&L Impact (USD)", fontweight="bold", fontsize=11)
        ax1.set_title("Average Impact by Category (with range)", fontweight="bold", fontsize=12, pad=20)
        ax1.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax1.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, means)):
            label = f"${val:,.0f}"
            x_pos = val + (means.abs().max() * 0.02 if val >= 0 else -means.abs().max() * 0.02)
            ha = "left" if val >= 0 else "right"
            ax1.text(x_pos, i, label, va="center", ha=ha, fontsize=9, fontweight="bold")

        # Stacked bar showing scenario counts
        ax2.barh(
            y_pos,
            category_stats["count"],
            color=colors_list,
            alpha=0.7,
            edgecolor="black",
        )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(categories)
        ax2.set_xlabel("Number of Scenarios", fontweight="bold", fontsize=11)
        ax2.set_title("Scenario Count by Category", fontweight="bold", fontsize=12, pad=20)
        ax2.grid(True, alpha=0.3, axis="x")

        # Add count labels
        for i, count in enumerate(category_stats["count"]):
            ax2.text(count + 0.1, i, f"{int(count)}", va="center", fontsize=10, fontweight="bold")

        plt.tight_layout()
        output_file = self.output_dir / "stress_category_impact.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def create_interactive_dashboard(self, results: Dict[str, Dict]) -> Optional[Path]:
        """Create an interactive Plotly dashboard.

        Args:
            results: Stress test results

        Returns:
            Path to saved HTML file, or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        df = pd.DataFrame.from_dict(results, orient="index")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "P&L Impact by Scenario",
                "Impact by Category",
                "Severity Distribution",
                "Impact vs Baseline MTM",
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "pie"}, {"type": "scatter"}],
            ],
        )

        # 1. Bar chart - scenarios by impact
        df_sorted = df.sort_values("pnl_impact")
        fig.add_trace(
            go.Bar(
                x=df_sorted["pnl_impact"],
                y=df_sorted["scenario"],
                orientation="h",
                marker=dict(
                    color=df_sorted["pnl_impact"],
                    colorscale="RdYlGn",
                    showscale=False,
                ),
                text=[f"{x:+.1f}%" for x in df_sorted["pnl_impact_pct"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Impact: $%{x:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 2. Box plot - by category
        categories = df["category"].unique()
        for category in categories:
            cat_data = df[df["category"] == category]
            fig.add_trace(
                go.Box(
                    y=cat_data["pnl_impact"],
                    name=category,
                    boxmean="sd",
                ),
                row=1,
                col=2,
            )

        # 3. Pie chart - severity distribution
        severity_counts = df["severity"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                hole=0.3,
            ),
            row=2,
            col=1,
        )

        # 4. Scatter plot - impact vs baseline
        fig.add_trace(
            go.Scatter(
                x=df["baseline_mtm"],
                y=df["pnl_impact"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=df["pnl_impact_pct"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Impact %", x=1.15),
                ),
                text=df["scenario"],
                hovertemplate="<b>%{text}</b><br>Baseline: $%{x:,.0f}<br>Impact: $%{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="P&L Impact (USD)", row=1, col=1)
        fig.update_yaxes(title_text="P&L Impact (USD)", row=1, col=2)
        fig.update_xaxes(title_text="Baseline MTM (USD)", row=2, col=2)
        fig.update_yaxes(title_text="P&L Impact (USD)", row=2, col=2)

        fig.update_layout(
            title_text="Stress Test Interactive Dashboard",
            height=900,
            showlegend=True,
        )

        output_file = self.output_dir / "stress_interactive_dashboard.html"
        fig.write_html(str(output_file))

        return output_file

    def create_3d_risk_surface(self, results: Dict[str, Dict]) -> Optional[Path]:
        """Create a 3D surface plot of risk factors.

        Args:
            results: Stress test results

        Returns:
            Path to saved HTML file, or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        df = pd.DataFrame.from_dict(results, orient="index")

        # Create 3D scatter plot
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=df["baseline_mtm"],
                    y=df["pnl_impact_pct"],
                    z=df["stressed_mtm"],
                    mode="markers+text",
                    marker=dict(
                        size=8,
                        color=df["pnl_impact"],
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="P&L Impact (USD)"),
                    ),
                    text=df["scenario"],
                    textposition="top center",
                    hovertemplate="<b>%{text}</b><br>"
                    + "Baseline: $%{x:,.0f}<br>"
                    + "Impact: %{y:.1f}%<br>"
                    + "Stressed: $%{z:,.0f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title="3D Stress Test Risk Surface",
            scene=dict(
                xaxis_title="Baseline MTM (USD)",
                yaxis_title="Impact (%)",
                zaxis_title="Stressed MTM (USD)",
            ),
            height=800,
        )

        output_file = self.output_dir / "stress_3d_risk_surface.html"
        fig.write_html(str(output_file))

        return output_file

    def plot_sensitivity_analysis(
        self, sensitivity_df: pd.DataFrame, shock_type: str
    ) -> Path:
        """Create visualization for sensitivity analysis results.

        Args:
            sensitivity_df: DataFrame from analyze_shock_sensitivity
            shock_type: Type of shock being analyzed

        Returns:
            Path to saved chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Line plot showing sensitivity curve
        x_values = range(len(sensitivity_df))
        ax1.plot(
            x_values,
            sensitivity_df["pnl_impact"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="#3498db",
        )
        ax1.fill_between(
            x_values,
            sensitivity_df["pnl_impact"],
            0,
            alpha=0.3,
            color="#3498db",
        )
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax1.set_xticks(x_values)
        ax1.set_xticklabels(sensitivity_df["shock_value"], rotation=45)
        ax1.set_xlabel(f"{shock_type} Shock", fontweight="bold", fontsize=11)
        ax1.set_ylabel("P&L Impact (USD)", fontweight="bold", fontsize=11)
        ax1.set_title(f"Sensitivity to {shock_type}", fontweight="bold", fontsize=12, pad=15)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, val in enumerate(sensitivity_df["pnl_impact"]):
            ax1.text(
                i, val, f"${val:,.0f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9
            )

        # Bar chart showing percentage impact
        colors = ["#27ae60" if x >= 0 else "#e74c3c" for x in sensitivity_df["pnl_impact_pct"]]
        ax2.bar(x_values, sensitivity_df["pnl_impact_pct"], color=colors, alpha=0.7, edgecolor="black")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.set_xticks(x_values)
        ax2.set_xticklabels(sensitivity_df["shock_value"], rotation=45)
        ax2.set_xlabel(f"{shock_type} Shock", fontweight="bold", fontsize=11)
        ax2.set_ylabel("P&L Impact (%)", fontweight="bold", fontsize=11)
        ax2.set_title(f"Percentage Impact - {shock_type}", fontweight="bold", fontsize=12, pad=15)
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for i, val in enumerate(sensitivity_df["pnl_impact_pct"]):
            ax2.text(
                i, val, f"{val:+.1f}%", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9, fontweight="bold"
            )

        plt.tight_layout()
        output_file = self.output_dir / f"sensitivity_{shock_type.lower()}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file


def main():
    """Demo of stress test visualization."""
    # This would be called from stress_testing.py with actual results
    print("This module is designed to be imported and used with stress test results.")
    print("Run stress_testing.py to see it in action.")


if __name__ == "__main__":
    main()
