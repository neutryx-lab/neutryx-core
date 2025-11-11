#!/usr/bin/env python3
"""Interactive XVA Dashboard using Plotly

Creates an interactive HTML dashboard with:
- XVA waterfall charts
- Exposure profile time series
- Counterparty comparison
- CSA impact analysis
- Interactive drill-down capabilities
"""
import json
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


class InteractiveXVADashboard:
    """Create interactive Plotly dashboards for XVA analysis."""

    def __init__(self, xva_results: List):
        """Initialize dashboard creator.

        Args:
            xva_results: List of XVA results from standalone calculator
        """
        self.results = xva_results

    def create_waterfall_chart(self) -> go.Figure:
        """Create interactive waterfall chart."""
        # Aggregate portfolio-level XVA
        total_cva = sum(r.cva for r in self.results)
        total_dva = sum(r.dva for r in self.results)
        total_fva = sum(r.fva for r in self.results)
        total_mva = sum(r.mva for r in self.results)
        total_xva = sum(r.total_xva for r in self.results)

        fig = go.Figure(go.Waterfall(
            name="XVA",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["CVA", "DVA<br>(Benefit)", "FVA", "MVA", "Total XVA"],
            textposition="outside",
            text=[f"${total_cva:,.0f}", f"-${total_dva:,.0f}",
                  f"${total_fva:,.0f}", f"${total_mva:,.0f}", f"${total_xva:,.0f}"],
            y=[total_cva, -total_dva, total_fva, total_mva, total_xva],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#2ecc71"}},
            increasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}},
        ))

        fig.update_layout(
            title="Portfolio XVA Waterfall",
            showlegend=False,
            height=600,
            font=dict(size=12),
            yaxis_title="Amount (USD)",
            hovermode="x unified"
        )

        return fig

    def create_exposure_profiles_chart(self) -> go.Figure:
        """Create interactive exposure profiles chart."""
        fig = go.Figure()

        for result in self.results:
            label = result.counterparty_name[:30]
            if result.has_csa:
                label += " (CSA)"

            fig.add_trace(go.Scatter(
                x=result.time_grid,
                y=result.exposure_profile,
                mode='lines',
                name=label,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Time: %{x:.2f} years<br>' +
                             'Exposure: $%{y:,.0f}<br>' +
                             '<extra></extra>'
            ))

        fig.update_layout(
            title="Expected Positive Exposure Profiles",
            xaxis_title="Time (Years)",
            yaxis_title="Expected Positive Exposure (USD)",
            height=600,
            hovermode="x unified",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    def create_xva_by_counterparty_chart(self) -> go.Figure:
        """Create stacked bar chart of XVA by counterparty."""
        counterparties = [r.counterparty_name[:25] for r in self.results]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='CVA',
            x=counterparties,
            y=[r.cva for r in self.results],
            marker_color='#e74c3c',
            hovertemplate='<b>CVA</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='DVA (Benefit)',
            x=counterparties,
            y=[-r.dva for r in self.results],
            marker_color='#2ecc71',
            hovertemplate='<b>DVA</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='FVA',
            x=counterparties,
            y=[r.fva for r in self.results],
            marker_color='#f39c12',
            hovertemplate='<b>FVA</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='MVA',
            x=counterparties,
            y=[r.mva for r in self.results],
            marker_color='#9b59b6',
            hovertemplate='<b>MVA</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="XVA Components by Counterparty",
            xaxis_title="Counterparty",
            yaxis_title="XVA Amount (USD)",
            barmode='stack',
            height=600,
            hovermode="x unified"
        )

        return fig

    def create_csa_comparison_chart(self) -> go.Figure:
        """Create CSA impact comparison chart."""
        csa_results = [r for r in self.results if r.has_csa]
        non_csa_results = [r for r in self.results if not r.has_csa]

        if not csa_results or not non_csa_results:
            return None

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Total XVA", "Average Exposure"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Total XVA
        csa_total = sum(r.total_xva for r in csa_results)
        non_csa_total = sum(r.total_xva for r in non_csa_results)

        fig.add_trace(
            go.Bar(
                x=["With CSA", "Without CSA"],
                y=[csa_total, non_csa_total],
                marker_color=["#2ecc71", "#e74c3c"],
                text=[f"${csa_total:,.0f}", f"${non_csa_total:,.0f}"],
                textposition='outside',
                name="Total XVA",
                hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Average exposure
        csa_avg_exp = sum(r.expected_exposure for r in csa_results) / len(csa_results)
        non_csa_avg_exp = sum(r.expected_exposure for r in non_csa_results) / len(non_csa_results)

        fig.add_trace(
            go.Bar(
                x=["With CSA", "Without CSA"],
                y=[csa_avg_exp, non_csa_avg_exp],
                marker_color=["#2ecc71", "#e74c3c"],
                text=[f"${csa_avg_exp:,.0f}", f"${non_csa_avg_exp:,.0f}"],
                textposition='outside',
                name="Avg Exposure",
                hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="XVA (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Exposure (USD)", row=1, col=2)

        fig.update_layout(
            title_text="CSA Impact Analysis",
            showlegend=False,
            height=500
        )

        return fig

    def create_xva_heatmap(self) -> go.Figure:
        """Create heatmap of XVA components."""
        counterparties = [r.counterparty_name[:20] for r in self.results]
        components = ["CVA", "DVA", "FVA", "MVA"]

        # Create data matrix
        data_matrix = [
            [r.cva for r in self.results],
            [r.dva for r in self.results],
            [r.fva for r in self.results],
            [r.mva for r in self.results],
        ]

        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=counterparties,
            y=components,
            colorscale="RdYlGn_r",
            text=[[f"${val:,.0f}" for val in row] for row in data_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b><br>' +
                         '%{x}<br>' +
                         'Value: $%{z:,.2f}<br>' +
                         '<extra></extra>',
            colorbar=dict(title="USD")
        ))

        fig.update_layout(
            title="XVA Components Heatmap",
            xaxis_title="Counterparty",
            yaxis_title="XVA Component",
            height=500
        )

        return fig

    def create_summary_metrics(self) -> go.Figure:
        """Create summary metrics indicators."""
        total_cva = sum(r.cva for r in self.results)
        total_dva = sum(r.dva for r in self.results)
        total_fva = sum(r.fva for r in self.results)
        total_mva = sum(r.mva for r in self.results)
        total_xva = sum(r.total_xva for r in self.results)

        # Create indicator subplot
        fig = make_subplots(
            rows=1, cols=5,
            specs=[[{"type": "indicator"}, {"type": "indicator"},
                   {"type": "indicator"}, {"type": "indicator"},
                   {"type": "indicator"}]],
            subplot_titles=("CVA", "DVA", "FVA", "MVA", "Total XVA")
        )

        fig.add_trace(go.Indicator(
            mode="number",
            value=total_cva,
            number={'prefix': "$", 'valueformat': ",.0f"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode="number",
            value=total_dva,
            number={'prefix': "$", 'valueformat': ",.0f"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode="number",
            value=total_fva,
            number={'prefix': "$", 'valueformat': ",.0f"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ), row=1, col=3)

        fig.add_trace(go.Indicator(
            mode="number",
            value=total_mva,
            number={'prefix': "$", 'valueformat': ",.0f"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ), row=1, col=4)

        fig.add_trace(go.Indicator(
            mode="number",
            value=total_xva,
            number={'prefix': "$", 'valueformat': ",.0f"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ), row=1, col=5)

        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        return fig

    def create_full_dashboard(self, output_file: Path):
        """Create comprehensive interactive dashboard.

        Args:
            output_file: Path to save HTML dashboard
        """
        print("Creating interactive XVA dashboard...")

        # Create all charts
        metrics_fig = self.create_summary_metrics()
        waterfall_fig = self.create_waterfall_chart()
        exposure_fig = self.create_exposure_profiles_chart()
        counterparty_fig = self.create_xva_by_counterparty_chart()
        csa_fig = self.create_csa_comparison_chart()
        heatmap_fig = self.create_xva_heatmap()

        # Combine into single HTML
        html_parts = []

        # Header
        html_parts.append("""
        <html>
        <head>
            <title>XVA Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .chart-container {
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 {
                    margin: 0;
                }
                .subtitle {
                    margin: 5px 0 0 0;
                    font-size: 14px;
                    opacity: 0.9;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏦 XVA Analysis Dashboard</h1>
                <p class="subtitle">Interactive Portfolio XVA Analytics | Generated: """ +
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
        """)

        # Add each chart
        for title, fig in [
            ("Summary Metrics", metrics_fig),
            ("XVA Waterfall", waterfall_fig),
            ("Exposure Profiles", exposure_fig),
            ("XVA by Counterparty", counterparty_fig),
            ("CSA Impact", csa_fig),
            ("XVA Heatmap", heatmap_fig)
        ]:
            if fig:
                html_parts.append(f'<div class="chart-container">')
                html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
                html_parts.append('</div>')

        html_parts.append("""
            <div class="header" style="margin-top: 30px;">
                <p style="margin: 0; text-align: center; font-size: 12px;">
                    Generated with Neutryx Risk Analytics Platform |
                    Standalone XVA Calculator
                </p>
            </div>
        </body>
        </html>
        """)

        # Write to file
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        print(f"✓ Dashboard saved: {output_file.name}")
        print(f"  Open in browser: file://{output_file.absolute()}")

        return output_file


def main():
    """Main entry point - load results and create dashboard."""
    print("=" * 80)
    print("Interactive XVA Dashboard Generator")
    print("=" * 80)
    print()

    # Load latest XVA results
    reports_dir = Path(__file__).parent / "reports"
    xva_files = list(reports_dir.glob("xva_analysis_*.json"))

    if not xva_files:
        print("Error: No XVA analysis results found!")
        print("Please run standalone_xva_demo.py first to generate results.")
        sys.exit(1)

    latest_file = max(xva_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading XVA results from: {latest_file.name}")

    with open(latest_file, "r") as f:
        data = json.load(f)

    print(f"✓ Loaded results for {data['portfolio_summary']['num_netting_sets']} netting sets")
    print()

    # For dashboard, we need to reconstruct XVAResult objects
    # This is a simplified version - in production, we'd serialize/deserialize properly
    from standalone_xva_demo import XVAResult
    import numpy as np

    results = []
    for ns in data["netting_sets"]:
        # Create dummy exposure profile for visualization
        time_grid = np.linspace(0, 5, 61)
        exposure_profile = np.maximum(0, ns["expected_exposure"] * np.exp(-0.1 * time_grid))

        result = XVAResult(
            netting_set_id=ns["netting_set_id"],
            counterparty_name=ns["counterparty"],
            has_csa=ns["has_csa"],
            num_trades=ns["num_trades"],
            gross_notional=ns["gross_notional"],
            net_mtm=ns["net_mtm"],
            expected_exposure=ns["expected_exposure"],
            peak_exposure=ns["peak_exposure"],
            exposure_profile=exposure_profile,
            time_grid=time_grid,
            cva=ns["cva"],
            dva=ns["dva"],
            fva=ns["fva"],
            mva=ns["mva"],
            total_xva=ns["total_xva"]
        )
        results.append(result)

    # Create dashboard
    dashboard = InteractiveXVADashboard(results)

    output_file = Path(__file__).parent / "sample_outputs" / "xva_dashboard.html"
    dashboard.create_full_dashboard(output_file)

    print()
    print("=" * 80)
    print("Dashboard Generation Complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
