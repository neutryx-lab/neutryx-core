#!/usr/bin/env python3
"""Compliance Dashboard - Interactive Visualizations

This module creates interactive compliance dashboards and visualizations:
- Real-time limit utilization gauges
- Alert status overview
- Breach timeline
- Counterparty/desk heat maps
- Regulatory reporting charts

Outputs:
- Interactive HTML dashboards (Plotly)
- Static PNG charts (Matplotlib)
- PDF reports
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
    print("Warning: Plotly not available. Interactive dashboards will be disabled.")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from compliance_monitoring import ComplianceReport, AlertLevel

# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


class ComplianceDashboard:
    """Create compliance monitoring dashboards and visualizations."""

    def __init__(self, output_dir: Path):
        """Initialize the dashboard generator.

        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme for alert levels
        self.colors = {
            AlertLevel.GREEN: "#27ae60",     # Green
            AlertLevel.YELLOW: "#f39c12",    # Yellow/Orange
            AlertLevel.RED: "#e74c3c",       # Red
            AlertLevel.BREACH: "#c0392b",    # Dark Red
        }

    def create_all_dashboards(
        self,
        report: ComplianceReport,
        create_interactive: bool = True
    ) -> Dict[str, Path]:
        """Create all compliance dashboards.

        Args:
            report: ComplianceReport object
            create_interactive: Whether to create interactive Plotly dashboards

        Returns:
            Dictionary mapping dashboard name to file path
        """
        print("Generating compliance dashboards...")
        print()

        dashboards = {}

        # Static dashboards (matplotlib)
        dashboard = self.plot_compliance_overview(report)
        dashboards["compliance_overview"] = dashboard
        print("✓ Compliance overview")

        dashboard = self.plot_limit_utilization(report)
        dashboards["limit_utilization"] = dashboard
        print("✓ Limit utilization chart")

        dashboard = self.plot_breach_analysis(report)
        dashboards["breach_analysis"] = dashboard
        print("✓ Breach analysis")

        dashboard = self.plot_scope_breakdown(report)
        dashboards["scope_breakdown"] = dashboard
        print("✓ Scope breakdown")

        # Interactive dashboard (Plotly)
        if create_interactive and PLOTLY_AVAILABLE:
            dashboard = self.create_interactive_dashboard(report)
            dashboards["interactive_dashboard"] = dashboard
            print("✓ Interactive dashboard (HTML)")

            dashboard = self.create_gauge_dashboard(report)
            dashboards["gauge_dashboard"] = dashboard
            print("✓ Gauge dashboard (HTML)")

        print()
        print(f"All dashboards saved to: {self.output_dir}")
        return dashboards

    def plot_compliance_overview(self, report: ComplianceReport) -> Path:
        """Create compliance overview dashboard.

        Args:
            report: ComplianceReport object

        Returns:
            Path to saved chart
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(
            f"Compliance Monitoring Dashboard - {report.portfolio_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98
        )

        # 1. Health Score Gauge (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        health_score = report.summary_stats['health_score']
        self._plot_health_gauge(ax1, health_score)

        # 2. Alert Status Breakdown (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        counts = [report.limits_ok, report.limits_warning, report.limits_critical, report.limits_breached]
        labels = ['OK', 'Warning', 'Critical', 'Breached']
        colors = [self.colors[AlertLevel.GREEN], self.colors[AlertLevel.YELLOW],
                 self.colors[AlertLevel.RED], self.colors[AlertLevel.BREACH]]
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title("Alert Status Distribution", fontweight="bold", pad=10)

        # 3. Key Metrics (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        metrics_text = f"""
KEY METRICS

Total Limits: {report.total_limits}

🟢 OK: {report.limits_ok}
🟡 Warning: {report.limits_warning}
🔴 Critical: {report.limits_critical}
🚨 Breached: {report.limits_breached}

Avg Utilization: {report.summary_stats['avg_utilization']:.1f}%
Max Utilization: {report.summary_stats['max_utilization']:.1f}%

Health Score: {health_score:.1f}%
        """
        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 4. Top Breaches/Warnings (middle row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        critical = [b for b in report.breaches
                   if b.alert_level in [AlertLevel.BREACH, AlertLevel.RED, AlertLevel.YELLOW]]
        critical = sorted(critical, key=lambda x: x.utilization_pct, reverse=True)[:10]

        if critical:
            limit_names = [b.limit_name[:30] for b in critical]
            utilizations = [b.utilization_pct for b in critical]
            bar_colors = [self.colors[b.alert_level] for b in critical]

            y_pos = np.arange(len(limit_names))
            bars = ax4.barh(y_pos, utilizations, color=bar_colors, alpha=0.7, edgecolor='black')

            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(limit_names, fontsize=9)
            ax4.set_xlabel("Utilization (%)", fontweight="bold")
            ax4.set_title("Top Limit Utilizations", fontweight="bold", pad=10)
            ax4.axvline(x=70, color='orange', linestyle='--', alpha=0.5, label='Warning (70%)')
            ax4.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='Critical (90%)')
            ax4.axvline(x=100, color='darkred', linestyle='-', alpha=0.7, linewidth=2, label='Limit (100%)')
            ax4.legend(loc='lower right', fontsize=8)
            ax4.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, utilizations)):
                ax4.text(val + 2, i, f"{val:.1f}%", va='center', fontsize=8, fontweight='bold')

        # 5. Utilization Distribution (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        all_utils = [b.utilization_pct for b in report.breaches]
        ax5.hist(all_utils, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax5.axvline(x=70, color='orange', linestyle='--', alpha=0.7, label='Warning')
        ax5.axvline(x=90, color='red', linestyle='--', alpha=0.7, label='Critical')
        ax5.axvline(x=100, color='darkred', linestyle='-', alpha=0.7, linewidth=2, label='Limit')
        ax5.set_xlabel("Utilization (%)", fontweight="bold")
        ax5.set_ylabel("Frequency", fontweight="bold")
        ax5.set_title("Utilization Distribution", fontweight="bold", pad=10)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. By Scope (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        scope_data = {}
        for breach in report.breaches:
            scope = breach.scope
            if scope not in scope_data:
                scope_data[scope] = {level: 0 for level in AlertLevel}
            scope_data[scope][breach.alert_level] += 1

        scopes = list(scope_data.keys())
        if scopes:
            x = np.arange(len(scopes))
            width = 0.2

            for i, level in enumerate([AlertLevel.GREEN, AlertLevel.YELLOW, AlertLevel.RED, AlertLevel.BREACH]):
                counts = [scope_data[s].get(level, 0) for s in scopes]
                offset = width * (i - 1.5)
                ax6.bar(x + offset, counts, width, label=level.value.title(),
                       color=self.colors[level], alpha=0.8)

            ax6.set_xticks(x)
            ax6.set_xticklabels([s.title() for s in scopes], rotation=15)
            ax6.set_ylabel("Count", fontweight="bold")
            ax6.set_title("Alerts by Scope", fontweight="bold", pad=10)
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3, axis='y')

        # 7. By Limit Type (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        type_data = {}
        for breach in report.breaches:
            ltype = breach.limit_type
            if ltype not in type_data:
                type_data[ltype] = 0
            type_data[ltype] += 1

        if type_data:
            types = list(type_data.keys())
            counts = list(type_data.values())
            ax7.barh(range(len(types)), counts, color='#3498db', alpha=0.7, edgecolor='black')
            ax7.set_yticks(range(len(types)))
            ax7.set_yticklabels([t.replace('_', ' ').title() for t in types], fontsize=9)
            ax7.set_xlabel("Count", fontweight="bold")
            ax7.set_title("Limits by Type", fontweight="bold", pad=10)
            ax7.grid(True, alpha=0.3, axis='x')

            # Add count labels
            for i, count in enumerate(counts):
                ax7.text(count + 0.1, i, str(count), va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / "compliance_overview.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def _plot_health_gauge(self, ax, health_score: float):
        """Plot health score gauge.

        Args:
            ax: Matplotlib axis
            health_score: Health score (0-100)
        """
        # Determine color
        if health_score >= 80:
            color = self.colors[AlertLevel.GREEN]
        elif health_score >= 60:
            color = self.colors[AlertLevel.YELLOW]
        else:
            color = self.colors[AlertLevel.RED]

        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background arc
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'gray', linewidth=15, alpha=0.3)

        # Colored arc for health score
        theta_health = np.linspace(0, np.pi * health_score / 100, 100)
        ax.plot(r * np.cos(theta_health), r * np.sin(theta_health), color=color, linewidth=15)

        # Text
        ax.text(0, -0.3, f"{health_score:.1f}%", ha='center', va='center',
               fontsize=20, fontweight='bold')
        ax.text(0, -0.5, "Health Score", ha='center', va='center', fontsize=12)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title("Compliance Health", fontweight="bold", pad=10)

    def plot_limit_utilization(self, report: ComplianceReport) -> Path:
        """Create detailed limit utilization chart.

        Args:
            report: ComplianceReport object

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, max(8, len(report.breaches) * 0.3)))

        # Sort by utilization
        breaches = sorted(report.breaches, key=lambda x: x.utilization_pct, reverse=True)

        limit_names = [b.limit_name[:40] for b in breaches]
        utilizations = [b.utilization_pct for b in breaches]
        colors_list = [self.colors[b.alert_level] for b in breaches]

        y_pos = np.arange(len(limit_names))
        bars = ax.barh(y_pos, utilizations, color=colors_list, alpha=0.7, edgecolor='black')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(limit_names, fontsize=8)
        ax.set_xlabel("Utilization (%)", fontweight="bold", fontsize=12)
        ax.set_title(f"Limit Utilization - {report.portfolio_name}", fontweight="bold", fontsize=14, pad=20)

        # Reference lines
        ax.axvline(x=70, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Warning (70%)')
        ax.axvline(x=90, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Critical (90%)')
        ax.axvline(x=100, color='darkred', linestyle='-', alpha=0.7, linewidth=3, label='Limit (100%)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

        # Value labels
        for i, (bar, val) in enumerate(zip(bars, utilizations)):
            label = f"{val:.1f}%"
            x_pos = min(val + 3, ax.get_xlim()[1] - 5)
            ax.text(x_pos, i, label, va='center', fontsize=7, fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / "compliance_limit_utilization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def plot_breach_analysis(self, report: ComplianceReport) -> Path:
        """Create breach analysis charts.

        Args:
            report: ComplianceReport object

        Returns:
            Path to saved chart
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        fig.suptitle(f"Breach Analysis - {report.portfolio_name}",
                    fontsize=14, fontweight='bold')

        # 1. Breaches by severity
        severity_counts = {
            'OK': report.limits_ok,
            'Warning': report.limits_warning,
            'Critical': report.limits_critical,
            'Breached': report.limits_breached,
        }

        ax1.bar(severity_counts.keys(), severity_counts.values(),
               color=[self.colors[AlertLevel.GREEN], self.colors[AlertLevel.YELLOW],
                     self.colors[AlertLevel.RED], self.colors[AlertLevel.BREACH]],
               alpha=0.8, edgecolor='black')
        ax1.set_ylabel("Count", fontweight='bold')
        ax1.set_title("Limits by Severity", fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, axis='y')

        for i, (k, v) in enumerate(severity_counts.items()):
            ax1.text(i, v + 0.1, str(v), ha='center', fontweight='bold', fontsize=11)

        # 2. Top breaches
        critical_breaches = [b for b in report.breaches
                            if b.alert_level in [AlertLevel.BREACH, AlertLevel.RED]]
        critical_breaches = sorted(critical_breaches, key=lambda x: x.utilization_pct, reverse=True)[:10]

        if critical_breaches:
            names = [b.limit_name[:25] for b in critical_breaches]
            utils = [b.utilization_pct for b in critical_breaches]
            bar_colors = [self.colors[b.alert_level] for b in critical_breaches]

            ax2.barh(range(len(names)), utils, color=bar_colors, alpha=0.7, edgecolor='black')
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=9)
            ax2.set_xlabel("Utilization (%)", fontweight='bold')
            ax2.set_title("Top Critical Limits", fontweight='bold', pad=10)
            ax2.axvline(x=100, color='darkred', linestyle='-', linewidth=2, alpha=0.7)
            ax2.grid(True, alpha=0.3, axis='x')

        # 3. Utilization ranges
        ranges = {'0-50%': 0, '50-70%': 0, '70-90%': 0, '90-100%': 0, '>100%': 0}
        for breach in report.breaches:
            util = breach.utilization_pct
            if util > 100:
                ranges['>100%'] += 1
            elif util > 90:
                ranges['90-100%'] += 1
            elif util > 70:
                ranges['70-90%'] += 1
            elif util > 50:
                ranges['50-70%'] += 1
            else:
                ranges['0-50%'] += 1

        range_colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
        ax3.bar(ranges.keys(), ranges.values(), color=range_colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel("Count", fontweight='bold')
        ax3.set_xlabel("Utilization Range", fontweight='bold')
        ax3.set_title("Limits by Utilization Range", fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, axis='y')

        for i, (k, v) in enumerate(ranges.items()):
            if v > 0:
                ax3.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

        # 4. Breach amounts (for actual breaches)
        actual_breaches = [b for b in report.breaches if b.breach_amount is not None]
        if actual_breaches:
            actual_breaches = sorted(actual_breaches, key=lambda x: x.breach_amount, reverse=True)[:10]
            names = [b.limit_name[:25] for b in actual_breaches]
            amounts = [b.breach_amount for b in actual_breaches]

            ax4.barh(range(len(names)), amounts, color=self.colors[AlertLevel.BREACH],
                    alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels(names, fontsize=9)
            ax4.set_xlabel("Breach Amount", fontweight='bold')
            ax4.set_title("Top Breach Amounts", fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3, axis='x')

            for i, amt in enumerate(amounts):
                ax4.text(amt + amt * 0.02, i, f"${amt:,.0f}", va='center', fontsize=8)
        else:
            ax4.text(0.5, 0.5, "No Limit Breaches", ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14, fontweight='bold', color='green')
            ax4.axis('off')

        plt.tight_layout()
        output_file = self.output_dir / "compliance_breach_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def plot_scope_breakdown(self, report: ComplianceReport) -> Path:
        """Create scope-based breakdown charts.

        Args:
            report: ComplianceReport object

        Returns:
            Path to saved chart
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        fig.suptitle(f"Scope Breakdown - {report.portfolio_name}",
                    fontsize=14, fontweight='bold')

        # Group by scope
        scope_groups = {}
        for breach in report.breaches:
            scope = breach.scope
            if scope not in scope_groups:
                scope_groups[scope] = []
            scope_groups[scope].append(breach)

        # 1. Count by scope
        scope_counts = {scope: len(breaches) for scope, breaches in scope_groups.items()}
        ax1.pie(scope_counts.values(), labels=[s.title() for s in scope_counts.keys()],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title("Limits by Scope", fontweight='bold', pad=10)

        # 2. Average utilization by scope
        scope_avg_util = {
            scope: np.mean([b.utilization_pct for b in breaches])
            for scope, breaches in scope_groups.items()
        }

        ax2.bar(scope_avg_util.keys(), scope_avg_util.values(),
               color='#3498db', alpha=0.7, edgecolor='black')
        ax2.axhline(y=70, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel("Average Utilization (%)", fontweight='bold')
        ax2.set_title("Average Utilization by Scope", fontweight='bold', pad=10)
        ax2.set_xticklabels([s.title() for s in scope_avg_util.keys()], rotation=15)
        ax2.grid(True, alpha=0.3, axis='y')

        for i, (k, v) in enumerate(scope_avg_util.items()):
            ax2.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

        # 3. Max utilization by scope
        scope_max_util = {
            scope: max([b.utilization_pct for b in breaches])
            for scope, breaches in scope_groups.items()
        }

        ax3.bar(scope_max_util.keys(), scope_max_util.values(),
               color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.axhline(y=100, color='darkred', linestyle='-', linewidth=2, alpha=0.7)
        ax3.set_ylabel("Maximum Utilization (%)", fontweight='bold')
        ax3.set_title("Max Utilization by Scope", fontweight='bold', pad=10)
        ax3.set_xticklabels([s.title() for s in scope_max_util.keys()], rotation=15)
        ax3.grid(True, alpha=0.3, axis='y')

        for i, (k, v) in enumerate(scope_max_util.items()):
            ax3.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

        # 4. Stacked bar by alert level
        x = np.arange(len(scope_groups))
        width = 0.6

        bottoms = np.zeros(len(scope_groups))
        for level in [AlertLevel.GREEN, AlertLevel.YELLOW, AlertLevel.RED, AlertLevel.BREACH]:
            counts = [
                sum(1 for b in breaches if b.alert_level == level)
                for breaches in scope_groups.values()
            ]
            ax4.bar(x, counts, width, label=level.value.title(),
                   color=self.colors[level], alpha=0.8, bottom=bottoms)
            bottoms += counts

        ax4.set_xticks(x)
        ax4.set_xticklabels([s.title() for s in scope_groups.keys()], rotation=15)
        ax4.set_ylabel("Count", fontweight='bold')
        ax4.set_title("Alert Levels by Scope", fontweight='bold', pad=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / "compliance_scope_breakdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def create_interactive_dashboard(self, report: ComplianceReport) -> Optional[Path]:
        """Create interactive Plotly dashboard.

        Args:
            report: ComplianceReport object

        Returns:
            Path to HTML file, or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Alert Status Distribution",
                "Limit Utilization",
                "Utilization Histogram",
                "By Scope",
                "By Limit Type",
                "Critical Limits",
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
            row_heights=[0.33, 0.33, 0.34],
        )

        # 1. Alert Status Pie Chart
        counts = [report.limits_ok, report.limits_warning, report.limits_critical, report.limits_breached]
        labels = ['OK', 'Warning', 'Critical', 'Breached']
        colors_list = [self.colors[AlertLevel.GREEN], self.colors[AlertLevel.YELLOW],
                      self.colors[AlertLevel.RED], self.colors[AlertLevel.BREACH]]

        fig.add_trace(
            go.Pie(labels=labels, values=counts, marker_colors=colors_list, hole=0.3),
            row=1, col=1
        )

        # 2. Top Limit Utilizations
        breaches_sorted = sorted(report.breaches, key=lambda x: x.utilization_pct, reverse=True)[:15]
        limit_names = [b.limit_name[:30] for b in breaches_sorted]
        utilizations = [b.utilization_pct for b in breaches_sorted]
        bar_colors = [self.colors[b.alert_level] for b in breaches_sorted]

        fig.add_trace(
            go.Bar(
                y=limit_names,
                x=utilizations,
                orientation='h',
                marker_color=bar_colors,
                text=[f"{u:.1f}%" for u in utilizations],
                textposition='outside',
                hovertemplate="<b>%{y}</b><br>Utilization: %{x:.1f}%<extra></extra>",
            ),
            row=1, col=2
        )

        # 3. Utilization Histogram
        all_utils = [b.utilization_pct for b in report.breaches]
        fig.add_trace(
            go.Histogram(x=all_utils, nbinsx=20, marker_color='#3498db'),
            row=2, col=1
        )

        # 4. By Scope
        scope_data = {}
        for breach in report.breaches:
            scope = breach.scope
            if scope not in scope_data:
                scope_data[scope] = 0
            scope_data[scope] += 1

        fig.add_trace(
            go.Bar(
                x=list(scope_data.keys()),
                y=list(scope_data.values()),
                marker_color='#9b59b6',
                text=list(scope_data.values()),
                textposition='outside',
            ),
            row=2, col=2
        )

        # 5. By Limit Type
        type_data = {}
        for breach in report.breaches:
            ltype = breach.limit_type.replace('_', ' ').title()
            if ltype not in type_data:
                type_data[ltype] = 0
            type_data[ltype] += 1

        fig.add_trace(
            go.Bar(
                x=list(type_data.keys()),
                y=list(type_data.values()),
                marker_color='#e67e22',
                text=list(type_data.values()),
                textposition='outside',
            ),
            row=3, col=1
        )

        # 6. Critical Limits
        critical = [b for b in report.breaches
                   if b.alert_level in [AlertLevel.BREACH, AlertLevel.RED]]
        critical = sorted(critical, key=lambda x: x.utilization_pct, reverse=True)[:10]

        if critical:
            fig.add_trace(
                go.Bar(
                    y=[b.limit_name[:25] for b in critical],
                    x=[b.utilization_pct for b in critical],
                    orientation='h',
                    marker_color=[self.colors[b.alert_level] for b in critical],
                    text=[f"{b.utilization_pct:.1f}%" for b in critical],
                    textposition='outside',
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_xaxes(title_text="Utilization (%)", row=1, col=2)
        fig.update_xaxes(title_text="Utilization (%)", row=2, col=1)
        fig.update_xaxes(title_text="Scope", row=2, col=2)
        fig.update_xaxes(title_text="Limit Type", row=3, col=1, tickangle=-45)
        fig.update_xaxes(title_text="Utilization (%)", row=3, col=2)

        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=1)

        fig.update_layout(
            title_text=f"Compliance Dashboard - {report.portfolio_name}",
            height=1200,
            showlegend=False,
        )

        output_file = self.output_dir / "compliance_interactive_dashboard.html"
        fig.write_html(str(output_file))

        return output_file

    def create_gauge_dashboard(self, report: ComplianceReport) -> Optional[Path]:
        """Create gauge-style dashboard with utilization indicators.

        Args:
            report: ComplianceReport object

        Returns:
            Path to HTML file, or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Select top limits to display as gauges
        top_limits = sorted(report.breaches, key=lambda x: x.utilization_pct, reverse=True)[:9]

        # Create subplots for gauges
        fig = make_subplots(
            rows=3,
            cols=3,
            specs=[[{"type": "indicator"}] * 3] * 3,
            subplot_titles=[b.limit_name[:30] for b in top_limits],
        )

        for i, breach in enumerate(top_limits):
            row = i // 3 + 1
            col = i % 3 + 1

            # Determine color based on utilization
            if breach.utilization_pct >= 100:
                gauge_color = self.colors[AlertLevel.BREACH]
            elif breach.utilization_pct >= 90:
                gauge_color = self.colors[AlertLevel.RED]
            elif breach.utilization_pct >= 70:
                gauge_color = self.colors[AlertLevel.YELLOW]
            else:
                gauge_color = self.colors[AlertLevel.GREEN]

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=breach.utilization_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{breach.limit_type.replace('_', ' ').title()}"},
                    delta={'reference': 70, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 120]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "lightyellow"},
                            {'range': [90, 100], 'color': "lightcoral"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 100}
                    }
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            title_text=f"Limit Utilization Gauges - {report.portfolio_name}",
            height=900,
        )

        output_file = self.output_dir / "compliance_gauge_dashboard.html"
        fig.write_html(str(output_file))

        return output_file


def main():
    """Demo of compliance dashboard generation."""
    print("This module is designed to be imported and used with compliance reports.")
    print("Run compliance_demo.py to see it in action.")


if __name__ == "__main__":
    main()
