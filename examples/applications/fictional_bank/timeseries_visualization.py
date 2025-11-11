#!/usr/bin/env python3
"""Time Series Visualization

Create comprehensive visualizations for time series analysis:
- Historical trends with moving averages
- Volatility charts
- Forecast charts with confidence intervals
- Exposure evolution by counterparty
- Interactive dashboards
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from timeseries_analysis import TimeSeriesData, ForecastResult

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


class TimeSeriesVisualizer:
    """Create time series visualizations."""

    def __init__(self, output_dir: Path):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_all_charts(
        self,
        mtm_history: TimeSeriesData,
        forecast: ForecastResult,
        moving_averages: Dict[int, List[float]],
        rolling_vol: List[float],
        exposure_history: Optional[Dict[str, TimeSeriesData]] = None,
        create_interactive: bool = True
    ) -> Dict[str, Path]:
        """Create all time series charts.

        Args:
            mtm_history: Historical MTM data
            forecast: Forecast results
            moving_averages: Moving averages by window size
            rolling_vol: Rolling volatility
            exposure_history: Optional exposure history by counterparty
            create_interactive: Whether to create interactive charts

        Returns:
            Dictionary mapping chart name to file path
        """
        print("Generating time series visualizations...")
        print()

        charts = {}

        # Static charts
        chart = self.plot_trend_with_ma(mtm_history, moving_averages)
        charts["trend_with_ma"] = chart
        print("✓ Trend with moving averages")

        chart = self.plot_volatility(mtm_history, rolling_vol)
        charts["volatility"] = chart
        print("✓ Volatility chart")

        chart = self.plot_forecast(mtm_history, forecast)
        charts["forecast"] = chart
        print("✓ Forecast chart")

        if exposure_history:
            chart = self.plot_exposure_evolution(exposure_history)
            charts["exposure_evolution"] = chart
            print("✓ Exposure evolution")

        # Interactive dashboard
        if create_interactive and PLOTLY_AVAILABLE:
            chart = self.create_interactive_dashboard(
                mtm_history, forecast, moving_averages, exposure_history
            )
            charts["interactive_dashboard"] = chart
            print("✓ Interactive dashboard")

        print()
        print(f"All charts saved to: {self.output_dir}")
        return charts

    def plot_trend_with_ma(
        self,
        data: TimeSeriesData,
        moving_averages: Dict[int, List[float]]
    ) -> Path:
        """Plot trend with moving averages.

        Args:
            data: Time series data
            moving_averages: Moving averages by window

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot actual values
        ax.plot(data.dates, data.values, label="Actual", linewidth=2, color="#2c3e50", alpha=0.8)

        # Plot moving averages
        colors = ["#3498db", "#e74c3c", "#f39c12"]
        for i, (window, ma_values) in enumerate(sorted(moving_averages.items())):
            ax.plot(data.dates, ma_values, label=f"{window}-day MA",
                   linewidth=1.5, color=colors[i % len(colors)], alpha=0.7)

        ax.set_xlabel("Date", fontweight="bold", fontsize=12)
        ax.set_ylabel(f"{data.name} ({data.units})", fontweight="bold", fontsize=12)
        ax.set_title(f"{data.name} - Trend with Moving Averages",
                    fontweight="bold", fontsize=14, pad=20)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_file = self.output_dir / "timeseries_trend_ma.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_volatility(
        self,
        data: TimeSeriesData,
        rolling_vol: List[float],
        window: int = 20
    ) -> Path:
        """Plot rolling volatility.

        Args:
            data: Time series data
            rolling_vol: Rolling volatility values
            window: Window size used

        Returns:
            Path to saved chart
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top: Price
        ax1.plot(data.dates, data.values, color="#2c3e50", linewidth=1.5)
        ax1.set_ylabel(f"{data.name} ({data.units})", fontweight="bold", fontsize=11)
        ax1.set_title(f"{data.name} and Rolling Volatility",
                     fontweight="bold", fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)

        # Bottom: Volatility
        ax2.plot(data.dates, rolling_vol, color="#e74c3c", linewidth=1.5)
        ax2.fill_between(data.dates, rolling_vol, alpha=0.3, color="#e74c3c")
        ax2.set_xlabel("Date", fontweight="bold", fontsize=11)
        ax2.set_ylabel(f"Volatility ({window}-day)", fontweight="bold", fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_file = self.output_dir / "timeseries_volatility.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_forecast(
        self,
        history: TimeSeriesData,
        forecast: ForecastResult,
        num_historical: int = 60
    ) -> Path:
        """Plot forecast with confidence intervals.

        Args:
            history: Historical data
            forecast: Forecast results
            num_historical: Number of historical days to show

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot recent historical data
        hist_dates = history.dates[-num_historical:]
        hist_values = history.values[-num_historical:]
        ax.plot(hist_dates, hist_values, label="Historical",
               linewidth=2, color="#2c3e50")

        # Plot forecast mean
        ax.plot(forecast.dates, forecast.mean, label="Forecast (Mean)",
               linewidth=2, color="#3498db", linestyle="--")

        # Plot confidence intervals
        ax.fill_between(forecast.dates, forecast.percentile_5, forecast.percentile_95,
                       alpha=0.3, color="#3498db", label="90% Confidence")
        ax.fill_between(forecast.dates, forecast.percentile_1, forecast.percentile_99,
                       alpha=0.15, color="#3498db", label="98% Confidence")

        # Plot some sample scenarios
        for i, scenario in enumerate(forecast.scenarios[:10]):
            ax.plot(forecast.dates, scenario, alpha=0.1, color="#95a5a6", linewidth=0.5)

        ax.set_xlabel("Date", fontweight="bold", fontsize=12)
        ax.set_ylabel(f"{history.name} ({history.units})", fontweight="bold", fontsize=12)
        ax.set_title(f"{history.name} - Monte Carlo Forecast",
                    fontweight="bold", fontsize=14, pad=20)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add vertical line at forecast start
        ax.axvline(x=history.dates[-1], color="red", linestyle=":", linewidth=2, alpha=0.7)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_file = self.output_dir / "timeseries_forecast.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_exposure_evolution(
        self,
        exposure_history: Dict[str, TimeSeriesData],
        top_n: int = 5
    ) -> Path:
        """Plot exposure evolution for top counterparties.

        Args:
            exposure_history: Exposure history by counterparty
            top_n: Number of top counterparties to show

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get top N by final exposure
        sorted_cps = sorted(
            exposure_history.items(),
            key=lambda x: x[1].values[-1],
            reverse=True
        )[:top_n]

        # Plot each counterparty
        colors = plt.cm.tab10(np.linspace(0, 1, top_n))

        for i, (cp_id, data) in enumerate(sorted_cps):
            ax.plot(data.dates, data.values, label=data.name,
                   linewidth=2, color=colors[i], alpha=0.8)

        ax.set_xlabel("Date", fontweight="bold", fontsize=12)
        ax.set_ylabel(f"Exposure ({sorted_cps[0][1].units})", fontweight="bold", fontsize=12)
        ax.set_title(f"Top {top_n} Counterparty Exposures - Evolution",
                    fontweight="bold", fontsize=14, pad=20)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_file = self.output_dir / "timeseries_exposure_evolution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def create_interactive_dashboard(
        self,
        mtm_history: TimeSeriesData,
        forecast: ForecastResult,
        moving_averages: Dict[int, List[float]],
        exposure_history: Optional[Dict[str, TimeSeriesData]] = None
    ) -> Optional[Path]:
        """Create interactive Plotly dashboard.

        Args:
            mtm_history: Historical MTM data
            forecast: Forecast results
            moving_averages: Moving averages
            exposure_history: Optional exposure history

        Returns:
            Path to HTML file, or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Historical MTM with Moving Averages",
                "Monte Carlo Forecast",
                "Exposure Evolution" if exposure_history else "MTM Distribution",
                "Returns Distribution"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ]
        )

        # 1. Historical with MAs
        fig.add_trace(
            go.Scatter(x=mtm_history.dates, y=mtm_history.values,
                      mode="lines", name="Actual", line=dict(color="#2c3e50", width=2)),
            row=1, col=1
        )

        colors_ma = ["#3498db", "#e74c3c", "#f39c12"]
        for i, (window, ma) in enumerate(sorted(moving_averages.items())):
            fig.add_trace(
                go.Scatter(x=mtm_history.dates, y=ma,
                          mode="lines", name=f"{window}-day MA",
                          line=dict(color=colors_ma[i], width=1.5, dash="dash")),
                row=1, col=1
            )

        # 2. Forecast
        fig.add_trace(
            go.Scatter(x=forecast.dates, y=forecast.mean,
                      mode="lines", name="Forecast", line=dict(color="#3498db", width=2)),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=forecast.dates + forecast.dates[::-1],
                      y=forecast.percentile_95 + forecast.percentile_5[::-1],
                      fill="toself", fillcolor="rgba(52, 152, 219, 0.3)",
                      line=dict(width=0), name="90% CI", showlegend=False),
            row=1, col=2
        )

        # 3. Exposure evolution or MTM distribution
        if exposure_history:
            sorted_cps = sorted(
                exposure_history.items(),
                key=lambda x: x[1].values[-1],
                reverse=True
            )[:5]

            for cp_id, data in sorted_cps:
                fig.add_trace(
                    go.Scatter(x=data.dates, y=data.values,
                              mode="lines", name=data.name[:20]),
                    row=2, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(x=mtm_history.dates, y=mtm_history.values,
                          mode="lines", name="MTM", line=dict(color="#2c3e50")),
                row=2, col=1
            )

        # 4. Returns distribution
        returns = np.diff(np.log(mtm_history.values))
        fig.add_trace(
            go.Histogram(x=returns * 100, name="Returns", marker_color="#3498db",
                        nbinsx=50),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Return (%)", row=2, col=2)

        fig.update_yaxes(title_text=f"{mtm_history.name} ({mtm_history.units})", row=1, col=1)
        fig.update_yaxes(title_text=f"{mtm_history.name} ({mtm_history.units})", row=1, col=2)
        fig.update_yaxes(title_text="Exposure", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        fig.update_layout(
            title_text="Time Series Analysis Dashboard",
            height=900,
            showlegend=True
        )

        output_file = self.output_dir / "timeseries_interactive_dashboard.html"
        fig.write_html(str(output_file))

        return output_file


def main():
    """Demo of time series visualization."""
    print("This module is designed to be imported and used with time series data.")
    print("Run timeseries_demo.py to see it in action.")


if __name__ == "__main__":
    main()
