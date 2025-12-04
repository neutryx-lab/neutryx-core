#!/usr/bin/env python3
"""Time Series Analysis for Portfolio Monitoring

This module provides comprehensive time series analysis capabilities:
- Historical portfolio MTM tracking
- Exposure evolution over time
- Trend analysis and forecasting
- Monte Carlo simulation for future projections
- Volatility analysis
- Performance metrics

Key Features:
- Historical data generation and tracking
- Rolling statistics (moving averages, volatility)
- Scenario-based simulations
- Confidence intervals and VaR
- Interactive visualizations
"""
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import get_portfolio_summary


@dataclass
class TimeSeriesData:
    """Time series data container."""
    dates: List[datetime]
    values: List[float]
    name: str
    units: str = "USD"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'date': self.dates,
            self.name: self.values
        }).set_index('date')

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'units': self.units,
            'data': [
                {'date': d.isoformat(), 'value': v}
                for d, v in zip(self.dates, self.values)
            ]
        }


@dataclass
class ForecastResult:
    """Forecast results container."""
    dates: List[datetime]
    mean: List[float]
    std: List[float]
    percentile_5: List[float]
    percentile_95: List[float]
    percentile_1: List[float]
    percentile_99: List[float]
    scenarios: List[List[float]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'date': self.dates,
            'mean': self.mean,
            'std': self.std,
            'p5': self.percentile_5,
            'p95': self.percentile_95,
            'p1': self.percentile_1,
            'p99': self.percentile_99,
        }).set_index('date')


class TimeSeriesAnalyzer:
    """Analyze and forecast portfolio time series."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the analyzer.

        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Historical data storage
        self.mtm_history: Optional[TimeSeriesData] = None
        self.exposure_history: Dict[str, TimeSeriesData] = {}

    def generate_historical_data(
        self,
        portfolio: Any,
        book_hierarchy: Any,
        num_days: int = 252,  # 1 year of business days
        start_date: Optional[datetime] = None,
        volatility: float = 0.02,  # 2% daily volatility
        drift: float = 0.0001,  # Small positive drift
        seed: Optional[int] = None
    ) -> TimeSeriesData:
        """Generate synthetic historical MTM data.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy
            num_days: Number of historical days to generate
            start_date: Start date (default: num_days before today)
            volatility: Daily volatility (annualized / sqrt(252))
            drift: Daily drift rate
            seed: Random seed for reproducibility

        Returns:
            TimeSeriesData with historical MTM
        """
        if seed is not None:
            np.random.seed(seed)

        # Get current portfolio MTM as starting point
        summary = get_portfolio_summary(portfolio, book_hierarchy)
        current_mtm = summary['total_mtm']

        # Generate dates
        if start_date is None:
            start_date = datetime.now() - timedelta(days=num_days)

        dates = [start_date + timedelta(days=i) for i in range(num_days)]

        # Generate returns using geometric Brownian motion
        daily_returns = np.random.normal(drift, volatility, num_days)

        # Calculate cumulative returns and MTM values
        cumulative_returns = np.exp(np.cumsum(daily_returns))

        # Work backwards from current MTM
        initial_mtm = current_mtm / cumulative_returns[-1]
        mtm_values = initial_mtm * cumulative_returns

        self.mtm_history = TimeSeriesData(
            dates=dates,
            values=mtm_values.tolist(),
            name="Portfolio MTM",
            units="USD"
        )

        return self.mtm_history

    def generate_exposure_history(
        self,
        portfolio: Any,
        book_hierarchy: Any,
        num_days: int = 252,
        start_date: Optional[datetime] = None,
        volatility: float = 0.015,
        seed: Optional[int] = None
    ) -> Dict[str, TimeSeriesData]:
        """Generate historical exposure data by counterparty.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy
            num_days: Number of historical days
            start_date: Start date
            volatility: Daily volatility
            seed: Random seed

        Returns:
            Dictionary of TimeSeriesData by counterparty
        """
        if seed is not None:
            np.random.seed(seed)

        summary = get_portfolio_summary(portfolio, book_hierarchy)

        if start_date is None:
            start_date = datetime.now() - timedelta(days=num_days)

        dates = [start_date + timedelta(days=i) for i in range(num_days)]

        for cp_id, cp_info in summary['counterparties'].items():
            current_exposure = abs(cp_info['net_mtm'])

            # Generate returns
            daily_returns = np.random.normal(0, volatility, num_days)
            cumulative_returns = np.exp(np.cumsum(daily_returns))

            # Work backwards
            initial_exposure = current_exposure / cumulative_returns[-1]
            exposure_values = initial_exposure * cumulative_returns

            self.exposure_history[cp_id] = TimeSeriesData(
                dates=dates,
                values=exposure_values.tolist(),
                name=f"{cp_info['name']} Exposure",
                units="USD"
            )

        return self.exposure_history

    def calculate_statistics(self, data: TimeSeriesData) -> Dict[str, float]:
        """Calculate statistical metrics for time series.

        Args:
            data: Time series data

        Returns:
            Dictionary of statistics
        """
        values = np.array(data.values)
        returns = np.diff(np.log(values))  # Log returns

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'current': float(values[-1]),
            'return_mean': float(np.mean(returns)),
            'return_std': float(np.std(returns)),
            'volatility_annual': float(np.std(returns) * np.sqrt(252)),  # Annualized
            'sharpe_ratio': float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0,
            'max_drawdown': float(self._calculate_max_drawdown(values)),
        }

    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown.

        Args:
            values: Array of values

        Returns:
            Maximum drawdown as percentage
        """
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        return float(np.min(drawdowns)) * 100  # As percentage

    def calculate_moving_averages(
        self,
        data: TimeSeriesData,
        windows: List[int] = [5, 20, 50]
    ) -> Dict[int, List[float]]:
        """Calculate moving averages.

        Args:
            data: Time series data
            windows: List of window sizes (in days)

        Returns:
            Dictionary mapping window size to moving average values
        """
        df = data.to_dataframe()
        mas = {}

        for window in windows:
            ma = df[data.name].rolling(window=window).mean()
            mas[window] = ma.tolist()

        return mas

    def calculate_rolling_volatility(
        self,
        data: TimeSeriesData,
        window: int = 20
    ) -> List[float]:
        """Calculate rolling volatility.

        Args:
            data: Time series data
            window: Rolling window size

        Returns:
            List of rolling volatility values
        """
        df = data.to_dataframe()
        returns = np.log(df[data.name] / df[data.name].shift(1))
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

        return rolling_vol.tolist()

    def forecast_montecarlo(
        self,
        data: TimeSeriesData,
        num_days: int = 60,  # ~3 months
        num_scenarios: int = 1000,
        volatility: Optional[float] = None,
        drift: Optional[float] = None,
        seed: Optional[int] = None
    ) -> ForecastResult:
        """Forecast using Monte Carlo simulation.

        Args:
            data: Historical time series data
            num_days: Number of days to forecast
            num_scenarios: Number of Monte Carlo scenarios
            volatility: Daily volatility (if None, estimated from data)
            drift: Daily drift rate (if None, estimated from data)
            seed: Random seed

        Returns:
            ForecastResult with mean, std, and percentiles
        """
        if seed is not None:
            np.random.seed(seed)

        # Estimate parameters from historical data if not provided
        values = np.array(data.values)
        returns = np.diff(np.log(values))

        if volatility is None:
            volatility = float(np.std(returns))

        if drift is None:
            drift = float(np.mean(returns))

        # Current value
        current_value = values[-1]

        # Generate forecast dates
        last_date = data.dates[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(num_days)]

        # Run Monte Carlo simulation
        scenarios = np.zeros((num_scenarios, num_days))

        for i in range(num_scenarios):
            daily_returns = np.random.normal(drift, volatility, num_days)
            cumulative_returns = np.exp(np.cumsum(daily_returns))
            scenarios[i, :] = current_value * cumulative_returns

        # Calculate statistics across scenarios
        mean_forecast = np.mean(scenarios, axis=0)
        std_forecast = np.std(scenarios, axis=0)
        p1_forecast = np.percentile(scenarios, 1, axis=0)
        p5_forecast = np.percentile(scenarios, 5, axis=0)
        p95_forecast = np.percentile(scenarios, 95, axis=0)
        p99_forecast = np.percentile(scenarios, 99, axis=0)

        return ForecastResult(
            dates=forecast_dates,
            mean=mean_forecast.tolist(),
            std=std_forecast.tolist(),
            percentile_1=p1_forecast.tolist(),
            percentile_5=p5_forecast.tolist(),
            percentile_95=p95_forecast.tolist(),
            percentile_99=p99_forecast.tolist(),
            scenarios=[scenarios[i, :].tolist() for i in range(min(100, num_scenarios))]  # Store first 100 scenarios
        )

    def calculate_var(
        self,
        data: TimeSeriesData,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """Calculate Value at Risk (VaR).

        Args:
            data: Time series data
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon: Time horizon in days

        Returns:
            Dictionary with VaR metrics
        """
        values = np.array(data.values)
        returns = np.diff(np.log(values))

        # Scale to time horizon
        scaled_returns = returns * np.sqrt(time_horizon)

        # Calculate VaR
        current_value = values[-1]
        var_percentile = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        var_absolute = current_value * (np.exp(var_percentile) - 1)

        # Calculate CVaR (Expected Shortfall)
        tail_returns = scaled_returns[scaled_returns <= var_percentile]
        cvar_percentile = np.mean(tail_returns) if len(tail_returns) > 0 else var_percentile
        cvar_absolute = current_value * (np.exp(cvar_percentile) - 1)

        return {
            'var_relative': float(var_percentile * 100),  # As percentage
            'var_absolute': float(var_absolute),
            'cvar_relative': float(cvar_percentile * 100),
            'cvar_absolute': float(cvar_absolute),
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'current_value': current_value,
        }

    def generate_report(
        self,
        historical_stats: Dict[str, float],
        forecast: ForecastResult,
        var_metrics: Dict[str, float]
    ) -> Path:
        """Generate comprehensive time series analysis report.

        Args:
            historical_stats: Historical statistics
            forecast: Forecast results
            var_metrics: VaR metrics

        Returns:
            Path to JSON report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"timeseries_analysis_{timestamp}.json"

        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'timeseries_analysis',
            },
            'historical_statistics': historical_stats,
            'var_metrics': var_metrics,
            'forecast_summary': {
                'num_days': len(forecast.dates),
                'forecast_start': forecast.dates[0].isoformat(),
                'forecast_end': forecast.dates[-1].isoformat(),
                'mean_final': forecast.mean[-1],
                'std_final': forecast.std[-1],
                'p5_final': forecast.percentile_5[-1],
                'p95_final': forecast.percentile_95[-1],
            },
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✓ Time series analysis report: {report_file.name}")
        return report_file

    def print_summary(self, stats: Dict[str, float], var_metrics: Dict[str, float]):
        """Print analysis summary to console.

        Args:
            stats: Historical statistics
            var_metrics: VaR metrics
        """
        print("=" * 80)
        print("TIME SERIES ANALYSIS SUMMARY")
        print("=" * 80)
        print()

        print("Historical Statistics:")
        print("-" * 80)
        print(f"  Current Value:        ${stats['current']:>15,.2f}")
        print(f"  Mean:                 ${stats['mean']:>15,.2f}")
        print(f"  Std Deviation:        ${stats['std']:>15,.2f}")
        print(f"  Min:                  ${stats['min']:>15,.2f}")
        print(f"  Max:                  ${stats['max']:>15,.2f}")
        print()
        print(f"  Mean Return:          {stats['return_mean']*100:>15.4f}%")
        print(f"  Return Volatility:    {stats['return_std']*100:>15.4f}%")
        print(f"  Annual Volatility:    {stats['volatility_annual']*100:>15.2f}%")
        print(f"  Sharpe Ratio:         {stats['sharpe_ratio']:>15.2f}")
        print(f"  Max Drawdown:         {stats['max_drawdown']:>15.2f}%")
        print()

        print("Risk Metrics (VaR):")
        print("-" * 80)
        print(f"  Confidence Level:     {var_metrics['confidence_level']*100:>15.1f}%")
        print(f"  Time Horizon:         {var_metrics['time_horizon']:>15d} day(s)")
        print(f"  Value at Risk:        ${var_metrics['var_absolute']:>15,.2f}  ({var_metrics['var_relative']:>6.2f}%)")
        print(f"  Conditional VaR:      ${var_metrics['cvar_absolute']:>15,.2f}  ({var_metrics['cvar_relative']:>6.2f}%)")
        print()


def main():
    """Demo of time series analysis."""
    print("=" * 80)
    print("Fictional Bank - Time Series Analysis Demo")
    print("=" * 80)
    print()

    # Load portfolio
    from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio

    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(output_dir=Path(__file__).parent / "reports")

    # Generate historical data
    print("Generating historical MTM data (252 days)...")
    mtm_history = analyzer.generate_historical_data(
        portfolio, book_hierarchy,
        num_days=252,
        volatility=0.02,
        drift=0.0001,
        seed=42
    )
    print(f"✓ Generated {len(mtm_history.dates)} days of historical data")
    print()

    # Calculate statistics
    print("Calculating statistics...")
    stats = analyzer.calculate_statistics(mtm_history)
    print("✓ Statistics calculated")
    print()

    # Calculate VaR
    print("Calculating Value at Risk...")
    var_metrics = analyzer.calculate_var(mtm_history, confidence_level=0.95, time_horizon=1)
    print("✓ VaR calculated")
    print()

    # Generate forecast
    print("Running Monte Carlo forecast (60 days, 1000 scenarios)...")
    forecast = analyzer.forecast_montecarlo(
        mtm_history,
        num_days=60,
        num_scenarios=1000,
        seed=42
    )
    print(f"✓ Forecast generated: {len(forecast.dates)} days")
    print()

    # Print summary
    analyzer.print_summary(stats, var_metrics)

    # Generate report
    report_file = analyzer.generate_report(stats, forecast, var_metrics)
    print()

    print("=" * 80)
    print("Time Series Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
