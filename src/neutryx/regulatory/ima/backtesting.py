"""Backtesting framework with Traffic Light Approach for IMA.

The backtesting framework compares predicted risk measures (ES/VaR) against actual P&L
to validate model performance. Basel III requires:

1. ES backtesting (primary): Compare ES forecast to actual P&L
2. VaR backtesting (supplementary): Count VaR exceedances (Basel II legacy)
3. Traffic Light Approach: Green/Amber/Red zones based on number of exceptions

The traffic light approach uses binomial distribution to determine zones:
- Green zone: Model performs well, no capital add-on
- Amber zone: Model acceptable but requires monitoring, small add-on
- Red zone: Model inadequate, must recalibrate or use standardized approach

References
----------
- Basel Committee on Banking Supervision (2016). "Minimum capital requirements for
  market risk"
- Basel Committee on Banking Supervision (2006). "International Convergence of
  Capital Measurement and Capital Standards" (Basel II - VaR backtesting)
- Christoffersen, P. (1998). "Evaluating Interval Forecasts", International
  Economic Review, 39(4), 841-862.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import stats


class TrafficLightZone(str, Enum):
    """Traffic light zones for backtesting results."""
    GREEN = "green"     # Model performs well
    AMBER = "amber"     # Model acceptable with monitoring
    RED = "red"         # Model inadequate


@dataclass
class BacktestException:
    """Individual backtesting exception (breach)."""
    date: date
    actual_pnl: float
    predicted_risk_measure: float  # ES or VaR
    severity: float  # How much actual loss exceeded prediction
    is_outlier: bool = False


@dataclass
class BacktestResult:
    """Backtesting result with traffic light classification."""

    # Core statistics
    num_exceptions: int
    num_observations: int
    exception_rate: float
    expected_exceptions: float

    # Traffic light
    traffic_light_zone: TrafficLightZone
    capital_multiplier: float  # Add-on for capital requirement

    # Exception details
    exceptions: List[BacktestException] = field(default_factory=list)

    # Statistical tests
    kupiec_pof_statistic: Optional[float] = None  # Proportion of Failures test
    kupiec_pof_pvalue: Optional[float] = None
    christoffersen_test_statistic: Optional[float] = None
    christoffersen_test_pvalue: Optional[float] = None

    # Additional metrics
    average_severity: float = 0.0
    max_severity: float = 0.0
    coverage_level: float = 0.975  # e.g., 97.5% for ES

    # Time period
    test_period_start: Optional[date] = None
    test_period_end: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_exceptions": self.num_exceptions,
            "num_observations": self.num_observations,
            "exception_rate": self.exception_rate,
            "expected_exceptions": self.expected_exceptions,
            "traffic_light_zone": self.traffic_light_zone.value,
            "capital_multiplier": self.capital_multiplier,
            "kupiec_pof_pvalue": self.kupiec_pof_pvalue,
            "christoffersen_test_pvalue": self.christoffersen_test_pvalue,
            "average_severity": self.average_severity,
            "max_severity": self.max_severity,
            "coverage_level": self.coverage_level,
        }


# Basel II Traffic Light thresholds for VaR (99% confidence, 250 days)
# Number of exceptions -> Zone
VAR_TRAFFIC_LIGHT_THRESHOLDS = {
    "green_max": 4,   # 0-4 exceptions: green zone
    "amber_max": 9,   # 5-9 exceptions: amber zone
    # 10+ exceptions: red zone
}

# Capital multiplier based on number of exceptions (Basel II)
VAR_CAPITAL_MULTIPLIERS = {
    0: 3.00, 1: 3.00, 2: 3.00, 3: 3.00, 4: 3.00,  # Green zone
    5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85,  # Amber zone
    # 10+: Red zone, multiplier = 4.00 or model rejection
}


def calculate_traffic_light_zone(
    num_exceptions: int,
    num_observations: int,
    coverage_level: float = 0.99,
    use_var_thresholds: bool = True
) -> Tuple[TrafficLightZone, float]:
    """Calculate traffic light zone and capital multiplier.

    Parameters
    ----------
    num_exceptions : int
        Number of times actual loss exceeded forecast
    num_observations : int
        Total number of observations
    coverage_level : float, optional
        Coverage level (default: 0.99 for VaR, 0.975 for ES)
    use_var_thresholds : bool, optional
        Use Basel II VaR thresholds (True) or calculate dynamically (False)

    Returns
    -------
    zone : TrafficLightZone
        Traffic light zone
    multiplier : float
        Capital requirement multiplier
    """
    if use_var_thresholds:
        # Use Basel II predefined thresholds (for 99% VaR, 250 days)
        if num_exceptions <= VAR_TRAFFIC_LIGHT_THRESHOLDS["green_max"]:
            zone = TrafficLightZone.GREEN
            multiplier = VAR_CAPITAL_MULTIPLIERS.get(num_exceptions, 3.00)
        elif num_exceptions <= VAR_TRAFFIC_LIGHT_THRESHOLDS["amber_max"]:
            zone = TrafficLightZone.AMBER
            multiplier = VAR_CAPITAL_MULTIPLIERS.get(num_exceptions, 3.85)
        else:
            zone = TrafficLightZone.RED
            multiplier = 4.00  # Maximum multiplier
    else:
        # Calculate zones based on binomial confidence intervals
        alpha = 1 - coverage_level
        expected_exceptions = alpha * num_observations

        # Use binomial distribution to determine zones
        # Green zone: within 95% confidence interval
        # Amber zone: within 99% confidence interval
        # Red zone: outside 99% confidence interval

        # 95% confidence interval for green zone
        green_upper = stats.binom.ppf(0.975, num_observations, alpha)

        # 99% confidence interval for amber zone
        amber_upper = stats.binom.ppf(0.995, num_observations, alpha)

        if num_exceptions <= green_upper:
            zone = TrafficLightZone.GREEN
            multiplier = 3.00
        elif num_exceptions <= amber_upper:
            zone = TrafficLightZone.AMBER
            # Linear interpolation between 3.00 and 3.85
            t = (num_exceptions - green_upper) / (amber_upper - green_upper)
            multiplier = 3.00 + t * 0.85
        else:
            zone = TrafficLightZone.RED
            multiplier = 4.00

    return zone, multiplier


def backtest_var(
    actual_pnl: Array,
    var_forecasts: Array,
    coverage_level: float = 0.99,
    dates: Optional[List[date]] = None
) -> BacktestResult:
    """Backtest VaR forecasts using traffic light approach.

    Parameters
    ----------
    actual_pnl : Array
        Actual P&L (losses as negative values)
    var_forecasts : Array
        VaR forecasts (positive values indicating risk)
    coverage_level : float, optional
        Coverage level (typically 0.99 for Basel II)
    dates : List[date], optional
        Dates for each observation

    Returns
    -------
    BacktestResult
        Backtesting results with traffic light classification
    """
    pnl = np.array(actual_pnl)
    var = np.array(var_forecasts)

    if len(pnl) != len(var):
        raise ValueError("P&L and VaR arrays must have same length")

    n_obs = len(pnl)
    alpha = 1 - coverage_level
    expected_exceptions = alpha * n_obs

    # Identify VaR exceptions (actual loss > VaR forecast)
    # Loss is negative, so exception when: -pnl > var  =>  pnl < -var
    exceptions_mask = pnl < -var
    exception_indices = np.where(exceptions_mask)[0]
    n_exceptions = len(exception_indices)

    # Create exception objects
    exceptions = []
    for idx in exception_indices:
        severity = float((-pnl[idx]) - var[idx])  # How much loss exceeded VaR
        exception = BacktestException(
            date=dates[idx] if dates else None,
            actual_pnl=float(pnl[idx]),
            predicted_risk_measure=float(var[idx]),
            severity=severity,
        )
        exceptions.append(exception)

    # Calculate exception rate
    exception_rate = n_exceptions / n_obs

    # Determine traffic light zone
    zone, multiplier = calculate_traffic_light_zone(
        n_exceptions, n_obs, coverage_level, use_var_thresholds=True
    )

    # Kupiec Proportion of Failures (POF) test
    # H0: Exception rate = expected rate
    # Test statistic: -2 * ln(L(p0)/L(p1))
    # where p0 = alpha (expected), p1 = observed exception rate
    if n_exceptions > 0 and n_exceptions < n_obs:
        p1 = n_exceptions / n_obs
        lr_stat = -2 * (
            n_exceptions * np.log(alpha / p1) +
            (n_obs - n_exceptions) * np.log((1 - alpha) / (1 - p1))
        )
        kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
    else:
        lr_stat = np.nan
        kupiec_pvalue = np.nan

    # Christoffersen independence test
    # Tests if exceptions are independent (not clustered)
    if n_exceptions >= 2:
        # Count transitions: (no exception, no exception), (no exception, exception), etc.
        transitions = np.zeros((2, 2))
        for i in range(len(exceptions_mask) - 1):
            prev_state = int(exceptions_mask[i])
            curr_state = int(exceptions_mask[i + 1])
            transitions[prev_state, curr_state] += 1

        # Calculate likelihood ratio test statistic
        n00, n01 = transitions[0, 0], transitions[0, 1]
        n10, n11 = transitions[1, 0], transitions[1, 1]

        if n00 + n01 > 0 and n10 + n11 > 0 and n01 > 0 and n11 > 0:
            p01 = n01 / (n00 + n01)
            p11 = n11 / (n10 + n11)
            p = (n01 + n11) / n_obs

            christoffersen_stat = -2 * (
                (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p) -
                n00 * np.log(1 - p01) - n01 * np.log(p01) -
                n10 * np.log(1 - p11) - n11 * np.log(p11)
            )
            christoffersen_pvalue = 1 - stats.chi2.cdf(christoffersen_stat, df=1)
        else:
            christoffersen_stat = np.nan
            christoffersen_pvalue = np.nan
    else:
        christoffersen_stat = np.nan
        christoffersen_pvalue = np.nan

    # Calculate severity statistics
    if exceptions:
        avg_severity = float(np.mean([e.severity for e in exceptions]))
        max_severity = float(max([e.severity for e in exceptions]))
    else:
        avg_severity = 0.0
        max_severity = 0.0

    result = BacktestResult(
        num_exceptions=n_exceptions,
        num_observations=n_obs,
        exception_rate=exception_rate,
        expected_exceptions=expected_exceptions,
        traffic_light_zone=zone,
        capital_multiplier=multiplier,
        exceptions=exceptions,
        kupiec_pof_statistic=float(lr_stat) if not np.isnan(lr_stat) else None,
        kupiec_pof_pvalue=float(kupiec_pvalue) if not np.isnan(kupiec_pvalue) else None,
        christoffersen_test_statistic=float(christoffersen_stat) if not np.isnan(christoffersen_stat) else None,
        christoffersen_test_pvalue=float(christoffersen_pvalue) if not np.isnan(christoffersen_pvalue) else None,
        average_severity=avg_severity,
        max_severity=max_severity,
        coverage_level=coverage_level,
        test_period_start=dates[0] if dates else None,
        test_period_end=dates[-1] if dates else None,
    )

    return result


def backtest_expected_shortfall(
    actual_pnl: Array,
    es_forecasts: Array,
    coverage_level: float = 0.975,
    dates: Optional[List[date]] = None
) -> BacktestResult:
    """Backtest Expected Shortfall forecasts.

    ES backtesting is more complex than VaR as we need to check if the
    average of tail losses matches the ES forecast.

    Parameters
    ----------
    actual_pnl : Array
        Actual P&L (losses as negative)
    es_forecasts : Array
        ES forecasts (positive values)
    coverage_level : float, optional
        Coverage level (typically 0.975 for Basel III)
    dates : List[date], optional
        Dates for observations

    Returns
    -------
    BacktestResult
        Backtesting results
    """
    pnl = np.array(actual_pnl)
    es = np.array(es_forecasts)

    if len(pnl) != len(es):
        raise ValueError("P&L and ES arrays must have same length")

    n_obs = len(pnl)
    alpha = 1 - coverage_level

    # For ES backtesting, we identify observations where loss exceeds VaR threshold
    # Then check if average of these losses is consistent with ES forecast

    # First, identify tail events (beyond VaR threshold)
    # Approximate VaR from ES (ES is typically 1.2-1.5x VaR at 97.5%)
    var_approx = es * 0.85  # Rough approximation

    # Identify exceptions
    exceptions_mask = pnl < -var_approx
    exception_indices = np.where(exceptions_mask)[0]
    n_exceptions = len(exception_indices)

    # Create exception objects
    exceptions = []
    for idx in exception_indices:
        actual_loss = -pnl[idx]
        severity = float(actual_loss - es[idx])
        exception = BacktestException(
            date=dates[idx] if dates else None,
            actual_pnl=float(pnl[idx]),
            predicted_risk_measure=float(es[idx]),
            severity=severity,
            is_outlier=(severity > es[idx] * 0.5)  # Flag extreme outliers
        )
        exceptions.append(exception)

    exception_rate = n_exceptions / n_obs
    expected_exceptions = alpha * n_obs

    # ES-specific tests
    # 1. Check if average tail loss matches ES forecast
    if n_exceptions > 0:
        tail_losses = -pnl[exceptions_mask]
        avg_tail_loss = np.mean(tail_losses)
        avg_es_forecast = np.mean(es[exceptions_mask])

        # Simple t-test: is average tail loss significantly different from ES?
        if n_exceptions > 1:
            t_stat = (avg_tail_loss - avg_es_forecast) / (np.std(tail_losses) / np.sqrt(n_exceptions))
            es_test_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_exceptions - 1))
        else:
            es_test_pvalue = np.nan
    else:
        es_test_pvalue = np.nan

    # Determine zone (use similar approach to VaR but with adjusted thresholds)
    zone, multiplier = calculate_traffic_light_zone(
        n_exceptions, n_obs, coverage_level, use_var_thresholds=False
    )

    # Calculate severity statistics
    if exceptions:
        avg_severity = float(np.mean([e.severity for e in exceptions]))
        max_severity = float(max([e.severity for e in exceptions]))
    else:
        avg_severity = 0.0
        max_severity = 0.0

    result = BacktestResult(
        num_exceptions=n_exceptions,
        num_observations=n_obs,
        exception_rate=exception_rate,
        expected_exceptions=expected_exceptions,
        traffic_light_zone=zone,
        capital_multiplier=multiplier,
        exceptions=exceptions,
        kupiec_pof_pvalue=float(es_test_pvalue) if not np.isnan(es_test_pvalue) else None,
        average_severity=avg_severity,
        max_severity=max_severity,
        coverage_level=coverage_level,
        test_period_start=dates[0] if dates else None,
        test_period_end=dates[-1] if dates else None,
    )

    return result


def rolling_backtest(
    dates: List[date],
    actual_pnl: Array,
    risk_forecasts: Array,
    window_days: int = 250,
    risk_measure: str = "var",
    coverage_level: float = 0.99
) -> List[Tuple[date, BacktestResult]]:
    """Perform rolling backtests over time.

    Parameters
    ----------
    dates : List[date]
        Dates for observations
    actual_pnl : Array
        Actual P&L series
    risk_forecasts : Array
        Risk measure forecasts (VaR or ES)
    window_days : int, optional
        Rolling window size (default: 250)
    risk_measure : str, optional
        "var" or "es"
    coverage_level : float, optional
        Coverage level

    Returns
    -------
    List[Tuple[date, BacktestResult]]
        Rolling backtest results
    """
    n_obs = len(dates)
    results = []

    backtest_func = backtest_var if risk_measure == "var" else backtest_expected_shortfall

    for i in range(window_days, n_obs + 1):
        window_start = i - window_days
        window_end = i

        window_dates = dates[window_start:window_end]
        window_pnl = actual_pnl[window_start:window_end]
        window_forecasts = risk_forecasts[window_start:window_end]

        result = backtest_func(
            window_pnl,
            window_forecasts,
            coverage_level=coverage_level,
            dates=window_dates
        )

        results.append((dates[window_end - 1], result))

    return results


__all__ = [
    "TrafficLightZone",
    "BacktestException",
    "BacktestResult",
    "calculate_traffic_light_zone",
    "backtest_var",
    "backtest_expected_shortfall",
    "rolling_backtest",
    "VAR_TRAFFIC_LIGHT_THRESHOLDS",
    "VAR_CAPITAL_MULTIPLIERS",
]
