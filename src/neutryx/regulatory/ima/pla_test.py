"""P&L Attribution (PLA) Test for Internal Models Approach.

The PLA test compares risk-theoretical P&L (RTPL) from the risk management model
against hypothetical P&L (HPL) from the actual front-office pricing models.

Per Basel III/FRTB:
- Test must be passed for a trading desk to qualify for IMA
- Calculated using 12 months of daily data (≥250 observations)
- Two test statistics: Spearman correlation and Kolmogorov-Smirnov test
- Thresholds: Spearman ρ ≥ 0.85 or KS < 0.09 for amber zone

References
----------
- Basel Committee on Banking Supervision (2019). "Minimum capital requirements for
  market risk - Explanatory note on the minimum capital requirements for market risk"
- Basel Committee on Banking Supervision (2022). "Instructions for Basel III
  monitoring: Market risk"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import stats


class PLATestResult(str, Enum):
    """PLA test result classification."""
    GREEN = "green"      # Pass: desk qualifies for IMA
    AMBER = "amber"      # Warning: additional monitoring required
    RED = "red"          # Fail: desk must use standardized approach


@dataclass
class PLAMetrics:
    """P&L Attribution test metrics."""

    # Core PLA statistics
    spearman_correlation: float
    kolmogorov_smirnov_statistic: float
    mean_absolute_difference: float
    root_mean_squared_error: float

    # Test result
    test_result: PLATestResult
    passes_test: bool

    # Additional statistics
    mean_unexplained_pnl: float
    variance_ratio: float  # Var(HPL-RTPL) / Var(HPL)
    num_observations: int
    test_period_start: Optional[date] = None
    test_period_end: Optional[date] = None

    # Threshold information
    spearman_threshold: float = 0.85
    ks_threshold: float = 0.09

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spearman_correlation": self.spearman_correlation,
            "kolmogorov_smirnov_statistic": self.kolmogorov_smirnov_statistic,
            "mean_absolute_difference": self.mean_absolute_difference,
            "root_mean_squared_error": self.root_mean_squared_error,
            "test_result": self.test_result.value,
            "passes_test": self.passes_test,
            "mean_unexplained_pnl": self.mean_unexplained_pnl,
            "variance_ratio": self.variance_ratio,
            "num_observations": self.num_observations,
            "spearman_threshold": self.spearman_threshold,
            "ks_threshold": self.ks_threshold,
        }


def calculate_pla_metrics(
    hypothetical_pnl: Array,
    risk_theoretical_pnl: Array,
    spearman_threshold: float = 0.85,
    ks_threshold: float = 0.09
) -> PLAMetrics:
    """Calculate P&L Attribution test metrics.

    Parameters
    ----------
    hypothetical_pnl : Array
        Hypothetical P&L from front-office pricing models
        Shape: (n_days,)
    risk_theoretical_pnl : Array
        Risk-theoretical P&L from risk management models
        Shape: (n_days,)
    spearman_threshold : float, optional
        Minimum Spearman correlation for amber zone (default: 0.85)
    ks_threshold : float, optional
        Maximum KS statistic for amber zone (default: 0.09)

    Returns
    -------
    PLAMetrics
        P&L attribution test results

    Examples
    --------
    >>> hpl = jnp.array([100, -50, 75, -25, 150])
    >>> rtpl = jnp.array([95, -45, 80, -30, 145])
    >>> metrics = calculate_pla_metrics(hpl, rtpl)
    >>> print(f"Spearman: {metrics.spearman_correlation:.3f}")
    """
    # Ensure arrays are numpy for scipy compatibility
    hpl = np.array(hypothetical_pnl)
    rtpl = np.array(risk_theoretical_pnl)

    if len(hpl) != len(rtpl):
        raise ValueError("HPL and RTPL must have same length")

    n_obs = len(hpl)

    if n_obs < 250:
        import warnings
        warnings.warn(
            f"PLA test requires at least 250 observations (12 months), got {n_obs}",
            UserWarning
        )

    # 1. Spearman rank correlation
    # Measures monotonic relationship between HPL and RTPL
    spearman_corr, _ = stats.spearmanr(hpl, rtpl)

    # 2. Kolmogorov-Smirnov test
    # Tests if unexplained P&L follows a symmetric distribution around zero
    unexplained_pnl = hpl - rtpl

    # Standardize unexplained P&L
    std_unexplained = unexplained_pnl / np.std(hpl) if np.std(hpl) > 0 else unexplained_pnl

    # KS test against standard normal distribution
    # We want the distribution to be centered at zero with small deviations
    ks_stat, _ = stats.kstest(std_unexplained, 'norm', args=(0, 1))

    # 3. Mean Absolute Difference
    mad = float(np.mean(np.abs(unexplained_pnl)))

    # 4. Root Mean Squared Error
    rmse = float(np.sqrt(np.mean(unexplained_pnl ** 2)))

    # 5. Mean unexplained P&L (should be close to zero)
    mean_unexplained = float(np.mean(unexplained_pnl))

    # 6. Variance ratio
    # Var(unexplained) / Var(HPL) - should be small
    var_ratio = float(np.var(unexplained_pnl) / np.var(hpl)) if np.var(hpl) > 0 else 0.0

    # Determine test result based on thresholds
    # Green zone: Spearman ≥ 0.85 AND KS < 0.09
    # Amber zone: Either threshold not met (but close)
    # Red zone: Both thresholds not met or far from thresholds

    passes_spearman = spearman_corr >= spearman_threshold
    passes_ks = ks_stat < ks_threshold

    if passes_spearman and passes_ks:
        test_result = PLATestResult.GREEN
        passes_test = True
    elif (spearman_corr >= spearman_threshold * 0.9 or
          ks_stat < ks_threshold * 1.2):
        # Close to thresholds - amber zone
        test_result = PLATestResult.AMBER
        passes_test = False
    else:
        test_result = PLATestResult.RED
        passes_test = False

    metrics = PLAMetrics(
        spearman_correlation=float(spearman_corr),
        kolmogorov_smirnov_statistic=float(ks_stat),
        mean_absolute_difference=mad,
        root_mean_squared_error=rmse,
        test_result=test_result,
        passes_test=passes_test,
        mean_unexplained_pnl=mean_unexplained,
        variance_ratio=var_ratio,
        num_observations=n_obs,
        spearman_threshold=spearman_threshold,
        ks_threshold=ks_threshold,
    )

    return metrics


def calculate_rolling_pla(
    dates: List[date],
    hypothetical_pnl: Array,
    risk_theoretical_pnl: Array,
    window_days: int = 250
) -> List[Tuple[date, PLAMetrics]]:
    """Calculate rolling P&L attribution test over time.

    Parameters
    ----------
    dates : List[date]
        Dates for each P&L observation
    hypothetical_pnl : Array
        Hypothetical P&L series
    risk_theoretical_pnl : Array
        Risk-theoretical P&L series
    window_days : int, optional
        Rolling window size (default: 250 for 12 months)

    Returns
    -------
    List[Tuple[date, PLAMetrics]]
        List of (date, metrics) for each rolling window
    """
    if len(dates) != len(hypothetical_pnl) or len(dates) != len(risk_theoretical_pnl):
        raise ValueError("Dates and P&L arrays must have same length")

    n_obs = len(dates)
    results = []

    for i in range(window_days, n_obs + 1):
        window_start_idx = i - window_days
        window_end_idx = i

        window_hpl = hypothetical_pnl[window_start_idx:window_end_idx]
        window_rtpl = risk_theoretical_pnl[window_start_idx:window_end_idx]

        metrics = calculate_pla_metrics(window_hpl, window_rtpl)

        # Add date information
        metrics.test_period_start = dates[window_start_idx]
        metrics.test_period_end = dates[window_end_idx - 1]

        results.append((dates[window_end_idx - 1], metrics))

    return results


def diagnose_pla_failures(
    hypothetical_pnl: Array,
    risk_theoretical_pnl: Array,
    threshold_percentile: float = 95
) -> Dict[str, Any]:
    """Diagnose reasons for PLA test failures.

    Provides detailed analysis of where and why the risk model differs
    from the front-office model.

    Parameters
    ----------
    hypothetical_pnl : Array
        Hypothetical P&L
    risk_theoretical_pnl : Array
        Risk-theoretical P&L
    threshold_percentile : float, optional
        Percentile for identifying outliers (default: 95)

    Returns
    -------
    dict
        Diagnostic information including:
        - largest_discrepancies: indices and values of biggest differences
        - systematic_bias: whether there's consistent over/under estimation
        - outlier_dates: dates with unusually large differences
        - correlation_by_sign: separate correlations for positive/negative P&L
    """
    hpl = np.array(hypothetical_pnl)
    rtpl = np.array(risk_theoretical_pnl)

    unexplained = hpl - rtpl

    # 1. Identify largest discrepancies
    abs_unexplained = np.abs(unexplained)
    threshold = np.percentile(abs_unexplained, threshold_percentile)
    outlier_indices = np.where(abs_unexplained > threshold)[0]

    largest_discrepancies = [
        {
            "index": int(idx),
            "hpl": float(hpl[idx]),
            "rtpl": float(rtpl[idx]),
            "difference": float(unexplained[idx]),
            "pct_diff": float(100 * unexplained[idx] / hpl[idx]) if hpl[idx] != 0 else np.inf
        }
        for idx in outlier_indices
    ]

    # Sort by absolute difference
    largest_discrepancies.sort(key=lambda x: abs(x["difference"]), reverse=True)

    # 2. Check for systematic bias
    systematic_bias = {
        "mean_difference": float(np.mean(unexplained)),
        "median_difference": float(np.median(unexplained)),
        "positive_bias_pct": float(100 * np.sum(unexplained > 0) / len(unexplained)),
        "mean_when_hpl_positive": float(np.mean(unexplained[hpl > 0])) if np.sum(hpl > 0) > 0 else 0.0,
        "mean_when_hpl_negative": float(np.mean(unexplained[hpl < 0])) if np.sum(hpl < 0) > 0 else 0.0,
    }

    # 3. Correlation by sign (check if model works differently for gains vs losses)
    positive_mask = hpl > 0
    negative_mask = hpl < 0

    if np.sum(positive_mask) > 1 and np.sum(negative_mask) > 1:
        corr_positive, _ = stats.spearmanr(hpl[positive_mask], rtpl[positive_mask])
        corr_negative, _ = stats.spearmanr(hpl[negative_mask], rtpl[negative_mask])
    else:
        corr_positive = corr_negative = np.nan

    correlation_by_sign = {
        "correlation_gains": float(corr_positive),
        "correlation_losses": float(corr_negative),
    }

    # 4. Time series properties
    # Check if errors are autocorrelated (suggests model misspecification)
    if len(unexplained) > 10:
        # Lag-1 autocorrelation
        lag1_corr = np.corrcoef(unexplained[:-1], unexplained[1:])[0, 1]
    else:
        lag1_corr = np.nan

    time_series_properties = {
        "lag1_autocorrelation": float(lag1_corr),
        "durbin_watson": float(np.sum(np.diff(unexplained)**2) / np.sum(unexplained**2))
        if np.sum(unexplained**2) > 0 else np.nan,
    }

    diagnosis = {
        "largest_discrepancies": largest_discrepancies[:10],  # Top 10
        "systematic_bias": systematic_bias,
        "correlation_by_sign": correlation_by_sign,
        "time_series_properties": time_series_properties,
        "outlier_threshold": float(threshold),
        "num_outliers": len(outlier_indices),
    }

    return diagnosis


__all__ = [
    "PLATestResult",
    "PLAMetrics",
    "calculate_pla_metrics",
    "calculate_rolling_pla",
    "diagnose_pla_failures",
]
