"""Performance metrics calculation for strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a strategy."""

    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk metrics
    volatility: float
    annualized_volatility: float
    downside_deviation: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # in days
    recovery_time: int  # in days

    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk at 95%
    cvar_95: float  # Conditional VaR at 95%

    # Win/loss metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Other
    num_periods: int
    num_trades: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "Total Return": f"{self.total_return:.2%}",
            "Annualized Return": f"{self.annualized_return:.2%}",
            "Volatility": f"{self.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "VaR (95%)": f"{self.var_95:.2%}",
        }


def calculate_sharpe_ratio(
    returns: pd.Series | Array,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = np.array(returns)

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)

    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series | Array,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """Calculate Sortino ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        target_return: Target return for downside deviation

    Returns:
        Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = np.array(returns)

    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < target_return] - target_return

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0

    sortino = (
        np.sqrt(periods_per_year)
        * np.mean(excess_returns)
        / np.std(downside_returns)
    )

    return float(sortino)


def calculate_max_drawdown(
    returns: pd.Series | Array,
) -> tuple[float, int, int]:
    """Calculate maximum drawdown and recovery metrics.

    Args:
        returns: Return series

    Returns:
        Tuple of (max_drawdown, max_drawdown_duration, recovery_time)
    """
    if isinstance(returns, pd.Series):
        returns_array = returns.values
    else:
        returns_array = np.array(returns)

    if len(returns_array) == 0:
        return 0.0, 0, 0

    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns_array)

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max

    # Max drawdown
    max_dd = float(np.min(drawdown))

    # Max drawdown duration
    max_dd_duration = 0
    current_dd_duration = 0

    for dd in drawdown:
        if dd < 0:
            current_dd_duration += 1
            max_dd_duration = max(max_dd_duration, current_dd_duration)
        else:
            current_dd_duration = 0

    # Recovery time (time to recover from max drawdown)
    max_dd_idx = np.argmin(drawdown)
    recovery_time = 0

    for i in range(max_dd_idx + 1, len(drawdown)):
        if drawdown[i] >= 0:
            recovery_time = i - max_dd_idx
            break
    else:
        recovery_time = len(drawdown) - max_dd_idx  # Still in drawdown

    return max_dd, max_dd_duration, recovery_time


def calculate_calmar_ratio(
    returns: pd.Series | Array,
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Return series
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = np.array(returns)

    if len(returns) == 0:
        return 0.0

    # Annualized return
    total_return = np.prod(1 + returns) - 1
    years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(returns)

    if abs(max_dd) < 1e-10:
        return 0.0

    calmar = annualized_return / abs(max_dd)

    return float(calmar)


def calculate_information_ratio(
    returns: pd.Series | Array,
    benchmark_returns: pd.Series | Array,
    periods_per_year: int = 252,
) -> float:
    """Calculate information ratio.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Information ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values

    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)

    if len(returns) == 0 or len(returns) != len(benchmark_returns):
        return 0.0

    # Active returns
    active_returns = returns - benchmark_returns

    if np.std(active_returns) == 0:
        return 0.0

    # Information ratio
    ir = (
        np.sqrt(periods_per_year)
        * np.mean(active_returns)
        / np.std(active_returns)
    )

    return float(ir)


def calculate_var(
    returns: pd.Series | Array,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Value at Risk.

    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR value (as positive number for loss)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = np.array(returns)

    if len(returns) == 0:
        return 0.0

    var = -float(np.percentile(returns, (1 - confidence_level) * 100))

    return var


def calculate_cvar(
    returns: pd.Series | Array,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        CVaR value (as positive number for loss)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    returns = np.array(returns)

    if len(returns) == 0:
        return 0.0

    var_threshold = -calculate_var(returns, confidence_level)
    tail_losses = returns[returns <= var_threshold]

    if len(tail_losses) == 0:
        return 0.0

    cvar = -float(np.mean(tail_losses))

    return cvar


def calculate_comprehensive_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    trades: Optional[pd.DataFrame] = None,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics.

    Args:
        returns: Return series
        benchmark_returns: Optional benchmark returns for information ratio
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        trades: Optional trade DataFrame for trade statistics

    Returns:
        PerformanceMetrics object
    """
    returns_array = returns.values

    # Return metrics
    total_return = float(np.prod(1 + returns_array) - 1)
    years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    cumulative_return = total_return

    # Volatility
    volatility = float(np.std(returns_array))
    annualized_volatility = volatility * np.sqrt(periods_per_year)

    # Downside deviation
    downside_returns = returns_array[returns_array < 0]
    downside_deviation = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0

    # Risk-adjusted ratios
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(returns, periods_per_year)

    if benchmark_returns is not None:
        information = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
    else:
        information = 0.0

    # Drawdown metrics
    max_dd, max_dd_duration, recovery_time = calculate_max_drawdown(returns)

    # Average drawdown
    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    avg_drawdown = float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0

    # Distribution metrics
    skewness = float(pd.Series(returns_array).skew())
    kurtosis = float(pd.Series(returns_array).kurtosis())
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)

    # Trade metrics
    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    largest_win = 0.0
    largest_loss = 0.0
    num_trades = 0

    if trades is not None and len(trades) > 0:
        if "pnl" in trades.columns:
            pnls = trades["pnl"].values
            winning = pnls[pnls > 0]
            losing = pnls[pnls < 0]

            num_trades = len(pnls)
            win_rate = len(winning) / num_trades if num_trades > 0 else 0.0

            gross_profit = np.sum(winning) if len(winning) > 0 else 0.0
            gross_loss = abs(np.sum(losing)) if len(losing) > 0 else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            avg_win = float(np.mean(winning)) if len(winning) > 0 else 0.0
            avg_loss = float(np.mean(losing)) if len(losing) > 0 else 0.0
            largest_win = float(np.max(winning)) if len(winning) > 0 else 0.0
            largest_loss = float(np.min(losing)) if len(losing) > 0 else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        cumulative_return=cumulative_return,
        volatility=volatility,
        annualized_volatility=annualized_volatility,
        downside_deviation=downside_deviation,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        information_ratio=information,
        max_drawdown=max_dd,
        avg_drawdown=avg_drawdown,
        max_drawdown_duration=max_dd_duration,
        recovery_time=recovery_time,
        skewness=skewness,
        kurtosis=kurtosis,
        var_95=var_95,
        cvar_95=cvar_95,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        num_periods=len(returns),
        num_trades=num_trades,
    )


__all__ = [
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "calculate_information_ratio",
    "calculate_var",
    "calculate_cvar",
    "calculate_comprehensive_metrics",
]
