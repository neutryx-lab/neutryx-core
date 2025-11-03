"""Risk metrics: VaR, CVaR (Expected Shortfall), and related measures.

This module provides functions for computing Value at Risk (VaR),
Conditional Value at Risk (CVaR), and other risk measures from
simulated portfolio or option value distributions.
"""
import jax.numpy as jnp

from neutryx.core.engine import Array


def value_at_risk(returns: Array, confidence_level: float = 0.95) -> float:
    """Compute Value at Risk (VaR) at given confidence level.

    VaR is the maximum loss not exceeded with a given confidence level.
    For example, 95% VaR is the loss level that will not be exceeded
    95% of the time.

    Parameters
    ----------
    returns : Array
        Distribution of returns or P&L values
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% confidence)

    Returns
    -------
    float
        Value at Risk (positive number represents loss)
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must be between 0 and 1")

    # VaR is the negative of the (1 - confidence_level) quantile
    # We negate returns to get losses
    losses = -returns
    var = jnp.quantile(losses, confidence_level)

    return float(var)


def conditional_value_at_risk(returns: Array, confidence_level: float = 0.95) -> float:
    """Compute Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    CVaR is the expected loss given that the loss exceeds VaR.
    It provides information about the tail risk beyond VaR.

    Parameters
    ----------
    returns : Array
        Distribution of returns or P&L values
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% confidence)

    Returns
    -------
    float
        Conditional Value at Risk (positive number represents loss)
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must be between 0 and 1")

    # Convert returns to losses
    losses = -returns

    # Compute VaR first
    var = jnp.quantile(losses, confidence_level)

    # CVaR is the average of losses that exceed VaR
    tail_losses = losses[losses >= var]

    if tail_losses.size == 0:
        return float(var)

    cvar = tail_losses.mean()

    return float(cvar)


def expected_shortfall(returns: Array, alpha: float = 0.95) -> float:
    """Alias for CVaR - Expected Shortfall.

    Parameters
    ----------
    returns : Array
        Distribution of returns or P&L values
    alpha : float
        Confidence level

    Returns
    -------
    float
        Expected Shortfall
    """
    return conditional_value_at_risk(returns, alpha)


def portfolio_var(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95,
) -> float:
    """Compute portfolio VaR from position weights and scenario returns.

    Parameters
    ----------
    positions : Array
        Position sizes/weights for each asset, shape [n_assets]
    returns_scenarios : Array
        Scenario returns for each asset, shape [n_scenarios, n_assets]
    confidence_level : float
        Confidence level

    Returns
    -------
    float
        Portfolio Value at Risk
    """
    # Portfolio returns = weighted sum of asset returns
    portfolio_returns = jnp.dot(returns_scenarios, positions)

    return value_at_risk(portfolio_returns, confidence_level)


def portfolio_cvar(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95,
) -> float:
    """Compute portfolio CVaR from position weights and scenario returns.

    Parameters
    ----------
    positions : Array
        Position sizes/weights for each asset, shape [n_assets]
    returns_scenarios : Array
        Scenario returns for each asset, shape [n_scenarios, n_assets]
    confidence_level : float
        Confidence level

    Returns
    -------
    float
        Portfolio Conditional Value at Risk
    """
    portfolio_returns = jnp.dot(returns_scenarios, positions)

    return conditional_value_at_risk(portfolio_returns, confidence_level)


def downside_deviation(returns: Array, threshold: float = 0.0) -> float:
    """Compute downside deviation (semi-standard deviation).

    Measures volatility of returns below a threshold (typically 0).

    Parameters
    ----------
    returns : Array
        Distribution of returns
    threshold : float
        Threshold for downside (typically 0 for no loss)

    Returns
    -------
    float
        Downside deviation
    """
    downside_returns = jnp.where(returns < threshold, returns - threshold, 0.0)
    return float(jnp.sqrt(jnp.mean(downside_returns**2)))


def maximum_drawdown(cumulative_returns: Array) -> float:
    """Compute maximum drawdown from cumulative return series.

    Maximum drawdown is the largest peak-to-trough decline.

    Parameters
    ----------
    cumulative_returns : Array
        Cumulative returns over time, shape [n_periods]

    Returns
    -------
    float
        Maximum drawdown (positive number)
    """
    # Compute running maximum
    running_max = jnp.maximum.accumulate(cumulative_returns)

    # Drawdown at each point
    drawdown = running_max - cumulative_returns

    # Maximum drawdown
    max_dd = jnp.max(drawdown)

    return float(max_dd)


def sharpe_ratio(returns: Array, risk_free_rate: float = 0.0) -> float:
    """Compute Sharpe ratio.

    Measures risk-adjusted return as (mean return - risk-free rate) / std(return).

    Parameters
    ----------
    returns : Array
        Distribution of returns
    risk_free_rate : float
        Risk-free rate (per period)

    Returns
    -------
    float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    return float(jnp.mean(excess_returns) / (jnp.std(excess_returns) + 1e-10))


def sortino_ratio(returns: Array, risk_free_rate: float = 0.0, threshold: float = 0.0) -> float:
    """Compute Sortino ratio.

    Similar to Sharpe ratio but uses downside deviation instead of standard deviation.

    Parameters
    ----------
    returns : Array
        Distribution of returns
    risk_free_rate : float
        Risk-free rate (per period)
    threshold : float
        Minimum acceptable return (typically 0)

    Returns
    -------
    float
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate
    dd = downside_deviation(returns, threshold)

    return float(jnp.mean(excess_returns) / (dd + 1e-10))


def compute_all_risk_metrics(
    returns: Array,
    confidence_levels: list = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute comprehensive set of risk metrics.

    Parameters
    ----------
    returns : Array
        Distribution of returns or P&L values
    confidence_levels : list, optional
        List of confidence levels for VaR/CVaR (default: [0.95, 0.99])
    risk_free_rate : float
        Risk-free rate for Sharpe/Sortino ratios

    Returns
    -------
    dict
        Dictionary containing all risk metrics
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    metrics = {
        "mean": float(jnp.mean(returns)),
        "std": float(jnp.std(returns)),
        "skewness": float(
            jnp.mean(((returns - jnp.mean(returns)) / jnp.std(returns)) ** 3)
        ),
        "kurtosis": float(
            jnp.mean(((returns - jnp.mean(returns)) / jnp.std(returns)) ** 4)
        ),
        "min": float(jnp.min(returns)),
        "max": float(jnp.max(returns)),
        "downside_deviation": downside_deviation(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
    }

    # Add VaR and CVaR for each confidence level
    for cl in confidence_levels:
        metrics[f"var_{int(cl*100)}"] = value_at_risk(returns, cl)
        metrics[f"cvar_{int(cl*100)}"] = conditional_value_at_risk(returns, cl)

    return metrics


__all__ = [
    "value_at_risk",
    "conditional_value_at_risk",
    "expected_shortfall",
    "portfolio_var",
    "portfolio_cvar",
    "downside_deviation",
    "maximum_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "compute_all_risk_metrics",
]
