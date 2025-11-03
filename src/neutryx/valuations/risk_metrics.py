"""Risk metrics: VaR, CVaR (Expected Shortfall), and related measures.

This module provides functions for computing Value at Risk (VaR),
Conditional Value at Risk (CVaR), and other risk measures from
simulated portfolio or option value distributions.

Includes multiple VaR methodologies:
- Historical VaR: Based on historical return distribution
- Parametric VaR: Assumes normal distribution
- Monte Carlo VaR: From simulated scenarios
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import jax
import jax.numpy as jnp
from scipy import stats

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


class VaRMethod(str, Enum):
    """VaR calculation methodology."""

    HISTORICAL = "Historical"  # Based on historical returns
    PARAMETRIC = "Parametric"  # Assumes normal distribution (variance-covariance)
    MONTE_CARLO = "MonteCarlo"  # From Monte Carlo simulations
    CORNISH_FISHER = "CornishFisher"  # Parametric with skewness/kurtosis adjustment


def historical_var(
    returns: Array,
    confidence_level: float = 0.95,
    window: Optional[int] = None,
) -> float:
    """Calculate Historical VaR from empirical return distribution.

    Parameters
    ----------
    returns : Array
        Historical returns or P&L values
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% VaR)
    window : int, optional
        Rolling window size. If None, uses all returns

    Returns
    -------
    float
        Historical VaR

    Notes
    -----
    Historical VaR = empirical quantile of loss distribution
    No distributional assumptions required
    """
    if window is not None:
        returns = returns[-window:]

    return value_at_risk(returns, confidence_level)


def parametric_var(
    returns: Array,
    confidence_level: float = 0.95,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> float:
    """Calculate Parametric VaR assuming normal distribution.

    Parameters
    ----------
    returns : Array
        Historical returns (used to estimate mean/std if not provided)
    confidence_level : float
        Confidence level
    mean : float, optional
        Mean return. If None, estimated from returns
    std : float, optional
        Standard deviation. If None, estimated from returns

    Returns
    -------
    float
        Parametric VaR

    Notes
    -----
    Parametric VaR = -μ + σ * z_α
    where z_α is the normal quantile at confidence level α
    Assumes returns are normally distributed
    """
    if mean is None:
        mean = float(jnp.mean(returns))
    if std is None:
        std = float(jnp.std(returns))

    # Normal quantile at confidence level
    z_score = stats.norm.ppf(confidence_level)

    # VaR = -mean + std * z_score (negative of the quantile)
    var = -mean + std * z_score

    return float(var)


def monte_carlo_var(
    simulated_returns: Array,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Monte Carlo VaR from simulated scenarios.

    Parameters
    ----------
    simulated_returns : Array
        Simulated future returns/P&L from Monte Carlo, shape [n_scenarios]
    confidence_level : float
        Confidence level

    Returns
    -------
    float
        Monte Carlo VaR

    Notes
    -----
    Monte Carlo VaR = quantile of simulated loss distribution
    Can capture non-normal distributions and complex risk factors
    """
    return value_at_risk(simulated_returns, confidence_level)


def cornish_fisher_var(
    returns: Array,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Cornish-Fisher VaR with skewness/kurtosis adjustment.

    Parameters
    ----------
    returns : Array
        Historical returns
    confidence_level : float
        Confidence level

    Returns
    -------
    float
        Cornish-Fisher VaR

    Notes
    -----
    Modified VaR that adjusts for non-normality using Cornish-Fisher expansion:
    z_CF = z + (z²-1)*S/6 + (z³-3z)*(K-3)/24 - (2z³-5z)*S²/36

    where:
    - z is normal quantile
    - S is skewness
    - K is kurtosis
    """
    mean = float(jnp.mean(returns))
    std = float(jnp.std(returns))

    # Calculate moments
    standardized = (returns - mean) / std
    skewness = float(jnp.mean(standardized**3))
    kurtosis = float(jnp.mean(standardized**4))

    # Normal quantile
    z = stats.norm.ppf(confidence_level)

    # Cornish-Fisher adjustment
    z_cf = (
        z
        + (z**2 - 1) * skewness / 6.0
        + (z**3 - 3 * z) * (kurtosis - 3) / 24.0
        - (2 * z**3 - 5 * z) * skewness**2 / 36.0
    )

    # VaR with adjustment
    var = -mean + std * z_cf

    return float(var)


def calculate_var(
    returns: Array,
    confidence_level: float = 0.95,
    method: VaRMethod = VaRMethod.HISTORICAL,
    **kwargs,
) -> float:
    """Calculate VaR using specified methodology.

    Parameters
    ----------
    returns : Array
        Returns or P&L distribution
    confidence_level : float
        Confidence level
    method : VaRMethod
        VaR calculation method
    **kwargs
        Additional method-specific parameters

    Returns
    -------
    float
        Value at Risk

    Examples
    --------
    >>> returns = jnp.array([-0.02, 0.01, -0.01, 0.03, -0.015])
    >>> var_hist = calculate_var(returns, 0.95, VaRMethod.HISTORICAL)
    >>> var_param = calculate_var(returns, 0.95, VaRMethod.PARAMETRIC)
    """
    if method == VaRMethod.HISTORICAL:
        return historical_var(returns, confidence_level, **kwargs)
    elif method == VaRMethod.PARAMETRIC:
        return parametric_var(returns, confidence_level, **kwargs)
    elif method == VaRMethod.MONTE_CARLO:
        return monte_carlo_var(returns, confidence_level)
    elif method == VaRMethod.CORNISH_FISHER:
        return cornish_fisher_var(returns, confidence_level)
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def incremental_var(
    portfolio_returns: Array,
    position_returns: Array,
    confidence_level: float = 0.95,
) -> float:
    """Calculate incremental VaR of adding a position to portfolio.

    Parameters
    ----------
    portfolio_returns : Array
        Current portfolio returns
    position_returns : Array
        Returns of position to add
    confidence_level : float
        Confidence level

    Returns
    -------
    float
        Incremental VaR (difference in VaR)

    Notes
    -----
    IVaR = VaR(portfolio + position) - VaR(portfolio)
    Measures marginal risk contribution of adding the position
    """
    var_current = value_at_risk(portfolio_returns, confidence_level)
    combined_returns = portfolio_returns + position_returns
    var_new = value_at_risk(combined_returns, confidence_level)

    return var_new - var_current


def component_var(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95,
) -> Array:
    """Calculate component VaR for each position.

    Parameters
    ----------
    positions : Array
        Position weights, shape [n_assets]
    returns_scenarios : Array
        Scenario returns, shape [n_scenarios, n_assets]
    confidence_level : float
        Confidence level

    Returns
    -------
    Array
        Component VaR for each position, shape [n_assets]

    Notes
    -----
    Component VaR decomposes total portfolio VaR into contributions
    from each position. Sum of component VaRs equals total VaR.
    """
    # Portfolio returns
    portfolio_returns = jnp.dot(returns_scenarios, positions)
    portfolio_var = value_at_risk(portfolio_returns, confidence_level)

    # Calculate beta of each asset to portfolio
    portfolio_std = jnp.std(portfolio_returns)

    component_vars = []
    for i in range(len(positions)):
        asset_returns = returns_scenarios[:, i]
        # Beta = Cov(asset, portfolio) / Var(portfolio)
        covariance = jnp.mean(
            (asset_returns - jnp.mean(asset_returns))
            * (portfolio_returns - jnp.mean(portfolio_returns))
        )
        beta = covariance / (portfolio_std**2 + 1e-10)

        # Component VaR = position * beta * VaR
        comp_var = positions[i] * beta * portfolio_var
        component_vars.append(comp_var)

    return jnp.array(component_vars)


def marginal_var(
    positions: Array,
    returns_scenarios: Array,
    confidence_level: float = 0.95,
    delta: float = 0.01,
) -> Array:
    """Calculate marginal VaR (sensitivity of VaR to position changes).

    Parameters
    ----------
    positions : Array
        Position weights, shape [n_assets]
    returns_scenarios : Array
        Scenario returns, shape [n_scenarios, n_assets]
    confidence_level : float
        Confidence level
    delta : float
        Small change in position for finite difference

    Returns
    -------
    Array
        Marginal VaR for each position, shape [n_assets]

    Notes
    -----
    Marginal VaR = ∂VaR/∂position_i
    Measures how VaR changes with small increase in position
    """
    marginal_vars = []

    for i in range(len(positions)):
        # Increase position i by delta
        positions_plus = positions.at[i].add(delta)
        var_plus = portfolio_var(positions_plus, returns_scenarios, confidence_level)

        # Decrease position i by delta
        positions_minus = positions.at[i].add(-delta)
        var_minus = portfolio_var(positions_minus, returns_scenarios, confidence_level)

        # Finite difference
        mvar = (var_plus - var_minus) / (2 * delta)
        marginal_vars.append(mvar)

    return jnp.array(marginal_vars)


def backtest_var(
    realized_returns: Array,
    var_forecasts: Array,
    confidence_level: float = 0.95,
) -> dict:
    """Backtest VaR model by comparing forecasts to realized returns.

    Parameters
    ----------
    realized_returns : Array
        Actual realized returns
    var_forecasts : Array
        Forecasted VaR values (same length as realized_returns)
    confidence_level : float
        Confidence level used for VaR

    Returns
    -------
    dict
        Backtest results including:
        - violations: number of times loss exceeded VaR
        - violation_rate: percentage of violations
        - expected_violations: expected number based on confidence level
        - kupiec_pvalue: p-value from Kupiec test
        - pass_backtest: whether model passes at 5% significance

    Notes
    -----
    VaR should be exceeded (1 - confidence_level) % of the time.
    Kupiec test checks if violation rate is statistically consistent.
    """
    # Convert returns to losses (positive = loss)
    losses = -realized_returns

    # Count violations (losses exceeding VaR)
    violations = jnp.sum(losses > var_forecasts)
    n = len(realized_returns)

    violation_rate = float(violations / n)
    expected_rate = 1 - confidence_level
    expected_violations = n * expected_rate

    # Kupiec Likelihood Ratio test
    if violations == 0:
        lr_stat = 0.0
    elif violations == n:
        lr_stat = float("inf")
    else:
        # LR = -2 * ln((p^x * (1-p)^(n-x)) / (f^x * (1-f)^(n-x)))
        # where p = expected rate, f = observed rate
        p = expected_rate
        f = violation_rate
        x = violations

        lr_stat = -2 * (
            x * jnp.log(p / f) + (n - x) * jnp.log((1 - p) / (1 - f))
        )

    # P-value from chi-squared distribution (df=1)
    pvalue = 1 - stats.chi2.cdf(float(lr_stat), df=1)

    # Pass if p-value > 0.05 (fail to reject null hypothesis)
    pass_backtest = pvalue > 0.05

    return {
        "violations": int(violations),
        "violation_rate": violation_rate,
        "expected_violations": expected_violations,
        "expected_rate": expected_rate,
        "kupiec_lr_stat": float(lr_stat),
        "kupiec_pvalue": pvalue,
        "pass_backtest": pass_backtest,
    }


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
    # Enhanced VaR methods
    "VaRMethod",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "cornish_fisher_var",
    "calculate_var",
    "incremental_var",
    "component_var",
    "marginal_var",
    "backtest_var",
]
