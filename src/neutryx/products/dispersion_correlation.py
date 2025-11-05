"""Dispersion and Correlation Trading Products.

This module implements comprehensive dispersion and correlation trading strategies:
- Index variance swaps
- Single-name variance swaps
- Correlation swaps
- Dispersion trading strategies
- Implied vs realized correlation trades

These products are widely used by equity derivatives desks for volatility arbitrage
and correlation trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import Array

from .base import PathProduct


# ============================================================================
# Index Variance Swaps
# ============================================================================


@dataclass
class IndexVarianceSwap(PathProduct):
    """Index variance swap.

    A variance swap on an equity index where the buyer receives realized
    variance and pays the strike variance. Commonly used for index volatility
    exposure.

    Attributes
    ----------
    T : float
        Time to maturity in years
    strike_variance : float
        Strike variance (in variance units, e.g., 0.04 for 20% vol)
    notional_per_point : float
        Notional per variance point (vega notional = notional / (2*sqrt(K_var)))
    annualization_factor : float
        Factor to annualize variance (252 for daily observations)
    cap : float or None
        Optional cap on realized variance
    floor : float or None
        Optional floor on realized variance

    Notes
    -----
    Variance swap payoff:
        P&L = Notional * (Realized_Var - Strike_Var)

    Where realized variance is calculated as:
        σ²_realized = (252/N) * Σ[ln(S_i / S_{i-1})]²

    Common index variance swaps:
    - SPX (S&P 500) - most liquid
    - NDX (Nasdaq 100)
    - SX5E (Euro Stoxx 50)
    - NKY (Nikkei 225)

    Example
    -------
    >>> # SPX variance swap, strike at 20% vol (0.04 variance)
    >>> var_swap = IndexVarianceSwap(
    ...     T=0.25,  # 3 months
    ...     strike_variance=0.04,
    ...     notional_per_point=50_000.0,  # $50k per variance point
    ...     annualization_factor=252.0
    ... )
    >>> # If realized variance is 0.05 (22.4% vol), profit per variance point
    >>> # P&L = 50,000 * (0.05 - 0.04) = $500
    """

    T: float
    strike_variance: float
    notional_per_point: float = 50_000.0
    annualization_factor: float = 252.0
    cap: Optional[float] = None
    floor: Optional[float] = None

    def payoff_path(self, path: Array) -> Array:
        """Calculate variance swap payoff from index path.

        Parameters
        ----------
        path : Array
            Index price path

        Returns
        -------
        Array
            Variance swap payoff
        """
        path = jnp.asarray(path)

        # Calculate log returns
        log_returns = jnp.diff(jnp.log(path))

        # Realized variance (annualized)
        n_obs = len(log_returns)
        realized_var = (self.annualization_factor / n_obs) * jnp.sum(log_returns ** 2)

        # Apply cap/floor if specified
        if self.cap is not None:
            realized_var = jnp.minimum(realized_var, self.cap)
        if self.floor is not None:
            realized_var = jnp.maximum(realized_var, self.floor)

        # Payoff
        variance_diff = realized_var - self.strike_variance
        payoff = self.notional_per_point * variance_diff

        return payoff

    def vega_notional(self) -> float:
        """Calculate vega notional (volatility sensitivity).

        Returns
        -------
        float
            Vega notional

        Notes
        -----
        Vega notional = Variance notional / (2 * sqrt(strike_variance))

        This converts variance notional to volatility space.
        """
        strike_vol = jnp.sqrt(self.strike_variance)
        return self.notional_per_point / (2.0 * strike_vol)


# ============================================================================
# Single-Name Variance Swaps
# ============================================================================


@dataclass
class SingleNameVarianceSwap(PathProduct):
    """Single-name (single-stock) variance swap.

    A variance swap on an individual stock. Similar to index variance swaps
    but typically have higher volatility and different liquidity profiles.

    Attributes
    ----------
    T : float
        Time to maturity in years
    strike_variance : float
        Strike variance
    notional_per_point : float
        Notional per variance point
    annualization_factor : float
        Factor to annualize variance
    cap : float or None
        Optional cap on realized variance
    floor : float or None
        Optional floor on realized variance
    ticker : str
        Stock ticker symbol (for reference)

    Notes
    -----
    Single-stock variance swaps are exposed to:
    - Idiosyncratic volatility (company-specific risk)
    - Earnings announcements
    - Corporate events (M&A, restructuring)
    - Liquidity risk

    They typically trade at wider bid-ask spreads than index variance swaps.

    Example
    -------
    >>> # AAPL variance swap
    >>> var_swap = SingleNameVarianceSwap(
    ...     T=0.5,
    ...     strike_variance=0.06,  # 24.5% vol
    ...     notional_per_point=10_000.0,
    ...     ticker="AAPL"
    ... )
    """

    T: float
    strike_variance: float
    notional_per_point: float = 10_000.0
    annualization_factor: float = 252.0
    cap: Optional[float] = None
    floor: Optional[float] = None
    ticker: str = ""

    def payoff_path(self, path: Array) -> Array:
        """Calculate variance swap payoff from stock path.

        Parameters
        ----------
        path : Array
            Stock price path

        Returns
        -------
        Array
            Variance swap payoff
        """
        path = jnp.asarray(path)

        # Calculate log returns
        log_returns = jnp.diff(jnp.log(path))

        # Realized variance (annualized)
        n_obs = len(log_returns)
        realized_var = (self.annualization_factor / n_obs) * jnp.sum(log_returns ** 2)

        # Apply cap/floor if specified
        if self.cap is not None:
            realized_var = jnp.minimum(realized_var, self.cap)
        if self.floor is not None:
            realized_var = jnp.maximum(realized_var, self.floor)

        # Payoff
        variance_diff = realized_var - self.strike_variance
        payoff = self.notional_per_point * variance_diff

        return payoff

    def vega_notional(self) -> float:
        """Calculate vega notional."""
        strike_vol = jnp.sqrt(self.strike_variance)
        return self.notional_per_point / (2.0 * strike_vol)


# ============================================================================
# Correlation Swaps
# ============================================================================


@dataclass
class CorrelationSwap(PathProduct):
    """Correlation swap on pair-wise correlation between assets.

    A swap where the buyer receives realized correlation and pays
    strike correlation. Used for pure correlation exposure without
    variance exposure.

    Attributes
    ----------
    T : float
        Time to maturity in years
    strike_correlation : float
        Strike correlation (e.g., 0.50 for 50% correlation)
    notional_per_point : float
        Notional per correlation point (often $100k per 1% correlation)
    num_assets : int
        Number of assets in correlation basket
    correlation_type : str
        'average' for average pairwise correlation,
        'specific' for specific pair
    asset_1_idx : int
        Index of first asset (for specific pair correlation)
    asset_2_idx : int
        Index of second asset (for specific pair correlation)

    Notes
    -----
    Correlation swap payoff:
        P&L = Notional * (Realized_Corr - Strike_Corr) * 100

    Realized correlation for a pair:
        ρ_{12} = Σ(r1_i * r2_i) / sqrt(Σr1_i² * Σr2_i²)

    Average pairwise correlation:
        ρ_avg = (2/(N*(N-1))) * ΣΣ_{i<j} ρ_{ij}

    Correlation swaps are pure correlation plays, hedging out
    variance exposure through delta hedging.

    Example
    -------
    >>> # Average correlation swap on 5 stocks
    >>> corr_swap = CorrelationSwap(
    ...     T=1.0,
    ...     strike_correlation=0.50,
    ...     notional_per_point=100_000.0,
    ...     num_assets=5,
    ...     correlation_type='average'
    ... )
    >>> # If realized correlation is 60%, P&L = 100k * (0.60 - 0.50) * 100 = $1M
    """

    T: float
    strike_correlation: float
    notional_per_point: float = 100_000.0
    num_assets: int = 2
    correlation_type: str = 'average'  # 'average' or 'specific'
    asset_1_idx: int = 0
    asset_2_idx: int = 1

    def payoff_path(self, paths: Array) -> Array:
        """Calculate correlation swap payoff.

        Parameters
        ----------
        paths : Array
            Array of shape (num_assets, num_steps) with paths for all assets

        Returns
        -------
        Array
            Correlation swap payoff
        """
        paths = jnp.asarray(paths)

        if paths.ndim == 1:
            # Single asset, no correlation
            return 0.0

        # Calculate log returns for each asset
        log_returns = jnp.diff(jnp.log(paths), axis=1)

        if self.correlation_type == 'specific':
            # Specific pair correlation
            returns_1 = log_returns[self.asset_1_idx, :]
            returns_2 = log_returns[self.asset_2_idx, :]

            # Pearson correlation
            mean_1 = jnp.mean(returns_1)
            mean_2 = jnp.mean(returns_2)

            cov = jnp.mean((returns_1 - mean_1) * (returns_2 - mean_2))
            var_1 = jnp.mean((returns_1 - mean_1) ** 2)
            var_2 = jnp.mean((returns_2 - mean_2) ** 2)

            realized_corr = cov / jnp.sqrt(var_1 * var_2 + 1e-10)

        else:  # 'average'
            # Average pairwise correlation
            n_assets = paths.shape[0]

            # Calculate correlation matrix
            # Center returns
            returns_centered = log_returns - jnp.mean(log_returns, axis=1, keepdims=True)

            # Covariance matrix
            cov_matrix = jnp.dot(returns_centered, returns_centered.T) / returns_centered.shape[1]

            # Standard deviations
            stds = jnp.sqrt(jnp.diag(cov_matrix) + 1e-10)

            # Correlation matrix
            corr_matrix = cov_matrix / jnp.outer(stds, stds)

            # Average off-diagonal correlations
            # Sum all correlations, subtract diagonal (which is n_assets * 1)
            # Divide by number of pairs: n*(n-1)/2
            n_pairs = n_assets * (n_assets - 1) / 2
            realized_corr = (jnp.sum(corr_matrix) - n_assets) / (2 * n_pairs)

        # Payoff (multiply by 100 for percentage points)
        correlation_diff = (realized_corr - self.strike_correlation)
        payoff = self.notional_per_point * correlation_diff * 100

        return payoff


# ============================================================================
# Dispersion Trading Strategies
# ============================================================================


@dataclass
class BasicDispersionStrategy(PathProduct):
    """Basic dispersion trade: Long single-stock variance, short index variance.

    The classic dispersion trade profits when individual stock volatilities
    are higher than implied by index volatility (i.e., low correlation).

    Attributes
    ----------
    T : float
        Time to maturity in years
    index_strike_var : float
        Strike variance for index variance swap (short)
    stock_strike_var : Array
        Strike variances for each stock variance swap (long)
    index_notional : float
        Notional for index variance swap
    stock_notionals : Array
        Notionals for each stock variance swap
    num_stocks : int
        Number of stocks
    index_weights : Array or None
        Weights of stocks in index

    Notes
    -----
    Dispersion trade P&L:
        P&L = Σ(Stock_Var_PnL) - Index_Var_PnL

    The trade profits when:
    - Realized correlation is lower than implied correlation
    - Individual stocks realize higher vol than expected
    - Index realizes lower vol than expected

    Position sizing typically aims for equal vega across stocks
    and the index leg.

    Example
    -------
    >>> # Dispersion on 10 stocks
    >>> dispersion = BasicDispersionStrategy(
    ...     T=0.25,
    ...     index_strike_var=0.04,  # 20% index vol
    ...     stock_strike_var=jnp.array([0.06] * 10),  # 24.5% stock vol
    ...     index_notional=500_000.0,
    ...     stock_notionals=jnp.array([50_000.0] * 10)
    ... )
    """

    T: float
    index_strike_var: float
    stock_strike_var: Array
    index_notional: float
    stock_notionals: Array
    num_stocks: int = 10
    index_weights: Optional[Array] = None
    annualization_factor: float = 252.0

    def __post_init__(self):
        if self.index_weights is None:
            # Equal weights if not specified
            object.__setattr__(self, 'index_weights',
                             jnp.ones(self.num_stocks) / self.num_stocks)

        self.stock_strike_var = jnp.asarray(self.stock_strike_var)
        self.stock_notionals = jnp.asarray(self.stock_notionals)
        self.index_weights = jnp.asarray(self.index_weights)

    def payoff_path(self, paths: Array) -> Array:
        """Calculate dispersion strategy payoff.

        Parameters
        ----------
        paths : Array
            Array of shape (num_stocks+1, num_steps)
            First num_stocks rows are individual stocks, last row is index

        Returns
        -------
        Array
            Total dispersion strategy P&L
        """
        paths = jnp.asarray(paths)

        if paths.ndim != 2:
            return 0.0

        # Separate stock paths and index path
        stock_paths = paths[:-1, :]
        index_path = paths[-1, :]

        # Calculate stock variance P&Ls (long positions)
        stock_log_returns = jnp.diff(jnp.log(stock_paths), axis=1)
        n_obs = stock_log_returns.shape[1]

        stock_realized_vars = (self.annualization_factor / n_obs) * jnp.sum(
            stock_log_returns ** 2, axis=1
        )

        stock_var_pnls = self.stock_notionals * (
            stock_realized_vars - self.stock_strike_var
        )
        total_stock_pnl = jnp.sum(stock_var_pnls)

        # Calculate index variance P&L (short position)
        index_log_returns = jnp.diff(jnp.log(index_path))
        index_realized_var = (self.annualization_factor / n_obs) * jnp.sum(
            index_log_returns ** 2
        )

        index_var_pnl = self.index_notional * (
            index_realized_var - self.index_strike_var
        )

        # Total P&L (long stocks, short index)
        total_pnl = total_stock_pnl - index_var_pnl

        return total_pnl


@dataclass
class ImpliedCorrelationStrategy(PathProduct):
    """Implied vs realized correlation strategy.

    Trade that isolates correlation by hedging out variance exposure.
    Profits when realized correlation differs from implied correlation.

    Attributes
    ----------
    T : float
        Time to maturity in years
    implied_correlation : float
        Implied correlation from market prices
    index_var_strike : float
        Index variance strike
    avg_stock_var_strike : float
        Average stock variance strike
    notional_per_corr_point : float
        Notional per correlation point
    num_stocks : int
        Number of stocks in basket

    Notes
    -----
    Implied correlation can be extracted from variance swap prices:

        σ²_index = ρ_impl * σ²_avg_stock + (1-ρ_impl) * σ²_idiosyncratic

    Simplifying (assuming equal stock vols):

        ρ_impl ≈ (σ²_index) / (σ²_avg_stock)

    Strategy P&L:
        P&L = Notional * (ρ_realized - ρ_implied) * 10000 (bps)

    Example
    -------
    >>> # Trade implied vs realized correlation
    >>> corr_strat = ImpliedCorrelationStrategy(
    ...     T=0.5,
    ...     implied_correlation=0.55,  # 55% from market
    ...     index_var_strike=0.04,
    ...     avg_stock_var_strike=0.06,
    ...     notional_per_corr_point=200_000.0,
    ...     num_stocks=20
    ... )
    """

    T: float
    implied_correlation: float
    index_var_strike: float
    avg_stock_var_strike: float
    notional_per_corr_point: float = 200_000.0
    num_stocks: int = 10
    annualization_factor: float = 252.0

    def payoff_path(self, paths: Array) -> Array:
        """Calculate implied correlation strategy payoff.

        Parameters
        ----------
        paths : Array
            Array of shape (num_stocks+1, num_steps)
            Last row is index

        Returns
        -------
        Array
            Correlation strategy P&L
        """
        paths = jnp.asarray(paths)

        if paths.ndim != 2:
            return 0.0

        # Separate stock and index paths
        stock_paths = paths[:-1, :]
        index_path = paths[-1, :]

        # Calculate realized variances
        stock_log_returns = jnp.diff(jnp.log(stock_paths), axis=1)
        index_log_returns = jnp.diff(jnp.log(index_path))

        n_obs = stock_log_returns.shape[1]

        stock_realized_vars = (self.annualization_factor / n_obs) * jnp.sum(
            stock_log_returns ** 2, axis=1
        )
        avg_stock_var = jnp.mean(stock_realized_vars)

        index_realized_var = (self.annualization_factor / n_obs) * jnp.sum(
            index_log_returns ** 2
        )

        # Calculate realized correlation
        # For equal-weighted index: σ²_index ≈ ρ * σ²_avg
        realized_correlation = index_realized_var / (avg_stock_var + 1e-10)
        realized_correlation = jnp.clip(realized_correlation, 0.0, 1.0)

        # P&L from correlation difference
        corr_diff = realized_correlation - self.implied_correlation
        pnl = self.notional_per_corr_point * corr_diff * 10000  # Convert to bps

        return pnl


@dataclass
class RealizedCorrelationDispersion(PathProduct):
    """Realized correlation-based dispersion strategy.

    Advanced dispersion trade that explicitly tracks realized correlation
    and adjusts exposures dynamically.

    Attributes
    ----------
    T : float
        Time to maturity in years
    target_correlation : float
        Target correlation for the strategy
    index_notional : float
        Index variance swap notional
    stock_notionals : Array
        Stock variance swap notionals
    correlation_notional : float
        Additional notional for correlation exposure
    num_stocks : int
        Number of stocks

    Notes
    -----
    This strategy combines:
    1. Dispersion trade (long stocks, short index variance)
    2. Explicit correlation overlay
    3. Dynamic rebalancing based on realized correlation

    Total P&L = Dispersion_PnL + Correlation_PnL

    Example
    -------
    >>> # Advanced dispersion with correlation overlay
    >>> strat = RealizedCorrelationDispersion(
    ...     T=1.0,
    ...     target_correlation=0.50,
    ...     index_notional=1_000_000.0,
    ...     stock_notionals=jnp.array([100_000.0] * 10),
    ...     correlation_notional=500_000.0,
    ...     num_stocks=10
    ... )
    """

    T: float
    target_correlation: float
    index_notional: float
    stock_notionals: Array
    correlation_notional: float = 500_000.0
    num_stocks: int = 10
    annualization_factor: float = 252.0

    def __post_init__(self):
        self.stock_notionals = jnp.asarray(self.stock_notionals)

    def payoff_path(self, paths: Array) -> Array:
        """Calculate realized correlation dispersion payoff.

        Parameters
        ----------
        paths : Array
            Array of shape (num_stocks+1, num_steps)
            Last row is index

        Returns
        -------
        Array
            Total strategy P&L
        """
        paths = jnp.asarray(paths)

        if paths.ndim != 2:
            return 0.0

        # Separate stock and index paths
        stock_paths = paths[:-1, :]
        index_path = paths[-1, :]

        # Calculate returns
        stock_log_returns = jnp.diff(jnp.log(stock_paths), axis=1)
        index_log_returns = jnp.diff(jnp.log(index_path))

        # Calculate average pairwise correlation
        returns_centered = stock_log_returns - jnp.mean(
            stock_log_returns, axis=1, keepdims=True
        )
        cov_matrix = jnp.dot(returns_centered, returns_centered.T) / returns_centered.shape[1]
        stds = jnp.sqrt(jnp.diag(cov_matrix) + 1e-10)
        corr_matrix = cov_matrix / jnp.outer(stds, stds)

        n_assets = stock_paths.shape[0]
        n_pairs = n_assets * (n_assets - 1) / 2
        realized_correlation = (jnp.sum(corr_matrix) - n_assets) / (2 * n_pairs)

        # Correlation P&L
        corr_diff = realized_correlation - self.target_correlation
        correlation_pnl = self.correlation_notional * corr_diff * 10000

        # Return total P&L (correlation component only for now)
        # In practice, this would be combined with dispersion P&L
        return correlation_pnl


__all__ = [
    "IndexVarianceSwap",
    "SingleNameVarianceSwap",
    "CorrelationSwap",
    "BasicDispersionStrategy",
    "ImpliedCorrelationStrategy",
    "RealizedCorrelationDispersion",
]
