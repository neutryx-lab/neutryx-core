"""Hybrid and multi-asset products.

This module implements hybrid products combining multiple asset classes:
- Quanto CDS
- FX-hybrid equity options
- Inflation-linked FX options
- Cross-currency exotics
- Multi-asset worst-of/best-of options
- Correlation swaps
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax
import jax.numpy as jnp

from neutryx.products.base import Product, PathProduct

Array = jnp.ndarray


@dataclass
class QuantoCDS(Product):
    """Quanto CDS - Credit default swap with FX quanto feature.

    CDS where protection payment is in a different currency than the reference
    obligation, without FX conversion (fixed quanto rate).

    Args:
        T: Time to maturity (years)
        notional_domestic: Notional in domestic currency
        spread: CDS spread (bps)
        recovery_rate: Recovery rate on default
        quanto_fx_rate: Fixed FX rate for quanto conversion
        fx_vol: Volatility of FX rate
        credit_fx_correlation: Correlation between credit and FX
        coupon_freq: Coupon payment frequency per year
    """

    T: float
    notional_domestic: float
    spread: float
    recovery_rate: float = 0.4
    quanto_fx_rate: float = 1.0
    fx_vol: float = 0.1
    credit_fx_correlation: float = -0.3  # Typical negative correlation
    coupon_freq: int = 4

    def payoff_terminal(self, state: Array) -> Array:
        """Calculate payoff given terminal state.

        Args:
            state: Array [default_indicator, fx_rate]

        Returns:
            Quanto-adjusted CDS payoff
        """
        default_occurred = state[0] > 0.5
        fx_rate = state[1] if state.shape[0] > 1 else self.quanto_fx_rate

        # Premium leg (in domestic currency)
        dt = 1.0 / self.coupon_freq
        num_payments = int(self.T * self.coupon_freq)
        premium_leg = self.spread * 1e-4 * num_payments * dt

        # Protection leg (quanto-adjusted)
        protection_leg = jnp.where(
            default_occurred,
            (1.0 - self.recovery_rate) * self.quanto_fx_rate,  # Fixed quanto rate
            0.0
        )

        # Quanto adjustment for correlation
        quanto_adjustment = jnp.exp(
            self.credit_fx_correlation * self.fx_vol * jnp.sqrt(self.T)
        )

        return self.notional_domestic * quanto_adjustment * (protection_leg - premium_leg)


@dataclass
class FXHybridEquityOption(Product):
    """FX-hybrid equity option.

    Equity option with payoff in foreign currency. Combines equity exposure
    with FX exposure.

    Args:
        T: Time to maturity (years)
        K_equity: Strike price for equity
        K_fx: Strike for FX rate (if applicable)
        option_type: 'call' or 'put' on equity
        notional: Notional amount
        fx_protected: If True, includes FX hedge at K_fx
        equity_fx_correlation: Correlation between equity and FX
    """

    T: float
    K_equity: float
    K_fx: float = 1.0
    option_type: Literal['call', 'put'] = 'call'
    notional: float = 1.0
    fx_protected: bool = False
    equity_fx_correlation: float = 0.0

    def payoff_terminal(self, state: Array) -> Array:
        """Calculate payoff given terminal state.

        Args:
            state: Array [equity_price, fx_rate]

        Returns:
            Payoff in domestic currency
        """
        equity_price = state[0]
        fx_rate = state[1] if state.shape[0] > 1 else 1.0

        # Equity payoff in foreign currency
        if self.option_type == 'call':
            equity_payoff = jnp.maximum(equity_price - self.K_equity, 0.0)
        else:
            equity_payoff = jnp.maximum(self.K_equity - equity_price, 0.0)

        # Convert to domestic currency
        if self.fx_protected:
            # FX-protected: convert at strike FX rate
            domestic_payoff = equity_payoff * self.K_fx
        else:
            # Unprotected: convert at spot FX rate
            domestic_payoff = equity_payoff * fx_rate

        return self.notional * domestic_payoff


@dataclass
class InflationLinkedFXOption(Product):
    """Inflation-linked FX option.

    FX option where strike or payoff is adjusted for inflation differential
    between two currencies.

    Args:
        T: Time to maturity (years)
        K: FX strike rate
        option_type: 'call' or 'put'
        notional: Notional amount
        domestic_inflation_rate: Expected domestic inflation rate
        foreign_inflation_rate: Expected foreign inflation rate
        inflation_adjustment_type: 'strike' or 'payoff'
    """

    T: float
    K: float
    option_type: Literal['call', 'put'] = 'call'
    notional: float = 1.0
    domestic_inflation_rate: float = 0.02
    foreign_inflation_rate: float = 0.02
    inflation_adjustment_type: Literal['strike', 'payoff'] = 'strike'

    def payoff_terminal(self, state: Array) -> Array:
        """Calculate payoff given terminal state.

        Args:
            state: Array [fx_rate, domestic_cpi, foreign_cpi]

        Returns:
            Inflation-adjusted payoff
        """
        fx_rate = state[0]
        domestic_cpi = state[1] if state.shape[0] > 1 else jnp.exp(
            self.domestic_inflation_rate * self.T
        )
        foreign_cpi = state[2] if state.shape[0] > 2 else jnp.exp(
            self.foreign_inflation_rate * self.T
        )

        # Inflation adjustment factor
        inflation_ratio = domestic_cpi / foreign_cpi

        if self.inflation_adjustment_type == 'strike':
            # Adjust strike by inflation differential
            adjusted_strike = self.K * inflation_ratio
            if self.option_type == 'call':
                payoff = jnp.maximum(fx_rate - adjusted_strike, 0.0)
            else:
                payoff = jnp.maximum(adjusted_strike - fx_rate, 0.0)
        else:  # 'payoff'
            # Standard payoff, then adjust by inflation
            if self.option_type == 'call':
                payoff = jnp.maximum(fx_rate - self.K, 0.0) * inflation_ratio
            else:
                payoff = jnp.maximum(self.K - fx_rate, 0.0) * inflation_ratio

        return self.notional * payoff


@dataclass
class CrossCurrencyExotic(PathProduct):
    """Cross-currency exotic option.

    Generic exotic option involving multiple currencies, such as:
    - Cross-currency barrier options
    - Currency basket options
    - Multi-FX digital options

    Args:
        T: Time to maturity (years)
        strikes: Array of strikes for each FX pair
        barrier: Optional barrier level
        barrier_type: 'down-and-out', 'up-and-out', 'down-and-in', 'up-and-in'
        option_type: 'call' or 'put'
        notional: Notional amount
        weights: Weights for basket (if basket option)
    """

    T: float
    strikes: Array
    barrier: Optional[float] = None
    barrier_type: str = 'down-and-out'
    option_type: Literal['call', 'put'] = 'call'
    notional: float = 1.0
    weights: Optional[Array] = None

    def payoff_path(self, path: Array) -> Array:
        """Calculate payoff given FX rate paths.

        Args:
            path: Array of shape (num_fx_pairs, num_steps) with FX rates

        Returns:
            Payoff considering barrier and basket features
        """
        if path.ndim == 1:
            # Single FX pair, terminal value
            return self._terminal_payoff(path[-1])

        # Multiple FX pairs or path dependency
        terminal_rates = path[:, -1] if path.ndim == 2 else path

        # Check barrier condition if applicable
        barrier_breached = False
        if self.barrier is not None:
            if 'down' in self.barrier_type:
                barrier_breached = jnp.any(path <= self.barrier)
            else:  # 'up'
                barrier_breached = jnp.any(path >= self.barrier)

        # Calculate basket value if weights provided
        if self.weights is not None:
            basket_value = jnp.sum(self.weights * terminal_rates)
            strike = jnp.sum(self.weights * self.strikes)
        else:
            basket_value = terminal_rates[0] if terminal_rates.ndim > 0 else terminal_rates
            strike = self.strikes[0] if self.strikes.ndim > 0 else self.strikes

        # Vanilla payoff
        if self.option_type == 'call':
            payoff = jnp.maximum(basket_value - strike, 0.0)
        else:
            payoff = jnp.maximum(strike - basket_value, 0.0)

        # Apply barrier logic
        if self.barrier is not None:
            if 'out' in self.barrier_type:
                payoff = jnp.where(barrier_breached, 0.0, payoff)
            else:  # 'in'
                payoff = jnp.where(barrier_breached, payoff, 0.0)

        return self.notional * payoff

    def _terminal_payoff(self, spot: float) -> float:
        """Helper for terminal payoff."""
        if self.option_type == 'call':
            return self.notional * jnp.maximum(spot - self.strikes[0], 0.0)
        else:
            return self.notional * jnp.maximum(self.strikes[0] - spot, 0.0)


@dataclass
class MultiAssetWorstOfBestOf(Product):
    """Multi-asset worst-of or best-of option.

    Option that pays based on the worst or best performing asset in a basket.

    Args:
        T: Time to maturity (years)
        strikes: Strike prices for each asset
        option_type: 'call' or 'put'
        payoff_type: 'worst-of' or 'best-of'
        notional: Notional amount
        participation_rate: Participation in performance (default 1.0)
        num_assets: Number of assets in basket
    """

    T: float
    strikes: Array
    option_type: Literal['call', 'put'] = 'call'
    payoff_type: Literal['worst-of', 'best-of'] = 'worst-of'
    notional: float = 1.0
    participation_rate: float = 1.0
    num_assets: int = 2

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate payoff given terminal asset prices.

        Args:
            spots: Array of terminal prices for each asset

        Returns:
            Worst-of or best-of payoff
        """
        # Calculate individual payoffs
        if self.option_type == 'call':
            individual_payoffs = jnp.maximum(spots - self.strikes, 0.0)
        else:
            individual_payoffs = jnp.maximum(self.strikes - spots, 0.0)

        # Select worst or best
        if self.payoff_type == 'worst-of':
            payoff = jnp.min(individual_payoffs)
        else:  # 'best-of'
            payoff = jnp.max(individual_payoffs)

        return self.notional * self.participation_rate * payoff

    def rainbow_payoff(self, spots: Array, rank: int) -> Array:
        """Calculate rainbow payoff for a specific rank.

        Args:
            spots: Array of terminal prices
            rank: Which ranked performance to pay (1=best, n=worst)

        Returns:
            Payoff based on ranked performance
        """
        # Calculate performances
        if self.option_type == 'call':
            performances = spots - self.strikes
        else:
            performances = self.strikes - spots

        # Sort performances (descending for call, ascending for put)
        if self.option_type == 'call':
            sorted_perfs = jnp.sort(performances)[::-1]
        else:
            sorted_perfs = jnp.sort(performances)

        # Get performance at specified rank (0-indexed)
        ranked_perf = sorted_perfs[rank - 1]

        return self.notional * self.participation_rate * jnp.maximum(ranked_perf, 0.0)


@dataclass
class CorrelationSwap(PathProduct):
    """Correlation swap.

    Swap that pays the difference between realized correlation and a strike.

    Args:
        T: Time to maturity (years)
        strike_correlation: Strike correlation
        notional: Notional amount per correlation point
        num_assets: Number of assets in correlation calculation
        observation_freq: Number of observations per year
    """

    T: float
    strike_correlation: float
    notional: float = 1000000.0  # Notional per correlation point
    num_assets: int = 2
    observation_freq: int = 252  # Daily observations

    def payoff_path(self, paths: Array) -> Array:
        """Calculate payoff given asset price paths.

        Args:
            paths: Array of shape (num_assets, num_steps) with price paths

        Returns:
            Payoff based on realized correlation
        """
        # Calculate returns
        if paths.ndim == 1:
            return 0.0  # Need multiple assets

        returns = jnp.diff(jnp.log(paths), axis=1)

        # Calculate realized correlation
        if self.num_assets == 2:
            # Simple pairwise correlation
            corr_matrix = jnp.corrcoef(returns)
            realized_corr = corr_matrix[0, 1]
        else:
            # Average pairwise correlation
            corr_matrix = jnp.corrcoef(returns)
            # Extract upper triangular (excluding diagonal)
            n = self.num_assets
            num_pairs = n * (n - 1) // 2
            upper_tri_indices = jnp.triu_indices(n, k=1)
            pairwise_corrs = corr_matrix[upper_tri_indices]
            realized_corr = jnp.mean(pairwise_corrs)

        # Payoff: (realized - strike) * notional
        correlation_diff = realized_corr - self.strike_correlation

        return self.notional * correlation_diff


@dataclass
class DispersionOption(Product):
    """Dispersion option for variance dispersion trading.

    Pays the difference between average single-stock variance and index variance.

    Args:
        T: Time to maturity (years)
        strike: Strike dispersion level
        notional: Notional amount
        num_stocks: Number of stocks in index
        index_weight: Weight of index variance (typically -1 for short)
    """

    T: float
    strike: float
    notional: float = 1000000.0
    num_stocks: int = 50
    index_weight: float = -1.0

    def payoff_terminal(self, variances: Array) -> Array:
        """Calculate payoff given realized variances.

        Args:
            variances: Array [stock_var_1, ..., stock_var_n, index_var]

        Returns:
            Dispersion payoff
        """
        # Split stock and index variances
        stock_variances = variances[:-1]
        index_variance = variances[-1]

        # Average stock variance
        avg_stock_variance = jnp.mean(stock_variances)

        # Dispersion = avg(stock variances) - index variance
        dispersion = avg_stock_variance + self.index_weight * index_variance

        # Payoff
        return self.notional * (dispersion - self.strike)
