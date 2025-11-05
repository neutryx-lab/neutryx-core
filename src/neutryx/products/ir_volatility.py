"""Interest rate volatility products.

This module implements trading products on interest rate volatility:
- Swaption straddles and strangles
- Volatility dispersion swaps
- Forward variance swaps on IR
- Caplet variance swaps
- Correlation swaps between rates
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from neutryx.products.base import Product, PathProduct


@dataclass
class SwaptionStraddle(Product):
    """Swaption straddle (ATM payer + receiver at same strike).

    A long swaption straddle profits from volatility in either direction.
    Payoff = max(S - K, 0) + max(K - S, 0) = |S - K|

    where S is the swap rate at expiry and K is the strike.

    Parameters
    ----------
    T : float
        Option expiry in years
    K : float
        Strike rate (typically ATM forward swap rate)
    annuity : float
        Present value of one basis point (swap annuity)
    notional : float
        Notional amount

    Notes
    -----
    The straddle is a pure volatility play. The payoff is always positive
    if S != K, with profit = |S - K| * annuity * notional.

    Greeks:
    - Delta: zero at ATM (offsetting payer/receiver deltas)
    - Gamma: positive (double that of a single option)
    - Vega: positive (sum of payer and receiver vegas)

    Example
    -------
    >>> straddle = SwaptionStraddle(T=1.0, K=0.05, annuity=4.5, notional=1_000_000)
    >>> # If swap rate moves to 5.5%
    >>> payoff = straddle.payoff_terminal(0.055)  # 50 bps * annuity * notional
    """

    T: float
    K: float
    annuity: float
    notional: float = 1_000_000.0

    def payoff_terminal(self, swap_rate: Array) -> Array:
        """Calculate straddle payoff at expiry.

        Parameters
        ----------
        swap_rate : Array
            Forward swap rate at option expiry

        Returns
        -------
        Array
            Payoff = |swap_rate - strike| * annuity * notional
        """
        swap_rate = jnp.asarray(swap_rate, dtype=jnp.float32)
        intrinsic = jnp.abs(swap_rate - self.K)
        return self.notional * self.annuity * intrinsic


@dataclass
class SwaptionStrangle(Product):
    """Swaption strangle (OTM payer + OTM receiver at different strikes).

    A strangle is cheaper than a straddle but requires larger rate moves to profit.
    Payoff = max(S - K_high, 0) + max(K_low - S, 0)

    Parameters
    ----------
    T : float
        Option expiry in years
    K_low : float
        Lower strike (receiver strike)
    K_high : float
        Upper strike (payer strike)
    annuity : float
        Present value of one basis point
    notional : float
        Notional amount

    Notes
    -----
    The strangle profits from large rate moves in either direction.
    It's cheaper than a straddle because both options are OTM.

    Typical setup:
    - K_low = ATM - 50 bps (receiver)
    - K_high = ATM + 50 bps (payer)

    Example
    -------
    >>> strangle = SwaptionStrangle(
    ...     T=1.0, K_low=0.045, K_high=0.055,
    ...     annuity=4.5, notional=1_000_000
    ... )
    >>> # Profit if swap rate moves outside [4.5%, 5.5%]
    >>> payoff_up = strangle.payoff_terminal(0.06)  # 50 bps profit
    >>> payoff_down = strangle.payoff_terminal(0.04)  # 50 bps profit
    >>> payoff_middle = strangle.payoff_terminal(0.05)  # Zero
    """

    T: float
    K_low: float
    K_high: float
    annuity: float
    notional: float = 1_000_000.0

    def payoff_terminal(self, swap_rate: Array) -> Array:
        """Calculate strangle payoff at expiry.

        Parameters
        ----------
        swap_rate : Array
            Forward swap rate at option expiry

        Returns
        -------
        Array
            Payoff from payer + receiver options
        """
        swap_rate = jnp.asarray(swap_rate, dtype=jnp.float32)

        # Payer payoff (upper strike)
        payer = jnp.maximum(swap_rate - self.K_high, 0.0)

        # Receiver payoff (lower strike)
        receiver = jnp.maximum(self.K_low - swap_rate, 0.0)

        return self.notional * self.annuity * (payer + receiver)


@dataclass
class IRVarianceSwap(PathProduct):
    """Interest rate variance swap.

    Variance swap on an interest rate (e.g., 3M LIBOR, 10Y swap rate).
    Payoff = (realized_variance - strike_variance) * notional_per_variance_point

    Parameters
    ----------
    T : float
        Maturity in years
    strike_variance : float
        Strike variance (annual, e.g., 0.04 for 20% vol squared)
    notional_per_variance_point : float
        Notional per variance point (vega notional)
    observation_frequency : int
        Number of observations per year (e.g., 252 for daily)

    Notes
    -----
    IR variance swaps allow traders to take pure exposure to interest rate
    volatility without delta risk.

    Realized variance is computed as:
        RV = (252/n) * Σ(r_i - r_{i-1})^2 / r_{i-1}^2

    where n is the number of observations.

    Example
    -------
    >>> var_swap = IRVarianceSwap(
    ...     T=1.0,
    ...     strike_variance=0.04,  # 20% vol squared
    ...     notional_per_variance_point=10_000
    ... )
    >>> # If realized variance is 25% (0.0625)
    >>> # P&L = (0.0625 - 0.04) * 10_000 = 225 profit
    """

    T: float
    strike_variance: float
    notional_per_variance_point: float = 10_000.0
    observation_frequency: int = 252

    def payoff_path(self, path: Array) -> Array:
        """Calculate variance swap payoff from rate path.

        Parameters
        ----------
        path : Array
            Array of interest rate observations

        Returns
        -------
        Array
            Payoff = (realized_variance - strike) * notional
        """
        path = jnp.asarray(path, dtype=jnp.float32)

        # Calculate log returns
        log_returns = jnp.log(path[1:] / path[:-1])

        # Realized variance (annualized)
        n_obs = len(log_returns)
        realized_var = (self.observation_frequency / n_obs) * jnp.sum(log_returns**2)

        # Payoff
        variance_diff = realized_var - self.strike_variance
        return self.notional_per_variance_point * variance_diff


@dataclass
class CapletVarianceSwap(PathProduct):
    """Variance swap on caplet implied volatilities.

    Pays out based on realized variance of caplet implied vols.
    This is a meta-volatility product - trading the volatility of volatility.

    Parameters
    ----------
    T : float
        Maturity in years
    strike_vol_variance : float
        Strike variance of implied vols
    notional_per_vol_variance_point : float
        Notional per vol variance point
    caplet_tenor : float
        Tenor of the caplet to track (e.g., 3M)

    Notes
    -----
    Caplet variance swaps allow trading on the stability of the vol surface.
    They profit when implied vols are more volatile than expected.

    Realized vol-of-vol is computed from a time series of caplet IVs.

    Example
    -------
    >>> caplet_var_swap = CapletVarianceSwap(
    ...     T=1.0,
    ...     strike_vol_variance=0.01,  # 10% vol-of-vol squared
    ...     notional_per_vol_variance_point=50_000,
    ...     caplet_tenor=0.25
    ... )
    """

    T: float
    strike_vol_variance: float
    notional_per_vol_variance_point: float = 50_000.0
    caplet_tenor: float = 0.25

    def payoff_path(self, path: Array) -> Array:
        """Calculate payoff from implied vol path.

        Parameters
        ----------
        path : Array
            Time series of caplet implied volatilities

        Returns
        -------
        Array
            Payoff based on realized vol variance
        """
        path = jnp.asarray(path, dtype=jnp.float32)

        # Calculate log returns of implied vols
        log_returns = jnp.log(path[1:] / path[:-1])

        # Realized variance of vols (annualized)
        n_obs = len(log_returns)
        realized_vol_var = (252 / n_obs) * jnp.sum(log_returns**2)

        # Payoff
        variance_diff = realized_vol_var - self.strike_vol_variance
        return self.notional_per_vol_variance_point * variance_diff


@dataclass
class ForwardIRVarianceSwap(Product):
    """Forward-starting interest rate variance swap.

    Variance swap that starts at a future date T1 and ends at T2.
    Payoff = (realized_variance[T1, T2] - strike) * notional

    Parameters
    ----------
    T : float
        End date in years (inherited from Product)
    T1 : float
        Forward start date in years
    strike_variance : float
        Strike variance (forward variance)
    notional_per_variance_point : float
        Notional per variance point

    Notes
    -----
    Forward variance swaps allow trading forward volatility.
    They are used to:
    - Take views on future volatility levels
    - Hedge forward volatility exposure
    - Arbitrage between spot and forward vol markets

    The forward variance relationship:
        σ²_fwd = (σ²_T2 * T2 - σ²_T1 * T1) / (T2 - T1)

    Example
    -------
    >>> fwd_var_swap = ForwardIRVarianceSwap(
    ...     T=2.0, T1=1.0,
    ...     strike_variance=0.0441,  # 21% forward vol
    ...     notional_per_variance_point=10_000
    ... )
    """

    T: float  # End of variance period (maturity)
    T1: float  # Start of variance period
    strike_variance: float
    notional_per_variance_point: float = 10_000.0

    def __post_init__(self):
        """Validate forward dates."""
        if self.T <= self.T1:
            raise ValueError(f"T ({self.T}) must be > T1 ({self.T1})")

    def payoff_terminal(self, realized_variance: Array) -> Array:
        """Calculate payoff from realized forward variance.

        Parameters
        ----------
        realized_variance : Array
            Realized variance over [T1, T2] period

        Returns
        -------
        Array
            Payoff based on variance difference
        """
        realized_variance = jnp.asarray(realized_variance, dtype=jnp.float32)
        variance_diff = realized_variance - self.strike_variance
        return self.notional_per_variance_point * variance_diff


@dataclass
class RateCorrelationSwap(PathProduct):
    """Correlation swap between two interest rates.

    Pays based on realized correlation between two rate paths.
    Payoff = (realized_correlation - strike_correlation) * notional

    Parameters
    ----------
    T : float
        Maturity in years
    strike_correlation : float
        Strike correlation (between -1 and 1)
    notional_per_correlation_point : float
        Notional per correlation point (e.g., per 1% correlation)

    Notes
    -----
    Rate correlation swaps allow trading on:
    - Correlation between different tenors (e.g., 2Y vs 10Y)
    - Correlation between different currencies
    - Correlation breakdown during stress

    Common pairs:
    - 2Y vs 10Y swap rates (curve steepening/flattening)
    - 3M LIBOR vs 10Y swap (short vs long rates)
    - USD vs EUR swap rates (cross-currency correlation)

    Example
    -------
    >>> corr_swap = RateCorrelationSwap(
    ...     T=1.0,
    ...     strike_correlation=0.7,  # Expect 70% correlation
    ...     notional_per_correlation_point=100_000
    ... )
    >>> # If realized correlation is 85%
    >>> # P&L = (0.85 - 0.70) * 100_000 = 15_000 profit
    """

    T: float
    strike_correlation: float
    notional_per_correlation_point: float = 100_000.0

    def payoff_path(self, path: Array) -> Array:
        """Calculate correlation swap payoff.

        Parameters
        ----------
        path : Array
            Array of shape (2, n_steps) containing two rate paths

        Returns
        -------
        Array
            Payoff based on realized correlation
        """
        path = jnp.asarray(path, dtype=jnp.float32)

        if path.ndim != 2 or path.shape[0] != 2:
            raise ValueError(
                f"Path must have shape (2, n_steps), got {path.shape}"
            )

        # Extract both rate paths
        rates1 = path[0, :]
        rates2 = path[1, :]

        # Calculate log returns
        returns1 = jnp.log(rates1[1:] / rates1[:-1])
        returns2 = jnp.log(rates2[1:] / rates2[:-1])

        # Realized correlation
        correlation = jnp.corrcoef(returns1, returns2)[0, 1]

        # Payoff
        correlation_diff = correlation - self.strike_correlation
        return self.notional_per_correlation_point * correlation_diff


@dataclass
class VolatilityDispersionSwap(PathProduct):
    """Volatility dispersion swap across rate tenors.

    Pays the difference between:
    - Average volatility of individual rates
    - Volatility of the average rate (or index)

    Payoff = (avg_individual_vol - index_vol) * notional

    Parameters
    ----------
    T : float
        Maturity in years
    strike_dispersion : float
        Strike dispersion level
    notional_per_dispersion_point : float
        Notional per dispersion point
    n_rates : int
        Number of rates in the basket

    Notes
    -----
    Dispersion swaps profit from de-correlation. They pay when:
    - Individual rates become more volatile
    - Correlation between rates decreases

    Common in:
    - Rate curve dispersion (2Y, 5Y, 10Y, 30Y)
    - Cross-currency rate dispersion
    - Caplet tenor dispersion

    Example
    -------
    >>> disp_swap = VolatilityDispersionSwap(
    ...     T=1.0,
    ...     strike_dispersion=0.05,
    ...     notional_per_dispersion_point=10_000,
    ...     n_rates=4  # 2Y, 5Y, 10Y, 30Y
    ... )
    """

    T: float
    strike_dispersion: float
    notional_per_dispersion_point: float = 10_000.0
    n_rates: int = 4

    def payoff_path(self, path: Array) -> Array:
        """Calculate dispersion swap payoff.

        Parameters
        ----------
        path : Array
            Array of shape (n_rates, n_steps) containing rate paths

        Returns
        -------
        Array
            Payoff based on realized dispersion
        """
        path = jnp.asarray(path, dtype=jnp.float32)

        if path.ndim != 2:
            raise ValueError(f"Path must be 2D, got shape {path.shape}")

        # Calculate realized vol for each rate
        individual_vols = []
        for i in range(path.shape[0]):
            returns = jnp.log(path[i, 1:] / path[i, :-1])
            vol = jnp.std(returns) * jnp.sqrt(252)  # Annualized
            individual_vols.append(vol)

        avg_individual_vol = jnp.mean(jnp.array(individual_vols))

        # Calculate vol of the equal-weighted index
        index = jnp.mean(path, axis=0)
        index_returns = jnp.log(index[1:] / index[:-1])
        index_vol = jnp.std(index_returns) * jnp.sqrt(252)

        # Realized dispersion
        realized_dispersion = avg_individual_vol - index_vol

        # Payoff
        dispersion_diff = realized_dispersion - self.strike_dispersion
        return self.notional_per_dispersion_point * dispersion_diff


def compute_forward_variance_strike(
    spot_variance_t1: float,
    spot_variance_t2: float,
    t1: float,
    t2: float,
) -> float:
    """Compute fair forward variance strike.

    Parameters
    ----------
    spot_variance_t1 : float
        Variance (vol²) for maturity T1
    spot_variance_t2 : float
        Variance (vol²) for maturity T2
    t1 : float
        First maturity in years
    t2 : float
        Second maturity in years

    Returns
    -------
    float
        Fair forward variance strike

    Notes
    -----
    Forward variance relationship:
        σ²_fwd(T1, T2) = [σ²(T2) * T2 - σ²(T1) * T1] / (T2 - T1)

    Example
    -------
    >>> # 1Y spot vol = 20%, 2Y spot vol = 22%
    >>> fwd_var = compute_forward_variance_strike(
    ...     spot_variance_t1=0.20**2,
    ...     spot_variance_t2=0.22**2,
    ...     t1=1.0,
    ...     t2=2.0
    ... )
    >>> fwd_vol = jnp.sqrt(fwd_var)  # Forward vol 1Y-2Y
    """
    if t2 <= t1:
        raise ValueError(f"t2 ({t2}) must be > t1 ({t1})")

    # Total variance for each maturity
    total_var_t1 = spot_variance_t1 * t1
    total_var_t2 = spot_variance_t2 * t2

    # Forward variance
    forward_variance = (total_var_t2 - total_var_t1) / (t2 - t1)

    return float(forward_variance)


__all__ = [
    'SwaptionStraddle',
    'SwaptionStrangle',
    'IRVarianceSwap',
    'CapletVarianceSwap',
    'ForwardIRVarianceSwap',
    'RateCorrelationSwap',
    'VolatilityDispersionSwap',
    'compute_forward_variance_strike',
]
