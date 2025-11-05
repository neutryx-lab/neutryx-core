"""Comprehensive FX Vanilla and Exotic Derivatives.

This module implements a complete suite of FX derivatives including:
- FX Forwards and Non-Deliverable Forwards
- Vanilla Options (European and American)
- Digital Options (Cash-or-Nothing and Asset-or-Nothing)
- Barrier Options (Single and Double, all variants)
- Asian Options (Arithmetic and Geometric averages)
- Lookback Options (Fixed and Floating strike)

All products follow the Garman-Kohlhagen model for FX option pricing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from .base import Product, PathProduct
from ._utils import compute_d1_d2_fx


# ============================================================================
# FX Forwards
# ============================================================================


@dataclass(frozen=True)
class FXForward:
    """FX Forward contract.

    A binding agreement to exchange currencies at a future date at a
    predetermined rate.

    Attributes
    ----------
    spot : float
        Current FX spot rate (domestic per foreign)
    forward_rate : float
        Agreed forward rate
    expiry : float
        Time to expiry in years
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    notional_foreign : float
        Notional amount in foreign currency
    is_long : bool
        True if long foreign currency (pay domestic, receive foreign)

    Notes
    -----
    Fair forward rate: F = S * exp((r_d - r_f) * T)

    Mark-to-market value:
    V = N_f * (F_market - F_contract) * exp(-r_d * T)

    where:
    - N_f = notional in foreign currency
    - F_market = current fair forward rate
    - F_contract = contracted forward rate

    Example
    -------
    >>> # Long EUR/USD forward: agree to buy EUR at 1.12 in 1 year
    >>> forward = FXForward(
    ...     spot=1.10,
    ...     forward_rate=1.12,
    ...     expiry=1.0,
    ...     domestic_rate=0.05,  # USD rate
    ...     foreign_rate=0.02,   # EUR rate
    ...     notional_foreign=1_000_000.0,
    ...     is_long=True
    ... )
    >>> mtm_value = forward.mark_to_market()
    """

    spot: float
    forward_rate: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    notional_foreign: float = 1.0
    is_long: bool = True

    def fair_forward_rate(self) -> float:
        """Calculate fair forward rate using interest rate parity.

        Returns
        -------
        float
            Fair forward rate
        """
        F = self.spot * jnp.exp((self.domestic_rate - self.foreign_rate) * self.expiry)
        return float(F)

    def mark_to_market(self) -> float:
        """Calculate mark-to-market value of the forward contract.

        Returns
        -------
        float
            Present value of the forward contract in domestic currency
        """
        fair_rate = self.fair_forward_rate()

        if self.is_long:
            # Long: benefit if fair rate > contract rate
            pnl = fair_rate - self.forward_rate
        else:
            # Short: benefit if fair rate < contract rate
            pnl = self.forward_rate - fair_rate

        # PV of payoff
        value = self.notional_foreign * pnl * jnp.exp(-self.domestic_rate * self.expiry)
        return float(value)

    def settlement_payoff(self, spot_at_expiry: float) -> float:
        """Calculate settlement payoff given spot at expiry.

        Parameters
        ----------
        spot_at_expiry : float
            Spot FX rate at expiry

        Returns
        -------
        float
            Settlement amount in domestic currency
        """
        if self.is_long:
            # Long: pay forward_rate, receive spot_at_expiry
            pnl = spot_at_expiry - self.forward_rate
        else:
            # Short: receive forward_rate, pay spot_at_expiry
            pnl = self.forward_rate - spot_at_expiry

        return float(self.notional_foreign * pnl)


# ============================================================================
# FX American Options
# ============================================================================


@dataclass
class FXAmericanOption(PathProduct):
    """American-style FX option with early exercise capability.

    Can be exercised at any time up to expiry. Requires path-dependent
    valuation methods like Longstaff-Schwartz Monte Carlo.

    Attributes
    ----------
    strike : float
        Strike FX rate
    T : float
        Time to maturity in years
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    is_call : bool
        True for call option, False for put option
    notional : float
        Notional amount in foreign currency

    Notes
    -----
    American options are worth at least as much as European options
    due to the early exercise feature. For FX options:

    - Call: right to buy foreign currency at strike
    - Put: right to sell foreign currency at strike

    Early exercise is optimal when the immediate exercise value exceeds
    the option's continuation value.

    Example
    -------
    >>> option = FXAmericanOption(
    ...     strike=1.12,
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     is_call=True,
    ...     notional=1_000_000.0
    ... )
    >>> # Price using LSM or finite differences
    """

    strike: float
    T: float
    domestic_rate: float
    foreign_rate: float
    is_call: bool = True
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute terminal immediate exercise value.

        For American options with LSM pricing, this provides the
        terminal exercise value used in backward induction.
        """
        path = jnp.asarray(path)
        terminal_spot = path[-1]
        return self.immediate_exercise(terminal_spot)

    def immediate_exercise(self, spot: jnp.ndarray) -> jnp.ndarray:
        """Compute immediate exercise value at any time.

        Parameters
        ----------
        spot : Array
            Spot FX rates

        Returns
        -------
        Array
            Immediate exercise values
        """
        spot = jnp.asarray(spot)
        if self.is_call:
            intrinsic = spot - self.strike
        else:
            intrinsic = self.strike - spot

        return jnp.maximum(intrinsic, 0.0) * self.notional


# ============================================================================
# FX Digital Options (Enhanced)
# ============================================================================


@dataclass(frozen=True)
class FXDigitalAssetOrNothing:
    """FX digital option that pays the asset if condition is met.

    Attributes
    ----------
    spot : float
        Current FX spot rate
    strike : float
        Strike price
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    volatility : float
        FX volatility
    is_call : bool
        True for call (pays if S_T > K), False for put (pays if S_T < K)
    notional : float
        Notional amount

    Notes
    -----
    - Call pays S_T if S_T > K, otherwise 0
    - Put pays S_T if S_T < K, otherwise 0

    This is different from cash-or-nothing which pays a fixed amount.

    Pricing formula:
    - Asset-or-nothing call: S * exp(-r_f * T) * N(d1)
    - Asset-or-nothing put: S * exp(-r_f * T) * N(-d1)

    Example
    -------
    >>> digital = FXDigitalAssetOrNothing(
    ...     spot=1.10,
    ...     strike=1.12,
    ...     expiry=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     volatility=0.10,
    ...     is_call=True
    ... )
    >>> price = digital.price()
    """

    spot: float
    strike: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    is_call: bool = True
    notional: float = 1.0

    def price(self) -> float:
        """Price the asset-or-nothing digital option.

        Returns
        -------
        float
            Option price in domestic currency
        """
        if self.expiry <= 0:
            if self.is_call:
                payoff = self.spot if self.spot > self.strike else 0.0
            else:
                payoff = self.spot if self.spot < self.strike else 0.0
            return float(payoff * self.notional)

        d1, _ = compute_d1_d2_fx(
            self.spot,
            self.strike,
            self.expiry,
            self.domestic_rate,
            self.foreign_rate,
            self.volatility
        )

        if self.is_call:
            price = self.spot * jnp.exp(-self.foreign_rate * self.expiry) * norm.cdf(d1)
        else:
            price = self.spot * jnp.exp(-self.foreign_rate * self.expiry) * norm.cdf(-d1)

        return float(price * self.notional)

    def delta(self) -> float:
        """Compute delta of the asset-or-nothing digital.

        Returns
        -------
        float
            Delta (∂V/∂S)
        """
        if self.expiry <= 0:
            return 0.0

        d1, _ = compute_d1_d2_fx(
            self.spot,
            self.strike,
            self.expiry,
            self.domestic_rate,
            self.foreign_rate,
            self.volatility
        )

        sqrt_T = jnp.sqrt(self.expiry)

        if self.is_call:
            delta = jnp.exp(-self.foreign_rate * self.expiry) * (
                norm.cdf(d1) + self.spot / (self.volatility * sqrt_T) * norm.pdf(d1)
            )
        else:
            delta = jnp.exp(-self.foreign_rate * self.expiry) * (
                norm.cdf(-d1) - self.spot / (self.volatility * sqrt_T) * norm.pdf(d1)
            )

        return float(delta * self.notional)


# ============================================================================
# FX Asian Options
# ============================================================================


@dataclass
class FXAsianArithmetic(PathProduct):
    """Arithmetic-average FX Asian option (fixed strike).

    Parameters
    ----------
    strike : float
        Strike FX rate
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    is_call : bool
        True for call, False for put
    notional : float
        Notional amount in foreign currency

    Notes
    -----
    Payoff based on arithmetic average of FX rates:
    - Call: max(Avg(S) - K, 0) * N
    - Put: max(K - Avg(S), 0) * N

    Asian options are cheaper than vanilla options due to the
    averaging feature which reduces volatility.

    Example
    -------
    >>> asian = FXAsianArithmetic(
    ...     strike=1.12,
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     is_call=True,
    ...     notional=1_000_000.0
    ... )
    """

    strike: float
    T: float
    domestic_rate: float
    foreign_rate: float
    is_call: bool = True
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        avg_rate = path.mean()

        if self.is_call:
            intrinsic = avg_rate - self.strike
        else:
            intrinsic = self.strike - avg_rate

        return jnp.maximum(intrinsic, 0.0) * self.notional


@dataclass
class FXAsianGeometric(PathProduct):
    """Geometric-average FX Asian option (fixed strike).

    Parameters
    ----------
    strike : float
        Strike FX rate
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    is_call : bool
        True for call, False for put
    notional : float
        Notional amount

    Notes
    -----
    Payoff based on geometric average:
    - Geometric mean = exp(mean(log(S_i)))

    Geometric Asian options have closed-form pricing formulas
    and are typically cheaper than arithmetic Asian options.

    Example
    -------
    >>> asian = FXAsianGeometric(
    ...     strike=1.12,
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     is_call=True
    ... )
    """

    strike: float
    T: float
    domestic_rate: float
    foreign_rate: float
    is_call: bool = True
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        log_rates = jnp.log(jnp.maximum(path, 1e-10))
        geometric_avg = jnp.exp(log_rates.mean())

        if self.is_call:
            intrinsic = geometric_avg - self.strike
        else:
            intrinsic = self.strike - geometric_avg

        return jnp.maximum(intrinsic, 0.0) * self.notional


@dataclass
class FXAsianArithmeticFloatingStrike(PathProduct):
    """Arithmetic-average FX Asian option with floating strike.

    Parameters
    ----------
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    is_call : bool
        True for call, False for put
    notional : float
        Notional amount

    Notes
    -----
    Strike is the average FX rate, payoff based on terminal rate:
    - Call: max(S_T - Avg(S), 0) * N
    - Put: max(Avg(S) - S_T, 0) * N

    Useful for hedging average exchange rate exposure.

    Example
    -------
    >>> asian = FXAsianArithmeticFloatingStrike(
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     is_call=True
    ... )
    """

    T: float
    domestic_rate: float
    foreign_rate: float
    is_call: bool = True
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        avg_rate = path.mean()
        terminal_rate = path[-1]

        if self.is_call:
            intrinsic = terminal_rate - avg_rate
        else:
            intrinsic = avg_rate - terminal_rate

        return jnp.maximum(intrinsic, 0.0) * self.notional


@dataclass
class FXAsianGeometricFloatingStrike(PathProduct):
    """Geometric-average FX Asian option with floating strike.

    Parameters
    ----------
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign rate
    is_call : bool
        True for call, False for put
    notional : float
        Notional amount

    Notes
    -----
    Strike is the geometric average, payoff based on terminal rate:
    - Call: max(S_T - GeoAvg(S), 0) * N
    - Put: max(GeoAvg(S) - S_T, 0) * N
    """

    T: float
    domestic_rate: float
    foreign_rate: float
    is_call: bool = True
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        log_rates = jnp.log(jnp.maximum(path, 1e-10))
        geometric_avg = jnp.exp(log_rates.mean())
        terminal_rate = path[-1]

        if self.is_call:
            intrinsic = terminal_rate - geometric_avg
        else:
            intrinsic = geometric_avg - terminal_rate

        return jnp.maximum(intrinsic, 0.0) * self.notional


# ============================================================================
# FX Lookback Options
# ============================================================================


@dataclass
class FXLookbackFloatingStrikeCall(PathProduct):
    """Floating-strike FX lookback call option.

    Parameters
    ----------
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    notional : float
        Notional amount

    Notes
    -----
    Pays the difference between terminal rate and minimum rate:
    Payoff = (S_T - min(S_t)) * N

    Allows holder to "buy" at the lowest rate observed.

    Example
    -------
    >>> lookback = FXLookbackFloatingStrikeCall(
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     notional=1_000_000.0
    ... )
    """

    T: float
    domestic_rate: float
    foreign_rate: float
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        return (path[-1] - path.min()) * self.notional


@dataclass
class FXLookbackFloatingStrikePut(PathProduct):
    """Floating-strike FX lookback put option.

    Parameters
    ----------
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    notional : float
        Notional amount

    Notes
    -----
    Pays the difference between maximum rate and terminal rate:
    Payoff = (max(S_t) - S_T) * N

    Allows holder to "sell" at the highest rate observed.

    Example
    -------
    >>> lookback = FXLookbackFloatingStrikePut(
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02
    ... )
    """

    T: float
    domestic_rate: float
    foreign_rate: float
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        return (path.max() - path[-1]) * self.notional


@dataclass
class FXLookbackFixedStrikeCall(PathProduct):
    """Fixed-strike FX lookback call option.

    Parameters
    ----------
    strike : float
        Strike FX rate
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    notional : float
        Notional amount

    Notes
    -----
    Pays based on maximum rate observed:
    Payoff = max(max(S_t) - K, 0) * N

    Holder benefits from the best rate during the option's life.

    Example
    -------
    >>> lookback = FXLookbackFixedStrikeCall(
    ...     strike=1.12,
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02
    ... )
    """

    strike: float
    T: float
    domestic_rate: float
    foreign_rate: float
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        max_rate = path.max()
        return jnp.maximum(max_rate - self.strike, 0.0) * self.notional


@dataclass
class FXLookbackFixedStrikePut(PathProduct):
    """Fixed-strike FX lookback put option.

    Parameters
    ----------
    strike : float
        Strike FX rate
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    notional : float
        Notional amount

    Notes
    -----
    Pays based on minimum rate observed:
    Payoff = max(K - min(S_t), 0) * N

    Holder benefits from the lowest rate during the option's life.

    Example
    -------
    >>> lookback = FXLookbackFixedStrikePut(
    ...     strike=1.12,
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02
    ... )
    """

    strike: float
    T: float
    domestic_rate: float
    foreign_rate: float
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        min_rate = path.min()
        return jnp.maximum(self.strike - min_rate, 0.0) * self.notional


@dataclass
class FXLookbackPartialFixedStrikeCall(PathProduct):
    """Partial lookback call with observation window.

    Parameters
    ----------
    strike : float
        Strike FX rate
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    observation_start : float
        Start of observation window (as fraction of total time)
    notional : float
        Notional amount

    Notes
    -----
    Only observes rates during a portion of the option's life,
    reducing premium while maintaining lookback features.

    Example
    -------
    >>> # Lookback starting at 50% of option life
    >>> lookback = FXLookbackPartialFixedStrikeCall(
    ...     strike=1.12,
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     observation_start=0.5
    ... )
    """

    strike: float
    T: float
    domestic_rate: float
    foreign_rate: float
    observation_start: float = 0.0
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        n = len(path)
        start_idx = int(self.observation_start * (n - 1))
        observed_path = path[start_idx:]
        max_rate = observed_path.max()
        return jnp.maximum(max_rate - self.strike, 0.0) * self.notional


@dataclass
class FXLookbackPartialFixedStrikePut(PathProduct):
    """Partial lookback put with observation window.

    Parameters
    ----------
    strike : float
        Strike FX rate
    T : float
        Time to maturity
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    observation_start : float
        Start of observation window (as fraction of total time)
    notional : float
        Notional amount

    Notes
    -----
    Similar to fixed-strike lookback put, but only observes rates
    during a portion of the option's life.

    Example
    -------
    >>> lookback = FXLookbackPartialFixedStrikePut(
    ...     strike=1.12,
    ...     T=1.0,
    ...     domestic_rate=0.05,
    ...     foreign_rate=0.02,
    ...     observation_start=0.5
    ... )
    """

    strike: float
    T: float
    domestic_rate: float
    foreign_rate: float
    observation_start: float = 0.0
    notional: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff from path."""
        path = jnp.asarray(path)
        n = len(path)
        start_idx = int(self.observation_start * (n - 1))
        observed_path = path[start_idx:]
        min_rate = observed_path.min()
        return jnp.maximum(self.strike - min_rate, 0.0) * self.notional


__all__ = [
    # Forwards
    "FXForward",
    # American Options
    "FXAmericanOption",
    # Digital Options
    "FXDigitalAssetOrNothing",
    # Asian Options
    "FXAsianArithmetic",
    "FXAsianGeometric",
    "FXAsianArithmeticFloatingStrike",
    "FXAsianGeometricFloatingStrike",
    # Lookback Options
    "FXLookbackFloatingStrikeCall",
    "FXLookbackFloatingStrikePut",
    "FXLookbackFixedStrikeCall",
    "FXLookbackFixedStrikePut",
    "FXLookbackPartialFixedStrikeCall",
    "FXLookbackPartialFixedStrikePut",
]
