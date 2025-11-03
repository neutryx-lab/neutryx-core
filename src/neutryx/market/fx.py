"""
FX (foreign exchange) market data structures and utilities.

This module provides infrastructure for multi-currency pricing:
- FX spot rate management with triangulation
- FX forward curves
- FX volatility surfaces
- Quanto adjustments for cross-currency derivatives
- Cross-currency basis spreads
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from .base import Curve, VolatilitySurface


@dataclass
class FXSpot:
    """
    FX spot rate manager with triangulation support.

    Manages FX spot rates between currency pairs, automatically handling
    inverse rates and triangulation through a base currency.

    Attributes:
        base_currency: Base currency for triangulation (typically "USD")
        rates: Dict mapping (from_ccy, to_ccy) to spot rates

    Example:
        >>> fx = FXSpot(base_currency="USD")
        >>> fx.add_rate("EUR", "USD", 1.10)  # 1 EUR = 1.10 USD
        >>> fx.add_rate("GBP", "USD", 1.30)  # 1 GBP = 1.30 USD
        >>> rate = fx.get_rate("EUR", "GBP")  # Auto-triangulate
        >>> abs(rate - 1.10/1.30) < 1e-6  # EUR/GBP via USD
        True
    """

    base_currency: str = "USD"
    rates: Dict[Tuple[str, str], float] = None

    def __post_init__(self):
        """Initialize rates dict if not provided."""
        if self.rates is None:
            object.__setattr__(self, "rates", {})

    def add_rate(self, from_ccy: str, to_ccy: str, rate: float) -> None:
        """
        Add FX spot rate.

        Args:
            from_ccy: Source currency (e.g., "EUR")
            to_ccy: Target currency (e.g., "USD")
            rate: Spot rate (units of to_ccy per unit of from_ccy)
        """
        if rate <= 0:
            raise ValueError(f"FX rate must be positive, got {rate}")

        self.rates[(from_ccy, to_ccy)] = rate

    def get_rate(self, from_ccy: str, to_ccy: str) -> float:
        """
        Get FX spot rate, with automatic triangulation.

        Args:
            from_ccy: Source currency
            to_ccy: Target currency

        Returns:
            Spot rate (units of to_ccy per unit of from_ccy)

        Raises:
            KeyError: If rate cannot be determined (even via triangulation)

        Note:
            Lookup order:
            1. Direct rate (from_ccy, to_ccy)
            2. Inverse rate 1/(to_ccy, from_ccy)
            3. Triangulation via base currency
        """
        if from_ccy == to_ccy:
            return 1.0

        # Try direct lookup
        pair = (from_ccy, to_ccy)
        if pair in self.rates:
            return self.rates[pair]

        # Try inverse
        inverse_pair = (to_ccy, from_ccy)
        if inverse_pair in self.rates:
            return 1.0 / self.rates[inverse_pair]

        # Try triangulation via base currency
        if from_ccy != self.base_currency and to_ccy != self.base_currency:
            try:
                # from_ccy -> base -> to_ccy
                from_to_base = self.get_rate(from_ccy, self.base_currency)
                base_to_to = self.get_rate(self.base_currency, to_ccy)
                return from_to_base * base_to_to
            except KeyError:
                pass

        raise KeyError(
            f"Cannot determine FX rate for {from_ccy}/{to_ccy} "
            f"(base currency: {self.base_currency})"
        )

    def list_pairs(self) -> list[Tuple[str, str]]:
        """Return list of all currency pairs with direct rates."""
        return list(self.rates.keys())

    def __repr__(self) -> str:
        return f"FXSpot(base={self.base_currency}, pairs={len(self.rates)})"


@dataclass
class FXForwardCurve:
    """
    FX forward rate curve.

    Represents term structure of FX forward rates for a currency pair.
    Can be constructed explicitly or implied from discount curves via
    covered interest parity.

    Attributes:
        from_ccy: Source currency
        to_ccy: Target currency
        spot: Spot FX rate
        times: Pillar times (in years)
        forwards: Forward rates at each pillar

    Example:
        >>> import jax.numpy as jnp
        >>> curve = FXForwardCurve(
        ...     from_ccy="EUR",
        ...     to_ccy="USD",
        ...     spot=1.10,
        ...     times=jnp.array([0.25, 0.5, 1.0, 2.0]),
        ...     forwards=jnp.array([1.105, 1.11, 1.12, 1.14])
        ... )
        >>> f = curve.forward(1.0)  # 1-year forward rate
    """

    from_ccy: str
    to_ccy: str
    spot: float
    times: Array
    forwards: Array

    def __post_init__(self):
        """Validate inputs."""
        if len(self.times) != len(self.forwards):
            raise ValueError("times and forwards must have same length")
        if self.spot <= 0:
            raise ValueError("spot must be positive")

        # Convert to JAX arrays
        object.__setattr__(self, "times", jnp.asarray(self.times))
        object.__setattr__(self, "forwards", jnp.asarray(self.forwards))

    def forward(self, t: float | Array) -> float | Array:
        """
        Get FX forward rate at time t.

        Args:
            t: Time in years

        Returns:
            Forward FX rate F(0, t)
        """
        t_arr = jnp.asarray(t)
        return jnp.interp(t_arr, self.times, self.forwards)

    def value(self, t: float | Array) -> float | Array:
        """Alias for forward(t) to implement Curve protocol."""
        return self.forward(t)

    def __call__(self, t: float | Array) -> float | Array:
        """Alias for forward(t)."""
        return self.forward(t)

    def forward_points(self, t: float | Array) -> float | Array:
        """
        Compute forward points: F(t) - S(0).

        Args:
            t: Time in years

        Returns:
            Forward points
        """
        return self.forward(t) - self.spot

    def implied_yield_diff(self, t: float) -> float:
        """
        Compute implied interest rate differential.

        From covered interest parity: F/S = exp((r_to - r_from)*t)
        => r_to - r_from = ln(F/S) / t

        Args:
            t: Time in years

        Returns:
            Implied rate differential (continuously compounded)
        """
        fwd = float(self.forward(t))
        return jnp.log(fwd / self.spot) / t

    @staticmethod
    def from_interest_parity(
        from_ccy: str,
        to_ccy: str,
        spot: float,
        discount_curve_from: Curve,
        discount_curve_to: Curve,
        times: Array
    ) -> FXForwardCurve:
        """
        Construct FX forward curve from covered interest parity.

        F(t) = S * DF_to(t) / DF_from(t)

        Args:
            from_ccy: Source currency
            to_ccy: Target currency
            spot: Spot FX rate
            discount_curve_from: Discount curve for from_ccy
            discount_curve_to: Discount curve for to_ccy
            times: Pillar times for forward curve

        Returns:
            FXForwardCurve instance
        """
        times_arr = jnp.asarray(times)
        df_from = discount_curve_from.value(times_arr)
        df_to = discount_curve_to.value(times_arr)

        forwards = spot * df_to / df_from

        return FXForwardCurve(
            from_ccy=from_ccy,
            to_ccy=to_ccy,
            spot=spot,
            times=times_arr,
            forwards=forwards
        )


@dataclass
class FXVolatilitySurface:
    """
    FX implied volatility surface.

    Represents volatility surface for FX options on a currency pair.
    Typically quotes are in delta space for FX (25-delta RR, 25-delta BF, ATM).

    Attributes:
        from_ccy: Source currency
        to_ccy: Target currency
        expiries: Expiry times (in years)
        strikes: Strike levels (or delta values if using delta space)
        vols: Implied volatilities (grid: expiries x strikes)

    Example:
        >>> import jax.numpy as jnp
        >>> surf = FXVolatilitySurface(
        ...     from_ccy="EUR",
        ...     to_ccy="USD",
        ...     expiries=jnp.array([0.25, 0.5, 1.0]),
        ...     strikes=jnp.array([1.05, 1.10, 1.15]),
        ...     vols=jnp.array([
        ...         [0.10, 0.09, 0.10],  # 3M tenor
        ...         [0.11, 0.10, 0.11],  # 6M tenor
        ...         [0.12, 0.11, 0.12],  # 1Y tenor
        ...     ])
        ... )
        >>> vol = surf.implied_vol(0.5, 1.10)  # 6M, ATM
    """

    from_ccy: str
    to_ccy: str
    expiries: Array
    strikes: Array
    vols: Array  # Shape: (n_expiries, n_strikes)

    def __post_init__(self):
        """Validate inputs."""
        self.expiries = jnp.asarray(self.expiries)
        self.strikes = jnp.asarray(self.strikes)
        self.vols = jnp.asarray(self.vols)

        expected_shape = (len(self.expiries), len(self.strikes))
        if self.vols.shape != expected_shape:
            raise ValueError(
                f"vols shape {self.vols.shape} doesn't match "
                f"expected {expected_shape} from expiries x strikes"
            )

    def implied_vol(self, expiry: float | Array, strike: float | Array) -> float | Array:
        """
        Get implied volatility via bilinear interpolation.

        Args:
            expiry: Time to expiry in years
            strike: Strike price (or delta if using delta space)

        Returns:
            Implied volatility (annualized)
        """
        expiry_arr = jnp.asarray(expiry)
        strike_arr = jnp.asarray(strike)

        # Find indices for bilinear interpolation
        # For simplicity, using 1D interpolation on flattened coordinates
        # Production code would use proper 2D interpolation

        # Interpolate on expiry dimension first
        vol_at_expiry = jnp.array([
            jnp.interp(strike_arr, self.strikes, self.vols[i, :])
            for i in range(len(self.expiries))
        ])

        # Then interpolate on strike dimension
        result = jnp.interp(expiry_arr, self.expiries, vol_at_expiry)

        return result

    def value(self, expiry: float | Array, strike: float | Array) -> float | Array:
        """Alias for implied_vol to implement Surface protocol."""
        return self.implied_vol(expiry, strike)

    def __call__(self, expiry: float | Array, strike: float | Array) -> float | Array:
        """Alias for implied_vol."""
        return self.implied_vol(expiry, strike)

    def atm_vol(self, expiry: float) -> float:
        """
        Get ATM (at-the-money) volatility for given expiry.

        Assumes middle strike is ATM.

        Args:
            expiry: Time to expiry

        Returns:
            ATM implied volatility
        """
        mid_strike = self.strikes[len(self.strikes) // 2]
        return float(self.implied_vol(expiry, mid_strike))


def quanto_drift_adjustment(
    vol_asset: float,
    vol_fx: float,
    corr_asset_fx: float,
    domestic_rate: float,
    foreign_rate: float
) -> float:
    """
    Compute quanto drift adjustment for cross-currency derivatives.

    For an asset in foreign currency, priced in domestic currency without
    FX conversion (quanto payoff), the drift adjustment is:

        adjustment = -ρ_SF * σ_S * σ_F

    where:
    - ρ_SF: correlation between asset and FX rate
    - σ_S: asset volatility
    - σ_F: FX volatility

    Args:
        vol_asset: Asset volatility (annualized)
        vol_fx: FX volatility (annualized)
        corr_asset_fx: Correlation between asset and FX rate
        domestic_rate: Domestic risk-free rate (unused in basic adjustment)
        foreign_rate: Foreign risk-free rate (unused in basic adjustment)

    Returns:
        Drift adjustment to add to foreign rate

    Example:
        >>> # SPX quanto option in EUR (no FX conversion)
        >>> # SPX vol = 20%, USD/EUR vol = 10%, correlation = 0.3
        >>> adj = quanto_drift_adjustment(0.20, 0.10, 0.3, 0.02, 0.05)
        >>> # Use adjusted drift: mu_quanto = r_foreign + adjustment
    """
    return -corr_asset_fx * vol_asset * vol_fx


def quanto_adjusted_forward(
    spot: float,
    time: float,
    domestic_rate: float,
    foreign_rate: float,
    dividend_yield: float,
    vol_asset: float,
    vol_fx: float,
    corr_asset_fx: float
) -> float:
    """
    Compute quanto-adjusted forward price for an asset.

    For a foreign asset with quanto payoff (no FX conversion):

        F_quanto = S * exp((r_f - q + quanto_adjustment) * T)

    where quanto_adjustment = -ρ * σ_S * σ_FX

    Args:
        spot: Current asset price in foreign currency
        time: Time to maturity in years
        domestic_rate: Domestic risk-free rate
        foreign_rate: Foreign risk-free rate
        dividend_yield: Asset dividend yield
        vol_asset: Asset volatility
        vol_fx: FX volatility
        corr_asset_fx: Correlation between asset and FX

    Returns:
        Quanto-adjusted forward price

    Example:
        >>> # Nikkei quanto forward in USD
        >>> F = quanto_adjusted_forward(
        ...     spot=30000.0,
        ...     time=1.0,
        ...     domestic_rate=0.02,   # USD rate
        ...     foreign_rate=0.005,   # JPY rate
        ...     dividend_yield=0.02,
        ...     vol_asset=0.20,       # Nikkei vol
        ...     vol_fx=0.12,          # USD/JPY vol
        ...     corr_asset_fx=-0.3    # Typical negative correlation
        ... )
    """
    adjustment = quanto_drift_adjustment(
        vol_asset, vol_fx, corr_asset_fx,
        domestic_rate, foreign_rate
    )

    # Forward with quanto adjustment
    # F = S * exp((r_f - q + adj) * T)
    drift = foreign_rate - dividend_yield + adjustment

    return spot * jnp.exp(drift * time)


@dataclass
class CrossCurrencyBasisSpread:
    """
    Cross-currency basis spread term structure.

    Represents the basis spread adjustment to covered interest parity,
    observed in FX swap markets (especially post-2008).

    The actual FX forward includes basis:
        F(t) = S * (DF_to(t) / DF_from(t)) * exp(basis(t) * t)

    Attributes:
        from_ccy: Source currency
        to_ccy: Target currency
        times: Pillar times
        spreads: Basis spreads at each pillar (in bps, e.g., -15 for -15bps)

    Example:
        >>> basis = CrossCurrencyBasisSpread(
        ...     from_ccy="EUR",
        ...     to_ccy="USD",
        ...     times=jnp.array([0.25, 0.5, 1.0, 5.0]),
        ...     spreads=jnp.array([-10, -12, -15, -20])  # bps
        ... )
        >>> spread_1y = basis.spread(1.0)  # Get 1Y basis
    """

    from_ccy: str
    to_ccy: str
    times: Array
    spreads: Array  # In basis points

    def __post_init__(self):
        """Validate inputs."""
        if len(self.times) != len(self.spreads):
            raise ValueError("times and spreads must have same length")

        object.__setattr__(self, "times", jnp.asarray(self.times))
        object.__setattr__(self, "spreads", jnp.asarray(self.spreads))

    def spread(self, t: float | Array) -> float | Array:
        """
        Get basis spread at time t (in basis points).

        Args:
            t: Time in years

        Returns:
            Basis spread in bps
        """
        t_arr = jnp.asarray(t)
        return jnp.interp(t_arr, self.times, self.spreads)

    def spread_decimal(self, t: float | Array) -> float | Array:
        """
        Get basis spread at time t as decimal (divide by 10000).

        Args:
            t: Time in years

        Returns:
            Basis spread as decimal (e.g., -0.0015 for -15bps)
        """
        return self.spread(t) / 10000.0

    def adjust_forward(
        self,
        forward_no_basis: float | Array,
        t: float | Array
    ) -> float | Array:
        """
        Apply basis adjustment to FX forward.

        F_adjusted = F_no_basis * exp(basis * t)

        Args:
            forward_no_basis: Forward rate without basis
            t: Time in years

        Returns:
            Basis-adjusted forward rate
        """
        basis_dec = self.spread_decimal(t)
        t_arr = jnp.asarray(t)
        return forward_no_basis * jnp.exp(basis_dec * t_arr)
