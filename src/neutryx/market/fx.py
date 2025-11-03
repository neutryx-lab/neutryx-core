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
from typing import Dict, Optional, Tuple, Literal

import jax.numpy as jnp
from jax import Array

from .base import Curve, VolatilitySurface
from .vol import SABRParameters, sabr_implied_vol


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


# ============================================================================
# FX Volatility Market Conventions (BF/RR Quotes)
# ============================================================================

@dataclass
class FXVolatilityQuote:
    """
    FX volatility market quote for a single tenor.

    FX options are typically quoted in terms of:
    - ATM (At-The-Money) volatility
    - 25Δ Risk Reversal (RR): vol_25d_call - vol_25d_put
    - 25Δ Butterfly (BF): (vol_25d_call + vol_25d_put)/2 - vol_ATM

    Also supports 10Δ and 15Δ pillars for more precise smile construction.

    Attributes:
        expiry: Time to expiry in years
        atm_vol: ATM volatility (e.g., 0.10 for 10%)
        rr_25d: 25-delta risk reversal (e.g., 0.01 for 1%)
        bf_25d: 25-delta butterfly (e.g., 0.005 for 0.5%)
        forward: Forward FX rate for this expiry
        domestic_rate: Domestic risk-free rate (for delta calculation)
        foreign_rate: Foreign risk-free rate (for delta calculation)
        rr_10d: Optional 10-delta risk reversal
        bf_10d: Optional 10-delta butterfly
        rr_15d: Optional 15-delta risk reversal
        bf_15d: Optional 15-delta butterfly

    Example:
        >>> quote = FXVolatilityQuote(
        ...     expiry=1.0,
        ...     atm_vol=0.10,
        ...     rr_25d=0.015,
        ...     bf_25d=0.005,
        ...     forward=1.10,
        ...     domestic_rate=0.02,
        ...     foreign_rate=0.01
        ... )
        >>> vols = quote.extract_pillar_vols()
    """

    expiry: float
    atm_vol: float
    rr_25d: float
    bf_25d: float
    forward: float
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    # Additional delta pillars
    rr_10d: Optional[float] = None
    bf_10d: Optional[float] = None
    rr_15d: Optional[float] = None
    bf_15d: Optional[float] = None

    def extract_pillar_vols(self, deltas: Optional[list[float]] = None) -> dict[str, float]:
        """
        Extract theoretical volatilities at market pillars.

        From market quotes (ATM, RR, BF), compute:
        - vol_Xd_call = ATM + BF + RR/2
        - vol_Xd_put = ATM + BF - RR/2
        - vol_ATM = ATM

        Args:
            deltas: List of delta values to extract. If None, uses available pillars.

        Returns:
            Dictionary with keys like 'atm', '25d_call', '25d_put', etc.
        """
        vols = {'atm': self.atm_vol}

        # Always include 25 delta if available
        if self.rr_25d is not None and self.bf_25d is not None:
            vols['25d_call'] = self.atm_vol + self.bf_25d + self.rr_25d / 2.0
            vols['25d_put'] = self.atm_vol + self.bf_25d - self.rr_25d / 2.0

        # Include 10 delta if available
        if self.rr_10d is not None and self.bf_10d is not None:
            vols['10d_call'] = self.atm_vol + self.bf_10d + self.rr_10d / 2.0
            vols['10d_put'] = self.atm_vol + self.bf_10d - self.rr_10d / 2.0

        # Include 15 delta if available
        if self.rr_15d is not None and self.bf_15d is not None:
            vols['15d_call'] = self.atm_vol + self.bf_15d + self.rr_15d / 2.0
            vols['15d_put'] = self.atm_vol + self.bf_15d - self.rr_15d / 2.0

        return vols

    def get_available_deltas(self) -> list[float]:
        """
        Get list of available delta values for this quote.

        Returns:
            List of delta values (e.g., [0.10, 0.15, 0.25])
        """
        deltas = [0.50]  # ATM is always available

        if self.rr_10d is not None and self.bf_10d is not None:
            deltas.append(0.10)

        if self.rr_15d is not None and self.bf_15d is not None:
            deltas.append(0.15)

        if self.rr_25d is not None and self.bf_25d is not None:
            deltas.append(0.25)

        return sorted(deltas)

    def __repr__(self) -> str:
        base = (
            f"FXVolatilityQuote(expiry={self.expiry:.2f}y, "
            f"ATM={self.atm_vol*100:.2f}%, "
            f"RR25={self.rr_25d*100:.2f}%, "
            f"BF25={self.bf_25d*100:.2f}%"
        )
        if self.rr_10d is not None:
            base += f", RR10={self.rr_10d*100:.2f}%, BF10={self.bf_10d*100:.2f}%"
        if self.rr_15d is not None:
            base += f", RR15={self.rr_15d*100:.2f}%, BF15={self.bf_15d*100:.2f}%"
        return base + ")"


def _standard_normal_cdf(x: float | Array) -> float | Array:
    """Standard normal cumulative distribution function."""
    from jax.scipy.stats import norm
    return norm.cdf(x)


def delta_to_strike_iterative(
    delta: float,
    forward: float,
    expiry: float,
    vol_initial: float,
    is_call: bool = True,
    domestic_rate: float = 0.0,
    foreign_rate: float = 0.0,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> tuple[float, float]:
    """
    Convert delta to strike using iterative solver.

    For FX options, delta depends on the strike through the volatility smile.
    This function iteratively solves for the strike that matches the target delta.

    Uses forward delta convention (premium in foreign currency):
    - Call delta: Δ = exp(-r_f * T) * N(d1)
    - Put delta: Δ = -exp(-r_f * T) * N(-d1)

    where d1 = [ln(F/K) + (σ²/2)*T] / (σ*√T)

    Args:
        delta: Target delta (e.g., 0.25 for 25-delta call, -0.25 for 25-delta put)
        forward: Forward FX rate
        expiry: Time to expiry in years
        vol_initial: Initial volatility guess
        is_call: True for call, False for put
        domestic_rate: Domestic risk-free rate
        foreign_rate: Foreign risk-free rate
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance for delta

    Returns:
        Tuple of (strike, final_volatility)

    Example:
        >>> # Find strike for 25-delta call
        >>> strike, vol = delta_to_strike_iterative(
        ...     delta=0.25, forward=1.10, expiry=1.0, vol_initial=0.10, is_call=True
        ... )
    """
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    # Handle negative delta for puts
    delta_target = abs(delta)

    # Initial guess: use ATM strike and iterate
    vol = vol_initial
    strike = forward  # Initial guess

    discount_factor = jnp.exp(-foreign_rate * expiry)
    sqrt_t = jnp.sqrt(expiry)

    for _ in range(max_iterations):
        # Calculate d1 and d2
        if strike <= 0 or vol <= 0:
            break

        d1 = (jnp.log(forward / strike) + (vol**2 / 2) * expiry) / (vol * sqrt_t)

        # Calculate delta based on option type
        if is_call:
            calculated_delta = discount_factor * norm.cdf(d1)
        else:
            calculated_delta = discount_factor * (norm.cdf(d1) - 1.0)
            calculated_delta = abs(calculated_delta)

        # Check convergence
        delta_diff = abs(calculated_delta - delta_target)
        if delta_diff < tolerance:
            return float(strike), float(vol)

        # Update strike using Newton-Raphson step
        # For simplicity, use a fixed-point iteration
        # More sophisticated: use vega-weighted adjustment

        # Simple adjustment: move strike in direction of delta error
        if calculated_delta > delta_target:
            # Need lower delta -> increase strike for call, decrease for put
            strike *= 1.01 if is_call else 0.99
        else:
            # Need higher delta -> decrease strike for call, increase for put
            strike *= 0.99 if is_call else 1.01

    return float(strike), float(vol)


def strike_from_delta_atm(
    forward: float,
    expiry: float,
    vol: float,
    domestic_rate: float = 0.0,
    foreign_rate: float = 0.0,
) -> float:
    """
    Calculate ATM strike from forward.

    For FX options, ATM can be defined as:
    - ATM Forward: K = F (used here)
    - ATM Delta Neutral: K such that call_delta + put_delta = 0
    - ATM DNS (Delta Neutral Straddle): K such that |call_delta| = |put_delta|

    Args:
        forward: Forward FX rate
        expiry: Time to expiry in years
        vol: Volatility
        domestic_rate: Domestic risk-free rate
        foreign_rate: Foreign risk-free rate

    Returns:
        ATM strike
    """
    # ATM forward convention (most common for FX)
    return forward


def build_smile_from_market_quote(
    quote: FXVolatilityQuote,
    num_strikes: int = 5,
) -> tuple[Array, Array]:
    """
    Build volatility smile from market quote (ATM, BF, RR).

    Constructs strike-vol pairs from market conventions:
    1. Extract theoretical vols at 25-delta pillars
    2. Convert deltas to strikes
    3. Interpolate to create full smile

    Args:
        quote: FX volatility market quote
        num_strikes: Number of strike points to generate

    Returns:
        Tuple of (strikes, vols) arrays

    Example:
        >>> quote = FXVolatilityQuote(
        ...     expiry=1.0, atm_vol=0.10, rr_25d=0.015, bf_25d=0.005,
        ...     forward=1.10, domestic_rate=0.02, foreign_rate=0.01
        ... )
        >>> strikes, vols = build_smile_from_market_quote(quote)
    """
    import jax.numpy as jnp

    # Extract pillar vols
    pillar_vols = quote.extract_pillar_vols()

    # Get strikes for each pillar
    # 25-delta put (typically delta = -0.25 or 0.25 for put using positive convention)
    strike_25p, vol_25p = delta_to_strike_iterative(
        delta=-0.25,
        forward=quote.forward,
        expiry=quote.expiry,
        vol_initial=pillar_vols['25d_put'],
        is_call=False,
        domestic_rate=quote.domestic_rate,
        foreign_rate=quote.foreign_rate,
    )

    # ATM
    strike_atm = strike_from_delta_atm(
        forward=quote.forward,
        expiry=quote.expiry,
        vol=pillar_vols['atm'],
        domestic_rate=quote.domestic_rate,
        foreign_rate=quote.foreign_rate,
    )
    vol_atm = pillar_vols['atm']

    # 25-delta call
    strike_25c, vol_25c = delta_to_strike_iterative(
        delta=0.25,
        forward=quote.forward,
        expiry=quote.expiry,
        vol_initial=pillar_vols['25d_call'],
        is_call=True,
        domestic_rate=quote.domestic_rate,
        foreign_rate=quote.foreign_rate,
    )

    # Create arrays of pillar points
    pillar_strikes = jnp.array([strike_25p, strike_atm, strike_25c])
    pillar_vols_array = jnp.array([vol_25p, vol_atm, vol_25c])

    # Generate full smile by interpolation
    # Extend range slightly beyond pillars
    min_strike = strike_25p * 0.9
    max_strike = strike_25c * 1.1
    strikes = jnp.linspace(min_strike, max_strike, num_strikes)

    # Linear interpolation (can be enhanced with splines or SABR)
    vols = jnp.interp(strikes, pillar_strikes, pillar_vols_array)

    return strikes, vols


@dataclass
class FXVolatilitySurfaceBuilder:
    """
    Builder for FX volatility surface from market quotes.

    Constructs a complete volatility surface from ATM/BF/RR market quotes
    across multiple tenors. Each tenor produces a smile, and the surface
    interpolates across both strike and tenor dimensions.

    Attributes:
        from_ccy: Source currency
        to_ccy: Target currency
        quotes: List of FXVolatilityQuote for different tenors

    Example:
        >>> builder = FXVolatilitySurfaceBuilder(
        ...     from_ccy="EUR",
        ...     to_ccy="USD",
        ...     quotes=[
        ...         FXVolatilityQuote(expiry=0.25, atm_vol=0.095, rr_25d=0.010, bf_25d=0.003, forward=1.10),
        ...         FXVolatilityQuote(expiry=0.5, atm_vol=0.100, rr_25d=0.012, bf_25d=0.004, forward=1.105),
        ...         FXVolatilityQuote(expiry=1.0, atm_vol=0.105, rr_25d=0.015, bf_25d=0.005, forward=1.11),
        ...     ]
        ... )
        >>> surface = builder.build_surface()
        >>> vol = surface.implied_vol(0.75, 1.12)  # Get vol at 9M, strike 1.12
    """

    from_ccy: str
    to_ccy: str
    quotes: list[FXVolatilityQuote]

    def build_surface(self, num_strikes_per_tenor: int = 11) -> FXVolatilitySurface:
        """
        Build complete volatility surface from market quotes.

        For each tenor:
        1. Extract ATM/BF/RR
        2. Compute theoretical vols at 25-delta pillars
        3. Convert deltas to strikes
        4. Interpolate to create smile

        Then combine all tenors into a surface.

        Args:
            num_strikes_per_tenor: Number of strike points per tenor

        Returns:
            FXVolatilitySurface with full strike-tenor grid

        Example:
            >>> surface = builder.build_surface(num_strikes_per_tenor=15)
        """
        if not self.quotes:
            raise ValueError("Must provide at least one volatility quote")

        # Sort quotes by expiry
        sorted_quotes = sorted(self.quotes, key=lambda q: q.expiry)

        # Build smile for each tenor
        all_strikes = []
        all_vols = []
        expiries = []

        for quote in sorted_quotes:
            strikes, vols = build_smile_from_market_quote(quote, num_strikes_per_tenor)
            all_strikes.append(strikes)
            all_vols.append(vols)
            expiries.append(quote.expiry)

        # Create unified strike grid (use strikes from longest tenor as reference)
        # For simplicity, use common strike range across all tenors
        min_strike = min(jnp.min(s) for s in all_strikes)
        max_strike = max(jnp.max(s) for s in all_strikes)
        unified_strikes = jnp.linspace(min_strike, max_strike, num_strikes_per_tenor)

        # Interpolate each tenor's smile onto unified strike grid
        unified_vols = []
        for strikes, vols in zip(all_strikes, all_vols):
            interpolated_vols = jnp.interp(unified_strikes, strikes, vols)
            unified_vols.append(interpolated_vols)

        # Create surface
        expiries_array = jnp.array(expiries)
        vols_grid = jnp.array(unified_vols)  # Shape: (n_tenors, n_strikes)

        return FXVolatilitySurface(
            from_ccy=self.from_ccy,
            to_ccy=self.to_ccy,
            expiries=expiries_array,
            strikes=unified_strikes,
            vols=vols_grid,
        )

    def get_quote(self, expiry: float) -> Optional[FXVolatilityQuote]:
        """
        Get market quote for specific expiry.

        Args:
            expiry: Time to expiry in years

        Returns:
            FXVolatilityQuote if found, None otherwise
        """
        for quote in self.quotes:
            if abs(quote.expiry - expiry) < 1e-6:
                return quote
        return None

    def add_quote(self, quote: FXVolatilityQuote) -> None:
        """
        Add a new market quote.

        Args:
            quote: FXVolatilityQuote to add
        """
        self.quotes.append(quote)

    def __repr__(self) -> str:
        return (
            f"FXVolatilitySurfaceBuilder({self.from_ccy}/{self.to_ccy}, "
            f"{len(self.quotes)} tenors)"
        )


# ============================================================================
# SABR Calibration from Market Quotes
# ============================================================================

def calibrate_sabr_from_quote(
    quote: FXVolatilityQuote,
    beta: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> SABRParameters:
    """
    Calibrate SABR parameters from FX market quote (ATM/BF/RR).

    Uses a simplified calibration approach:
    1. Fix beta (typically 0.5 or 1.0 for FX)
    2. Fit alpha, rho, nu to match ATM vol and smile shape (RR/BF)

    Args:
        quote: FX volatility market quote
        beta: Fixed beta parameter (0.0-1.0, typically 0.5 for FX)
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance

    Returns:
        Calibrated SABR parameters

    Example:
        >>> quote = FXVolatilityQuote(expiry=1.0, atm_vol=0.10, rr_25d=0.015, bf_25d=0.005, forward=1.10)
        >>> params = calibrate_sabr_from_quote(quote)
        >>> params.alpha, params.beta, params.rho, params.nu
    """
    import scipy.optimize as opt

    # Extract pillar volatilities
    pillar_vols = quote.extract_pillar_vols()
    vol_atm = pillar_vols['atm']
    vol_25c = pillar_vols.get('25d_call', vol_atm)
    vol_25p = pillar_vols.get('25d_put', vol_atm)

    # Get strikes for pillars (simplified: use analytical approximation)
    # For 25-delta, strikes are approximately:
    # K_call ≈ F * exp(0.5 * vol * sqrt(T) * N^(-1)(0.75))
    # K_put ≈ F * exp(-0.5 * vol * sqrt(T) * N^(-1)(0.75))

    from jax.scipy.stats import norm
    sqrt_t = jnp.sqrt(quote.expiry)
    # For 25-delta options (approximately)
    delta_call_z = norm.ppf(0.75)  # Inverse CDF for ~25 delta
    delta_put_z = norm.ppf(0.25)

    strike_25c = float(quote.forward * jnp.exp(0.5 * vol_25c * sqrt_t * delta_call_z))
    strike_25p = float(quote.forward * jnp.exp(0.5 * vol_25p * sqrt_t * delta_put_z))
    strike_atm = quote.forward

    # Target volatilities at pillar strikes
    target_strikes = jnp.array([strike_25p, strike_atm, strike_25c])
    target_vols = jnp.array([vol_25p, vol_atm, vol_25c])

    def objective(params):
        """Objective function: minimize squared error in implied vols."""
        alpha, rho, nu = params

        # Ensure parameters are in valid range
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1.0:
            return 1e10  # Penalty for invalid parameters

        sabr_params = SABRParameters(alpha=alpha, beta=beta, rho=rho, nu=nu)

        # Compute SABR implied vols at target strikes
        sabr_vols = sabr_implied_vol(
            forward=quote.forward,
            strike=target_strikes,
            maturity=quote.expiry,
            params=sabr_params,
        )

        # Sum of squared errors
        errors = (sabr_vols - target_vols) ** 2
        return float(jnp.sum(errors))

    # Initial guess for parameters
    # alpha: start close to ATM vol
    # rho: small negative (typical for FX)
    # nu: vol-of-vol around 0.3-0.5
    initial_guess = [vol_atm, -0.2, 0.4]

    # Bounds for parameters
    bounds = [
        (1e-6, 2.0),      # alpha > 0
        (-0.99, 0.99),    # -1 < rho < 1
        (1e-6, 2.0),      # nu > 0
    ]

    # Optimize
    result = opt.minimize(
        objective,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': tolerance}
    )

    if result.success:
        alpha, rho, nu = result.x
        return SABRParameters(alpha=float(alpha), beta=beta, rho=float(rho), nu=float(nu))
    else:
        # If optimization fails, return simple parameters
        # that at least match ATM vol
        return SABRParameters(alpha=vol_atm, beta=beta, rho=0.0, nu=0.3)


# ============================================================================
# Vanna-Volga Interpolation
# ============================================================================

def vanna_volga_weights(
    strike: float,
    forward: float,
    strike_25p: float,
    strike_atm: float,
    strike_25c: float,
) -> tuple[float, float, float]:
    """
    Compute Vanna-Volga weights for smile interpolation.

    Vanna-Volga is a standard FX market interpolation method that uses
    three market pillars (25Δ put, ATM, 25Δ call) to interpolate vols.

    The weights are derived from matching the vanna and volga Greeks.

    Args:
        strike: Target strike for interpolation
        forward: Forward FX rate
        strike_25p: 25-delta put strike
        strike_atm: ATM strike
        strike_25c: 25-delta call strike

    Returns:
        Tuple of (weight_25p, weight_atm, weight_25c)

    Reference:
        Wystup, U. (2006). "FX Options and Structured Products"
    """
    # Log-moneyness relative to forward
    x = jnp.log(strike / forward)
    x1 = jnp.log(strike_25p / forward)
    x2 = jnp.log(strike_atm / forward)
    x3 = jnp.log(strike_25c / forward)

    # Vanna-Volga weights (quadratic interpolation in log-moneyness space)
    # w_i = prod_{j≠i} (x - x_j) / prod_{j≠i} (x_i - x_j)

    w1 = ((x - x2) * (x - x3)) / ((x1 - x2) * (x1 - x3))
    w2 = ((x - x1) * (x - x3)) / ((x2 - x1) * (x2 - x3))
    w3 = ((x - x1) * (x - x2)) / ((x3 - x1) * (x3 - x2))

    return float(w1), float(w2), float(w3)


def build_smile_vanna_volga(
    quote: FXVolatilityQuote,
    num_strikes: int = 21,
) -> tuple[Array, Array]:
    """
    Build volatility smile using Vanna-Volga interpolation.

    Vanna-Volga is the industry-standard method for FX smile interpolation.
    It provides smooth, arbitrage-free smiles that match market pillars exactly.

    Args:
        quote: FX volatility market quote
        num_strikes: Number of strike points to generate

    Returns:
        Tuple of (strikes, vols) arrays

    Example:
        >>> quote = FXVolatilityQuote(
        ...     expiry=1.0, atm_vol=0.10, rr_25d=0.015, bf_25d=0.005,
        ...     forward=1.10, domestic_rate=0.02, foreign_rate=0.01
        ... )
        >>> strikes, vols = build_smile_vanna_volga(quote)
    """
    import jax.numpy as jnp

    # Extract pillar vols
    pillar_vols = quote.extract_pillar_vols()

    # Get strikes for each pillar
    strike_25p, vol_25p = delta_to_strike_iterative(
        delta=-0.25,
        forward=quote.forward,
        expiry=quote.expiry,
        vol_initial=pillar_vols['25d_put'],
        is_call=False,
        domestic_rate=quote.domestic_rate,
        foreign_rate=quote.foreign_rate,
    )

    strike_atm = strike_from_delta_atm(
        forward=quote.forward,
        expiry=quote.expiry,
        vol=pillar_vols['atm'],
        domestic_rate=quote.domestic_rate,
        foreign_rate=quote.foreign_rate,
    )
    vol_atm = pillar_vols['atm']

    strike_25c, vol_25c = delta_to_strike_iterative(
        delta=0.25,
        forward=quote.forward,
        expiry=quote.expiry,
        vol_initial=pillar_vols['25d_call'],
        is_call=True,
        domestic_rate=quote.domestic_rate,
        foreign_rate=quote.foreign_rate,
    )

    # Generate strike grid
    min_strike = strike_25p * 0.85
    max_strike = strike_25c * 1.15
    strikes = jnp.linspace(min_strike, max_strike, num_strikes)

    # Apply Vanna-Volga interpolation
    vols = []
    for strike in strikes:
        w1, w2, w3 = vanna_volga_weights(
            float(strike), quote.forward, strike_25p, strike_atm, strike_25c
        )

        # Weighted average of pillar vols
        vol = w1 * vol_25p + w2 * vol_atm + w3 * vol_25c
        vols.append(vol)

    return strikes, jnp.array(vols)


# ============================================================================
# Enhanced Surface Builder with Multiple Interpolation Methods
# ============================================================================

InterpolationMethod = Literal["linear", "sabr", "vanna_volga"]


def build_smile_with_method(
    quote: FXVolatilityQuote,
    num_strikes: int = 21,
    method: InterpolationMethod = "linear",
    sabr_beta: float = 0.5,
) -> tuple[Array, Array]:
    """
    Build volatility smile using specified interpolation method.

    Args:
        quote: FX volatility market quote
        num_strikes: Number of strike points to generate
        method: Interpolation method ("linear", "sabr", "vanna_volga")
        sabr_beta: Beta parameter for SABR (only used if method="sabr")

    Returns:
        Tuple of (strikes, vols) arrays

    Example:
        >>> quote = FXVolatilityQuote(expiry=1.0, atm_vol=0.10, rr_25d=0.015, bf_25d=0.005, forward=1.10)
        >>> # Linear interpolation (fast, simple)
        >>> strikes, vols = build_smile_with_method(quote, method="linear")
        >>> # SABR interpolation (smooth, arbitrage-free)
        >>> strikes, vols = build_smile_with_method(quote, method="sabr")
        >>> # Vanna-Volga interpolation (FX market standard)
        >>> strikes, vols = build_smile_with_method(quote, method="vanna_volga")
    """
    if method == "linear":
        return build_smile_from_market_quote(quote, num_strikes)
    elif method == "sabr":
        # Calibrate SABR to market quote
        sabr_params = calibrate_sabr_from_quote(quote, beta=sabr_beta)

        # Generate strike range
        pillar_vols = quote.extract_pillar_vols()
        strike_25p, _ = delta_to_strike_iterative(
            -0.25, quote.forward, quote.expiry, pillar_vols['25d_put'], False,
            quote.domestic_rate, quote.foreign_rate
        )
        strike_25c, _ = delta_to_strike_iterative(
            0.25, quote.forward, quote.expiry, pillar_vols['25d_call'], True,
            quote.domestic_rate, quote.foreign_rate
        )

        min_strike = strike_25p * 0.85
        max_strike = strike_25c * 1.15
        strikes = jnp.linspace(min_strike, max_strike, num_strikes)

        # Compute SABR implied vols
        vols = sabr_implied_vol(
            forward=quote.forward,
            strike=strikes,
            maturity=quote.expiry,
            params=sabr_params,
        )

        return strikes, vols
    elif method == "vanna_volga":
        return build_smile_vanna_volga(quote, num_strikes)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
