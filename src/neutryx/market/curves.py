"""Interest-rate curve bootstrapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

# Type alias for all supported market rate instruments
MarketInstrument = Union[
    "Deposit", "FRA", "Future", "FixedRateSwap", "OIS", "TenorBasisSwap", "CurrencyBasisSwap"
]

import jax.numpy as jnp
from jax import Array

ArrayLike = Union[float, Array]


@dataclass
class FlatCurve:
    """
    Simple continuously-compounded flat discount curve.

    Implements the DiscountCurve protocol with a constant rate.

    Attributes:
        r: Continuously-compounded rate (default 0.01 = 1%)
    """

    r: float = 0.01

    def df(self, t: ArrayLike) -> ArrayLike:
        """Compute discount factor: DF(t) = exp(-r*t)."""
        t_arr = jnp.asarray(t)
        return jnp.exp(-self.r * t_arr)

    def value(self, t: ArrayLike) -> ArrayLike:
        """Alias for df(t) to satisfy Curve protocol."""
        return self.df(t)

    def __call__(self, t: ArrayLike) -> ArrayLike:
        """Alias for df(t) for convenient syntax: curve(t)."""
        return self.df(t)

    def zero_rate(self, t: ArrayLike) -> ArrayLike:
        """Return the constant zero rate."""
        t_arr = jnp.asarray(t)
        return jnp.full_like(t_arr, self.r, dtype=float)

    def forward_rate(self, t1: ArrayLike, t2: ArrayLike) -> ArrayLike:
        """Return the constant forward rate (same as zero rate)."""
        # For flat curve, forward rate is constant
        t1_arr = jnp.asarray(t1)
        return jnp.full_like(t1_arr, self.r, dtype=float)


@dataclass
class Deposit:
    """Simple money-market deposit used for bootstrapping."""

    maturity: float
    rate: float

    def bootstrap(self) -> Tuple[float, float]:
        """Return the maturity and discount factor implied by the deposit."""

        discount = 1.0 / (1.0 + self.rate * self.maturity)
        return self.maturity, discount


@dataclass
class FixedRateSwap:
    """Par swap specification for bootstrapping discount factors."""

    fixed_rate: float
    payment_times: Sequence[float]
    accrual_factors: Sequence[float]

    def bootstrap(self, curve: "BootstrappedCurve") -> Tuple[float, float]:
        if len(self.payment_times) != len(self.accrual_factors):
            raise ValueError("Payment times and accrual factors must have the same length")

        if not self.payment_times:
            raise ValueError("Swap must contain at least one payment")

        known_value = 0.0
        for time, accrual in zip(self.payment_times[:-1], self.accrual_factors[:-1]):
            discount = curve.node_df(time)
            if discount is None:
                raise ValueError(
                    f"Discount factor for {time}y must be bootstrapped before the swap"
                )
            known_value += accrual * float(discount)

        final_time = self.payment_times[-1]
        final_accrual = self.accrual_factors[-1]

        numerator = 1.0 - self.fixed_rate * known_value
        denominator = 1.0 + self.fixed_rate * final_accrual
        discount_final = numerator / denominator

        return final_time, discount_final


@dataclass
class FRA:
    """
    Forward Rate Agreement (FRA) for bootstrapping.

    A FRA is a contract that fixes the interest rate for a future period.
    For example, a 3x6 FRA fixes the 3-month rate starting in 3 months.

    Attributes:
        start: Start time of the forward period (in years)
        end: End time of the forward period (in years)
        rate: Fixed rate of the FRA (simple rate, not continuously compounded)

    Example:
        >>> # 3x6 FRA at 5.5%
        >>> fra = FRA(start=0.25, end=0.5, rate=0.055)
    """

    start: float
    end: float
    rate: float

    def bootstrap(self, curve: "BootstrappedCurve") -> Tuple[float, float]:
        """
        Bootstrap discount factor at end time using FRA rate.

        The FRA rate implies:
        DF(start) / DF(end) = 1 + rate * (end - start)

        Therefore:
        DF(end) = DF(start) / (1 + rate * (end - start))
        """
        if self.start >= self.end:
            raise ValueError("FRA start must be before end")

        if self.start < 0 or self.end < 0:
            raise ValueError("FRA times must be non-negative")

        # Get discount factor at start time
        df_start = curve.node_df(self.start)
        if df_start is None:
            raise ValueError(
                f"Discount factor for {self.start}y must be bootstrapped before the FRA"
            )

        # Calculate discount factor at end time
        accrual = self.end - self.start
        df_end = float(df_start) / (1.0 + self.rate * accrual)

        return self.end, df_end


@dataclass
class Future:
    """
    Interest rate future for bootstrapping.

    Interest rate futures are standardized exchange-traded contracts.
    The future price is quoted as 100 - implied rate (%).

    Attributes:
        start: Start time of the forward period (in years)
        end: End time of the forward period (in years)
        price: Future price (e.g., 94.5 implies 5.5% rate)
        convexity_adjustment: Adjustment for difference between futures and forwards (default 0.0)

    Example:
        >>> # 3-month future starting in 0.25 years, price 94.5 (5.5% implied)
        >>> future = Future(start=0.25, end=0.5, price=94.5, convexity_adjustment=0.0001)
    """

    start: float
    end: float
    price: float
    convexity_adjustment: float = 0.0

    def bootstrap(self, curve: "BootstrappedCurve") -> Tuple[float, float]:
        """
        Bootstrap discount factor at end time using future price.

        The future price implies a rate:
        implied_rate = (100 - price) / 100

        Adjusting for convexity:
        forward_rate = implied_rate - convexity_adjustment

        Then apply FRA formula:
        DF(end) = DF(start) / (1 + forward_rate * (end - start))
        """
        if self.start >= self.end:
            raise ValueError("Future start must be before end")

        if self.start < 0 or self.end < 0:
            raise ValueError("Future times must be non-negative")

        # Get discount factor at start time
        df_start = curve.node_df(self.start)
        if df_start is None:
            raise ValueError(
                f"Discount factor for {self.start}y must be bootstrapped before the future"
            )

        # Extract implied rate from future price
        # Price = 100 - rate(%), so rate = (100 - price) / 100
        implied_rate = (100.0 - self.price) / 100.0

        # Apply convexity adjustment
        # Futures are margined daily, creating convexity bias vs forwards
        forward_rate = implied_rate - self.convexity_adjustment

        # Calculate discount factor at end time
        accrual = self.end - self.start
        df_end = float(df_start) / (1.0 + forward_rate * accrual)

        return self.end, df_end


@dataclass
class OIS:
    """
    Overnight Index Swap (OIS) for bootstrapping RFR discount curves.

    OIS swaps exchange fixed vs overnight floating rate (e.g., SOFR, ESTR, TONAR).
    Used to build the risk-free discount curve for collateralized derivatives.

    Attributes:
        fixed_rate: Fixed rate of the OIS
        payment_times: Times of fixed leg payments (in years)
        accrual_factors: Accrual factors for each payment period
        compounding: Compounding convention ("compound" or "averaging")

    Example:
        >>> # 2Y SOFR OIS at 5.5% with semi-annual payments
        >>> ois = OIS(
        ...     fixed_rate=0.055,
        ...     payment_times=[0.5, 1.0, 1.5, 2.0],
        ...     accrual_factors=[0.5, 0.5, 0.5, 0.5],
        ...     compounding="compound"
        ... )
    """

    fixed_rate: float
    payment_times: Sequence[float]
    accrual_factors: Sequence[float]
    compounding: str = "compound"  # "compound" or "averaging"

    def bootstrap(self, curve: "BootstrappedCurve") -> Tuple[float, float]:
        """
        Bootstrap discount factor at final maturity using OIS rate.

        For OIS, the floating leg is computed as:
        - Compound: (1/DF(start) - 1/DF(end)) = compound rate
        - Averaging: Simple average of daily rates

        For simplicity, we treat OIS like a standard fixed rate swap here.
        In production, the compounding logic would be more sophisticated.
        """
        if len(self.payment_times) != len(self.accrual_factors):
            raise ValueError("Payment times and accrual factors must have the same length")

        if not self.payment_times:
            raise ValueError("OIS must contain at least one payment")

        known_value = 0.0
        for time, accrual in zip(self.payment_times[:-1], self.accrual_factors[:-1]):
            discount = curve.node_df(time)
            if discount is None:
                raise ValueError(
                    f"Discount factor for {time}y must be bootstrapped before the OIS"
                )
            known_value += accrual * float(discount)

        final_time = self.payment_times[-1]
        final_accrual = self.accrual_factors[-1]

        # Par OIS: PV(fixed leg) = PV(floating leg)
        # PV(fixed leg) = fixed_rate * sum(DF(t_i) * accrual_i)
        # PV(floating leg) = 1 - DF(T) for compounded OIS
        numerator = 1.0 - self.fixed_rate * known_value
        denominator = 1.0 + self.fixed_rate * final_accrual
        discount_final = numerator / denominator

        return final_time, discount_final


@dataclass
class TenorBasisSwap:
    """
    Tenor basis swap for bootstrapping projection curves.

    A tenor basis swap exchanges floating payments at different tenors
    (e.g., 3M LIBOR vs 6M LIBOR) plus a basis spread.

    Attributes:
        basis_spread: Basis spread added to the shorter tenor leg (in decimal)
        payment_times: Times of payments (in years)
        accrual_factors: Accrual factors for each payment period
        short_tenor: Short tenor (e.g., "3M")
        long_tenor: Long tenor (e.g., "6M")
        discount_curve: Discount curve for present value calculation

    Example:
        >>> # 3M vs 6M basis swap with 10bp spread
        >>> basis = TenorBasisSwap(
        ...     basis_spread=0.0010,
        ...     payment_times=[0.5, 1.0],
        ...     accrual_factors=[0.5, 0.5],
        ...     short_tenor="3M",
        ...     long_tenor="6M"
        ... )
    """

    basis_spread: float
    payment_times: Sequence[float]
    accrual_factors: Sequence[float]
    short_tenor: str
    long_tenor: str
    discount_curve: Union["BootstrappedCurve", None] = None

    def bootstrap(
        self, curve: "BootstrappedCurve", discount_curve: "BootstrappedCurve" = None
    ) -> Tuple[float, float]:
        """
        Bootstrap forward rate at final maturity using tenor basis spread.

        The basis swap implies:
        PV(short_tenor + basis) = PV(long_tenor)

        This is used to build projection curves for different tenors
        given a discount curve and another projection curve.
        """
        if len(self.payment_times) != len(self.accrual_factors):
            raise ValueError("Payment times and accrual factors must have the same length")

        if not self.payment_times:
            raise ValueError("Tenor basis swap must contain at least one payment")

        # Use provided discount curve or fall back to the curve being built
        disc_curve = discount_curve or self.discount_curve or curve

        known_value = 0.0
        for time, accrual in zip(self.payment_times[:-1], self.accrual_factors[:-1]):
            # Get discount factor for present value
            df = disc_curve.node_df(time)
            if df is None:
                raise ValueError(
                    f"Discount factor for {time}y must be available before tenor basis"
                )
            known_value += accrual * float(df)

        final_time = self.payment_times[-1]
        final_accrual = self.accrual_factors[-1]

        # Solve for the forward rate that makes PV = 0
        # This is a simplified version; production would use iterative solving
        df_final = disc_curve.node_df(final_time)
        if df_final is None:
            raise ValueError(
                f"Discount factor for {final_time}y must be available"
            )

        # For now, return the final time and a forward rate adjustment
        # In practice, this would bootstrap a projection curve node
        forward_adjustment = self.basis_spread
        return final_time, float(df_final) * (1.0 + forward_adjustment)


@dataclass
class CurrencyBasisSwap:
    """
    Cross-currency basis swap for multi-currency curve building.

    A currency basis swap exchanges floating payments in different currencies,
    typically LIBOR vs LIBOR + basis spread, with initial and final notional exchanges.

    Attributes:
        basis_spread: Basis spread added to the foreign currency leg (in decimal)
        payment_times: Times of payments (in years)
        accrual_factors: Accrual factors for each payment period
        domestic_currency: Domestic currency (e.g., "USD")
        foreign_currency: Foreign currency (e.g., "JPY")
        fx_spot: Spot FX rate (domestic per foreign)
        domestic_discount_curve: Domestic currency discount curve
        foreign_projection_curve: Foreign currency projection curve (optional)

    Example:
        >>> # USD/JPY 3M basis swap with -15bp spread on JPY leg
        >>> xcs = CurrencyBasisSwap(
        ...     basis_spread=-0.0015,
        ...     payment_times=[0.25, 0.5, 0.75, 1.0],
        ...     accrual_factors=[0.25] * 4,
        ...     domestic_currency="USD",
        ...     foreign_currency="JPY",
        ...     fx_spot=110.0
        ... )
    """

    basis_spread: float
    payment_times: Sequence[float]
    accrual_factors: Sequence[float]
    domestic_currency: str
    foreign_currency: str
    fx_spot: float
    domestic_discount_curve: Union["BootstrappedCurve", None] = None
    foreign_projection_curve: Union["BootstrappedCurve", None] = None

    def bootstrap(
        self,
        curve: "BootstrappedCurve",
        domestic_discount_curve: "BootstrappedCurve" = None,
    ) -> Tuple[float, float]:
        """
        Bootstrap foreign discount curve using cross-currency basis.

        The currency basis swap implies:
        PV_domestic(domestic_leg) = FX * PV_foreign(foreign_leg + basis)

        This is used to build collateralized discount curves in foreign currencies.
        """
        if len(self.payment_times) != len(self.accrual_factors):
            raise ValueError("Payment times and accrual factors must have the same length")

        if not self.payment_times:
            raise ValueError("Currency basis swap must contain at least one payment")

        # Use provided discount curve or fall back to stored curve
        dom_curve = domestic_discount_curve or self.domestic_discount_curve

        if dom_curve is None:
            raise ValueError("Domestic discount curve must be provided")

        known_value = 0.0
        for time, accrual in zip(self.payment_times[:-1], self.accrual_factors[:-1]):
            # Get discount factor from domestic curve
            df_dom = dom_curve.node_df(time)
            if df_dom is None:
                raise ValueError(
                    f"Domestic discount factor for {time}y must be available"
                )

            # Get discount factor from foreign curve being built
            df_for = curve.node_df(time)
            if df_for is None:
                raise ValueError(
                    f"Foreign discount factor for {time}y must be bootstrapped before XCS"
                )

            known_value += accrual * float(df_for)

        final_time = self.payment_times[-1]
        final_accrual = self.accrual_factors[-1]

        # Solve for foreign discount factor
        # PV_dom = FX * PV_for implies:
        # DF_for(T) = (1 - basis * known_value) / (1 + basis * final_accrual)
        # This is simplified; production would use full XCS pricing formula
        df_dom_final = dom_curve.node_df(final_time)
        if df_dom_final is None:
            raise ValueError(
                f"Domestic discount factor for {final_time}y must be available"
            )

        numerator = 1.0 - self.basis_spread * known_value
        denominator = 1.0 + self.basis_spread * final_accrual
        df_foreign_final = float(df_dom_final) * numerator / denominator

        return final_time, df_foreign_final


class BootstrappedCurve:
    """
    Piecewise log-linear discount curve built from money-market instruments.

    Implements the DiscountCurve protocol with log-linear interpolation between nodes.

    The curve is bootstrapped from deposits, FRAs, futures, and swaps, producing
    a discount factor at each instrument maturity. Between nodes, log-linear
    interpolation is used to ensure smooth forward rates.

    Supported instruments:
        - Deposit: Simple money market deposits
        - FRA: Forward Rate Agreements
        - Future: Interest rate futures (e.g., Eurodollar, SOFR futures)
        - FixedRateSwap: Par swaps for longer maturities

    Example:
        >>> instruments = [
        ...     Deposit(maturity=0.25, rate=0.05),
        ...     FRA(start=0.25, end=0.5, rate=0.052),
        ...     Future(start=0.5, end=0.75, price=94.7),
        ...     FixedRateSwap(fixed_rate=0.055, payment_times=[1.0, 2.0], accrual_factors=[1.0, 1.0])
        ... ]
        >>> curve = BootstrappedCurve(instruments)
        >>> curve.df(1.5)  # Get discount factor at 1.5 years
    """

    def __init__(self, instruments: Iterable[MarketInstrument]):
        self._nodes: Dict[float, float] = {0.0: 1.0}
        ordered_instruments = sorted(instruments, key=_instrument_maturity)
        for instrument in ordered_instruments:
            maturity, discount = self._bootstrap_instrument(instrument)
            self.add_node(maturity, discount)

        self._rebuild_arrays()

    def _bootstrap_instrument(self, instrument: MarketInstrument) -> Tuple[float, float]:
        if isinstance(instrument, Deposit):
            return instrument.bootstrap()
        if isinstance(instrument, FRA):
            return instrument.bootstrap(self)
        if isinstance(instrument, Future):
            return instrument.bootstrap(self)
        if isinstance(instrument, FixedRateSwap):
            return instrument.bootstrap(self)
        if isinstance(instrument, OIS):
            return instrument.bootstrap(self)
        if isinstance(instrument, TenorBasisSwap):
            return instrument.bootstrap(self)
        if isinstance(instrument, CurrencyBasisSwap):
            return instrument.bootstrap(self)
        raise TypeError(f"Unsupported instrument type: {type(instrument)!r}")

    def _rebuild_arrays(self) -> None:
        items: List[Tuple[float, float]] = sorted(self._nodes.items())
        self._times = jnp.array([t for t, _ in items])
        self._dfs = jnp.array([df for _, df in items])

    def add_node(self, maturity: float, discount: float) -> None:
        if maturity <= 0:
            raise ValueError("Maturity must be positive")
        if discount <= 0:
            raise ValueError("Discount factors must be positive")
        self._nodes[maturity] = discount

    def node_df(self, maturity: float) -> Union[float, None]:
        return self._nodes.get(maturity)

    def df(self, t: ArrayLike) -> ArrayLike:
        """Compute discount factor with log-linear interpolation."""
        t_arr = jnp.asarray(t)
        if t_arr.ndim == 0 and float(t_arr) in self._nodes:
            return self._nodes[float(t_arr)]

        log_df = jnp.log(self._dfs)
        interpolated = jnp.exp(jnp.interp(t_arr, self._times, log_df))
        return interpolated

    def value(self, t: ArrayLike) -> ArrayLike:
        """Alias for df(t) to satisfy Curve protocol."""
        return self.df(t)

    def __call__(self, t: ArrayLike) -> ArrayLike:
        """Alias for df(t) for convenient syntax: curve(t)."""
        return self.df(t)

    def zero_rate(self, t: ArrayLike) -> ArrayLike:
        """Compute continuously-compounded zero rate: r(t) = -ln(DF(t))/t."""
        t_arr = jnp.asarray(t)
        discount = self.df(t_arr)
        safe_t = jnp.where(t_arr == 0.0, jnp.nan, t_arr)
        zero_rates = -jnp.log(discount) / safe_t
        if jnp.isscalar(zero_rates):
            return 0.0 if float(t_arr) == 0.0 else zero_rates
        zero_rates = jnp.where(t_arr == 0.0, 0.0, zero_rates)
        return zero_rates

    def forward_rate(self, t0: ArrayLike, t1: ArrayLike) -> ArrayLike:
        """
        Compute continuously-compounded forward rate between t0 and t1.

        Forward rate f(t0,t1) satisfies: DF(t0,t1) = exp(-f(t0,t1)*(t1-t0))
        where DF(t0,t1) = DF(t1) / DF(t0).

        Args:
            t0: Start time(s)
            t1: End time(s)

        Returns:
            Forward rate(s) f(t0, t1)
        """
        t0_arr = jnp.asarray(t0)
        t1_arr = jnp.asarray(t1)

        df0 = self.df(t0_arr)
        df1 = self.df(t1_arr)
        delta = t1_arr - t0_arr
        if jnp.isscalar(delta):
            if float(delta) == 0.0:
                raise ValueError("t1 must be greater than t0 for forward_rate computation")
            return (df0 / df1 - 1.0) / float(delta)

        delta_safe = jnp.where(delta == 0.0, jnp.nan, delta)
        forward = (df0 / df1 - 1.0) / delta_safe
        return jnp.where(delta == 0.0, jnp.nan, forward)


def _instrument_maturity(instrument: MarketInstrument) -> float:
    """Extract the maturity time from a market instrument."""
    if isinstance(instrument, Deposit):
        return instrument.maturity
    if isinstance(instrument, FRA):
        return instrument.end
    if isinstance(instrument, Future):
        return instrument.end
    if isinstance(instrument, FixedRateSwap):
        return instrument.payment_times[-1]
    if isinstance(instrument, OIS):
        return instrument.payment_times[-1]
    if isinstance(instrument, TenorBasisSwap):
        return instrument.payment_times[-1]
    if isinstance(instrument, CurrencyBasisSwap):
        return instrument.payment_times[-1]
    raise TypeError(f"Unsupported instrument type: {type(instrument)!r}")


@dataclass
class DividendYieldCurve:
    """
    Piecewise-constant dividend yield curve.

    Implements the Curve protocol for dividend yields. Used for equity forwards
    and equity derivatives pricing.

    Attributes:
        times: Pillar times (in years from reference)
        yields: Dividend yields at each pillar (continuously compounded)

    Example:
        >>> curve = DividendYieldCurve(
        ...     times=jnp.array([0.0, 1.0, 2.0, 5.0]),
        ...     yields=jnp.array([0.02, 0.025, 0.03, 0.03])
        ... )
        >>> q = curve(1.5)  # Get dividend yield at 1.5 years
    """

    times: Array
    yields: Array

    def __post_init__(self):
        """Validate curve inputs."""
        if len(self.times) != len(self.yields):
            raise ValueError("times and yields must have same length")
        if len(self.times) == 0:
            raise ValueError("Curve must have at least one point")

        # Convert to JAX arrays
        object.__setattr__(self, "times", jnp.asarray(self.times))
        object.__setattr__(self, "yields", jnp.asarray(self.yields))

    def value(self, t: ArrayLike) -> ArrayLike:
        """
        Get dividend yield at time t using piecewise-constant interpolation.

        Args:
            t: Time(s) in years

        Returns:
            Dividend yield(s) at time t
        """
        t_arr = jnp.asarray(t)
        # Use right-continuous step function (previous value)
        return jnp.interp(t_arr, self.times, self.yields, left=self.yields[0])

    def __call__(self, t: ArrayLike) -> ArrayLike:
        """Alias for value(t)."""
        return self.value(t)

    def integrated_yield(self, t: float) -> float:
        """
        Compute integral of yield from 0 to t.

        Used for forward price calculations: F = S * exp(∫q(s)ds).

        Args:
            t: Time in years

        Returns:
            Integral of yield from 0 to t
        """
        # Piecewise-constant integration
        integral = 0.0
        prev_time = 0.0

        for i, time in enumerate(self.times):
            if time >= t:
                # Add partial segment
                integral += float(self.yields[i]) * (t - prev_time)
                break
            else:
                # Add full segment
                if i < len(self.times) - 1:
                    integral += float(self.yields[i]) * (time - prev_time)
                    prev_time = time

        # Handle extrapolation beyond last pillar
        if t > self.times[-1]:
            integral += float(self.yields[-1]) * (t - float(self.times[-1]))

        return integral


@dataclass
class ForwardRateCurve:
    """
    Forward rate curve for LIBOR/SOFR-style forward rates.

    Implements the Curve protocol for forward rates. Can be used for
    multi-curve pricing where discounting and projection curves differ.

    Attributes:
        tenor: Forward tenor (e.g., "3M", "6M") as string
        times: Pillar times (in years from reference)
        forwards: Forward rates at each pillar (continuously compounded)

    Example:
        >>> curve = ForwardRateCurve(
        ...     tenor="3M",
        ...     times=jnp.array([0.25, 0.5, 1.0, 2.0, 5.0]),
        ...     forwards=jnp.array([0.05, 0.052, 0.053, 0.055, 0.056])
        ... )
        >>> f = curve(1.5)  # Get 3M forward rate at 1.5 years
    """

    tenor: str
    times: Array
    forwards: Array

    def __post_init__(self):
        """Validate curve inputs."""
        if len(self.times) != len(self.forwards):
            raise ValueError("times and forwards must have same length")
        if len(self.times) == 0:
            raise ValueError("Curve must have at least one point")

        # Convert to JAX arrays
        object.__setattr__(self, "times", jnp.asarray(self.times))
        object.__setattr__(self, "forwards", jnp.asarray(self.forwards))

    def value(self, t: ArrayLike) -> ArrayLike:
        """
        Get forward rate at time t using linear interpolation.

        Args:
            t: Time(s) in years

        Returns:
            Forward rate(s) at time t
        """
        t_arr = jnp.asarray(t)
        return jnp.interp(t_arr, self.times, self.forwards)

    def __call__(self, t: ArrayLike) -> ArrayLike:
        """Alias for value(t)."""
        return self.value(t)

    def forward_df(self, t1: float, t2: float) -> float:
        """
        Compute forward discount factor from t1 to t2.

        Uses the forward rates to compute DF(t1, t2).

        Args:
            t1: Start time
            t2: End time

        Returns:
            Forward discount factor
        """
        # Simple approximation: use average forward rate
        # For production, would integrate the forward curve
        avg_forward = float(self.value((t1 + t2) / 2))
        return jnp.exp(-avg_forward * (t2 - t1))
