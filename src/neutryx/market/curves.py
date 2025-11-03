"""Interest-rate curve bootstrapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

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


class BootstrappedCurve:
    """
    Piecewise log-linear discount curve built from money-market instruments.

    Implements the DiscountCurve protocol with log-linear interpolation between nodes.

    The curve is bootstrapped from deposits and swaps, producing a discount factor
    at each instrument maturity. Between nodes, log-linear interpolation is used
    to ensure smooth forward rates.
    """

    def __init__(self, instruments: Iterable[Union[Deposit, FixedRateSwap]]):
        self._nodes: Dict[float, float] = {0.0: 1.0}
        ordered_instruments = sorted(instruments, key=_instrument_maturity)
        for instrument in ordered_instruments:
            maturity, discount = self._bootstrap_instrument(instrument)
            self.add_node(maturity, discount)

        self._rebuild_arrays()

    def _bootstrap_instrument(self, instrument: Union[Deposit, FixedRateSwap]) -> Tuple[float, float]:
        if isinstance(instrument, Deposit):
            return instrument.bootstrap()
        if isinstance(instrument, FixedRateSwap):
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

        # Forward rate: -ln(df1/df0) / (t1-t0)
        return -jnp.log(df1 / df0) / (t1_arr - t0_arr)


def _instrument_maturity(instrument: Union[Deposit, FixedRateSwap]) -> float:
    if isinstance(instrument, Deposit):
        return instrument.maturity
    if isinstance(instrument, FixedRateSwap):
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
