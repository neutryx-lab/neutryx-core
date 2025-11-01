"""Interest-rate curve bootstrapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import jax.numpy as jnp

ArrayLike = Union[float, jnp.ndarray]


@dataclass
class FlatCurve:
    """Simple continuously-compounded flat curve."""

    r: float = 0.01

    def df(self, t: ArrayLike) -> ArrayLike:
        t_arr = jnp.asarray(t)
        return jnp.exp(-self.r * t_arr)


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
    """Piecewise log-linear discount curve built from money-market instruments."""

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
        t_arr = jnp.asarray(t)
        if t_arr.ndim == 0 and float(t_arr) in self._nodes:
            return self._nodes[float(t_arr)]

        log_df = jnp.log(self._dfs)
        interpolated = jnp.exp(jnp.interp(t_arr, self._times, log_df))
        return interpolated

    def zero_rate(self, t: ArrayLike) -> ArrayLike:
        t_arr = jnp.asarray(t)
        discount = self.df(t_arr)
        safe_t = jnp.where(t_arr == 0.0, jnp.nan, t_arr)
        zero_rates = -jnp.log(discount) / safe_t
        if jnp.isscalar(zero_rates):
            return 0.0 if float(t_arr) == 0.0 else zero_rates
        zero_rates = jnp.where(t_arr == 0.0, 0.0, zero_rates)
        return zero_rates

    def forward_rate(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        df0 = float(self.df(t0))
        df1 = float(self.df(t1))
        return (df0 / df1 - 1.0) / (t1 - t0)


def _instrument_maturity(instrument: Union[Deposit, FixedRateSwap]) -> float:
    if isinstance(instrument, Deposit):
        return instrument.maturity
    if isinstance(instrument, FixedRateSwap):
        return instrument.payment_times[-1]
    raise TypeError(f"Unsupported instrument type: {type(instrument)!r}")
