"""Binomial tree generators and pricing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

Array = np.ndarray
ExerciseStyle = Literal["european", "american"]


class PayoffFn(Protocol):
    """Protocol for payoff functions evaluated on the asset price."""

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        ...


@dataclass
class BinomialModel:
    """Parameters for a Cox-Ross-Rubinstein style binomial tree."""

    S0: float
    r: float
    q: float
    sigma: float
    T: float
    steps: int
    dtype: np.dtype = np.float64

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.T <= 0.0:
            raise ValueError("maturity T must be positive")

    @property
    def dt(self) -> float:
        return float(self.T) / float(self.steps)

    @property
    def up(self) -> float:
        dt = self.dt
        return float(np.exp(self.sigma * np.sqrt(dt)))

    @property
    def down(self) -> float:
        u = self.up
        return float(1.0 / u)

    @property
    def discount(self) -> float:
        return float(np.exp(-self.r * self.dt))

    @property
    def prob(self) -> float:
        u, d = self.up, self.down
        p = risk_neutral_prob(self.r, self.q, self.dt, u, d)
        return float(p)


def binomial_parameters(sigma: float, dt: float) -> tuple[float, float]:
    """Return up/down factors for a Cox-Ross-Rubinstein binomial tree."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    vol = np.asarray(sigma, dtype=np.float64)
    up = np.exp(vol * np.sqrt(dt))
    down = 1.0 / up
    return float(up), float(down)


def risk_neutral_prob(r: float, q: float, dt: float, up: float, down: float) -> float:
    """Compute risk-neutral probability for a binomial tree."""

    growth = np.exp((r - q) * dt)
    denom = up - down
    if denom == 0.0:
        raise ValueError("up and down factors must differ")
    p = (growth - down) / denom
    if not (0.0 <= p <= 1.0):
        raise ValueError("risk-neutral probability outside [0, 1]")
    return float(p)


def generate_binomial_tree(
    S0: float,
    up: float,
    down: float,
    steps: int,
    *,
    dtype: np.dtype = np.float64,
) -> Array:
    """Generate a recombining binomial price tree."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    dtype_np = np.dtype(dtype)
    tree = np.full((steps + 1, steps + 1), np.nan, dtype=dtype_np)
    for i in range(steps + 1):
        j = np.arange(i + 1, dtype=dtype_np)
        level_prices = (S0 * (up ** j) * (down ** (i - j))).astype(dtype_np)
        tree[: i + 1, i] = level_prices
    return tree


def _call_payoff(payoff: PayoffFn, spots: np.ndarray, dtype: np.dtype) -> np.ndarray:
    values = payoff(np.asarray(spots, dtype=dtype))
    return np.asarray(values, dtype=dtype)


def _backward_induction(
    tree: Array,
    payoff: PayoffFn,
    prob: float,
    discount: float,
    *,
    exercise: ExerciseStyle,
) -> np.ndarray:
    """Run backward induction over a price tree to compute option values."""

    dtype = tree.dtype
    steps = tree.shape[1] - 1
    values = _call_payoff(payoff, tree[: steps + 1, steps], dtype)
    if values.shape[0] != steps + 1:
        raise ValueError("payoff must return a vector matching terminal nodes")
    for i in range(steps - 1, -1, -1):
        continuation = discount * (
            prob * values[1 : i + 2] + (1.0 - prob) * values[0 : i + 1]
        )
        if exercise == "american":
            immediate = _call_payoff(payoff, tree[: i + 1, i], dtype)
            continuation = np.maximum(continuation, immediate)
        values = continuation
    return values


def price_binomial(
    model: BinomialModel,
    payoff: PayoffFn,
    *,
    exercise: ExerciseStyle = "european",
) -> float:
    """Price an option via backward induction on a binomial tree."""

    up, down = model.up, model.down
    prob = model.prob
    discount = model.discount
    tree = generate_binomial_tree(model.S0, up, down, model.steps, dtype=model.dtype)
    values = _backward_induction(tree, payoff, prob, discount, exercise=exercise)
    return float(values[0])
