"""Ladder option payoffs.

Ladder options are path-dependent options where profits are "locked in"
when the underlying reaches certain predetermined levels (rungs) during
the option's life. Once a rung is hit, that level becomes the new minimum
payout, protecting gains even if the market subsequently declines.

Key Features:
- Lock in profits at predetermined levels
- Protection against downside after hitting rungs
- More expensive than vanilla options due to profit protection
"""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import PathProduct


@dataclass
class LadderCall(PathProduct):
    """Ladder call option with fixed rungs.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    rungs : jnp.ndarray
        Array of ladder levels (ascending order)

    Notes
    -----
    A ladder call option allows the holder to lock in profits when the
    underlying price reaches certain predetermined levels (rungs).

    At maturity:
    - If no rungs were hit: max(S_T - K, 0) [vanilla call]
    - If rungs were hit: max(highest_rung_hit - K, S_T - K, 0)

    Example:
    - Strike K = 100
    - Rungs at [110, 120, 130]
    - If spot reaches 125 during life (hitting rungs 110 and 120):
      * Minimum payoff at maturity = 120 - 100 = 20
      * If S_T = 115 at maturity: payoff = 20 (locked in)
      * If S_T = 135 at maturity: payoff = 35 (greater than locked 20)
    """

    K: float
    T: float
    rungs: jnp.ndarray

    def __post_init__(self):
        self.rungs = jnp.asarray(self.rungs)
        # Ensure rungs are sorted in ascending order
        self.rungs = jnp.sort(self.rungs)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Calculate ladder call payoff based on path.

        The payoff is the maximum of:
        1. The highest rung hit during the path
        2. The terminal payoff (S_T - K)
        3. Zero
        """
        path = jnp.asarray(path)

        # Find the highest price along the path
        max_price = path.max()

        # Determine the highest rung hit
        # A rung is hit if max_price >= rung_level
        rungs_hit = max_price >= self.rungs

        # Get the highest rung hit (or 0 if none hit)
        if jnp.any(rungs_hit):
            highest_rung = self.rungs[rungs_hit][-1]  # Last true value
            locked_in_profit = highest_rung - self.K
        else:
            locked_in_profit = 0.0

        # Terminal payoff
        terminal_price = path[-1]
        terminal_payoff = terminal_price - self.K

        # Payoff is max of locked-in profit and terminal payoff
        return jnp.maximum(locked_in_profit, jnp.maximum(terminal_payoff, 0.0))


@dataclass
class LadderPut(PathProduct):
    """Ladder put option with fixed rungs.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    rungs : jnp.ndarray
        Array of ladder levels (descending order)

    Notes
    -----
    A ladder put option allows the holder to lock in profits when the
    underlying price falls to certain predetermined levels (rungs).

    At maturity:
    - If no rungs were hit: max(K - S_T, 0) [vanilla put]
    - If rungs were hit: max(K - lowest_rung_hit, K - S_T, 0)

    Example:
    - Strike K = 100
    - Rungs at [90, 80, 70] (descending)
    - If spot falls to 75 during life (hitting rungs 90 and 80):
      * Minimum payoff at maturity = 100 - 80 = 20
      * If S_T = 85 at maturity: payoff = 20 (locked in)
      * If S_T = 65 at maturity: payoff = 35 (greater than locked 20)
    """

    K: float
    T: float
    rungs: jnp.ndarray

    def __post_init__(self):
        self.rungs = jnp.asarray(self.rungs)
        # Ensure rungs are sorted in descending order
        self.rungs = jnp.sort(self.rungs)[::-1]

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Calculate ladder put payoff based on path.

        The payoff is the maximum of:
        1. K minus the lowest rung hit during the path
        2. The terminal payoff (K - S_T)
        3. Zero
        """
        path = jnp.asarray(path)

        # Find the lowest price along the path
        min_price = path.min()

        # Determine the lowest rung hit
        # A rung is hit if min_price <= rung_level
        rungs_hit = min_price <= self.rungs

        # Get the lowest rung hit (or 0 if none hit)
        if jnp.any(rungs_hit):
            lowest_rung = self.rungs[rungs_hit][-1]  # Last true value (lowest due to descending order)
            locked_in_profit = self.K - lowest_rung
        else:
            locked_in_profit = 0.0

        # Terminal payoff
        terminal_price = path[-1]
        terminal_payoff = self.K - terminal_price

        # Payoff is max of locked-in profit and terminal payoff
        return jnp.maximum(locked_in_profit, jnp.maximum(terminal_payoff, 0.0))


@dataclass
class PercentageLadderCall(PathProduct):
    """Percentage ladder call option.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    rung_percentages : jnp.ndarray
        Array of percentage levels above strike (e.g., [0.1, 0.2, 0.3] for 110%, 120%, 130% of K)

    Notes
    -----
    Similar to LadderCall but rungs are defined as percentages of the strike
    rather than absolute levels. This is more intuitive for structuring.

    Example:
    - Strike K = 100
    - Rung percentages = [0.10, 0.20, 0.30] (i.e., rungs at 110, 120, 130)
    """

    K: float
    T: float
    rung_percentages: jnp.ndarray

    def __post_init__(self):
        self.rung_percentages = jnp.asarray(self.rung_percentages)
        self.rung_percentages = jnp.sort(self.rung_percentages)
        # Convert percentages to absolute levels
        self.rungs = self.K * (1.0 + self.rung_percentages)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Calculate percentage ladder call payoff."""
        path = jnp.asarray(path)
        max_price = path.max()

        rungs_hit = max_price >= self.rungs

        if jnp.any(rungs_hit):
            highest_rung = self.rungs[rungs_hit][-1]
            locked_in_profit = highest_rung - self.K
        else:
            locked_in_profit = 0.0

        terminal_price = path[-1]
        terminal_payoff = terminal_price - self.K

        return jnp.maximum(locked_in_profit, jnp.maximum(terminal_payoff, 0.0))


@dataclass
class PercentageLadderPut(PathProduct):
    """Percentage ladder put option.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    rung_percentages : jnp.ndarray
        Array of percentage levels below strike (e.g., [-0.1, -0.2, -0.3] for 90%, 80%, 70% of K)

    Notes
    -----
    Similar to LadderPut but rungs are defined as percentages of the strike.
    Negative percentages indicate levels below strike.

    Example:
    - Strike K = 100
    - Rung percentages = [-0.10, -0.20, -0.30] (i.e., rungs at 90, 80, 70)
    """

    K: float
    T: float
    rung_percentages: jnp.ndarray

    def __post_init__(self):
        self.rung_percentages = jnp.asarray(self.rung_percentages)
        self.rung_percentages = jnp.sort(self.rung_percentages)[::-1]  # Descending
        # Convert percentages to absolute levels
        self.rungs = self.K * (1.0 + self.rung_percentages)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Calculate percentage ladder put payoff."""
        path = jnp.asarray(path)
        min_price = path.min()

        rungs_hit = min_price <= self.rungs

        if jnp.any(rungs_hit):
            lowest_rung = self.rungs[rungs_hit][-1]
            locked_in_profit = self.K - lowest_rung
        else:
            locked_in_profit = 0.0

        terminal_price = path[-1]
        terminal_payoff = self.K - terminal_price

        return jnp.maximum(locked_in_profit, jnp.maximum(terminal_payoff, 0.0))


__all__ = [
    "LadderCall",
    "LadderPut",
    "PercentageLadderCall",
    "PercentageLadderPut",
]
