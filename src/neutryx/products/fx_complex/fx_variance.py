"""FX Variance Swaps with FX-specific features.

Implements variance swaps on FX rates with:
- Correlation adjustments between FX and vol
- Quanto effects for cross-currency variance
- Corridor and conditional variance structures
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import jit

from ..base import PathProduct

Array = jnp.ndarray


@dataclass
class FXVarianceSwap(PathProduct):
    """FX Variance Swap with correlation and quanto adjustments.

    A variance swap on an FX rate, accounting for the specific dynamics
    of FX markets including correlation between spot and volatility.

    Attributes:
        T: Maturity in years
        notional: Variance notional (in variance points)
        strike: Variance strike (e.g., 0.0144 for 12% vol)
        quanto_adjustment: Quanto adjustment factor for cross-currency
        correlation_adjustment: Adjustment for spot-vol correlation
        observation_frequency: Days per year for observations (252 for daily)
        cap_level: Optional cap on realized variance
        floor_level: Optional floor on realized variance
    """

    notional: float
    strike: float
    quanto_adjustment: float = 0.0
    correlation_adjustment: float = 0.0
    observation_frequency: int = 252
    cap_level: float | None = None
    floor_level: float | None = None

    @property
    def requires_path(self) -> bool:
        """Variance swaps require full path for realized variance calculation."""
        return True

    def payoff_path(self, path: Array) -> Array:
        """Calculate variance swap payoff from FX path.

        Args:
            path: Array of FX spot prices over time

        Returns:
            Payoff = Notional × (Realized Variance - Strike)
        """
        # Calculate realized variance
        realized_var = self._calculate_realized_variance(path)

        # Apply cap/floor if specified
        if self.cap_level is not None:
            realized_var = jnp.minimum(realized_var, self.cap_level)
        if self.floor_level is not None:
            realized_var = jnp.maximum(realized_var, self.floor_level)

        # Apply quanto and correlation adjustments
        adjusted_realized_var = realized_var + self.quanto_adjustment + self.correlation_adjustment

        # Payoff calculation
        payoff = self.notional * (adjusted_realized_var - self.strike)

        return payoff

    def _calculate_realized_variance(self, path: Array) -> float:
        """Calculate realized variance from price path.

        Uses log returns method:
        σ²_realized = (1/N) × Σ[ln(S_i / S_{i-1})]² × annualization
        """
        # Log returns
        log_returns = jnp.log(path[1:] / path[:-1])

        # Sum of squared log returns
        sum_squared_returns = jnp.sum(log_returns**2)

        # Annualized realized variance
        n = len(log_returns)
        variance = (self.observation_frequency / n) * sum_squared_returns

        return variance

    def vega_notional(self) -> float:
        """Convert variance notional to vega notional.

        Vega notional = Variance notional / (2 × √Strike)

        Returns:
            Vega notional in volatility points
        """
        return float(self.notional / (2.0 * jnp.sqrt(self.strike)))

    def payoff_terminal(self, spot: Array) -> Array:
        """Variance swap requires path for calculation."""
        raise NotImplementedError("Variance swap requires full path")


@dataclass
class CorridorVarianceSwap(PathProduct):
    """Corridor Variance Swap - only accrues variance when FX is in range.

    Variance accumulates only when spot is within a specified corridor.
    Popular in FX markets for directional variance bets.

    Attributes:
        T: Maturity in years
        notional: Variance notional
        strike: Variance strike
        lower_barrier: Lower corridor boundary
        upper_barrier: Upper corridor boundary
        observation_frequency: Observations per year
        accrue_inside: If True, accrue when inside corridor; if False, outside
    """

    notional: float
    strike: float
    lower_barrier: float
    upper_barrier: float
    observation_frequency: int = 252
    accrue_inside: bool = True

    @property
    def requires_path(self) -> bool:
        return True

    def payoff_path(self, path: Array) -> Array:
        """Calculate corridor variance swap payoff.

        Args:
            path: FX spot price path

        Returns:
            Payoff based on conditional variance
        """
        # Determine which observations are in the corridor
        if self.accrue_inside:
            in_corridor = (path >= self.lower_barrier) & (path <= self.upper_barrier)
        else:
            in_corridor = (path < self.lower_barrier) | (path > self.upper_barrier)

        # Calculate log returns
        log_returns = jnp.log(path[1:] / path[:-1])

        # Only include returns when in corridor
        # Align corridor mask with returns (use starting point of each period)
        corridor_mask = in_corridor[:-1]

        # Squared returns in corridor
        masked_squared_returns = jnp.where(corridor_mask, log_returns**2, 0.0)

        # Count observations in corridor
        n_in_corridor = jnp.sum(corridor_mask)

        # Conditional variance (annualized)
        # Only divide by observations that were in corridor
        conditional_var = jnp.where(
            n_in_corridor > 0,
            (self.observation_frequency / n_in_corridor) * jnp.sum(masked_squared_returns),
            0.0,
        )

        # Payoff
        payoff = self.notional * (conditional_var - self.strike)

        return payoff

    def payoff_terminal(self, spot: Array) -> Array:
        raise NotImplementedError("Corridor variance swap requires full path")


@dataclass
class ConditionalVarianceSwap(PathProduct):
    """Conditional Variance Swap - variance depends on trigger condition.

    Variance accrues differently depending on market conditions.
    Can have different strikes for different regimes.

    Attributes:
        T: Maturity in years
        notional: Variance notional
        strike_base: Base variance strike
        strike_conditional: Conditional variance strike
        trigger_level: Level that triggers conditional regime
        trigger_type: 'spot' (based on spot level) or 'vol' (based on realized vol)
        observation_frequency: Observations per year
    """

    notional: float
    strike_base: float
    strike_conditional: float
    trigger_level: float
    trigger_type: str = "spot"  # 'spot' or 'vol'
    observation_frequency: int = 252

    @property
    def requires_path(self) -> bool:
        return True

    def payoff_path(self, path: Array) -> Array:
        """Calculate conditional variance swap payoff.

        Args:
            path: FX spot price path

        Returns:
            Payoff based on conditional variance and regime
        """
        # Calculate realized variance
        log_returns = jnp.log(path[1:] / path[:-1])
        sum_squared_returns = jnp.sum(log_returns**2)
        n = len(log_returns)
        realized_var = (self.observation_frequency / n) * sum_squared_returns

        if self.trigger_type == "spot":
            # Check if spot crossed trigger level
            triggered = jnp.any(path >= self.trigger_level)
        else:  # vol trigger
            # Check if realized vol exceeded trigger
            realized_vol = jnp.sqrt(realized_var)
            triggered = realized_vol >= self.trigger_level

        # Use conditional strike if triggered
        effective_strike = jnp.where(triggered, self.strike_conditional, self.strike_base)

        # Payoff
        payoff = self.notional * (realized_var - effective_strike)

        return payoff

    def payoff_terminal(self, spot: Array) -> Array:
        raise NotImplementedError("Conditional variance swap requires full path")


@jit
def fx_variance_fair_strike(
    spot: float,
    forward: float,
    vol_atm: float,
    vol_smile_adjustment: float = 0.0,
    correlation_spot_vol: float = 0.0,
    time_to_maturity: float = 1.0,
) -> float:
    """Calculate fair strike for FX variance swap.

    Uses replicating portfolio of options to determine fair variance strike.

    Args:
        spot: Current spot FX rate
        forward: Forward FX rate to maturity
        vol_atm: At-the-money volatility
        vol_smile_adjustment: Adjustment for volatility smile/skew
        correlation_spot_vol: Correlation between spot and volatility
        time_to_maturity: Time to maturity in years

    Returns:
        Fair variance strike (in variance units)

    Notes:
        Fair variance strike ≈ σ²_ATM + smile adjustment + correlation adjustment
    """
    # Base variance from ATM vol
    base_variance = vol_atm**2

    # Add smile adjustment (contribution from OTM options)
    variance_with_smile = base_variance + vol_smile_adjustment

    # Correlation adjustment for FX (spot-vol correlation effect)
    # Positive correlation increases variance
    correlation_term = 2.0 * correlation_spot_vol * vol_atm * (jnp.log(forward / spot) / time_to_maturity)

    fair_strike = variance_with_smile + correlation_term

    return fair_strike


@jit
def fx_variance_convexity_adjustment(
    variance_strike: float,
    vol_of_vol: float,
    time_to_maturity: float,
) -> float:
    """Calculate convexity adjustment for variance swap pricing.

    Accounts for the difference between variance swap and volatility swap.

    Args:
        variance_strike: Variance strike level
        vol_of_vol: Volatility of volatility
        time_to_maturity: Time to maturity

    Returns:
        Convexity adjustment to add to variance strike

    Notes:
        Adjustment ≈ (1/4) × (vol_of_vol)² × K_var × T
    """
    adjustment = 0.25 * (vol_of_vol**2) * variance_strike * time_to_maturity
    return adjustment


__all__ = [
    "FXVarianceSwap",
    "CorridorVarianceSwap",
    "ConditionalVarianceSwap",
    "fx_variance_fair_strike",
    "fx_variance_convexity_adjustment",
]
