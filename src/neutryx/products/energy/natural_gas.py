"""Natural gas derivatives - swaps, options, and storage contracts."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import jit

from ..base import Product, PathProduct

Array = jnp.ndarray


@dataclass
class NaturalGasOption(Product):
    """Natural gas futures option.

    Natural gas has high volatility (50%+) and strong seasonality.
    Uses Black-76 model for futures options.

    Attributes:
        T: Time to maturity
        strike: Strike price ($/MMBtu)
        is_call: True for call
        notional: Contract size (MMBtu, typically 10,000)
        volatility: Implied volatility (typically 40-60%)
        risk_free_rate: Risk-free rate
    """

    strike: float
    is_call: bool = True
    notional: float = 10_000.0  # 1 contract = 10,000 MMBtu
    volatility: float = 0.50
    risk_free_rate: float = 0.03

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate gas option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, futures_price: float) -> float:
        """Price using Black-76 model."""
        from .oil import black_76_option

        return float(
            black_76_option(
                F=futures_price,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class GasSwap(Product):
    """Natural gas swap.

    Exchanges floating gas prices for fixed price over delivery period.

    Attributes:
        T: Maturity
        notional: Total volume (MMBtu)
        fixed_price: Fixed price ($/MMBtu)
        payment_frequency: Payments per year
        is_payer: True if paying fixed, receiving floating
    """

    notional: float
    fixed_price: float
    payment_frequency: int = 12  # Monthly
    is_payer: bool = True

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Simplified swap payoff (single settlement)."""
        if self.is_payer:
            # Pay fixed, receive floating
            return self.notional * (spot - self.fixed_price)
        else:
            return self.notional * (self.fixed_price - spot)


@dataclass
class SeasonalGasOption(Product):
    """Seasonal natural gas option.

    Options on gas for specific seasons (winter/summer).
    Captures seasonality premium in gas prices.

    Attributes:
        T: Time to maturity
        strike: Strike price
        is_call: True for call
        season: 'winter' or 'summer'
        notional: Contract size
        volatility: Seasonal volatility (higher in winter)
        risk_free_rate: Risk-free rate
    """

    strike: float
    is_call: bool = True
    season: str = "winter"  # 'winter' or 'summer'
    notional: float = 10_000.0
    volatility: float | None = None  # Will be set based on season
    risk_free_rate: float = 0.03

    def __post_init__(self):
        """Set seasonality-adjusted volatility."""
        if self.volatility is None:
            # Winter gas typically has higher vol
            if self.season == "winter":
                self.volatility = 0.60  # Higher winter volatility
            else:
                self.volatility = 0.40  # Lower summer volatility

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate seasonal option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, futures_price: float) -> float:
        """Price with season-adjusted volatility."""
        from .oil import black_76_option

        return float(
            black_76_option(
                F=futures_price,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class GasStorageContract(PathProduct):
    """Natural gas storage contract (swing/take-or-pay).

    Contract allowing flexible injection/withdrawal from storage
    subject to capacity and rate constraints.

    Attributes:
        T: Contract maturity
        max_storage: Maximum storage capacity (MMBtu)
        max_injection_rate: Max injection rate per period
        max_withdrawal_rate: Max withdrawal rate per period
        injection_cost: Cost per MMBtu injected
        withdrawal_cost: Cost per MMBtu withdrawn
        fixing_times: Times for injection/withdrawal decisions
        current_inventory: Starting inventory level
    """

    max_storage: float
    max_injection_rate: float
    max_withdrawal_rate: float
    injection_cost: float
    withdrawal_cost: float
    fixing_times: Array
    current_inventory: float = 0.0

    def __post_init__(self):
        """Ensure arrays."""
        self.fixing_times = jnp.asarray(self.fixing_times)

    @property
    def requires_path(self) -> bool:
        return True

    def payoff_path(self, path: Array) -> Array:
        """Calculate optimal storage strategy payoff.

        Greedy strategy: inject when prices low, withdraw when high.
        """
        path = jnp.asarray(path)
        n_steps = len(path)

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        fixing_indices = jnp.round(self.fixing_times / dt).astype(int)
        fixing_indices = jnp.clip(fixing_indices, 0, n_steps - 1)

        inventory = self.current_inventory
        total_payoff = 0.0

        for idx in fixing_indices:
            price = path[idx]

            # Simple strategy: inject if price below threshold, withdraw if above
            # Threshold = average of injection and withdrawal costs
            threshold = (self.injection_cost + self.withdrawal_cost) / 2.0

            if price < threshold and inventory < self.max_storage:
                # Inject
                inject_qty = jnp.minimum(
                    self.max_injection_rate, self.max_storage - inventory
                )
                cost = inject_qty * (price + self.injection_cost)
                total_payoff -= cost
                inventory += inject_qty

            elif price > threshold and inventory > 0:
                # Withdraw
                withdraw_qty = jnp.minimum(self.max_withdrawal_rate, inventory)
                revenue = withdraw_qty * (price - self.withdrawal_cost)
                total_payoff += revenue
                inventory -= withdraw_qty

        # Final inventory value (sell at final price)
        final_value = inventory * path[-1]
        total_payoff += final_value

        return total_payoff


__all__ = [
    "NaturalGasOption",
    "GasSwap",
    "SeasonalGasOption",
    "GasStorageContract",
]
