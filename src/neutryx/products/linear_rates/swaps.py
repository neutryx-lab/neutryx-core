"""Interest rate swap products with multi-curve framework.

Implements:
- Interest Rate Swaps (IRS): Fixed-floating vanilla swaps with multi-curve discounting
- Overnight Index Swaps (OIS): Swaps based on overnight rates (SOFR, ESTR, SONIA)
- Cross-Currency Swaps (CCS): Swaps with FX reset and currency exchange
- Basis Swaps: Tenor basis (3M vs 6M) and currency basis swaps
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit, vmap

from ..base import Product

Array = jnp.ndarray


class SwapType(Enum):
    """Type of swap position."""

    PAYER = "payer"  # Pay fixed, receive floating
    RECEIVER = "receiver"  # Receive fixed, pay floating


class DayCount(Enum):
    """Day count conventions."""

    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"


class Tenor(Enum):
    """Standard tenors for rate fixing."""

    OVERNIGHT = "ON"
    ONE_MONTH = "1M"
    THREE_MONTH = "3M"
    SIX_MONTH = "6M"
    TWELVE_MONTH = "12M"


@dataclass
class InterestRateSwap(Product):
    """Interest rate swap with multi-curve framework.

    A vanilla fixed-floating swap where one party pays a fixed rate
    and receives a floating rate (typically LIBOR/SOFR). Supports
    multi-curve discounting where the forward curve differs from
    the discount curve.

    Attributes:
        T: Maturity in years
        notional: Notional principal amount
        fixed_rate: Fixed rate (annualized)
        swap_type: PAYER (pay fixed) or RECEIVER (receive fixed)
        payment_frequency: Payments per year (e.g., 2 for semiannual)
        day_count: Day count convention
        spread: Spread on floating leg (in bps)
        discount_curve_rates: Discount curve rates (for PV calculation)
        forward_curve_rates: Forward curve rates (for floating rate projection)
    """

    notional: float
    fixed_rate: float
    swap_type: SwapType = SwapType.PAYER
    payment_frequency: int = 2  # Semiannual
    day_count: DayCount = DayCount.ACT_360
    spread: float = 0.0  # Spread in decimal (e.g., 0.001 for 10bps)
    discount_curve_rates: Array | None = None
    forward_curve_rates: Array | None = None

    def __post_init__(self):
        """Initialize curves if not provided."""
        if self.discount_curve_rates is None:
            # Default to flat curve at fixed rate
            n_payments = int(self.T * self.payment_frequency)
            self.discount_curve_rates = jnp.full(n_payments, self.fixed_rate)
        if self.forward_curve_rates is None:
            # Default to same as discount curve
            self.forward_curve_rates = self.discount_curve_rates

    @property
    def requires_path(self) -> bool:
        """IRS pricing does not require full path simulation."""
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate swap payoff (present value).

        For an IRS, the terminal payoff is the present value of all
        fixed and floating cash flows, discounted to present.

        Args:
            spot: Current spot rate (not used for IRS, uses curves)

        Returns:
            Present value of the swap
        """
        return self._calculate_pv()

    def _calculate_pv(self) -> float:
        """Calculate present value of the swap."""
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency

        # Generate payment times
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        # Calculate discount factors
        discount_factors = jnp.exp(-self.discount_curve_rates[:n_payments] * payment_times)

        # Fixed leg PV
        fixed_cash_flows = self.notional * self.fixed_rate * period_length
        fixed_leg_pv = jnp.sum(fixed_cash_flows * discount_factors)

        # Floating leg PV (using forward curve for rate projection)
        floating_rates = self.forward_curve_rates[:n_payments] + self.spread
        floating_cash_flows = self.notional * floating_rates * period_length
        floating_leg_pv = jnp.sum(floating_cash_flows * discount_factors)

        # Net value depends on swap type
        if self.swap_type == SwapType.PAYER:
            # Pay fixed, receive floating
            return floating_leg_pv - fixed_leg_pv
        else:
            # Receive fixed, pay floating
            return fixed_leg_pv - floating_leg_pv

    def par_rate(self) -> float:
        """Calculate the par swap rate (break-even fixed rate).

        Returns:
            The fixed rate that makes the swap have zero value
        """
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        discount_factors = jnp.exp(-self.discount_curve_rates[:n_payments] * payment_times)

        # Floating leg PV
        floating_rates = self.forward_curve_rates[:n_payments] + self.spread
        floating_cash_flows = self.notional * floating_rates * period_length
        floating_leg_pv = jnp.sum(floating_cash_flows * discount_factors)

        # Annuity factor
        annuity = jnp.sum(discount_factors) * period_length * self.notional

        return float(floating_leg_pv / annuity)


@dataclass
class OvernightIndexSwap(Product):
    """Overnight Index Swap (OIS).

    A swap where the floating leg is based on daily compounding of
    an overnight rate (e.g., SOFR, ESTR, SONIA). OIS is the market
    standard for risk-free rate discounting.

    Attributes:
        T: Maturity in years
        notional: Notional principal
        fixed_rate: Fixed rate (annualized)
        swap_type: PAYER or RECEIVER
        overnight_rates: Array of daily overnight rates
        payment_frequency: Payments per year
    """

    notional: float
    fixed_rate: float
    swap_type: SwapType = SwapType.PAYER
    overnight_rates: Array | None = None
    payment_frequency: int = 1  # Annual for OIS typically
    day_count: DayCount = DayCount.ACT_360

    def __post_init__(self):
        """Initialize overnight rates if not provided."""
        if self.overnight_rates is None:
            # Default to flat overnight rate equal to fixed rate
            n_days = int(self.T * 360)  # Approximate trading days
            self.overnight_rates = jnp.full(n_days, self.fixed_rate)

    @property
    def requires_path(self) -> bool:
        """OIS requires daily rate path for compounding."""
        return True

    def payoff_path(self, path: Array) -> Array:
        """Calculate OIS payoff with daily compounding.

        Args:
            path: Array of daily overnight rates

        Returns:
            Present value of the OIS
        """
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency

        # Calculate compound interest for each payment period
        days_per_period = int(360 * period_length)  # ACT/360 convention
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        # Discount factors
        discount_factors = jnp.exp(-self.fixed_rate * payment_times)

        # Fixed leg PV
        fixed_cash_flows = self.notional * self.fixed_rate * period_length
        fixed_leg_pv = jnp.sum(fixed_cash_flows * discount_factors)

        # Floating leg: compound overnight rates
        floating_leg_pv = 0.0
        for i in range(n_payments):
            start_idx = i * days_per_period
            end_idx = min((i + 1) * days_per_period, len(path))
            period_rates = path[start_idx:end_idx]

            # Daily compounding: (1 + r1/360) * (1 + r2/360) * ... - 1
            compound_factor = jnp.prod(1.0 + period_rates / 360.0)
            floating_rate = (compound_factor - 1.0) / period_length
            floating_cash_flow = self.notional * floating_rate
            floating_leg_pv += floating_cash_flow * discount_factors[i]

        # Net value
        if self.swap_type == SwapType.PAYER:
            return floating_leg_pv - fixed_leg_pv
        else:
            return fixed_leg_pv - floating_leg_pv

    def payoff_terminal(self, spot: Array) -> Array:
        """Simplified OIS payoff using spot rate as constant overnight rate.

        Note: This is a simplified approximation. For accurate OIS pricing with
        daily compounding, use `payoff_path` method with actual daily overnight rates.

        Args:
            spot: Spot overnight rate to use as constant rate

        Returns:
            Approximate present value of the OIS
        """
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        # Discount factors
        discount_factors = jnp.exp(-spot * payment_times)

        # Fixed leg PV
        fixed_cash_flows = self.notional * self.fixed_rate * period_length
        fixed_leg_pv = jnp.sum(fixed_cash_flows * discount_factors)

        # Floating leg: approximate with constant overnight rate (spot)
        # Daily compounding over each period: (1 + r/360)^days - 1
        days_per_period = int(360 * period_length)
        compound_factor = jnp.power(1.0 + spot / 360.0, days_per_period)
        floating_rate = (compound_factor - 1.0) / period_length
        floating_cash_flows = self.notional * floating_rate * period_length
        floating_leg_pv = jnp.sum(floating_cash_flows * discount_factors)

        # Net value
        if self.swap_type == SwapType.PAYER:
            return floating_leg_pv - fixed_leg_pv
        else:
            return fixed_leg_pv - floating_leg_pv


@dataclass
class CrossCurrencySwap(Product):
    """Cross-Currency Swap with FX reset.

    Swaps cash flows in two different currencies. Typically involves
    exchange of notional at inception and maturity, plus periodic
    interest payments. Supports FX reset where the foreign currency
    notional is reset periodically based on spot FX rate.

    Attributes:
        T: Maturity in years
        notional_domestic: Notional in domestic currency
        notional_foreign: Notional in foreign currency
        domestic_rate: Fixed/floating rate in domestic currency
        foreign_rate: Fixed/floating rate in foreign currency
        fx_spot: Current FX rate (domestic per foreign)
        fx_reset: Whether to reset FX at each payment
        payment_frequency: Payments per year
        discount_curve_domestic: Discount curve for domestic currency
        discount_curve_foreign: Discount curve for foreign currency
    """

    notional_domestic: float
    notional_foreign: float
    domestic_rate: float
    foreign_rate: float
    fx_spot: float  # Exchange rate (domestic/foreign)
    fx_reset: bool = True
    payment_frequency: int = 2
    discount_curve_domestic: Array | None = None
    discount_curve_foreign: Array | None = None

    def __post_init__(self):
        """Initialize discount curves."""
        n_payments = int(self.T * self.payment_frequency)
        if self.discount_curve_domestic is None:
            self.discount_curve_domestic = jnp.full(n_payments, self.domestic_rate)
        if self.discount_curve_foreign is None:
            self.discount_curve_foreign = jnp.full(n_payments, self.foreign_rate)

    @property
    def requires_path(self) -> bool:
        """CCS with FX reset requires FX path."""
        return self.fx_reset

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate CCS payoff without FX reset.

        Args:
            spot: Terminal FX rate

        Returns:
            Present value of CCS in domestic currency
        """
        if self.fx_reset:
            raise NotImplementedError("FX reset requires path")

        return self._calculate_pv_no_reset(spot)

    def payoff_path(self, path: Array) -> Array:
        """Calculate CCS payoff with FX reset.

        Args:
            path: Array of FX rates over time

        Returns:
            Present value in domestic currency
        """
        if not self.fx_reset:
            return self._calculate_pv_no_reset(path[-1])

        return self._calculate_pv_with_reset(path)

    def _calculate_pv_no_reset(self, fx_terminal: float) -> float:
        """PV calculation without FX reset."""
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        # Domestic leg
        discount_factors_domestic = jnp.exp(
            -self.discount_curve_domestic[:n_payments] * payment_times
        )
        domestic_cash_flows = self.notional_domestic * self.domestic_rate * period_length
        domestic_pv = jnp.sum(domestic_cash_flows * discount_factors_domestic)

        # Foreign leg (converted to domestic currency)
        discount_factors_foreign = jnp.exp(
            -self.discount_curve_foreign[:n_payments] * payment_times
        )
        foreign_cash_flows = self.notional_foreign * self.foreign_rate * period_length
        foreign_pv = jnp.sum(foreign_cash_flows * discount_factors_foreign)
        foreign_pv_domestic = foreign_pv * fx_terminal

        # Notional exchange at maturity
        notional_exchange = (
            self.notional_foreign * fx_terminal - self.notional_domestic
        ) * discount_factors_domestic[-1]

        # Net value: receive foreign, pay domestic
        return foreign_pv_domestic + notional_exchange - domestic_pv

    def _calculate_pv_with_reset(self, fx_path: Array) -> float:
        """PV calculation with FX reset at each payment."""
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        # Sample FX path at payment dates
        sample_indices = (payment_times * len(fx_path) / self.T).astype(int) - 1
        sample_indices = jnp.clip(sample_indices, 0, len(fx_path) - 1)
        fx_at_payments = fx_path[sample_indices]

        # Discount factors
        discount_factors_domestic = jnp.exp(
            -self.discount_curve_domestic[:n_payments] * payment_times
        )

        # Domestic leg
        domestic_cash_flows = self.notional_domestic * self.domestic_rate * period_length
        domestic_pv = jnp.sum(domestic_cash_flows * discount_factors_domestic)

        # Foreign leg with FX reset
        # Reset notional = initial_notional * (FX_current / FX_initial)
        reset_notionals = self.notional_foreign * (fx_at_payments / self.fx_spot)
        foreign_cash_flows = reset_notionals * self.foreign_rate * period_length
        foreign_pv_domestic = jnp.sum(
            foreign_cash_flows * fx_at_payments * discount_factors_domestic
        )

        # Final notional exchange with reset
        final_notional = reset_notionals[-1]
        notional_exchange = (
            final_notional * fx_at_payments[-1] - self.notional_domestic
        ) * discount_factors_domestic[-1]

        return foreign_pv_domestic + notional_exchange - domestic_pv


@dataclass
class BasisSwap(Product):
    """Basis swap (tenor basis or currency basis).

    Swaps floating cash flows based on two different reference rates:
    - Tenor basis: Same currency, different tenors (e.g., 3M LIBOR vs 6M LIBOR)
    - Currency basis: Different currencies, both floating (with basis spread)

    Attributes:
        T: Maturity in years
        notional: Notional principal
        tenor_1: First reference rate tenor
        tenor_2: Second reference rate tenor
        basis_spread: Basis spread (added to tenor_2 leg)
        payment_frequency: Payments per year
        forward_curve_1: Forward curve for first tenor
        forward_curve_2: Forward curve for second tenor
        discount_curve: Discount curve for PV calculation
    """

    notional: float
    tenor_1: Tenor = Tenor.THREE_MONTH
    tenor_2: Tenor = Tenor.SIX_MONTH
    basis_spread: float = 0.0  # Spread on second leg
    payment_frequency: int = 4  # Quarterly
    forward_curve_1: Array | None = None
    forward_curve_2: Array | None = None
    discount_curve: Array | None = None

    def __post_init__(self):
        """Initialize curves."""
        n_payments = int(self.T * self.payment_frequency)
        if self.forward_curve_1 is None:
            self.forward_curve_1 = jnp.full(n_payments, 0.03)
        if self.forward_curve_2 is None:
            self.forward_curve_2 = jnp.full(n_payments, 0.03)
        if self.discount_curve is None:
            self.discount_curve = jnp.full(n_payments, 0.03)

    @property
    def requires_path(self) -> bool:
        """Basis swap does not require path."""
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate basis swap payoff.

        Args:
            spot: Not used for basis swap

        Returns:
            Present value of basis swap
        """
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        # Discount factors
        discount_factors = jnp.exp(-self.discount_curve[:n_payments] * payment_times)

        # First leg cash flows
        leg1_cash_flows = self.notional * self.forward_curve_1[:n_payments] * period_length

        # Second leg cash flows (with basis spread)
        leg2_rates = self.forward_curve_2[:n_payments] + self.basis_spread
        leg2_cash_flows = self.notional * leg2_rates * period_length

        # PV of both legs
        leg1_pv = jnp.sum(leg1_cash_flows * discount_factors)
        leg2_pv = jnp.sum(leg2_cash_flows * discount_factors)

        # Net value: receive leg2, pay leg1
        return leg2_pv - leg1_pv

    def par_spread(self) -> float:
        """Calculate par basis spread (break-even spread).

        Returns:
            Basis spread that makes the swap have zero value
        """
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency
        payment_times = jnp.arange(1, n_payments + 1) * period_length

        discount_factors = jnp.exp(-self.discount_curve[:n_payments] * payment_times)

        # Leg 1 PV
        leg1_cash_flows = self.notional * self.forward_curve_1[:n_payments] * period_length
        leg1_pv = jnp.sum(leg1_cash_flows * discount_factors)

        # Leg 2 PV without spread
        leg2_cash_flows = self.notional * self.forward_curve_2[:n_payments] * period_length
        leg2_pv = jnp.sum(leg2_cash_flows * discount_factors)

        # Annuity factor for leg 2
        annuity = jnp.sum(discount_factors) * period_length * self.notional

        # Spread that makes net value zero
        return float((leg1_pv - leg2_pv) / annuity)


__all__ = [
    "SwapType",
    "DayCount",
    "Tenor",
    "InterestRateSwap",
    "OvernightIndexSwap",
    "CrossCurrencySwap",
    "BasisSwap",
]
