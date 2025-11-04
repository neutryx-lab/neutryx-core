"""Advanced interest rate products.

This module implements sophisticated interest rate derivatives:
- Bermudan swaptions (with LSM and grid methods)
- Callable/putable bonds
- CMS spread range accruals
- Constant maturity swaps (CMS)
- Yield curve options
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
from jax import lax

from neutryx.products.base import Product, PathProduct

Array = jnp.ndarray


@dataclass
class BermudanSwaption(PathProduct):
    """Bermudan swaption - swaption with multiple exercise dates.

    Can be priced using Longstaff-Schwartz Monte Carlo or grid-based methods.

    Args:
        T: Final maturity (years)
        K: Strike rate (fixed rate of underlying swap)
        notional: Notional amount
        exercise_dates: Array of exercise dates
        option_type: 'payer' (right to pay fixed) or 'receiver'
        tenor: Tenor of underlying swap (years)
        payment_freq: Payment frequency per year
    """

    T: float
    K: float
    notional: float
    exercise_dates: Array
    option_type: Literal['payer', 'receiver'] = 'payer'
    tenor: float = 10.0
    payment_freq: int = 2

    def intrinsic_value(self, swap_rate: float, time: float) -> float:
        """Calculate intrinsic value of swaption at given time.

        Args:
            swap_rate: Current swap rate
            time: Current time

        Returns:
            Intrinsic value
        """
        remaining_tenor = self.tenor - (self.T - time)
        if remaining_tenor <= 0:
            return 0.0

        # Annuity calculation (simplified)
        num_payments = int(remaining_tenor * self.payment_freq)
        dt = 1.0 / self.payment_freq
        annuity = num_payments * dt  # Simplified, should use discount factors

        if self.option_type == 'payer':
            intrinsic = jnp.maximum(swap_rate - self.K, 0.0) * annuity
        else:
            intrinsic = jnp.maximum(self.K - swap_rate, 0.0) * annuity

        return self.notional * intrinsic

    def payoff_path(self, path: Array) -> Array:
        """Calculate payoff using Longstaff-Schwartz method.

        Args:
            path: Array of shape (num_steps,) containing swap rates at each time

        Returns:
            Optimal exercise value
        """
        # This is a simplified version - full LSM requires regression
        # across multiple paths
        if path.ndim != 1:
            return 0.0

        # Find exercise values at each exercise date
        exercise_values = jnp.array([
            self.intrinsic_value(path[int(t * len(path) / self.T)], t)
            if t < self.T else 0.0
            for t in self.exercise_dates
        ])

        # Simple strategy: exercise at first positive intrinsic value
        # (Full LSM would use regression to estimate continuation value)
        return jnp.max(exercise_values)

    def price_lsm(self, paths: Array, discount_factors: Array,
                  basis_functions: Optional[Callable] = None) -> float:
        """Price using Longstaff-Schwartz Monte Carlo.

        Args:
            paths: Array of shape (num_paths, num_steps) with swap rate paths
            discount_factors: Discount factors for each time step
            basis_functions: Callable for regression basis (default: polynomials)

        Returns:
            Bermudan swaption price
        """
        num_paths, num_steps = paths.shape

        # Default basis functions: 1, x, x^2, x^3
        if basis_functions is None:
            def basis_functions(x):
                return jnp.array([jnp.ones_like(x), x, x**2, x**3]).T

        # Initialize continuation values
        num_exercises = len(self.exercise_dates)
        exercise_indices = (self.exercise_dates / self.T * (num_steps - 1)).astype(int)

        # Start from the last exercise date
        cashflows = jnp.zeros(num_paths)

        # Backward induction
        for i in range(num_exercises - 1, -1, -1):
            ex_idx = exercise_indices[i]
            ex_time = self.exercise_dates[i]

            # Current swap rates
            current_rates = paths[:, ex_idx]

            # Intrinsic values
            intrinsic = jax.vmap(lambda r: self.intrinsic_value(r, ex_time))(current_rates)

            if i == num_exercises - 1:
                # At last exercise, just take intrinsic
                cashflows = intrinsic
            else:
                # Regression to estimate continuation value
                # Only regress on in-the-money paths
                itm_mask = intrinsic > 0

                if jnp.sum(itm_mask) > 0:
                    # Basis functions of current rate
                    X = basis_functions(current_rates[itm_mask])
                    Y = cashflows[itm_mask] * discount_factors[ex_idx]

                    # Least squares regression
                    coeffs = jnp.linalg.lstsq(X, Y)[0]
                    continuation_value = jnp.zeros_like(cashflows)
                    continuation_value = continuation_value.at[itm_mask].set(
                        basis_functions(current_rates[itm_mask]) @ coeffs
                    )

                    # Exercise if intrinsic > continuation
                    exercise_now = intrinsic > continuation_value
                    cashflows = jnp.where(exercise_now, intrinsic, cashflows)

        # Discount and average
        return jnp.mean(cashflows * discount_factors[0])


@dataclass
class CallablePutableBond(PathProduct):
    """Callable or putable bond with embedded options.

    Args:
        T: Maturity (years)
        face_value: Face value
        coupon_rate: Annual coupon rate
        call_dates: Dates at which issuer can call (if callable)
        put_dates: Dates at which holder can put (if putable)
        call_prices: Call prices at each call date
        put_prices: Put prices at each put date
        payment_freq: Coupon payment frequency per year
    """

    T: float
    face_value: float
    coupon_rate: float
    call_dates: Optional[Array] = None
    put_dates: Optional[Array] = None
    call_prices: Optional[Array] = None
    put_prices: Optional[Array] = None
    payment_freq: int = 2

    def payoff_path(self, path: Array) -> Array:
        """Calculate bond value along interest rate path.

        Args:
            path: Array of short rates

        Returns:
            Bond value considering optimal call/put strategy
        """
        # Simplified backward induction
        # In practice, would use full lattice or Monte Carlo with LSM

        # Final payoff at maturity
        value = self.face_value

        # Work backwards, checking call/put options
        if self.call_dates is not None and self.call_prices is not None:
            # Issuer will call if bond value > call price
            for i, (call_date, call_price) in enumerate(
                zip(self.call_dates, self.call_prices)
            ):
                # Simple check: if we're past call date, bond worth call price
                value = jnp.minimum(value, call_price)

        if self.put_dates is not None and self.put_prices is not None:
            # Holder will put if bond value < put price
            for i, (put_date, put_price) in enumerate(
                zip(self.put_dates, self.put_prices)
            ):
                value = jnp.maximum(value, put_price)

        # Add coupon payments
        coupon = self.coupon_rate * self.face_value / self.payment_freq
        num_payments = int(self.T * self.payment_freq)
        total_coupons = coupon * num_payments

        return value + total_coupons


@dataclass
class CMSSpreadRangeAccrual(PathProduct):
    """CMS spread range accrual note.

    Accrues interest only when the spread between two CMS rates is within a range.

    Args:
        T: Maturity (years)
        notional: Notional amount
        base_rate: Base interest rate
        spread_lower: Lower bound of accrual range
        spread_upper: Upper bound of accrual range
        cms_tenor_1: Tenor of first CMS rate (years)
        cms_tenor_2: Tenor of second CMS rate (years)
        observation_freq: Number of observations per year
    """

    T: float
    notional: float
    base_rate: float
    spread_lower: float
    spread_upper: float
    cms_tenor_1: float = 10.0
    cms_tenor_2: float = 2.0
    observation_freq: int = 252

    def payoff_path(self, path: Array) -> Array:
        """Calculate payoff based on CMS spread path.

        Args:
            path: Array of shape (2, num_steps) with [CMS_10Y, CMS_2Y]

        Returns:
            Accrued interest
        """
        if path.ndim != 2:
            return 0.0

        cms_long = path[0, :]
        cms_short = path[1, :]

        # CMS spread
        spread = cms_long - cms_short

        # Check if spread is in range
        in_range = (spread >= self.spread_lower) & (spread <= self.spread_upper)

        # Accrual fraction
        accrual_fraction = jnp.mean(in_range)

        # Total interest
        return self.notional * self.base_rate * self.T * accrual_fraction


@dataclass
class ConstantMaturitySwap(Product):
    """Constant maturity swap (CMS).

    Swap where one leg pays a constant maturity swap rate instead of LIBOR.

    Args:
        T: Swap maturity (years)
        notional: Notional amount
        cms_tenor: Tenor of CMS rate (e.g., 10Y)
        fixed_rate: Fixed rate on other leg (if applicable)
        payment_freq: Payment frequency per year
        is_fixed_vs_cms: If True, fixed vs CMS; if False, CMS vs LIBOR
    """

    T: float
    notional: float
    cms_tenor: float = 10.0
    fixed_rate: float = 0.03
    payment_freq: int = 2
    is_fixed_vs_cms: bool = True

    def payoff_terminal(self, cms_rate: Array) -> Array:
        """Calculate swap value given CMS rate.

        Args:
            cms_rate: Terminal CMS rate

        Returns:
            Swap value
        """
        # Simplified - should integrate over payment dates
        dt = 1.0 / self.payment_freq
        num_payments = int(self.T * self.payment_freq)

        if self.is_fixed_vs_cms:
            # Fixed vs CMS: pay fixed, receive CMS
            value = (cms_rate - self.fixed_rate) * num_payments * dt
        else:
            # CMS vs LIBOR: would need LIBOR rate as well
            value = cms_rate * num_payments * dt

        return self.notional * value

    def convexity_adjustment(self, forward_rate: float, volatility: float,
                            time_to_expiry: float) -> float:
        """Calculate CMS convexity adjustment.

        Args:
            forward_rate: Forward swap rate
            volatility: Swap rate volatility
            time_to_expiry: Time to payment

        Returns:
            Convexity adjustment
        """
        # Simplified convexity adjustment formula
        # Full formula would depend on swap annuity duration

        # Annuity duration approximation
        duration = (1.0 - (1.0 + forward_rate)**(-self.cms_tenor)) / forward_rate

        # Convexity adjustment
        adjustment = 0.5 * volatility**2 * time_to_expiry * duration * forward_rate

        return adjustment


@dataclass
class YieldCurveOption(Product):
    """Option on yield curve shape.

    Options on curve steepness, level, or curvature.

    Args:
        T: Option maturity (years)
        notional: Notional amount
        option_type: 'steepener', 'flattener', 'level', 'butterfly'
        K: Strike level
        tenors: Array of relevant tenors for the option
    """

    T: float
    notional: float
    option_type: Literal['steepener', 'flattener', 'level', 'butterfly'] = 'steepener'
    K: float = 0.0
    tenors: Array = None  # Will be set in __post_init__

    def __post_init__(self):
        if self.tenors is None:
            object.__setattr__(self, 'tenors', jnp.array([2.0, 10.0]))  # Default: 2Y-10Y spread

    def payoff_terminal(self, rates: Array) -> Array:
        """Calculate payoff given terminal yield curve.

        Args:
            rates: Array of rates at specified tenors

        Returns:
            Payoff based on curve shape
        """
        if self.option_type == 'steepener':
            # Pay if curve steepens (long - short > K)
            spread = rates[-1] - rates[0]
            payoff = jnp.maximum(spread - self.K, 0.0)

        elif self.option_type == 'flattener':
            # Pay if curve flattens (long - short < K)
            spread = rates[-1] - rates[0]
            payoff = jnp.maximum(self.K - spread, 0.0)

        elif self.option_type == 'level':
            # Pay based on average level
            avg_level = jnp.mean(rates)
            payoff = jnp.maximum(avg_level - self.K, 0.0)

        elif self.option_type == 'butterfly':
            # Pay based on curvature (belly - wings)
            if len(rates) >= 3:
                curvature = 2 * rates[1] - rates[0] - rates[2]
                payoff = jnp.maximum(curvature - self.K, 0.0)
            else:
                payoff = 0.0

        else:
            payoff = 0.0

        return self.notional * payoff


@dataclass
class RangeAccrualSwap(PathProduct):
    """Range accrual swap.

    Interest rate swap where floating leg accrues only when reference rate
    is within specified range.

    Args:
        T: Maturity (years)
        notional: Notional amount
        fixed_rate: Fixed rate
        range_lower: Lower bound of accrual range
        range_upper: Upper bound of accrual range
        payment_freq: Payment frequency per year
    """

    T: float
    notional: float
    fixed_rate: float
    range_lower: float
    range_upper: float
    payment_freq: int = 4

    def payoff_path(self, path: Array) -> Array:
        """Calculate swap value based on rate path.

        Args:
            path: Array of reference rates over time

        Returns:
            Net swap value
        """
        # Check which observations are in range
        in_range = (path >= self.range_lower) & (path <= self.range_upper)

        # Accrual fraction
        accrual_fraction = jnp.mean(in_range)

        # Floating leg (accrued)
        avg_rate = jnp.mean(path)
        floating_leg = avg_rate * self.T * accrual_fraction

        # Fixed leg
        fixed_leg = self.fixed_rate * self.T

        # Net value
        return self.notional * (floating_leg - fixed_leg)


@dataclass
class TargetRedemptionNote(PathProduct):
    """Target redemption note (TARN).

    Note that automatically redeems when cumulative coupon reaches target.

    Args:
        T: Maximum maturity (years)
        notional: Notional amount
        target_coupon: Target cumulative coupon
        coupon_rate: Base coupon rate
        payment_freq: Payment frequency per year
    """

    T: float
    notional: float
    target_coupon: float
    coupon_rate: float
    payment_freq: int = 4

    def payoff_path(self, path: Array) -> Array:
        """Calculate payoff considering early redemption.

        Args:
            path: Array of reference rates for coupon calculation

        Returns:
            Total payout including coupons and principal
        """
        dt = 1.0 / self.payment_freq
        num_periods = int(self.T * self.payment_freq)

        cumulative_coupon = 0.0
        total_payment = 0.0

        # Simulate coupon payments
        for i in range(min(num_periods, len(path))):
            coupon = self.coupon_rate * self.notional * dt
            cumulative_coupon += coupon
            total_payment += coupon

            # Check if target reached
            if cumulative_coupon >= self.target_coupon:
                # Early redemption
                total_payment += self.notional
                return total_payment

        # If target not reached, pay at maturity
        total_payment += self.notional
        return total_payment


@dataclass
class SnowballNote(PathProduct):
    """Snowball note with memory coupon feature.

    A snowball note pays coupons that accumulate (snowball) if not paid.
    Typical structure:
    - Each period, coupon = Base Rate + Memory (unpaid coupons from past)
    - If reference rate stays within range, coupon is paid
    - If outside range, coupon is not paid and added to memory
    - Often includes knock-out barrier

    Args:
        T: Maturity (years)
        notional: Notional amount
        base_coupon_rate: Base coupon rate per period
        range_lower: Lower bound of accrual range
        range_upper: Upper bound of accrual range
        payment_freq: Payment frequency per year
        knock_out_barrier: Optional knock-out barrier level
        memory_cap: Optional cap on accumulated memory
    """

    T: float
    notional: float
    base_coupon_rate: float
    range_lower: float
    range_upper: float
    payment_freq: int = 4
    knock_out_barrier: Optional[float] = None
    memory_cap: Optional[float] = None

    def payoff_path(self, path: Array) -> Array:
        """Calculate snowball payoff based on rate path.

        Args:
            path: Array of reference rates over time

        Returns:
            Total value including all coupon payments
        """
        dt = 1.0 / self.payment_freq
        n_periods = int(self.T * self.payment_freq)

        total_payment = 0.0
        memory_coupon = 0.0
        knocked_out = False

        # Simulate each coupon period
        for i in range(min(n_periods, len(path))):
            current_rate = path[i]

            # Check knock-out
            if self.knock_out_barrier is not None:
                if current_rate >= self.knock_out_barrier:
                    knocked_out = True
                    # Early redemption at par plus accrued
                    total_payment += self.notional
                    return total_payment

            # Check if in range
            in_range = (current_rate >= self.range_lower) and (current_rate <= self.range_upper)

            if in_range:
                # Pay base coupon + memory
                period_coupon = self.base_coupon_rate + memory_coupon
                total_payment += self.notional * period_coupon * dt
                memory_coupon = 0.0  # Reset memory
            else:
                # Accumulate to memory
                memory_coupon += self.base_coupon_rate

                # Apply memory cap if specified
                if self.memory_cap is not None:
                    memory_coupon = float(jnp.minimum(memory_coupon, self.memory_cap))

        # At maturity, return notional
        if not knocked_out:
            total_payment += self.notional

        return total_payment


@dataclass
class AutocallableNote(PathProduct):
    """Autocallable note (also known as auto-redemption note).

    Automatically redeems (calls) if reference rate exceeds barrier on observation dates.

    Args:
        T: Maximum maturity (years)
        notional: Notional amount
        call_barrier: Barrier level for autocall
        coupon_rate: Coupon rate per period
        call_dates: Array of autocall observation dates
        call_prices: Array of call prices (as % of notional)
        memory_coupon: Whether coupons accumulate if not paid
    """

    T: float
    notional: float
    call_barrier: float
    coupon_rate: float
    call_dates: Array
    call_prices: Optional[Array] = None
    memory_coupon: bool = True

    def __post_init__(self):
        if self.call_prices is None:
            # Default: call at par plus accrued coupon
            n_calls = len(self.call_dates)
            object.__setattr__(
                self, 'call_prices', jnp.ones(n_calls) * 1.0  # 100% of notional
            )

    def payoff_path(self, path: Array) -> Array:
        """Calculate autocallable payoff.

        Args:
            path: Array of reference rates over time

        Returns:
            Total value including redemption and coupons
        """
        n_steps = len(path)
        dt = self.T / n_steps

        accumulated_coupon = 0.0
        total_payment = 0.0

        # Check each call date
        for i, call_date in enumerate(self.call_dates):
            if call_date >= self.T:
                continue

            # Find corresponding path index
            idx = int(call_date / dt)
            if idx >= n_steps:
                continue

            current_rate = path[idx]

            # Check if above barrier
            if current_rate >= self.call_barrier:
                # Autocall triggered
                redemption = self.notional * self.call_prices[i]

                # Pay accumulated coupon if memory feature
                if self.memory_coupon:
                    n_periods = i + 1
                    coupon_payment = self.notional * self.coupon_rate * n_periods
                else:
                    coupon_payment = self.notional * self.coupon_rate

                total_payment = redemption + coupon_payment
                return total_payment
            else:
                # No call, accumulate coupon
                if self.memory_coupon:
                    accumulated_coupon += self.coupon_rate

        # No autocall occurred, pay at maturity
        final_coupon = (
            accumulated_coupon * self.notional
            if self.memory_coupon
            else self.coupon_rate * self.notional * len(self.call_dates)
        )
        total_payment = self.notional + final_coupon

        return total_payment


@dataclass
class RatchetCapFloor(PathProduct):
    """Ratchet cap/floor with dynamic strike.

    The strike adjusts (ratchets) based on previous fixings.

    Args:
        T: Maturity (years)
        notional: Notional amount
        initial_strike: Initial strike rate
        ratchet_rate: Rate at which strike adjusts (e.g., 0.01 = 1%)
        is_cap: True for cap, False for floor
        payment_freq: Payment frequency per year
        global_floor: Optional global floor on strike
        global_cap: Optional global cap on strike
    """

    T: float
    notional: float
    initial_strike: float
    ratchet_rate: float
    is_cap: bool = True
    payment_freq: int = 4
    global_floor: Optional[float] = None
    global_cap: Optional[float] = None

    def payoff_path(self, path: Array) -> Array:
        """Calculate ratchet cap/floor payoff.

        Args:
            path: Array of reference rates over time

        Returns:
            Total value of all caplet/floorlet payments
        """
        dt = 1.0 / self.payment_freq
        n_periods = int(self.T * self.payment_freq)

        total_payment = 0.0
        current_strike = self.initial_strike

        for i in range(min(n_periods, len(path))):
            current_rate = path[i]

            # Calculate payoff
            if self.is_cap:
                payoff = jnp.maximum(current_rate - current_strike, 0.0)
            else:
                payoff = jnp.maximum(current_strike - current_rate, 0.0)

            total_payment += self.notional * payoff * dt

            # Ratchet the strike
            current_strike = current_rate + self.ratchet_rate

            # Apply global bounds
            if self.global_floor is not None:
                current_strike = jnp.maximum(current_strike, self.global_floor)
            if self.global_cap is not None:
                current_strike = jnp.minimum(current_strike, self.global_cap)

        return total_payment
