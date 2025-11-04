"""Credit derivatives products.

This module implements credit derivatives including:
- CDX/iTraxx indices
- Synthetic CDO tranche pricing
- n-th to default baskets
- Credit-linked notes
- Loan CDS
- Contingent CDS
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from neutryx.products.base import Product, PathProduct

Array = jnp.ndarray


@dataclass
class CDSIndex(Product):
    """CDX/iTraxx index - weighted basket of single-name CDS.

    Args:
        T: Time to maturity (years)
        notional: Total notional amount
        spread: Index spread (bps)
        recovery_rate: Expected recovery rate on default (typically 0.4)
        coupon_freq: Coupon payment frequency per year (typically 4 for quarterly)
        num_names: Number of reference entities in the index (typically 125)
        weights: Optional weights for each name (defaults to equal weights)
    """

    T: float
    notional: float
    spread: float
    recovery_rate: float = 0.4
    coupon_freq: int = 4
    num_names: int = 125
    weights: Optional[Array] = None

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(self, 'weights', jnp.ones(self.num_names) / self.num_names)

    def payoff_terminal(self, hazard_rates: Array) -> Array:
        """Calculate index value given hazard rates.

        Args:
            hazard_rates: Array of hazard rates for each reference entity

        Returns:
            Present value of the index
        """
        # Survival probabilities
        survival_probs = jnp.exp(-hazard_rates * self.T)

        # Default probabilities
        default_probs = 1.0 - survival_probs

        # Premium leg (value of spread payments)
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Calculate weighted average survival probability at each payment date
        def premium_pv(t):
            surv = jnp.exp(-hazard_rates * t)
            weighted_surv = jnp.sum(self.weights * surv)
            return self.spread * 1e-4 * weighted_surv * dt

        premium_leg = jnp.sum(jax.vmap(premium_pv)(times))

        # Protection leg (value of default payments)
        weighted_default = jnp.sum(self.weights * default_probs)
        protection_leg = (1.0 - self.recovery_rate) * weighted_default

        # Index value = protection leg - premium leg
        return self.notional * (protection_leg - premium_leg)


@dataclass
class SyntheticCDO(Product):
    """Synthetic CDO tranche.

    Prices tranches of a synthetic CDO using the Gaussian copula model.

    Args:
        T: Time to maturity (years)
        notional: Tranche notional
        attachment: Lower bound of tranche (as fraction of portfolio)
        detachment: Upper bound of tranche (as fraction of portfolio)
        spread: Tranche spread (bps)
        recovery_rate: Expected recovery rate
        correlation: Default correlation between entities
        num_names: Number of reference entities
        coupon_freq: Coupon payment frequency per year
    """

    T: float
    notional: float
    attachment: float
    detachment: float
    spread: float
    recovery_rate: float = 0.4
    correlation: float = 0.3
    num_names: int = 125
    coupon_freq: int = 4

    def expected_loss(self, default_prob: float, num_defaults: int) -> float:
        """Calculate expected loss given number of defaults.

        Args:
            default_prob: Unconditional default probability
            num_defaults: Number of defaults

        Returns:
            Expected tranche loss
        """
        loss_per_name = (1.0 - self.recovery_rate) / self.num_names
        portfolio_loss = num_defaults * loss_per_name

        # Tranche loss calculation
        tranche_loss = jnp.maximum(
            0.0,
            jnp.minimum(portfolio_loss - self.attachment,
                       self.detachment - self.attachment)
        ) / (self.detachment - self.attachment)

        return tranche_loss

    def payoff_terminal(self, default_prob: Array) -> Array:
        """Calculate tranche value given default probability.

        Uses large homogeneous pool approximation with Gaussian copula.

        Args:
            default_prob: Default probability for reference entities

        Returns:
            Present value of the tranche
        """
        # Convert default probability to conditional threshold
        K = norm.ppf(default_prob)

        # Calculate conditional default probability for various market factors
        def conditional_loss(market_factor: float) -> float:
            # Conditional default probability given market factor
            rho = self.correlation
            cond_threshold = (K - jnp.sqrt(rho) * market_factor) / jnp.sqrt(1 - rho)
            cond_default_prob = norm.cdf(cond_threshold)

            # Expected number of defaults
            expected_defaults = self.num_names * cond_default_prob

            # Expected tranche loss
            return self.expected_loss(default_prob, expected_defaults)

        # Integrate over market factor distribution (Gaussian quadrature)
        n_points = 50
        x, w = np.polynomial.hermite.hermgauss(n_points)
        x = jnp.array(x) * jnp.sqrt(2.0)  # Scale for N(0,1)
        w = jnp.array(w) / jnp.sqrt(jnp.pi)

        expected_tranche_loss = jnp.sum(
            jax.vmap(lambda xi: conditional_loss(xi))(x) * w
        )

        # Premium leg
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Simplified: assume constant outstanding notional
        premium_leg = self.spread * 1e-4 * jnp.sum(
            (1.0 - expected_tranche_loss) * dt * jnp.ones_like(times)
        )

        # Protection leg
        protection_leg = expected_tranche_loss

        return self.notional * (protection_leg - premium_leg)


@dataclass
class NthToDefaultBasket(Product):
    """n-th to default basket.

    Credit derivative that pays on the n-th default in a basket of reference entities.

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        n: Which default triggers payout (1st, 2nd, 3rd, etc.)
        num_names: Number of reference entities in basket
        recovery_rate: Expected recovery rate
        correlation: Default correlation
    """

    T: float
    notional: float
    n: int
    num_names: int
    recovery_rate: float = 0.4
    correlation: float = 0.3

    def payoff_terminal(self, default_times: Array) -> Array:
        """Calculate payoff given default times.

        Args:
            default_times: Array of default times for each entity

        Returns:
            Payoff if n-th default occurs before maturity
        """
        # Sort default times
        sorted_times = jnp.sort(default_times)

        # Get n-th default time (0-indexed, so n-1)
        nth_default_time = sorted_times[self.n - 1]

        # Pay if n-th default occurs before maturity
        default_occurred = nth_default_time <= self.T

        return jnp.where(
            default_occurred,
            self.notional * (1.0 - self.recovery_rate),
            0.0
        )

    def price_monte_carlo(self, default_probs: Array, num_sims: int = 10000,
                         random_key: Optional[Array] = None) -> float:
        """Price using Monte Carlo simulation with copula.

        Args:
            default_probs: Array of default probabilities for each entity
            num_sims: Number of Monte Carlo simulations
            random_key: JAX random key

        Returns:
            Expected present value
        """
        if random_key is None:
            random_key = jax.random.PRNGKey(0)

        # Generate correlated default times using Gaussian copula
        key1, key2 = jax.random.split(random_key)

        # Common factor
        common_factor = jax.random.normal(key1, (num_sims,))

        # Idiosyncratic factors
        idio_factors = jax.random.normal(key2, (num_sims, self.num_names))

        # Combine with correlation structure
        rho = self.correlation
        latent = jnp.sqrt(rho) * common_factor[:, None] + jnp.sqrt(1 - rho) * idio_factors

        # Convert to uniform and then to default times
        uniform = norm.cdf(latent)
        default_times = -jnp.log(1 - uniform) / (-jnp.log(1 - default_probs) / self.T)

        # Calculate payoffs
        payoffs = jax.vmap(self.payoff_terminal)(default_times)

        return jnp.mean(payoffs)


@dataclass
class CreditLinkedNote(Product):
    """Credit-linked note.

    Bond with embedded credit protection. Pays coupon plus return of principal,
    unless credit event occurs.

    Args:
        T: Time to maturity (years)
        principal: Principal amount
        coupon_rate: Annual coupon rate
        credit_spread: Credit spread (bps)
        recovery_rate: Recovery rate on default
        coupon_freq: Coupon payment frequency per year
    """

    T: float
    principal: float
    coupon_rate: float
    credit_spread: float
    recovery_rate: float = 0.4
    coupon_freq: int = 2

    def payoff_terminal(self, default_indicator: Array) -> Array:
        """Calculate payoff at maturity.

        Args:
            default_indicator: 1 if default occurred, 0 otherwise

        Returns:
            Total payoff including coupons and principal
        """
        # Coupon payments
        coupon_payment = self.coupon_rate * self.principal / self.coupon_freq
        num_payments = int(self.T * self.coupon_freq)
        total_coupons = coupon_payment * num_payments

        # Principal repayment or recovery
        principal_payment = jnp.where(
            default_indicator > 0.5,
            self.principal * self.recovery_rate,
            self.principal
        )

        return total_coupons + principal_payment

    def fair_coupon(self, hazard_rate: float, risk_free_rate: float) -> float:
        """Calculate fair coupon rate given hazard rate.

        Args:
            hazard_rate: Constant hazard rate
            risk_free_rate: Risk-free rate

        Returns:
            Fair coupon rate
        """
        # Survival probability
        survival_prob = jnp.exp(-hazard_rate * self.T)

        # Expected present value of principal
        discount = jnp.exp(-risk_free_rate * self.T)
        expected_principal = discount * (
            survival_prob * self.principal +
            (1 - survival_prob) * self.principal * self.recovery_rate
        )

        # Annuity factor for coupons
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt, self.T + dt, dt)
        annuity = jnp.sum(
            jnp.exp(-risk_free_rate * times) * jnp.exp(-hazard_rate * times)
        )

        # Fair coupon: (Principal - PV(expected principal)) / (annuity * principal)
        fair_rate = (self.principal - expected_principal) / (annuity * self.principal)

        return fair_rate


@dataclass
class LoanCDS(Product):
    """Loan credit default swap.

    CDS on a syndicated loan with specific features like delayed settlement
    and modified restructuring.

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        spread: CDS spread (bps)
        recovery_rate: Expected recovery rate (typically lower for loans)
        coupon_freq: Coupon frequency per year
        settlement_delay: Settlement delay in years after default
    """

    T: float
    notional: float
    spread: float
    recovery_rate: float = 0.6  # Higher recovery for senior secured loans
    coupon_freq: int = 4
    settlement_delay: float = 0.083  # ~1 month

    def payoff_terminal(self, default_time: Array) -> Array:
        """Calculate payoff given default time.

        Args:
            default_time: Time of default (or T+1 if no default)

        Returns:
            Net present value of protection and premium legs
        """
        # Check if default occurred before maturity
        default_occurred = default_time <= self.T

        # Premium leg: spread payments until default or maturity
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Accrued premium
        payment_times = jnp.where(times <= default_time, times, 0.0)
        premium_leg = self.spread * 1e-4 * jnp.sum(payment_times > 0) * dt

        # Protection leg: loss given default
        protection_leg = jnp.where(
            default_occurred,
            (1.0 - self.recovery_rate),
            0.0
        )

        return self.notional * (protection_leg - premium_leg)


@dataclass
class ContingentCDS(Product):
    """Contingent credit default swap.

    CDS that is activated only if a specified trigger event occurs.
    For example, a CCDS that only activates if an equity price falls below a barrier.

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        spread: CDS spread (bps)
        recovery_rate: Recovery rate on default
        trigger_barrier: Barrier level that activates the CDS
        trigger_type: Type of trigger ('down', 'up')
        coupon_freq: Coupon payment frequency per year
    """

    T: float
    notional: float
    spread: float
    recovery_rate: float = 0.4
    trigger_barrier: float = 80.0  # Trigger level (e.g., stock price)
    trigger_type: str = 'down'  # 'down' or 'up'
    coupon_freq: int = 4

    @property
    def requires_path(self) -> bool:
        return True

    def payoff_path(self, path: Array) -> Array:
        """Calculate payoff given price path.

        Args:
            path: Array of [trigger_asset_prices, default_indicator]

        Returns:
            Payoff of contingent CDS
        """
        # Extract trigger asset path and default indicator
        if path.ndim == 2:
            trigger_path = path[0, :]
            default_time_indicator = path[1, -1]
        else:
            # Simple case: single terminal value
            return 0.0

        # Check if trigger condition was hit
        if self.trigger_type == 'down':
            trigger_hit = jnp.any(trigger_path <= self.trigger_barrier)
        else:  # 'up'
            trigger_hit = jnp.any(trigger_path >= self.trigger_barrier)

        if not trigger_hit:
            return 0.0

        # If triggered, calculate standard CDS payoff
        default_occurred = default_time_indicator > 0.5

        # Premium leg
        dt = 1.0 / self.coupon_freq
        num_payments = int(self.T * self.coupon_freq)
        premium_leg = self.spread * 1e-4 * num_payments * dt

        # Protection leg
        protection_leg = jnp.where(
            default_occurred,
            (1.0 - self.recovery_rate),
            0.0
        )

        return self.notional * (protection_leg - premium_leg)
