"""Credit derivatives products.

This module implements credit derivatives including:
- Single-name CDS
- CDX/iTraxx indices
- Synthetic CDO/CLO tranche pricing
- Total Return Swaps
- n-th to default baskets
- Credit-linked notes
- Loan CDS
- Contingent CDS

Theoretical foundations:
- Hazard rate models for default probability
- Reduced-form credit models (Jarrow-Turnbull, Duffie-Singleton)
- Gaussian copula models for portfolio credit risk
- Large homogeneous pool (LHP) approximation for CDO pricing
- Credit triangle: PD = 1 - exp(-hazard_rate * T)
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
class CreditDefaultSwap(Product):
    """Single-name Credit Default Swap (CDS).

    A CDS is a credit derivative contract that provides protection against default
    of a reference entity. The protection buyer pays periodic premiums (spread)
    to the protection seller. If a credit event occurs, the seller pays the buyer
    the loss given default (1 - recovery_rate) times the notional.

    The CDS value is the difference between the protection leg (expected loss) and
    the premium leg (present value of spread payments).

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        spread: CDS spread in basis points (e.g., 100 = 1%)
        recovery_rate: Expected recovery rate on default (typically 0.4 for senior unsecured)
        coupon_freq: Number of premium payments per year (typically 4 for quarterly)
        upfront_payment: Upfront payment (positive = paid by protection buyer)

    References:
        - O'Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives.
        - Hull, J., & White, A. (2000). Valuing credit default swaps I & II.
    """

    T: float
    notional: float
    spread: float  # in basis points
    recovery_rate: float = 0.4
    coupon_freq: int = 4
    upfront_payment: float = 0.0

    def survival_probability(self, hazard_rate: float, t: float) -> float:
        """Calculate survival probability to time t.

        Args:
            hazard_rate: Constant hazard rate (intensity of default)
            t: Time horizon

        Returns:
            Probability of no default by time t
        """
        return jnp.exp(-hazard_rate * t)

    def default_probability(self, hazard_rate: float, t: float) -> float:
        """Calculate cumulative default probability by time t.

        Args:
            hazard_rate: Constant hazard rate
            t: Time horizon

        Returns:
            Probability of default by time t
        """
        return 1.0 - self.survival_probability(hazard_rate, t)

    def premium_leg_pv(self, hazard_rate: float, discount_rate: float) -> float:
        """Calculate present value of premium leg.

        The premium leg consists of:
        1. Regular premium payments (spread * notional * dt) at each coupon date
        2. Accrued premium in case of default between payment dates

        Args:
            hazard_rate: Constant hazard rate
            discount_rate: Risk-free discount rate

        Returns:
            Present value of all premium payments
        """
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Regular premium payments (paid if no default)
        def pv_payment(t: float) -> float:
            survival = self.survival_probability(hazard_rate, t)
            discount = jnp.exp(-discount_rate * t)
            return self.spread * 1e-4 * dt * survival * discount

        regular_pv = jnp.sum(jax.vmap(pv_payment)(times))

        # Accrued premium (approximation: assume default occurs mid-period)
        def pv_accrued(t: float) -> float:
            # Default probability in period [t-dt, t]
            surv_start = self.survival_probability(hazard_rate, jnp.maximum(0, t - dt))
            surv_end = self.survival_probability(hazard_rate, t)
            default_in_period = surv_start - surv_end

            # Discount to middle of period
            discount = jnp.exp(-discount_rate * (t - dt/2))

            return self.spread * 1e-4 * (dt/2) * default_in_period * discount

        accrued_pv = jnp.sum(jax.vmap(pv_accrued)(times))

        return self.notional * (regular_pv + accrued_pv)

    def protection_leg_pv(self, hazard_rate: float, discount_rate: float) -> float:
        """Calculate present value of protection leg.

        The protection leg pays (1 - recovery_rate) * notional if default occurs.
        We integrate over all possible default times.

        Args:
            hazard_rate: Constant hazard rate
            discount_rate: Risk-free discount rate

        Returns:
            Present value of protection payment
        """
        # Loss given default
        lgd = 1.0 - self.recovery_rate

        # Approximate integral using discrete time steps
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt/2, self.T + dt/2, dt)

        def pv_protection(t: float) -> float:
            # Probability density of default at time t
            # f(t) = hazard_rate * exp(-hazard_rate * t)
            default_density = hazard_rate * jnp.exp(-hazard_rate * t)

            # Discount factor
            discount = jnp.exp(-discount_rate * t)

            return lgd * default_density * discount * dt

        return self.notional * jnp.sum(jax.vmap(pv_protection)(times))

    def payoff_terminal(self, hazard_rate: Array) -> Array:
        """Calculate CDS value given hazard rate.

        CDS value = Protection Leg PV - Premium Leg PV - Upfront Payment

        Args:
            hazard_rate: Hazard rate of reference entity

        Returns:
            Present value of CDS from protection buyer's perspective
        """
        # Assume risk-neutral measure: discount rate = hazard rate (simplification)
        # In practice, use risk-free rate
        discount_rate = 0.0  # Can be parameterized

        protection = self.protection_leg_pv(hazard_rate, discount_rate)
        premium = self.premium_leg_pv(hazard_rate, discount_rate)

        return protection - premium - self.upfront_payment

    def fair_spread(self, hazard_rate: float, discount_rate: float = 0.0) -> float:
        """Calculate fair spread that makes CDS value zero at inception.

        Fair spread = Protection Leg PV / Premium Leg PV (with spread = 1bp)

        Args:
            hazard_rate: Constant hazard rate
            discount_rate: Risk-free discount rate

        Returns:
            Fair spread in basis points
        """
        # Calculate protection leg
        protection = self.protection_leg_pv(hazard_rate, discount_rate)

        # Calculate premium leg sensitivity (risky PV01)
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt, self.T + dt, dt)

        risky_pv01 = jnp.sum(
            jax.vmap(lambda t: self.survival_probability(hazard_rate, t) *
                     jnp.exp(-discount_rate * t) * dt)(times)
        )

        return (protection / (self.notional * risky_pv01 * 1e-4)) if risky_pv01 > 0 else 0.0

    def credit_dv01(self, hazard_rate: float, discount_rate: float = 0.0) -> float:
        """Calculate credit DV01 (dollar value of 1 basis point spread change).

        Args:
            hazard_rate: Constant hazard rate
            discount_rate: Risk-free discount rate

        Returns:
            Change in CDS value for 1bp increase in spread
        """
        dt = 1.0 / self.coupon_freq
        times = jnp.arange(dt, self.T + dt, dt)

        risky_pv01 = jnp.sum(
            jax.vmap(lambda t: self.survival_probability(hazard_rate, t) *
                     jnp.exp(-discount_rate * t) * dt)(times)
        )

        return self.notional * risky_pv01 * 1e-4


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
class TotalReturnSwap(Product):
    """Total Return Swap (TRS) on a credit asset.

    A TRS allows one party (total return receiver) to receive the total economic
    return (coupons + price appreciation/depreciation) of a reference asset without
    owning it, while paying a floating rate (typically LIBOR + spread) to the payer.

    The TRS is commonly used for:
    - Gaining exposure to credit assets without balance sheet impact
    - Financing positions
    - Regulatory capital arbitrage
    - Hedging credit risk

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        asset_coupon: Coupon rate of reference asset (annual)
        funding_spread: Spread over LIBOR paid by total return receiver (bps)
        initial_asset_price: Initial price of reference asset (as % of par)
        recovery_rate: Recovery rate if asset defaults
        payment_freq: Payment frequency per year

    References:
        - British Bankers' Association (1999). Credit Derivatives Report.
        - Das, S. (2000). Credit Derivatives: Trading & Management of Credit.
    """

    T: float
    notional: float
    asset_coupon: float  # Annual coupon rate (e.g., 0.05 for 5%)
    funding_spread: float  # Spread in basis points
    initial_asset_price: float = 100.0  # Price as % of par
    recovery_rate: float = 0.4
    payment_freq: int = 4

    def total_return_leg(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate total return leg (paid to receiver).

        Total return = Coupons + (Final Price - Initial Price)

        Args:
            final_asset_price: Terminal price of reference asset
            hazard_rate: Hazard rate of reference asset
            libor_rate: LIBOR rate for discounting

        Returns:
            Present value of total return payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Coupon payments (if no default)
        coupon_payment = self.asset_coupon / self.payment_freq

        def pv_coupon(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return coupon_payment * survival * discount

        total_coupons = jnp.sum(jax.vmap(pv_coupon)(times))

        # Price appreciation/depreciation
        survival = jnp.exp(-hazard_rate * self.T)
        default_prob = 1.0 - survival

        # If no default: return final price
        # If default: return recovery value
        expected_final = (
            survival * final_asset_price +
            default_prob * self.recovery_rate * 100.0
        )

        price_return = (expected_final - self.initial_asset_price) / 100.0
        discount = jnp.exp(-libor_rate * self.T)

        return self.notional * (total_coupons + price_return * discount)

    def funding_leg(self, libor_rate: float, hazard_rate: float = 0.0) -> float:
        """Calculate funding leg (paid by receiver).

        Funding = (LIBOR + spread) paid periodically

        Args:
            libor_rate: LIBOR rate
            hazard_rate: Hazard rate (affects payment if counterparty defaults)

        Returns:
            Present value of funding payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        funding_rate = libor_rate + self.funding_spread * 1e-4

        def pv_funding(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return funding_rate * dt * survival * discount

        return self.notional * jnp.sum(jax.vmap(pv_funding)(times))

    def payoff_terminal(self, state: Array) -> Array:
        """Calculate TRS value.

        TRS value = Total Return Leg - Funding Leg

        Args:
            state: Array containing [final_asset_price, hazard_rate, libor_rate]

        Returns:
            Present value of TRS from receiver's perspective
        """
        # Unpack state
        if state.size >= 3:
            final_price = state[0]
            hazard_rate = state[1]
            libor_rate = state[2]
        else:
            # Default values if not enough inputs
            final_price = 100.0
            hazard_rate = state[0] if state.size >= 1 else 0.01
            libor_rate = 0.03

        total_return = self.total_return_leg(final_price, hazard_rate, libor_rate)
        funding = self.funding_leg(libor_rate, hazard_rate)

        return total_return - funding

    def breakeven_spread(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate breakeven funding spread.

        The spread that makes TRS value zero at inception.

        Args:
            final_asset_price: Expected final price of reference asset
            hazard_rate: Hazard rate
            libor_rate: LIBOR rate

        Returns:
            Breakeven spread in basis points
        """
        total_return = self.total_return_leg(final_asset_price, hazard_rate, libor_rate)

        # Calculate funding leg with 1bp spread
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        pv01 = jnp.sum(
            jax.vmap(lambda t: jnp.exp(-hazard_rate * t) *
                     jnp.exp(-libor_rate * t) * dt)(times)
        )

        # Breakeven: total_return = (libor + spread) * pv01
        libor_pv = libor_rate * self.notional * pv01
        spread_in_decimal = (total_return - libor_pv) / (self.notional * pv01)

        return spread_in_decimal * 1e4  # Convert to basis points


@dataclass
class CollateralizedLoanObligation(Product):
    """Collateralized Loan Obligation (CLO) tranche.

    CLOs are structured finance securities backed by a pool of loans, typically
    leveraged loans. The cash flows from the loan pool are divided into tranches
    with different seniority levels.

    Key differences from CDOs:
    - Backed by actual loans (not synthetic)
    - Higher recovery rates (60-70% vs 40% for bonds)
    - Floating rate assets (loans tied to LIBOR/SOFR)
    - Active management of loan portfolio

    Args:
        T: Time to maturity (years)
        notional: Tranche notional
        attachment: Lower bound of tranche (as fraction of portfolio)
        detachment: Upper bound of tranche (as fraction of portfolio)
        spread: Tranche spread over LIBOR (bps)
        recovery_rate: Expected recovery rate for loans (typically 0.6-0.7)
        correlation: Default correlation between loans
        num_loans: Number of loans in portfolio
        payment_freq: Payment frequency per year
        loan_coupon: Average coupon rate of underlying loans (spread over LIBOR)

    References:
        - Tavakoli, J. (2003). Collateralized Debt Obligations and Structured Finance.
        - Goodman, L. & Fabozzi, F. (2002). Collateralized Debt Obligations.
    """

    T: float
    notional: float
    attachment: float
    detachment: float
    spread: float  # Tranche spread in bps
    recovery_rate: float = 0.65  # Higher for senior secured loans
    correlation: float = 0.25
    num_loans: int = 100
    payment_freq: int = 4
    loan_coupon: float = 400.0  # Average loan spread in bps (e.g., L+400)

    def tranche_loss(self, portfolio_loss: float) -> float:
        """Calculate tranche loss given portfolio loss.

        Args:
            portfolio_loss: Total portfolio loss as fraction of notional

        Returns:
            Tranche loss as fraction of tranche size
        """
        # Loss allocated to tranche
        loss_to_tranche = jnp.maximum(
            0.0,
            jnp.minimum(
                portfolio_loss - self.attachment,
                self.detachment - self.attachment
            )
        )

        # Normalize by tranche size
        tranche_size = self.detachment - self.attachment
        return loss_to_tranche / tranche_size if tranche_size > 0 else 0.0

    def expected_tranche_loss_lhp(self, default_prob: float) -> float:
        """Calculate expected tranche loss using Large Homogeneous Pool (LHP).

        Uses Gaussian copula model with numerical integration.

        Args:
            default_prob: Unconditional default probability

        Returns:
            Expected loss for the tranche
        """
        # Convert default probability to threshold
        K = norm.ppf(default_prob)

        # Conditional loss function
        def conditional_loss(market_factor: float) -> float:
            # Conditional default probability given market factor
            rho = self.correlation
            threshold = (K - jnp.sqrt(rho) * market_factor) / jnp.sqrt(1 - rho)
            cond_default_prob = norm.cdf(threshold)

            # Portfolio loss (in LHP: E[loss] = default_prob * LGD)
            lgd = 1.0 - self.recovery_rate
            portfolio_loss = cond_default_prob * lgd

            # Tranche loss
            return self.tranche_loss(portfolio_loss)

        # Numerical integration using Gauss-Hermite quadrature
        n_points = 50
        x, w = np.polynomial.hermite.hermgauss(n_points)
        x = jnp.array(x) * jnp.sqrt(2.0)  # Scale for N(0,1)
        w = jnp.array(w) / jnp.sqrt(jnp.pi)

        expected_loss = jnp.sum(
            jax.vmap(conditional_loss)(x) * w
        )

        return expected_loss

    def interest_waterfall(
        self,
        libor_rate: float,
        default_prob: float,
        tranche_outstanding: float = 1.0
    ) -> float:
        """Calculate interest payments considering waterfall structure.

        In a CLO, interest is paid in order of seniority (waterfall).

        Args:
            libor_rate: LIBOR rate
            default_prob: Portfolio default probability
            tranche_outstanding: Fraction of tranche still outstanding

        Returns:
            Expected interest payment to tranche
        """
        # Interest rate for this tranche
        tranche_rate = libor_rate + self.spread * 1e-4

        # Available interest from loan pool
        pool_rate = libor_rate + self.loan_coupon * 1e-4

        # Survival probability
        survival = 1.0 - default_prob

        # Interest payment (simplified: assumes senior tranches get full payment)
        # More sophisticated model would consider full waterfall
        interest = tranche_rate * tranche_outstanding * survival

        return interest

    def payoff_terminal(self, params: Array) -> Array:
        """Calculate CLO tranche value.

        Args:
            params: Array containing [default_prob, libor_rate]

        Returns:
            Present value of CLO tranche
        """
        # Unpack parameters
        if params.size >= 2:
            default_prob = params[0]
            libor_rate = params[1]
        else:
            default_prob = params[0] if params.size >= 1 else 0.03
            libor_rate = 0.03

        # Expected tranche loss
        expected_loss = self.expected_tranche_loss_lhp(default_prob)

        # Interest payments
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        def pv_interest(t: float) -> float:
            # Outstanding tranche notional
            outstanding = 1.0 - expected_loss  # Simplified

            # Interest payment
            interest = self.interest_waterfall(libor_rate, default_prob, outstanding)

            # Discount factor
            discount = jnp.exp(-libor_rate * t)

            return interest * dt * discount

        total_interest = jnp.sum(jax.vmap(pv_interest)(times))

        # Principal repayment
        remaining_principal = 1.0 - expected_loss
        discount = jnp.exp(-libor_rate * self.T)
        principal_pv = remaining_principal * discount

        # Total value
        return self.notional * (total_interest + principal_pv)

    def expected_return(self, default_prob: float, libor_rate: float) -> float:
        """Calculate expected return for equity tranche investors.

        Args:
            default_prob: Portfolio default probability
            libor_rate: LIBOR rate

        Returns:
            Expected annualized return
        """
        pv = self.payoff_terminal(jnp.array([default_prob, libor_rate]))
        implied_rate = (pv / self.notional) ** (1.0 / self.T) - 1.0
        return implied_rate


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
class TotalReturnSwap(Product):
    """Total Return Swap (TRS) on a credit asset.

    A TRS allows one party (total return receiver) to receive the total economic
    return (coupons + price appreciation/depreciation) of a reference asset without
    owning it, while paying a floating rate (typically LIBOR + spread) to the payer.

    The TRS is commonly used for:
    - Gaining exposure to credit assets without balance sheet impact
    - Financing positions
    - Regulatory capital arbitrage
    - Hedging credit risk

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        asset_coupon: Coupon rate of reference asset (annual)
        funding_spread: Spread over LIBOR paid by total return receiver (bps)
        initial_asset_price: Initial price of reference asset (as % of par)
        recovery_rate: Recovery rate if asset defaults
        payment_freq: Payment frequency per year

    References:
        - British Bankers' Association (1999). Credit Derivatives Report.
        - Das, S. (2000). Credit Derivatives: Trading & Management of Credit.
    """

    T: float
    notional: float
    asset_coupon: float  # Annual coupon rate (e.g., 0.05 for 5%)
    funding_spread: float  # Spread in basis points
    initial_asset_price: float = 100.0  # Price as % of par
    recovery_rate: float = 0.4
    payment_freq: int = 4

    def total_return_leg(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate total return leg (paid to receiver).

        Total return = Coupons + (Final Price - Initial Price)

        Args:
            final_asset_price: Terminal price of reference asset
            hazard_rate: Hazard rate of reference asset
            libor_rate: LIBOR rate for discounting

        Returns:
            Present value of total return payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Coupon payments (if no default)
        coupon_payment = self.asset_coupon / self.payment_freq

        def pv_coupon(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return coupon_payment * survival * discount

        total_coupons = jnp.sum(jax.vmap(pv_coupon)(times))

        # Price appreciation/depreciation
        survival = jnp.exp(-hazard_rate * self.T)
        default_prob = 1.0 - survival

        # If no default: return final price
        # If default: return recovery value
        expected_final = (
            survival * final_asset_price +
            default_prob * self.recovery_rate * 100.0
        )

        price_return = (expected_final - self.initial_asset_price) / 100.0
        discount = jnp.exp(-libor_rate * self.T)

        return self.notional * (total_coupons + price_return * discount)

    def funding_leg(self, libor_rate: float, hazard_rate: float = 0.0) -> float:
        """Calculate funding leg (paid by receiver).

        Funding = (LIBOR + spread) paid periodically

        Args:
            libor_rate: LIBOR rate
            hazard_rate: Hazard rate (affects payment if counterparty defaults)

        Returns:
            Present value of funding payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        funding_rate = libor_rate + self.funding_spread * 1e-4

        def pv_funding(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return funding_rate * dt * survival * discount

        return self.notional * jnp.sum(jax.vmap(pv_funding)(times))

    def payoff_terminal(self, state: Array) -> Array:
        """Calculate TRS value.

        TRS value = Total Return Leg - Funding Leg

        Args:
            state: Array containing [final_asset_price, hazard_rate, libor_rate]

        Returns:
            Present value of TRS from receiver's perspective
        """
        # Unpack state
        if state.size >= 3:
            final_price = state[0]
            hazard_rate = state[1]
            libor_rate = state[2]
        else:
            # Default values if not enough inputs
            final_price = 100.0
            hazard_rate = state[0] if state.size >= 1 else 0.01
            libor_rate = 0.03

        total_return = self.total_return_leg(final_price, hazard_rate, libor_rate)
        funding = self.funding_leg(libor_rate, hazard_rate)

        return total_return - funding

    def breakeven_spread(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate breakeven funding spread.

        The spread that makes TRS value zero at inception.

        Args:
            final_asset_price: Expected final price of reference asset
            hazard_rate: Hazard rate
            libor_rate: LIBOR rate

        Returns:
            Breakeven spread in basis points
        """
        total_return = self.total_return_leg(final_asset_price, hazard_rate, libor_rate)

        # Calculate funding leg with 1bp spread
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        pv01 = jnp.sum(
            jax.vmap(lambda t: jnp.exp(-hazard_rate * t) *
                     jnp.exp(-libor_rate * t) * dt)(times)
        )

        # Breakeven: total_return = (libor + spread) * pv01
        libor_pv = libor_rate * self.notional * pv01
        spread_in_decimal = (total_return - libor_pv) / (self.notional * pv01)

        return spread_in_decimal * 1e4  # Convert to basis points


@dataclass
class CollateralizedLoanObligation(Product):
    """Collateralized Loan Obligation (CLO) tranche.

    CLOs are structured finance securities backed by a pool of loans, typically
    leveraged loans. The cash flows from the loan pool are divided into tranches
    with different seniority levels.

    Key differences from CDOs:
    - Backed by actual loans (not synthetic)
    - Higher recovery rates (60-70% vs 40% for bonds)
    - Floating rate assets (loans tied to LIBOR/SOFR)
    - Active management of loan portfolio

    Args:
        T: Time to maturity (years)
        notional: Tranche notional
        attachment: Lower bound of tranche (as fraction of portfolio)
        detachment: Upper bound of tranche (as fraction of portfolio)
        spread: Tranche spread over LIBOR (bps)
        recovery_rate: Expected recovery rate for loans (typically 0.6-0.7)
        correlation: Default correlation between loans
        num_loans: Number of loans in portfolio
        payment_freq: Payment frequency per year
        loan_coupon: Average coupon rate of underlying loans (spread over LIBOR)

    References:
        - Tavakoli, J. (2003). Collateralized Debt Obligations and Structured Finance.
        - Goodman, L. & Fabozzi, F. (2002). Collateralized Debt Obligations.
    """

    T: float
    notional: float
    attachment: float
    detachment: float
    spread: float  # Tranche spread in bps
    recovery_rate: float = 0.65  # Higher for senior secured loans
    correlation: float = 0.25
    num_loans: int = 100
    payment_freq: int = 4
    loan_coupon: float = 400.0  # Average loan spread in bps (e.g., L+400)

    def tranche_loss(self, portfolio_loss: float) -> float:
        """Calculate tranche loss given portfolio loss.

        Args:
            portfolio_loss: Total portfolio loss as fraction of notional

        Returns:
            Tranche loss as fraction of tranche size
        """
        # Loss allocated to tranche
        loss_to_tranche = jnp.maximum(
            0.0,
            jnp.minimum(
                portfolio_loss - self.attachment,
                self.detachment - self.attachment
            )
        )

        # Normalize by tranche size
        tranche_size = self.detachment - self.attachment
        return loss_to_tranche / tranche_size if tranche_size > 0 else 0.0

    def expected_tranche_loss_lhp(self, default_prob: float) -> float:
        """Calculate expected tranche loss using Large Homogeneous Pool (LHP).

        Uses Gaussian copula model with numerical integration.

        Args:
            default_prob: Unconditional default probability

        Returns:
            Expected loss for the tranche
        """
        # Convert default probability to threshold
        K = norm.ppf(default_prob)

        # Conditional loss function
        def conditional_loss(market_factor: float) -> float:
            # Conditional default probability given market factor
            rho = self.correlation
            threshold = (K - jnp.sqrt(rho) * market_factor) / jnp.sqrt(1 - rho)
            cond_default_prob = norm.cdf(threshold)

            # Portfolio loss (in LHP: E[loss] = default_prob * LGD)
            lgd = 1.0 - self.recovery_rate
            portfolio_loss = cond_default_prob * lgd

            # Tranche loss
            return self.tranche_loss(portfolio_loss)

        # Numerical integration using Gauss-Hermite quadrature
        n_points = 50
        x, w = np.polynomial.hermite.hermgauss(n_points)
        x = jnp.array(x) * jnp.sqrt(2.0)  # Scale for N(0,1)
        w = jnp.array(w) / jnp.sqrt(jnp.pi)

        expected_loss = jnp.sum(
            jax.vmap(conditional_loss)(x) * w
        )

        return expected_loss

    def interest_waterfall(
        self,
        libor_rate: float,
        default_prob: float,
        tranche_outstanding: float = 1.0
    ) -> float:
        """Calculate interest payments considering waterfall structure.

        In a CLO, interest is paid in order of seniority (waterfall).

        Args:
            libor_rate: LIBOR rate
            default_prob: Portfolio default probability
            tranche_outstanding: Fraction of tranche still outstanding

        Returns:
            Expected interest payment to tranche
        """
        # Interest rate for this tranche
        tranche_rate = libor_rate + self.spread * 1e-4

        # Available interest from loan pool
        pool_rate = libor_rate + self.loan_coupon * 1e-4

        # Survival probability
        survival = 1.0 - default_prob

        # Interest payment (simplified: assumes senior tranches get full payment)
        # More sophisticated model would consider full waterfall
        interest = tranche_rate * tranche_outstanding * survival

        return interest

    def payoff_terminal(self, params: Array) -> Array:
        """Calculate CLO tranche value.

        Args:
            params: Array containing [default_prob, libor_rate]

        Returns:
            Present value of CLO tranche
        """
        # Unpack parameters
        if params.size >= 2:
            default_prob = params[0]
            libor_rate = params[1]
        else:
            default_prob = params[0] if params.size >= 1 else 0.03
            libor_rate = 0.03

        # Expected tranche loss
        expected_loss = self.expected_tranche_loss_lhp(default_prob)

        # Interest payments
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        def pv_interest(t: float) -> float:
            # Outstanding tranche notional
            outstanding = 1.0 - expected_loss  # Simplified

            # Interest payment
            interest = self.interest_waterfall(libor_rate, default_prob, outstanding)

            # Discount factor
            discount = jnp.exp(-libor_rate * t)

            return interest * dt * discount

        total_interest = jnp.sum(jax.vmap(pv_interest)(times))

        # Principal repayment
        remaining_principal = 1.0 - expected_loss
        discount = jnp.exp(-libor_rate * self.T)
        principal_pv = remaining_principal * discount

        # Total value
        return self.notional * (total_interest + principal_pv)

    def expected_return(self, default_prob: float, libor_rate: float) -> float:
        """Calculate expected return for equity tranche investors.

        Args:
            default_prob: Portfolio default probability
            libor_rate: LIBOR rate

        Returns:
            Expected annualized return
        """
        pv = self.payoff_terminal(jnp.array([default_prob, libor_rate]))
        implied_rate = (pv / self.notional) ** (1.0 / self.T) - 1.0
        return implied_rate


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
class TotalReturnSwap(Product):
    """Total Return Swap (TRS) on a credit asset.

    A TRS allows one party (total return receiver) to receive the total economic
    return (coupons + price appreciation/depreciation) of a reference asset without
    owning it, while paying a floating rate (typically LIBOR + spread) to the payer.

    The TRS is commonly used for:
    - Gaining exposure to credit assets without balance sheet impact
    - Financing positions
    - Regulatory capital arbitrage
    - Hedging credit risk

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        asset_coupon: Coupon rate of reference asset (annual)
        funding_spread: Spread over LIBOR paid by total return receiver (bps)
        initial_asset_price: Initial price of reference asset (as % of par)
        recovery_rate: Recovery rate if asset defaults
        payment_freq: Payment frequency per year

    References:
        - British Bankers' Association (1999). Credit Derivatives Report.
        - Das, S. (2000). Credit Derivatives: Trading & Management of Credit.
    """

    T: float
    notional: float
    asset_coupon: float  # Annual coupon rate (e.g., 0.05 for 5%)
    funding_spread: float  # Spread in basis points
    initial_asset_price: float = 100.0  # Price as % of par
    recovery_rate: float = 0.4
    payment_freq: int = 4

    def total_return_leg(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate total return leg (paid to receiver).

        Total return = Coupons + (Final Price - Initial Price)

        Args:
            final_asset_price: Terminal price of reference asset
            hazard_rate: Hazard rate of reference asset
            libor_rate: LIBOR rate for discounting

        Returns:
            Present value of total return payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Coupon payments (if no default)
        coupon_payment = self.asset_coupon / self.payment_freq

        def pv_coupon(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return coupon_payment * survival * discount

        total_coupons = jnp.sum(jax.vmap(pv_coupon)(times))

        # Price appreciation/depreciation
        survival = jnp.exp(-hazard_rate * self.T)
        default_prob = 1.0 - survival

        # If no default: return final price
        # If default: return recovery value
        expected_final = (
            survival * final_asset_price +
            default_prob * self.recovery_rate * 100.0
        )

        price_return = (expected_final - self.initial_asset_price) / 100.0
        discount = jnp.exp(-libor_rate * self.T)

        return self.notional * (total_coupons + price_return * discount)

    def funding_leg(self, libor_rate: float, hazard_rate: float = 0.0) -> float:
        """Calculate funding leg (paid by receiver).

        Funding = (LIBOR + spread) paid periodically

        Args:
            libor_rate: LIBOR rate
            hazard_rate: Hazard rate (affects payment if counterparty defaults)

        Returns:
            Present value of funding payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        funding_rate = libor_rate + self.funding_spread * 1e-4

        def pv_funding(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return funding_rate * dt * survival * discount

        return self.notional * jnp.sum(jax.vmap(pv_funding)(times))

    def payoff_terminal(self, state: Array) -> Array:
        """Calculate TRS value.

        TRS value = Total Return Leg - Funding Leg

        Args:
            state: Array containing [final_asset_price, hazard_rate, libor_rate]

        Returns:
            Present value of TRS from receiver's perspective
        """
        # Unpack state
        if state.size >= 3:
            final_price = state[0]
            hazard_rate = state[1]
            libor_rate = state[2]
        else:
            # Default values if not enough inputs
            final_price = 100.0
            hazard_rate = state[0] if state.size >= 1 else 0.01
            libor_rate = 0.03

        total_return = self.total_return_leg(final_price, hazard_rate, libor_rate)
        funding = self.funding_leg(libor_rate, hazard_rate)

        return total_return - funding

    def breakeven_spread(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate breakeven funding spread.

        The spread that makes TRS value zero at inception.

        Args:
            final_asset_price: Expected final price of reference asset
            hazard_rate: Hazard rate
            libor_rate: LIBOR rate

        Returns:
            Breakeven spread in basis points
        """
        total_return = self.total_return_leg(final_asset_price, hazard_rate, libor_rate)

        # Calculate funding leg with 1bp spread
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        pv01 = jnp.sum(
            jax.vmap(lambda t: jnp.exp(-hazard_rate * t) *
                     jnp.exp(-libor_rate * t) * dt)(times)
        )

        # Breakeven: total_return = (libor + spread) * pv01
        libor_pv = libor_rate * self.notional * pv01
        spread_in_decimal = (total_return - libor_pv) / (self.notional * pv01)

        return spread_in_decimal * 1e4  # Convert to basis points


@dataclass
class CollateralizedLoanObligation(Product):
    """Collateralized Loan Obligation (CLO) tranche.

    CLOs are structured finance securities backed by a pool of loans, typically
    leveraged loans. The cash flows from the loan pool are divided into tranches
    with different seniority levels.

    Key differences from CDOs:
    - Backed by actual loans (not synthetic)
    - Higher recovery rates (60-70% vs 40% for bonds)
    - Floating rate assets (loans tied to LIBOR/SOFR)
    - Active management of loan portfolio

    Args:
        T: Time to maturity (years)
        notional: Tranche notional
        attachment: Lower bound of tranche (as fraction of portfolio)
        detachment: Upper bound of tranche (as fraction of portfolio)
        spread: Tranche spread over LIBOR (bps)
        recovery_rate: Expected recovery rate for loans (typically 0.6-0.7)
        correlation: Default correlation between loans
        num_loans: Number of loans in portfolio
        payment_freq: Payment frequency per year
        loan_coupon: Average coupon rate of underlying loans (spread over LIBOR)

    References:
        - Tavakoli, J. (2003). Collateralized Debt Obligations and Structured Finance.
        - Goodman, L. & Fabozzi, F. (2002). Collateralized Debt Obligations.
    """

    T: float
    notional: float
    attachment: float
    detachment: float
    spread: float  # Tranche spread in bps
    recovery_rate: float = 0.65  # Higher for senior secured loans
    correlation: float = 0.25
    num_loans: int = 100
    payment_freq: int = 4
    loan_coupon: float = 400.0  # Average loan spread in bps (e.g., L+400)

    def tranche_loss(self, portfolio_loss: float) -> float:
        """Calculate tranche loss given portfolio loss.

        Args:
            portfolio_loss: Total portfolio loss as fraction of notional

        Returns:
            Tranche loss as fraction of tranche size
        """
        # Loss allocated to tranche
        loss_to_tranche = jnp.maximum(
            0.0,
            jnp.minimum(
                portfolio_loss - self.attachment,
                self.detachment - self.attachment
            )
        )

        # Normalize by tranche size
        tranche_size = self.detachment - self.attachment
        return loss_to_tranche / tranche_size if tranche_size > 0 else 0.0

    def expected_tranche_loss_lhp(self, default_prob: float) -> float:
        """Calculate expected tranche loss using Large Homogeneous Pool (LHP).

        Uses Gaussian copula model with numerical integration.

        Args:
            default_prob: Unconditional default probability

        Returns:
            Expected loss for the tranche
        """
        # Convert default probability to threshold
        K = norm.ppf(default_prob)

        # Conditional loss function
        def conditional_loss(market_factor: float) -> float:
            # Conditional default probability given market factor
            rho = self.correlation
            threshold = (K - jnp.sqrt(rho) * market_factor) / jnp.sqrt(1 - rho)
            cond_default_prob = norm.cdf(threshold)

            # Portfolio loss (in LHP: E[loss] = default_prob * LGD)
            lgd = 1.0 - self.recovery_rate
            portfolio_loss = cond_default_prob * lgd

            # Tranche loss
            return self.tranche_loss(portfolio_loss)

        # Numerical integration using Gauss-Hermite quadrature
        n_points = 50
        x, w = np.polynomial.hermite.hermgauss(n_points)
        x = jnp.array(x) * jnp.sqrt(2.0)  # Scale for N(0,1)
        w = jnp.array(w) / jnp.sqrt(jnp.pi)

        expected_loss = jnp.sum(
            jax.vmap(conditional_loss)(x) * w
        )

        return expected_loss

    def interest_waterfall(
        self,
        libor_rate: float,
        default_prob: float,
        tranche_outstanding: float = 1.0
    ) -> float:
        """Calculate interest payments considering waterfall structure.

        In a CLO, interest is paid in order of seniority (waterfall).

        Args:
            libor_rate: LIBOR rate
            default_prob: Portfolio default probability
            tranche_outstanding: Fraction of tranche still outstanding

        Returns:
            Expected interest payment to tranche
        """
        # Interest rate for this tranche
        tranche_rate = libor_rate + self.spread * 1e-4

        # Available interest from loan pool
        pool_rate = libor_rate + self.loan_coupon * 1e-4

        # Survival probability
        survival = 1.0 - default_prob

        # Interest payment (simplified: assumes senior tranches get full payment)
        # More sophisticated model would consider full waterfall
        interest = tranche_rate * tranche_outstanding * survival

        return interest

    def payoff_terminal(self, params: Array) -> Array:
        """Calculate CLO tranche value.

        Args:
            params: Array containing [default_prob, libor_rate]

        Returns:
            Present value of CLO tranche
        """
        # Unpack parameters
        if params.size >= 2:
            default_prob = params[0]
            libor_rate = params[1]
        else:
            default_prob = params[0] if params.size >= 1 else 0.03
            libor_rate = 0.03

        # Expected tranche loss
        expected_loss = self.expected_tranche_loss_lhp(default_prob)

        # Interest payments
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        def pv_interest(t: float) -> float:
            # Outstanding tranche notional
            outstanding = 1.0 - expected_loss  # Simplified

            # Interest payment
            interest = self.interest_waterfall(libor_rate, default_prob, outstanding)

            # Discount factor
            discount = jnp.exp(-libor_rate * t)

            return interest * dt * discount

        total_interest = jnp.sum(jax.vmap(pv_interest)(times))

        # Principal repayment
        remaining_principal = 1.0 - expected_loss
        discount = jnp.exp(-libor_rate * self.T)
        principal_pv = remaining_principal * discount

        # Total value
        return self.notional * (total_interest + principal_pv)

    def expected_return(self, default_prob: float, libor_rate: float) -> float:
        """Calculate expected return for equity tranche investors.

        Args:
            default_prob: Portfolio default probability
            libor_rate: LIBOR rate

        Returns:
            Expected annualized return
        """
        pv = self.payoff_terminal(jnp.array([default_prob, libor_rate]))
        implied_rate = (pv / self.notional) ** (1.0 / self.T) - 1.0
        return implied_rate


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


@dataclass
class TotalReturnSwap(Product):
    """Total Return Swap (TRS) on a credit asset.

    A TRS allows one party (total return receiver) to receive the total economic
    return (coupons + price appreciation/depreciation) of a reference asset without
    owning it, while paying a floating rate (typically LIBOR + spread) to the payer.

    The TRS is commonly used for:
    - Gaining exposure to credit assets without balance sheet impact
    - Financing positions
    - Regulatory capital arbitrage
    - Hedging credit risk

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        asset_coupon: Coupon rate of reference asset (annual)
        funding_spread: Spread over LIBOR paid by total return receiver (bps)
        initial_asset_price: Initial price of reference asset (as % of par)
        recovery_rate: Recovery rate if asset defaults
        payment_freq: Payment frequency per year

    References:
        - British Bankers' Association (1999). Credit Derivatives Report.
        - Das, S. (2000). Credit Derivatives: Trading & Management of Credit.
    """

    T: float
    notional: float
    asset_coupon: float  # Annual coupon rate (e.g., 0.05 for 5%)
    funding_spread: float  # Spread in basis points
    initial_asset_price: float = 100.0  # Price as % of par
    recovery_rate: float = 0.4
    payment_freq: int = 4

    def total_return_leg(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate total return leg (paid to receiver).

        Total return = Coupons + (Final Price - Initial Price)

        Args:
            final_asset_price: Terminal price of reference asset
            hazard_rate: Hazard rate of reference asset
            libor_rate: LIBOR rate for discounting

        Returns:
            Present value of total return payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        # Coupon payments (if no default)
        coupon_payment = self.asset_coupon / self.payment_freq

        def pv_coupon(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return coupon_payment * survival * discount

        total_coupons = jnp.sum(jax.vmap(pv_coupon)(times))

        # Price appreciation/depreciation
        survival = jnp.exp(-hazard_rate * self.T)
        default_prob = 1.0 - survival

        # If no default: return final price
        # If default: return recovery value
        expected_final = (
            survival * final_asset_price +
            default_prob * self.recovery_rate * 100.0
        )

        price_return = (expected_final - self.initial_asset_price) / 100.0
        discount = jnp.exp(-libor_rate * self.T)

        return self.notional * (total_coupons + price_return * discount)

    def funding_leg(self, libor_rate: float, hazard_rate: float = 0.0) -> float:
        """Calculate funding leg (paid by receiver).

        Funding = (LIBOR + spread) paid periodically

        Args:
            libor_rate: LIBOR rate
            hazard_rate: Hazard rate (affects payment if counterparty defaults)

        Returns:
            Present value of funding payments
        """
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        funding_rate = libor_rate + self.funding_spread * 1e-4

        def pv_funding(t: float) -> float:
            survival = jnp.exp(-hazard_rate * t)
            discount = jnp.exp(-libor_rate * t)
            return funding_rate * dt * survival * discount

        return self.notional * jnp.sum(jax.vmap(pv_funding)(times))

    def payoff_terminal(self, state: Array) -> Array:
        """Calculate TRS value.

        TRS value = Total Return Leg - Funding Leg

        Args:
            state: Array containing [final_asset_price, hazard_rate, libor_rate]

        Returns:
            Present value of TRS from receiver's perspective
        """
        # Unpack state
        if state.size >= 3:
            final_price = state[0]
            hazard_rate = state[1]
            libor_rate = state[2]
        else:
            # Default values if not enough inputs
            final_price = 100.0
            hazard_rate = state[0] if state.size >= 1 else 0.01
            libor_rate = 0.03

        total_return = self.total_return_leg(final_price, hazard_rate, libor_rate)
        funding = self.funding_leg(libor_rate, hazard_rate)

        return total_return - funding

    def breakeven_spread(
        self,
        final_asset_price: float,
        hazard_rate: float,
        libor_rate: float
    ) -> float:
        """Calculate breakeven funding spread.

        The spread that makes TRS value zero at inception.

        Args:
            final_asset_price: Expected final price of reference asset
            hazard_rate: Hazard rate
            libor_rate: LIBOR rate

        Returns:
            Breakeven spread in basis points
        """
        total_return = self.total_return_leg(final_asset_price, hazard_rate, libor_rate)

        # Calculate funding leg with 1bp spread
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        pv01 = jnp.sum(
            jax.vmap(lambda t: jnp.exp(-hazard_rate * t) *
                     jnp.exp(-libor_rate * t) * dt)(times)
        )

        # Breakeven: total_return = (libor + spread) * pv01
        libor_pv = libor_rate * self.notional * pv01
        spread_in_decimal = (total_return - libor_pv) / (self.notional * pv01)

        return spread_in_decimal * 1e4  # Convert to basis points


@dataclass
class CollateralizedLoanObligation(Product):
    """Collateralized Loan Obligation (CLO) tranche.

    CLOs are structured finance securities backed by a pool of loans, typically
    leveraged loans. The cash flows from the loan pool are divided into tranches
    with different seniority levels.

    Key differences from CDOs:
    - Backed by actual loans (not synthetic)
    - Higher recovery rates (60-70% vs 40% for bonds)
    - Floating rate assets (loans tied to LIBOR/SOFR)
    - Active management of loan portfolio

    Args:
        T: Time to maturity (years)
        notional: Tranche notional
        attachment: Lower bound of tranche (as fraction of portfolio)
        detachment: Upper bound of tranche (as fraction of portfolio)
        spread: Tranche spread over LIBOR (bps)
        recovery_rate: Expected recovery rate for loans (typically 0.6-0.7)
        correlation: Default correlation between loans
        num_loans: Number of loans in portfolio
        payment_freq: Payment frequency per year
        loan_coupon: Average coupon rate of underlying loans (spread over LIBOR)

    References:
        - Tavakoli, J. (2003). Collateralized Debt Obligations and Structured Finance.
        - Goodman, L. & Fabozzi, F. (2002). Collateralized Debt Obligations.
    """

    T: float
    notional: float
    attachment: float
    detachment: float
    spread: float  # Tranche spread in bps
    recovery_rate: float = 0.65  # Higher for senior secured loans
    correlation: float = 0.25
    num_loans: int = 100
    payment_freq: int = 4
    loan_coupon: float = 400.0  # Average loan spread in bps (e.g., L+400)

    def tranche_loss(self, portfolio_loss: float) -> float:
        """Calculate tranche loss given portfolio loss.

        Args:
            portfolio_loss: Total portfolio loss as fraction of notional

        Returns:
            Tranche loss as fraction of tranche size
        """
        # Loss allocated to tranche
        loss_to_tranche = jnp.maximum(
            0.0,
            jnp.minimum(
                portfolio_loss - self.attachment,
                self.detachment - self.attachment
            )
        )

        # Normalize by tranche size
        tranche_size = self.detachment - self.attachment
        return loss_to_tranche / tranche_size if tranche_size > 0 else 0.0

    def expected_tranche_loss_lhp(self, default_prob: float) -> float:
        """Calculate expected tranche loss using Large Homogeneous Pool (LHP).

        Uses Gaussian copula model with numerical integration.

        Args:
            default_prob: Unconditional default probability

        Returns:
            Expected loss for the tranche
        """
        # Convert default probability to threshold
        K = norm.ppf(default_prob)

        # Conditional loss function
        def conditional_loss(market_factor: float) -> float:
            # Conditional default probability given market factor
            rho = self.correlation
            threshold = (K - jnp.sqrt(rho) * market_factor) / jnp.sqrt(1 - rho)
            cond_default_prob = norm.cdf(threshold)

            # Portfolio loss (in LHP: E[loss] = default_prob * LGD)
            lgd = 1.0 - self.recovery_rate
            portfolio_loss = cond_default_prob * lgd

            # Tranche loss
            return self.tranche_loss(portfolio_loss)

        # Numerical integration using Gauss-Hermite quadrature
        n_points = 50
        x, w = np.polynomial.hermite.hermgauss(n_points)
        x = jnp.array(x) * jnp.sqrt(2.0)  # Scale for N(0,1)
        w = jnp.array(w) / jnp.sqrt(jnp.pi)

        expected_loss = jnp.sum(
            jax.vmap(conditional_loss)(x) * w
        )

        return expected_loss

    def interest_waterfall(
        self,
        libor_rate: float,
        default_prob: float,
        tranche_outstanding: float = 1.0
    ) -> float:
        """Calculate interest payments considering waterfall structure.

        In a CLO, interest is paid in order of seniority (waterfall).

        Args:
            libor_rate: LIBOR rate
            default_prob: Portfolio default probability
            tranche_outstanding: Fraction of tranche still outstanding

        Returns:
            Expected interest payment to tranche
        """
        # Interest rate for this tranche
        tranche_rate = libor_rate + self.spread * 1e-4

        # Available interest from loan pool
        pool_rate = libor_rate + self.loan_coupon * 1e-4

        # Survival probability
        survival = 1.0 - default_prob

        # Interest payment (simplified: assumes senior tranches get full payment)
        # More sophisticated model would consider full waterfall
        interest = tranche_rate * tranche_outstanding * survival

        return interest

    def payoff_terminal(self, params: Array) -> Array:
        """Calculate CLO tranche value.

        Args:
            params: Array containing [default_prob, libor_rate]

        Returns:
            Present value of CLO tranche
        """
        # Unpack parameters
        if params.size >= 2:
            default_prob = params[0]
            libor_rate = params[1]
        else:
            default_prob = params[0] if params.size >= 1 else 0.03
            libor_rate = 0.03

        # Expected tranche loss
        expected_loss = self.expected_tranche_loss_lhp(default_prob)

        # Interest payments
        dt = 1.0 / self.payment_freq
        times = jnp.arange(dt, self.T + dt, dt)

        def pv_interest(t: float) -> float:
            # Outstanding tranche notional
            outstanding = 1.0 - expected_loss  # Simplified

            # Interest payment
            interest = self.interest_waterfall(libor_rate, default_prob, outstanding)

            # Discount factor
            discount = jnp.exp(-libor_rate * t)

            return interest * dt * discount

        total_interest = jnp.sum(jax.vmap(pv_interest)(times))

        # Principal repayment
        remaining_principal = 1.0 - expected_loss
        discount = jnp.exp(-libor_rate * self.T)
        principal_pv = remaining_principal * discount

        # Total value
        return self.notional * (total_interest + principal_pv)

    def expected_return(self, default_prob: float, libor_rate: float) -> float:
        """Calculate expected return for equity tranche investors.

        Args:
            default_prob: Portfolio default probability
            libor_rate: LIBOR rate

        Returns:
            Expected annualized return
        """
        pv = self.payoff_terminal(jnp.array([default_prob, libor_rate]))
        implied_rate = (pv / self.notional) ** (1.0 / self.T) - 1.0
        return implied_rate
