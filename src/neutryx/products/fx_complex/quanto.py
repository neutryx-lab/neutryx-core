"""Quanto products - cross-currency derivatives with currency protection.

Quanto products allow investors to gain exposure to foreign assets
while eliminating FX risk. The payoff in domestic currency is based
on the foreign asset performance but without FX conversion.

Key feature: Quanto adjustment accounts for correlation between
asset returns and FX rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm

from ..base import Product

Array = jnp.ndarray


class QuantoType(Enum):
    """Type of quanto payoff."""

    FIXED = "fixed"  # Fixed quanto (domestic notional regardless of FX)
    FLOATING = "floating"  # Floating quanto (notional varies with FX)


@dataclass
class QuantoOption(Product):
    """Quanto option on foreign asset, paid in domestic currency.

    A quanto option allows an investor to gain exposure to a foreign asset
    without FX risk. The payoff is based on the foreign asset price but
    paid in domestic currency at a fixed exchange rate.

    Example: A USD-based investor buys a quanto call on Nikkei.
    - If Nikkei rises 10%, payoff = $10 per $100 notional
    - No matter what USD/JPY does

    Attributes:
        T: Maturity in years
        strike: Strike price (in foreign asset units)
        is_call: True for call, False for put
        domestic_rate: Domestic risk-free rate
        foreign_rate: Foreign risk-free rate
        asset_vol: Volatility of foreign asset
        fx_vol: Volatility of FX rate
        correlation: Correlation between asset and FX rate
        fixed_fx_rate: Fixed FX rate for conversion (1.0 for unit conversion)
    """

    strike: float
    is_call: bool = True
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    asset_vol: float = 0.20
    fx_vol: float = 0.10
    correlation: float = 0.0
    fixed_fx_rate: float = 1.0

    @property
    def requires_path(self) -> bool:
        """Quanto option can be priced analytically."""
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate quanto option payoff.

        Args:
            spot: Foreign asset spot price

        Returns:
            Option payoff in domestic currency
        """
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)

        # Convert at fixed FX rate
        return self.fixed_fx_rate * payoff

    def price(self, spot: float) -> float:
        """Price quanto option using adjusted Black-Scholes.

        The quanto adjustment modifies the drift of the foreign asset
        to account for the correlation between asset and FX.

        Returns:
            Present value of quanto option
        """
        # Quanto drift adjustment
        quanto_drift = self.correlation * self.asset_vol * self.fx_vol

        # Adjusted foreign rate
        adjusted_foreign_rate = self.foreign_rate - quanto_drift

        # Black-Scholes with adjusted drift
        d1 = (
            jnp.log(spot / self.strike)
            + (self.domestic_rate - adjusted_foreign_rate + 0.5 * self.asset_vol**2) * self.T
        ) / (self.asset_vol * jnp.sqrt(self.T))
        d2 = d1 - self.asset_vol * jnp.sqrt(self.T)

        if self.is_call:
            price = self.fixed_fx_rate * (
                spot * jnp.exp(-adjusted_foreign_rate * self.T) * norm.cdf(d1)
                - self.strike * jnp.exp(-self.domestic_rate * self.T) * norm.cdf(d2)
            )
        else:
            price = self.fixed_fx_rate * (
                self.strike * jnp.exp(-self.domestic_rate * self.T) * norm.cdf(-d2)
                - spot * jnp.exp(-adjusted_foreign_rate * self.T) * norm.cdf(-d1)
            )

        return float(price)


@dataclass
class QuantoForward(Product):
    """Quanto forward contract on foreign asset.

    A forward contract on a foreign asset, settled in domestic currency
    at a fixed exchange rate, eliminating FX risk.

    Attributes:
        T: Maturity in years
        forward_price: Forward price of foreign asset
        domestic_rate: Domestic risk-free rate
        foreign_rate: Foreign risk-free rate
        asset_vol: Asset volatility
        fx_vol: FX volatility
        correlation: Asset-FX correlation
        fixed_fx_rate: Fixed FX rate for settlement
        is_long: True for long position
    """

    forward_price: float
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    asset_vol: float = 0.20
    fx_vol: float = 0.10
    correlation: float = 0.0
    fixed_fx_rate: float = 1.0
    is_long: bool = True

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate quanto forward payoff.

        Args:
            spot: Foreign asset spot price at maturity

        Returns:
            Forward payoff in domestic currency
        """
        payoff = spot - self.forward_price
        if not self.is_long:
            payoff = -payoff

        return self.fixed_fx_rate * payoff

    def fair_forward_price(self, spot: float) -> float:
        """Calculate fair forward price with quanto adjustment.

        Args:
            spot: Current foreign asset spot price

        Returns:
            Fair forward price accounting for quanto effect
        """
        # Quanto drift adjustment
        quanto_drift = self.correlation * self.asset_vol * self.fx_vol

        # Fair forward price
        fair_forward = spot * jnp.exp(
            (self.domestic_rate - self.foreign_rate + quanto_drift) * self.T
        )

        return float(fair_forward)


@dataclass
class QuantoSwap(Product):
    """Quanto swap - exchange foreign asset returns for fixed rate.

    A swap where one party receives the return on a foreign asset
    (in domestic currency at fixed FX rate) and pays a fixed rate.

    Attributes:
        T: Maturity in years
        notional: Notional in domestic currency
        fixed_rate: Fixed rate to pay/receive (annualized)
        payment_frequency: Payments per year
        domestic_rate: Domestic risk-free rate
        foreign_rate: Foreign risk-free rate
        asset_vol: Foreign asset volatility
        fx_vol: FX volatility
        correlation: Asset-FX correlation
        is_payer: True if paying fixed, receiving floating
    """

    notional: float
    fixed_rate: float
    payment_frequency: int = 4  # Quarterly
    domestic_rate: float = 0.0
    foreign_rate: float = 0.0
    asset_vol: float = 0.20
    fx_vol: float = 0.10
    correlation: float = 0.0
    is_payer: bool = True

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Simplified quanto swap payoff (for terminal settlement).

        Args:
            spot: Not used for swap pricing

        Returns:
            Present value of quanto swap
        """
        return self._calculate_pv()

    def _calculate_pv(self) -> float:
        """Calculate present value of quanto swap."""
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency

        # Discount factors
        payment_times = jnp.arange(1, n_payments + 1) * period_length
        discount_factors = jnp.exp(-self.domestic_rate * payment_times)

        # Fixed leg PV
        fixed_cash_flows = self.notional * self.fixed_rate * period_length
        fixed_leg_pv = jnp.sum(fixed_cash_flows * discount_factors)

        # Floating leg PV (with quanto adjustment)
        quanto_drift = self.correlation * self.asset_vol * self.fx_vol
        adjusted_rate = self.foreign_rate - quanto_drift

        # Simplified: assume flat quanto-adjusted returns
        floating_cash_flows = self.notional * adjusted_rate * period_length
        floating_leg_pv = jnp.sum(floating_cash_flows * discount_factors)

        # Net value
        if self.is_payer:
            return floating_leg_pv - fixed_leg_pv
        else:
            return fixed_leg_pv - floating_leg_pv


@jit
def quanto_drift_adjustment(
    asset_vol: float,
    fx_vol: float,
    correlation: float,
) -> float:
    """Calculate quanto drift adjustment.

    The quanto adjustment accounts for the correlation between
    the foreign asset and the FX rate.

    Args:
        asset_vol: Volatility of foreign asset
        fx_vol: Volatility of FX rate (domestic/foreign)
        correlation: Correlation between asset returns and FX changes

    Returns:
        Drift adjustment to apply to foreign asset

    Notes:
        Adjustment = ρ × σ_asset × σ_fx

        Positive correlation means foreign asset and domestic currency
        are positively correlated, which increases the domestic
        currency value of the asset.
    """
    return correlation * asset_vol * fx_vol


@jit
def quanto_option_delta(
    spot: float,
    strike: float,
    time_to_maturity: float,
    domestic_rate: float,
    foreign_rate: float,
    asset_vol: float,
    fx_vol: float,
    correlation: float,
    is_call: bool = True,
) -> float:
    """Calculate delta of quanto option.

    Args:
        spot: Foreign asset spot price
        strike: Strike price
        time_to_maturity: Time to maturity
        domestic_rate: Domestic rate
        foreign_rate: Foreign rate
        asset_vol: Asset volatility
        fx_vol: FX volatility
        correlation: Asset-FX correlation
        is_call: True for call, False for put

    Returns:
        Delta (sensitivity to foreign asset price)
    """
    # Quanto adjustment
    quanto_drift = correlation * asset_vol * fx_vol
    adjusted_foreign_rate = foreign_rate - quanto_drift

    # d1 calculation
    sqrt_t = jnp.sqrt(time_to_maturity)
    d1 = (
        jnp.log(spot / strike)
        + (domestic_rate - adjusted_foreign_rate + 0.5 * asset_vol**2) * time_to_maturity
    ) / (asset_vol * sqrt_t)

    # Delta (use jnp.where for JAX compatibility)
    delta_call = jnp.exp(-adjusted_foreign_rate * time_to_maturity) * norm.cdf(d1)
    delta_put = -jnp.exp(-adjusted_foreign_rate * time_to_maturity) * norm.cdf(-d1)

    delta = jnp.where(is_call, delta_call, delta_put)

    return delta


class QuantoDrift:
    """Helper class for quanto drift calculations and analytics."""

    @staticmethod
    @jit
    def calculate_adjustment(
        asset_vol: float,
        fx_vol: float,
        correlation: float,
    ) -> float:
        """Calculate quanto drift adjustment."""
        return correlation * asset_vol * fx_vol

    @staticmethod
    @jit
    def implied_correlation(
        market_quanto_price: float,
        theoretical_price_no_quanto: float,
        asset_vol: float,
        fx_vol: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Imply correlation from market quanto prices.

        Args:
            market_quanto_price: Observed quanto option price
            theoretical_price_no_quanto: BS price without quanto adjustment
            asset_vol: Asset volatility
            fx_vol: FX volatility
            sensitivity: Price sensitivity to correlation

        Returns:
            Implied correlation between asset and FX
        """
        # Simplified: assume linear relationship
        price_diff = market_quanto_price - theoretical_price_no_quanto
        implied_corr = price_diff / (sensitivity * asset_vol * fx_vol)

        # Clip to valid correlation range
        return jnp.clip(implied_corr, -1.0, 1.0)

    @staticmethod
    @jit
    def quanto_vs_vanilla_spread(
        spot: float,
        strike: float,
        time_to_maturity: float,
        rates: tuple[float, float],  # (domestic, foreign)
        vols: tuple[float, float],  # (asset, fx)
        correlation: float,
    ) -> float:
        """Calculate spread between quanto and vanilla option prices.

        Args:
            spot: Foreign asset spot
            strike: Strike price
            time_to_maturity: Time to maturity
            rates: (domestic_rate, foreign_rate)
            vols: (asset_vol, fx_vol)
            correlation: Asset-FX correlation

        Returns:
            Price difference (quanto - vanilla)
        """
        domestic_rate, foreign_rate = rates
        asset_vol, fx_vol = vols

        quanto_adj = correlation * asset_vol * fx_vol

        # This is a simplified approximation
        # Full calculation requires pricing both options
        spread_approximation = (
            spot * quanto_adj * time_to_maturity * jnp.exp(-foreign_rate * time_to_maturity)
        )

        return spread_approximation


__all__ = [
    "QuantoType",
    "QuantoOption",
    "QuantoForward",
    "QuantoSwap",
    "QuantoDrift",
    "quanto_drift_adjustment",
    "quanto_option_delta",
]
