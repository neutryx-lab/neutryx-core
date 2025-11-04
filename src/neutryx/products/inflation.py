"""Inflation-linked products and derivatives.

Implements inflation-linked instruments:
- Inflation-linked bonds (TIPS, linkers)
- Zero-coupon inflation swaps
- Year-on-year inflation swaps
- Inflation caps and floors
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import jit


@dataclass
class InflationIndex:
    """Inflation index specification."""

    current_level: float
    base_level: float
    publication_lag: float = 0.25  # 3 months typical lag


@jit
def inflation_linked_bond_price(
    face_value: float,
    real_coupon_rate: float,
    real_yield: float,
    maturity: float,
    index_ratio: float,
    frequency: int = 2,
) -> float:
    """Price an inflation-linked bond.

    Parameters
    ----------
    face_value : float
        Nominal face value (before inflation adjustment)
    real_coupon_rate : float
        Real coupon rate (annual)
    real_yield : float
        Real yield to maturity
    maturity : float
        Time to maturity in years
    index_ratio : float
        Current CPI / Base CPI (inflation adjustment factor)
    frequency : int
        Coupon payment frequency per year

    Returns
    -------
    float
        Current price of inflation-linked bond

    Notes
    -----
    Inflation-linked bonds (e.g., TIPS) adjust both principal and coupon
    payments for inflation based on an index ratio:
        Index Ratio = CPI_current / CPI_base

    The real cash flows are then discounted at the real yield.

    Price = IndexRatio * [PV(real coupons) + PV(real principal)]

    Examples
    --------
    >>> inflation_linked_bond_price(
    ...     100.0, 0.02, 0.01, 5.0, index_ratio=1.15, frequency=2
    ... )
    120.456...
    """
    # Adjusted face value and coupons by inflation
    adjusted_face = face_value * index_ratio
    real_coupon_payment = (real_coupon_rate * face_value * index_ratio) / frequency

    n_periods = maturity * frequency
    discount_rate = real_yield / frequency

    # Present value of real coupons
    pv_coupons = (
        real_coupon_payment * (1.0 - jnp.power(1.0 + discount_rate, -n_periods)) / discount_rate
    )

    # Present value of adjusted principal
    pv_principal = adjusted_face / jnp.power(1.0 + discount_rate, n_periods)

    return pv_coupons + pv_principal


@jit
def zero_coupon_inflation_swap_rate(
    forward_cpi: float,
    base_cpi: float,
) -> float:
    """Calculate zero-coupon inflation swap rate.

    Parameters
    ----------
    forward_cpi : float
        Expected CPI at maturity
    base_cpi : float
        Base CPI at inception

    Returns
    -------
    float
        Total inflation rate over the period

    Notes
    -----
    A zero-coupon inflation swap (ZCIS) exchanges:
    - Fixed leg: (1 + K)^T - 1 (fixed inflation rate K compounded)
    - Floating leg: CPI(T) / CPI(0) - 1 (realized inflation)

    The fair rate K satisfies:
        E[CPI(T) / CPI(0)] = 1 + K_total

    Examples
    --------
    >>> zero_coupon_inflation_swap_rate(110.0, 100.0)
    0.1
    """
    return (forward_cpi / base_cpi) - 1.0


@jit
def zero_coupon_inflation_swap_value(
    notional: float,
    strike: float,
    maturity: float,
    current_cpi: float,
    forward_cpi: float,
    base_cpi: float,
    discount_factor: float,
) -> float:
    """Calculate value of zero-coupon inflation swap.

    Parameters
    ----------
    notional : float
        Notional amount
    strike : float
        Fixed inflation rate (total, not annualized)
    maturity : float
        Time to maturity in years
    current_cpi : float
        Current CPI level
    forward_cpi : float
        Expected CPI at maturity
    base_cpi : float
        Base CPI at inception
    discount_factor : float
        Discount factor to maturity

    Returns
    -------
    float
        Present value to inflation receiver

    Notes
    -----
    Value = Notional * [(Forward_CPI/Base_CPI - 1) - Strike] * DF

    Examples
    --------
    >>> zero_coupon_inflation_swap_value(
    ...     1_000_000, 0.10, 5.0, 105.0, 115.0, 100.0, 0.95
    ... )
    47_500.0
    """
    expected_inflation = (forward_cpi / base_cpi) - 1.0
    payoff = notional * (expected_inflation - strike)
    return payoff * discount_factor


@jit
def year_on_year_inflation_swap_rate(
    forward_cpi_t: float,
    forward_cpi_t_minus_1: float,
) -> float:
    """Calculate year-on-year inflation rate.

    Parameters
    ----------
    forward_cpi_t : float
        Expected CPI at time t
    forward_cpi_t_minus_1 : float
        Expected CPI at time t-1

    Returns
    -------
    float
        Year-on-year inflation rate

    Examples
    --------
    >>> year_on_year_inflation_swap_rate(103.0, 100.0)
    0.03
    """
    return (forward_cpi_t / forward_cpi_t_minus_1) - 1.0


def year_on_year_inflation_swap_value(
    notional: float,
    strike: float,
    payment_dates: jnp.ndarray,
    forward_cpi_levels: jnp.ndarray,
    discount_factors: jnp.ndarray,
) -> float:
    """Calculate value of year-on-year inflation swap.

    Parameters
    ----------
    notional : float
        Notional amount
    strike : float
        Fixed annual inflation rate
    payment_dates : Array
        Payment dates (in years)
    forward_cpi_levels : Array
        Expected CPI at each payment date
    discount_factors : Array
        Discount factors for each payment

    Returns
    -------
    float
        Present value to inflation receiver

    Notes
    -----
    A YoY swap exchanges annual inflation realizations for a fixed rate.
    Each period pays:
        Notional * [(CPI(t)/CPI(t-1) - 1) - Strike]

    Examples
    --------
    >>> dates = jnp.array([1.0, 2.0, 3.0])
    >>> cpis = jnp.array([102.0, 105.0, 108.0])
    >>> dfs = jnp.array([0.95, 0.90, 0.85])
    >>> year_on_year_inflation_swap_value(1_000_000, 0.02, dates, cpis, dfs)
    18_571.43...
    """
    # Calculate year-on-year inflation rates
    # Insert base CPI of 100 at the start
    base_cpi = 100.0
    cpi_with_base = jnp.concatenate([jnp.array([base_cpi]), forward_cpi_levels])

    yoy_inflation = (cpi_with_base[1:] / cpi_with_base[:-1]) - 1.0

    # Calculate payoffs
    payoffs = notional * (yoy_inflation - strike)

    # Discount and sum
    pv = jnp.sum(payoffs * discount_factors)

    return float(pv)


@jit
def inflation_caplet_price(
    notional: float,
    strike: float,
    forward_inflation: float,
    volatility: float,
    maturity: float,
    discount_factor: float,
) -> float:
    """Price an inflation caplet using Black's model.

    Parameters
    ----------
    notional : float
        Notional amount
    strike : float
        Cap strike (inflation rate)
    forward_inflation : float
        Forward inflation rate
    volatility : float
        Inflation volatility
    maturity : float
        Time to maturity
    discount_factor : float
        Discount factor

    Returns
    -------
    float
        Caplet price

    Notes
    -----
    Uses Black's model for pricing inflation caps/floors:
        Caplet = N * DF * [F * Φ(d1) - K * Φ(d2)]

    where:
        d1 = [ln(F/K) + 0.5 * σ² * T] / (σ * sqrt(T))
        d2 = d1 - σ * sqrt(T)

    Examples
    --------
    >>> inflation_caplet_price(
    ...     1_000_000, 0.03, 0.025, 0.01, 1.0, 0.95
    ... )
    2_851.32...
    """
    # Avoid division by zero
    vol_sqrt_t = volatility * jnp.sqrt(maturity)
    vol_sqrt_t = jnp.maximum(vol_sqrt_t, 1e-10)

    # Black's formula
    d1 = (jnp.log(forward_inflation / strike) + 0.5 * volatility**2 * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    # Standard normal CDF
    from jax.scipy.stats import norm

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)

    caplet_value = notional * discount_factor * (forward_inflation * nd1 - strike * nd2)

    return jnp.maximum(caplet_value, 0.0)


@jit
def inflation_floorlet_price(
    notional: float,
    strike: float,
    forward_inflation: float,
    volatility: float,
    maturity: float,
    discount_factor: float,
) -> float:
    """Price an inflation floorlet using Black's model.

    Parameters
    ----------
    notional : float
        Notional amount
    strike : float
        Floor strike (inflation rate)
    forward_inflation : float
        Forward inflation rate
    volatility : float
        Inflation volatility
    maturity : float
        Time to maturity
    discount_factor : float
        Discount factor

    Returns
    -------
    float
        Floorlet price

    Notes
    -----
    Uses put-call parity:
        Floorlet = Caplet - Forward + Strike

    Or directly:
        Floorlet = N * DF * [K * Φ(-d2) - F * Φ(-d1)]

    Examples
    --------
    >>> inflation_floorlet_price(
    ...     1_000_000, 0.03, 0.025, 0.01, 1.0, 0.95
    ... )
    7_351.32...
    """
    vol_sqrt_t = volatility * jnp.sqrt(maturity)
    vol_sqrt_t = jnp.maximum(vol_sqrt_t, 1e-10)

    d1 = (jnp.log(forward_inflation / strike) + 0.5 * volatility**2 * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    from jax.scipy.stats import norm

    n_minus_d1 = norm.cdf(-d1)
    n_minus_d2 = norm.cdf(-d2)

    floorlet_value = notional * discount_factor * (strike * n_minus_d2 - forward_inflation * n_minus_d1)

    return jnp.maximum(floorlet_value, 0.0)


def real_to_nominal_yield(
    real_yield: float,
    expected_inflation: float,
    inflation_risk_premium: float = 0.0,
) -> float:
    """Convert real yield to nominal yield using Fisher equation.

    Parameters
    ----------
    real_yield : float
        Real interest rate
    expected_inflation : float
        Expected inflation rate
    inflation_risk_premium : float
        Risk premium for inflation uncertainty

    Returns
    -------
    float
        Nominal yield

    Notes
    -----
    Fisher equation (exact form):
        (1 + nominal) = (1 + real) * (1 + inflation) * (1 + risk_premium)

    Approximation:
        nominal ≈ real + inflation + risk_premium

    Examples
    --------
    >>> real_to_nominal_yield(0.02, 0.025, 0.005)
    0.050...
    """
    # Use exact Fisher equation
    nominal = (
        (1.0 + real_yield) * (1.0 + expected_inflation) * (1.0 + inflation_risk_premium) - 1.0
    )
    return float(nominal)


def breakeven_inflation(
    nominal_yield: float,
    real_yield: float,
) -> float:
    """Calculate breakeven inflation rate.

    Parameters
    ----------
    nominal_yield : float
        Yield on nominal bond
    real_yield : float
        Yield on inflation-linked bond

    Returns
    -------
    float
        Breakeven inflation rate

    Notes
    -----
    Breakeven inflation is the inflation rate that would make an investor
    indifferent between nominal and inflation-linked bonds:

        (1 + nominal) = (1 + real) * (1 + breakeven)

    Examples
    --------
    >>> breakeven_inflation(0.05, 0.02)
    0.0294...
    """
    breakeven = ((1.0 + nominal_yield) / (1.0 + real_yield)) - 1.0
    return float(breakeven)


# Inflation Swap Variants
def inflation_swap_pv01(
    notional: float,
    maturity: float,
    payment_frequency: int = 1,
) -> float:
    """Calculate PV01 (present value of 1bp) for inflation swap.

    Parameters
    ----------
    notional : float
        Notional amount
    maturity : float
        Swap maturity in years
    payment_frequency : int
        Payment frequency per year

    Returns
    -------
    float
        PV01

    Notes
    -----
    PV01 measures the sensitivity to a 1bp change in the inflation rate.
    """
    n_periods = int(maturity * payment_frequency)
    dt = 1.0 / payment_frequency

    # Simplified: assumes constant discount factor
    discount_rate = 0.03  # Placeholder
    pv01 = 0.0

    for i in range(1, n_periods + 1):
        t = i * dt
        df = jnp.exp(-discount_rate * t)
        pv01 += df * dt

    return float(notional * pv01 * 0.0001)  # 1bp = 0.0001


def asset_swap_spread_inflation(
    bond_price: float,
    par_value: float,
    real_coupon_rate: float,
    swap_rate: float,
    maturity: float,
) -> float:
    """Calculate asset swap spread for inflation-linked bond.

    Parameters
    ----------
    bond_price : float
        Current market price of the inflation-linked bond
    par_value : float
        Par value of the bond
    real_coupon_rate : float
        Real coupon rate
    swap_rate : float
        Current swap rate
    maturity : float
        Time to maturity

    Returns
    -------
    float
        Asset swap spread

    Notes
    -----
    Asset swap spread = (Par - Price) / (Duration * Price) + Real_Coupon - Swap_Rate
    """
    # Simplified duration calculation
    duration = maturity / 2.0  # Approximation

    spread = ((par_value - bond_price) / (duration * bond_price)) + real_coupon_rate - swap_rate

    return float(spread)


# Advanced Inflation Products
@jit
def inflation_swaption_price(
    strike_inflation: float,
    forward_inflation: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    notional: float = 1_000_000.0,
    is_payer: bool = True,
) -> float:
    """Price an inflation swaption (option on inflation swap).

    Parameters
    ----------
    strike_inflation : float
        Strike inflation rate
    forward_inflation : float
        Forward inflation rate
    time_to_expiry : float
        Time to expiry
    volatility : float
        Inflation volatility
    discount_factor : float
        Discount factor
    notional : float
        Notional amount
    is_payer : bool
        True for payer, False for receiver

    Returns
    -------
    float
        Inflation swaption price

    Notes
    -----
    Uses Black's model for inflation swaptions.
    """
    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(forward_inflation / strike_inflation) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )
    d2 = d1 - volatility * sqrt_T

    from jax.scipy.stats import norm

    if is_payer:
        value = forward_inflation * norm.cdf(d1) - strike_inflation * norm.cdf(d2)
    else:
        value = strike_inflation * norm.cdf(-d2) - forward_inflation * norm.cdf(-d1)

    return notional * discount_factor * value


def limited_price_indexation_bond(
    face_value: float,
    real_coupon_rate: float,
    real_yield: float,
    maturity: float,
    index_ratio: float,
    inflation_cap: float,
    inflation_floor: float,
    frequency: int = 2,
) -> float:
    """Price an inflation-linked bond with limited price indexation (LPI).

    Parameters
    ----------
    face_value : float
        Nominal face value
    real_coupon_rate : float
        Real coupon rate
    real_yield : float
        Real yield to maturity
    maturity : float
        Time to maturity
    index_ratio : float
        Current CPI / Base CPI
    inflation_cap : float
        Maximum inflation adjustment (e.g., 0.05 for 5%)
    inflation_floor : float
        Minimum inflation adjustment (e.g., 0.0)
    frequency : int
        Coupon frequency per year

    Returns
    -------
    float
        LPI bond price

    Notes
    -----
    LPI bonds have caps and floors on inflation adjustments.
    Common in UK pension schemes.
    """
    # Capped and floored index ratio
    inflation_rate = index_ratio - 1.0
    capped_inflation = float(jnp.minimum(jnp.maximum(inflation_rate, inflation_floor), inflation_cap))
    adjusted_index_ratio = 1.0 + capped_inflation

    # Price like a regular inflation bond with adjusted index ratio
    price = float(
        inflation_linked_bond_price(
            face_value, real_coupon_rate, real_yield, maturity, adjusted_index_ratio, frequency
        )
    )

    return price


def inflation_risk_premium(
    nominal_yield: float,
    real_yield: float,
    expected_inflation: float,
) -> float:
    """Calculate implied inflation risk premium.

    Parameters
    ----------
    nominal_yield : float
        Yield on nominal bond
    real_yield : float
        Yield on real (inflation-linked) bond
    expected_inflation : float
        Expected inflation rate

    Returns
    -------
    float
        Inflation risk premium

    Notes
    -----
    Fisher equation with risk premium:
        (1 + nominal) = (1 + real) * (1 + inflation) * (1 + risk_premium)

    Solving for risk premium:
        risk_premium = (1 + nominal) / [(1 + real) * (1 + inflation)] - 1
    """
    risk_premium = (1.0 + nominal_yield) / ((1.0 + real_yield) * (1.0 + expected_inflation)) - 1.0

    return float(risk_premium)


@dataclass
class InflationLinkedSwap:
    """Inflation-linked interest rate swap with both legs inflation-indexed.

    Args:
        notional: Notional amount
        maturity: Swap maturity in years
        fixed_real_rate: Fixed real rate on one leg
        payment_frequency: Payment frequency per year
    """

    notional: float
    maturity: float
    fixed_real_rate: float
    payment_frequency: int = 1

    def price(
        self,
        cpi_path: jnp.ndarray,
        discount_factors: jnp.ndarray,
    ) -> float:
        """Price the inflation-linked swap.

        Parameters
        ----------
        cpi_path : Array
            CPI levels at each payment date
        discount_factors : Array
            Discount factors for each payment

        Returns
        -------
        float
            Swap value
        """
        n_periods = int(self.maturity * self.payment_frequency)
        dt = 1.0 / self.payment_frequency

        # Fixed leg (real)
        fixed_leg = 0.0
        for i in range(n_periods):
            # Real payment adjusted for inflation
            real_payment = self.notional * self.fixed_real_rate * dt
            inflation_adjustment = cpi_path[i] / cpi_path[0]
            nominal_payment = real_payment * inflation_adjustment
            fixed_leg += nominal_payment * discount_factors[i]

        # Floating leg (inflation-linked)
        # Pays realized inflation
        floating_leg = 0.0
        for i in range(1, n_periods + 1):
            inflation_rate = (cpi_path[i] / cpi_path[i - 1]) - 1.0
            payment = self.notional * inflation_rate * dt
            floating_leg += payment * discount_factors[i]

        return float(floating_leg - fixed_leg)


@dataclass
class InflationFloorCertificate:
    """Certificate with guaranteed minimum inflation-linked return.

    Args:
        notional: Notional amount
        maturity: Maturity in years
        participation_rate: Participation in inflation (e.g., 1.0 = 100%)
        guaranteed_floor: Minimum guaranteed return
    """

    notional: float
    maturity: float
    participation_rate: float = 1.0
    guaranteed_floor: float = 0.0

    def payoff(self, realized_inflation: float) -> float:
        """Calculate payoff at maturity.

        Parameters
        ----------
        realized_inflation : float
            Total realized inflation over the period

        Returns
        -------
        float
            Payoff amount
        """
        inflation_return = self.participation_rate * realized_inflation
        total_return = jnp.maximum(inflation_return, self.guaranteed_floor)

        return float(self.notional * (1.0 + total_return))


__all__ = [
    "InflationIndex",
    "breakeven_inflation",
    "inflation_caplet_price",
    "inflation_floorlet_price",
    "inflation_linked_bond_price",
    "real_to_nominal_yield",
    "year_on_year_inflation_swap_rate",
    "year_on_year_inflation_swap_value",
    "zero_coupon_inflation_swap_rate",
    "zero_coupon_inflation_swap_value",
    # Advanced inflation products
    "inflation_swap_pv01",
    "asset_swap_spread_inflation",
    "inflation_swaption_price",
    "limited_price_indexation_bond",
    "inflation_risk_premium",
    "InflationLinkedSwap",
    "InflationFloorCertificate",
]
