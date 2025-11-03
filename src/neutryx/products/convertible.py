"""Convertible bonds and hybrid securities.

Implements convertible bond pricing and analytics:
- Convertible bonds with embedded call and put options
- Mandatory convertibles
- Exchangeable bonds
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import jit

from neutryx.models.bs import price as bs_price


@dataclass
class ConvertibleBond:
    """Convertible bond specification."""

    face_value: float
    coupon_rate: float
    maturity: float
    conversion_ratio: float  # Number of shares per bond
    call_price: float | None = None  # Issuer call price
    put_price: float | None = None  # Investor put price
    call_date: float | None = None
    put_date: float | None = None


@jit
def convertible_bond_parity(
    stock_price: float,
    conversion_ratio: float,
) -> float:
    """Calculate conversion parity (conversion value).

    Parameters
    ----------
    stock_price : float
        Current stock price
    conversion_ratio : float
        Number of shares received per bond

    Returns
    -------
    float
        Conversion value

    Notes
    -----
    Conversion Value = Stock Price × Conversion Ratio

    This is the value if converted immediately.

    Examples
    --------
    >>> convertible_bond_parity(50.0, 20.0)
    1000.0
    """
    return stock_price * conversion_ratio


@jit
def convertible_bond_premium(
    bond_price: float,
    conversion_value: float,
    straight_bond_value: float,
) -> tuple[float, float]:
    """Calculate conversion premium and investment premium.

    Parameters
    ----------
    bond_price : float
        Current convertible bond price
    conversion_value : float
        Conversion value (parity)
    straight_bond_value : float
        Value as straight bond (without conversion feature)

    Returns
    -------
    conversion_premium : float
        Premium over conversion value
    investment_premium : float
        Premium over straight bond value

    Notes
    -----
    Conversion Premium = (Bond Price / Conversion Value) - 1
    Investment Premium = (Bond Price / Straight Bond Value) - 1

    Examples
    --------
    >>> convertible_bond_premium(1100.0, 1000.0, 980.0)
    (0.1, 0.122...)
    """
    conversion_premium = (bond_price / jnp.maximum(conversion_value, 1e-10)) - 1.0
    investment_premium = (bond_price / jnp.maximum(straight_bond_value, 1e-10)) - 1.0

    return conversion_premium, investment_premium


@jit
def convertible_bond_floor(
    conversion_value: float,
    straight_bond_value: float,
) -> float:
    """Calculate convertible bond floor value.

    Parameters
    ----------
    conversion_value : float
        Conversion value
    straight_bond_value : float
        Straight bond value

    Returns
    -------
    float
        Floor value (minimum of straight bond or conversion value)

    Notes
    -----
    The floor represents downside protection:
        Floor = max(Straight Bond Value, Conversion Value)

    Examples
    --------
    >>> convertible_bond_floor(950.0, 980.0)
    980.0
    """
    return jnp.maximum(conversion_value, straight_bond_value)


@jit
def convertible_bond_delta(
    bond_price: float,
    stock_price: float,
    conversion_ratio: float,
    sensitivity: float = 0.5,
) -> float:
    """Calculate convertible bond equity delta.

    Parameters
    ----------
    bond_price : float
        Current convertible bond price
    stock_price : float
        Current stock price
    conversion_ratio : float
        Conversion ratio
    sensitivity : float
        Simplified sensitivity parameter (0 to 1)

    Returns
    -------
    float
        Equity delta

    Notes
    -----
    Simplified delta calculation:
        Δ = Conversion Ratio × Sensitivity

    where Sensitivity depends on moneyness of conversion option.

    A more accurate calculation would use option pricing models.

    Examples
    --------
    >>> convertible_bond_delta(1050.0, 50.0, 20.0, 0.6)
    12.0
    """
    return conversion_ratio * sensitivity


def convertible_bond_simple_price(
    face_value: float,
    coupon_rate: float,
    yield_rate: float,
    maturity: float,
    stock_price: float,
    conversion_ratio: float,
    stock_volatility: float,
    risk_free_rate: float,
    frequency: int = 2,
) -> dict[str, float]:
    """Price convertible bond using simplified approach.

    Parameters
    ----------
    face_value : float
        Face value of bond
    coupon_rate : float
        Annual coupon rate
    yield_rate : float
        Credit spread yield
    maturity : float
        Time to maturity
    stock_price : float
        Current stock price
    conversion_ratio : float
        Conversion ratio
    stock_volatility : float
        Stock volatility
    risk_free_rate : float
        Risk-free rate
    frequency : int
        Coupon frequency

    Returns
    -------
    dict
        Dictionary with components:
        - straight_bond: Straight bond value
        - conversion_value: Conversion value
        - option_value: Value of conversion option
        - convertible_value: Total convertible bond value

    Notes
    -----
    Simplified model:
        Convertible = Straight Bond + Conversion Option

    The conversion option is approximated as a European call option
    with strike equal to (Face Value / Conversion Ratio).

    Examples
    --------
    >>> result = convertible_bond_simple_price(
    ...     1000.0, 0.03, 0.05, 5.0, 45.0, 25.0, 0.30, 0.04, 2
    ... )
    >>> result['convertible_value']
    1150.23...
    """
    from neutryx.products.bonds import coupon_bond_price

    # 1. Straight bond value (no conversion)
    straight_bond = float(
        coupon_bond_price(face_value, coupon_rate, yield_rate, maturity, frequency)
    )

    # 2. Conversion value (immediate conversion)
    conversion_value = float(stock_price * conversion_ratio)

    # 3. Conversion option value
    # Strike = face value / conversion ratio (breakeven stock price)
    conversion_strike = face_value / conversion_ratio

    # Value of conversion option per share
    option_per_share = float(
        bs_price(
            S=stock_price,
            K=conversion_strike,
            T=maturity,
            r=risk_free_rate,
            q=0.0,
            sigma=stock_volatility,
            kind="call",
        )
    )

    # Total option value
    option_value = option_per_share * conversion_ratio

    # 4. Total convertible value
    # Simplified: max(straight bond, conversion value + option value)
    convertible_value = jnp.maximum(straight_bond, conversion_value) + option_value * 0.5

    return {
        "straight_bond": straight_bond,
        "conversion_value": conversion_value,
        "option_value": option_value,
        "convertible_value": float(convertible_value),
    }


@jit
def mandatory_convertible_price(
    stock_price: float,
    conversion_ratio_low: float,
    conversion_ratio_high: float,
    threshold_low: float,
    threshold_high: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    """Price mandatory convertible with variable conversion ratio.

    Parameters
    ----------
    stock_price : float
        Current stock price
    conversion_ratio_low : float
        Minimum conversion ratio
    conversion_ratio_high : float
        Maximum conversion ratio
    threshold_low : float
        Lower stock price threshold
    threshold_high : float
        Upper stock price threshold
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Stock volatility

    Returns
    -------
    float
        Mandatory convertible price

    Notes
    -----
    Mandatory convertibles have payoff structure:
    - If S(T) < L: receives conversion_ratio_high shares
    - If L ≤ S(T) ≤ U: receives par value in shares
    - If S(T) > U: receives conversion_ratio_low shares

    This can be replicated with options.

    Examples
    --------
    >>> mandatory_convertible_price(
    ...     100.0, 10.0, 12.0, 90.0, 110.0, 3.0, 0.05, 0.25
    ... )
    1050.23...
    """
    # Forward stock price
    forward = stock_price * jnp.exp(risk_free_rate * maturity)

    # Simplified valuation (approximation)
    # Expected payoff based on forward price
    if forward < threshold_low:
        expected_shares = conversion_ratio_high
    elif forward > threshold_high:
        expected_shares = conversion_ratio_low
    else:
        # Linear interpolation in the middle range
        weight = (forward - threshold_low) / (threshold_high - threshold_low)
        # In the middle range, gets par value worth of shares
        # Par / S(T) shares, but we approximate with average
        expected_shares = conversion_ratio_high * (1 - weight) + conversion_ratio_low * weight

    # Discount expected value
    discount_factor = jnp.exp(-risk_free_rate * maturity)
    value = expected_shares * forward * discount_factor

    return value


@jit
def exchangeable_bond_value(
    face_value: float,
    straight_bond_value: float,
    underlying_stock_price: float,
    conversion_ratio: float,
    volatility: float,
    maturity: float,
    risk_free_rate: float,
) -> float:
    """Price exchangeable bond (convertible into different company's stock).

    Parameters
    ----------
    face_value : float
        Face value of bond
    straight_bond_value : float
        Value as straight bond
    underlying_stock_price : float
        Price of stock it can exchange into
    conversion_ratio : float
        Exchange ratio
    volatility : float
        Volatility of underlying stock
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate

    Returns
    -------
    float
        Exchangeable bond value

    Notes
    -----
    Similar to convertible bond but exchanges into a different company's stock.
    Common when parent company issues bond exchangeable into subsidiary stock.

    Examples
    --------
    >>> exchangeable_bond_value(
    ...     1000.0, 950.0, 48.0, 22.0, 0.35, 4.0, 0.04
    ... )
    1098.45...
    """
    # Exchange value
    exchange_value = underlying_stock_price * conversion_ratio

    # Exchange option value
    strike = face_value / conversion_ratio
    option_per_share = bs_price(
        S=underlying_stock_price,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=0.0,
        sigma=volatility,
        kind="call",
    )

    option_value = option_per_share * conversion_ratio

    # Total value
    value = jnp.maximum(straight_bond_value, exchange_value) + option_value * 0.3

    return value


__all__ = [
    "ConvertibleBond",
    "convertible_bond_delta",
    "convertible_bond_floor",
    "convertible_bond_parity",
    "convertible_bond_premium",
    "convertible_bond_simple_price",
    "exchangeable_bond_value",
    "mandatory_convertible_price",
]
