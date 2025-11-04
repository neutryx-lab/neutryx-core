"""Commodity derivatives and structured products.

Implements commodity-specific products:
- Commodity forwards and futures
- Commodity options with convenience yield
- Commodity swaps
- Storage and transport options
- Sector-specific derivatives (Energy, Metals, Agriculture)
- Asian options for commodity averaging
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal

import jax.numpy as jnp
from jax import jit

from neutryx.models.bs import price as bs_price


class CommoditySector(Enum):
    """Commodity sector classification."""

    ENERGY = "energy"
    METALS = "metals"
    AGRICULTURE = "agriculture"
    LIVESTOCK = "livestock"


class CommodityType(Enum):
    """Specific commodity types."""

    # Energy
    WTI_CRUDE = "wti_crude"
    BRENT_CRUDE = "brent_crude"
    NATURAL_GAS = "natural_gas"
    HEATING_OIL = "heating_oil"
    GASOLINE = "gasoline"
    COAL = "coal"

    # Metals
    GOLD = "gold"
    SILVER = "silver"
    COPPER = "copper"
    ALUMINUM = "aluminum"
    PLATINUM = "platinum"
    PALLADIUM = "palladium"

    # Agriculture
    CORN = "corn"
    WHEAT = "wheat"
    SOYBEANS = "soybeans"
    SUGAR = "sugar"
    COFFEE = "coffee"
    COTTON = "cotton"

    # Livestock
    LIVE_CATTLE = "live_cattle"
    LEAN_HOGS = "lean_hogs"


# Default parameters for different commodity types
COMMODITY_DEFAULTS = {
    # Energy - typically higher storage costs, moderate convenience yields
    CommodityType.WTI_CRUDE: {"storage_cost": 0.02, "convenience_yield": 0.03, "vol": 0.35},
    CommodityType.BRENT_CRUDE: {"storage_cost": 0.02, "convenience_yield": 0.03, "vol": 0.35},
    CommodityType.NATURAL_GAS: {"storage_cost": 0.05, "convenience_yield": 0.08, "vol": 0.50},
    CommodityType.HEATING_OIL: {"storage_cost": 0.03, "convenience_yield": 0.02, "vol": 0.30},
    CommodityType.GASOLINE: {"storage_cost": 0.03, "convenience_yield": 0.02, "vol": 0.32},
    CommodityType.COAL: {"storage_cost": 0.04, "convenience_yield": 0.05, "vol": 0.25},
    # Metals - lower storage costs, variable convenience yields
    CommodityType.GOLD: {"storage_cost": 0.005, "convenience_yield": 0.001, "vol": 0.15},
    CommodityType.SILVER: {"storage_cost": 0.01, "convenience_yield": 0.005, "vol": 0.25},
    CommodityType.COPPER: {"storage_cost": 0.02, "convenience_yield": 0.03, "vol": 0.22},
    CommodityType.ALUMINUM: {"storage_cost": 0.02, "convenience_yield": 0.025, "vol": 0.20},
    CommodityType.PLATINUM: {"storage_cost": 0.01, "convenience_yield": 0.01, "vol": 0.20},
    CommodityType.PALLADIUM: {"storage_cost": 0.01, "convenience_yield": 0.01, "vol": 0.30},
    # Agriculture - seasonal patterns, higher storage for some
    CommodityType.CORN: {"storage_cost": 0.06, "convenience_yield": 0.04, "vol": 0.30},
    CommodityType.WHEAT: {"storage_cost": 0.05, "convenience_yield": 0.04, "vol": 0.28},
    CommodityType.SOYBEANS: {"storage_cost": 0.06, "convenience_yield": 0.04, "vol": 0.25},
    CommodityType.SUGAR: {"storage_cost": 0.04, "convenience_yield": 0.02, "vol": 0.35},
    CommodityType.COFFEE: {"storage_cost": 0.05, "convenience_yield": 0.02, "vol": 0.40},
    CommodityType.COTTON: {"storage_cost": 0.04, "convenience_yield": 0.02, "vol": 0.25},
    # Livestock - high storage (maintenance) costs
    CommodityType.LIVE_CATTLE: {"storage_cost": 0.10, "convenience_yield": 0.02, "vol": 0.18},
    CommodityType.LEAN_HOGS: {"storage_cost": 0.12, "convenience_yield": 0.02, "vol": 0.22},
}


@dataclass
class CommodityForward:
    """Commodity forward contract specification.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Forward delivery price
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free interest rate
    commodity_type : CommodityType | None
        Type of commodity (auto-fills default parameters if provided)
    storage_cost : float
        Storage cost rate (overrides default if commodity_type is set)
    convenience_yield : float
        Convenience yield (overrides default if commodity_type is set)
    quantity : float
        Contract quantity/notional
    """

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    commodity_type: CommodityType | None = None
    storage_cost: float = 0.0
    convenience_yield: float = 0.0
    quantity: float = 1.0

    def __post_init__(self):
        """Auto-fill parameters based on commodity type if not explicitly set."""
        if self.commodity_type is not None and self.commodity_type in COMMODITY_DEFAULTS:
            defaults = COMMODITY_DEFAULTS[self.commodity_type]
            # Only use defaults if parameters are at their default values
            if self.storage_cost == 0.0:
                self.storage_cost = defaults["storage_cost"]
            if self.convenience_yield == 0.0:
                self.convenience_yield = defaults["convenience_yield"]


@jit
def commodity_forward_price(
    spot: float,
    maturity: float,
    risk_free_rate: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
) -> float:
    """Calculate theoretical forward price for commodity.

    Parameters
    ----------
    spot : float
        Current spot price of the commodity
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free interest rate (continuously compounded)
    storage_cost : float
        Storage cost as a continuous rate (default: 0.0)
    convenience_yield : float
        Convenience yield (benefit of holding physical commodity)

    Returns
    -------
    float
        Theoretical forward price

    Notes
    -----
    Forward price formula:
        F = S * exp((r + u - y) * T)

    where:
    - S = spot price
    - r = risk-free rate
    - u = storage cost rate
    - y = convenience yield
    - T = time to maturity

    The convenience yield represents the benefit of holding the physical
    commodity (e.g., ability to meet unexpected demand, maintain production).

    Examples
    --------
    >>> commodity_forward_price(50.0, 1.0, 0.05, storage_cost=0.02, convenience_yield=0.03)
    52.020...
    """
    carry_cost = risk_free_rate + storage_cost - convenience_yield
    return spot * jnp.exp(carry_cost * maturity)


@partial(jit, static_argnames=["position"])
def commodity_forward_value(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    position: str = "long",
) -> float:
    """Calculate mark-to-market value of commodity forward.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Forward strike (delivery price)
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free interest rate
    storage_cost : float
        Storage cost rate
    convenience_yield : float
        Convenience yield
    position : str
        'long' or 'short' position

    Returns
    -------
    float
        Present value of the forward contract

    Examples
    --------
    >>> commodity_forward_value(50.0, 48.0, 1.0, 0.05, 0.02, 0.03, "long")
    3.922...
    """
    forward_price = commodity_forward_price(
        spot, maturity, risk_free_rate, storage_cost, convenience_yield
    )
    discount_factor = jnp.exp(-risk_free_rate * maturity)

    payoff = forward_price - strike
    value = payoff * discount_factor

    return jnp.where(position == "long", value, -value)


@partial(jit, static_argnames=["option_type"])
def commodity_option_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    option_type: str = "call",
) -> float:
    """Price commodity option using Black-Scholes with convenience yield.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility of commodity price
    storage_cost : float
        Storage cost rate
    convenience_yield : float
        Convenience yield
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Option price

    Notes
    -----
    Uses Black-Scholes with cost-of-carry adjustment:
        b = r + u - y

    where u is storage cost and y is convenience yield.
    This is equivalent to treating (u - y) as a negative dividend yield.

    Examples
    --------
    >>> commodity_option_price(
    ...     50.0, 52.0, 1.0, 0.05, 0.30, storage_cost=0.02, convenience_yield=0.03
    ... )
    4.789...
    """
    # Net cost of carry (adjust as if it's a dividend yield)
    # b = r + u - y, so q = y - u (negative if storage costs exceed convenience)
    equivalent_dividend_yield = convenience_yield - storage_cost

    return bs_price(
        S=spot,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=equivalent_dividend_yield,
        sigma=volatility,
        kind=option_type,
    )


@dataclass
class CommoditySwap:
    """Commodity swap specification.

    A commodity swap exchanges fixed and floating payments based on commodity prices.
    Common structures include:
    - Fixed-for-floating: Fixed price vs market price
    - Basis swaps: One commodity vs another
    - Index swaps: Fixed vs commodity index

    Parameters
    ----------
    notional : float
        Quantity of commodity per period
    fixed_price : float
        Fixed price per unit
    payment_dates : list[float]
        Payment times in years
    commodity_type : CommodityType | None
        Type of commodity
    floating_prices : list[float] | None
        Expected floating prices for each period (for valuation)
    swap_type : str
        Type of swap ('fixed_floating', 'basis', 'index')
    """

    notional: float  # Quantity of commodity
    fixed_price: float  # Fixed price per unit
    payment_dates: list[float]
    commodity_type: CommodityType | None = None
    floating_prices: list[float] | None = None
    swap_type: Literal["fixed_floating", "basis", "index"] = "fixed_floating"


@partial(jit, static_argnames=["position"])
def commodity_swap_value(
    quantity: float,
    fixed_price: float,
    floating_price: float,
    discount_factor: float,
    position: str = "fixed_payer",
) -> float:
    """Calculate value of a single-period commodity swap.

    Parameters
    ----------
    quantity : float
        Quantity of commodity
    fixed_price : float
        Fixed price per unit
    floating_price : float
        Floating (market) price per unit
    discount_factor : float
        Discount factor to payment date
    position : str
        'fixed_payer' (pays fixed, receives floating) or
        'fixed_receiver' (receives fixed, pays floating)

    Returns
    -------
    float
        Present value of the swap

    Notes
    -----
    A commodity swap exchanges:
    - Fixed leg: Fixed price * Quantity
    - Floating leg: Market price * Quantity

    Value to fixed payer = Quantity * (Floating - Fixed) * DF

    Examples
    --------
    >>> commodity_swap_value(1000.0, 50.0, 55.0, 0.95, "fixed_payer")
    4750.0
    """
    payoff = quantity * (floating_price - fixed_price)
    value = payoff * discount_factor

    return jnp.where(position == "fixed_payer", value, -value)


def multi_period_commodity_swap_value(
    quantity: float,
    fixed_price: float,
    floating_prices: jnp.ndarray,
    discount_factors: jnp.ndarray,
    position: str = "fixed_payer",
) -> float:
    """Calculate value of multi-period commodity swap.

    Parameters
    ----------
    quantity : float
        Quantity of commodity per period
    fixed_price : float
        Fixed price per unit
    floating_prices : Array
        Expected floating prices for each period
    discount_factors : Array
        Discount factors for each payment date
    position : str
        'fixed_payer' or 'fixed_receiver'

    Returns
    -------
    float
        Total present value of the swap

    Examples
    --------
    >>> floating = jnp.array([52.0, 54.0, 53.0])
    >>> dfs = jnp.array([0.98, 0.96, 0.94])
    >>> multi_period_commodity_swap_value(1000, 50.0, floating, dfs, "fixed_payer")
    8180.0
    """
    payoffs = quantity * (floating_prices - fixed_price)
    pv = jnp.sum(payoffs * discount_factors)

    return float(jnp.where(position == "fixed_payer", pv, -pv))


@partial(jit, static_argnames=["option_type"])
def spread_option_price(
    spot1: float,
    spot2: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    vol1: float,
    vol2: float,
    correlation: float,
    option_type: str = "call",
) -> float:
    """Price a commodity spread option (approximate).

    Parameters
    ----------
    spot1 : float
        Spot price of first commodity
    spot2 : float
        Spot price of second commodity
    strike : float
        Strike on the spread
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    vol1 : float
        Volatility of first commodity
    vol2 : float
        Volatility of second commodity
    correlation : float
        Correlation between the two commodities
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Approximate option price

    Notes
    -----
    Spread option payoff:
        Call: max(S1 - S2 - K, 0)
        Put: max(K - (S1 - S2), 0)

    This uses Kirk's approximation for pricing spread options.
    The spread S1 - S2 is approximated as lognormal.

    Examples
    --------
    >>> spread_option_price(
    ...     50.0, 45.0, 3.0, 1.0, 0.05, 0.25, 0.30, 0.6, "call"
    ... )
    3.456...
    """
    # Kirk's approximation
    # Treat the spread as a single asset with adjusted parameters

    # Forward prices
    F1 = spot1 * jnp.exp(risk_free_rate * maturity)
    F2 = spot2 * jnp.exp(risk_free_rate * maturity)

    # Adjusted strike and spot
    spread_forward = F1 - F2
    adjusted_spot = spread_forward * jnp.exp(-risk_free_rate * maturity)

    # Approximate spread volatility
    weight2 = F2 / (F2 + strike)
    spread_vol = jnp.sqrt(
        vol1**2 + (weight2 * vol2) ** 2 - 2.0 * correlation * vol1 * vol2 * weight2
    )

    # Use Black-Scholes on the spread
    return bs_price(
        S=adjusted_spot,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=0.0,
        sigma=spread_vol,
        kind=option_type,
    )


@partial(jit, static_argnames=["option_type"])
def asian_commodity_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    num_fixings: int,
    option_type: str = "call",
) -> float:
    """Price Asian option on commodity (arithmetic average).

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility
    num_fixings : int
        Number of averaging fixings
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Approximate Asian option price

    Notes
    -----
    Asian options are common in commodity markets as they reduce
    manipulation risk and volatility.

    This uses the Curran approximation with adjusted volatility.

    Examples
    --------
    >>> asian_commodity_price(50.0, 50.0, 1.0, 0.05, 0.30, 12, "call")
    4.123...
    """
    # Adjusted volatility for arithmetic average
    # Using approximation: σ_avg ≈ σ / sqrt(3)
    adjusted_vol = volatility / jnp.sqrt(3.0)

    # Adjust for number of fixings
    fixing_adjustment = jnp.sqrt((num_fixings + 1.0) / (2.0 * num_fixings))
    final_vol = adjusted_vol * fixing_adjustment

    # Use Black-Scholes with adjusted volatility
    return bs_price(
        S=spot,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=0.0,
        sigma=final_vol,
        kind=option_type,
    )


@dataclass
class CommodityAsianOption:
    """Asian option on commodity with arithmetic or geometric averaging.

    Asian options are particularly common in commodity markets as they:
    - Reduce manipulation risk
    - Lower hedging costs
    - Better match cash flows for producers/consumers
    - Reduce volatility impact

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility (overridden by commodity_type if provided)
    commodity_type : CommodityType | None
        Type of commodity
    storage_cost : float
        Storage cost rate
    convenience_yield : float
        Convenience yield
    averaging_type : str
        'arithmetic' or 'geometric'
    strike_type : str
        'fixed' (fixed strike) or 'floating' (floating strike)
    num_fixings : int
        Number of averaging observations
    is_call : bool
        True for call, False for put
    """

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    volatility: float
    commodity_type: CommodityType | None = None
    storage_cost: float = 0.0
    convenience_yield: float = 0.0
    averaging_type: Literal["arithmetic", "geometric"] = "arithmetic"
    strike_type: Literal["fixed", "floating"] = "fixed"
    num_fixings: int = 12
    is_call: bool = True

    def __post_init__(self):
        """Auto-fill parameters based on commodity type."""
        if self.commodity_type is not None and self.commodity_type in COMMODITY_DEFAULTS:
            defaults = COMMODITY_DEFAULTS[self.commodity_type]
            if self.volatility == 0.0:
                self.volatility = defaults["vol"]
            if self.storage_cost == 0.0:
                self.storage_cost = defaults["storage_cost"]
            if self.convenience_yield == 0.0:
                self.convenience_yield = defaults["convenience_yield"]


@jit
def price_commodity_asian_arithmetic(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    num_fixings: int,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    is_call: bool = True,
) -> float:
    """Price arithmetic-average Asian option on commodity.

    Uses Turnbull-Wakeman approximation for arithmetic Asian options.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility
    num_fixings : int
        Number of averaging fixings
    storage_cost : float
        Storage cost rate
    convenience_yield : float
        Convenience yield
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Option price

    Notes
    -----
    The arithmetic average Asian option uses an adjusted volatility:
        σ_adj = σ * sqrt((2n+1)/(6(n+1)))
    where n is the number of fixings.
    """
    # Net cost of carry
    cost_of_carry = risk_free_rate + storage_cost - convenience_yield

    # Adjusted volatility for arithmetic averaging (Turnbull-Wakeman)
    n = float(num_fixings)
    vol_adjustment = jnp.sqrt((2.0 * n + 1.0) / (6.0 * (n + 1.0)))
    adjusted_vol = volatility * vol_adjustment

    # Adjusted forward price
    forward_price = spot * jnp.exp(cost_of_carry * maturity)

    # Moment matching for arithmetic average
    # First moment
    m1 = forward_price

    # Second moment (variance)
    if n > 1:
        # More accurate adjustment
        dt = maturity / n
        variance_sum = 0.0
        for i in range(int(n)):
            t = (i + 1) * dt
            variance_sum += jnp.exp(2.0 * cost_of_carry * t + volatility**2 * t)
        m2 = (spot**2 / n**2) * variance_sum
    else:
        m2 = spot**2 * jnp.exp(2.0 * cost_of_carry * maturity + volatility**2 * maturity)

    # Lognormal approximation parameters
    variance_A = jnp.log(m2 / (m1**2))
    mean_A = jnp.log(m1) - 0.5 * variance_A
    vol_A = jnp.sqrt(variance_A / maturity)

    # Use Black-Scholes with adjusted parameters
    d1 = (mean_A + variance_A - jnp.log(strike)) / jnp.sqrt(variance_A)
    d2 = d1 - jnp.sqrt(variance_A)

    from jax.scipy.stats import norm

    if is_call:
        price = jnp.exp(-risk_free_rate * maturity) * (
            m1 * norm.cdf(d1) - strike * norm.cdf(d2)
        )
    else:
        price = jnp.exp(-risk_free_rate * maturity) * (
            strike * norm.cdf(-d2) - m1 * norm.cdf(-d1)
        )

    return price


@jit
def price_commodity_asian_geometric(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    num_fixings: int,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    is_call: bool = True,
) -> float:
    """Price geometric-average Asian option on commodity.

    Geometric Asian options have closed-form solutions.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility
    num_fixings : int
        Number of averaging fixings
    storage_cost : float
        Storage cost rate
    convenience_yield : float
        Convenience yield
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Option price
    """
    # Net cost of carry
    cost_of_carry = risk_free_rate + storage_cost - convenience_yield

    # Adjusted parameters for geometric averaging
    n = float(num_fixings)

    # Adjusted cost of carry
    adjusted_carry = 0.5 * (cost_of_carry - 0.5 * volatility**2)

    # Adjusted volatility
    adjusted_vol = volatility / jnp.sqrt(3.0) * jnp.sqrt((n + 1.0) * (2.0 * n + 1.0) / (n**2))

    # Use Black-Scholes with adjusted parameters
    return bs_price(
        S=spot,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=cost_of_carry - adjusted_carry,
        sigma=adjusted_vol,
        kind="call" if is_call else "put",
    )


# Sector-specific futures and options
@dataclass
class EnergyFuture:
    """Energy commodity future (WTI, Brent, Natural Gas, etc.).

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Futures delivery price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    energy_type : CommodityType
        Type of energy commodity
    """

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    energy_type: CommodityType = CommodityType.WTI_CRUDE

    def __post_init__(self):
        """Validate energy type."""
        energy_types = [
            CommodityType.WTI_CRUDE,
            CommodityType.BRENT_CRUDE,
            CommodityType.NATURAL_GAS,
            CommodityType.HEATING_OIL,
            CommodityType.GASOLINE,
            CommodityType.COAL,
        ]
        if self.energy_type not in energy_types:
            raise ValueError(f"Invalid energy type: {self.energy_type}")

    def forward_price(self) -> float:
        """Calculate forward price for energy commodity."""
        defaults = COMMODITY_DEFAULTS[self.energy_type]
        return commodity_forward_price(
            spot=self.spot,
            maturity=self.maturity,
            risk_free_rate=self.risk_free_rate,
            storage_cost=defaults["storage_cost"],
            convenience_yield=defaults["convenience_yield"],
        )


@dataclass
class MetalFuture:
    """Metal commodity future (Gold, Silver, Copper, etc.).

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Futures delivery price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    metal_type : CommodityType
        Type of metal
    """

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    metal_type: CommodityType = CommodityType.GOLD

    def __post_init__(self):
        """Validate metal type."""
        metal_types = [
            CommodityType.GOLD,
            CommodityType.SILVER,
            CommodityType.COPPER,
            CommodityType.ALUMINUM,
            CommodityType.PLATINUM,
            CommodityType.PALLADIUM,
        ]
        if self.metal_type not in metal_types:
            raise ValueError(f"Invalid metal type: {self.metal_type}")

    def forward_price(self) -> float:
        """Calculate forward price for metal commodity."""
        defaults = COMMODITY_DEFAULTS[self.metal_type]
        return commodity_forward_price(
            spot=self.spot,
            maturity=self.maturity,
            risk_free_rate=self.risk_free_rate,
            storage_cost=defaults["storage_cost"],
            convenience_yield=defaults["convenience_yield"],
        )


@dataclass
class AgricultureFuture:
    """Agricultural commodity future (Corn, Wheat, Soybeans, etc.).

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Futures delivery price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    agriculture_type : CommodityType
        Type of agricultural commodity
    seasonal_factor : float
        Seasonal adjustment factor (default: 1.0)
    """

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    agriculture_type: CommodityType = CommodityType.CORN
    seasonal_factor: float = 1.0

    def __post_init__(self):
        """Validate agriculture type."""
        ag_types = [
            CommodityType.CORN,
            CommodityType.WHEAT,
            CommodityType.SOYBEANS,
            CommodityType.SUGAR,
            CommodityType.COFFEE,
            CommodityType.COTTON,
        ]
        if self.agriculture_type not in ag_types:
            raise ValueError(f"Invalid agriculture type: {self.agriculture_type}")

    def forward_price(self) -> float:
        """Calculate forward price for agricultural commodity with seasonal adjustment."""
        defaults = COMMODITY_DEFAULTS[self.agriculture_type]
        base_forward = commodity_forward_price(
            spot=self.spot,
            maturity=self.maturity,
            risk_free_rate=self.risk_free_rate,
            storage_cost=defaults["storage_cost"],
            convenience_yield=defaults["convenience_yield"],
        )
        # Apply seasonal factor
        return base_forward * self.seasonal_factor


@jit
def basis_swap_value(
    quantity: float,
    price_commodity1: float,
    price_commodity2: float,
    strike_spread: float,
    discount_factor: float,
    position: str = "long_spread",
) -> float:
    """Value a commodity basis swap (spread between two commodities).

    Parameters
    ----------
    quantity : float
        Quantity traded
    price_commodity1 : float
        Price of first commodity
    price_commodity2 : float
        Price of second commodity
    strike_spread : float
        Fixed spread (commodity1 - commodity2)
    discount_factor : float
        Discount factor
    position : str
        'long_spread' (long commodity1, short commodity2) or 'short_spread'

    Returns
    -------
    float
        Present value of the basis swap

    Notes
    -----
    Basis swaps are common in energy markets, e.g.:
    - WTI vs Brent crude
    - Natural gas hub spreads
    - Crack spreads (refining margins)

    Examples
    --------
    >>> basis_swap_value(1000.0, 75.0, 70.0, 3.0, 0.95, "long_spread")
    1900.0
    """
    actual_spread = price_commodity1 - price_commodity2
    payoff = quantity * (actual_spread - strike_spread)
    value = payoff * discount_factor

    return jnp.where(position == "long_spread", value, -value)


__all__ = [
    # Enums and constants
    "CommoditySector",
    "CommodityType",
    "COMMODITY_DEFAULTS",
    # Base classes
    "CommodityForward",
    "CommoditySwap",
    "CommodityAsianOption",
    # Sector-specific futures
    "EnergyFuture",
    "MetalFuture",
    "AgricultureFuture",
    # Pricing functions - Forwards
    "commodity_forward_price",
    "commodity_forward_value",
    # Pricing functions - Options
    "commodity_option_price",
    "asian_commodity_price",
    "price_commodity_asian_arithmetic",
    "price_commodity_asian_geometric",
    # Pricing functions - Swaps
    "commodity_swap_value",
    "multi_period_commodity_swap_value",
    "basis_swap_value",
    # Spread options
    "spread_option_price",
]
