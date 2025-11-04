"""Dividend forecasting and dividend curve construction.

This module provides tools for forecasting equity dividends, constructing
dividend curves, and pricing dividend derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array


class DividendType(Enum):
    """Type of dividend payment."""
    CASH = "cash"  # Fixed cash dividend
    YIELD = "yield"  # Continuous dividend yield
    PROPORTIONAL = "proportional"  # Proportional to stock price
    DISCRETE = "discrete"  # Discrete cash dividends at specific dates


@dataclass
class DividendSchedule:
    """Schedule of discrete dividend payments.

    Attributes:
        ex_dates: Ex-dividend dates (in years from reference)
        amounts: Dividend amounts (in currency or as proportion)
        dividend_type: Type of dividend
    """

    ex_dates: Array
    amounts: Array
    dividend_type: DividendType = DividendType.CASH

    def __post_init__(self):
        """Validate dividend schedule."""
        self.ex_dates = jnp.asarray(self.ex_dates)
        self.amounts = jnp.asarray(self.amounts)

        if len(self.ex_dates) != len(self.amounts):
            raise ValueError("ex_dates and amounts must have same length")

        # Sort by ex-date
        sorted_indices = jnp.argsort(self.ex_dates)
        object.__setattr__(self, "ex_dates", self.ex_dates[sorted_indices])
        object.__setattr__(self, "amounts", self.amounts[sorted_indices])

    def total_dividends_before(self, T: float) -> float:
        """Compute total dividends paid before time T.

        Args:
            T: Time horizon

        Returns:
            Total dividend amount
        """
        mask = self.ex_dates < T
        return float(jnp.sum(self.amounts[mask]))

    def pv_dividends(self, T: float, discount_curve: Callable[[float], float]) -> float:
        """Compute present value of dividends paid before time T.

        Args:
            T: Time horizon
            discount_curve: Discount factor curve

        Returns:
            PV of dividends
        """
        mask = self.ex_dates < T
        pv = 0.0

        for t, div in zip(self.ex_dates[mask], self.amounts[mask]):
            df = discount_curve(float(t))
            pv += div * df

        return float(pv)


@dataclass
class DividendForecast:
    """Dividend forecast model.

    Forecasts future dividends based on historical patterns and analyst estimates.

    Attributes:
        ticker: Stock ticker
        historical_dividends: Historical dividend data
        growth_rate: Assumed dividend growth rate
        payout_ratio: Target dividend payout ratio
        analyst_estimates: Optional analyst dividend estimates
    """

    ticker: str
    historical_dividends: List[Tuple[float, float]]  # (time, amount) pairs
    growth_rate: float = 0.03  # 3% default growth
    payout_ratio: Optional[float] = None
    analyst_estimates: Optional[Dict[float, float]] = None

    def forecast(self, horizon: float, frequency: int = 4) -> DividendSchedule:
        """Forecast dividends over a given horizon.

        Args:
            horizon: Forecast horizon in years
            frequency: Dividend frequency per year (4 = quarterly)

        Returns:
            Forecasted dividend schedule
        """
        if not self.historical_dividends:
            raise ValueError("No historical dividends available")

        # Get last dividend
        last_time, last_amount = self.historical_dividends[-1]

        # Forecast future dividends
        n_periods = int(horizon * frequency)
        dt = 1.0 / frequency

        ex_dates = []
        amounts = []

        for i in range(1, n_periods + 1):
            t = last_time + i * dt

            # Check if analyst estimate available
            if self.analyst_estimates and t in self.analyst_estimates:
                amount = self.analyst_estimates[t]
            else:
                # Grow from last dividend
                years_ahead = t - last_time
                amount = last_amount * (1 + self.growth_rate) ** years_ahead

            ex_dates.append(t)
            amounts.append(amount)

        return DividendSchedule(jnp.array(ex_dates), jnp.array(amounts), DividendType.CASH)

    def estimate_growth_rate(self) -> float:
        """Estimate dividend growth rate from historical data.

        Returns:
            Estimated annualized growth rate
        """
        if len(self.historical_dividends) < 2:
            return self.growth_rate

        # Compute growth rates between consecutive dividends
        times = [t for t, _ in self.historical_dividends]
        amounts = [a for _, a in self.historical_dividends]

        growth_rates = []
        for i in range(1, len(amounts)):
            dt = times[i] - times[i - 1]
            if dt > 0 and amounts[i - 1] > 0:
                growth = (amounts[i] / amounts[i - 1]) ** (1.0 / dt) - 1.0
                growth_rates.append(growth)

        if growth_rates:
            return float(jnp.mean(jnp.array(growth_rates)))
        else:
            return self.growth_rate


@dataclass
class DividendYieldCurve:
    """Continuous dividend yield curve.

    For index options and equity derivatives, dividends are often
    modeled as a continuous yield.

    Attributes:
        tenors: Pillar times
        yields: Dividend yields (continuously compounded)
    """

    tenors: Array
    yields: Array

    def __post_init__(self):
        """Validate curve."""
        self.tenors = jnp.asarray(self.tenors)
        self.yields = jnp.asarray(self.yields)

        if len(self.tenors) != len(self.yields):
            raise ValueError("Tenors and yields must have same length")

    def get_yield(self, t: float) -> float:
        """Get dividend yield at time t.

        Args:
            t: Time

        Returns:
            Dividend yield
        """
        return float(jnp.interp(t, self.tenors, self.yields))

    def integrated_yield(self, T: float) -> float:
        """Compute integrated yield: ∫₀ᵀ q(t) dt.

        Args:
            T: Time horizon

        Returns:
            Integrated yield
        """
        # Piecewise linear integration
        integral = 0.0
        prev_time = 0.0
        prev_yield = self.yields[0]

        for time, yield_val in zip(self.tenors, self.yields):
            if T <= time:
                # Partial segment (trapezoidal rule)
                integral += 0.5 * (prev_yield + yield_val) * (T - prev_time)
                break
            else:
                # Full segment
                integral += 0.5 * (prev_yield + yield_val) * (time - prev_time)
                prev_time = time
                prev_yield = yield_val

        # Extrapolate if T > last tenor
        if T > self.tenors[-1]:
            integral += self.yields[-1] * (T - float(self.tenors[-1]))

        return float(integral)

    def forward_price_factor(self, T: float) -> float:
        """Compute forward price adjustment factor: exp(-∫₀ᵀ q(t) dt).

        Args:
            T: Time to maturity

        Returns:
            Adjustment factor for forward price
        """
        return jnp.exp(-self.integrated_yield(T))


def discrete_to_continuous_yield(
    dividend_schedule: DividendSchedule, spot_price: float, T: float
) -> float:
    """Convert discrete dividends to equivalent continuous yield.

    Args:
        dividend_schedule: Discrete dividend schedule
        spot_price: Current spot price
        T: Time horizon

    Returns:
        Equivalent continuous dividend yield
    """
    # Total dividends
    total_div = dividend_schedule.total_dividends_before(T)

    # Approximate continuous yield: q ≈ (total_div / S) / T
    if spot_price <= 0 or T <= 0:
        return 0.0

    return (total_div / spot_price) / T


@dataclass
class DividendStrip:
    """Dividend futures/strips pricing.

    Dividend strips are derivatives on future dividend payments.

    Attributes:
        start_date: Start of dividend collection period
        end_date: End of dividend collection period
        forward_dividends: Expected dividends in period
    """

    start_date: float
    end_date: float
    forward_dividends: float

    def fair_value(self, discount_curve: Callable[[float], float]) -> float:
        """Compute fair value of dividend strip.

        Args:
            discount_curve: Discount factor curve

        Returns:
            Fair value (PV of expected dividends)
        """
        # Approximate dividend timing at midpoint
        mid_time = (self.start_date + self.end_date) / 2
        df = discount_curve(mid_time)

        return self.forward_dividends * df


@dataclass
class DividendSwap:
    """Dividend swap contract.

    A dividend swap exchanges fixed dividends for realized dividends.

    Attributes:
        notional: Notional amount
        fixed_rate: Fixed dividend rate
        maturity: Swap maturity
        underlying: Underlying stock/index
    """

    notional: float
    fixed_rate: float
    maturity: float
    underlying: str

    def fair_strike(self, dividend_forecast: DividendForecast) -> float:
        """Compute fair strike (fixed rate) for dividend swap.

        Args:
            dividend_forecast: Dividend forecast model

        Returns:
            Fair strike rate
        """
        # Forecast dividends over swap life
        schedule = dividend_forecast.forecast(self.maturity)

        # Total expected dividends
        total_div = schedule.total_dividends_before(self.maturity)

        # Fair strike = total expected dividends / notional
        return total_div / self.notional


@dataclass
class DividendCurveBuilder:
    """Build dividend curves from market data.

    Calibrates dividend curves from:
    - Equity forwards
    - Dividend futures
    - Analyst forecasts
    - Historical patterns
    """

    spot_price: float
    forward_prices: Dict[float, float]  # tenor -> forward price
    discount_curve: Callable[[float], float]
    dividend_estimates: Optional[Dict[float, float]] = None

    def build_yield_curve(self) -> DividendYieldCurve:
        """Build implied dividend yield curve from forward prices.

        Forward price: F(T) = S exp((r - q) T)
        Implied yield: q(T) = r - (1/T) log(F/S)

        Returns:
            Implied dividend yield curve
        """
        tenors = sorted(self.forward_prices.keys())
        yields = []

        for T in tenors:
            F = self.forward_prices[T]

            # Get risk-free rate from discount curve
            df = self.discount_curve(T)
            r = -jnp.log(df) / T

            # Implied dividend yield
            q = r - (1.0 / T) * jnp.log(F / self.spot_price)
            yields.append(float(q))

        return DividendYieldCurve(jnp.array(tenors), jnp.array(yields))

    def build_discrete_schedule(
        self, forecast_model: Optional[DividendForecast] = None
    ) -> DividendSchedule:
        """Build discrete dividend schedule.

        Args:
            forecast_model: Optional dividend forecast model

        Returns:
            Discrete dividend schedule
        """
        if forecast_model is not None:
            # Use forecast model
            max_tenor = max(self.forward_prices.keys())
            return forecast_model.forecast(max_tenor)

        elif self.dividend_estimates is not None:
            # Use provided estimates
            times = sorted(self.dividend_estimates.keys())
            amounts = [self.dividend_estimates[t] for t in times]
            return DividendSchedule(jnp.array(times), jnp.array(amounts), DividendType.CASH)

        else:
            # Imply from forwards
            # Use simple heuristic: distribute implied dividends uniformly
            yield_curve = self.build_yield_curve()

            times = []
            amounts = []

            for T in sorted(self.forward_prices.keys()):
                q = yield_curve.get_yield(T)
                # Approximate dividend: D ≈ S × q × T
                div = self.spot_price * q * 0.25  # Quarterly approximation
                for i in range(1, int(T * 4) + 1):
                    t = i * 0.25
                    if t <= T:
                        times.append(t)
                        amounts.append(div)

            return DividendSchedule(jnp.array(times), jnp.array(amounts), DividendType.CASH)


def gordon_growth_model(div_current: float, growth_rate: float, required_return: float) -> float:
    """Gordon Growth Model for stock valuation.

    P = D₁ / (r - g)

    where D₁ is next year's dividend, r is required return, g is growth rate.

    Args:
        div_current: Current dividend
        growth_rate: Perpetual growth rate
        required_return: Required rate of return

    Returns:
        Intrinsic stock value
    """
    if required_return <= growth_rate:
        raise ValueError("Required return must exceed growth rate")

    div_next = div_current * (1 + growth_rate)
    return div_next / (required_return - growth_rate)


def dividend_discount_model(dividends: List[float], growth_rate: float, required_return: float) -> float:
    """Multi-period dividend discount model.

    P = Σ D_t / (1 + r)^t + Terminal Value / (1 + r)^n

    Args:
        dividends: Projected dividends for explicit forecast period
        growth_rate: Terminal growth rate
        required_return: Required rate of return

    Returns:
        Intrinsic stock value
    """
    pv = 0.0

    # PV of explicit forecast dividends
    for t, div in enumerate(dividends, start=1):
        pv += div / (1 + required_return) ** t

    # Terminal value (Gordon growth model)
    n = len(dividends)
    terminal_div = dividends[-1] * (1 + growth_rate)
    terminal_value = terminal_div / (required_return - growth_rate)
    pv += terminal_value / (1 + required_return) ** n

    return pv
