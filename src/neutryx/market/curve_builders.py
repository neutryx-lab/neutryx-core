"""
Currency-specific curve builders for major markets.

This module provides pre-configured curve builders for major currencies,
handling market conventions and instrument selection automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from neutryx.market.curves import (
    BootstrappedCurve,
    Deposit,
    FRA,
    Future,
    OIS,
    FixedRateSwap,
    TenorBasisSwap,
    MarketInstrument,
)
from neutryx.market.multi_curve import (
    CurveDefinition,
    MultiCurveBuilder,
    MultiCurveEnvironment,
)
from neutryx.market.rate_indices import (
    get_rfr_index,
    get_swap_convention,
    SOFR,
    ESTR,
    SONIA,
    TONAR,
)


@dataclass
class CurveMarketData:
    """
    Market data for curve construction.

    Attributes:
        reference_date: Valuation date
        deposits: Deposit rates by maturity (in years)
        fras: FRA rates by (start, end) in years
        futures: Future prices by (start, end) in years
        ois_rates: OIS swap rates by maturity (in years)
        swap_rates: Standard swap rates by maturity (in years)
        basis_spreads: Tenor basis spreads by (short_tenor, long_tenor, maturity)
    """

    reference_date: date
    deposits: Dict[float, float]
    fras: Dict[tuple[float, float], float] = None
    futures: Dict[tuple[float, float], float] = None
    ois_rates: Dict[float, float] = None
    swap_rates: Dict[float, float] = None
    basis_spreads: Dict[tuple[str, str, float], float] = None

    def __post_init__(self):
        if self.fras is None:
            self.fras = {}
        if self.futures is None:
            self.futures = {}
        if self.ois_rates is None:
            self.ois_rates = {}
        if self.swap_rates is None:
            self.swap_rates = {}
        if self.basis_spreads is None:
            self.basis_spreads = {}


class SOFRCurveBuilder:
    """
    Builder for USD SOFR discount curve.

    Constructs the USD risk-free rate curve using:
    - Short end: SOFR deposit rates (ON, 1W, 1M, 3M)
    - Medium term: SOFR futures (CME SOFR futures)
    - Long end: SOFR OIS swaps (1Y-30Y)
    """

    @staticmethod
    def build(market_data: CurveMarketData) -> BootstrappedCurve:
        """
        Build SOFR discount curve from market data.

        Args:
            market_data: Market rates and prices

        Returns:
            Bootstrapped SOFR curve
        """
        instruments: List[MarketInstrument] = []

        # Add deposits (short end: ON to 3M)
        for maturity, rate in sorted(market_data.deposits.items()):
            if maturity <= 0.25:  # Up to 3 months
                instruments.append(Deposit(maturity=maturity, rate=rate))

        # Add SOFR futures (medium term: 3M to 2Y)
        for (start, end), price in sorted(market_data.futures.items()):
            if 0.25 <= start <= 2.0:
                # SOFR futures trade like Eurodollar futures
                # Price = 100 - implied rate
                instruments.append(
                    Future(start=start, end=end, price=price, convexity_adjustment=0.0)
                )

        # Add SOFR OIS (long end: 1Y to 30Y)
        convention = get_swap_convention("USD")
        for maturity, rate in sorted(market_data.ois_rates.items()):
            if maturity >= 1.0:
                # Generate payment schedule based on convention
                n_payments = int(maturity * convention.fixed_leg_frequency)
                payment_times = [
                    i / convention.fixed_leg_frequency
                    for i in range(1, n_payments + 1)
                ]
                accrual_factors = [1.0 / convention.fixed_leg_frequency] * n_payments

                instruments.append(
                    OIS(
                        fixed_rate=rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                        compounding="compound",
                    )
                )

        return BootstrappedCurve(instruments)


class ESTRCurveBuilder:
    """
    Builder for EUR ESTR (€STR) discount curve.

    Constructs the EUR risk-free rate curve using:
    - Short end: ESTR deposit rates
    - Medium term: ESTR futures (Eurex €STR futures)
    - Long end: ESTR OIS swaps
    """

    @staticmethod
    def build(market_data: CurveMarketData) -> BootstrappedCurve:
        """
        Build ESTR discount curve from market data.

        Args:
            market_data: Market rates and prices

        Returns:
            Bootstrapped ESTR curve
        """
        instruments: List[MarketInstrument] = []

        # Add deposits (short end)
        for maturity, rate in sorted(market_data.deposits.items()):
            if maturity <= 0.25:
                instruments.append(Deposit(maturity=maturity, rate=rate))

        # Add ESTR futures (medium term)
        for (start, end), price in sorted(market_data.futures.items()):
            if 0.25 <= start <= 2.0:
                instruments.append(
                    Future(start=start, end=end, price=price, convexity_adjustment=0.0)
                )

        # Add ESTR OIS (long end)
        convention = get_swap_convention("EUR")
        for maturity, rate in sorted(market_data.ois_rates.items()):
            if maturity >= 1.0:
                # EUR OIS typically annual fixed vs daily ESTR
                n_payments = int(maturity * convention.fixed_leg_frequency)
                payment_times = [
                    i / convention.fixed_leg_frequency
                    for i in range(1, n_payments + 1)
                ]
                accrual_factors = [1.0 / convention.fixed_leg_frequency] * n_payments

                instruments.append(
                    OIS(
                        fixed_rate=rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                        compounding="compound",
                    )
                )

        return BootstrappedCurve(instruments)


class EONIACurveBuilder:
    """
    Builder for EUR EONIA discount curve (legacy).

    EONIA was replaced by ESTR in 2022, but legacy curves may still use it.
    EONIA = ESTR - 8.5 bps (fixed spread during transition).
    """

    @staticmethod
    def build(market_data: CurveMarketData) -> BootstrappedCurve:
        """
        Build EONIA discount curve from market data.

        Args:
            market_data: Market rates and prices

        Returns:
            Bootstrapped EONIA curve
        """
        # EONIA curve construction is similar to ESTR
        # Apply fixed spread adjustment: EONIA = ESTR - 8.5bp
        spread_adjustment = -0.000085

        instruments: List[MarketInstrument] = []

        # Add deposits with spread adjustment
        for maturity, rate in sorted(market_data.deposits.items()):
            if maturity <= 0.25:
                adjusted_rate = rate + spread_adjustment
                instruments.append(Deposit(maturity=maturity, rate=adjusted_rate))

        # Add EONIA OIS (long end)
        convention = get_swap_convention("EUR")
        for maturity, rate in sorted(market_data.ois_rates.items()):
            if maturity >= 1.0:
                adjusted_rate = rate + spread_adjustment
                n_payments = int(maturity * convention.fixed_leg_frequency)
                payment_times = [
                    i / convention.fixed_leg_frequency
                    for i in range(1, n_payments + 1)
                ]
                accrual_factors = [1.0 / convention.fixed_leg_frequency] * n_payments

                instruments.append(
                    OIS(
                        fixed_rate=adjusted_rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                        compounding="compound",
                    )
                )

        return BootstrappedCurve(instruments)


class SONIACurveBuilder:
    """
    Builder for GBP SONIA discount curve.

    Constructs the GBP risk-free rate curve using:
    - Short end: SONIA deposit rates
    - Medium term: Short Sterling futures (SOFR-style)
    - Long end: SONIA OIS swaps
    """

    @staticmethod
    def build(market_data: CurveMarketData) -> BootstrappedCurve:
        """
        Build SONIA discount curve from market data.

        Args:
            market_data: Market rates and prices

        Returns:
            Bootstrapped SONIA curve
        """
        instruments: List[MarketInstrument] = []

        # Add deposits (short end)
        for maturity, rate in sorted(market_data.deposits.items()):
            if maturity <= 0.25:
                instruments.append(Deposit(maturity=maturity, rate=rate))

        # Add Short Sterling futures (medium term)
        for (start, end), price in sorted(market_data.futures.items()):
            if 0.25 <= start <= 2.0:
                instruments.append(
                    Future(start=start, end=end, price=price, convexity_adjustment=0.0)
                )

        # Add SONIA OIS (long end)
        convention = get_swap_convention("GBP")
        for maturity, rate in sorted(market_data.ois_rates.items()):
            if maturity >= 1.0:
                # GBP OIS typically annual fixed vs daily SONIA
                n_payments = int(maturity * convention.fixed_leg_frequency)
                payment_times = [
                    i / convention.fixed_leg_frequency
                    for i in range(1, n_payments + 1)
                ]
                accrual_factors = [1.0 / convention.fixed_leg_frequency] * n_payments

                instruments.append(
                    OIS(
                        fixed_rate=rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                        compounding="compound",
                    )
                )

        return BootstrappedCurve(instruments)


class TONARCurveBuilder:
    """
    Builder for JPY TONAR (Tokyo Overnight Average Rate) discount curve.

    Constructs the JPY risk-free rate curve using:
    - Short end: TONAR deposit rates
    - Medium term: Euroyen futures
    - Long end: TONAR OIS swaps
    """

    @staticmethod
    def build(market_data: CurveMarketData) -> BootstrappedCurve:
        """
        Build TONAR discount curve from market data.

        Args:
            market_data: Market rates and prices

        Returns:
            Bootstrapped TONAR curve
        """
        instruments: List[MarketInstrument] = []

        # Add deposits (short end)
        for maturity, rate in sorted(market_data.deposits.items()):
            if maturity <= 0.25:
                instruments.append(Deposit(maturity=maturity, rate=rate))

        # Add Euroyen futures (medium term)
        for (start, end), price in sorted(market_data.futures.items()):
            if 0.25 <= start <= 2.0:
                instruments.append(
                    Future(start=start, end=end, price=price, convexity_adjustment=0.0)
                )

        # Add TONAR OIS (long end)
        convention = get_swap_convention("JPY")
        for maturity, rate in sorted(market_data.ois_rates.items()):
            if maturity >= 1.0:
                n_payments = int(maturity * convention.fixed_leg_frequency)
                payment_times = [
                    i / convention.fixed_leg_frequency
                    for i in range(1, n_payments + 1)
                ]
                accrual_factors = [1.0 / convention.fixed_leg_frequency] * n_payments

                instruments.append(
                    OIS(
                        fixed_rate=rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                        compounding="compound",
                    )
                )

        return BootstrappedCurve(instruments)


def build_multi_curve_environment(
    currency: str,
    market_data: CurveMarketData,
    tenors: Optional[List[str]] = None,
) -> MultiCurveEnvironment:
    """
    Build a complete multi-curve environment for a currency.

    Constructs:
    1. Risk-free discount curve (OIS/RFR)
    2. Projection curves for specified tenors (e.g., 3M, 6M)

    Args:
        currency: Currency code (USD, EUR, GBP, JPY)
        market_data: Market rates and prices
        tenors: List of tenors for projection curves (default: ["3M"])

    Returns:
        MultiCurveEnvironment with discount and projection curves
    """
    if tenors is None:
        tenors = ["3M"]

    builder = MultiCurveBuilder()

    # Build discount curve based on currency
    rfr_builders = {
        "USD": SOFRCurveBuilder,
        "EUR": ESTRCurveBuilder,
        "GBP": SONIACurveBuilder,
        "JPY": TONARCurveBuilder,
    }

    if currency not in rfr_builders:
        raise ValueError(f"Unsupported currency: {currency}")

    rfr_builder_class = rfr_builders[currency]
    rfr_curve = rfr_builder_class.build(market_data)

    # Add discount curve to multi-curve builder
    # Create instruments list from the bootstrapped curve
    discount_instruments: List[MarketInstrument] = []

    # Add deposits
    for maturity, rate in sorted(market_data.deposits.items()):
        if maturity <= 0.25:
            discount_instruments.append(Deposit(maturity=maturity, rate=rate))

    # Add OIS swaps
    convention = get_swap_convention(currency)
    for maturity, rate in sorted(market_data.ois_rates.items()):
        if maturity >= 1.0:
            n_payments = int(maturity * convention.fixed_leg_frequency)
            payment_times = [
                i / convention.fixed_leg_frequency for i in range(1, n_payments + 1)
            ]
            accrual_factors = [1.0 / convention.fixed_leg_frequency] * n_payments

            discount_instruments.append(
                OIS(
                    fixed_rate=rate,
                    payment_times=payment_times,
                    accrual_factors=accrual_factors,
                    compounding="compound",
                )
            )

    builder.add_curve_definition(
        CurveDefinition(
            name=f"{currency}-OIS",
            currency=currency,
            curve_type="discount",
            instruments=discount_instruments,
        )
    )

    # Build projection curves for each tenor
    for tenor in tenors:
        projection_instruments: List[MarketInstrument] = []

        # Add FRAs for short end
        for (start, end), rate in sorted(market_data.fras.items()):
            projection_instruments.append(FRA(start=start, end=end, rate=rate))

        # Add futures for medium term
        for (start, end), price in sorted(market_data.futures.items()):
            projection_instruments.append(
                Future(start=start, end=end, price=price, convexity_adjustment=0.0)
            )

        # Add tenor swaps for long end
        for maturity, rate in sorted(market_data.swap_rates.items()):
            if maturity >= 1.0:
                n_payments = int(maturity * convention.float_leg_frequency)
                payment_times = [
                    i / convention.float_leg_frequency
                    for i in range(1, n_payments + 1)
                ]
                accrual_factors = [1.0 / convention.float_leg_frequency] * n_payments

                projection_instruments.append(
                    FixedRateSwap(
                        fixed_rate=rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                    )
                )

        if projection_instruments:
            builder.add_curve_definition(
                CurveDefinition(
                    name=f"{currency}-{tenor}",
                    currency=currency,
                    curve_type="projection",
                    tenor=tenor,
                    instruments=projection_instruments,
                    depends_on=[f"{currency}-OIS"],
                )
            )

    return builder.build()
