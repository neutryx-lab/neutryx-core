"""
Interest Rate Swap (IRS) Generator

High-level API for generating Interest Rate Swap trades using market conventions.
Supports both vanilla fixed-floating swaps and provides integration with the
InterestRateSwap product class for pricing.
"""

from datetime import date
from typing import Optional, Tuple

import jax.numpy as jnp

from neutryx.portfolio.contracts.trade import Trade
from neutryx.portfolio.trade_generation.factory import (
    TradeFactory,
    TradeGenerationRequest,
    TradeGenerationResult,
)
from neutryx.market.convention_profiles import ProductTypeConvention
from neutryx.products.linear_rates.swaps import (
    InterestRateSwap,
    SwapType,
    DayCount,
)
from neutryx.core.dates.schedule import Frequency, Schedule
from neutryx.core.dates.day_count import (
    DayCountConvention,
    Actual360,
    Actual365Fixed,
    ActualActual,
    Thirty360,
)


class IRSGenerator:
    """
    Generator for Interest Rate Swap trades

    This class provides a high-level interface for creating IRS trades
    that conform to market conventions. It wraps the TradeFactory and
    provides IRS-specific convenience methods.
    """

    def __init__(self, factory: Optional[TradeFactory] = None):
        """
        Initialize IRS generator

        Args:
            factory: Optional TradeFactory instance (creates default if not provided)
        """
        self.factory = factory or TradeFactory()

    def _map_day_count(self, day_count_conv: DayCountConvention) -> DayCount:
        """Map DayCountConvention to product DayCount enum"""
        mapping = {
            DayCountConvention.ACT_360: DayCount.ACT_360,
            DayCountConvention.ACT_365: DayCount.ACT_365,
            DayCountConvention.ACT_ACT: DayCount.ACT_ACT,
            DayCountConvention.THIRTY_360: DayCount.THIRTY_360,
        }
        return mapping.get(day_count_conv, DayCount.ACT_360)

    def _map_swap_type(self, swap_type_str: str) -> SwapType:
        """Map string swap type to SwapType enum"""
        if swap_type_str.upper() == "PAYER":
            return SwapType.PAYER
        elif swap_type_str.upper() == "RECEIVER":
            return SwapType.RECEIVER
        else:
            raise ValueError(f"Invalid swap type: {swap_type_str}. Must be 'PAYER' or 'RECEIVER'")

    def _create_irs_product(
        self,
        notional: float,
        fixed_rate: float,
        maturity_years: float,
        swap_type: str,
        payment_frequency: int,
        day_count: DayCountConvention,
        spread: float = 0.0,
    ) -> InterestRateSwap:
        """Create InterestRateSwap product instance"""
        return InterestRateSwap(
            T=maturity_years,
            notional=notional,
            fixed_rate=fixed_rate,
            swap_type=self._map_swap_type(swap_type),
            payment_frequency=payment_frequency,
            day_count=self._map_day_count(day_count),
            spread=spread,
        )

    def generate(
        self,
        currency: str,
        trade_date: date,
        tenor: str,
        notional: float,
        fixed_rate: float,
        counterparty_id: str,
        swap_type: str = "PAYER",
        spread: float = 0.0,
        # Convention overrides
        fixed_leg_frequency: Optional[Frequency] = None,
        floating_leg_frequency: Optional[Frequency] = None,
        fixed_leg_day_count: Optional[DayCountConvention] = None,
        floating_leg_day_count: Optional[DayCountConvention] = None,
        # Trade metadata
        trade_number: Optional[str] = None,
        book_id: Optional[str] = None,
        desk_id: Optional[str] = None,
        trader_id: Optional[str] = None,
    ) -> TradeGenerationResult:
        """
        Generate an Interest Rate Swap trade

        Args:
            currency: Currency code (e.g., "USD", "EUR")
            trade_date: Trade execution date
            tenor: Tenor string (e.g., "5Y", "10Y")
            notional: Notional amount
            fixed_rate: Fixed rate (e.g., 0.045 for 4.5%)
            counterparty_id: Counterparty identifier
            swap_type: "PAYER" (pay fixed) or "RECEIVER" (receive fixed)
            spread: Spread over floating index (default 0.0)

            # Convention overrides (optional)
            fixed_leg_frequency: Override fixed leg payment frequency
            floating_leg_frequency: Override floating leg payment frequency
            fixed_leg_day_count: Override fixed leg day count
            floating_leg_day_count: Override floating leg day count

            # Trade metadata (optional)
            trade_number: External trade number
            book_id: Book identifier
            desk_id: Desk identifier
            trader_id: Trader identifier

        Returns:
            TradeGenerationResult containing Trade, Product, Schedules, and Warnings

        Example:
            >>> generator = IRSGenerator()
            >>> result = generator.generate(
            ...     currency="USD",
            ...     trade_date=date(2024, 1, 15),
            ...     tenor="5Y",
            ...     notional=10_000_000,
            ...     fixed_rate=0.045,
            ...     counterparty_id="CP-001",
            ...     swap_type="PAYER",
            ... )
            >>> print(result.trade)
            >>> print(result.has_warnings())
        """
        # Create request
        request = TradeGenerationRequest(
            currency=currency,
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            trade_date=trade_date,
            tenor=tenor,
            notional=notional,
            counterparty_id=counterparty_id,
            fixed_rate=fixed_rate,
            spread=spread,
            swap_type=swap_type,
            fixed_leg_frequency=fixed_leg_frequency,
            floating_leg_frequency=floating_leg_frequency,
            fixed_leg_day_count=fixed_leg_day_count,
            floating_leg_day_count=floating_leg_day_count,
            trade_number=trade_number,
            book_id=book_id,
            desk_id=desk_id,
            trader_id=trader_id,
        )

        # Generate trade using factory
        result = self.factory.generate_trade(request)

        # Create InterestRateSwap product instance
        fixed_leg = result.product["fixed_leg"]
        maturity_years = (result.trade.maturity_date - result.trade.effective_date).days / 365.25

        irs_product = self._create_irs_product(
            notional=notional,
            fixed_rate=fixed_rate,
            maturity_years=maturity_years,
            swap_type=swap_type,
            payment_frequency=fixed_leg["frequency"].value,  # Frequency enum value is payments per year
            day_count=fixed_leg["day_count"],
            spread=spread,
        )

        # Store product in result
        result.product = irs_product

        # Also store in trade product_details for serialization
        result.trade.product_details["irs_product"] = {
            "T": irs_product.T,
            "notional": irs_product.notional,
            "fixed_rate": irs_product.fixed_rate,
            "swap_type": irs_product.swap_type.value,
            "payment_frequency": irs_product.payment_frequency,
            "day_count": irs_product.day_count.value,
            "spread": irs_product.spread,
        }

        return result


def generate_irs_trade(
    currency: str,
    trade_date: date,
    tenor: str,
    notional: float,
    fixed_rate: float,
    counterparty_id: str,
    swap_type: str = "PAYER",
    **kwargs
) -> Tuple[Trade, InterestRateSwap, TradeGenerationResult]:
    """
    Convenience function to generate an IRS trade

    This is a simplified interface that returns the Trade, Product, and full result.

    Args:
        currency: Currency code
        trade_date: Trade execution date
        tenor: Tenor string (e.g., "5Y")
        notional: Notional amount
        fixed_rate: Fixed rate
        counterparty_id: Counterparty ID
        swap_type: "PAYER" or "RECEIVER"
        **kwargs: Additional parameters (see IRSGenerator.generate)

    Returns:
        Tuple of (Trade, InterestRateSwap, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_irs_trade(
        ...     "USD",
        ...     date(2024, 1, 15),
        ...     "5Y",
        ...     10_000_000,
        ...     0.045,
        ...     "CP-001",
        ... )
        >>> print(f"Trade ID: {trade.id}")
        >>> print(f"Par rate: {product.par_rate():.4%}")
        >>> if result.has_warnings():
        ...     print("Warnings:")
        ...     for warning in result.get_warnings():
        ...         print(f"  - {warning}")
    """
    generator = IRSGenerator()
    result = generator.generate(
        currency=currency,
        trade_date=trade_date,
        tenor=tenor,
        notional=notional,
        fixed_rate=fixed_rate,
        counterparty_id=counterparty_id,
        swap_type=swap_type,
        **kwargs
    )
    return result.trade, result.product, result
