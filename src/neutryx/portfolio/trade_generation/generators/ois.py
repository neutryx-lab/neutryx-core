"""
Overnight Index Swap (OIS) Generator

High-level API for generating Overnight Index Swap trades using market conventions.
Supports swaps based on overnight rates such as SOFR, ESTR, SONIA, and TONAR.
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
    OvernightIndexSwap,
    SwapType,
    DayCount,
)
from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import DayCountConvention


class OISGenerator:
    """
    Generator for Overnight Index Swap trades

    This class provides a high-level interface for creating OIS trades
    that conform to market conventions. It wraps the TradeFactory and
    provides OIS-specific convenience methods.
    """

    def __init__(self, factory: Optional[TradeFactory] = None):
        """
        Initialize OIS generator

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

    def _create_ois_product(
        self,
        notional: float,
        fixed_rate: float,
        maturity_years: float,
        swap_type: str,
        payment_frequency: int,
        day_count: DayCountConvention,
    ) -> OvernightIndexSwap:
        """Create OvernightIndexSwap product instance"""
        return OvernightIndexSwap(
            T=maturity_years,
            notional=notional,
            fixed_rate=fixed_rate,
            swap_type=self._map_swap_type(swap_type),
            payment_frequency=payment_frequency,
            day_count=self._map_day_count(day_count),
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
        Generate an Overnight Index Swap trade

        Args:
            currency: Currency code (e.g., "USD" for SOFR, "EUR" for ESTR)
            trade_date: Trade execution date
            tenor: Tenor string (e.g., "1Y", "2Y", "5Y")
            notional: Notional amount
            fixed_rate: Fixed rate (e.g., 0.043 for 4.3%)
            counterparty_id: Counterparty identifier
            swap_type: "PAYER" (pay fixed) or "RECEIVER" (receive fixed)

            # Convention overrides (optional)
            fixed_leg_frequency: Override fixed leg payment frequency (typically ANNUAL)
            floating_leg_frequency: Override floating leg payment frequency (typically ANNUAL)
            fixed_leg_day_count: Override fixed leg day count (typically ACT/360)
            floating_leg_day_count: Override floating leg day count (typically ACT/360)

            # Trade metadata (optional)
            trade_number: External trade number
            book_id: Book identifier
            desk_id: Desk identifier
            trader_id: Trader identifier

        Returns:
            TradeGenerationResult containing Trade, Product, Schedules, and Warnings

        Example:
            >>> generator = OISGenerator()
            >>> result = generator.generate(
            ...     currency="USD",
            ...     trade_date=date(2024, 1, 15),
            ...     tenor="2Y",
            ...     notional=20_000_000,
            ...     fixed_rate=0.043,
            ...     counterparty_id="CP-002",
            ...     swap_type="PAYER",
            ... )
            >>> print(result.trade)
            >>> print(f"Floating index: {result.product['floating_leg']['rate_index'].name}")
        """
        # Create request
        request = TradeGenerationRequest(
            currency=currency,
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            trade_date=trade_date,
            tenor=tenor,
            notional=notional,
            counterparty_id=counterparty_id,
            fixed_rate=fixed_rate,
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

        # Create OvernightIndexSwap product instance
        fixed_leg = result.product["fixed_leg"]
        maturity_years = (result.trade.maturity_date - result.trade.effective_date).days / 365.25

        ois_product = self._create_ois_product(
            notional=notional,
            fixed_rate=fixed_rate,
            maturity_years=maturity_years,
            swap_type=swap_type,
            payment_frequency=fixed_leg["frequency"].value,  # Frequency enum value is payments per year
            day_count=fixed_leg["day_count"],
        )

        # Store product in result
        result.product = ois_product

        # Also store in trade product_details for serialization
        result.trade.product_details["ois_product"] = {
            "T": ois_product.T,
            "notional": ois_product.notional,
            "fixed_rate": ois_product.fixed_rate,
            "swap_type": ois_product.swap_type.value,
            "payment_frequency": ois_product.payment_frequency,
            "day_count": ois_product.day_count.value,
        }

        return result


def generate_ois_trade(
    currency: str,
    trade_date: date,
    tenor: str,
    notional: float,
    fixed_rate: float,
    counterparty_id: str,
    swap_type: str = "PAYER",
    **kwargs
) -> Tuple[Trade, OvernightIndexSwap, TradeGenerationResult]:
    """
    Convenience function to generate an OIS trade

    This is a simplified interface that returns the Trade, Product, and full result.

    Args:
        currency: Currency code (e.g., "USD" for SOFR OIS)
        trade_date: Trade execution date
        tenor: Tenor string (e.g., "2Y")
        notional: Notional amount
        fixed_rate: Fixed rate
        counterparty_id: Counterparty ID
        swap_type: "PAYER" or "RECEIVER"
        **kwargs: Additional parameters (see OISGenerator.generate)

    Returns:
        Tuple of (Trade, OvernightIndexSwap, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_ois_trade(
        ...     "USD",
        ...     date(2024, 1, 15),
        ...     "2Y",
        ...     20_000_000,
        ...     0.043,
        ...     "CP-002",
        ... )
        >>> print(f"Trade ID: {trade.id}")
        >>> print(f"Overnight rate: {product.overnight_rates[0]:.4%}")
        >>> if result.has_warnings():
        ...     print("Warnings:")
        ...     for warning in result.get_warnings():
        ...         print(f"  - {warning}")
    """
    generator = OISGenerator()
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
