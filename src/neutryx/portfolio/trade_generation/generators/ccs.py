"""
Cross-Currency Swap (CCS) Generator

High-level API for generating Cross-Currency Swap trades using market conventions.
CCS swaps exchange cash flows in two different currencies, typically involving:
- Exchange of notional at inception and maturity
- Periodic interest payments in both currencies
- Optional FX reset mechanism
"""

from datetime import date, timedelta
from typing import Optional, Tuple

import jax.numpy as jnp

from neutryx.portfolio.contracts.trade import Trade
from neutryx.portfolio.trade_generation.factory import (
    TradeFactory,
    TradeGenerationRequest,
    TradeGenerationResult,
)
from neutryx.market.convention_profiles import ProductTypeConvention
from neutryx.products.linear_rates.swaps import CrossCurrencySwap
from neutryx.core.dates.schedule import Frequency


class CCSGenerator:
    """
    Generator for Cross-Currency Swap trades

    This class provides a high-level interface for creating cross-currency swap
    trades that conform to market conventions. CCS exchanges cash flows in two
    different currencies.
    """

    def __init__(self, factory: Optional[TradeFactory] = None):
        """
        Initialize CCS generator

        Args:
            factory: Optional TradeFactory instance (creates default if not provided)
        """
        self.factory = factory or TradeFactory()

    def _frequency_to_payments_per_year(self, frequency: Frequency) -> int:
        """
        Convert Frequency to payment frequency (payments per year)

        Args:
            frequency: Frequency enum value

        Returns:
            Payment frequency as integer (payments per year)
        """
        mapping = {
            Frequency.ANNUAL: 1,
            Frequency.SEMI_ANNUAL: 2,
            Frequency.QUARTERLY: 4,
            Frequency.MONTHLY: 12,
        }

        return mapping.get(frequency, 2)  # Default to semi-annual

    def _create_ccs_product(
        self,
        notional_domestic: float,
        notional_foreign: float,
        domestic_rate: float,
        foreign_rate: float,
        fx_spot: float,
        maturity_years: float,
        fx_reset: bool,
        payment_frequency: int,
    ) -> CrossCurrencySwap:
        """Create CrossCurrencySwap product instance"""
        return CrossCurrencySwap(
            T=maturity_years,
            notional_domestic=notional_domestic,
            notional_foreign=notional_foreign,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            fx_spot=fx_spot,
            fx_reset=fx_reset,
            payment_frequency=payment_frequency,
        )

    def generate(
        self,
        currency_pair: str,  # e.g., "USDEUR", "USDJPY"
        trade_date: date,
        maturity_years: float,
        notional_domestic: float,
        notional_foreign: float,
        domestic_rate: float,
        foreign_rate: float,
        fx_spot: float,
        counterparty_id: str,
        fx_reset: bool = True,
        # Trade metadata
        trade_number: Optional[str] = None,
        book_id: Optional[str] = None,
        desk_id: Optional[str] = None,
        trader_id: Optional[str] = None,
    ) -> TradeGenerationResult:
        """
        Generate a Cross-Currency Swap trade

        Args:
            currency_pair: Currency pair code (e.g., "USDEUR" for USD/EUR)
            trade_date: Trade execution date
            maturity_years: Maturity in years
            notional_domestic: Notional amount in domestic currency
            notional_foreign: Notional amount in foreign currency
            domestic_rate: Interest rate for domestic currency leg
            foreign_rate: Interest rate for foreign currency leg
            fx_spot: Current FX spot rate (domestic per foreign)
            counterparty_id: Counterparty identifier
            fx_reset: Whether to reset FX at each payment (default True)

            # Trade metadata (optional)
            trade_number: External trade number
            book_id: Book identifier
            desk_id: Desk identifier
            trader_id: Trader identifier

        Returns:
            TradeGenerationResult containing Trade, Product, and Warnings

        Example:
            >>> generator = CCSGenerator()
            >>> result = generator.generate(
            ...     currency_pair="USDEUR",
            ...     trade_date=date(2024, 1, 15),
            ...     maturity_years=5.0,
            ...     notional_domestic=100_000_000,  # USD
            ...     notional_foreign=90_000_000,    # EUR
            ...     domestic_rate=0.04,  # 4% USD
            ...     foreign_rate=0.03,   # 3% EUR
            ...     fx_spot=1.11,  # USD/EUR rate
            ...     counterparty_id="CP-001",
            ... )
            >>> print(result.trade)
        """
        # Extract currencies from pair (e.g., "USDEUR" -> "USD", "EUR")
        # Validate format first before checking profile
        if len(currency_pair) != 6:
            raise ValueError(
                f"Invalid currency pair format: '{currency_pair}'. "
                "Expected 6-character format like 'USDEUR', 'USDJPY'"
            )

        domestic_currency = currency_pair[:3]
        foreign_currency = currency_pair[3:6]

        # Get convention profile for CCS
        from neutryx.market.convention_profiles import get_convention_profile
        profile = get_convention_profile(currency_pair, ProductTypeConvention.CROSS_CURRENCY_SWAP)

        if profile is None:
            raise ValueError(f"No Cross-Currency Swap convention profile found for {currency_pair}")

        # Payment frequency (use the higher frequency leg)
        domestic_freq = profile.fixed_leg.frequency if profile.fixed_leg else Frequency.SEMI_ANNUAL
        foreign_freq = profile.floating_leg.frequency if profile.floating_leg else Frequency.ANNUAL

        payment_frequency = max(
            self._frequency_to_payments_per_year(domestic_freq),
            self._frequency_to_payments_per_year(foreign_freq),
        )

        # Calculate dates
        effective_date = trade_date + timedelta(days=profile.spot_lag)
        maturity_date = effective_date + timedelta(days=int(maturity_years * 365))

        # Create CCS product
        ccs_product = self._create_ccs_product(
            notional_domestic=notional_domestic,
            notional_foreign=notional_foreign,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            fx_spot=fx_spot,
            maturity_years=maturity_years,
            fx_reset=fx_reset,
            payment_frequency=payment_frequency,
        )

        # Create Trade
        from neutryx.portfolio.contracts.trade import (
            Trade,
            ProductType,
            TradeStatus,
            SettlementType as TradeSettlementType,
        )
        from uuid import uuid4

        trade = Trade(
            id=str(uuid4()),
            trade_number=trade_number,
            counterparty_id=counterparty_id,
            book_id=book_id,
            desk_id=desk_id,
            trader_id=trader_id,
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=trade_date,
            effective_date=effective_date,
            maturity_date=maturity_date,
            status=TradeStatus.ACTIVE,
            notional=notional_domestic,  # Use domestic notional as primary
            currency=domestic_currency,
            settlement_type=TradeSettlementType.PHYSICAL,  # CCS typically involves notional exchange
            product_details={
                "product_subtype": "CROSS_CURRENCY_SWAP",
                "currency_pair": currency_pair,
                "domestic_currency": domestic_currency,
                "foreign_currency": foreign_currency,
                "notional_domestic": notional_domestic,
                "notional_foreign": notional_foreign,
                "domestic_rate": domestic_rate,
                "foreign_rate": foreign_rate,
                "fx_spot": fx_spot,
                "fx_reset": fx_reset,
                "payment_frequency": payment_frequency,
                "maturity_years": maturity_years,
                "ccs_product": {
                    "T": ccs_product.T,
                    "notional_domestic": ccs_product.notional_domestic,
                    "notional_foreign": ccs_product.notional_foreign,
                    "domestic_rate": ccs_product.domestic_rate,
                    "foreign_rate": ccs_product.foreign_rate,
                    "fx_spot": ccs_product.fx_spot,
                    "fx_reset": ccs_product.fx_reset,
                    "payment_frequency": ccs_product.payment_frequency,
                },
            },
            convention_profile_id=profile.get_profile_id(),
            generated_from_convention=True,
        )

        # Create validation result
        from neutryx.portfolio.trade_generation.validation import ValidationResult
        validation_result = ValidationResult(convention_profile=profile)

        # Create result
        result = TradeGenerationResult(
            trade=trade,
            product=ccs_product,
            schedules=None,  # Could add schedule generation in future
            validation_result=validation_result,
            convention_profile=profile,
        )

        return result


def generate_ccs_trade(
    currency_pair: str,
    trade_date: date,
    maturity_years: float,
    notional_domestic: float,
    notional_foreign: float,
    domestic_rate: float,
    foreign_rate: float,
    fx_spot: float,
    counterparty_id: str,
    fx_reset: bool = True,
    **kwargs
) -> Tuple[Trade, CrossCurrencySwap, TradeGenerationResult]:
    """
    Convenience function to generate a Cross-Currency Swap trade

    This is a simplified interface that returns the Trade, Product, and full result.

    Args:
        currency_pair: Currency pair (e.g., "USDEUR")
        trade_date: Trade execution date
        maturity_years: Maturity in years
        notional_domestic: Notional in domestic currency
        notional_foreign: Notional in foreign currency
        domestic_rate: Interest rate for domestic currency
        foreign_rate: Interest rate for foreign currency
        fx_spot: FX spot rate
        counterparty_id: Counterparty ID
        fx_reset: Whether to reset FX at each payment
        **kwargs: Additional parameters (see CCSGenerator.generate)

    Returns:
        Tuple of (Trade, CrossCurrencySwap, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_ccs_trade(
        ...     "USDEUR",
        ...     date(2024, 1, 15),
        ...     5.0,  # 5 years
        ...     100_000_000,  # 100M USD
        ...     90_000_000,   # 90M EUR
        ...     0.04,  # 4% USD
        ...     0.03,  # 3% EUR
        ...     1.11,  # USD/EUR
        ...     "CP-001",
        ... )
        >>> print(f"Trade ID: {trade.id}")
        >>> print(f"FX Reset: {product.fx_reset}")
    """
    generator = CCSGenerator()
    result = generator.generate(
        currency_pair=currency_pair,
        trade_date=trade_date,
        maturity_years=maturity_years,
        notional_domestic=notional_domestic,
        notional_foreign=notional_foreign,
        domestic_rate=domestic_rate,
        foreign_rate=foreign_rate,
        fx_spot=fx_spot,
        counterparty_id=counterparty_id,
        fx_reset=fx_reset,
        **kwargs
    )
    return result.trade, result.product, result
