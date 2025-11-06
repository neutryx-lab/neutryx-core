"""Pricing bridge connecting trades to pricing engines.

Extracts pricing parameters from trade objects and interfaces with pricing engines:
- Parameter extraction from Trade and product_details
- Batch pricing workflows
- Portfolio revaluation
- Market data integration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

from neutryx.api.rest import VanillaOptionRequest
from neutryx.portfolio.contracts.trade import ProductType, Trade
from neutryx.core.engine import MCConfig, price_vanilla_mc


@dataclass
class PricingResult:
    """Result of pricing a single trade."""

    trade_id: str
    price: float
    pricing_date: date
    success: bool
    error_message: Optional[str] = None
    greeks: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketData:
    """Market data snapshot for pricing."""

    pricing_date: date
    spot_prices: Dict[str, float] = field(default_factory=dict)  # instrument_id -> spot
    volatilities: Dict[str, float] = field(default_factory=dict)  # instrument_id -> vol
    interest_rates: Dict[str, float] = field(default_factory=dict)  # currency -> rate
    dividend_yields: Dict[str, float] = field(default_factory=dict)  # instrument_id -> yield
    fx_rates: Dict[str, float] = field(default_factory=dict)  # currency_pair -> rate
    discount_curves: Dict[str, List[float]] = field(default_factory=dict)  # currency -> curve


class PricingBridge:
    """Bridge between trade management and pricing engines.

    Example:
        >>> bridge = PricingBridge()
        >>> market_data = MarketData(
        ...     pricing_date=date.today(),
        ...     spot_prices={"AAPL": 150.0},
        ...     volatilities={"AAPL": 0.25},
        ...     interest_rates={"USD": 0.05}
        ... )
        >>> 
        >>> trade = Trade(
        ...     id="TRD-001",
        ...     product_type=ProductType.EQUITY_OPTION,
        ...     product_details={
        ...         "underlying": "AAPL",
        ...         "strike": 155.0,
        ...         "is_call": True,
        ...         ...
        ...     }
        ... )
        >>> 
        >>> result = bridge.price_trade(trade, market_data)
        >>> print(f"Price: ${result.price:,.2f}")
    """

    def __init__(self, mc_config: Optional[MCConfig] = None, seed: int = 42):
        """Initialize pricing bridge.

        Args:
            mc_config: Monte Carlo configuration
            seed: Random seed for reproducibility
        """
        self.mc_config = mc_config or MCConfig(steps=252, paths=100_000)
        self.seed = seed
        self._key = jax.random.PRNGKey(seed)

    def price_trade(self, trade: Trade, market_data: MarketData) -> PricingResult:
        """Price a single trade.

        Args:
            trade: Trade to price
            market_data: Market data snapshot

        Returns:
            Pricing result
        """
        try:
            if trade.product_type == ProductType.EQUITY_OPTION:
                price = self._price_equity_option(trade, market_data)
            elif trade.product_type == ProductType.FX_OPTION:
                price = self._price_fx_option(trade, market_data)
            elif trade.product_type == ProductType.INTEREST_RATE_SWAP:
                price = self._price_interest_rate_swap(trade, market_data)
            else:
                return PricingResult(
                    trade_id=trade.id,
                    price=0.0,
                    pricing_date=market_data.pricing_date,
                    success=False,
                    error_message=f"Unsupported product type: {trade.product_type}",
                )

            return PricingResult(
                trade_id=trade.id,
                price=float(price),
                pricing_date=market_data.pricing_date,
                success=True,
            )

        except Exception as e:
            return PricingResult(
                trade_id=trade.id,
                price=0.0,
                pricing_date=market_data.pricing_date,
                success=False,
                error_message=str(e),
            )

    def _price_equity_option(self, trade: Trade, market_data: MarketData) -> float:
        """Price an equity option."""
        if not trade.product_details:
            raise ValueError("Missing product_details for equity option")

        details = trade.product_details
        underlying = details.get("underlying")
        strike = details.get("strike")
        is_call = details.get("is_call", True)

        if not underlying or strike is None:
            raise ValueError("Missing underlying or strike in product_details")

        # Get market data
        spot = market_data.spot_prices.get(underlying)
        vol = market_data.volatilities.get(underlying)
        rate = market_data.interest_rates.get(trade.currency or "USD", 0.05)
        dividend = market_data.dividend_yields.get(underlying, 0.0)

        if spot is None or vol is None:
            raise ValueError(f"Missing market data for {underlying}")

        # Calculate maturity
        if not trade.maturity_date:
            raise ValueError("Missing maturity_date")

        maturity = (trade.maturity_date - market_data.pricing_date).days / 365.0
        if maturity <= 0:
            return max(0.0, (spot - strike) if is_call else (strike - spot))

        # Price using Monte Carlo
        self._key, subkey = jax.random.split(self._key)
        price = price_vanilla_mc(
            subkey,
            spot,
            strike,
            maturity,
            rate,
            dividend,
            vol,
            self.mc_config,
            is_call=is_call,
        )

        return float(price)

    def _price_fx_option(self, trade: Trade, market_data: MarketData) -> float:
        """Price an FX option."""
        if not trade.product_details:
            raise ValueError("Missing product_details for FX option")

        details = trade.product_details
        currency_pair = details.get("currency_pair")  # e.g., "EURUSD"
        strike = details.get("strike")
        is_call = details.get("is_call", True)

        if not currency_pair or strike is None:
            raise ValueError("Missing currency_pair or strike")

        # Get market data
        spot_rate = market_data.fx_rates.get(currency_pair)
        vol = market_data.volatilities.get(currency_pair)

        # Extract domestic and foreign rates
        domestic_ccy = currency_pair[3:6]  # e.g., "USD"
        foreign_ccy = currency_pair[0:3]  # e.g., "EUR"
        domestic_rate = market_data.interest_rates.get(domestic_ccy, 0.04)
        foreign_rate = market_data.interest_rates.get(foreign_ccy, 0.02)

        if spot_rate is None or vol is None:
            raise ValueError(f"Missing market data for {currency_pair}")

        # Calculate maturity
        if not trade.maturity_date:
            raise ValueError("Missing maturity_date")

        maturity = (trade.maturity_date - market_data.pricing_date).days / 365.0
        if maturity <= 0:
            return max(0.0, (spot_rate - strike) if is_call else (strike - spot_rate))

        # Price using Monte Carlo (with FX carry adjustment)
        self._key, subkey = jax.random.split(self._key)
        price = price_vanilla_mc(
            subkey,
            spot_rate,
            strike,
            maturity,
            domestic_rate,
            foreign_rate,  # Foreign rate acts as dividend yield
            vol,
            self.mc_config,
            is_call=is_call,
        )

        return float(price)

    def _price_interest_rate_swap(self, trade: Trade, market_data: MarketData) -> float:
        """Price an interest rate swap."""
        from neutryx.products.swap import price_vanilla_swap

        if not trade.product_details:
            raise ValueError("Missing product_details for interest rate swap")

        details = trade.product_details
        fixed_rate = details.get("fixed_rate")
        floating_rate = details.get("floating_rate")
        payment_frequency = details.get("payment_frequency", 2)
        pay_fixed = details.get("pay_fixed", True)

        if fixed_rate is None or floating_rate is None:
            raise ValueError("Missing fixed_rate or floating_rate")

        if not trade.notional:
            raise ValueError("Missing notional for swap")

        # Calculate maturity
        if not trade.maturity_date:
            raise ValueError("Missing maturity_date")

        maturity = (trade.maturity_date - market_data.pricing_date).days / 365.0
        if maturity <= 0:
            return 0.0

        # Get discount rate
        discount_rate = market_data.interest_rates.get(trade.currency or "USD", 0.05)

        # Price swap
        value = price_vanilla_swap(
            notional=trade.notional,
            fixed_rate=fixed_rate,
            floating_rate=floating_rate,
            maturity=maturity,
            payment_frequency=payment_frequency,
            discount_rate=discount_rate,
            pay_fixed=pay_fixed,
        )

        return float(value)

    def price_portfolio(
        self, trades: List[Trade], market_data: MarketData, update_mtm: bool = True
    ) -> List[PricingResult]:
        """Price multiple trades.

        Args:
            trades: List of trades to price
            market_data: Market data snapshot
            update_mtm: If True, update trade MTM values

        Returns:
            List of pricing results
        """
        results = []

        for trade in trades:
            result = self.price_trade(trade, market_data)
            results.append(result)

            # Update MTM if requested and pricing succeeded
            if update_mtm and result.success:
                trade.update_mtm(result.price, market_data.pricing_date)

        return results

    def extract_pricing_parameters(self, trade: Trade) -> Dict[str, Any]:
        """Extract pricing parameters from a trade.

        Args:
            trade: Trade to extract parameters from

        Returns:
            Dictionary of pricing parameters
        """
        params = {
            "trade_id": trade.id,
            "product_type": trade.product_type.value,
            "notional": trade.notional,
            "currency": trade.currency,
            "maturity_date": trade.maturity_date,
        }

        if trade.product_details:
            params.update(trade.product_details)

        return params

    def create_market_data_snapshot(
        self,
        pricing_date: date,
        spot_prices: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        interest_rates: Optional[Dict[str, float]] = None,
    ) -> MarketData:
        """Create a market data snapshot.

        Args:
            pricing_date: Pricing date
            spot_prices: Spot prices by instrument
            volatilities: Volatilities by instrument
            interest_rates: Interest rates by currency

        Returns:
            Market data snapshot
        """
        return MarketData(
            pricing_date=pricing_date,
            spot_prices=spot_prices or {},
            volatilities=volatilities or {},
            interest_rates=interest_rates or {},
        )


__all__ = [
    "PricingResult",
    "MarketData",
    "PricingBridge",
]
