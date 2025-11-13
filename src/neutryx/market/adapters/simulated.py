"""Simulated market data adapter for development and testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from .base import AdapterConfig, BaseMarketDataAdapter, ConnectionState
from ..data_models import (
    BondQuote,
    CommodityQuote,
    CreditSpreadQuote,
    DataQuality,
    EquityQuote,
    FXQuote,
    InterestRateQuote,
    MarketDataPoint,
    VolatilityQuote,
    YieldCurve,
    DataRequest,
)
from ..storage.security_master import CorporateActionEvent


@dataclass
class SimulatedConfig(AdapterConfig):
    """Configuration for the simulated adapter."""

    seed: Optional[int] = None
    fx_levels: Dict[str, float] = field(
        default_factory=lambda: {
            "EURUSD": 1.10,
            "USDJPY": 110.0,
            "GBPUSD": 1.30,
        }
    )
    equity_levels: Dict[str, float] = field(
        default_factory=lambda: {
            "AAPL US Equity": 185.0,
            "MSFT US Equity": 340.0,
        }
    )
    rate_levels: Dict[str, float] = field(
        default_factory=lambda: {
            "USD:SOFR:1M": 0.052,
            "USD:SOFR:3M": 0.054,
            "EUR:ESTR:3M": 0.033,
        }
    )
    commodity_levels: Dict[str, float] = field(
        default_factory=lambda: {
            "CL": 72.0,
            "GC": 1950.0,
        }
    )
    credit_spreads: Dict[str, float] = field(
        default_factory=lambda: {
            "US-Treasury": 0.0,
            "ACME_CORP": 0.0125,
        }
    )


class SimulatedAdapter(BaseMarketDataAdapter):
    """Market data adapter that generates simulated quotes."""

    def __init__(self, config: SimulatedConfig):
        super().__init__(config)
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._connected = False
        self._state: Dict[str, float] = {}

    def connect(self) -> bool:
        self._connected = True
        self._notify_connection_state_change(ConnectionState.CONNECTED)
        return True

    def disconnect(self) -> bool:
        self._connected = False
        self._notify_connection_state_change(ConnectionState.DISCONNECTED)
        return True

    def is_connected(self) -> bool:
        return self._connected

    def get_equity_quote(
        self, ticker: str, exchange: Optional[str] = None
    ) -> Optional[EquityQuote]:
        base = self.config.equity_levels.get(ticker, 150.0)
        price = self._random_walk(f"equity:{ticker}", base, 1.5)
        spread = abs(self._rng.normal(0, max(price * 0.0005, 0.005)))

        return EquityQuote(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            ticker=ticker,
            exchange=exchange or "SIM",
            price=price,
            bid=price - spread,
            ask=price + spread,
            volume=max(int(self._rng.normal(1e6, 1e5)), 0),
            open_price=price * (1 - self._rng.normal(0, 0.005)),
            high=price * (1 + abs(self._rng.normal(0, 0.01))),
            low=price * (1 - abs(self._rng.normal(0, 0.01))),
            close=price * (1 - self._rng.normal(0, 0.003)),
            currency="USD",
        )

    def get_bond_quote(
        self, identifier: str, id_type: str = "isin"
    ) -> Optional[BondQuote]:
        base_price = 100.0
        price = self._random_walk(f"bond:{identifier}", base_price, 0.25)
        yield_to_maturity = max(0.0, self._rng.normal(0.03, 0.002))
        spread = max(0.0, self._rng.normal(0.015, 0.002))

        return BondQuote(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            isin=identifier if id_type == "isin" else "",
            cusip=identifier if id_type == "cusip" else "",
            price=price,
            yield_to_maturity=yield_to_maturity,
            accrued_interest=price * 0.01,
            spread=spread * 1e4,
            duration=5.0,
            convexity=50.0,
            maturity_date=date.today() + timedelta(days=365 * 5),
            coupon_rate=0.025,
            currency="USD",
        )

    def get_fx_quote(
        self, base_currency: str, quote_currency: str
    ) -> Optional[FXQuote]:
        pair_key = f"{base_currency}{quote_currency}".upper()
        base = self.config.fx_levels.get(pair_key, 1.10)
        spot = self._random_walk(f"fx:{pair_key}", base, base * 0.002)
        spread = abs(self._rng.normal(0, max(spot * 0.0002, 0.0001)))

        return FXQuote(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            currency_pair=f"{base_currency}/{quote_currency}",
            base_currency=base_currency,
            quote_currency=quote_currency,
            spot=spot,
            bid=spot - spread,
            ask=spot + spread,
        )

    def get_interest_rate(
        self, rate_type: str, currency: str, tenor: str
    ) -> Optional[InterestRateQuote]:
        key = f"{currency}:{rate_type}:{tenor}".upper()
        base = self.config.rate_levels.get(key, 0.035)
        rate = max(0.0, self._random_walk(f"rate:{key}", base, 0.0005))

        return InterestRateQuote(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            rate_type=rate_type,
            currency=currency,
            tenor=tenor,
            rate=rate,
            curve_name=f"{currency}-{rate_type}",
        )

    def get_commodity_quote(
        self, commodity_code: str, exchange: Optional[str] = None
    ) -> Optional[CommodityQuote]:
        base = self.config.commodity_levels.get(commodity_code, 50.0)
        price = self._random_walk(
            f"commodity:{commodity_code}", base, base * 0.01
        )

        return CommodityQuote(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            commodity_code=commodity_code,
            commodity_name=commodity_code,
            price=price,
            unit="unit",
            contract_month=None,
            exchange=exchange or "SIM",
            currency="USD",
        )

    def get_credit_spread(
        self, issuer: str, tenor: str, currency: str = "USD"
    ) -> Optional[CreditSpreadQuote]:
        base = self.config.credit_spreads.get(issuer, 0.01)
        spread = max(0.0, self._random_walk(f"credit:{issuer}", base, 0.001))

        return CreditSpreadQuote(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            issuer=issuer,
            ticker=issuer,
            currency=currency,
            tenor=tenor,
            spread=spread,
            recovery_rate=0.4,
        )

    def get_volatility_quote(
        self, underlying: str, strike: float, expiry: date, option_type: str = "call"
    ) -> Optional[VolatilityQuote]:
        base_vol = 0.20
        vol = max(0.05, self._random_walk(f"vol:{underlying}", base_vol, 0.01))

        return VolatilityQuote(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            volatility=vol,
            option_type=option_type,
        )

    def get_yield_curve(
        self, curve_name: str, currency: str = "USD"
    ) -> Optional[YieldCurve]:
        tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        base_rate = 0.03
        rates = [
            max(0.0, base_rate + 0.002 * tenor + self._rng.normal(0, 0.0005))
            for tenor in tenors
        ]

        return YieldCurve(
            timestamp=datetime.utcnow(),
            source=self.config.adapter_name,
            quality=DataQuality.REALTIME,
            curve_name=curve_name,
            currency=currency,
            tenors=tenors,
            rates=rates,
            curve_type="zero",
            interpolation_method="linear",
        )

    def get_corporate_actions(
        self, identifier: str, start_date: date, end_date: date
    ) -> List[CorporateActionEvent]:
        """Retrieve corporate action events for the specified security.

        For simulated adapter, returns an empty list as no corporate actions
        are generated in the simulation.

        Args:
            identifier: Security identifier
            start_date: Start date for corporate action search
            end_date: End date for corporate action search

        Returns:
            Empty list (simulated adapter doesn't generate corporate actions)
        """
        return []

    def _execute_request_internal(self, request: DataRequest) -> List[MarketDataPoint]:
        return []

    def _random_walk(self, key: str, base: float, sigma: float) -> float:
        last = self._state.get(key, base)
        shock = self._rng.normal(0, sigma)
        updated = max(0.0001, last + shock)
        self._state[key] = updated
        return updated
