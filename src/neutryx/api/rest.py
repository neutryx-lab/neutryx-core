"""REST API exposing pricing workflows via FastAPI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, List, Sequence

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from neutryx.api.config import (
    DEFAULT_COUNTERPARTY_PD_ANNUAL,
    DEFAULT_FUNDING_SPREAD_BPS,
    DEFAULT_IM_SPREAD,
    DEFAULT_LGD,
    DEFAULT_MC_PATHS,
    DEFAULT_OWN_LGD,
    DEFAULT_OWN_PD_ANNUAL,
    DEFAULT_SEED,
    RiskCurve,
)
from neutryx.api.portfolio_store import (
    PortfolioStore,
    PortfolioStoreSettings,
    create_portfolio_store,
)
from neutryx.api.schemas import (
    CVARequest,
    AuctionResultPayload,
    FpMLParseRequest,
    FpMLPriceRequest,
    FpMLValidateRequest,
    FVARequest,
    QuoteSubmissionRequest,
    QuoteSummary,
    MVARequest,
    PortfolioXVARequest,
    ProfileRequest,
    RFQCreateRequest,
    RFQStatusEvent,
    RFQStatusUpdateRequest,
    RFQSummary,
    VanillaOptionRequest,
)
from neutryx.core.engine import MCConfig, price_vanilla_mc, simulate_gbm
from neutryx.integrations import fpml
from neutryx.integrations.clearing.rfq import Quote, RFQManager
from neutryx.portfolio.contracts.trade import ProductType
from neutryx.products.swap import swap_value
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.xva.fva import fva
from neutryx.valuations.xva.mva import mva
from neutryx.infrastructure.observability import setup_observability

# Import authentication components
from neutryx.api.auth.endpoints import router as auth_router


def _to_array(values: Sequence[float] | float, *, match: int) -> jnp.ndarray:
    """Convert scalar or sequence to JAX array with length checking."""
    if isinstance(values, (int, float)):
        return jnp.full((match,), float(values), dtype=jnp.float32)
    seq = list(values)
    if len(seq) != match:
        raise HTTPException(status_code=400, detail="Sequence length mismatch")
    return jnp.asarray(seq, dtype=jnp.float32)


def _prepare_market_data(
    base_market_data: dict[str, Any], steps: int, paths: int
) -> dict[str, Any]:
    """Prepare market data by adding Monte Carlo configuration."""
    market_data = base_market_data.copy()
    market_data.setdefault("steps", steps)
    market_data.setdefault("paths", paths)
    return market_data


@dataclass(frozen=True)
class EquityParams:
    """Normalized market data for an equity underlyer."""

    spot: float
    volatility: float
    dividend: float
    currency: str | None


@dataclass(frozen=True)
class FXParams:
    """Normalized market data for an FX currency pair."""

    spot: float
    volatility: float
    domestic_currency: str
    foreign_currency: str
    domestic_rate: float
    foreign_rate: float


@dataclass(frozen=True)
class RateCurveData:
    """Normalized short-rate information and optional curves."""

    rate: float
    volatility: float | None
    discount_curve: jnp.ndarray | None
    forward_curve: jnp.ndarray | None


class MarketDataView:
    """Utility to access structured market data with validation."""

    def __init__(
        self,
        *,
        equities: dict[str, dict[str, Any]],
        fx_pairs: dict[str, dict[str, Any]],
        rates: dict[str, dict[str, Any]],
    ) -> None:
        self._equities = equities
        self._fx_pairs = fx_pairs
        self._rates = rates

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "MarketDataView":
        data = payload or {}
        equities = cls._parse_equities(data)
        fx_pairs = cls._parse_fx(data)
        rates = cls._parse_rates(data)
        return cls(equities=equities, fx_pairs=fx_pairs, rates=rates)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_mapping(value: Any, field: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise HTTPException(
                status_code=400,
                detail=f"market_data.{field} must be a mapping",
            )
        return value

    @staticmethod
    def _optional_float(value: Any, field: str, *, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid numeric value for market_data.{field}"
            ) from exc

    @staticmethod
    def _parse_equities(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        equities: dict[str, dict[str, Any]] = {}
        structured = MarketDataView._ensure_mapping(data.get("equities"), "equities")
        for name, entry in structured.items():
            if not isinstance(entry, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"market_data.equities['{name}'] must be a mapping",
                )
            equities[name] = {
                "spot": MarketDataView._optional_float(entry.get("spot"), f"equities.{name}.spot"),
                "volatility": MarketDataView._optional_float(
                    entry.get("volatility", entry.get("vol")), f"equities.{name}.volatility"
                ),
                "dividend": MarketDataView._optional_float(
                    entry.get("dividend", entry.get("dividend_yield", 0.0)),
                    f"equities.{name}.dividend",
                    default=0.0,
                ),
                "currency": entry.get("currency"),
            }

        legacy_spots = MarketDataView._ensure_mapping(data.get("spots"), "spots")
        for name, value in legacy_spots.items():
            equities.setdefault(name, {})["spot"] = MarketDataView._optional_float(
                value, f"spots.{name}"
            )

        legacy_vols = MarketDataView._ensure_mapping(data.get("vols"), "vols")
        for name, value in legacy_vols.items():
            equities.setdefault(name, {})["volatility"] = MarketDataView._optional_float(
                value, f"vols.{name}"
            )

        legacy_div = MarketDataView._ensure_mapping(data.get("dividends"), "dividends")
        for name, value in legacy_div.items():
            equities.setdefault(name, {})["dividend"] = MarketDataView._optional_float(
                value, f"dividends.{name}", default=0.0
            )

        for value in equities.values():
            value.setdefault("dividend", 0.0)
            value.setdefault("currency", None)
        return equities

    @staticmethod
    def _parse_fx(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        fx_pairs: dict[str, dict[str, Any]] = {}
        structured = MarketDataView._ensure_mapping(data.get("fx"), "fx")
        for pair, entry in structured.items():
            if not isinstance(entry, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"market_data.fx['{pair}'] must be a mapping",
                )
            fx_pairs[pair] = {
                "spot": MarketDataView._optional_float(entry.get("spot", entry.get("rate")), f"fx.{pair}.spot"),
                "volatility": MarketDataView._optional_float(
                    entry.get("volatility", entry.get("vol")), f"fx.{pair}.volatility"
                ),
                "domestic_currency": entry.get("domestic_currency"),
                "foreign_currency": entry.get("foreign_currency"),
                "domestic_rate": MarketDataView._optional_float(
                    entry.get("domestic_rate"), f"fx.{pair}.domestic_rate"
                ),
                "foreign_rate": MarketDataView._optional_float(
                    entry.get("foreign_rate"), f"fx.{pair}.foreign_rate"
                ),
            }

        legacy_fx = MarketDataView._ensure_mapping(data.get("fx_rates"), "fx_rates")
        for pair, value in legacy_fx.items():
            fx_pairs.setdefault(pair, {})["spot"] = MarketDataView._optional_float(
                value, f"fx_rates.{pair}"
            )

        legacy_vols = MarketDataView._ensure_mapping(data.get("vols"), "vols")
        for pair, value in legacy_vols.items():
            if pair not in fx_pairs:
                # Only treat entries that look like FX pairs (6 letters)
                if isinstance(pair, str) and len(pair) == 6:
                    fx_pairs.setdefault(pair, {})["volatility"] = MarketDataView._optional_float(
                        value, f"vols.{pair}"
                    )

        return fx_pairs

    @staticmethod
    def _coerce_curve(value: Any, field: str) -> jnp.ndarray | None:
        if value is None:
            return None
        if isinstance(value, dict):
            if "values" not in value:
                raise HTTPException(
                    status_code=400,
                    detail=f"market_data.{field} must provide 'values'",
                )
            value = value["values"]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return jnp.asarray([float(v) for v in value], dtype=jnp.float32)
        raise HTTPException(
            status_code=400, detail=f"market_data.{field} must be a sequence of numbers"
        )

    @staticmethod
    def _parse_rates(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rates: dict[str, dict[str, Any]] = {}
        structured = MarketDataView._ensure_mapping(data.get("rates"), "rates")
        for ccy, entry in structured.items():
            if isinstance(entry, dict):
                rate = MarketDataView._optional_float(entry.get("rate", entry.get("value")), f"rates.{ccy}.rate", default=0.0)
                rates[ccy] = {
                    "rate": rate if rate is not None else 0.0,
                    "volatility": MarketDataView._optional_float(
                        entry.get("volatility", entry.get("vol")), f"rates.{ccy}.volatility"
                    ),
                    "discount_curve": MarketDataView._coerce_curve(
                        entry.get("discount_curve"), f"rates.{ccy}.discount_curve"
                    )
                    if entry.get("discount_curve") is not None
                    else None,
                    "forward_curve": MarketDataView._coerce_curve(
                        entry.get("forward_curve"), f"rates.{ccy}.forward_curve"
                    )
                    if entry.get("forward_curve") is not None
                    else None,
                }
            else:
                rate = MarketDataView._optional_float(entry, f"rates.{ccy}", default=0.0)
                rates[ccy] = {
                    "rate": rate if rate is not None else 0.0,
                    "volatility": None,
                    "discount_curve": None,
                    "forward_curve": None,
                }

        discount_curves = MarketDataView._ensure_mapping(
            data.get("discount_curves"), "discount_curves"
        )
        for ccy, curve in discount_curves.items():
            rates.setdefault(ccy, {"rate": 0.0, "volatility": None, "forward_curve": None})[
                "discount_curve"
            ] = MarketDataView._coerce_curve(curve, f"discount_curves.{ccy}")

        forward_curves = MarketDataView._ensure_mapping(
            data.get("forward_curves"), "forward_curves"
        )
        for ccy, curve in forward_curves.items():
            rates.setdefault(ccy, {"rate": 0.0, "volatility": None, "discount_curve": None})[
                "forward_curve"
            ] = MarketDataView._coerce_curve(curve, f"forward_curves.{ccy}")

        return rates

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def require_equity(self, name: str) -> EquityParams:
        params = self._equities.get(name)
        if params is None:
            raise HTTPException(
                status_code=400, detail=f"Missing market data for equity '{name}'"
            )
        spot = params.get("spot")
        vol = params.get("volatility")
        if spot is None:
            raise HTTPException(
                status_code=400, detail=f"Equity '{name}' is missing a spot quote"
            )
        if vol is None:
            raise HTTPException(
                status_code=400, detail=f"Equity '{name}' is missing a volatility quote"
            )
        dividend = params.get("dividend", 0.0)
        currency = params.get("currency")
        return EquityParams(
            spot=float(spot),
            volatility=float(vol),
            dividend=float(dividend),
            currency=str(currency) if currency is not None else None,
        )

    def require_fx(self, pair: str) -> FXParams:
        params = self._fx_pairs.get(pair)
        if params is None:
            raise HTTPException(
                status_code=400, detail=f"Missing market data for FX pair '{pair}'"
            )
        spot = params.get("spot")
        vol = params.get("volatility")
        if spot is None:
            raise HTTPException(
                status_code=400, detail=f"FX pair '{pair}' is missing a spot rate"
            )
        if vol is None:
            raise HTTPException(
                status_code=400, detail=f"FX pair '{pair}' is missing a volatility quote"
            )
        domestic = params.get("domestic_currency")
        foreign = params.get("foreign_currency")
        if (domestic is None or foreign is None) and isinstance(pair, str) and len(pair) >= 6:
            foreign = pair[:3]
            domestic = pair[3:6]
        if domestic is None or foreign is None:
            raise HTTPException(
                status_code=400,
                detail=f"FX pair '{pair}' must define domestic and foreign currencies",
            )
        domestic_rate = params.get("domestic_rate")
        foreign_rate = params.get("foreign_rate")
        return FXParams(
            spot=float(spot),
            volatility=float(vol),
            domestic_currency=str(domestic),
            foreign_currency=str(foreign),
            domestic_rate=float(domestic_rate) if domestic_rate is not None else self.get_rate(domestic),
            foreign_rate=float(foreign_rate) if foreign_rate is not None else self.get_rate(foreign),
        )

    def get_rate(self, currency: str) -> float:
        params = self._rates.get(currency)
        if not params:
            return 0.0
        return float(params.get("rate", 0.0))

    def require_rate(self, currency: str) -> RateCurveData:
        params = self._rates.get(currency)
        if params is None:
            raise HTTPException(
                status_code=400, detail=f"Missing rate data for currency '{currency}'"
            )
        return RateCurveData(
            rate=float(params.get("rate", 0.0)),
            volatility=(
                float(params["volatility"])
                if params.get("volatility") is not None
                else None
            ),
            discount_curve=params.get("discount_curve"),
            forward_curve=params.get("forward_curve"),
        )

    def get_discount_curve(self, currency: str) -> jnp.ndarray | None:
        params = self._rates.get(currency)
        if not params:
            return None
        curve = params.get("discount_curve")
        return curve if curve is not None else None

def _extract_swap_trade_info(trade: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Extract trade information from a swap."""
    return {
        "product_type": "InterestRateSwap",
        "notional": request["notional"],
        "fixed_rate": request["fixed_rate"],
        "floating_rate": request["floating_rate"],
        "maturity": request["maturity"],
        "payment_frequency": request["payment_frequency"],
    }


def _extract_option_trade_info(trade: Any) -> dict[str, Any]:
    """Extract trade information from an option."""
    if trade.equityOption:
        return {
            "product_type": "EquityOption",
            "option_type": trade.equityOption.optionType.value,
            "strike": float(trade.equityOption.strike.strikePrice),
            "underlyer": trade.equityOption.underlyer.instrumentId,
        }
    elif trade.fxOption:
        return {
            "product_type": "FxOption",
            "strike": float(trade.fxOption.strike.rate),
        }
    return {}


def _price_option(request: Any, seed: int, steps: int, paths: int) -> float:
    """Price an option using Monte Carlo simulation."""
    key = jax.random.PRNGKey(seed)
    mc_cfg = MCConfig(steps=steps, paths=paths)
    price = price_vanilla_mc(
        key,
        request.spot,
        request.strike,
        request.maturity,
        request.rate,
        request.dividend,
        request.volatility,
        mc_cfg,
        is_call=request.call,
    )
    return float(price)


def _get_portfolio_trades(
    portfolio: Any, netting_set_id: str | None
) -> tuple[list[Any], Any | None, str]:
    """Get trades for a portfolio or netting set.

    Returns:
        (trades, netting_set, scope)
    """
    if netting_set_id:
        netting_set = portfolio.get_netting_set(netting_set_id)
        if not netting_set:
            raise HTTPException(
                status_code=404, detail=f"Netting set '{netting_set_id}' not found"
            )
        trades = portfolio.get_trades_by_netting_set(netting_set_id)
        scope = f"netting_set:{netting_set_id}"
    else:
        trades = list(portfolio.trades.values())
        netting_set = None
        scope = "portfolio"
    return trades, netting_set, scope


def _parse_valuation_date(value: str) -> datetime.date:
    """Parse valuation date provided in ISO format."""

    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid valuation date") from exc


def _resolve_time_grid(trades: list[Any], valuation_date: datetime.date, payload: Any) -> jnp.ndarray:
    """Determine the time grid used for exposure simulation."""

    if payload.time_grid:
        times = jnp.asarray(payload.time_grid, dtype=jnp.float32)
        if times.ndim != 1 or times.shape[0] < 2:
            raise HTTPException(
                status_code=400,
                detail="time_grid must contain at least two entries",
            )
        if not jnp.all(times[1:] >= times[:-1]):
            raise HTTPException(status_code=400, detail="time_grid must be non-decreasing")
        return times

    mc_cfg = payload.monte_carlo or {}
    steps = int(mc_cfg.get("steps", 32))
    if steps <= 0:
        raise HTTPException(status_code=400, detail="Monte Carlo steps must be positive")

    max_maturity = 0.0
    for trade in trades:
        ttm = trade.time_to_maturity(valuation_date)
        if ttm is not None:
            max_maturity = max(max_maturity, float(ttm))

    horizon = max(max_maturity, 1.0 / 365.0)
    return jnp.linspace(0.0, horizon, steps + 1, dtype=jnp.float32)


def _build_mc_config(payload: Any, steps: int) -> tuple[MCConfig, int]:
    cfg_dict = payload.monte_carlo or {}
    paths = int(cfg_dict.get("paths", DEFAULT_MC_PATHS))
    if paths <= 0:
        raise HTTPException(status_code=400, detail="Monte Carlo paths must be positive")
    seed = int(cfg_dict.get("seed", DEFAULT_SEED))
    return MCConfig(steps=steps, paths=paths), seed


def _market_bucket(market_data: dict[str, Any], name: str) -> dict[str, Any]:
    bucket = market_data.get(name, {}) if market_data else {}
    return bucket if isinstance(bucket, dict) else {}


def _profile_to_curve(
    profile: ProfileRequest | None,
    *,
    length: int,
    default: jnp.ndarray | Sequence[float] | float,
) -> jnp.ndarray:
    """Convert optional profile requests into arrays aligned with the time grid."""

    values: Sequence[float] | float | None
    if profile is None:
        values = None
    else:
        values = profile.values
    try:
        curve = RiskCurve.from_sequence(values, length=length, default=default)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return curve.as_array().astype(jnp.float32)


def _resolve_discount_curve(
    times: jnp.ndarray,
    payload: PortfolioXVARequest,
    market_data: dict[str, Any],
    base_currency: str,
    *,
    market_view: MarketDataView | None = None,
) -> jnp.ndarray:
    length = int(times.shape[0])
    if payload.discount_curve is not None:
        curve = payload.discount_curve.to_array()
        if curve.shape[0] != length:
            raise HTTPException(status_code=400, detail="Discount curve length mismatch")
        return curve.astype(jnp.float32)

    if market_view is None and market_data:
        market_view = MarketDataView.from_payload(market_data)

    if market_view is not None:
        curve = market_view.get_discount_curve(base_currency)
        if curve is not None:
            if curve.shape[0] != length:
                raise HTTPException(status_code=400, detail="Discount curve length mismatch")
            return jnp.asarray(curve, dtype=jnp.float32)
        rate = market_view.get_rate(base_currency)
    else:
        rate = float(_market_bucket(market_data, "rates").get(base_currency, 0.0))

    return jnp.exp(-float(rate) * times)


def _resolve_market_curve(
    market_data: dict[str, Any],
    key: str,
    length: int,
    *,
    default_value: float,
) -> jnp.ndarray:
    if market_data:
        raw = market_data.get(key)
        if isinstance(raw, dict) and "values" in raw:
            arr = jnp.asarray(raw["values"], dtype=jnp.float32)
        elif isinstance(raw, Sequence):
            arr = jnp.asarray(raw, dtype=jnp.float32)
        else:
            arr = None
        if arr is not None:
            if arr.shape[0] != length:
                raise HTTPException(status_code=400, detail=f"{key} length mismatch")
            return arr
    return jnp.full((length,), float(default_value), dtype=jnp.float32)

def _simulate_trade_exposure(
    trade: Any,
    *,
    key: jax.Array,
    cfg: MCConfig,
    times: jnp.ndarray,
    valuation_date: datetime.date,
    market: MarketDataView,
) -> jnp.ndarray:
    """Return pathwise exposure for a trade using a simple intrinsic model."""

    maturity = trade.time_to_maturity(valuation_date) or 0.0
    horizon = float(times[-1])
    if horizon <= 0.0:
        return jnp.zeros((cfg.paths, times.shape[0]), dtype=jnp.float32)

    mask = (times <= maturity + 1e-12).astype(jnp.float32)[None, :]
    product_type = trade.product_type
    if not isinstance(product_type, ProductType):
        try:
            product_type = ProductType(str(product_type))
        except ValueError:
            product_type = ProductType.OTHER

    if product_type == ProductType.EQUITY_OPTION:
        exposures = _simulate_equity_option_exposure(
            trade,
            key=key,
            cfg=cfg,
            horizon=horizon,
            market=market,
        )
    elif product_type == ProductType.FX_OPTION:
        exposures = _simulate_fx_option_exposure(
            trade,
            key=key,
            cfg=cfg,
            horizon=horizon,
            market=market,
        )
    elif product_type == ProductType.INTEREST_RATE_SWAP:
        exposures = _simulate_swap_exposure(
            trade,
            key=key,
            cfg=cfg,
            times=times,
            maturity=maturity,
            market=market,
        )
    else:
        exposures = _constant_mtm_exposure(trade, cfg, times)

    return exposures * mask


def _constant_mtm_exposure(trade: Any, cfg: MCConfig, times: jnp.ndarray) -> jnp.ndarray:
    base_mtm = trade.get_mtm(default=0.0)
    return jnp.full((cfg.paths, times.shape[0]), float(base_mtm), dtype=jnp.float32)


def _resolve_option_notional(trade: Any, details: dict[str, Any]) -> tuple[float, float]:
    notional = details.get("notional", trade.notional)
    if notional is None:
        notional = 1.0
    notional = float(abs(notional))
    direction = 1.0
    if details.get("is_long") is not None:
        direction = 1.0 if bool(details.get("is_long")) else -1.0
    elif trade.notional is not None and trade.notional < 0:
        direction = -1.0
    return notional, direction


def _simulate_equity_option_exposure(
    trade: Any,
    *,
    key: jax.Array,
    cfg: MCConfig,
    horizon: float,
    market: MarketDataView,
) -> jnp.ndarray:
    details = trade.product_details or {}
    underlying = details.get("underlying")
    if underlying is None:
        raise HTTPException(status_code=400, detail="Equity option missing underlying")
    if "strike" not in details:
        raise HTTPException(status_code=400, detail="Equity option missing strike")

    option_type = details.get("option_type")
    is_call = bool(details.get("is_call", option_type != "put"))
    strike = float(details["strike"])

    equity = market.require_equity(str(underlying))
    currency = trade.currency or equity.currency or "USD"
    rate = market.get_rate(currency)

    paths = jnp.asarray(
        simulate_gbm(
            key,
            equity.spot,
            rate - equity.dividend,
            equity.volatility,
            horizon,
            cfg,
            return_full=True,
        ).values,
        dtype=jnp.float32,
    )
    intrinsic = (
        jnp.maximum(paths - strike, 0.0)
        if is_call
        else jnp.maximum(strike - paths, 0.0)
    )
    notional, direction = _resolve_option_notional(trade, details)
    return direction * notional * intrinsic


def _simulate_fx_option_exposure(
    trade: Any,
    *,
    key: jax.Array,
    cfg: MCConfig,
    horizon: float,
    market: MarketDataView,
) -> jnp.ndarray:
    details = trade.product_details or {}
    currency_pair = details.get("currency_pair")
    if currency_pair is None:
        raise HTTPException(status_code=400, detail="FX option missing currency_pair")
    if "strike" not in details:
        raise HTTPException(status_code=400, detail="FX option missing strike")

    option_type = details.get("option_type")
    is_call = bool(details.get("is_call", option_type != "put"))
    strike = float(details["strike"])

    fx_data = market.require_fx(str(currency_pair))

    paths = jnp.asarray(
        simulate_gbm(
            key,
            fx_data.spot,
            fx_data.domestic_rate - fx_data.foreign_rate,
            fx_data.volatility,
            horizon,
            cfg,
            return_full=True,
        ).values,
        dtype=jnp.float32,
    )
    intrinsic = (
        jnp.maximum(paths - strike, 0.0)
        if is_call
        else jnp.maximum(strike - paths, 0.0)
    )
    notional, direction = _resolve_option_notional(trade, details)
    return direction * notional * intrinsic


def _simulate_swap_exposure(
    trade: Any,
    *,
    key: jax.Array,
    cfg: MCConfig,
    times: jnp.ndarray,
    maturity: float,
    market: MarketDataView,
) -> jnp.ndarray:
    details = trade.product_details or {}
    notional = details.get("notional", trade.notional)
    if notional is None:
        raise HTTPException(status_code=400, detail="Swap trade missing notional")
    fixed_rate = details.get("fixed_rate")
    if fixed_rate is None:
        raise HTTPException(status_code=400, detail="Swap trade missing fixed_rate")

    payment_frequency = int(details.get("payment_frequency", 2))
    if payment_frequency <= 0:
        raise HTTPException(status_code=400, detail="payment_frequency must be positive")

    pay_fixed = bool(details.get("pay_fixed", True))
    currency = details.get("currency", trade.currency) or "USD"
    rate_curve = market.require_rate(currency)
    floating_rate = float(details.get("floating_rate", rate_curve.rate))
    discount_rate = float(details.get("discount_rate", rate_curve.rate))

    volatility = details.get("rate_volatility", details.get("volatility"))
    if volatility is None:
        volatility = rate_curve.volatility if rate_curve.volatility is not None else 0.01
    volatility = float(volatility)

    time_points = [float(t) for t in jax.device_get(times)]
    exposures_list: list[jnp.ndarray] = []

    for idx, t_point in enumerate(time_points):
        remaining = max(maturity - t_point, 0.0)
        if remaining <= 0.0:
            exposures_list.append(jnp.zeros((cfg.paths,), dtype=jnp.float32))
            continue

        n_payments = max(int(math.ceil(remaining * payment_frequency)), 1)
        year_fractions = jnp.full(
            (n_payments,), 1.0 / payment_frequency, dtype=jnp.float32
        )
        payment_times = jnp.arange(1, n_payments + 1, dtype=jnp.float32) / payment_frequency
        discount_factors = jnp.exp(-discount_rate * payment_times)

        key_t = jax.random.fold_in(key, idx + 1)
        std = volatility * math.sqrt(max(t_point, 1e-12))
        if std > 0:
            shocks = std * jax.random.normal(key_t, (cfg.paths,), dtype=jnp.float32)
        else:
            shocks = jnp.zeros((cfg.paths,), dtype=jnp.float32)

        floating_rates = jnp.asarray(floating_rate, dtype=jnp.float32) + shocks
        pv_fn = jax.vmap(
            lambda fr: swap_value(
                float(notional),
                float(fixed_rate),
                fr,
                year_fractions,
                discount_factors,
                pay_fixed,
            )
        )
        exposures_list.append(pv_fn(floating_rates).astype(jnp.float32))

    return jnp.stack(exposures_list, axis=1)


def _compute_exposures(
    trades: list[Any],
    payload: Any,
    valuation_date: datetime.date,
    market: MarketDataView,
) -> dict[str, jnp.ndarray]:
    time_grid = _resolve_time_grid(trades, valuation_date, payload)
    mc_cfg, seed = _build_mc_config(payload, int(time_grid.shape[0] - 1))
    key = jax.random.PRNGKey(seed)

    exposures_sum: jnp.ndarray | None = None
    for idx, trade in enumerate(trades):
        trade_key = jax.random.fold_in(key, idx)
        trade_exposure = _simulate_trade_exposure(
            trade,
            key=trade_key,
            cfg=mc_cfg,
            times=time_grid,
            valuation_date=valuation_date,
            market=market,
        )
        exposures_sum = (
            trade_exposure
            if exposures_sum is None
            else exposures_sum + trade_exposure
        )

    if exposures_sum is None:
        exposures_sum = jnp.zeros((mc_cfg.paths, time_grid.shape[0]), dtype=jnp.float32)

    expected_positive = jnp.maximum(exposures_sum, 0.0).mean(axis=0)
    expected_negative = jnp.maximum(-exposures_sum, 0.0).mean(axis=0)
    net_exposure = exposures_sum.mean(axis=0)
    return {
        "times": time_grid,
        "pathwise": exposures_sum,
        "epe": expected_positive,
        "ene": expected_negative,
        "net": net_exposure,
    }


def _calculate_cva(
    epe: jnp.ndarray,
    discount: jnp.ndarray,
    default_prob: jnp.ndarray,
    lgd_curve: jnp.ndarray,
) -> float:
    increments = jnp.diff(jnp.concatenate([jnp.array([0.0], dtype=default_prob.dtype), default_prob]))
    return float((discount * epe * increments * lgd_curve).sum())


def _calculate_dva(
    ene: jnp.ndarray,
    discount: jnp.ndarray,
    default_prob: jnp.ndarray,
    lgd_curve: jnp.ndarray,
) -> float:
    increments = jnp.diff(jnp.concatenate([jnp.array([0.0], dtype=default_prob.dtype), default_prob]))
    return float((discount * ene * increments * lgd_curve).sum())


def _calculate_fva(
    epe: jnp.ndarray,
    discount: jnp.ndarray,
    funding_curve: jnp.ndarray,
    has_csa: bool,
) -> float:
    effective_epe = jnp.zeros_like(epe) if has_csa else epe
    return float((discount * effective_epe * funding_curve).sum())


def _calculate_mva(
    initial_margin: jnp.ndarray,
    discount: jnp.ndarray,
    spread_curve: jnp.ndarray,
) -> float:
    return float((discount * initial_margin * spread_curve).sum())


def _calculate_total_xva(result: dict[str, Any]) -> float:
    """Calculate total XVA from individual components."""
    total = 0.0
    if result["cva"] is not None:
        total += result["cva"]
    if result["dva"] is not None:
        total -= result["dva"]  # DVA is a benefit
    if result["fva"] is not None:
        total += result["fva"]
    if result["mva"] is not None:
        total += result["mva"]
    return total


def create_app(
    *,
    portfolio_store: PortfolioStore | None = None,
    store_settings: PortfolioStoreSettings | None = None,
) -> FastAPI:
    app = FastAPI(title="Neutryx Pricing API", version="0.1.0")
    observability = setup_observability(app)
    metrics = observability.metrics

    # Include authentication router
    app.include_router(auth_router)

    # Portfolio storage backend
    portfolio_store = portfolio_store or create_portfolio_store(store_settings)
    rfq_manager = RFQManager()

    def _rfq_error(exc: ValueError) -> HTTPException:
        message = str(exc)
        status_code = 404 if "not found" in message.lower() else 400
        return HTTPException(status_code=status_code, detail=message)

    @app.post("/rfq", response_model=RFQSummary)
    def create_rfq_endpoint(payload: RFQCreateRequest) -> RFQSummary:
        try:
            rfq_kwargs = payload.to_rfq_kwargs()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rfq = rfq_manager.create_rfq(**rfq_kwargs)
        return RFQSummary.from_rfq(rfq)

    @app.get("/rfq/{rfq_id}", response_model=RFQSummary)
    def get_rfq(rfq_id: str) -> RFQSummary:
        rfq = rfq_manager.get_rfq(rfq_id)
        if rfq is None:
            raise HTTPException(status_code=404, detail=f"RFQ {rfq_id} not found")
        return RFQSummary.from_rfq(rfq)

    @app.get("/rfq/{rfq_id}/status/history", response_model=List[RFQStatusEvent])
    def get_rfq_history(rfq_id: str) -> List[RFQStatusEvent]:
        rfq = rfq_manager.get_rfq(rfq_id)
        if rfq is None:
            raise HTTPException(status_code=404, detail=f"RFQ {rfq_id} not found")
        return RFQSummary.from_rfq(rfq).status_history

    @app.get("/rfq/{rfq_id}/quotes", response_model=List[QuoteSummary])
    def list_quotes(rfq_id: str) -> List[QuoteSummary]:
        rfq = rfq_manager.get_rfq(rfq_id)
        if rfq is None:
            raise HTTPException(status_code=404, detail=f"RFQ {rfq_id} not found")
        quotes = rfq_manager.get_quotes(rfq_id)
        return [QuoteSummary.from_quote(q) for q in quotes]

    @app.post("/rfq/{rfq_id}/submit", response_model=RFQSummary)
    def submit_rfq_endpoint(rfq_id: str) -> RFQSummary:
        try:
            rfq = rfq_manager.submit_rfq(rfq_id)
        except ValueError as exc:  # pragma: no cover - defensive
            raise _rfq_error(exc) from exc
        return RFQSummary.from_rfq(rfq)

    @app.post("/rfq/{rfq_id}/quotes", response_model=QuoteSummary)
    def submit_quote_endpoint(rfq_id: str, payload: QuoteSubmissionRequest) -> QuoteSummary:
        rfq = rfq_manager.get_rfq(rfq_id)
        if rfq is None:
            raise HTTPException(status_code=404, detail=f"RFQ {rfq_id} not found")

        try:
            quote = Quote(rfq_id=rfq_id, **payload.to_quote_kwargs())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            submitted = rfq_manager.submit_quote(rfq_id, quote)
        except ValueError as exc:
            raise _rfq_error(exc) from exc
        return QuoteSummary.from_quote(submitted)

    @app.post("/rfq/{rfq_id}/status", response_model=RFQSummary)
    def update_rfq_status(rfq_id: str, payload: RFQStatusUpdateRequest) -> RFQSummary:
        try:
            rfq = rfq_manager.update_status(
                rfq_id,
                payload.status,
                reason=payload.reason,
                metadata=payload.metadata or None,
            )
        except ValueError as exc:
            raise _rfq_error(exc) from exc
        return RFQSummary.from_rfq(rfq)

    @app.post("/rfq/{rfq_id}/execute", response_model=AuctionResultPayload)
    def execute_rfq(rfq_id: str) -> AuctionResultPayload:
        try:
            result = rfq_manager.execute_auction(rfq_id)
        except ValueError as exc:
            raise _rfq_error(exc) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return AuctionResultPayload.from_result(result)

    @app.websocket("/ws/rfq/{rfq_id}")
    async def rfq_status_stream(websocket: WebSocket, rfq_id: str) -> None:
        await websocket.accept()
        rfq = rfq_manager.get_rfq(rfq_id)
        if rfq is None:
            await websocket.close(code=4404, reason=f"RFQ {rfq_id} not found")
            return

        queue = rfq_manager.register_state_listener(rfq_id)
        try:
            while True:
                update = await queue.get()
                await websocket.send_json(update)
        except WebSocketDisconnect:  # pragma: no cover - network condition
            pass
        finally:
            rfq_manager.unregister_state_listener(rfq_id, queue)

    @app.post("/price/vanilla")
    def price_vanilla(payload: VanillaOptionRequest) -> dict[str, float]:
        with metrics.time(
            "pricing.vanilla",
            labels={"channel": "http", "product": "vanilla_option"},
        ):
            mc_cfg = MCConfig(steps=payload.steps, paths=payload.paths, antithetic=payload.antithetic)
            seed = payload.seed if payload.seed is not None else 0
            key = jax.random.PRNGKey(int(seed))
            price = price_vanilla_mc(
                key,
                payload.spot,
                payload.strike,
                payload.maturity,
                payload.rate,
                payload.dividend,
                payload.volatility,
                mc_cfg,
                is_call=payload.call,
            )
        return {"price": float(price)}

    @app.post("/xva/cva")
    def compute_cva(payload: CVARequest) -> dict[str, float]:
        with metrics.time(
            "xva.cva",
            labels={"channel": "http", "product": "cva"},
        ):
            epe = payload.epe.to_array()
            discount = payload.discount.to_array()
            default_probability = payload.default_probability.to_array()
            if epe.shape[0] != discount.shape[0] or epe.shape[0] != default_probability.shape[0]:
                raise HTTPException(status_code=400, detail="Profiles must share the same length")
            value = cva(epe, discount, default_probability, lgd=float(payload.lgd))
        return {"cva": float(value)}

    @app.post("/xva/fva")
    def compute_fva(payload: FVARequest) -> dict[str, float]:
        with metrics.time(
            "xva.fva",
            labels={"channel": "http", "product": "fva"},
        ):
            epe = payload.epe.to_array()
            discount = payload.discount.to_array()
            spread = _to_array(payload.funding_spread, match=epe.shape[0])
            value = fva(epe, spread, discount)
        return {"fva": float(value)}

    @app.post("/xva/mva")
    def compute_mva(payload: MVARequest) -> dict[str, float]:
        with metrics.time(
            "xva.mva",
            labels={"channel": "http", "product": "mva"},
        ):
            margin = payload.initial_margin.to_array()
            discount = payload.discount.to_array()
            spread = _to_array(payload.spread, match=margin.shape[0])
            value = mva(margin, discount, spread)
        return {"mva": float(value)}

    # FpML endpoints
    @app.post("/fpml/price")
    def price_fpml(payload: FpMLPriceRequest) -> dict[str, Any]:
        """Price an FpML trade document.

        Accepts an FpML XML document and market data, parses the trade,
        converts it to Neutryx format, and returns the computed price.
        """
        with metrics.time(
            "pricing.fpml",
            labels={"channel": "http", "product": "fpml"},
        ):
            try:
                fpml_doc = fpml.parse_fpml(payload.fpml_xml)
                market_data = _prepare_market_data(payload.market_data, payload.steps, payload.paths)
                request = fpml.fpml_to_neutryx(fpml_doc, market_data)
                trade = fpml_doc.primary_trade

                if trade.swap and isinstance(request, dict):
                    return {
                        "price": request["value"],
                        "trade_date": trade.tradeHeader.tradeDate.isoformat(),
                        "trade_info": _extract_swap_trade_info(trade, request),
                    }

                price = _price_option(request, payload.seed, payload.steps, payload.paths)
                return {
                    "price": price,
                    "trade_date": trade.tradeHeader.tradeDate.isoformat(),
                    "trade_info": _extract_option_trade_info(trade),
                }
            except fpml.FpMLParseError as e:
                raise HTTPException(status_code=400, detail=f"FpML parse error: {e}") from e
            except fpml.FpMLMappingError as e:
                raise HTTPException(status_code=400, detail=f"FpML mapping error: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Pricing error: {e}") from e

    @app.post("/fpml/parse")
    def parse_fpml_doc(payload: FpMLParseRequest) -> dict[str, Any]:
        """Parse an FpML document and return its structure.

        Useful for validating FpML documents and inspecting trade details.
        """
        with metrics.time(
            "fpml.parse",
            labels={"channel": "http", "product": "fpml"},
        ):
            try:
                fpml_doc = fpml.parse_fpml(payload.fpml_xml)
                trade = fpml_doc.primary_trade

                result = {
                    "trade_date": trade.tradeHeader.tradeDate.isoformat(),
                    "parties": [{"id": p.id, "name": p.name} for p in fpml_doc.party],
                }

                if trade.equityOption:
                    equity_option = trade.equityOption
                    result["product"] = {
                        "type": "EquityOption",
                        "option_type": equity_option.optionType.value,
                        "exercise_type": equity_option.equityExercise.optionType.value,
                        "strike": float(equity_option.strike.strikePrice),
                        "underlyer": equity_option.underlyer.instrumentId,
                        "expiration": (
                            equity_option.equityExercise.expirationDate.unadjustedDate.isoformat()
                        ),
                        "number_of_options": float(equity_option.numberOfOptions),
                    }
                elif trade.fxOption:
                    fx_option = trade.fxOption
                    result["product"] = {
                        "type": "FxOption",
                        "exercise_type": fx_option.fxExercise.optionType.value,
                        "strike": float(fx_option.strike.rate),
                        "put_currency": fx_option.putCurrencyAmount.currency.value,
                        "put_amount": float(fx_option.putCurrencyAmount.amount),
                        "call_currency": fx_option.callCurrencyAmount.currency.value,
                        "call_amount": float(fx_option.callCurrencyAmount.amount),
                        "expiry": fx_option.fxExercise.expiryDate.isoformat(),
                    }
                elif trade.swap:
                    result["product"] = {
                        "type": "InterestRateSwap",
                        "streams": len(trade.swap.swapStream),
                    }

                return result
            except fpml.FpMLParseError as e:
                raise HTTPException(status_code=400, detail=f"FpML parse error: {e}") from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Parse error: {e}") from e

    @app.post("/fpml/validate")
    def validate_fpml_doc(payload: FpMLValidateRequest) -> dict[str, Any]:
        """Validate an FpML document structure.

        Returns validation status and any errors found.
        """
        with metrics.time(
            "fpml.validate",
            labels={"channel": "http", "product": "fpml"},
        ):
            try:
                fpml.parse_fpml(payload.fpml_xml)
                return {"valid": True, "message": "FpML document is valid"}
            except fpml.FpMLParseError as e:
                return {"valid": False, "message": f"Parse error: {e}"}
            except Exception as e:
                return {"valid": False, "message": f"Validation error: {e}"}

    # Portfolio management endpoints
    @app.post("/portfolio/register")
    def register_portfolio(portfolio_data: dict[str, Any]) -> dict[str, Any]:
        """Register a portfolio for XVA calculations.

        Accepts a portfolio serialized as JSON (from Portfolio.model_dump()).
        """
        with metrics.time(
            "portfolio.register",
            labels={"channel": "http", "product": "portfolio"},
        ):
            try:
                from neutryx.portfolio.portfolio import Portfolio

                portfolio = Portfolio.model_validate(portfolio_data)
                portfolio_store.save_portfolio(portfolio)

                return {
                    "status": "registered",
                    "portfolio_id": portfolio.name,
                    "summary": portfolio.summary(),
                }
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Portfolio registration error: {e}"
                ) from e

    @app.get("/portfolio/{portfolio_id}/summary")
    def get_portfolio_summary(portfolio_id: str) -> dict[str, Any]:
        """Get portfolio summary statistics."""
        portfolio = portfolio_store.get_portfolio(portfolio_id)
        if portfolio is None:
            raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")

        with metrics.time(
            "portfolio.summary",
            labels={"channel": "http", "product": "portfolio"},
        ):
            summary = {
                "portfolio_id": portfolio_id,
                "base_currency": portfolio.base_currency,
                **portfolio.summary(),
                "total_mtm": portfolio.calculate_total_mtm(),
                "gross_notional": portfolio.calculate_gross_notional(),
            }

            counterparty_summary = []
            for counterparty_id, counterparty in portfolio.counterparties.items():
                counterparty_trades = portfolio.get_trades_by_counterparty(counterparty_id)
                counterparty_summary.append(
                    {
                        "counterparty_id": counterparty_id,
                        "name": counterparty.name,
                        "entity_type": counterparty.entity_type.value,
                        "num_trades": len(counterparty_trades),
                        "net_mtm": portfolio.calculate_net_mtm_by_counterparty(counterparty_id),
                    }
                )

        summary["counterparties"] = counterparty_summary
        return summary

    @app.post("/portfolio/xva")
    def compute_portfolio_xva(payload: PortfolioXVARequest) -> dict[str, Any]:
        """Compute XVA adjustments for a portfolio or netting set using MC exposures."""
        portfolio = portfolio_store.get_portfolio(payload.portfolio_id)
        if portfolio is None:
            raise HTTPException(
                status_code=404, detail=f"Portfolio '{payload.portfolio_id}' not found"
            )
        trades, netting_set, scope = _get_portfolio_trades(portfolio, payload.netting_set_id)

        # Handle empty portfolio case
        if not trades:
            return {
                "scope": scope,
                "num_trades": 0,
                "cva": 0.0,
                "dva": 0.0,
                "fva": 0.0,
                "mva": 0.0,
                "total_xva": 0.0,
            }

        valuation_date = _parse_valuation_date(payload.valuation_date)
        market_data_raw = payload.market_data or {}
        market_view = MarketDataView.from_payload(market_data_raw)
        exposures = _compute_exposures(trades, payload, valuation_date, market_view)
        times = exposures["times"]
        length = int(times.shape[0])

        if netting_set:
            counterparty_id = netting_set.counterparty_id
            has_csa = netting_set.has_csa()
        else:
            counterparty_id = trades[0].counterparty_id
            has_csa = False

        times_list = [float(t) for t in jax.device_get(times)]
        cp_pd_from_portfolio = portfolio.get_counterparty_default_probabilities(counterparty_id, times_list)
        lgd_from_portfolio = portfolio.get_counterparty_lgd_curve(counterparty_id, times_list)

        discount_curve = _resolve_discount_curve(
            times,
            payload,
            market_data_raw,
            portfolio.base_currency,
            market_view=market_view,
        )

        lam_counterparty = -jnp.log1p(-DEFAULT_COUNTERPARTY_PD_ANNUAL)
        cp_pd_default = (
            cp_pd_from_portfolio.astype(jnp.float32)
            if cp_pd_from_portfolio is not None
            else 1.0 - jnp.exp(-lam_counterparty * times)
        )
        counterparty_pd = _profile_to_curve(
            payload.counterparty_pd, length=length, default=cp_pd_default
        )

        lam_own = -jnp.log1p(-DEFAULT_OWN_PD_ANNUAL)
        own_pd_default = 1.0 - jnp.exp(-lam_own * times)
        own_pd_curve = _profile_to_curve(payload.own_pd, length=length, default=own_pd_default)

        lgd_base = (
            lgd_from_portfolio.astype(jnp.float32)
            if lgd_from_portfolio is not None
            else jnp.full(
                (length,),
                float(payload.lgd if payload.lgd is not None else DEFAULT_LGD),
                dtype=jnp.float32,
            )
        )
        lgd_curve = _profile_to_curve(payload.lgd_curve, length=length, default=lgd_base)
        own_lgd_curve = _profile_to_curve(
            payload.own_lgd_curve,
            length=length,
            default=jnp.full((length,), DEFAULT_OWN_LGD, dtype=jnp.float32),
        )

        funding_bps = float(payload.funding_spread_bps or DEFAULT_FUNDING_SPREAD_BPS)
        funding_default = _resolve_market_curve(
            market_data_raw,
            "funding_curve",
            length,
            default_value=funding_bps / 10000.0,
        )
        funding_curve = _profile_to_curve(
            payload.funding_curve, length=length, default=funding_default
        )

        im_default = _resolve_market_curve(
            market_data_raw, "initial_margin", length, default_value=0.0
        )
        initial_margin_curve = _profile_to_curve(
            payload.initial_margin, length=length, default=im_default
        )
        im_spread_curve = _resolve_market_curve(
            market_data_raw, "im_spread", length, default_value=DEFAULT_IM_SPREAD
        )

        result = {
            "scope": scope,
            "num_trades": len(trades),
            "net_mtm": float(sum(t.get_mtm(default=0.0) for t in trades)),
            "positive_exposure": float(jnp.max(exposures["epe"])),
            "negative_exposure": float(jnp.max(exposures["ene"])),
            "times": jax.device_get(times).tolist(),
            "epe_profile": jax.device_get(exposures["epe"]).tolist(),
            "ene_profile": jax.device_get(exposures["ene"]).tolist(),
            "net_exposure_profile": jax.device_get(exposures["net"]).tolist(),
        }

        result["discount_curve"] = jax.device_get(discount_curve).tolist()
        result["counterparty_pd_curve"] = jax.device_get(counterparty_pd).tolist()
        result["own_pd_curve"] = jax.device_get(own_pd_curve).tolist()
        result["lgd_curve"] = jax.device_get(lgd_curve).tolist()
        result["funding_curve"] = jax.device_get(funding_curve).tolist()

        result["cva"] = (
            _calculate_cva(exposures["epe"], discount_curve, counterparty_pd, lgd_curve)
            if payload.compute_cva
            else None
        )
        result["dva"] = (
            _calculate_dva(exposures["ene"], discount_curve, own_pd_curve, own_lgd_curve)
            if payload.compute_dva
            else None
        )
        result["fva"] = (
            _calculate_fva(exposures["epe"], discount_curve, funding_curve, has_csa)
            if payload.compute_fva
            else None
        )
        result["mva"] = (
            _calculate_mva(initial_margin_curve, discount_curve, im_spread_curve)
            if payload.compute_mva
            else None
        )

        result["total_xva"] = _calculate_total_xva(result)
        result["valuation_date"] = payload.valuation_date
        result["has_csa"] = has_csa

        return result


    @app.get("/portfolio/{portfolio_id}/netting-sets")
    def list_netting_sets(portfolio_id: str) -> dict[str, Any]:
        """List all netting sets in a portfolio."""
        portfolio = portfolio_store.get_portfolio(portfolio_id)
        if portfolio is None:
            raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")

        netting_sets_info = []
        for netting_set_id, netting_set in portfolio.netting_sets.items():
            trades = portfolio.get_trades_by_netting_set(netting_set_id)
            counterparty = portfolio.get_counterparty(netting_set.counterparty_id)

            netting_sets_info.append(
                {
                    "netting_set_id": netting_set_id,
                    "counterparty_id": netting_set.counterparty_id,
                    "counterparty_name": counterparty.name if counterparty else None,
                    "has_csa": netting_set.has_csa(),
                    "num_trades": len(trades),
                    "net_mtm": sum(t.get_mtm(default=0.0) for t in trades),
                }
            )

        return {"portfolio_id": portfolio_id, "netting_sets": netting_sets_info}

    return app


__all__ = ["create_app"]
