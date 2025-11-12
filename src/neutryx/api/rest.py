"""REST API exposing pricing workflows via FastAPI."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException

from neutryx.api.config import (
    DEFAULT_COUNTERPARTY_PD_ANNUAL,
    DEFAULT_IM_SPREAD,
    DEFAULT_LGD,
    DEFAULT_FUNDING_SPREAD_BPS,
    DEFAULT_OWN_LGD,
    DEFAULT_OWN_PD_ANNUAL,
    DEFAULT_MC_PATHS,
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
    FpMLParseRequest,
    FpMLPriceRequest,
    FpMLValidateRequest,
    FVARequest,
    MVARequest,
    PortfolioXVARequest,
    ProfileRequest,
    VanillaOptionRequest,
)
from neutryx.core.engine import MCConfig, price_vanilla_mc, simulate_gbm
from neutryx.integrations import fpml
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
) -> jnp.ndarray:
    length = int(times.shape[0])
    if payload.discount_curve is not None:
        curve = payload.discount_curve.to_array()
        if curve.shape[0] != length:
            raise HTTPException(status_code=400, detail="Discount curve length mismatch")
        return curve.astype(jnp.float32)

    market_curve = None
    if market_data:
        raw = market_data.get("discount_curve")
        if isinstance(raw, dict) and "values" in raw:
            market_curve = raw["values"]
        elif isinstance(raw, Sequence):
            market_curve = raw

    if market_curve is not None:
        curve = jnp.asarray(market_curve, dtype=jnp.float32)
        if curve.shape[0] != length:
            raise HTTPException(status_code=400, detail="Discount curve length mismatch")
        return curve

    rate = float(_market_bucket(market_data, "rates").get(base_currency, 0.0))
    return jnp.exp(-rate * times)


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
    market_data: dict[str, Any],
) -> jnp.ndarray:
    """Return pathwise exposure for a trade using a simple intrinsic model."""

    maturity = trade.time_to_maturity(valuation_date) or 0.0
    horizon = float(times[-1])
    if horizon <= 0.0:
        return jnp.zeros((cfg.paths, times.shape[0]), dtype=jnp.float32)

    mask = (times <= maturity + 1e-12).astype(jnp.float32)[None, :]
    bucket_spot = _market_bucket(market_data, "spots")
    bucket_vol = _market_bucket(market_data, "vols")
    bucket_rates = _market_bucket(market_data, "rates")
    bucket_div = _market_bucket(market_data, "dividends")

    if trade.product_type.value == "EquityOption":
        details = trade.product_details or {}
        underlying = details.get("underlying")
        strike = float(details.get("strike", 0.0))
        is_call = bool(details.get("is_call", True))
        if underlying is None:
            raise HTTPException(status_code=400, detail="Equity option missing underlying")
        if underlying not in bucket_spot or underlying not in bucket_vol:
            raise HTTPException(status_code=400, detail=f"Missing market data for {underlying}")
        spot = float(bucket_spot[underlying])
        sigma = float(bucket_vol[underlying])
        currency = trade.currency or "USD"
        rate = float(bucket_rates.get(currency, 0.0))
        dividend = float(bucket_div.get(underlying, 0.0))
        paths = jnp.asarray(
            simulate_gbm(
                key,
                spot,
                rate - dividend,
                sigma,
                horizon,
                cfg,
                return_full=True,
            ).values,
            dtype=jnp.float32,
        )
        notional = abs(trade.notional) if trade.notional is not None else 1.0
        direction = 1.0
        if details.get("is_long") is not None:
            direction = 1.0 if details.get("is_long") else -1.0
        elif trade.notional is not None and trade.notional < 0:
            direction = -1.0
        intrinsic = jnp.maximum(paths - strike, 0.0) if is_call else jnp.maximum(strike - paths, 0.0)
        exposures = direction * float(notional) * intrinsic
        return exposures * mask

    base_mtm = trade.get_mtm(default=0.0)
    exposures = jnp.full((cfg.paths, times.shape[0]), float(base_mtm), dtype=jnp.float32)
    return exposures * mask


def _compute_exposures(
    trades: list[Any],
    payload: Any,
    valuation_date: datetime.date,
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
            market_data=payload.market_data,
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
        exposures = _compute_exposures(trades, payload, valuation_date)
        times = exposures["times"]
        length = int(times.shape[0])
        market_data = payload.market_data or {}

        if netting_set:
            counterparty_id = netting_set.counterparty_id
            has_csa = netting_set.has_csa()
        else:
            counterparty_id = trades[0].counterparty_id
            has_csa = False

        times_list = [float(t) for t in jax.device_get(times)]
        cp_pd_from_portfolio = portfolio.get_counterparty_default_probabilities(counterparty_id, times_list)
        lgd_from_portfolio = portfolio.get_counterparty_lgd_curve(counterparty_id, times_list)

        discount_curve = _resolve_discount_curve(times, payload, market_data, portfolio.base_currency)

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
            market_data,
            "funding_curve",
            length,
            default_value=funding_bps / 10000.0,
        )
        funding_curve = _profile_to_curve(
            payload.funding_curve, length=length, default=funding_default
        )

        im_default = _resolve_market_curve(market_data, "initial_margin", length, default_value=0.0)
        initial_margin_curve = _profile_to_curve(
            payload.initial_margin, length=length, default=im_default
        )
        im_spread_curve = _resolve_market_curve(
            market_data, "im_spread", length, default_value=DEFAULT_IM_SPREAD
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
