"""REST API exposing pricing workflows via FastAPI."""
from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException

from neutryx.api.config import (
    DEFAULT_COUNTERPARTY_PD_ANNUAL,
    DEFAULT_IM_RATIO,
    DEFAULT_IM_SPREAD,
    DEFAULT_OWN_LGD,
    DEFAULT_OWN_PD_ANNUAL,
)
from neutryx.api.schemas import (
    CVARequest,
    FpMLParseRequest,
    FpMLPriceRequest,
    FpMLValidateRequest,
    FVARequest,
    MVARequest,
    PortfolioXVARequest,
    VanillaOptionRequest,
)
from neutryx.core.engine import MCConfig, price_vanilla_mc
from neutryx.integrations import fpml
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.xva.fva import fva
from neutryx.valuations.xva.mva import mva
from neutryx.infrastructure.observability import setup_observability


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


def _compute_exposures(trades: list[Any]) -> tuple[float, float, float]:
    """Compute net MTM and exposures from trades.

    Returns:
        (net_mtm, positive_exposure, negative_exposure)
    """
    net_mtm = sum(t.get_mtm(default=0.0) for t in trades)
    positive_exposure = max(net_mtm, 0.0)
    negative_exposure = max(-net_mtm, 0.0)
    return net_mtm, positive_exposure, negative_exposure


def _get_credit_parameters(
    portfolio: Any, netting_set: Any | None, default_lgd: float
) -> tuple[float, bool]:
    """Get credit parameters (LGD and CSA status).

    Returns:
        (lgd, has_csa)
    """
    if netting_set:
        counterparty = portfolio.get_counterparty(netting_set.counterparty_id)
        lgd = counterparty.get_lgd() if counterparty else default_lgd
        has_csa = netting_set.has_csa()
    else:
        lgd = default_lgd
        has_csa = False
    return lgd, has_csa


def _calculate_cva(lgd: float, positive_exposure: float) -> float:
    """Calculate Credit Valuation Adjustment (simplified)."""
    return lgd * positive_exposure * DEFAULT_COUNTERPARTY_PD_ANNUAL


def _calculate_dva(negative_exposure: float) -> float:
    """Calculate Debit Valuation Adjustment (simplified)."""
    return DEFAULT_OWN_LGD * negative_exposure * DEFAULT_OWN_PD_ANNUAL


def _calculate_fva(
    positive_exposure: float, has_csa: bool, funding_spread_bps: float
) -> float:
    """Calculate Funding Valuation Adjustment (simplified)."""
    uncollateralized_exposure = 0.0 if has_csa else positive_exposure
    funding_spread = funding_spread_bps / 10000.0
    return funding_spread * uncollateralized_exposure


def _calculate_mva(trades: list[Any], has_csa: bool) -> float:
    """Calculate Margin Valuation Adjustment (simplified)."""
    if not has_csa:
        return 0.0
    gross_notional = sum(abs(t.notional or 0) for t in trades)
    im_estimate = DEFAULT_IM_RATIO * gross_notional
    return DEFAULT_IM_SPREAD * im_estimate


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


def create_app() -> FastAPI:
    app = FastAPI(title="Neutryx Pricing API", version="0.1.0")
    observability = setup_observability(app)
    metrics = observability.metrics

    # In-memory portfolio storage (for demo purposes)
    # In production, this would be a database or cache
    _portfolios: dict[str, Any] = {}

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
                _portfolios[portfolio.name] = portfolio

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
        if portfolio_id not in _portfolios:
            raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")

        portfolio = _portfolios[portfolio_id]

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
        """Compute XVA adjustments for a portfolio or netting set.

        This is a simplified implementation that uses current MTM values
        as a proxy for exposure profiles. In production, you would:
        1. Run Monte Carlo simulation for each trade
        2. Compute exposure profiles (EE, PFE, EPE, ENE)
        3. Apply CVA/DVA/FVA/MVA formulas with proper term structures
        """
        if payload.portfolio_id not in _portfolios:
            raise HTTPException(
                status_code=404, detail=f"Portfolio '{payload.portfolio_id}' not found"
            )

        portfolio = _portfolios[payload.portfolio_id]
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

        # Compute exposures and credit parameters
        net_mtm, positive_exposure, negative_exposure = _compute_exposures(trades)
        lgd, has_csa = _get_credit_parameters(portfolio, netting_set, payload.lgd)

        # Build result with basic metrics
        result = {
            "scope": scope,
            "num_trades": len(trades),
            "net_mtm": net_mtm,
            "positive_exposure": positive_exposure,
            "negative_exposure": negative_exposure,
        }

        # Calculate XVA components
        result["cva"] = (
            _calculate_cva(lgd, positive_exposure) if payload.compute_cva else None
        )
        result["dva"] = _calculate_dva(negative_exposure) if payload.compute_dva else None
        result["fva"] = (
            _calculate_fva(positive_exposure, has_csa, payload.funding_spread_bps)
            if payload.compute_fva
            else None
        )
        result["mva"] = _calculate_mva(trades, has_csa) if payload.compute_mva else None

        # Calculate total XVA
        result["total_xva"] = _calculate_total_xva(result)
        result["valuation_date"] = payload.valuation_date

        return result

    @app.get("/portfolio/{portfolio_id}/netting-sets")
    def list_netting_sets(portfolio_id: str) -> dict[str, Any]:
        """List all netting sets in a portfolio."""
        if portfolio_id not in _portfolios:
            raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")

        portfolio = _portfolios[portfolio_id]

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
