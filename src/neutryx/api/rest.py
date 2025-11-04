"""REST API exposing pricing workflows via FastAPI."""
from __future__ import annotations

from typing import Any, List, Sequence

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neutryx.core.engine import MCConfig, price_vanilla_mc
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.xva.fva import fva
from neutryx.valuations.xva.mva import mva


class VanillaOptionRequest(BaseModel):
    spot: float = Field(..., description="Current underlying spot level")
    strike: float = Field(..., description="Option strike price")
    maturity: float = Field(..., description="Time to maturity in years")
    rate: float = Field(0.0, description="Risk-free rate")
    dividend: float = Field(0.0, description="Dividend yield")
    volatility: float = Field(..., description="Volatility of the underlying")
    steps: int = Field(64, gt=0, description="Number of time steps")
    paths: int = Field(8192, gt=0, description="Number of Monte Carlo paths")
    antithetic: bool = Field(False, description="Use antithetic sampling")
    call: bool = Field(True, description="Price a call (False for put)")
    seed: int | None = Field(None, description="PRNG seed for simulation determinism")


class ProfileRequest(BaseModel):
    values: List[float]

    def to_array(self) -> jnp.ndarray:
        if not self.values:
            raise HTTPException(status_code=400, detail="Expected non-empty sequence")
        return jnp.asarray(self.values, dtype=jnp.float32)


class CVARequest(BaseModel):
    epe: ProfileRequest
    discount: ProfileRequest
    default_probability: ProfileRequest
    lgd: float = Field(0.6, description="Loss given default")


class FVARequest(BaseModel):
    epe: ProfileRequest
    discount: ProfileRequest
    funding_spread: Sequence[float] | float


class MVARequest(BaseModel):
    initial_margin: ProfileRequest
    discount: ProfileRequest
    spread: Sequence[float] | float


class PortfolioXVARequest(BaseModel):
    """Request to compute XVA for a portfolio or netting set."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    netting_set_id: str | None = Field(
        None, description="Netting set ID (if None, compute for entire portfolio)"
    )
    valuation_date: str = Field(..., description="Valuation date (ISO format)")
    compute_cva: bool = Field(True, description="Compute CVA")
    compute_dva: bool = Field(False, description="Compute DVA")
    compute_fva: bool = Field(False, description="Compute FVA")
    compute_mva: bool = Field(False, description="Compute MVA")
    lgd: float = Field(0.6, description="Loss given default (if not in counterparty data)")
    funding_spread_bps: float = Field(50.0, description="Funding spread in bps for FVA")


class PortfolioSummaryRequest(BaseModel):
    """Request to get portfolio summary statistics."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    valuation_date: str | None = Field(None, description="Valuation date (ISO format)")


class FpMLPriceRequest(BaseModel):
    """Request to price an FpML document."""

    fpml_xml: str = Field(..., description="FpML XML document")
    market_data: dict[str, Any] = Field(
        ..., description="Market data: spot, volatility, rate, dividend, etc."
    )
    steps: int = Field(252, gt=0, description="Number of time steps")
    paths: int = Field(100_000, gt=0, description="Number of Monte Carlo paths")
    seed: int = Field(42, description="Random seed")


class FpMLParseRequest(BaseModel):
    """Request to parse an FpML document."""

    fpml_xml: str = Field(..., description="FpML XML document")


class FpMLValidateRequest(BaseModel):
    """Request to validate an FpML document."""

    fpml_xml: str = Field(..., description="FpML XML document")


def _to_array(values: Sequence[float] | float, *, match: int) -> jnp.ndarray:
    if isinstance(values, (int, float)):
        return jnp.full((match,), float(values), dtype=jnp.float32)
    seq = list(values)
    if len(seq) != match:
        raise HTTPException(status_code=400, detail="Sequence length mismatch")
    return jnp.asarray(seq, dtype=jnp.float32)


def create_app() -> FastAPI:
    app = FastAPI(title="Neutryx Pricing API", version="0.1.0")

    # In-memory portfolio storage (for demo purposes)
    # In production, this would be a database or cache
    _portfolios: dict[str, Any] = {}

    @app.post("/price/vanilla")
    def price_vanilla(payload: VanillaOptionRequest) -> dict[str, float]:
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
        epe = payload.epe.to_array()
        discount = payload.discount.to_array()
        pd = payload.default_probability.to_array()
        if epe.shape[0] != discount.shape[0] or epe.shape[0] != pd.shape[0]:
            raise HTTPException(status_code=400, detail="Profiles must share the same length")
        value = cva(epe, discount, pd, lgd=float(payload.lgd))
        return {"cva": float(value)}

    @app.post("/xva/fva")
    def compute_fva(payload: FVARequest) -> dict[str, float]:
        epe = payload.epe.to_array()
        discount = payload.discount.to_array()
        spread = _to_array(payload.funding_spread, match=epe.shape[0])
        value = fva(epe, spread, discount)
        return {"fva": float(value)}

    @app.post("/xva/mva")
    def compute_mva(payload: MVARequest) -> dict[str, float]:
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
        try:
            from neutryx.integrations import fpml

            # Parse FpML
            fpml_doc = fpml.parse_fpml(payload.fpml_xml)

            # Add MC config to market data
            market_data = payload.market_data.copy()
            market_data.setdefault("steps", payload.steps)
            market_data.setdefault("paths", payload.paths)

            # Convert to Neutryx request
            request = fpml.fpml_to_neutryx(fpml_doc, market_data)

            # Extract trade info
            trade = fpml_doc.primary_trade
            trade_info = {}

            # Handle different product types
            if trade.swap:
                # Swap pricing - request is already a dict with the value
                if isinstance(request, dict):
                    swap_value = request["value"]
                    trade_info = {
                        "product_type": "InterestRateSwap",
                        "notional": request["notional"],
                        "fixed_rate": request["fixed_rate"],
                        "floating_rate": request["floating_rate"],
                        "maturity": request["maturity"],
                        "payment_frequency": request["payment_frequency"],
                    }
                    return {
                        "price": swap_value,
                        "trade_date": trade.tradeHeader.tradeDate.isoformat(),
                        "trade_info": trade_info,
                    }

            # Options pricing - use Monte Carlo
            key = jax.random.PRNGKey(payload.seed)
            mc_cfg = MCConfig(steps=payload.steps, paths=payload.paths)
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

            if trade.equityOption:
                trade_info = {
                    "product_type": "EquityOption",
                    "option_type": trade.equityOption.optionType.value,
                    "strike": float(trade.equityOption.strike.strikePrice),
                    "underlyer": trade.equityOption.underlyer.instrumentId,
                }
            elif trade.fxOption:
                trade_info = {
                    "product_type": "FxOption",
                    "strike": float(trade.fxOption.strike.rate),
                }

            return {
                "price": float(price),
                "trade_date": trade.tradeHeader.tradeDate.isoformat(),
                "trade_info": trade_info,
            }
        except fpml.FpMLParseError as e:
            raise HTTPException(status_code=400, detail=f"FpML parse error: {e}")
        except fpml.FpMLMappingError as e:
            raise HTTPException(status_code=400, detail=f"FpML mapping error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pricing error: {e}")

    @app.post("/fpml/parse")
    def parse_fpml_doc(payload: FpMLParseRequest) -> dict[str, Any]:
        """Parse an FpML document and return its structure.

        Useful for validating FpML documents and inspecting trade details.
        """
        try:
            from neutryx.integrations import fpml

            fpml_doc = fpml.parse_fpml(payload.fpml_xml)
            trade = fpml_doc.primary_trade

            result = {
                "trade_date": trade.tradeHeader.tradeDate.isoformat(),
                "parties": [{"id": p.id, "name": p.name} for p in fpml_doc.party],
            }

            if trade.equityOption:
                opt = trade.equityOption
                result["product"] = {
                    "type": "EquityOption",
                    "option_type": opt.optionType.value,
                    "exercise_type": opt.equityExercise.optionType.value,
                    "strike": float(opt.strike.strikePrice),
                    "underlyer": opt.underlyer.instrumentId,
                    "expiration": opt.equityExercise.expirationDate.unadjustedDate.isoformat(),
                    "number_of_options": float(opt.numberOfOptions),
                }
            elif trade.fxOption:
                opt = trade.fxOption
                result["product"] = {
                    "type": "FxOption",
                    "exercise_type": opt.fxExercise.optionType.value,
                    "strike": float(opt.strike.rate),
                    "put_currency": opt.putCurrencyAmount.currency.value,
                    "put_amount": float(opt.putCurrencyAmount.amount),
                    "call_currency": opt.callCurrencyAmount.currency.value,
                    "call_amount": float(opt.callCurrencyAmount.amount),
                    "expiry": opt.fxExercise.expiryDate.isoformat(),
                }
            elif trade.swap:
                result["product"] = {
                    "type": "InterestRateSwap",
                    "streams": len(trade.swap.swapStream),
                }

            return result
        except fpml.FpMLParseError as e:
            raise HTTPException(status_code=400, detail=f"FpML parse error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Parse error: {e}")

    @app.post("/fpml/validate")
    def validate_fpml_doc(payload: FpMLValidateRequest) -> dict[str, Any]:
        """Validate an FpML document structure.

        Returns validation status and any errors found.
        """
        try:
            from neutryx.integrations import fpml

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
            raise HTTPException(status_code=400, detail=f"Portfolio registration error: {e}")

    @app.get("/portfolio/{portfolio_id}/summary")
    def get_portfolio_summary(portfolio_id: str) -> dict[str, Any]:
        """Get portfolio summary statistics."""
        if portfolio_id not in _portfolios:
            raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")

        portfolio = _portfolios[portfolio_id]

        summary = {
            "portfolio_id": portfolio_id,
            "base_currency": portfolio.base_currency,
            **portfolio.summary(),
            "total_mtm": portfolio.calculate_total_mtm(),
            "gross_notional": portfolio.calculate_gross_notional(),
        }

        # Counterparty breakdown
        counterparty_summary = []
        for cp_id, cp in portfolio.counterparties.items():
            cp_trades = portfolio.get_trades_by_counterparty(cp_id)
            counterparty_summary.append(
                {
                    "counterparty_id": cp_id,
                    "name": cp.name,
                    "entity_type": cp.entity_type.value,
                    "num_trades": len(cp_trades),
                    "net_mtm": portfolio.calculate_net_mtm_by_counterparty(cp_id),
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

        # Get trades to analyze
        if payload.netting_set_id:
            netting_set = portfolio.get_netting_set(payload.netting_set_id)
            if not netting_set:
                raise HTTPException(
                    status_code=404,
                    detail=f"Netting set '{payload.netting_set_id}' not found",
                )
            trades = portfolio.get_trades_by_netting_set(payload.netting_set_id)
            scope = f"netting_set:{payload.netting_set_id}"
        else:
            trades = list(portfolio.trades.values())
            netting_set = None
            scope = "portfolio"

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

        # Compute net MTM as simplified exposure
        net_mtm = sum(t.get_mtm(default=0.0) for t in trades)
        positive_exposure = max(net_mtm, 0.0)
        negative_exposure = max(-net_mtm, 0.0)

        result = {
            "scope": scope,
            "num_trades": len(trades),
            "net_mtm": net_mtm,
            "positive_exposure": positive_exposure,
            "negative_exposure": negative_exposure,
        }

        # Get counterparty for credit info
        if netting_set:
            cp = portfolio.get_counterparty(netting_set.counterparty_id)
            lgd = cp.get_lgd() if cp else payload.lgd
            has_csa = netting_set.has_csa()
        else:
            lgd = payload.lgd
            has_csa = False

        # Simplified XVA calculations
        # In production, these would use proper exposure profiles and term structures

        if payload.compute_cva:
            # CVA = LGD * EPE * PD (simplified)
            # Using a simplified default probability of 1% per year
            pd_annual = 0.01
            cva_value = lgd * positive_exposure * pd_annual
            result["cva"] = cva_value
        else:
            result["cva"] = None

        if payload.compute_dva:
            # DVA = Our LGD * ENE * Our PD (simplified)
            # Assuming our own default probability of 0.5% per year
            our_pd_annual = 0.005
            dva_value = 0.6 * negative_exposure * our_pd_annual
            result["dva"] = dva_value
        else:
            result["dva"] = None

        if payload.compute_fva:
            # FVA = Funding spread * Average uncollateralized exposure
            # If CSA exists, FVA is reduced/eliminated
            if has_csa:
                uncollateralized_exposure = 0.0
            else:
                uncollateralized_exposure = positive_exposure

            funding_spread = payload.funding_spread_bps / 10000.0
            fva_value = funding_spread * uncollateralized_exposure
            result["fva"] = fva_value
        else:
            result["fva"] = None

        if payload.compute_mva:
            # MVA simplified calculation
            # In production, this requires initial margin simulation
            if has_csa:
                # Rough estimate: 6% of gross notional
                gross_notional = sum(abs(t.notional or 0) for t in trades)
                im_estimate = 0.06 * gross_notional
                mva_value = 0.015 * im_estimate  # 150bps spread on IM
            else:
                mva_value = 0.0
            result["mva"] = mva_value
        else:
            result["mva"] = None

        # Total XVA
        total_xva = 0.0
        if result["cva"] is not None:
            total_xva += result["cva"]
        if result["dva"] is not None:
            total_xva -= result["dva"]  # DVA is a benefit
        if result["fva"] is not None:
            total_xva += result["fva"]
        if result["mva"] is not None:
            total_xva += result["mva"]

        result["total_xva"] = total_xva
        result["valuation_date"] = payload.valuation_date

        return result

    @app.get("/portfolio/{portfolio_id}/netting-sets")
    def list_netting_sets(portfolio_id: str) -> dict[str, Any]:
        """List all netting sets in a portfolio."""
        if portfolio_id not in _portfolios:
            raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")

        portfolio = _portfolios[portfolio_id]

        netting_sets_info = []
        for ns_id, ns in portfolio.netting_sets.items():
            trades = portfolio.get_trades_by_netting_set(ns_id)
            cp = portfolio.get_counterparty(ns.counterparty_id)

            netting_sets_info.append(
                {
                    "netting_set_id": ns_id,
                    "counterparty_id": ns.counterparty_id,
                    "counterparty_name": cp.name if cp else None,
                    "has_csa": ns.has_csa(),
                    "num_trades": len(trades),
                    "net_mtm": sum(t.get_mtm(default=0.0) for t in trades),
                }
            )

        return {"portfolio_id": portfolio_id, "netting_sets": netting_sets_info}

    return app


__all__ = ["create_app"]
