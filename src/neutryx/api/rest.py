"""REST API exposing pricing workflows via FastAPI."""
from __future__ import annotations

from typing import Any, List, Sequence

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neutryx.core.engine import MCConfig, price_vanilla_mc
from neutryx.valuations.cva import cva
from neutryx.valuations.fva import fva
from neutryx.valuations.mva import mva


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
            from neutryx.bridge import fpml

            # Parse FpML
            fpml_doc = fpml.parse_fpml(payload.fpml_xml)

            # Add MC config to market data
            market_data = payload.market_data.copy()
            market_data.setdefault("steps", payload.steps)
            market_data.setdefault("paths", payload.paths)

            # Convert to Neutryx request
            request = fpml.fpml_to_neutryx(fpml_doc, market_data)

            # Price
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

            # Extract trade info
            trade = fpml_doc.primary_trade
            trade_info = {}
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
            from neutryx.bridge import fpml

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
            from neutryx.bridge import fpml

            fpml.parse_fpml(payload.fpml_xml)
            return {"valid": True, "message": "FpML document is valid"}
        except fpml.FpMLParseError as e:
            return {"valid": False, "message": f"Parse error: {e}"}
        except Exception as e:
            return {"valid": False, "message": f"Validation error: {e}"}

    return app


__all__ = ["create_app"]
