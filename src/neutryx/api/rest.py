"""REST API exposing pricing workflows via FastAPI."""
from __future__ import annotations

from typing import List, Sequence

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

    return app


__all__ = ["create_app"]
