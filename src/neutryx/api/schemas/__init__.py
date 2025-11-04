"""Shared API schema models for Neutryx.

This module contains Pydantic models for API requests and responses,
shared across different components to avoid circular dependencies.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import jax.numpy as jnp
from fastapi import HTTPException
from pydantic import BaseModel, Field


class VanillaOptionRequest(BaseModel):
    """Request model for vanilla option pricing."""

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
    """Request model for exposure or discount profiles."""

    values: List[float]

    def to_array(self) -> jnp.ndarray:
        """Convert values to JAX array."""
        if not self.values:
            raise HTTPException(status_code=400, detail="Expected non-empty sequence")
        return jnp.asarray(self.values, dtype=jnp.float32)


class CVARequest(BaseModel):
    """Request model for CVA calculation."""

    epe: ProfileRequest
    discount: ProfileRequest
    default_probability: ProfileRequest
    lgd: float = Field(0.6, description="Loss given default")


class FVARequest(BaseModel):
    """Request model for FVA calculation."""

    epe: ProfileRequest
    discount: ProfileRequest
    funding_spread: Sequence[float] | float


class MVARequest(BaseModel):
    """Request model for MVA calculation."""

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


__all__ = [
    "VanillaOptionRequest",
    "ProfileRequest",
    "CVARequest",
    "FVARequest",
    "MVARequest",
    "PortfolioXVARequest",
    "PortfolioSummaryRequest",
    "FpMLPriceRequest",
    "FpMLParseRequest",
    "FpMLValidateRequest",
]
