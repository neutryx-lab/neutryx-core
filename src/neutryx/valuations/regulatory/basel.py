"""Basel III capital ratio assessment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import jax.numpy as jnp


@dataclass(frozen=True)
class BaselExposure:
    """Single exposure amount with an associated Basel risk weight."""

    amount: float
    risk_weight: float
    description: str | None = None

    def risk_weighted_assets(self) -> float:
        """Return the risk-weighted asset contribution."""
        return self.amount * self.risk_weight


@dataclass(frozen=True)
class BaselCapitalInputs:
    """Regulatory capital components and leverage exposure."""

    cet1: float
    additional_tier1: float
    tier2: float
    leverage_exposure: float

    @property
    def tier1(self) -> float:
        """Common Equity Tier 1 + Additional Tier 1."""
        return self.cet1 + self.additional_tier1

    @property
    def total_capital(self) -> float:
        """Total regulatory capital (Tier 1 + Tier 2)."""
        return self.tier1 + self.tier2


@dataclass(frozen=True)
class BaselCapitalResult:
    """Summary of Basel III capital ratios and requirements."""

    rwa: float
    cet1_ratio: float
    tier1_ratio: float
    total_capital_ratio: float
    leverage_ratio: float
    required_cet1: float
    required_tier1: float
    required_total_capital: float
    surplus_cet1: float
    surplus_tier1: float
    surplus_total_capital: float
    meets_cet1_requirement: bool
    meets_tier1_requirement: bool
    meets_total_requirement: bool
    meets_leverage_requirement: bool

    def as_dict(self) -> Mapping[str, float | bool]:
        """Return a dictionary representation of the result."""
        return {
            "rwa": self.rwa,
            "cet1_ratio": self.cet1_ratio,
            "tier1_ratio": self.tier1_ratio,
            "total_capital_ratio": self.total_capital_ratio,
            "leverage_ratio": self.leverage_ratio,
            "required_cet1": self.required_cet1,
            "required_tier1": self.required_tier1,
            "required_total_capital": self.required_total_capital,
            "surplus_cet1": self.surplus_cet1,
            "surplus_tier1": self.surplus_tier1,
            "surplus_total_capital": self.surplus_total_capital,
            "meets_cet1_requirement": self.meets_cet1_requirement,
            "meets_tier1_requirement": self.meets_tier1_requirement,
            "meets_total_requirement": self.meets_total_requirement,
            "meets_leverage_requirement": self.meets_leverage_requirement,
        }


class BaselCapitalCalculator:
    """Assess Basel III capital ratios and requirements."""

    MIN_CET1_RATIO = 0.045
    MIN_TIER1_RATIO = 0.06
    MIN_TOTAL_RATIO = 0.08
    DEFAULT_LEVERAGE_RATIO = 0.03

    def __init__(
        self,
        *,
        capital_conservation_buffer: float = 0.025,
        countercyclical_buffer: float = 0.0,
        gsib_buffer: float = 0.0,
        leverage_ratio_min: float | None = None,
    ) -> None:
        self.capital_conservation_buffer = capital_conservation_buffer
        self.countercyclical_buffer = countercyclical_buffer
        self.gsib_buffer = gsib_buffer
        self.leverage_ratio_min = leverage_ratio_min or self.DEFAULT_LEVERAGE_RATIO

    @staticmethod
    def calculate_rwa(exposures: Iterable[BaselExposure]) -> float:
        """Aggregate risk-weighted assets for the provided exposure set."""
        return float(sum(exp.risk_weighted_assets() for exp in exposures))

    def assess_capital(
        self,
        capital: BaselCapitalInputs,
        rwa: float,
    ) -> BaselCapitalResult:
        """Compute Basel III ratios and determine surplus/shortfall."""
        buffer_total = self.capital_conservation_buffer + self.countercyclical_buffer + self.gsib_buffer

        cet1_requirement_ratio = self.MIN_CET1_RATIO + buffer_total
        tier1_requirement_ratio = self.MIN_TIER1_RATIO + buffer_total
        total_requirement_ratio = self.MIN_TOTAL_RATIO + buffer_total

        cet1_ratio = _safe_div(capital.cet1, rwa)
        tier1_ratio = _safe_div(capital.tier1, rwa)
        total_capital_ratio = _safe_div(capital.total_capital, rwa)
        leverage_ratio = _safe_div(capital.tier1, capital.leverage_exposure)

        required_cet1 = rwa * cet1_requirement_ratio
        required_tier1 = rwa * tier1_requirement_ratio
        required_total = rwa * total_requirement_ratio

        surplus_cet1 = capital.cet1 - required_cet1
        surplus_tier1 = capital.tier1 - required_tier1
        surplus_total = capital.total_capital - required_total

        return BaselCapitalResult(
            rwa=rwa,
            cet1_ratio=cet1_ratio,
            tier1_ratio=tier1_ratio,
            total_capital_ratio=total_capital_ratio,
            leverage_ratio=leverage_ratio,
            required_cet1=required_cet1,
            required_tier1=required_tier1,
            required_total_capital=required_total,
            surplus_cet1=surplus_cet1,
            surplus_tier1=surplus_tier1,
            surplus_total_capital=surplus_total,
            meets_cet1_requirement=cet1_ratio >= cet1_requirement_ratio,
            meets_tier1_requirement=tier1_ratio >= tier1_requirement_ratio,
            meets_total_requirement=total_capital_ratio >= total_requirement_ratio,
            meets_leverage_requirement=leverage_ratio >= self.leverage_ratio_min,
        )


def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator / denominator guarding against division by zero."""
    if jnp.isclose(denominator, 0.0):
        return jnp.inf if numerator > 0 else 0.0
    return float(numerator / denominator)
