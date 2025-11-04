"""High-level orchestration for regulatory capital calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from neutryx.valuations.regulatory.basel import (
    BaselCapitalCalculator,
    BaselCapitalInputs,
    BaselCapitalResult,
    BaselExposure,
)
from neutryx.valuations.regulatory.frtb import (
    FRTBResult,
    FRTBSensitivity,
    FRTBStandardizedApproach,
)
from neutryx.valuations.regulatory.saccr import (
    SACCRCalculator,
    SACCRResult,
    SACCRTrade,
)
from neutryx.valuations.simm.calculator import SIMMCalculator, SIMMResult
from neutryx.valuations.simm.sensitivities import RiskFactorSensitivity


@dataclass(frozen=True)
class RegulatoryCapitalSummary:
    """Container for consolidated regulatory capital outputs."""

    frtb: Optional[FRTBResult]
    saccr: Optional[SACCRResult]
    saccr_capital_requirement: Optional[float]
    basel: Optional[BaselCapitalResult]
    simm: Optional[SIMMResult]
    initial_margin: Optional[float]
    total_capital_requirement: Optional[float]

    def as_dict(self) -> dict[str, Optional[float]]:
        """Summarise key headline numbers as a dictionary."""
        return {
            "frtb_total": self.frtb.total_capital if self.frtb else None,
            "saccr_ead": self.saccr.ead if self.saccr else None,
            "saccr_capital_requirement": self.saccr_capital_requirement,
            "basel_required_total_capital": (
                self.basel.required_total_capital if self.basel else None
            ),
            "initial_margin": self.initial_margin,
            "total_capital_requirement": self.total_capital_requirement,
        }


class RegulatoryCapitalEngine:
    """Coordinate FRTB, SA-CCR, Basel III, and SIMM calculations."""

    def __init__(
        self,
        *,
        frtb_calculator: Optional[FRTBStandardizedApproach] = None,
        saccr_calculator: Optional[SACCRCalculator] = None,
        basel_calculator: Optional[BaselCapitalCalculator] = None,
        simm_calculator: Optional[SIMMCalculator] = None,
    ) -> None:
        self._frtb = frtb_calculator or FRTBStandardizedApproach()
        self._saccr = saccr_calculator or SACCRCalculator()
        self._basel = basel_calculator or BaselCapitalCalculator()
        self._simm = simm_calculator or SIMMCalculator()

    def run(
        self,
        *,
        frtb_sensitivities: Optional[Sequence[FRTBSensitivity]] = None,
        saccr_trades: Optional[Sequence[SACCRTrade]] = None,
        saccr_mark_to_market: float = 0.0,
        saccr_collateral: float = 0.0,
        saccr_risk_weight: Optional[float] = None,
        saccr_capital_ratio: float = 0.08,
        simm_sensitivities: Optional[Sequence[RiskFactorSensitivity]] = None,
        basel_exposures: Optional[Iterable[BaselExposure]] = None,
        basel_capital_inputs: Optional[BaselCapitalInputs] = None,
        basel_override_rwa: Optional[float] = None,
    ) -> RegulatoryCapitalSummary:
        """Execute all requested capital calculations and aggregate results."""

        frtb_result = (
            self._frtb.calculate(frtb_sensitivities)
            if frtb_sensitivities
            else None
        )

        saccr_result = None
        saccr_capital_requirement = None
        if saccr_trades:
            saccr_result = self._saccr.calculate(
                saccr_trades,
                mark_to_market=saccr_mark_to_market,
                collateral=saccr_collateral,
            )
            if saccr_risk_weight is not None:
                saccr_capital_requirement = saccr_result.capital_requirement(
                    saccr_risk_weight,
                    capital_ratio=saccr_capital_ratio,
                )

        simm_result = (
            self._simm.calculate(list(simm_sensitivities))
            if simm_sensitivities
            else None
        )

        basel_result = None
        if basel_capital_inputs is not None and (basel_exposures or basel_override_rwa):
            if basel_override_rwa is not None:
                rwa = basel_override_rwa
            else:
                rwa = self._basel.calculate_rwa(basel_exposures or [])
            basel_result = self._basel.assess_capital(basel_capital_inputs, rwa)

        total_capital_requirement: Optional[float] = None
        if basel_result is not None:
            total_capital_requirement = basel_result.required_total_capital
        elif saccr_capital_requirement is not None:
            total_capital_requirement = saccr_capital_requirement

        initial_margin = simm_result.total_im if simm_result else None

        return RegulatoryCapitalSummary(
            frtb=frtb_result,
            saccr=saccr_result,
            saccr_capital_requirement=saccr_capital_requirement,
            basel=basel_result,
            simm=simm_result,
            initial_margin=initial_margin,
            total_capital_requirement=total_capital_requirement,
        )
