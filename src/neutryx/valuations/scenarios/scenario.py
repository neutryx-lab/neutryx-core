"""Scenario definition and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Shock:
    """Definition of a risk factor shock.

    Parameters
    ----------
    risk_factor:
        Name of the risk factor being shocked (e.g. ``"IR_USD_1Y"``).
    shift:
        Magnitude of the shock. For ``kind="absolute"`` the shift represents
        an absolute move (e.g. +10 bps). For ``kind="relative"`` it represents
        a relative move expressed in fractional terms (e.g. ``0.01`` for a 1%
        increase).
    measure:
        Portfolio sensitivity measure used when translating the shock into PnL.
        Defaults to ``"delta"``.
    kind:
        Either ``"absolute"`` or ``"relative"``.
    """

    risk_factor: str
    shift: float
    measure: str = "delta"
    kind: str = "absolute"

    def impact(self, sensitivity: float, *, reference: Optional[float] = None) -> float:
        """Translate the configured shift into a PnL contribution."""

        if self.kind == "absolute":
            return sensitivity * self.shift
        if self.kind == "relative":
            if reference is None:
                raise ValueError(
                    "Relative shocks require a 'base' reference value in the "
                    "aggregated sensitivities."
                )
            return sensitivity * reference * self.shift
        raise ValueError(f"Unsupported shock kind: {self.kind}")


@dataclass
class ScenarioResult:
    """Holds the outcome of a scenario evaluation."""

    name: str
    contributions: Mapping[str, float]

    @property
    def total(self) -> float:
        return float(sum(self.contributions.values()))

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario": self.name,
            "total": self.total,
            "contributions": dict(self.contributions),
        }


@dataclass
class Scenario:
    """A scenario composed of multiple risk factor shocks."""

    name: str
    shocks: Tuple[Shock, ...] = field(default_factory=tuple)

    def evaluate(
        self, aggregated_sensitivities: Mapping[str, Mapping[str, float]]
    ) -> ScenarioResult:
        """Evaluate the scenario against portfolio sensitivities."""

        contributions: Dict[str, float] = {}
        for shock in self.shocks:
            measures = aggregated_sensitivities.get(shock.risk_factor, {})
            sensitivity = float(measures.get(shock.measure, 0.0))
            reference = measures.get("base")
            contributions[shock.risk_factor] = shock.impact(
                sensitivity, reference=reference
            )
        return ScenarioResult(self.name, dict(contributions))


@dataclass
class ScenarioReport:
    """Collection of scenario results with reporting helpers."""

    results: Tuple[ScenarioResult, ...]

    def totals(self) -> Dict[str, float]:
        return {result.name: result.total for result in self.results}

    def format_totals(self, *, precision: int = 4) -> str:
        header = f"{'Scenario':<20} {'PnL':>16}"
        lines = [header, "-" * len(header)]
        for result in self.results:
            lines.append(f"{result.name:<20} {result.total:>16.{precision}f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {result.name: result.to_dict() for result in self.results}


@dataclass
class ScenarioSet:
    """Container grouping multiple scenarios."""

    scenarios: Tuple[Scenario, ...]

    def evaluate(
        self, aggregated_sensitivities: Mapping[str, Mapping[str, float]]
    ) -> ScenarioReport:
        results = tuple(
            scenario.evaluate(aggregated_sensitivities) for scenario in self.scenarios
        )
        return ScenarioReport(results)
