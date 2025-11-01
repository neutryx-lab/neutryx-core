"""Utility functions for aggregating portfolio level metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping


def aggregate(trade_values: Iterable[float]) -> float:
    """Aggregate a sequence of trade values into a portfolio total."""

    return float(sum(trade_values))


def aggregate_sensitivities(
    trade_sensitivities: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate trade level sensitivities into portfolio level measures.

    Parameters
    ----------
    trade_sensitivities:
        Nested mapping keyed by trade identifier and then risk factor name.
        The innermost mapping contains measure names (``"delta"``, ``"vega``",
        etc.) mapped to numeric sensitivity values.

    Returns
    -------
    dict
        Dictionary keyed by risk factor with measure/value pairs aggregated
        across the provided trades.
    """

    aggregated: MutableMapping[str, MutableMapping[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for _, per_risk_factor in trade_sensitivities.items():
        for risk_factor, measures in per_risk_factor.items():
            agg_measures = aggregated[risk_factor]
            for measure, value in measures.items():
                if measure == "base":
                    if (
                        measure in agg_measures
                        and agg_measures[measure] != float(value)
                    ):
                        raise ValueError(
                            "Inconsistent base values encountered for risk factor "
                            f"'{risk_factor}' while aggregating sensitivities."
                        )
                    agg_measures[measure] = float(value)
                else:
                    agg_measures[measure] += float(value)

    return {risk_factor: dict(measures) for risk_factor, measures in aggregated.items()}


def format_sensitivity_report(
    aggregated_sensitivities: Mapping[str, Mapping[str, float]], *, precision: int = 4
) -> str:
    """Create a human readable sensitivity report.

    Parameters
    ----------
    aggregated_sensitivities:
        Mapping produced by :func:`aggregate_sensitivities`.
    precision:
        Number of decimal places displayed in the output report.
    """

    header = f"{'Risk Factor':<18} {'Measure':<10} {'Value':>16}"
    lines = [header, "-" * len(header)]

    for risk_factor in sorted(aggregated_sensitivities):
        measures = aggregated_sensitivities[risk_factor]
        for measure in sorted(measures):
            if measure == "base":
                continue
            value = measures[measure]
            lines.append(
                f"{risk_factor:<18} {measure:<10} {value:>16.{precision}f}"
            )

    return "\n".join(lines)
