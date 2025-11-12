from __future__ import annotations

from datetime import date

import jax.numpy as jnp
import pytest

from neutryx.market.credit.hazard import HazardRateCurve, SurvivalProbabilityCurve
from neutryx.market.environment import MarketDataEnvironment


def test_environment_uses_survival_probability_curve():
    curve = SurvivalProbabilityCurve(
        maturities=jnp.array([1.0, 3.0, 5.0]),
        survival_probabilities=jnp.array([0.97, 0.9, 0.78]),
    )
    env = MarketDataEnvironment(
        reference_date=date(2024, 1, 1),
        credit_curves={("CORP_A", "SENIOR"): curve},
    )

    times = jnp.array([0.0, 1.0, 2.0, 4.0, 6.0])
    survival = env.get_survival_probability("CORP_A", "SENIOR", times)

    assert bool(jnp.all(survival <= 1.0))
    assert bool(jnp.all(survival >= 0.0))
    # Survival curve should be non-increasing in time
    assert bool(jnp.all(survival[1:] <= survival[:-1] + 1e-12))


def test_with_credit_curve_accepts_hazard_curve():
    hazard_curve = HazardRateCurve(
        maturities=jnp.array([1.0, 2.0, 5.0]),
        intensities=jnp.array([0.01, 0.015, 0.02]),
    )
    env = MarketDataEnvironment(reference_date=date(2024, 1, 1))
    updated = env.with_credit_curve("CORP_B", "SENIOR", hazard_curve)

    query_time = jnp.array([1.0, 3.0])
    expected = hazard_curve.survival_probability(query_time)
    result = updated.get_survival_probability("CORP_B", "SENIOR", query_time)
    assert bool(jnp.allclose(result, expected))


def test_with_credit_curve_rejects_invalid_curve():
    class InvalidCurve:
        pass

    env = MarketDataEnvironment(reference_date=date(2024, 1, 1))

    with pytest.raises(TypeError):
        env.with_credit_curve("CORP_C", "SENIOR", InvalidCurve())  # type: ignore[arg-type]
