"""Tests for MarketDataEnvironment helper methods."""

from __future__ import annotations

from datetime import date

from neutryx.market.environment import MarketDataEnvironment


class DummyCurve:
    def value(self, t):  # pragma: no cover - simple protocol stub
        return t


class DummyForwardCurve(DummyCurve):
    pass


class DummyFxVolSurface:
    def implied_vol(self, expiry, strike):  # pragma: no cover - simple stub
        return expiry if isinstance(expiry, (int, float)) else 0.0


def _base_environment() -> MarketDataEnvironment:
    return MarketDataEnvironment(reference_date=date(2024, 1, 1))


def test_with_credit_curve_returns_new_environment_without_mutation():
    env = _base_environment()
    curve = DummyCurve()

    updated = env.with_credit_curve("CorpA", "Senior", curve)

    assert updated is not env
    assert ("CorpA", "Senior") not in env.credit_curves
    assert updated.credit_curves[("CorpA", "Senior")] is curve

    replacement = DummyCurve()
    replaced = updated.with_credit_curve("CorpA", "Senior", replacement)

    assert replaced is not updated
    assert updated.credit_curves[("CorpA", "Senior")] is curve
    assert replaced.credit_curves[("CorpA", "Senior")] is replacement


def test_with_fx_forward_curve_updates_pair_without_mutation():
    env = _base_environment()
    curve = DummyForwardCurve()

    updated = env.with_fx_forward_curve("EUR", "USD", curve)

    assert updated is not env
    assert ("EUR", "USD") not in env.fx_forward_curves
    assert updated.fx_forward_curves[("EUR", "USD")] is curve

    replacement = DummyForwardCurve()
    replaced = updated.with_fx_forward_curve("EUR", "USD", replacement)

    assert replaced is not updated
    assert updated.fx_forward_curves[("EUR", "USD")] is curve
    assert replaced.fx_forward_curves[("EUR", "USD")] is replacement


def test_with_fx_vol_surface_updates_pair_without_mutation():
    env = _base_environment()
    surface = DummyFxVolSurface()

    updated = env.with_fx_vol_surface("EUR", "USD", surface)

    assert updated is not env
    assert ("EUR", "USD") not in env.fx_vol_surfaces
    assert updated.fx_vol_surfaces[("EUR", "USD")] is surface

    replacement = DummyFxVolSurface()
    replaced = updated.with_fx_vol_surface("EUR", "USD", replacement)

    assert replaced is not updated
    assert updated.fx_vol_surfaces[("EUR", "USD")] is surface
    assert replaced.fx_vol_surfaces[("EUR", "USD")] is replacement

