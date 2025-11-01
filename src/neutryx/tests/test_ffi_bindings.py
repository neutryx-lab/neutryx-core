from __future__ import annotations

import types

import pytest

from neutryx.ffi import eigen, quantlib


def test_quantlib_binding_reports_availability() -> None:
    module = quantlib.get_quantlib_module()
    if quantlib.quantlib_available():
        assert isinstance(module, types.ModuleType)
    else:
        assert module is None
        fallback = quantlib.load_quantlib(prefer_fallback=True)
        with pytest.raises(RuntimeError):
            getattr(fallback, "SomeClass")


def test_eigen_binding_returns_stub_when_missing() -> None:
    module = eigen.get_eigen_module()
    if eigen.eigen_available():
        assert isinstance(module, types.ModuleType)
    else:
        assert module is None
        stub = eigen.load_eigen_kernels()
        with pytest.raises(RuntimeError):
            getattr(stub, "dot")


def test_require_helpers_raise_meaningful_errors() -> None:
    if quantlib.quantlib_available():
        assert quantlib.require_quantlib()
    else:
        with pytest.raises(RuntimeError):
            quantlib.require_quantlib()

    if eigen.eigen_available():
        assert eigen.require_eigen()
    else:
        with pytest.raises(RuntimeError):
            eigen.require_eigen()
