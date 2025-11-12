"""Tests for governance data flow lineage recording."""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import pytest


prometheus_stub = types.ModuleType("prometheus_client")
prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"


class _DummyCollector:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial container
        self._labels = {}

    def labels(self, *args, **kwargs) -> "_DummyCollector":
        return self

    def observe(self, *args, **kwargs) -> None:  # pragma: no cover - trivial container
        return None

    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - trivial container
        return None

    def time(self, *args, **kwargs):  # pragma: no cover - trivial container
        return self

    def __enter__(self):  # pragma: no cover - trivial container
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial container
        return None


prometheus_stub.Counter = lambda *args, **kwargs: _DummyCollector()
prometheus_stub.Histogram = lambda *args, **kwargs: _DummyCollector()
prometheus_stub.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
prometheus_stub.generate_latest = lambda *args, **kwargs: b""
sys.modules.setdefault("prometheus_client", prometheus_stub)

from neutryx.api.portfolio_store import InMemoryPortfolioStore
from neutryx.infrastructure.governance import (
    DataFlowInMemorySink,
    DataFlowRecorder,
    get_dataflow_recorder,
    set_dataflow_recorder,
)
from neutryx.infrastructure.workflows import CheckpointManager
from neutryx.portfolio.portfolio import Portfolio


@pytest.fixture()
def recorder_sink() -> tuple[DataFlowRecorder, DataFlowInMemorySink]:
    """Provide an isolated recorder + sink for lineage assertions."""

    original = get_dataflow_recorder()
    sink = DataFlowInMemorySink()
    recorder = DataFlowRecorder([sink])
    set_dataflow_recorder(recorder)
    try:
        yield recorder, sink
    finally:
        set_dataflow_recorder(original)


def test_record_flow_generates_lineage(recorder_sink: tuple[DataFlowRecorder, DataFlowInMemorySink]) -> None:
    recorder, sink = recorder_sink
    record = recorder.record_flow(job_id="test-job", source="unit-test", inputs={"key": "value"})

    assert record.lineage_id
    assert sink.records()[-1].lineage_id == record.lineage_id
    assert sink.records()[-1].inputs["key"] == "value"


def test_checkpoint_manager_embeds_lineage(recorder_sink: tuple[DataFlowRecorder, DataFlowInMemorySink], tmp_path: Path) -> None:
    recorder, sink = recorder_sink
    manager = CheckpointManager(directory=tmp_path)
    state: dict[str, object] = {"value": 42}

    manager.save(3, state)

    record = sink.records()[-1]
    assert record.job_id.startswith("checkpoint:")
    assert state["_metadata"]["lineage_id"] == record.lineage_id

    saved = manager._state_path(3)  # noqa: SLF001 - internal path used for verification
    with saved.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["state"]["_metadata"]["lineage_id"] == record.lineage_id


def test_portfolio_store_embeds_lineage(recorder_sink: tuple[DataFlowRecorder, DataFlowInMemorySink]) -> None:
    recorder, sink = recorder_sink
    store = InMemoryPortfolioStore()
    portfolio = Portfolio(name="demo-portfolio")

    store.save_portfolio(portfolio)

    record = sink.records()[-1]
    assert record.context["backend"] == "memory"
    assert portfolio.metadata["lineage_id"] == record.lineage_id

    stored = store.get_portfolio("demo-portfolio")
    assert stored is not None
    assert stored.metadata["lineage_id"] == record.lineage_id
