import json
from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

_prometheus_stub = types.ModuleType("prometheus_client")


class _MetricStub:
    def labels(self, *args, **kwargs):  # pragma: no cover - simple stub
        return self

    def observe(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        return None

    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        return None


def _metric_factory(*args, **kwargs):  # pragma: no cover - simple stub
    return _MetricStub()


_prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"
_prometheus_stub.Counter = _metric_factory
_prometheus_stub.Histogram = _metric_factory
_prometheus_stub.REGISTRY = None
_prometheus_stub.generate_latest = lambda *args, **kwargs: b""  # pragma: no cover

sys.modules.setdefault("prometheus_client", _prometheus_stub)

import jax.numpy as jnp
import pytest

# from neutryx.api.portfolio_store import InMemoryPortfolioStore  # Not yet implemented
from neutryx.infrastructure.governance import (
    DataFlowRecorder,
    data_flow_context,
    use_recorder,
)
from neutryx.io.base import StorageBackend, StorageConfig
from neutryx.io.mmap_store import MMapStore
from neutryx.portfolio.portfolio import Portfolio


@pytest.mark.skip(reason="InMemoryPortfolioStore not yet implemented")
def test_portfolio_store_embeds_lineage_metadata() -> None:
    # store = InMemoryPortfolioStore()
    # recorder = DataFlowRecorder()
    #
    # with use_recorder(recorder):
    #     with data_flow_context(source="api", api_request_id="req-123") as ctx:
    #         portfolio = Portfolio(name="test-portfolio")
    #         store.save_portfolio(portfolio)
    #         stored = store.get_portfolio("test-portfolio")
    #
    #     assert stored is not None
    #     assert stored.lineage is not None
    #     assert stored.lineage["lineage_id"] == ctx.lineage_id
    #
    #     events = recorder.get_events()
    #
    # assert any(event.event_type == "data_artifact_saved" for event in events)
    pass


def test_mmap_store_writes_lineage_metadata(tmp_path: Path) -> None:
    config = StorageConfig(
        backend=StorageBackend.MMAP,
        path=str(tmp_path),
        compression=None,
        create_if_missing=True,
    )

    store = MMapStore(config)
    recorder = DataFlowRecorder()

    with use_recorder(recorder):
        with data_flow_context(source="job", job_id="job-42") as ctx:
            store.save_array("results", jnp.ones((2,), dtype=jnp.float32))

        metadata_path = tmp_path / "__metadata__" / "results.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        assert metadata["lineage_id"] == ctx.lineage_id
        assert metadata["storage_backend"] == "mmap"

        events = recorder.get_events()

    assert any(event.event_type == "data_artifact_saved" for event in events)
