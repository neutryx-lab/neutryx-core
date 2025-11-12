# Data Flow Lineage Recording

This document describes how Neutryx records lineage metadata for compute jobs
and API driven persistence operations.

## Recorder overview

Lineage events are handled by `neutryx.infrastructure.governance.dataflow`.  The
module exposes a global `DataFlowRecorder` instance that generates lineage
identifiers and forwards structured `DataFlowRecord` events to registered sinks.
A lightweight in-memory sink is provided for testing.  Production deployments
can register custom sinks that forward events to the governance platform of
choice.

```python
from neutryx.infrastructure.governance import (
    DataFlowInMemorySink,
    get_dataflow_recorder,
)

recorder = get_dataflow_recorder()
recorder.register_sink(DataFlowInMemorySink())
```

Each record captures the job identifier, source component, timestamps, inputs,
outputs, and an immutable `lineage_id`.

## Automatic hooks

Two persistence code paths now emit lineage metadata automatically:

* `CheckpointManager.save` (core workflows) stores the lineage identifier under
  the `_metadata.lineage_id` key of the checkpoint state before serialisation.
* `PortfolioStore.save_portfolio` (API persistence) injects the lineage
  identifier into the `metadata` field of the `Portfolio` model.

The emitted records include useful context such as the storage backend, step
number, and structural information about the saved objects.  The identifiers can
be correlated with downstream artefacts produced by compute jobs.

## Testing and observability

Unit tests can replace the global recorder via `set_dataflow_recorder` to assert
that lineage events are emitted.  The helper ensures tests remain isolated while
reusing the production hooks:

```python
from neutryx.infrastructure.governance import (
    DataFlowInMemorySink,
    DataFlowRecorder,
    set_dataflow_recorder,
)

sink = DataFlowInMemorySink()
recorder = DataFlowRecorder([sink])
set_dataflow_recorder(recorder)
# ... run code under test ...
assert sink.records()
```

Administrators should monitor sink delivery errors and rotate lineage records
according to organisational governance policies.
