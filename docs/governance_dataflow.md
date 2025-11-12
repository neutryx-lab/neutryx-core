# Data Flow Lineage Operations Guide

The Neutryx platform now exposes a lightweight data-flow recorder that captures
lineage metadata across computational jobs and API requests. This guide
summarises how to activate lineage tracking and how storage/services propagate
the lineage identifier into persisted results.

## Activating a lineage context

Use the `data_flow_context` context manager from
`neutryx.infrastructure.governance` to describe the scope of a computation or
request. The context automatically publishes lifecycle events and makes the
`lineage_id` available to any downstream code.

```python
from neutryx.infrastructure.governance import data_flow_context, record_artifact

with data_flow_context(source="calibration", job_id="gbm-run-42") as ctx:
    # ctx.lineage_id is accessible to embed in custom metadata
    record_artifact("calibration-params", kind="metadata", metadata={"version": 2})
```

When no context is active, a new lineage identifier is generated automatically
for each artefact.

## Storage backends

Both the Zarr and memory-mapped stores enrich saved arrays and metadata with the
active lineage identifier. Additional attributes are attached automatically:

- `storage_backend`: storage implementation (`zarr` or `mmap`)
- `array_shape` / `array_dtype`: persisted array characteristics

Metadata is persisted even when the caller does not supply a metadata payload,
allowing governance services to reconstruct the provenance tree.

## Portfolio API persistence

The REST portfolio stores embed lineage information directly within the stored
`Portfolio` model. Every `save_portfolio` call records a `data_artifact_saved`
event and persists the lineage metadata. Consumers can inspect the
`portfolio.lineage` field to retrieve the provenance identifier associated with
that resource.

## Default recorder management

An in-memory `DataFlowRecorder` retains published events and can fan them out to
subscribers. Use `use_recorder` during tests or specialised workflows to
capture events without modifying global state:

```python
from neutryx.infrastructure.governance import DataFlowRecorder, use_recorder

recorder = DataFlowRecorder()
with use_recorder(recorder):
    ...  # run job/API call

assert recorder.get_events()
```

This mechanism enables integration with observability pipelines or external
governance systems that consume lineage events.
