"""
Metadata Tracker for spec.json management.

Tracks metadata and maintains audit trail in spec.json for SDD workflow.

Requirements covered: 12.1, 12.2, 12.3, 12.4, 12.5
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Types of audit events."""

    PHASE_TRANSITION = "phase_transition"
    APPROVAL_GRANTED = "approval_granted"
    VALIDATION_RUN = "validation_run"
    FAST_TRACK_USED = "fast_track_used"


class AuditEvent(BaseModel):
    """Represents an audit trail event."""

    timestamp: datetime
    event_type: AuditEventType
    details: Dict[str, Any]
    user: Optional[str] = None


class SpecMetadata(BaseModel):
    """Represents specification metadata from spec.json."""

    feature_name: str
    created_at: datetime
    updated_at: datetime
    language: str
    phase: str
    approvals: Dict[str, Dict[str, bool]]
    ready_for_implementation: bool
    schema_version: str = "1.0"
    audit_trail: List[AuditEvent] = Field(default_factory=list)


class MetadataTracker:
    """Track metadata and audit trail for specifications."""

    def load_metadata(self, spec_json_path: Path) -> SpecMetadata:
        """
        Load metadata from spec.json.

        Preconditions:
        - spec_json_path exists

        Postconditions:
        - Returns parsed metadata
        - Applies schema migration if needed
        """
        with open(spec_json_path, "r") as f:
            data = json.load(f)

        # Check if migration needed
        if "schema_version" not in data or "audit_trail" not in data:
            return self.migrate_schema(data)

        # Parse datetime fields
        data["created_at"] = datetime.fromisoformat(
            data["created_at"].replace("Z", "+00:00")
        )
        data["updated_at"] = datetime.fromisoformat(
            data["updated_at"].replace("Z", "+00:00")
        )

        # Parse audit trail events
        audit_events = []
        for event_data in data.get("audit_trail", []):
            event_data["timestamp"] = datetime.fromisoformat(
                event_data["timestamp"].replace("Z", "+00:00")
            )
            audit_events.append(AuditEvent(**event_data))

        data["audit_trail"] = audit_events

        return SpecMetadata(**data)

    def save_metadata(self, spec_json_path: Path, metadata: SpecMetadata) -> None:
        """
        Save metadata to spec.json.

        Preconditions:
        - spec_json_path is writable

        Postconditions:
        - spec.json updated with formatted JSON
        - updated_at timestamp refreshed
        """
        # Update timestamp
        metadata.updated_at = datetime.now(timezone.utc)

        # Convert to dict for JSON serialization
        data = metadata.model_dump(mode="json")

        # Format datetime fields as ISO 8601
        data["created_at"] = metadata.created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        data["updated_at"] = metadata.updated_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Format audit trail events
        audit_trail_data = []
        for event in metadata.audit_trail:
            event_dict = {
                "timestamp": event.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_type": event.event_type.value,
                "details": event.details,
            }
            if event.user:
                event_dict["user"] = event.user
            audit_trail_data.append(event_dict)

        data["audit_trail"] = audit_trail_data

        # Write formatted JSON
        with open(spec_json_path, "w") as f:
            json.dump(data, f, indent=2)

    def log_phase_transition(
        self, metadata: SpecMetadata, old_phase: str, new_phase: str
    ) -> SpecMetadata:
        """
        Log phase transition event.

        Returns updated metadata with audit event.
        """
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.PHASE_TRANSITION,
            details={"from": old_phase, "to": new_phase},
        )

        metadata.audit_trail.append(event)
        return metadata

    def log_approval(
        self, metadata: SpecMetadata, approval_type: str, approved: bool
    ) -> SpecMetadata:
        """
        Log approval grant event.

        Returns updated metadata with audit event.
        """
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.APPROVAL_GRANTED,
            details={"approval_type": approval_type, "approved": approved},
        )

        metadata.audit_trail.append(event)
        return metadata

    def log_fast_track(self, metadata: SpecMetadata, command: str) -> SpecMetadata:
        """
        Log fast-track mode usage.

        Returns updated metadata with audit event.
        """
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.FAST_TRACK_USED,
            details={"command": command},
        )

        metadata.audit_trail.append(event)
        return metadata

    def migrate_schema(
        self, old_metadata: Dict[str, Any], target_version: str = "1.0"
    ) -> SpecMetadata:
        """
        Migrate old spec.json to current schema.

        Handles missing audit_trail and schema_version fields.
        """
        # Add schema_version if missing
        if "schema_version" not in old_metadata:
            old_metadata["schema_version"] = target_version

        # Add audit_trail if missing
        if "audit_trail" not in old_metadata:
            old_metadata["audit_trail"] = []

        # Parse datetime fields
        old_metadata["created_at"] = datetime.fromisoformat(
            old_metadata["created_at"].replace("Z", "+00:00")
        )
        old_metadata["updated_at"] = datetime.fromisoformat(
            old_metadata["updated_at"].replace("Z", "+00:00")
        )

        return SpecMetadata(**old_metadata)
