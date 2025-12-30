"""
Test suite for Metadata Tracker for spec.json management.

Requirements covered: 12.1, 12.2, 12.3, 12.4, 12.5
"""
from pathlib import Path
from datetime import datetime
import json
import pytest
from .metadata_tracker import (
    MetadataTracker,
    SpecMetadata,
    AuditEvent,
    AuditEventType,
)


class TestMetadataTracker:
    """Test metadata tracker for spec.json management."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return MetadataTracker()

    @pytest.fixture
    def sample_spec_json(self, tmp_path):
        """Create sample spec.json file."""
        spec_data = {
            "feature_name": "test-feature",
            "created_at": "2025-12-30T00:00:00Z",
            "updated_at": "2025-12-30T10:00:00Z",
            "language": "en",
            "phase": "requirements-generated",
            "approvals": {
                "requirements": {"generated": True, "approved": True},
                "design": {"generated": False, "approved": False},
                "tasks": {"generated": False, "approved": False},
            },
            "ready_for_implementation": False,
        }
        file_path = tmp_path / "spec.json"
        file_path.write_text(json.dumps(spec_data, indent=2))
        return file_path

    @pytest.fixture
    def spec_json_with_audit(self, tmp_path):
        """Create spec.json with existing audit trail."""
        spec_data = {
            "feature_name": "test-feature",
            "created_at": "2025-12-30T00:00:00Z",
            "updated_at": "2025-12-30T10:00:00Z",
            "language": "en",
            "phase": "tasks-generated",
            "approvals": {
                "requirements": {"generated": True, "approved": True},
                "design": {"generated": True, "approved": True},
                "tasks": {"generated": True, "approved": True},
            },
            "ready_for_implementation": True,
            "schema_version": "1.0",
            "audit_trail": [
                {
                    "timestamp": "2025-12-30T08:00:00Z",
                    "event_type": "phase_transition",
                    "details": {"from": "initialized", "to": "requirements-generated"},
                }
            ],
        }
        file_path = tmp_path / "spec.json"
        file_path.write_text(json.dumps(spec_data, indent=2))
        return file_path

    def test_tracker_initialization(self, tracker):
        """Test tracker can be initialized."""
        assert tracker is not None
        assert isinstance(tracker, MetadataTracker)

    def test_load_metadata_from_file(self, tracker, sample_spec_json):
        """Test loading metadata from spec.json file."""
        metadata = tracker.load_metadata(sample_spec_json)

        assert metadata.feature_name == "test-feature"
        assert metadata.language == "en"
        assert metadata.phase == "requirements-generated"
        assert metadata.approvals["requirements"]["approved"] is True
        assert metadata.ready_for_implementation is False

    def test_load_metadata_applies_migration(self, tracker, sample_spec_json):
        """Test loading old spec.json applies schema migration."""
        metadata = tracker.load_metadata(sample_spec_json)

        # Should have default schema_version
        assert metadata.schema_version == "1.0"
        # Should have empty audit trail
        assert metadata.audit_trail == []

    def test_load_metadata_preserves_audit_trail(self, tracker, spec_json_with_audit):
        """Test loading spec.json preserves existing audit trail."""
        metadata = tracker.load_metadata(spec_json_with_audit)

        assert len(metadata.audit_trail) == 1
        assert metadata.audit_trail[0].event_type == AuditEventType.PHASE_TRANSITION
        assert metadata.audit_trail[0].details["from"] == "initialized"

    def test_save_metadata_updates_timestamp(self, tracker, sample_spec_json):
        """Test saving metadata updates the updated_at timestamp."""
        metadata = tracker.load_metadata(sample_spec_json)
        original_updated_at = metadata.updated_at

        # Save metadata
        tracker.save_metadata(sample_spec_json, metadata)

        # Reload and check timestamp
        reloaded = tracker.load_metadata(sample_spec_json)
        assert reloaded.updated_at > original_updated_at

    def test_save_metadata_preserves_data(self, tracker, sample_spec_json):
        """Test saving metadata preserves all data correctly."""
        metadata = tracker.load_metadata(sample_spec_json)
        metadata.phase = "design-generated"

        tracker.save_metadata(sample_spec_json, metadata)

        # Reload and verify
        reloaded = tracker.load_metadata(sample_spec_json)
        assert reloaded.phase == "design-generated"
        assert reloaded.feature_name == "test-feature"

    def test_log_phase_transition(self, tracker, sample_spec_json):
        """Test logging phase transition event."""
        metadata = tracker.load_metadata(sample_spec_json)

        updated_metadata = tracker.log_phase_transition(
            metadata, old_phase="requirements-generated", new_phase="design-generated"
        )

        assert len(updated_metadata.audit_trail) == 1
        event = updated_metadata.audit_trail[0]
        assert event.event_type == AuditEventType.PHASE_TRANSITION
        assert event.details["from"] == "requirements-generated"
        assert event.details["to"] == "design-generated"
        assert event.timestamp is not None

    def test_log_approval(self, tracker, sample_spec_json):
        """Test logging approval grant event."""
        metadata = tracker.load_metadata(sample_spec_json)

        updated_metadata = tracker.log_approval(
            metadata, approval_type="requirements", approved=True
        )

        assert len(updated_metadata.audit_trail) == 1
        event = updated_metadata.audit_trail[0]
        assert event.event_type == AuditEventType.APPROVAL_GRANTED
        assert event.details["approval_type"] == "requirements"
        assert event.details["approved"] is True

    def test_log_fast_track(self, tracker, sample_spec_json):
        """Test logging fast-track mode usage."""
        metadata = tracker.load_metadata(sample_spec_json)

        updated_metadata = tracker.log_fast_track(
            metadata, command="/kiro:spec-design -y"
        )

        assert len(updated_metadata.audit_trail) == 1
        event = updated_metadata.audit_trail[0]
        assert event.event_type == AuditEventType.FAST_TRACK_USED
        assert event.details["command"] == "/kiro:spec-design -y"

    def test_audit_trail_append_only(self, tracker, spec_json_with_audit):
        """Test audit trail is append-only (never removes events)."""
        metadata = tracker.load_metadata(spec_json_with_audit)
        initial_count = len(metadata.audit_trail)

        # Add new event
        updated_metadata = tracker.log_approval(metadata, "design", True)

        assert len(updated_metadata.audit_trail) == initial_count + 1
        # Original event still present
        assert updated_metadata.audit_trail[0].event_type == AuditEventType.PHASE_TRANSITION

    def test_migrate_schema_adds_audit_trail(self, tracker):
        """Test schema migration adds audit_trail if missing."""
        old_metadata = {
            "feature_name": "old-feature",
            "created_at": "2025-12-30T00:00:00Z",
            "updated_at": "2025-12-30T00:00:00Z",
            "language": "en",
            "phase": "initialized",
            "approvals": {},
            "ready_for_implementation": False,
        }

        migrated = tracker.migrate_schema(old_metadata)

        assert migrated.schema_version == "1.0"
        assert migrated.audit_trail == []

    def test_migrate_schema_adds_schema_version(self, tracker):
        """Test schema migration adds schema_version if missing."""
        old_metadata = {
            "feature_name": "old-feature",
            "created_at": "2025-12-30T00:00:00Z",
            "updated_at": "2025-12-30T00:00:00Z",
            "language": "en",
            "phase": "initialized",
            "approvals": {},
            "ready_for_implementation": False,
            "audit_trail": [],
        }

        migrated = tracker.migrate_schema(old_metadata)

        assert migrated.schema_version == "1.0"

    def test_multiple_events_maintain_chronological_order(self, tracker, sample_spec_json):
        """Test audit trail maintains chronological order."""
        metadata = tracker.load_metadata(sample_spec_json)

        # Add multiple events
        metadata = tracker.log_phase_transition(metadata, "init", "requirements-generated")
        metadata = tracker.log_approval(metadata, "requirements", True)
        metadata = tracker.log_phase_transition(metadata, "requirements-generated", "design-generated")

        # Check chronological order
        assert len(metadata.audit_trail) == 3
        assert metadata.audit_trail[0].event_type == AuditEventType.PHASE_TRANSITION
        assert metadata.audit_trail[1].event_type == AuditEventType.APPROVAL_GRANTED
        assert metadata.audit_trail[2].event_type == AuditEventType.PHASE_TRANSITION

        # Timestamps should be increasing
        assert metadata.audit_trail[0].timestamp <= metadata.audit_trail[1].timestamp
        assert metadata.audit_trail[1].timestamp <= metadata.audit_trail[2].timestamp
