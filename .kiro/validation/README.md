# SDD Validation Framework

Core validation utilities for Kiro-style Spec-Driven Development.

## Overview

This package provides the foundation layer for the SDD Remediation feature, implementing validation utilities that enforce quality standards throughout the specification workflow.

## Components

### 1. MarkdownParser (`markdown_parser.py`)
Parses Markdown documents (requirements.md, design.md, tasks.md) to extract structured data.

**Capabilities:**
- Extract requirement headings with numeric ID detection
- Parse acceptance criteria from nested lists
- Extract task items with hierarchy, details, and requirement references
- Handle code blocks and tables

**Requirements Covered:** 3.1, 3.2, 4.1, 4.2

### 2. MetadataTracker (`metadata_tracker.py`)
Manages spec.json metadata and maintains audit trail.

**Capabilities:**
- Load/save spec.json with schema validation
- Track phase transitions and approvals
- Maintain chronological audit trail
- Schema migration for backward compatibility

**Requirements Covered:** 12.1, 12.2, 12.3, 12.4, 12.5

### 3. SteeringLoader (`steering_loader.py`)
Loads steering context files for specification generation.

**Capabilities:**
- Recursive loading of `.kiro/steering/` directory
- Validate default steering files exist
- Token budget-based prioritization
- Warning generation for missing files

**Requirements Covered:** 5.1, 5.2, 5.3, 5.4, 5.5

### 4. ErrorFormatter (`error_formatter.py`)
Formats clear, actionable error messages.

**Capabilities:**
- Validation error formatting with context
- Before/after examples for corrections
- Success messages with next actions
- Phase gate and template error formatting

**Requirements Covered:** 10.1, 10.2, 10.3, 10.4, 10.5

### 5. LanguageHandler (`language_handler.py`)
Handles language localization for generated documents.

**Capabilities:**
- Read language setting from spec.json
- EARS keyword preservation validation
- Localization readiness checking

**Requirements Covered:** 11.1, 11.2, 11.3, 11.4, 11.5

## Usage

```python
from validation import (
    MarkdownParser,
    MetadataTracker,
    SteeringLoader,
    ErrorFormatter,
    LanguageHandler,
)

# Parse requirements
parser = MarkdownParser()
headings, criteria = parser.parse_requirements(Path("requirements.md"))

# Track metadata
tracker = MetadataTracker()
metadata = tracker.load_metadata(Path("spec.json"))
metadata = tracker.log_phase_transition(metadata, "init", "requirements-generated")
tracker.save_metadata(Path("spec.json"), metadata)

# Load steering context
loader = SteeringLoader()
context = loader.load_steering_context(Path(".kiro/steering"))

# Format errors
formatter = ErrorFormatter()
error_msg = formatter.format_phase_gate_error(
    current_phase="initialized",
    required_approvals=["requirements"],
    suggested_command="/kiro:spec-requirements"
)
```

## Testing

Run tests for all components:

```bash
pytest .kiro/validation/ -v
```

All tests follow Test-Driven Development (TDD) methodology:
- Tests written before implementation
- Comprehensive coverage of requirements
- Integration with pydantic models for type safety

## Architecture

The validation framework follows a layered architecture:

```
Utilities Layer
├── MarkdownParser    - Markdown AST extraction
├── MetadataTracker   - spec.json management
├── SteeringLoader    - Context loading
├── ErrorFormatter    - Error message formatting
└── LanguageHandler   - Localization support
```

All components are:
- Pure functions where possible
- Type-safe with pydantic models
- Tested with pytest
- Documented with docstrings

## Development

These utilities are designed to be used by higher-level validators:
- Template validation (Task 2)
- Phase validation (Task 3)
- EARS validation (Task 4)
- Requirement ID validation (Task 5)
- Gap analysis (Task 6)
- Design validation (Task 7)
- Task dependency analysis (Task 8)
- Implementation validation (Task 9)

## Status

✅ Task 1.1: MarkdownParser - Complete
✅ Task 1.2: MetadataTracker - Complete
✅ Task 1.3: SteeringLoader - Complete
✅ Task 1.4: ErrorFormatter - Complete
✅ Task 1.5: LanguageHandler - Complete

**Test Results:** 23 tests passed, 0 failed
