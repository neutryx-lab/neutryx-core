# Research & Design Decisions

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `sdd-remediation`
- **Discovery Scope**: Extension (existing system)
- **Key Findings**:
  - Current SDD framework exists with templates, rules, and workflow commands
  - Template structure uses consistent placeholder patterns ({{FEATURE_NAME}}, {{TIMESTAMP}}, {{PROJECT_DESCRIPTION}})
  - No validation layer exists for template completeness, workflow phases, or EARS format compliance
  - spec.json schema is minimal with basic approval tracking but lacks audit trail
  - Framework is purely prompt-based with no programmatic validation

## Research Log

### Existing SDD Framework Structure
- **Context**: Analyzed current `.kiro/` directory structure to understand extension points
- **Sources Consulted**:
  - `.kiro/settings/templates/specs/` - Template files
  - `.kiro/settings/rules/` - Generation rules
  - `CLAUDE.md` - SDD workflow documentation
- **Findings**:
  - Templates exist for: init.json, requirements.md, design.md, tasks.md, research.md, requirements-init.md
  - Rules exist for: EARS format, design principles, discovery, gap analysis, task generation, parallel analysis
  - Workflow follows 3-phase approval: Requirements → Design → Tasks → Implementation
  - All validation is currently manual through AI prompt instructions
  - No automated validation utilities or scripts exist
- **Implications**:
  - Remediation must add validation layer without disrupting existing template structure
  - Validation can be implemented as utility modules that work with existing file formats
  - Template versioning support is net-new capability

### Template Placeholder Patterns
- **Context**: Verified placeholder consistency across templates
- **Sources Consulted**:
  - `.kiro/settings/templates/specs/init.json`
  - `.kiro/settings/templates/specs/requirements-init.md`
- **Findings**:
  - Standard placeholders: {{FEATURE_NAME}}, {{TIMESTAMP}}, {{PROJECT_DESCRIPTION}}
  - Curly brace syntax is consistent
  - No template versioning metadata exists
- **Implications**:
  - Validation regex can check for these specific patterns
  - Template versioning will need new metadata field in templates

### spec.json Schema Analysis
- **Context**: Examined current spec.json structure for metadata tracking capabilities
- **Sources Consulted**:
  - `.kiro/settings/templates/specs/init.json`
  - `.kiro/specs/sdd-remediation/spec.json`
- **Findings**:
  - Current fields: feature_name, created_at, updated_at, language, phase, approvals, ready_for_implementation
  - approvals structure tracks: requirements, design, tasks (each with generated/approved flags)
  - No audit trail of phase transitions or approval timestamps
  - No record of fast-track mode usage
  - No schema version field for backward compatibility
- **Implications**:
  - Need to extend spec.json schema with audit_trail array
  - Must maintain backward compatibility with existing spec.json files
  - Schema migration logic needed for existing specifications

### EARS Format Validation Requirements
- **Context**: Understanding how to automate EARS compliance checking
- **Sources Consulted**:
  - `.kiro/settings/rules/ears-format.md`
- **Findings**:
  - Five EARS patterns: Event (When), State (While), Unwanted (If), Optional (Where), Ubiquitous (The [system] shall)
  - Pattern detection requires regex matching on sentence structure
  - Subject extraction needed to verify concrete system names
  - Combined patterns exist (e.g., "While [precondition], when [event], the [system] shall")
- **Implications**:
  - Validation logic needs pattern detection for all five types
  - Must parse Markdown to extract acceptance criteria
  - Should provide pattern-specific correction suggestions

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Pure Prompt-Based (Current) | All validation in AI prompts | Simple, no code needed | Inconsistent, not enforced | Current state |
| Validation Utilities | Python modules for validation | Programmatic enforcement, testable | Requires Python execution | Recommended |
| Pre-commit Hooks | Git hooks for validation | Automatic enforcement | Requires Git setup | Future enhancement |
| Hybrid Approach | Utilities + enhanced prompts | Best of both worlds | More complex | Selected approach |

## Design Decisions

### Decision: Hybrid Validation Architecture
- **Context**: Need to enforce validation rules while maintaining prompt-based workflow flexibility
- **Alternatives Considered**:
  1. Pure utility-based validation — Requires changing all workflow commands to Python scripts
  2. Enhanced prompts only — No enforcement mechanism, same issues as current
  3. Hybrid (utilities + prompts) — Utilities for validation logic, prompts orchestrate workflow
- **Selected Approach**: Hybrid architecture with validation utilities exposed through skill commands
- **Rationale**:
  - Maintains existing prompt-based workflow user experience
  - Adds programmatic validation where needed (template checking, EARS parsing, schema validation)
  - Utilities can be unit-tested for reliability
  - Allows gradual migration without breaking existing specs
- **Trade-offs**:
  - Benefits: Testable, enforceable, maintainable
  - Compromises: Adds Python dependencies, requires utility module development
- **Follow-up**: Validate utilities work with existing template structure in implementation phase

### Decision: Extended spec.json Schema with Audit Trail
- **Context**: Need comprehensive metadata tracking while maintaining backward compatibility
- **Alternatives Considered**:
  1. Separate audit.json file — Clean separation but fragmented metadata
  2. Extend spec.json with audit_trail array — Centralized, easier querying
  3. Use Git commits for audit trail — No structured data, harder to query
- **Selected Approach**: Extend spec.json with `audit_trail` array and `schema_version` field
- **Rationale**:
  - Centralized metadata in single file simplifies tooling
  - audit_trail array allows chronological event tracking
  - schema_version enables backward-compatible migrations
  - Minimal impact on existing spec.json parsing
- **Trade-offs**:
  - Benefits: Structured audit data, version-aware parsing
  - Compromises: Slightly larger spec.json files
- **Follow-up**: Implement migration logic for existing specs without audit_trail field

### Decision: Markdown Parser for Requirements Validation
- **Context**: Need to extract and validate acceptance criteria from requirements.md
- **Alternatives Considered**:
  1. Regex-based extraction — Fast but fragile with Markdown variations
  2. Full Markdown parser (markdown-it, mistune) — Robust but heavier dependency
  3. Lightweight AST parser (markdown) — Balance of robustness and simplicity
- **Selected Approach**: Python `markdown` library with custom tree walker for acceptance criteria
- **Rationale**:
  - Handles nested lists and code blocks correctly
  - Standard library, widely used
  - Can extract requirement IDs and acceptance criteria reliably
  - Extensible for future parsing needs
- **Trade-offs**:
  - Benefits: Robust, handles edge cases, standard library
  - Compromises: Slightly slower than pure regex
- **Follow-up**: Write parser tests for various Markdown formatting styles

### Decision: Validation Skill Commands Structure
- **Context**: How to expose validation utilities in the SDD workflow
- **Alternatives Considered**:
  1. Standalone Python scripts — User runs manually, not integrated
  2. Validation flags on existing commands (e.g., --validate) — Couples validation to generation
  3. Separate validation skill commands — Clear separation of concerns
- **Selected Approach**: Dedicated validation skills (`/kiro:validate-gap`, `/kiro:validate-design`, `/kiro:validate-impl`)
- **Rationale**:
  - Clear separation: generation vs validation
  - Optional: users can skip validation if desired
  - Composable: can chain validations in sequence
  - Extensible: can add new validators without changing generators
- **Trade-offs**:
  - Benefits: Flexible, composable, testable
  - Compromises: More commands to learn
- **Follow-up**: Ensure validation commands integrate smoothly with approval workflow

## Risks & Mitigations

- **Risk 1**: Backward compatibility breaks for existing specs
  - **Mitigation**: Schema versioning with migration logic, graceful fallback for missing fields

- **Risk 2**: Validation utilities add Python dependency overhead
  - **Mitigation**: Keep utilities lightweight, use standard library where possible, document dependency installation

- **Risk 3**: EARS validation produces false positives/negatives
  - **Mitigation**: Manual override flag, provide clear correction examples, iterative refinement based on usage

- **Risk 4**: Metadata tracking significantly increases spec.json file size
  - **Mitigation**: Audit trail uses minimal structure, consider optional pruning of old audit events

## References
- [EARS Format Guidelines](../.kiro/settings/rules/ears-format.md) — EARS syntax patterns for validation
- [Current Template Structure](../.kiro/settings/templates/specs/) — Existing placeholder patterns
- [SDD Workflow Documentation](../../../CLAUDE.md) — Current workflow and commands
