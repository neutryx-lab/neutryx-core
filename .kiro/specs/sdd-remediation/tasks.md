# Implementation Plan

## Foundation Layer

- [ ] 1. Build core utilities infrastructure
- [ ] 1.1 (P) Create Markdown parser with AST extraction
  - Implement parser using Python `markdown` library for requirements documents
  - Extract requirement headings with numeric ID detection
  - Extract acceptance criteria from nested lists under requirement sections
  - Extract task items with ID, description, details, and requirement references
  - Handle code blocks, tables, and nested list structures correctly
  - _Requirements: 3.1, 3.2, 4.1, 4.2_

- [ ] 1.2 (P) Build metadata tracker for spec.json management
  - Implement spec.json schema loading with pydantic models
  - Create audit trail event logging for phase transitions and approvals
  - Add timestamp management with ISO 8601 format
  - Implement schema migration from old format to new format with audit_trail field
  - Ensure atomic updates for phase and approval state changes
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 1.3 (P) Implement steering context loader
  - Build recursive file loader for `.kiro/steering/` directory
  - Validate default steering files exist (product.md, tech.md, structure.md)
  - Implement token estimation and budget-based prioritization by modification time
  - Add warning generation for missing default files or empty directory
  - Support loading custom steering files regardless of mode settings
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 1.4 (P) Create error formatter for consistent messaging
  - Build error message formatter with failure reason, file path, and remediation steps
  - Implement before/after example generation for validation failures
  - Add help text formatting with command syntax and options
  - Create success message formatter with next recommended action
  - Support template missing error with expected path and example structure
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 1.5 (P) Implement language handler for localization
  - Read language setting from spec.json with default to "en"
  - Create EARS keyword preservation validator
  - Implement localization readiness checker
  - Add Markdown structure validation for localized content
  - Support language field validation in templates
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

## Validation Core Components

- [ ] 2. Implement template validation system
- [ ] 2.1 Build template validator for placeholder checking
  - Implement placeholder pattern detection using regex for {{FEATURE_NAME}}, {{TIMESTAMP}}, {{PROJECT_DESCRIPTION}}
  - Validate all templates in `.kiro/settings/templates/specs/` directory
  - Check placeholder naming consistency across all template files
  - Report missing placeholders with file path and line number guidance
  - Add template versioning support for backward compatibility
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2.2 Integrate template validation into spec-init command
  - Call template validator before creating new specifications
  - Display validation errors with formatted messages using error formatter
  - Block spec initialization if template validation fails
  - Provide remediation guidance for template issues
  - Log template validation events to metadata tracker
  - _Requirements: 1.1, 1.2, 10.1, 10.2_

- [ ] 3. Build workflow phase validation system
- [ ] 3.1 Create phase validator for workflow gates
  - Implement phase prerequisite checking for design, tasks, and implementation
  - Validate approval state before allowing phase transitions
  - Detect spec.json corruption with schema validation
  - Generate phase status display showing current phase and required approvals
  - Implement atomic phase and approval state updates
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 3.2 Integrate phase validation into generation commands
  - Add phase validation to spec-design command (requires approved requirements)
  - Add phase validation to spec-tasks command (requires approved design)
  - Add phase validation to spec-impl command (requires all approvals)
  - Display detailed approval status on phase gate violations
  - Suggest `-y` flag or manual approval in error messages
  - _Requirements: 2.1, 2.2, 2.3, 10.1, 10.4_

- [ ] 4. Implement EARS format validation
- [ ] 4.1 Build EARS pattern detector with regex matching
  - Implement regex patterns for all five EARS types (Event, State, Unwanted, Optional, Ubiquitous)
  - Create subject extraction logic from EARS statements
  - Detect combined patterns (e.g., "While [X], when [Y], the [system] shall")
  - Validate subject is concrete (not generic "system") using steering context
  - Generate pattern-specific correction suggestions
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 4.2 Integrate EARS validation with requirements parsing
  - Use Markdown parser to extract acceptance criteria from requirements.md
  - Validate each criterion against EARS patterns
  - Report non-compliant criteria with line numbers and suggestions
  - Load examples from `.kiro/settings/rules/ears-format.md` for error messages
  - Support manual override with explicit user confirmation
  - _Requirements: 3.1, 3.2, 3.4, 3.5, 10.3_

- [ ] 5. Create requirement ID validation system
- [ ] 5.1 Build requirement ID validator
  - Implement numeric ID pattern detection (e.g., "Requirement 1", "1.", "2 Feature")
  - Detect alphabetic IDs (e.g., "Requirement A") and suggest numeric replacements
  - Auto-generate sequential numeric IDs for headings without IDs
  - Detect and report ID collisions with duplicate ID detection
  - Create ID normalization logic for alphabetic to numeric conversion
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5.2 Implement cross-reference validation
  - Validate requirement IDs referenced in design.md exist in requirements.md
  - Validate requirement IDs referenced in tasks.md exist in requirements.md
  - Report broken references with file locations and missing IDs
  - Track ID mappings during normalization for reference updates
  - _Requirements: 4.5_

## Orchestration Layer Validators

- [ ] 6. Build gap analysis validator
- [ ] 6.1 (P) Implement gap analyzer for requirement-to-code mapping
  - Parse requirements.md to extract requirement objectives
  - Search codebase for related components using keyword matching
  - Identify existing functionality, missing functionality, and integration points
  - Generate gap analysis report with confidence scores for component matches
  - Recommend greenfield vs brownfield approach based on findings
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 6.2 (P) Create gap analysis command integration
  - Implement `/kiro:validate-gap {feature}` command handler
  - Verify requirements are approved before running gap analysis
  - Save gap analysis report to `.kiro/specs/{feature}/gap-analysis.md` with timestamp
  - Format gap analysis results using error formatter for consistent output
  - Display summary with existing components, missing functionality, and recommended approach
  - _Requirements: 6.5, 10.5_

- [ ] 7. Implement design validation system
- [ ] 7.1 (P) Build design validator for quality review
  - Load design review criteria from `.kiro/settings/rules/design-review.md`
  - Parse design.md to extract components, technology stack, and architecture decisions
  - Check technology stack alignment with steering tech.md
  - Validate structural compliance, testability, performance, and security considerations
  - Generate design validation report with categorized issues (error, warning, info)
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 7.2 (P) Create design validation command integration
  - Implement `/kiro:validate-design {feature}` command handler
  - Support interactive Q&A for design clarification during validation
  - Update spec.json with design review timestamp and approval status
  - Provide specific feedback with file locations and suggested improvements
  - Format validation results using error formatter
  - _Requirements: 7.4, 7.5, 10.1, 10.3_

- [ ] 8. Build task dependency analysis system
- [ ] 8.1 (P) Implement task dependency analyzer
  - Parse tasks.md to extract task hierarchy and dependencies
  - Analyze file modification dependencies from task descriptions
  - Infer shared file dependencies from design component mappings
  - Detect circular dependencies and generate resolution suggestions
  - Identify parallelizable tasks with no shared dependencies
  - _Requirements: 8.1, 8.2, 8.3_

- [ ] 8.2 (P) Generate task execution graph
  - Build task dependency graph with critical path calculation
  - Identify parallel execution tracks for tasks
  - Generate optimal execution order recommendations
  - Include parallel markers `(P)` in tasks.md for parallelizable tasks
  - Format task execution graph for tasks.md output
  - _Requirements: 8.4, 8.5_

- [ ] 9. Create implementation validation system
- [ ] 9.1 (P) Build implementation validator
  - Parse requirements.md, design.md, and tasks.md for validation criteria
  - Search codebase for implementation artifacts matching requirements
  - Validate implementation follows design patterns from design.md
  - Check completed tasks have corresponding code changes
  - Generate implementation validation report with pass/fail per requirement
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 9.2 (P) Create implementation validation command integration
  - Implement `/kiro:validate-impl {feature}` command handler
  - Verify all approvals (requirements, design, tasks) exist before validation
  - Save implementation validation report to `.kiro/specs/{feature}/impl-validation.md`
  - Report missing functionality with requirement references
  - Display overall pass/fail status with detailed requirement breakdown
  - _Requirements: 9.5, 10.1_

## Integration and Testing

- [ ] 10. Integrate validators into workflow commands
- [ ] 10.1 Update spec-init to use template validator
  - Call template validator before spec initialization
  - Display validation errors if templates are incomplete
  - Log template validation to metadata tracker
  - _Requirements: 1.1, 1.2_

- [ ] 10.2 Update spec-requirements to use validators
  - Integrate steering context loader at command start
  - Add EARS validation during requirements generation
  - Add requirement ID validation after requirements creation
  - Update metadata tracker with generation events
  - _Requirements: 3.1, 4.1, 5.1_

- [ ] 10.3 Update spec-design to use validators
  - Integrate phase validator to check requirements approval
  - Load steering context for design generation
  - Add requirement ID cross-reference validation
  - Update metadata tracker with design generation events
  - _Requirements: 2.1, 4.5, 5.1_

- [ ] 10.4 Update spec-tasks to use validators
  - Integrate phase validator to check design approval
  - Add task dependency analysis during generation
  - Add requirement ID cross-reference validation
  - Update metadata tracker with task generation events
  - _Requirements: 2.2, 4.5, 8.1_

- [ ] 10.5 Update spec-impl to use validators
  - Integrate phase validator to check all approvals
  - Validate implementation readiness
  - Update metadata tracker with implementation events
  - _Requirements: 2.3_

- [ ] 11. Implement comprehensive testing
- [ ] 11.1* Unit tests for validation core components
  - Test TemplateValidator with valid/invalid templates
  - Test PhaseValidator with various approval states
  - Test EARSValidator with all five EARS patterns
  - Test RequirementIDValidator with numeric/alphabetic IDs
  - _Requirements: 1.1, 2.1, 3.1, 4.1_

- [ ] 11.2* Unit tests for utilities layer
  - Test MarkdownParser with complex documents
  - Test MetadataTracker schema migration
  - Test SteeringLoader with missing files and budget limits
  - Test ErrorFormatter message generation
  - _Requirements: 5.1, 10.1, 12.1_

- [ ] 11.3* Integration tests for workflow validation
  - Test end-to-end spec generation with validation
  - Test phase gate enforcement (reject invalid transitions)
  - Test EARS validation with real requirements.md
  - Test metadata tracking across multiple operations
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 12.1_

- [ ] 11.4* Integration tests for validator commands
  - Test `/kiro:validate-gap` with existing codebase
  - Test `/kiro:validate-design` with real design documents
  - Test `/kiro:validate-impl` with implemented features
  - Test error message clarity and actionability
  - _Requirements: 6.1, 7.1, 9.1, 10.1_

- [ ] 12. Documentation and examples
- [ ] 12.1 Update CLAUDE.md with new validation capabilities
  - Document new validator commands (/kiro:validate-gap, validate-design, validate-impl)
  - Update workflow section with validation integration points
  - Add troubleshooting section for common validation errors
  - Document `-y` flag behavior and fast-track auditing
  - _Requirements: 10.4, 10.5_

- [ ] 12.2 Create validation examples and templates
  - Add example validation error messages with remediation
  - Create template validation examples
  - Add EARS validation examples with before/after corrections
  - Document phase gate error scenarios
  - _Requirements: 10.2, 10.3_
