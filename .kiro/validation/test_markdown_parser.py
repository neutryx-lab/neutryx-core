"""
Test suite for Markdown parser used in SDD validation.

Requirements covered: 3.1, 3.2, 4.1, 4.2
"""
from pathlib import Path
import pytest
from .markdown_parser import (
    MarkdownParser,
    RequirementHeading,
    AcceptanceCriterion,
    TaskItem,
)


class TestMarkdownParser:
    """Test Markdown parser for SDD validation."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return MarkdownParser()

    @pytest.fixture
    def sample_requirements_md(self, tmp_path):
        """Create sample requirements.md file."""
        content = """# Requirements Document

## Requirement 1: Template Standardization

**Objective:** As a developer, I want all SDD templates to follow consistent structure.

#### Acceptance Criteria
1. The SDD framework shall validate that all template files contain required placeholders
2. When a template file is missing required placeholders, the SDD framework shall report specific missing placeholders
3. The SDD framework shall ensure all specification templates use identical placeholder naming conventions

### Requirement 2: Workflow Phase Validation

**Objective:** As a developer, I want strict phase validation.

#### Acceptance Criteria
1. When a user attempts to run `/kiro:spec-design` without approved requirements, the SDD framework shall block the operation
2. When a user attempts to run `/kiro:spec-tasks` without approved design, the SDD framework shall block the operation

## Requirement A: Alphabetic ID Test

Should be normalized to numeric.
"""
        file_path = tmp_path / "requirements.md"
        file_path.write_text(content)
        return file_path

    @pytest.fixture
    def sample_tasks_md(self, tmp_path):
        """Create sample tasks.md file."""
        content = """# Implementation Plan

## Foundation Layer

- [ ] 1. Build core utilities infrastructure
- [ ] 1.1 (P) Create Markdown parser with AST extraction
  - Implement parser using Python `markdown` library
  - Extract requirement headings with numeric ID detection
  - _Requirements: 3.1, 3.2, 4.1, 4.2_

- [ ] 1.2 (P) Build metadata tracker
  - Implement spec.json schema loading
  - _Requirements: 12.1, 12.2_

- [ ] 2. Implement template validation system
- [ ] 2.1 Build template validator for placeholder checking
  - Implement placeholder pattern detection
  - _Requirements: 1.1, 1.2_
"""
        file_path = tmp_path / "tasks.md"
        file_path.write_text(content)
        return file_path

    def test_parser_initialization(self, parser):
        """Test parser can be initialized."""
        assert parser is not None
        assert isinstance(parser, MarkdownParser)

    def test_parse_requirements_extracts_headings(self, parser, sample_requirements_md):
        """Test extraction of requirement headings with numeric IDs."""
        headings, criteria = parser.parse_requirements(sample_requirements_md)

        assert len(headings) >= 2

        # Check first requirement
        req1 = headings[0]
        assert req1.numeric_id == 1
        assert "Template Standardization" in req1.text
        assert req1.level == 2  # ## heading

        # Check second requirement
        req2 = headings[1]
        assert req2.numeric_id == 2
        assert "Workflow Phase Validation" in req2.text

    def test_parse_requirements_detects_alphabetic_ids(self, parser, sample_requirements_md):
        """Test detection of alphabetic requirement IDs."""
        headings, _ = parser.parse_requirements(sample_requirements_md)

        # Find alphabetic ID requirement
        alpha_req = [h for h in headings if h.text and "Alphabetic" in h.text]
        assert len(alpha_req) > 0
        assert alpha_req[0].numeric_id is None  # Should not have numeric ID

    def test_parse_requirements_extracts_acceptance_criteria(
        self, parser, sample_requirements_md
    ):
        """Test extraction of acceptance criteria from requirements."""
        _, criteria = parser.parse_requirements(sample_requirements_md)

        assert len(criteria) >= 5

        # Check first criterion
        first_criterion = criteria[0]
        assert "SDD framework shall validate" in first_criterion.text
        assert first_criterion.requirement_id == 1
        assert first_criterion.line_number > 0

    def test_parse_tasks_extracts_task_hierarchy(self, parser, sample_tasks_md):
        """Test extraction of task items with hierarchy."""
        tasks = parser.parse_tasks(sample_tasks_md)

        assert len(tasks) >= 4

        # Check task 1
        task1 = [t for t in tasks if t.task_id == "1"][0]
        assert "Build core utilities" in task1.description
        assert task1.parallel is False

        # Check task 1.1 (parallel)
        task1_1 = [t for t in tasks if t.task_id == "1.1"][0]
        assert "Create Markdown parser" in task1_1.description
        assert task1_1.parallel is True
        assert "3.1" in task1_1.requirements
        assert "3.2" in task1_1.requirements

    def test_parse_tasks_extracts_details(self, parser, sample_tasks_md):
        """Test extraction of task details as sub-items."""
        tasks = parser.parse_tasks(sample_tasks_md)

        task1_1 = [t for t in tasks if t.task_id == "1.1"][0]
        assert len(task1_1.details) >= 2
        assert any("Implement parser" in detail for detail in task1_1.details)
        assert any("Extract requirement headings" in detail for detail in task1_1.details)

    def test_extract_code_blocks(self, parser):
        """Test extraction of code blocks from Markdown."""
        content = """
# Example

Some text

```python
def hello():
    return "world"
```

More text

```bash
echo "test"
```
"""
        code_blocks = parser.extract_code_blocks(content)
        assert len(code_blocks) == 2
        assert "def hello()" in code_blocks[0]
        assert 'echo "test"' in code_blocks[1]

    def test_extract_code_blocks_filtered_by_language(self, parser):
        """Test extraction of code blocks filtered by language."""
        content = """
```python
def hello():
    pass
```

```bash
echo "test"
```

```python
def world():
    pass
```
"""
        python_blocks = parser.extract_code_blocks(content, language="python")
        assert len(python_blocks) == 2
        assert "def hello()" in python_blocks[0]
        assert "def world()" in python_blocks[1]

    def test_parse_requirements_handles_empty_file(self, parser, tmp_path):
        """Test parsing empty requirements file."""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")

        headings, criteria = parser.parse_requirements(empty_file)
        assert len(headings) == 0
        assert len(criteria) == 0

    def test_parse_tasks_handles_no_tasks(self, parser, tmp_path):
        """Test parsing file with no task checkboxes."""
        no_tasks = tmp_path / "no_tasks.md"
        no_tasks.write_text("# Just a heading\n\nSome text")

        tasks = parser.parse_tasks(no_tasks)
        assert len(tasks) == 0
