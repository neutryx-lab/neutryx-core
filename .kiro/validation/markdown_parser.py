"""
Markdown parser for SDD validation framework.

Parses requirements.md, design.md, and tasks.md to extract structured data
for validation purposes.

Requirements covered: 3.1, 3.2, 4.1, 4.2, 6, 7, 8, 9
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel


class RequirementHeading(BaseModel):
    """Represents a requirement heading from requirements.md."""

    level: int  # Heading level (1-6)
    text: str
    numeric_id: Optional[int] = None
    line_number: int


class AcceptanceCriterion(BaseModel):
    """Represents an acceptance criterion from requirements.md."""

    text: str
    line_number: int
    requirement_id: Optional[int] = None


class TaskItem(BaseModel):
    """Represents a task item from tasks.md."""

    task_id: str  # e.g., "1.1", "2.3"
    description: str
    details: List[str] = []
    requirements: List[str] = []  # Requirement IDs
    parallel: bool = False  # Has (P) marker
    line_number: int


class MarkdownParser:
    """Parse Markdown documents for SDD validation."""

    # Pattern to match numeric requirement IDs
    NUMERIC_ID_PATTERN = re.compile(r"^(?:Requirement\s+)?(\d+)")
    ALPHABETIC_ID_PATTERN = re.compile(r"^(?:Requirement\s+)?([A-Z])\b")

    # Pattern to match task checkboxes
    TASK_CHECKBOX_PATTERN = re.compile(r"^-\s+\[\s*[x ]?\s*\]\s+(.+)$")

    # Pattern to extract task ID (matches "1. Task" or "1.1 Task")
    TASK_ID_PATTERN = re.compile(r"^([\d.]+?)\.?\s+(.+)$")

    # Pattern to find requirement references
    REQ_REFERENCE_PATTERN = re.compile(r"_Requirements?:\s*([\d.,\s]+)_")

    def parse_requirements(
        self, requirements_md_path: Path
    ) -> Tuple[List[RequirementHeading], List[AcceptanceCriterion]]:
        """
        Extract requirement headings and acceptance criteria.

        Preconditions:
        - requirements_md_path exists and is valid Markdown

        Postconditions:
        - Returns all requirement headings with IDs
        - Returns all acceptance criteria with line numbers
        """
        if not requirements_md_path.exists():
            return [], []

        content = requirements_md_path.read_text()
        lines = content.split("\n")

        headings: List[RequirementHeading] = []
        criteria: List[AcceptanceCriterion] = []
        current_requirement_id: Optional[int] = None
        in_acceptance_criteria = False

        for line_num, line in enumerate(lines, start=1):
            # Check for headings
            if line.startswith("#"):
                heading_match = re.match(r"^(#+)\s+(.+)$", line)
                if heading_match:
                    level = len(heading_match.group(1))
                    text = heading_match.group(2).strip()

                    # Extract numeric ID
                    numeric_id = self._extract_numeric_id(text)

                    # If this is a requirement heading (level 2 or 3)
                    if level in [2, 3] and ("Requirement" in text or numeric_id):
                        headings.append(
                            RequirementHeading(
                                level=level,
                                text=text,
                                numeric_id=numeric_id,
                                line_number=line_num,
                            )
                        )
                        if numeric_id:
                            current_requirement_id = numeric_id

                    # Check if this is an acceptance criteria section
                    if "Acceptance Criteria" in text:
                        in_acceptance_criteria = True
                    elif level <= 3:  # New major section
                        in_acceptance_criteria = False

            # Check for numbered list items (acceptance criteria)
            elif in_acceptance_criteria and current_requirement_id:
                list_match = re.match(r"^\d+\.\s+(.+)$", line)
                if list_match:
                    criterion_text = list_match.group(1).strip()
                    criteria.append(
                        AcceptanceCriterion(
                            text=criterion_text,
                            line_number=line_num,
                            requirement_id=current_requirement_id,
                        )
                    )

        return headings, criteria

    def parse_tasks(self, tasks_md_path: Path) -> List[TaskItem]:
        """
        Extract task hierarchy and metadata.

        Returns all tasks with IDs, descriptions, details, requirements.
        """
        if not tasks_md_path.exists():
            return []

        content = tasks_md_path.read_text()
        lines = content.split("\n")

        tasks: List[TaskItem] = []
        current_task: Optional[TaskItem] = None

        for line_num, line in enumerate(lines, start=1):
            # Check for task checkbox
            checkbox_match = self.TASK_CHECKBOX_PATTERN.match(line)
            if checkbox_match:
                task_content = checkbox_match.group(1).strip()

                # Check for parallel marker (P)
                parallel = False
                if "(P)" in task_content:
                    parallel = True
                    task_content = task_content.replace("(P)", "").strip()

                # Extract task ID
                task_id_match = self.TASK_ID_PATTERN.match(task_content)
                if task_id_match:
                    task_id = task_id_match.group(1)
                    description = task_id_match.group(2).strip()

                    current_task = TaskItem(
                        task_id=task_id,
                        description=description,
                        parallel=parallel,
                        line_number=line_num,
                    )
                    tasks.append(current_task)
                else:
                    # Task without ID (e.g., subtask or continuation)
                    current_task = TaskItem(
                        task_id="",
                        description=task_content,
                        parallel=parallel,
                        line_number=line_num,
                    )
                    tasks.append(current_task)

            # Check for task details (sub-items starting with "  -")
            elif current_task and line.strip().startswith("- "):
                detail = line.strip()[2:].strip()

                # Check for requirement references
                req_match = self.REQ_REFERENCE_PATTERN.search(detail)
                if req_match:
                    req_ids = req_match.group(1)
                    # Parse requirement IDs
                    current_task.requirements = [
                        r.strip() for r in req_ids.replace(" ", "").split(",")
                    ]
                else:
                    current_task.details.append(detail)

        return tasks

    def extract_code_blocks(
        self, md_content: str, language: Optional[str] = None
    ) -> List[str]:
        """
        Extract code blocks from Markdown content.

        Optionally filter by language identifier.
        """
        code_blocks: List[str] = []
        in_code_block = False
        current_block: List[str] = []
        current_language: Optional[str] = None

        for line in md_content.split("\n"):
            # Check for code block start/end
            if line.strip().startswith("```"):
                if in_code_block:
                    # End of code block
                    if language is None or current_language == language:
                        code_blocks.append("\n".join(current_block))
                    current_block = []
                    current_language = None
                    in_code_block = False
                else:
                    # Start of code block
                    lang_match = re.match(r"^```(\w+)?", line.strip())
                    if lang_match:
                        current_language = lang_match.group(1)
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    def extract_tables(self, md_content: str) -> List[List[List[str]]]:
        """
        Extract tables from Markdown content.

        Returns list of tables, each table is list of rows.
        """
        tables: List[List[List[str]]] = []
        current_table: List[List[str]] = []
        in_table = False

        for line in md_content.split("\n"):
            # Check if line is a table row (contains |)
            if "|" in line:
                if not in_table:
                    in_table = True
                    current_table = []

                # Skip separator rows (like |---|---|)
                if re.match(r"^\s*\|[\s\-:]+\|\s*$", line):
                    continue

                # Parse table row
                cells = [cell.strip() for cell in line.split("|")]
                # Remove empty first/last cells from splitting
                cells = [c for c in cells if c]
                if cells:
                    current_table.append(cells)
            else:
                if in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                in_table = False

        # Add last table if exists
        if current_table:
            tables.append(current_table)

        return tables

    def _extract_numeric_id(self, text: str) -> Optional[int]:
        """Extract numeric ID from requirement heading text."""
        # Try numeric pattern first
        numeric_match = self.NUMERIC_ID_PATTERN.match(text)
        if numeric_match:
            return int(numeric_match.group(1))

        # Check for alphabetic ID (should return None)
        alpha_match = self.ALPHABETIC_ID_PATTERN.match(text)
        if alpha_match:
            return None  # Alphabetic IDs not supported

        return None
