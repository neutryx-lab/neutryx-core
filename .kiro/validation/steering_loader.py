"""
Steering Context Loader for SDD validation framework.

Loads and validates steering context files from .kiro/steering/.

Requirements covered: 5.1, 5.2, 5.3, 5.4, 5.5
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class SteeringFile(BaseModel):
    """Represents a steering file."""

    path: Path
    content: str
    modified_at: float  # Timestamp
    token_count: int


class SteeringContext(BaseModel):
    """Represents loaded steering context."""

    files: Dict[str, SteeringFile]
    total_tokens: int
    missing_defaults: List[str]
    warnings: List[str]


class SteeringLoader:
    """Load and validate steering context files."""

    DEFAULT_FILES = ["product.md", "tech.md", "structure.md"]

    def load_steering_context(
        self, steering_dir: Path, token_budget: Optional[int] = None
    ) -> SteeringContext:
        """
        Load all steering files recursively.

        Preconditions:
        - steering_dir exists

        Postconditions:
        - Returns steering context with all files
        - Prioritizes by modification timestamp if budget exceeded
        - Warns about missing default files
        """
        if not steering_dir.exists():
            return SteeringContext(
                files={},
                total_tokens=0,
                missing_defaults=self.DEFAULT_FILES.copy(),
                warnings=["Steering directory does not exist"],
            )

        # Check for missing defaults
        missing_defaults = self.validate_defaults_exist(steering_dir)

        # Load all Markdown files recursively
        all_files: List[SteeringFile] = []
        for md_file in steering_dir.rglob("*.md"):
            content = md_file.read_text()
            token_count = self.estimate_tokens(content)
            modified_at = md_file.stat().st_mtime

            all_files.append(
                SteeringFile(
                    path=md_file,
                    content=content,
                    modified_at=modified_at,
                    token_count=token_count,
                )
            )

        # Sort by modification time (most recent first)
        all_files.sort(key=lambda f: f.modified_at, reverse=True)

        # Apply token budget if specified
        if token_budget:
            all_files = self.prioritize_files(all_files, token_budget)

        # Build files dict
        files_dict = {str(f.path.relative_to(steering_dir)): f for f in all_files}

        # Calculate total tokens
        total_tokens = sum(f.token_count for f in all_files)

        # Warnings
        warnings: List[str] = []
        if missing_defaults:
            warnings.append(
                f"Missing default steering files: {', '.join(missing_defaults)}"
            )
        if not all_files:
            warnings.append("No steering files found - context may be incomplete")

        return SteeringContext(
            files=files_dict,
            total_tokens=total_tokens,
            missing_defaults=missing_defaults,
            warnings=warnings,
        )

    def validate_defaults_exist(self, steering_dir: Path) -> List[str]:
        """
        Check default steering files exist.

        Returns list of missing default files.
        """
        missing = []
        for default_file in self.DEFAULT_FILES:
            if not (steering_dir / default_file).exists():
                missing.append(default_file)
        return missing

    def estimate_tokens(self, content: str) -> int:
        """
        Estimate token count for content.

        Uses simple approximation: ~4 chars per token.
        """
        return len(content) // 4

    def prioritize_files(
        self, files: List[SteeringFile], token_budget: int
    ) -> List[SteeringFile]:
        """
        Prioritize files by modification time to fit budget.

        Returns list of files within token budget.
        """
        selected: List[SteeringFile] = []
        current_tokens = 0

        for file in files:
            if current_tokens + file.token_count <= token_budget:
                selected.append(file)
                current_tokens += file.token_count
            else:
                break

        return selected
