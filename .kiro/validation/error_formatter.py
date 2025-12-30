"""
Error Formatter for SDD validation framework.

Formats clear, actionable error messages with remediation guidance.

Requirements covered: 10.1, 10.2, 10.3, 10.4, 10.5
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ErrorContext(BaseModel):
    """Context for error message formatting."""

    failure_reason: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    remediation_steps: List[str]
    before_example: Optional[str] = None
    after_example: Optional[str] = None
    help_text: Optional[str] = None
    next_command: Optional[str] = None


class ErrorFormatter:
    """Format clear, actionable error messages."""

    def format_validation_error(self, error_type: str, context: ErrorContext) -> str:
        """
        Format validation error with context.

        Returns formatted Markdown error message.
        """
        lines = [f"## {error_type}"]
        lines.append("")
        lines.append(f"**Reason**: {context.failure_reason}")

        if context.file_path:
            location = context.file_path
            if context.line_number:
                location += f":{context.line_number}"
            lines.append(f"**Location**: {location}")

        lines.append("")
        lines.append("**Remediation Steps**:")
        for i, step in enumerate(context.remediation_steps, 1):
            lines.append(f"{i}. {step}")

        if context.before_example and context.after_example:
            lines.append("")
            lines.append("**Example**:")
            lines.append("")
            lines.append("Before:")
            lines.append("```")
            lines.append(context.before_example)
            lines.append("```")
            lines.append("")
            lines.append("After:")
            lines.append("```")
            lines.append(context.after_example)
            lines.append("```")

        if context.help_text:
            lines.append("")
            lines.append("**Help**:")
            lines.append(context.help_text)

        if context.next_command:
            lines.append("")
            lines.append(f"**Next Command**: `{context.next_command}`")

        return "\n".join(lines)

    def format_success_message(
        self, operation: str, result_summary: str, next_action: str
    ) -> str:
        """
        Format success message with next recommended action.

        Returns formatted Markdown success message.
        """
        lines = [f"## Success: {operation}"]
        lines.append("")
        lines.append(result_summary)
        lines.append("")
        lines.append(f"**Next Action**: {next_action}")
        return "\n".join(lines)

    def format_template_missing_error(self, template_path: str) -> str:
        """
        Format error for missing template with example structure.

        Includes expected path and template example.
        """
        context = ErrorContext(
            failure_reason=f"Template file not found: {template_path}",
            file_path=template_path,
            remediation_steps=[
                f"Create the template file at {template_path}",
                "Include required placeholders: {{FEATURE_NAME}}, {{TIMESTAMP}}, {{PROJECT_DESCRIPTION}}",
                "Ensure the file is readable",
            ],
            before_example=None,
            after_example="""# {{FEATURE_NAME}}

Created: {{TIMESTAMP}}

## Description
{{PROJECT_DESCRIPTION}}""",
            help_text="Templates should follow the standard SDD format with placeholders.",
        )

        return self.format_validation_error("Template Missing", context)

    def format_phase_gate_error(
        self,
        current_phase: str,
        required_approvals: List[str],
        suggested_command: str,
    ) -> str:
        """
        Format phase gate error with current status.

        Includes current phase, missing approvals, and guidance.
        """
        missing_str = ", ".join(required_approvals)
        context = ErrorContext(
            failure_reason=f"Phase gate violation: Cannot proceed from '{current_phase}'",
            remediation_steps=[
                f"Approve missing phases: {missing_str}",
                f"OR use fast-track mode: {suggested_command} -y",
                "Verify spec.json approval state",
            ],
            help_text=f"Current phase: {current_phase}\nRequired approvals: {missing_str}",
            next_command=suggested_command,
        )

        return self.format_validation_error("Phase Gate Error", context)
