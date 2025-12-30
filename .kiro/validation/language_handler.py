"""
Language Handler for SDD validation framework.

Handles language localization for generated documents.

Requirements covered: 11.1, 11.2, 11.3, 11.4, 11.5
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


class LanguageHandler:
    """Handle language localization for specs."""

    EARS_KEYWORDS = ["When", "If", "While", "Where", "shall"]

    def get_language(self, spec_json_path: Path) -> str:
        """
        Get language setting from spec.json.

        Returns language code, defaults to "en" if undefined.
        """
        if not spec_json_path.exists():
            return "en"

        with open(spec_json_path, "r") as f:
            data = json.load(f)

        return data.get("language", "en")

    def should_localize(self, language: str) -> bool:
        """
        Check if localization is needed.

        Returns False for "en", True for other languages.
        """
        return language != "en"

    def preserve_ears_keywords(self, content: str) -> bool:
        """
        Validate content preserves EARS keywords.

        Returns True if keywords are in English.
        """
        # Check if any EARS keywords are present and in English
        for keyword in self.EARS_KEYWORDS:
            # Look for the keyword in the content
            if keyword.lower() in content.lower():
                # Check if it appears in correct form
                pattern = rf"\b{keyword}\b"
                if not re.search(pattern, content):
                    return False  # Keyword found but not in correct form

        return True  # All keywords present are in correct form
