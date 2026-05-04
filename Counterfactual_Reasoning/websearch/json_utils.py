"""JSON utility functions for handling LLM-generated JSON responses."""

import json
import re


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from an LLM response that may be wrapped in markdown code blocks."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return text


def sanitize_json_string(text: str) -> str:
    r"""Fix invalid escape sequences in LLM-generated JSON (e.g. LaTeX \(, \alpha)."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    def fix_escapes(match):
        char = match.group(1)
        if char in ('"', "\\", "/", "b", "f", "n", "r", "t"):
            return match.group(0)
        if char == "u":
            return match.group(0)
        return "\\\\" + char

    return re.sub(r"\\(.)", fix_escapes, text)
