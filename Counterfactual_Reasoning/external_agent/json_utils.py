from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_response(response_text: str) -> str:
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def _extract_balanced_json_snippet(text: str) -> str:
    starts = [idx for idx, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1].strip()
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
    return text


def sanitize_json_string(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    def fix_escape(match: re.Match[str]) -> str:
        char = match.group(1)
        if char in ('"', "\\", "/", "b", "f", "n", "r", "t", "u"):
            return match.group(0)
        return "\\\\" + char

    return re.sub(r"\\(.)", fix_escape, text)


def parse_json_response(response_text: str) -> Any:
    extracted = extract_json_from_response(response_text)
    cleaned = sanitize_json_string(extracted)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        snippet = _extract_balanced_json_snippet(cleaned)
        repaired = sanitize_json_string(snippet)
        return json.loads(repaired)
