"""Render one dataset sample into a single user turn for the agent.

Each question_type needs a slightly different layout: multichoice needs
lettered options; boolean_support needs the claim; essay needs just the
question; experiment_code needs the data_code + unit_tests context.
"""

from __future__ import annotations

import string
from typing import Any


_LETTERS = string.ascii_uppercase


def _format_multichoice(sample: dict[str, Any]) -> str:
    options = sample.get("options") or []
    lines = [f"Question type: {sample.get('question_type')}", "", sample.get("question", "").strip(), "", "Options:"]
    for idx, opt in enumerate(options):
        letter = _LETTERS[idx] if idx < len(_LETTERS) else str(idx)
        lines.append(f"  {letter}. {opt.get('text', '').strip()}")
    lines.append("")
    lines.append("Respond with the single letter of the correct option inside <answer>...</answer>, then TERMINATE on the next line.")
    return "\n".join(lines)


def _format_boolean(sample: dict[str, Any]) -> str:
    lines = [
        f"Question type: {sample.get('question_type')}",
        "",
        sample.get("question", "").strip(),
        "",
        "Respond with <answer>Supported</answer> or <answer>Not Supported</answer>, then TERMINATE on the next line.",
    ]
    return "\n".join(lines)


def _format_essay(sample: dict[str, Any]) -> str:
    lines = [
        f"Question type: {sample.get('question_type')}",
        "",
        sample.get("question", "").strip(),
        "",
        "Write a concise, factual answer inside <answer>...</answer>, then TERMINATE on the next line.",
    ]
    return "\n".join(lines)


def _format_experiment_code(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    data_code = (metadata.get("data_code") or "").strip()
    unit_tests = metadata.get("unit_tests") or []
    lines = [
        f"Question type: {sample.get('question_type')}",
        "",
        sample.get("question", "").strip(),
    ]
    if data_code:
        lines += ["", "Context data_code (will be prepended when your answer is graded):", "```python", data_code, "```"]
    if unit_tests:
        lines += ["", "Unit tests your solution must pass:"]
        for i, ut in enumerate(unit_tests, 1):
            if isinstance(ut, dict):
                lines.append(f"  [{i}] {ut.get('name') or ut.get('test') or ut}")
            else:
                lines.append(f"  [{i}] {ut}")
    lines += [
        "",
        "Respond with the FINAL main_code only, inside <answer>...</answer>, no markdown fences, no commentary. Then TERMINATE on the next line.",
    ]
    return "\n".join(lines)


_BUILDERS = {
    "claim_choice": _format_multichoice,
    "one_hop_tail": _format_multichoice,
    "two_hop_tail": _format_multichoice,
    "boolean_support": _format_boolean,
    "essay": _format_essay,
    "experiment_code": _format_experiment_code,
}


def build_prompt(sample: dict[str, Any]) -> str:
    qtype = sample.get("question_type", "")
    builder = _BUILDERS.get(qtype)
    if builder is None:
        return sample.get("question", "").strip()
    return builder(sample)
