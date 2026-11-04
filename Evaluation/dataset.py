"""Read internal samples and the JSON-string fields in the public HF release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_NESTED_FIELDS = {
    "answer": dict, "options": list, "subgraph": dict, "provenance": dict,
    "grounding": dict, "quality": dict, "metadata": dict,
}


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Decode known nested fields without mutating the caller's record."""
    if not isinstance(sample, dict):
        raise ValueError("Each sample must be a JSON object")
    decoded = dict(sample)
    for field, expected_type in _NESTED_FIELDS.items():
        value = decoded.get(field)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Sample {sample.get('sample_id', '<unknown>')}: invalid JSON in {field}"
                ) from exc
        if value is not None and not isinstance(value, expected_type):
            raise ValueError(
                f"Sample {sample.get('sample_id', '<unknown>')}: "
                f"{field} must decode to {expected_type.__name__}"
            )
        if field in decoded:
            decoded[field] = value
    return decoded


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    samples = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                samples.append(normalize_sample(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return samples
