"""Evaluation result dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """One scoring verdict.

    ``score`` is always in [0.0, 1.0]:
      * multichoice / boolean : 0.0 or 1.0 (exact)
      * essay                 : continuous, from LLM judge
      * experiment_code       : passed_tests / total_tests

    ``is_correct`` is derived (``score >= 0.5`` by default); for reporting
    binary accuracy on essay/code, consumers can re-threshold themselves.

    ``picked`` is the caller's answer after parsing (e.g. an int index for
    multichoice, a bool for boolean, free text for essay, code string for
    experiment_code). ``expected`` is the ground-truth equivalent.
    """

    sample_id: str
    question_type: str
    score: float = 0.0
    is_correct: bool = False
    picked: Any = None
    expected: Any = None
    tier: str = "not_tagged"
    weight: float = 1.0
    corroboration_status: str = "not_requested"
    evidence_strength: str = "unknown"
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
