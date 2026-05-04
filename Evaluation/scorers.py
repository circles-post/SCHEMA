"""Per-question-type scoring primitives.

All scorers have the same output contract:

    ``(picked, expected, score: float, detail: dict, error: str)``

``error`` is an empty string on success; otherwise a short machine-readable
reason (``"unparseable_answer"``, ``"essay_requires_judge_config"``,
``"sandbox_unavailable"``, etc.). ``score`` is always in [0.0, 1.0].
"""

from __future__ import annotations

import re
import string
from typing import Any

from pubmed_graph.llm import InternChatClient
from pubmed_graph.utils import normalize_text

from question_generation.sandbox_runner import run_unit_tests


_LETTERS = "ABCDEFGHIJKL"
_BOOLEAN_TRUE_TOKENS = {"supported", "support", "yes", "true", "1", "y", "t"}
_BOOLEAN_FALSE_TOKENS = {
    "not_supported", "not supported", "unsupported", "no", "false", "0", "n", "f",
    "notsupported", "not",
}
_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return normalize_text(str(text)).casefold().strip()


# ---------------------------------------------------------------------------
# Multichoice: claim_choice / one_hop_tail / two_hop_tail
# ---------------------------------------------------------------------------
def _parse_multichoice(model_answer: Any, options: list[dict[str, Any]]) -> int:
    """Resolve ``model_answer`` to a 0-based index into ``options``.

    Accepts int, single letter ``A``–``H``, or exact-text match. Returns
    -1 on failure.
    """
    if not options:
        return -1
    n = len(options)

    if isinstance(model_answer, bool):
        # ``bool`` is a subclass of int; handle explicitly before the int branch
        # to avoid True → idx 1 surprises.
        return -1
    if isinstance(model_answer, int):
        return model_answer if 0 <= model_answer < n else -1

    if isinstance(model_answer, str):
        s = model_answer.strip()
        if not s:
            return -1
        # Single-letter form
        stripped = s.translate(_PUNCT_TRANSLATOR).strip().upper()
        if len(stripped) == 1 and stripped in _LETTERS[:n]:
            return _LETTERS.index(stripped)
        # Leading letter form like "A. foo" / "A) foo" / "A: foo"
        m = re.match(r"^\s*([A-H])\b", s.upper())
        if m and m.group(1) in _LETTERS[:n]:
            return _LETTERS.index(m.group(1))
        # Full-text match (normalized)
        target = _normalize(s)
        for i, opt in enumerate(options):
            if _normalize(opt.get("text", "")) == target:
                return i
        # Substring tie-break: unique option that contains the answer
        containing = [
            i for i, opt in enumerate(options)
            if target and target in _normalize(opt.get("text", ""))
        ]
        if len(containing) == 1:
            return containing[0]
    return -1


def score_multichoice(
    sample: dict[str, Any], model_answer: Any
) -> tuple[Any, Any, float, dict, str]:
    options = sample.get("options") or []
    picked = _parse_multichoice(model_answer, options)
    expected = next(
        (i for i, o in enumerate(options) if o.get("is_correct")), -1
    )
    detail = {
        "picked_index": picked,
        "expected_index": expected,
        "option_count": len(options),
    }
    if picked < 0:
        return model_answer, expected, 0.0, detail, "unparseable_answer"
    if expected < 0:
        return picked, expected, 0.0, detail, "no_correct_option"
    return picked, expected, (1.0 if picked == expected else 0.0), detail, ""


# ---------------------------------------------------------------------------
# Boolean: boolean_support
# ---------------------------------------------------------------------------
def _parse_boolean(model_answer: Any) -> bool | None:
    if isinstance(model_answer, bool):
        return model_answer
    if isinstance(model_answer, (int, float)):
        return bool(model_answer)
    if isinstance(model_answer, str):
        s = model_answer.strip().casefold().translate(_PUNCT_TRANSLATOR)
        s_flat = s.replace(" ", "")
        if s in _BOOLEAN_TRUE_TOKENS or s_flat in _BOOLEAN_TRUE_TOKENS:
            return True
        if s in _BOOLEAN_FALSE_TOKENS or s_flat in _BOOLEAN_FALSE_TOKENS:
            return False
        # Leading-word heuristic: e.g. "Supported. The claim..."
        head = s.split()[0] if s.split() else ""
        if head in _BOOLEAN_TRUE_TOKENS:
            return True
        if head in _BOOLEAN_FALSE_TOKENS:
            return False
    return None


def score_boolean(
    sample: dict[str, Any], model_answer: Any
) -> tuple[Any, Any, float, dict, str]:
    picked = _parse_boolean(model_answer)
    expected_text = _normalize((sample.get("answer") or {}).get("canonical_text", ""))
    expected = expected_text == "supported"
    detail = {"picked": picked, "expected": expected}
    if picked is None:
        return model_answer, expected, 0.0, detail, "unparseable_answer"
    return picked, expected, (1.0 if picked == expected else 0.0), detail, ""


# ---------------------------------------------------------------------------
# Essay: LLM-as-judge
# ---------------------------------------------------------------------------
_ESSAY_JUDGE_SYSTEM = (
    "You are an expert biomedical grader. Score a student's free-form answer "
    "against a reference answer produced from the same scientific evidence. "
    "Return JSON only."
)

_ESSAY_JUDGE_TEMPLATE = """Question:
{question}

Reference answer (ground truth):
{reference}

Student answer:
{student}

Grade the student answer strictly on scientific content overlap with the
reference. Ignore style differences. If the student answer contradicts the
reference, score low.

Return a single JSON object with:
  "score":     float in [0, 1], 1 = equivalent, 0 = contradictory or empty
  "rationale": <= 40 words explaining the score.
"""


def _judge_essay(
    question: str,
    reference: str,
    student: str,
    judge_model_config: dict[str, Any],
) -> tuple[float, str, str]:
    """Call the intern-s1-pro judge. Returns (score, rationale, error)."""
    if not judge_model_config:
        return 0.0, "", "essay_requires_judge_config"
    if not all(judge_model_config.get(k) for k in ("model", "base_url", "api_key")):
        return 0.0, "", "judge_config_incomplete"
    try:
        client = InternChatClient(judge_model_config)
        messages = [
            {"role": "system", "content": _ESSAY_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _ESSAY_JUDGE_TEMPLATE.format(
                    question=question.strip(),
                    reference=reference.strip(),
                    student=student.strip() or "<empty>",
                ),
            },
        ]
        resp = client.chat_json(
            messages,
            model=judge_model_config.get("model"),
            temperature=float(judge_model_config.get("temperature", 0.0) or 0.0),
            max_tokens=int(judge_model_config.get("max_tokens", 200) or 200),
        )
    except Exception as exc:
        return 0.0, "", f"judge_exception:{type(exc).__name__}"
    try:
        raw_score = float(resp.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        raw_score = 0.0
    score = max(0.0, min(1.0, raw_score))
    rationale = str(resp.get("rationale", "") or "")[:500]
    return score, rationale, ""


def score_essay(
    sample: dict[str, Any],
    model_answer: Any,
    judge_model_config: dict[str, Any] | None,
) -> tuple[Any, Any, float, dict, str]:
    if not isinstance(model_answer, str):
        return model_answer, None, 0.0, {}, "essay_requires_string_answer"
    reference = (sample.get("answer") or {}).get("text", "")
    question = sample.get("question", "")
    score, rationale, err = _judge_essay(
        question, reference, model_answer, judge_model_config or {}
    )
    detail = {
        "rationale": rationale,
        "reference_preview": reference[:200],
        "student_preview": model_answer[:200],
    }
    return model_answer, reference, score, detail, err


# ---------------------------------------------------------------------------
# VQA: yes/no or open-ended image QA
# ---------------------------------------------------------------------------
def score_vqa(
    sample: dict[str, Any],
    model_answer: Any,
    judge_model_config: dict[str, Any] | None = None,
) -> tuple[Any, Any, float, dict, str]:
    """Dispatch by ``metadata.vqa_format``.

    * ``yesno`` → exact match against the reference ("yes" / "no") via the
      same boolean parser used by ``boolean_support``.
    * ``open``  → LLM-as-judge compares the free-form answer to reference.
    """
    meta = sample.get("metadata") or {}
    vqa_format = str(meta.get("vqa_format") or "open")
    ref_answer = (sample.get("answer") or {}).get("text", "")

    if vqa_format == "yesno":
        # Ground truth: normalize ref to bool.
        ref_bool = _parse_boolean(ref_answer)
        picked = _parse_boolean(model_answer)
        detail = {"picked": picked, "expected": ref_bool, "vqa_format": "yesno"}
        if picked is None:
            return model_answer, ref_bool, 0.0, detail, "unparseable_answer"
        if ref_bool is None:
            return picked, ref_bool, 0.0, detail, "reference_not_yesno"
        return picked, ref_bool, (1.0 if picked == ref_bool else 0.0), detail, ""

    # open-ended → LLM-as-judge
    if not isinstance(model_answer, str):
        return model_answer, None, 0.0, {"vqa_format": "open"}, "vqa_requires_string_answer"
    question_q = str(meta.get("question_q") or sample.get("question", ""))
    score, rationale, err = _judge_essay(
        question_q, ref_answer, model_answer, judge_model_config or {}
    )
    detail = {
        "vqa_format": "open",
        "rationale": rationale,
        "reference_preview": str(ref_answer)[:200],
        "student_preview": model_answer[:200],
    }
    return model_answer, ref_answer, score, detail, err


# ---------------------------------------------------------------------------
# Experiment code: run unit_tests in sandbox
# ---------------------------------------------------------------------------
def score_experiment_code(
    sample: dict[str, Any],
    model_answer: Any,
    *,
    timeout: float | None = None,
) -> tuple[Any, Any, float, dict, str]:
    if not isinstance(model_answer, str) or not model_answer.strip():
        return model_answer, None, 0.0, {}, "code_requires_string_answer"
    metadata = sample.get("metadata") or {}
    data_code = str(metadata.get("data_code") or "")
    unit_tests = list(metadata.get("unit_tests") or [])
    if not unit_tests:
        return model_answer, None, 0.0, {}, "no_unit_tests"

    result = run_unit_tests(
        data_code=data_code,
        main_code=model_answer,
        unit_tests=unit_tests,
        timeout=timeout,
    )
    total = int(result.get("total", 0) or 0)
    passed = int(result.get("passed", 0) or 0)
    score = (passed / total) if total else 0.0
    detail = {
        "passed": passed,
        "total": total,
        "sandbox_status": result.get("sandbox_status", ""),
        "compile_error": (result.get("compile_error") or "")[:500],
        "stderr_head": (result.get("stderr") or "")[:500],
    }
    if result.get("sandbox_status") == "sandbox_disabled":
        return model_answer, "reference_main_code", score, detail, "sandbox_unavailable"
    return model_answer, "reference_main_code", score, detail, ""
