"""LLM-driven per-triple experiment_code generation (Plan C).

Instead of substituting head/tail into a hardcoded blueprint template, this
module asks ``intern-s1-pro`` to synthesize a complete, self-contained
experiment for a specific ``(head, relation, tail)`` triple, then lets
``sandbox_runner.evaluate_experiment_sample`` gate the result. Only specs
whose reference solution passes its own unit tests *and* whose masked
version fails at least one test are accepted — so a bad generation gets
rejected automatically, not silently embedded in the benchmark.

Shape of the returned dict (None on failure):

    {
        "task_family": "...",
        "research_direction": "...",
        "discipline": "life_sciences",
        "function_type": "quantitative_pipeline",
        "task_objective": "...",
        "research_focus": "...",
        "data_code": "...",              # defines load_* in data_en module
        "main_code": "...",              # full reference solution
        "incomplete_main_code": "...",    # one blank replaced with pass
        "incomplete_functions": ["..."],  # names of the blanked functions
        "unit_tests": [
            {"name": "...", "input": {...}, "expected_output": {...}},
            ...
        ],
        "generation_source": "llm",
        "generation_attempts": N,
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pubmed_graph.llm import InternChatClient

from . import sandbox_runner


_LOGGER = logging.getLogger("question_generation.experiment_llm")


_ALLOWED_IMPORTS = [
    "numpy", "pandas", "math", "statistics", "random", "json",
    "collections", "itertools", "functools", "re", "string",
    "sklearn", "scipy",
]

_REQUIRED_KEYS = (
    "task_family",
    "research_direction",
    "data_code",
    "main_code",
    "incomplete_main_code",
    "incomplete_functions",
    "unit_tests",
)


def _build_system_message() -> str:
    return (
        "You are a senior computational biologist who designs small, "
        "self-contained programming exercises for a scientific benchmark. "
        "Each exercise turns ONE biological claim into a Python task that "
        "a student must complete. Your output will be executed inside a "
        "sandbox with ONLY the standard library plus "
        + ", ".join(_ALLOWED_IMPORTS)
        + ". Your ``main_code`` must pass every ``unit_test`` you declare, "
        "and your ``incomplete_main_code`` must make at least one ``unit_test`` "
        "fail (otherwise the blank is pointless). Respond with STRICT JSON only, "
        "no prose."
    )


_DIFFICULTY_HINTS = {
    "easy": (
        "Blank out ONE helper function body only. The orchestration and the "
        "other helpers must remain intact."
    ),
    "medium": (
        "Blank out BOTH a helper function body AND one key numerical step "
        "inside the orchestration function."
    ),
    "hard": (
        "Blank out the orchestration function entirely (and optionally one "
        "helper), so the student must design the whole pipeline."
    ),
}


def _build_user_message(
    *,
    head: str,
    head_type: str,
    relation: str,
    tail: str,
    tail_type: str,
    evidence: str,
    difficulty: str,
) -> str:
    difficulty_hint = _DIFFICULTY_HINTS.get(difficulty, _DIFFICULTY_HINTS["easy"])
    # The sandbox harness wires ``data_en`` as an in-memory module, then
    # imports ``main_en``. main_code MUST ``from data_en import load_*``.
    # Unit tests call the function in ``incomplete_functions[0]`` or the
    # ``summarize_*`` orchestrator if no ``function`` hint is given.
    schema = {
        "task_family": "short snake_case tag, e.g. coip_ppi_detection",
        "research_direction": "one line describing the biological question",
        "discipline": "life_sciences",
        "function_type": "quantitative_pipeline",
        "task_objective": "one line: what the student must compute",
        "research_focus": "one line: the scientific claim under test",
        "data_code": (
            "Python module that defines load_<name>() returning a deterministic "
            "synthetic pandas.DataFrame or numpy array. Seed every RNG. NO I/O, "
            "NO network, NO file reads. Must be under 60 lines."
        ),
        "main_code": (
            "Python module that imports from data_en (e.g. `from data_en import "
            "load_<name>`), defines 1-2 helper functions AND a top-level "
            "``summarize_<family>`` orchestrator that returns a dict. Must "
            "compute quantities genuinely tied to the scientific claim. Under "
            "90 lines."
        ),
        "incomplete_main_code": (
            "Copy of main_code with EXACTLY the helper body(s) named in "
            "``incomplete_functions`` replaced by:\n"
            "    pass  # [Please complete the code]\n"
            "Do NOT blank out the imports or summarize_* orchestrator."
        ),
        "incomplete_functions": [
            "names of the functions whose body was replaced with pass"
        ],
        "unit_tests": [
            {
                "name": "descriptive_name",
                "function": "OPTIONAL: target helper name; defaults to summarize_*",
                "input": {"kwargs": "for the target function"},
                "expected_output": {
                    "key": "scalar/dict value; floats compared at 1e-2 tolerance"
                },
            }
        ],
    }
    return (
        f"Scientific claim (from a grounded triple):\n"
        f"  head:      {head}  (type: {head_type or 'unknown'})\n"
        f"  relation:  {relation}\n"
        f"  tail:      {tail}  (type: {tail_type or 'unknown'})\n"
        f"Evidence from the literature (one excerpt):\n"
        f"  {evidence or '(none)'}\n\n"
        f"Difficulty: {difficulty}. {difficulty_hint}\n\n"
        f"Design a SHORT, self-contained Python exercise that makes a student "
        f"operationalize this claim quantitatively. Requirements:\n"
        f"  1. data_code must define a load_* function producing deterministic "
        f"synthetic data that is *compatible with* the claim (e.g. if the claim "
        f"says A upregulates B, the synthetic data should reflect a positive "
        f"correlation). Use simple hard-coded values where possible — not "
        f"``np.random.randn``; prefer literal lists / arrays so that the "
        f"expected_output values are trivial to derive by hand.\n"
        f"  2. main_code must compute something non-trivial (a ratio, score, "
        f"p-value, correlation, classification, etc.) that the student could "
        f"actually derive from the data, and return it via summarize_<family>.\n"
        f"  3. unit_tests must be deterministic. Each ``expected_output`` must "
        f"be a value that the REAL main_code you wrote actually returns. "
        f"STRONGLY PREFER structural/cheap checks that are hard to get wrong:\n"
        f"       - check that the return dict has a specific key: "
        f"``{{\"has_score\": true}}``\n"
        f"       - check a boolean that follows from the sign of the claim: "
        f"``{{\"supported\": true}}`` or ``{{\"direction\": \"positive\"}}``\n"
        f"       - check an integer count: ``{{\"n_samples\": 5}}``\n"
        f"       - check a rounded float only when you have hand-traced the "
        f"math on the literal data — floats are compared with 1e-2 tolerance.\n"
        f"     Before you write ``expected_output`` for a float test, STEP "
        f"THROUGH the data_code + main_code on paper with the literal values, "
        f"then fill in the observed number. Do NOT guess.\n"
        f"  4. The incomplete version must genuinely break the unit tests "
        f"(replacing the helper body with ``pass`` makes it return None, "
        f"which will fail all non-trivial assertions).\n"
        f"  5. Use ONLY these imports: {', '.join(_ALLOWED_IMPORTS)}.\n"
        f"  6. Prefer 1-2 simple unit tests over 5 complex ones.\n\n"
        f"Return EXACTLY this JSON shape (no markdown, no prose):\n"
        f"{json.dumps(schema, indent=2)}"
    )


def _resolve_llm_config() -> dict[str, Any] | None:
    api_key = (
        os.environ.get("INTERN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not api_key:
        return None
    return {
        "api_key": api_key,
        "base_url": (
            os.environ.get("INTERN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://chat.intern-ai.org.cn/api/v1/"
        ),
        "model": os.environ.get("OPENAI_MODEL") or "intern-s1-pro",
        "temperature": 0.2,
        "max_tokens": 3200,
        "max_retries": 1,
        "request_timeout": 90.0,
    }


def _validate_shape(spec: Any) -> tuple[bool, str]:
    if not isinstance(spec, dict):
        return False, f"not a dict (got {type(spec).__name__})"
    for key in _REQUIRED_KEYS:
        if key not in spec:
            return False, f"missing key: {key}"
    if not isinstance(spec["data_code"], str) or "def load_" not in spec["data_code"]:
        return False, "data_code missing load_* definition"
    if not isinstance(spec["main_code"], str) or "def summarize_" not in spec["main_code"]:
        return False, "main_code missing summarize_* orchestrator"
    if not isinstance(spec["incomplete_main_code"], str):
        return False, "incomplete_main_code not a string"
    if spec["incomplete_main_code"] == spec["main_code"]:
        return False, "incomplete_main_code identical to main_code"
    if not isinstance(spec["incomplete_functions"], list) or not spec["incomplete_functions"]:
        return False, "incomplete_functions must be a non-empty list"
    if not isinstance(spec["unit_tests"], list) or len(spec["unit_tests"]) < 1:
        return False, "unit_tests must be a non-empty list"
    for i, ut in enumerate(spec["unit_tests"]):
        if not isinstance(ut, dict):
            return False, f"unit_tests[{i}] not a dict"
        if "name" not in ut or "expected_output" not in ut:
            return False, f"unit_tests[{i}] missing name/expected_output"
    return True, ""


def _call_llm(client: InternChatClient, messages: list[dict[str, str]], max_tokens: int) -> Any:
    return client.chat_json(messages, max_tokens=max_tokens)


def generate_experiment_via_llm(
    *,
    head: str,
    head_type: str,
    relation: str,
    tail: str,
    tail_type: str,
    evidence: str,
    difficulty: str,
    max_retries: int = 1,
) -> dict[str, Any] | None:
    """Ask the LLM for a per-triple experiment spec + validate via sandbox.

    Returns the accepted spec (dict) or None. On None the caller should
    either fall back to a hardcoded blueprint (hybrid mode) or reject the
    sample (pure LLM mode).
    """
    cfg = _resolve_llm_config()
    if cfg is None:
        _LOGGER.warning("LLM experiment generation unavailable (no API key)")
        return None
    try:
        client = InternChatClient(cfg)
    except Exception as exc:
        _LOGGER.warning("failed to construct LLM client: %s", exc)
        return None

    system = _build_system_message()
    user = _build_user_message(
        head=head, head_type=head_type, relation=relation,
        tail=tail, tail_type=tail_type, evidence=evidence,
        difficulty=difficulty,
    )

    last_error = ""
    prior_spec_json = ""
    for attempt in range(max_retries + 1):
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if last_error and prior_spec_json:
            # Include the model's previous attempt + the detailed sandbox
            # feedback so it can surgically patch expected_output values.
            messages.append({"role": "assistant", "content": prior_spec_json})
            messages.append({
                "role": "user",
                "content": (
                    f"{last_error}\n\n"
                    f"Return the CORRECTED strict JSON with the same schema. "
                    f"Keep data_code + main_code unchanged unless that is the "
                    f"problem; usually you just need to update expected_output "
                    f"to match the observed actual values. No markdown, no prose."
                ),
            })
        try:
            _LOGGER.info("LLM generate attempt %d/%d for %s %s %s",
                         attempt + 1, max_retries + 1, head, relation, tail)
            response = _call_llm(client, messages, max_tokens=int(cfg["max_tokens"]))
        except Exception as exc:
            last_error = f"LLM call failed: {exc}"
            _LOGGER.warning(last_error)
            continue
        ok, reason = _validate_shape(response)
        if not ok:
            last_error = f"schema violation: {reason}"
            prior_spec_json = json.dumps(response) if isinstance(response, (dict, list)) else ""
            _LOGGER.warning("LLM spec shape invalid: %s", reason)
            continue
        prior_spec_json = json.dumps(response)

        # Sandbox gate: reference must pass all unit tests; masked version
        # must fail at least one.
        try:
            verdict = sandbox_runner.evaluate_experiment_sample(
                data_code=response["data_code"],
                main_code=response["main_code"],
                incomplete_main_code=response["incomplete_main_code"],
                unit_tests=[dict(ut) for ut in response["unit_tests"]],
            )
        except Exception as exc:
            last_error = f"sandbox harness crashed: {exc}"
            _LOGGER.warning(last_error)
            continue
        if verdict.get("verdict") != "passed":
            ref = verdict.get("reference", {}) or {}
            ref_tests = ref.get("test_results") or []
            # Build a DETAILED feedback blob for the retry prompt so the
            # LLM can see expected vs actual. This is the difference between
            # 2% and 60% retry success rate: telling the model it's wrong
            # without telling it HOW rarely fixes anything.
            feedback_lines: list[str] = []
            if ref.get("compile_error"):
                feedback_lines.append(f"COMPILE ERROR in main_code: {ref['compile_error']}")
            for t in ref_tests:
                name = t.get("name", "?")
                if t.get("passed"):
                    continue
                if t.get("error"):
                    feedback_lines.append(
                        f"test '{name}': raised {t['error'][:200]}"
                    )
                else:
                    # include observed actual so LLM can copy it into expected
                    actual = t.get("actual")
                    feedback_lines.append(
                        f"test '{name}': expected != actual, observed actual={actual}"
                    )
            if not feedback_lines:
                feedback_lines.append("reference unit tests failed (no details)")
            inc = verdict.get("incomplete", {}) or {}
            if inc.get("failed", 0) == 0:
                feedback_lines.append(
                    "incomplete version still passes all tests — the blanked "
                    "function's output doesn't affect the test assertions"
                )
            last_error = (
                "sandbox rejected your spec. Specific failures:\n  - "
                + "\n  - ".join(feedback_lines)
                + "\nFIX: either change `expected_output` to match the actual "
                "observed values above, OR rewrite the code so it produces the "
                "values you originally claimed. Prefer the former when the "
                "observed value is stable and sensible."
            )
            _LOGGER.warning(
                "sandbox rejected LLM spec: ref=%s/%s inc_failed=%s reasons=%s feedback_lines=%d",
                ref.get("passed"), ref.get("total"),
                inc.get("failed"),
                verdict.get("rejection_reasons"),
                len(feedback_lines),
            )
            # Dump the full feedback text so we can see what the LLM is
            # being told on retry — without this, debugging the retry
            # loop is guesswork.
            for line in feedback_lines:
                _LOGGER.warning("  feedback: %s", line[:400])
            # Also dump the raw reference test_results (short form) so we
            # can see exactly what each test expected vs what was observed.
            for t in ref_tests:
                _LOGGER.warning(
                    "  ref_test: name=%r passed=%s expected_from_spec=%s actual=%r error=%r",
                    t.get("name"), t.get("passed"),
                    next((ut.get("expected_output") for ut in response.get("unit_tests", [])
                          if ut.get("name") == t.get("name")), None),
                    t.get("actual"), (t.get("error") or "")[:200],
                )
            continue

        # Success — annotate + return
        response.setdefault("task_family", "llm_experiment")
        response.setdefault("research_direction", f"{head} {relation} {tail}")
        response.setdefault("discipline", "life_sciences")
        response.setdefault("function_type", "quantitative_pipeline")
        response.setdefault("task_objective", response["research_direction"])
        response.setdefault("research_focus", response["research_direction"])
        response["_sandbox_evaluation"] = verdict
        response["generation_source"] = "llm"
        response["generation_attempts"] = attempt + 1
        return response

    _LOGGER.warning(
        "LLM experiment generation exhausted %d retries; last error: %s",
        max_retries + 1, last_error,
    )
    return None


__all__ = ["generate_experiment_via_llm"]
