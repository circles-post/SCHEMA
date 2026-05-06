#!/usr/bin/env python3
"""Background AGDebugger runner with dataset auto-eval and LLM trajectory debug.

What it does:
1) Starts AGDebugger backend in the background (or reuse an existing backend).
2) Loads a dataset component and feeds each question to AGDebugger automatically.
3) Runs the team until idle for each question.
4) If runtime error logs appear or answer is wrong, invokes external LLM planner
   to debug the trajectory through AGDebugger edit/insert APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from external_agent.integration import analyze_session_state
from external_agent_controller import AGDebuggerClient, LLMPlanner, execute_action
from model_routing import resolve_base_url_for_model, resolve_value_for_model

# Reuse existing dataset utilities from the workspace dataset package.
# Allow override via env so this runner is portable to other workstations.
DATASETS_DIR = Path(os.environ.get("AGDEBUGGER_DATASETS_DIR", ""))
if DATASETS_DIR.exists() and str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from browse_bio_graph_cluster_examples import (  # noqa: E402
    annotate_focus_nodes,
    load_component_nodes,
    load_examples_in_component,
    normalize_text,
)


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
DIRECT_ANSWER_PATTERNS = (
    re.compile(r"\b(?:correct|final)\s+answer\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+answer|answer)\s+(?:is|would be)\s+option\s*[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\b(?:choose|select|pick)\s+option\s*[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\boption\s*[a-z0-9]+\s+(?:is|would be|looks like)\s+(?:the\s+)?(?:correct|best|right)\b", re.IGNORECASE),
    re.compile(r"\b(?:supports|points to|favors)\s+option\s*[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"\b(?:therefore|thus|hence)\b.{0,80}\boption\s*[a-z0-9]+\b", re.IGNORECASE),
)
ANSWER_SELECTION_LANGUAGE_PATTERNS = (
    re.compile(r"\bthe\s+(?:most\s+scientifically\s+plausible|best|correct)\s+(?:explanation|choice|answer)\s+is\b", re.IGNORECASE),
    re.compile(r"\bthe\s+best\s+option\s+is\b", re.IGNORECASE),
    re.compile(r"\bthe\s+correct\s+choice\s+is\b", re.IGNORECASE),
    re.compile(r"\bfinal\s+(?:choice|answer)\b", re.IGNORECASE),
)
CORRECTED_CLAIM_CONTROL_RE = re.compile(
    r"\b(re-check|continue reasoning|do-not-carry-forward|required re-evaluation|actionable instruction|do not conclude|do not carry forward|looking at our options again|therefore[, ]|the correct answer is|therefore the correct answer|the answer is|let me analyze|from the search results|i can now see that)\b",
    re.IGNORECASE,
)
OPTION_REF_RE = re.compile(r"\boption\s*[a-z0-9]+\b", re.IGNORECASE)
REASONING_REPAIR_START = "[REASONING REPAIR]"
REASONING_REPAIR_END = "[END REASONING REPAIR]"
LEGACY_CONCEPT_PATCH_START = "[CONCEPT CORRECTION]"
LEGACY_CONCEPT_PATCH_END = "[END CONCEPT CORRECTION]"
SUPPORTED_PATCH_MARKERS = (
    (REASONING_REPAIR_START, REASONING_REPAIR_END),
    (LEGACY_CONCEPT_PATCH_START, LEGACY_CONCEPT_PATCH_END),
)


def _log_event(log_fh, event: str, **data):
    """Write a single JSONL event to the log file."""
    if log_fh is None:
        return
    entry = {"event": event, "ts": datetime.datetime.now().isoformat(), **data}
    log_fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    log_fh.flush()


def _resolve_log_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir = Path("./logs") / run_timestamp[:8]
    run_dir = date_dir / f"run_{run_timestamp}"

    run_log = args.run_log if args.run_log is not None else run_dir / "run.jsonl"
    server_log = args.server_log if args.server_log is not None else run_dir / "server.log"
    analysis_detail_log = (
        args.analysis_detail_log if args.analysis_detail_log is not None else run_dir / "analysis_detail.jsonl"
    )

    run_log = Path(run_log)
    server_log = Path(server_log)
    analysis_detail_log = Path(analysis_detail_log)
    run_dir = run_log.parent

    run_dir.mkdir(parents=True, exist_ok=True)
    server_log.parent.mkdir(parents=True, exist_ok=True)
    analysis_detail_log.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, run_log, server_log, analysis_detail_log


def _analysis_log_summary(analysis: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(analysis, dict):
        return {"has_analysis": False}
    claims = analysis.get("claims")
    judgments = analysis.get("judgments")
    concept_repair = analysis.get("concept_repair")
    return {
        "has_analysis": True,
        "analysis_error": analysis.get("analysis_error"),
        "analysis_fallback_reason": analysis.get("analysis_fallback_reason"),
        "analysis_task": analysis.get("analysis_task"),
        "analysis_use_websearch": analysis.get("analysis_use_websearch"),
        "analyzed_turn_count": analysis.get("analyzed_turn_count"),
        "extract_elapsed_sec": analysis.get("extract_elapsed_sec"),
        "judge_elapsed_sec": analysis.get("judge_elapsed_sec"),
        "claim_count": len(claims) if isinstance(claims, list) else None,
        "judgment_count": len(judgments) if isinstance(judgments, list) else None,
        "has_concept_repair": isinstance(concept_repair, dict),
        "hallucination_yes_count": analysis.get("hallucination_yes_count"),
        "verification_error_yes_count": analysis.get("verification_error_yes_count"),
    }


def _log_analysis_event(
    log_fh,
    *,
    phase: str,
    question_index: int,
    step: int | None,
    model: str,
    timeout_sec: float,
    use_websearch: bool,
    state: Dict[str, Any] | None = None,
    analysis: Dict[str, Any] | None = None,
    elapsed_sec: float | None = None,
    status: str = "ok",
    error: str | None = None,
) -> None:
    messages = _session_messages(state or {}) if isinstance(state, dict) else []
    assistant_messages = _assistant_messages(messages)
    _log_event(
        log_fh,
        "analysis_trace",
        phase=phase,
        index=question_index,
        step=step,
        status=status,
        model=model,
        timeout_sec=timeout_sec,
        use_websearch=use_websearch,
        elapsed_sec=elapsed_sec,
        error=error,
        session_message_count=len(messages),
        assistant_message_count=len(assistant_messages),
        analysis_summary=_analysis_log_summary(analysis),
        analysis=analysis,
    )


def format_task(example: Dict[str, Any]) -> str:
    lines = [f"Q: {example['question']}"]
    focus_node = normalize_text(example.get("focus_node", ""))
    if focus_node:
        lines.append(f"Focus node: {focus_node}")
    lines.append("")
    lines.append("Options:")
    for name, text in example["options"]:
        lines.append(f"  - {name}: {text}")
    return "\n".join(lines)


def normalize_answer(raw: str, num_options: int = 6) -> str:
    raw = raw.strip()
    letter_to_num = {chr(ord("A") + i): i + 1 for i in range(26)}
    cleaned = re.sub(r"^option[\s_:]*", "", raw, flags=re.IGNORECASE).strip()

    if len(cleaned) == 1 and cleaned.upper() in letter_to_num:
        n = letter_to_num[cleaned.upper()]
        if 1 <= n <= num_options:
            return f"option{n}"
    if cleaned.isdigit():
        return f"option{cleaned}"

    collapsed = raw.lower().replace(" ", "").replace("_", "")
    m = re.match(r"^option(\d+)$", collapsed)
    if m:
        return f"option{m.group(1)}"

    m = re.match(r"^option\s*([a-zA-Z])$", raw, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        if letter in letter_to_num:
            n = letter_to_num[letter]
            if 1 <= n <= num_options:
                return f"option{n}"

    return collapsed


def _forced_initial_answer_override() -> str:
    return os.environ.get("AGDEBUGGER_FORCE_INITIAL_ANSWER", "").strip()


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


def extract_answer_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    candidates: List[str] = []
    for msg in messages:
        for s in _iter_strings(msg):
            for m in ANSWER_TAG_RE.finditer(s):
                candidates.append(m.group(1).strip())
    return candidates[-1] if candidates else None


def _session_messages(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    current_session = state.get("current_session")
    history = state.get("message_history", {})
    if str(current_session) in history:
        return history[str(current_session)].get("messages", [])
    if current_session in history:
        return history[current_session].get("messages", [])
    return []


def _last_timestamp(messages: List[Dict[str, Any]]) -> int:
    ts_vals = [m.get("timestamp") for m in messages if isinstance(m.get("timestamp"), int)]
    return max(ts_vals) if ts_vals else -1


def _messages_after_timestamp(messages: List[Dict[str, Any]], after_ts: int) -> List[Dict[str, Any]]:
    return [m for m in messages if isinstance(m.get("timestamp"), int) and m["timestamp"] > after_ts]


def _scope_history_state_after_timestamp(state: Dict[str, Any], after_ts: int) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {"current_session": 0, "message_history": {0: {"messages": []}}}

    current_session = state.get("current_session", 0)
    history = state.get("message_history", {})
    session = history.get(str(current_session), history.get(current_session, {}))
    scoped_session = dict(session) if isinstance(session, dict) else {}
    scoped_session["messages"] = _messages_after_timestamp(
        scoped_session.get("messages", []) if isinstance(scoped_session.get("messages", []), list) else [],
        after_ts,
    )
    return {
        "current_session": current_session,
        "message_history": {
            str(current_session): scoped_session,
        },
    }


def _scope_snapshot_after_timestamp(snapshot: Dict[str, Any], after_ts: int) -> Dict[str, Any]:
    scoped = dict(snapshot)
    scoped["session_history"] = _scope_history_state_after_timestamp(snapshot.get("session_history", {}), after_ts)
    return scoped


def _has_termination_signal(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        for s in _iter_strings(msg):
            lowered = s.lower()
            if "terminate" in lowered or "text 'terminate' mentioned" in lowered:
                return True
        message_obj = msg.get("message")
        if isinstance(message_obj, dict) and str(message_obj.get("type", "")).lower() in {"stopmessage", "groupchattermination"}:
            return True
    return False


def _guess_manager_topic(topics: List[str]) -> str:
    for t in topics:
        if "manager" in t.lower():
            return t
    if not topics:
        raise RuntimeError("No topics returned by AGDebugger backend.")
    return topics[0]


def _is_error_log(entry: Dict[str, Any]) -> bool:
    level = str(entry.get("level", "")).upper()
    msg = str(entry.get("message", ""))
    if level in {"ERROR", "CRITICAL"}:
        return True
    needles = ("Traceback", "Exception", "ERROR", "[WARN]")
    return any(n in msg for n in needles)


def _default_debug_claim_task(claim_task: str | None) -> str:
    return claim_task or "scientific_concept_discovery"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalized_flag(value: Any) -> str:
    return str(value or "").strip().lower()


def _judgment_repair_type(judgment: Dict[str, Any]) -> str | None:
    hallucination = _normalized_flag(judgment.get("hallucination"))
    verification_error = _normalized_flag(judgment.get("verification_error"))
    if hallucination == "yes":
        return "hallucination"
    if verification_error == "yes":
        return "verification_error"
    return None


_CATEGORY_LABEL_PREFIXES = (
    "answer grounding",
    "answer alignment",
    "entity compatibility",
    "mapping claim",
    "alignment claim",
    "constraint claim",
)


def _is_category_label_concept_name(concept_name: str) -> bool:
    """Return True if *concept_name* is a category label (e.g. ``Answer
    grounding: ...``) rather than a real domain concept.

    These labels arrive when the upstream judge couldn't surface a
    ``concept_true_understanding`` and we'd otherwise echo the category
    name verbatim back into the trajectory, producing junk repairs like
    ``"The evidence does not support the previous mapping involving
    Answer grounding: [option redacted]."``.
    """
    if not concept_name:
        return False
    head = concept_name.strip().lower().lstrip("[(")
    return any(head.startswith(prefix) for prefix in _CATEGORY_LABEL_PREFIXES)


_NEGATION_LEAD_RE = re.compile(
    r"^\s*(?:the\s+)?(?:evidence|previous[\s\w]+|prior[\s\w]+)\s+"
    r"(?:does\s+not|did\s+not|was\s+not|is\s+not|cannot)",
    re.IGNORECASE,
)


def _extract_positive_evidence_hint(judgment_fields: Dict[str, str] | None) -> str:
    """Pick one positive evidence sentence from the raw judgment fields.

    Cat F: when ``concept_true_understanding`` is empty and
    ``_fallback_concept_guidance`` produces a pure-negation sentence
    (``"The evidence does not support ..."``), the replacement has zero
    actionable content and the agent has no new information to reason
    with. This helper returns the first non-empty, non-boilerplate line
    from the judgment's grounding fields so the caller can prepend it
    as a positive premise.
    """
    if not isinstance(judgment_fields, dict):
        return ""
    # Order matters: content_grounding is usually the most specific.
    for key in ("content_grounding", "reference_grounding", "hallucination", "verification_error"):
        value = judgment_fields.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if len(text) < 20:
            continue
        # Drop trivial phrases.
        lowered = text.lower()
        if lowered.startswith(("n/a", "none", "not applicable", "unknown", "unable")):
            continue
        # Take only the first sentence to keep the hint compact.
        first = _split_first_sentence(text)
        if len(first) >= 20:
            return first
    return ""


def _fallback_concept_guidance(
    *,
    concept_name: str,
    claim_text: str,
    reason: str,
    repair_type: str,
    error_type: str = "fact_error",
    judgment_fields: Dict[str, str] | None = None,
) -> str:
    # Strip category-label "concept names" so the guidance does not echo
    # the meta-label back into the trajectory as if it were a real concept.
    if _is_category_label_concept_name(concept_name):
        concept_label = "the prior inference"
    else:
        concept_label = concept_name or "the previously used concept"
    if error_type == "mapping_error":
        guidance = (
            f"The evidence does not support the previous mapping involving {concept_label}. "
            f"Re-examine the evidence and consider whether a different conclusion "
            f"is better supported, especially one that directly addresses the specific "
            f"aspect the question is asking about."
        )
    elif error_type == "constraint_error":
        guidance = (
            f"The previous claim about {concept_label} violated a subject or domain constraint. "
            f"Identify the correct entity class or domain for the question's subject "
            f"and re-evaluate which reasoning path is compatible with it."
        )
    elif error_type == "alignment_error":
        guidance = (
            f"The previous conclusion about {concept_label} did not directly answer the asked target. "
            f"Re-read the question to identify the specific mechanism, entity, or "
            f"relationship being asked about, then look for the reasoning path that "
            f"most precisely matches that target."
        )
    elif repair_type == "hallucination":
        guidance = (
            f"The previous claim about {concept_label} was incorrect or unsupported by evidence. "
            f"Discard this claim and re-derive the reasoning from verified evidence only."
        )
    else:
        guidance = (
            f"The previous claim about {concept_label} was not verified from the available evidence. "
            f"Do not rely on this claim; re-evaluate using only the evidence that has been confirmed."
        )
    if reason:
        guidance = f"{guidance} {reason}".strip()
    # Cat F: if the guidance is still a pure-negation sentence with no
    # reason attached, try to prepend one positive evidence hint from the
    # judgment's grounding fields so the replacement carries actionable
    # information instead of empty denial.
    evidence_hint = _extract_positive_evidence_hint(judgment_fields)
    if evidence_hint and _NEGATION_LEAD_RE.search(guidance):
        guidance = f"Evidence indicates: {evidence_hint.rstrip('.')}. {guidance}"
    return guidance


_SENTENCE_ABBREVIATIONS = (
    "e.g", "i.e", "etc", "vs", "cf", "al", "et al", "approx", "no",
    "fig", "figs", "eq", "eqs", "ref", "refs", "ca", "viz",
    "dr", "mr", "mrs", "ms", "prof", "st", "mt",
    # Genus / species abbreviations common in life-science text.
    "a", "b", "c", "d", "e", "h", "l", "m", "n", "p", "s", "t",
    "spp", "sp", "subsp", "var",
)


def _split_first_sentence(text: str) -> str:
    """Split *text* at the first real sentence boundary.

    Treats periods that follow common abbreviations (``E.``, ``e.g.``,
    ``et al.``, ``Fig.``) as non-terminal so single-letter genus
    abbreviations like ``E. coli`` are not severed mid-name.
    """
    if not text:
        return text
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in ".!?":
            # Need a following whitespace + capital/quote/digit to count as boundary.
            j = i + 1
            while j < n and text[j] == " ":
                j += 1
            if j >= n:
                return text.strip()
            nxt = text[j]
            # A digit following a period is more often a citation year
            # ("et al. 2020") than a new sentence — treat it as non-terminal.
            if not (nxt.isupper() or nxt in "\"'(["):
                i = j
                continue
            # Walk back to find the token preceding the period.
            k = i - 1
            while k >= 0 and (text[k].isalpha() or text[k] == "."):
                k -= 1
            token = text[k + 1 : i].lower().rstrip(".")
            if token in _SENTENCE_ABBREVIATIONS:
                i = j
                continue
            return text[: i + 1].strip()
        i += 1
    return text.strip()


def _sanitize_corrected_claim_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    sanitized = " ".join(text.split()).strip()
    sanitized = _redact_option_references(sanitized)
    if not sanitized:
        return ""
    if "<answer>" in sanitized.lower() or "terminate" in sanitized.lower():
        return ""
    first_sentence = _split_first_sentence(sanitized)
    sanitized = first_sentence or sanitized
    # Reject pathological micro-fragments — they nearly always come from
    # abbreviation mis-splits (e.g. "The murR deletion strain in E.").
    if len(sanitized) < 40:
        return ""
    if len(sanitized) > 280:
        return ""
    if CORRECTED_CLAIM_CONTROL_RE.search(sanitized):
        return ""
    if _contains_direct_answer_rewrite(sanitized):
        return ""
    return sanitized


_PLANNER_NEGATIVE_STATUSES = {
    "false", "incorrect", "wrong", "invalid", "off", "unsupported",
    "misaligned", "partially_supported", "partially supported",
    "potentially incorrect", "potentially_incorrect", "mismatched",
    "not supported", "not_supported", "off-target", "off_target",
}


def _planner_status_is_negative(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    if not v:
        return False
    if v in _PLANNER_NEGATIVE_STATUSES:
        return True
    # Catch loose forms like "potentially incorrect", "partially supported"
    for token in _PLANNER_NEGATIVE_STATUSES:
        if token in v:
            return True
    return False


def _planner_status_to_error_type(planner_judgment: Dict[str, Any]) -> str:
    alignment = planner_judgment.get("alignment_status")
    mapping = planner_judgment.get("mapping_status")
    grounding = planner_judgment.get("answer_grounding_status")
    if _planner_status_is_negative(alignment):
        return "alignment_error"
    if _planner_status_is_negative(mapping):
        return "mapping_error"
    if _planner_status_is_negative(grounding):
        return "fact_error"
    # All statuses look positive but planner still picked decision=repair.
    # Default to mapping_error — the planner's selected_claim_reason will
    # typically be a mapping/inference concern rather than a pure fact error.
    return "mapping_error"


def _sanitize_planner_reason(text: str) -> str:
    """Sanitize a planner ``selected_claim_reason`` for use as ``corrected_claim_text``.

    Less aggressive than ``_sanitize_corrected_claim_text``: keeps up to two
    sentences (planner rationales typically describe premise + inference) and
    caps at 400 chars, but still redacts option references and rejects direct
    answer injection / control phrases.
    """
    if not isinstance(text, str):
        return ""
    raw = " ".join(text.split()).strip()
    if not raw:
        return ""
    lowered_raw = raw.lower()
    if "<answer>" in lowered_raw or "terminate" in lowered_raw or "[answer redacted]" in lowered_raw:
        return ""
    # Reject sycophantic / meta re-evaluation phrasing that arrives when the
    # planner echoes the agent's own apologies back as a "reason". Injecting
    # these into the trajectory makes the agent randomly flip to another wrong
    # option instead of correcting the underlying mistake.
    _SYCOPHANT_PHRASES = (
        "you're absolutely right", "you are absolutely right",
        "you're right to question", "you are right to question",
        "let me re-evaluate", "let me reevaluate", "let me re evaluate",
        "let me reconsider", "i apologize", "my previous",
        "i was wrong", "i was incorrect", "thank you for pointing",
        "looking at this again", "on second thought",
    )
    if any(phrase in lowered_raw for phrase in _SYCOPHANT_PHRASES):
        return ""
    sanitized = _redact_option_references(raw)
    if not sanitized or "[answer redacted]" in sanitized.lower():
        return ""
    parts = re.split(r"(?<=[.!?])\s+", sanitized, maxsplit=2)
    sanitized = " ".join(parts[:2]).strip() if parts else sanitized
    if len(sanitized) > 400:
        sanitized = sanitized[:400].rsplit(". ", 1)[0].strip()
        if not sanitized.endswith(('.', '!', '?')):
            sanitized += "."
    if not sanitized:
        return ""
    if CORRECTED_CLAIM_CONTROL_RE.search(sanitized):
        return ""
    if _contains_direct_answer_rewrite(sanitized):
        return ""
    return sanitized


def _claim_error_type(claim: Dict[str, Any], judgment: Dict[str, Any]) -> str:
    category = str(claim.get("category", "")).strip().lower()
    source_type = str(claim.get("source_type", "")).strip().lower()
    concept_name = str(judgment.get("reference_name") or claim.get("source_ref") or "").strip().lower()

    if category == "constraint_claim" or source_type == "entity_compatibility":
        return "constraint_error"
    if category == "answer_alignment_claim" or source_type == "answer_alignment":
        return "alignment_error"
    if category == "mapping_claim" or source_type == "answer_grounding":
        return "mapping_error"
    if concept_name.startswith("entity compatibility:"):
        return "constraint_error"
    if concept_name.startswith("answer alignment:"):
        return "alignment_error"
    if concept_name.startswith("answer grounding:"):
        return "mapping_error"
    return "fact_error"


_REPR_LOOKING_ANCHOR_RE = re.compile(
    r"^[\[\(\{][\s\[\(\{]*['\"]?\w+['\"]?\s*:",
)


def _anchor_looks_like_repr(text: str) -> bool:
    """Heuristic: reject anchors that look like a Python repr of a list/dict.

    The extractor LLM occasionally emits ``context_snippet`` as a structured
    value (e.g. tool output list). After str()-coercion these look like
    ``"[{'content': '...'}]"`` and can never be substring-matched in the
    real trajectory text.

    Patterns caught (all produce un-anchorable garbage):
      - ``[{'content': ...``  (Python repr of list-of-dicts)
      - ``{'key': ...``       (Python repr of dict)
      - ``{"key": ...``       (JSON object)
      - ``[{"key": ...``      (JSON array of objects)
    """
    if not isinstance(text, str):
        return True
    stripped = text.lstrip()
    if not stripped:
        return False
    # Quick prefix check before running the regex
    first_char = stripped[0]
    if first_char not in ("[", "{", "("):
        return False
    if _REPR_LOOKING_ANCHOR_RE.match(stripped):
        return True
    return False


def _extract_best_anchor(claim: Dict[str, Any], judgment: Dict[str, Any]) -> str:
    """Extract the best available substring anchor for precise local repair.

    The summarized analysis payload used by the debug loop does not always
    preserve the extractor's full `context_snippet`, so we prefer richer
    fields when they exist and otherwise fall back to the claim text itself.
    """

    def _clean(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = " ".join(text.split()).strip()
        if _anchor_looks_like_repr(cleaned):
            return ""
        return cleaned

    claim_text = _clean(claim.get("text"))
    context_snippet_raw = claim.get("context_snippet") or claim.get("original_statement")
    if not context_snippet_raw and isinstance(claim.get("data"), dict):
        context_snippet_raw = claim.get("data", {}).get("context_snippet")
    context_snippet = _clean(context_snippet_raw)
    if context_snippet:
        return context_snippet

    original_statement = _clean(claim.get("original_statement"))
    if original_statement and claim_text:
        for sentence in re.split(r"(?<=[.!?])\s+", original_statement):
            sentence = _clean(sentence)
            if sentence and claim_text in sentence:
                return sentence

    judgment_reason = _clean(judgment.get("reason"))
    if claim_text:
        return claim_text
    if original_statement:
        return original_statement
    return judgment_reason


def _bridge_planner_concept(
    analysis_context: Dict[str, Any] | None,
    claims_by_id: Dict[Any, Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Phase 1 bridge: synthesize a repairable concept from ``planner_judgment``.

    When per-claim judge produced no ``hallucination=Yes`` verdicts but the
    planner judge chose ``decision=repair``, carry that judgment forward as a
    synthesized concept so strict mode has something to repair instead of
    halting with ``no_repairable_concepts``.

    PR-2 contract tightening:
      - Trust the planner's explicit ``repair_concept_name`` /
        ``incorrect_understanding`` / ``correct_understanding`` fields when
        present; those travel through the prompt as the authoritative repair
        description.
      - Allow bridging even when all three status fields look positive, as
        long as the planner supplied the new repair fields (the prompt now
        requires them whenever decision=="repair").
      - Still tolerate legacy planners that only return
        ``selected_claim_reason`` by sanitizing that string as a fallback.
      - Reject the bridge (return ``None``) when the planner asked for repair
        but supplied neither a concrete correct_understanding nor a
        sanitizable selected_claim_reason — that is the
        ``planner_repair_contract_violation`` case the caller will record.
    """
    if not analysis_context:
        return None
    planner_judgment = analysis_context.get("planner_judgment")
    if not isinstance(planner_judgment, dict):
        return None
    decision = str(planner_judgment.get("decision", "")).strip().lower()
    if decision != "repair":
        return None
    selected_claim_id = planner_judgment.get("selected_claim_id")
    claim: Dict[str, Any] = {}
    if selected_claim_id:
        fetched = claims_by_id.get(selected_claim_id)
        if isinstance(fetched, dict):
            claim = fetched

    # Prefer the planner-supplied correct_understanding; fall back to
    # sanitized selected_claim_reason for legacy planners.
    planner_correct = str(planner_judgment.get("correct_understanding", "")).strip()
    planner_incorrect = str(planner_judgment.get("incorrect_understanding", "")).strip()
    planner_concept_name = str(planner_judgment.get("repair_concept_name", "")).strip()
    planner_target_turn = planner_judgment.get("target_turn_number")

    corrected_claim_text = ""
    if planner_correct:
        corrected_claim_text = _sanitize_corrected_claim_text(planner_correct)
        if not corrected_claim_text:
            # Some planner responses are 2-3 sentences; sanitize with the
            # looser reason sanitizer before giving up.
            corrected_claim_text = _sanitize_planner_reason(planner_correct)
    if not corrected_claim_text:
        selected_reason = str(planner_judgment.get("selected_claim_reason", "")).strip()
        corrected_claim_text = _sanitize_planner_reason(selected_reason)
    if not corrected_claim_text:
        # Planner asked for repair but supplied no usable material.
        return None

    error_type = _planner_status_to_error_type(planner_judgment)
    repair_type = "planner_flagged"
    claim_text = (planner_incorrect or str(claim.get("text", "")).strip()).strip()
    concept_name = (
        planner_concept_name
        or str(claim.get("source_ref", "")).strip()
        or "the agent's selected reasoning step"
    )
    # Do not let the bridge replay any option label the planner accidentally
    # mentioned in the concept_name — the downstream replacement text must
    # never directly name an answer choice.
    concept_name = _redact_option_references(concept_name)

    # Fake a judgment dict so `_extract_best_anchor` can run on the synthesized
    # concept (it only reads anchor-related keys).
    synthetic_judgment = {
        "claim_id": selected_claim_id,
        "reference_name": concept_name,
        "reason": corrected_claim_text,
        "hallucination": "Yes",
        "verification_error": "No",
    }

    turn_number = claim.get("turn_number")
    if turn_number is None and isinstance(planner_target_turn, int):
        turn_number = planner_target_turn

    return {
        "claim_id": selected_claim_id or None,
        "turn_number": turn_number,
        "source_timestamp": claim.get("source_timestamp"),
        "analysis_assistant_turn": claim.get("analysis_assistant_turn"),
        "concept_name": concept_name,
        "incorrect_understanding": claim_text,
        "correct_understanding": corrected_claim_text,
        "corrected_claim_text": corrected_claim_text,
        "original_context": claim_text,
        "faulty_text_anchor": _extract_best_anchor(claim, synthetic_judgment) if claim else "",
        "reason": corrected_claim_text,
        "evidence_basis": [],
        "repair_type": repair_type,
        "error_type": error_type,
        "claim_category": claim.get("category"),
        "claim_source_type": claim.get("source_type"),
        "planner_bridged": True,
        "planner_target_turn_number": planner_target_turn if isinstance(planner_target_turn, int) else None,
    }


def _build_concept_repair_context(analysis_context: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not analysis_context:
        return None

    claims = analysis_context.get("claims", [])
    judgments = analysis_context.get("judgments", [])
    if not isinstance(claims, list) or not isinstance(judgments, list):
        return None

    claims_by_id = {
        claim.get("claim_id"): claim
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_id") is not None
    }

    hallucinated_concepts = []
    for judgment in judgments:
        if not isinstance(judgment, dict):
            continue
        repair_type = _judgment_repair_type(judgment)
        if repair_type is None:
            continue
        claim = claims_by_id.get(judgment.get("claim_id"), {})
        concept_name = judgment.get("reference_name", claim.get("source_ref"))
        claim_text = str(claim.get("text", "")).strip()
        reason = str(judgment.get("reason", "")).strip()
        correct_understanding = str(judgment.get("concept_true_understanding", "")).strip()
        error_type = _claim_error_type(claim, judgment)
        corrected_claim_text = _sanitize_corrected_claim_text(correct_understanding)
        if not corrected_claim_text:
            judgment_fields = {
                k: str(judgment.get(k, "")) for k in (
                    "content_grounding",
                    "reference_grounding",
                    "hallucination",
                    "verification_error",
                )
            }
            corrected_claim_text = _sanitize_corrected_claim_text(
                _fallback_concept_guidance(
                    concept_name=str(concept_name or "").strip(),
                    claim_text=claim_text,
                    reason=reason,
                    repair_type=repair_type,
                    error_type=error_type,
                    judgment_fields=judgment_fields,
                )
            )
        if not corrected_claim_text:
            corrected_claim_text = _redact_option_references(claim_text)
        hallucinated_concepts.append(
            {
                "claim_id": judgment.get("claim_id"),
                "turn_number": judgment.get("turn_number", claim.get("turn_number")),
                "source_timestamp": claim.get("source_timestamp"),
                "analysis_assistant_turn": claim.get("analysis_assistant_turn"),
                "concept_name": concept_name,
                "incorrect_understanding": claim_text,
                "correct_understanding": corrected_claim_text,
                "corrected_claim_text": corrected_claim_text,
                "original_context": claim_text,
                "faulty_text_anchor": _extract_best_anchor(claim, judgment),
                "reason": reason,
                "evidence_basis": judgment.get("reference_grounding", []),
                "repair_type": repair_type,
                "error_type": error_type,
                "claim_category": claim.get("category"),
                "claim_source_type": claim.get("source_type"),
            }
        )

    hallucinated_concepts.sort(
        key=lambda item: (
            0 if str(item.get("repair_type")) == "hallucination" else 1,
            float(item["turn_number"]) if isinstance(item.get("turn_number"), (int, float)) else float("inf"),
        )
    )

    if not hallucinated_concepts:
        bridged = _bridge_planner_concept(analysis_context, claims_by_id)
        if bridged is not None:
            hallucinated_concepts.append(bridged)

    if not hallucinated_concepts:
        # PR-2 diagnostic: record whether this "no concepts" state is caused
        # by the planner asking for repair but not supplying repair material
        # (contract violation) vs. the judge honestly finding nothing wrong.
        planner_judgment = analysis_context.get("planner_judgment")
        planner_wanted_repair = (
            isinstance(planner_judgment, dict)
            and str(planner_judgment.get("decision", "")).strip().lower() == "repair"
        )
        return {
            "workflow": [
                "extract fact, mapping, constraint, and answer-alignment claims from the assistant trajectory",
                "verify each claim against subject constraints and external evidence",
                "identify incorrect or unsupported local reasoning steps",
                "replace only the incorrect or unsupported local reasoning before rerunning",
            ],
            "hallucinated_concepts": [],
            "planner_repair_contract_violation": bool(planner_wanted_repair),
        }

    return {
        "workflow": [
            "extract fact, mapping, constraint, and answer-alignment claims from the assistant trajectory",
            "verify each claim against subject constraints and external evidence",
            "identify incorrect or unsupported local reasoning steps",
            "replace only the incorrect or unsupported local reasoning before rerunning",
        ],
        "hallucinated_concepts": hallucinated_concepts,
        "replacement_guidance": (
            "Prioritize the earliest assistant turn that introduced each repairable concept. "
            "Replace the incorrect or unsupported local reasoning with the provided corrected guidance, "
            "preserve unaffected reasoning, and only then let the trajectory continue."
        ),
    }


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _message_content(msg: Dict[str, Any]) -> str:
    message_obj = msg.get("message")
    if isinstance(message_obj, dict):
        if isinstance(message_obj.get("content"), str):
            return str(message_obj["content"])
        nested_message = message_obj.get("message")
        if isinstance(nested_message, dict) and isinstance(nested_message.get("content"), str):
            return str(nested_message["content"])
        response = message_obj.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("content"), str):
                return str(chat_message["content"])
    if isinstance(msg.get("content"), str):
        return str(msg["content"])
    return ""


def _message_source(msg: Dict[str, Any]) -> str:
    message_obj = msg.get("message")
    if isinstance(message_obj, dict):
        if isinstance(message_obj.get("source"), str):
            return str(message_obj["source"])
        nested_message = message_obj.get("message")
        if isinstance(nested_message, dict) and isinstance(nested_message.get("source"), str):
            return str(nested_message["source"])
        response = message_obj.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("source"), str):
                return str(chat_message["source"])
    if isinstance(msg.get("source"), str):
        return str(msg["source"])
    return ""


def _message_type(msg: Dict[str, Any]) -> str:
    nested = msg.get("message")
    if isinstance(nested, dict):
        inner_message = nested.get("message")
        if isinstance(inner_message, dict) and isinstance(inner_message.get("type"), str):
            return str(inner_message["type"])
        response = nested.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("type"), str):
                return str(chat_message["type"])
        if isinstance(nested.get("type"), str):
            return str(nested["type"])
    if isinstance(msg.get("type"), str):
        return str(msg["type"])
    return ""


def _is_editable_assistant_message(msg: Dict[str, Any]) -> bool:
    source = _message_source(msg)
    if not source or source == "user":
        return False
    if "GroupChatManager" in source or source.endswith("Termination"):
        return False

    message_type = _message_type(msg)
    if message_type in {"ThoughtEvent", "TextMessage", "ToolCallRequestEvent", "ToolCallExecutionEvent"}:
        return True

    nested = msg.get("message")
    return isinstance(nested, dict) and nested.get("type") == "GroupChatAgentResponse"


def _assistant_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [msg for msg in messages if _is_editable_assistant_message(msg)]


def _assistant_turn_number_for_message(messages: List[Dict[str, Any]], target_message: Dict[str, Any]) -> int | None:
    target_timestamp = target_message.get("timestamp")
    assistant_turn = 0
    for msg in messages:
        if not _is_editable_assistant_message(msg):
            continue
        assistant_turn += 1
        if msg is target_message:
            return assistant_turn
        if (
            isinstance(target_timestamp, int)
            and isinstance(msg.get("timestamp"), int)
            and msg.get("timestamp") == target_timestamp
        ):
            return assistant_turn
    return None


def _assistant_timestamp_for_turn(messages: List[Dict[str, Any]], target_turn: int | float | None) -> int | None:
    if not isinstance(target_turn, (int, float)):
        return None
    assistant_turn = 0
    for msg in messages:
        if not _is_editable_assistant_message(msg):
            continue
        assistant_turn += 1
        if assistant_turn == int(target_turn):
            timestamp = msg.get("timestamp")
            return int(timestamp) if isinstance(timestamp, int) else None
    return None


def _has_meaningful_agent_activity(messages: List[Dict[str, Any]]) -> bool:
    """Return True only if the timeout snapshot contains actual agent output.

    We do not want to treat pure bootstrap/user messages as a meaningful
    trajectory, because those cases should still be eligible for a retry.
    """
    if not messages:
        return False
    if _assistant_messages(messages):
        return True
    return extract_answer_from_messages(messages) is not None or _has_termination_signal(messages)


def _find_empty_agent_text_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for msg in messages:
        source = _message_source(msg)
        if not source or source == "user":
            continue
        if "GroupChatManager" in source or source.endswith("Termination"):
            continue
        if _message_type(msg) != "TextMessage":
            continue
        if _message_content(msg).strip():
            continue
        return msg
    return None


def _hallucinated_concepts(analysis_context: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    repair = (analysis_context or {}).get("concept_repair")
    if not isinstance(repair, dict):
        return []
    concepts = repair.get("hallucinated_concepts")
    if not isinstance(concepts, list):
        return []
    return [item for item in concepts if isinstance(item, dict)]


def _extract_repair_blocks(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    blocks: List[str] = []
    lowered = text.lower()
    for patch_start, patch_end in SUPPORTED_PATCH_MARKERS:
        start_l = patch_start.lower()
        end_l = patch_end.lower()
        cursor = 0
        while True:
            start_idx = lowered.find(start_l, cursor)
            if start_idx == -1:
                break
            end_idx = lowered.find(end_l, start_idx)
            if end_idx == -1:
                break
            end_idx += len(end_l)
            blocks.append(text[start_idx:end_idx])
            cursor = end_idx
    return blocks


def _contains_direct_answer_rewrite(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    if "<answer>" in lowered or "terminate" in lowered:
        return True
    check_text = lowered
    for patch_start, patch_end in SUPPORTED_PATCH_MARKERS:
        patch_start = patch_start.lower()
        patch_end = patch_end.lower()
        while patch_start in check_text:
            start_idx = check_text.index(patch_start)
            end_idx = check_text.find(patch_end, start_idx)
            if end_idx == -1:
                check_text = check_text[:start_idx]
            else:
                check_text = check_text[:start_idx] + check_text[end_idx + len(patch_end):]
    if any(pattern.search(check_text) for pattern in DIRECT_ANSWER_PATTERNS):
        return True
    return any(pattern.search(check_text) for pattern in ANSWER_SELECTION_LANGUAGE_PATTERNS)


def _repair_blocks_contain_option_identifiers(text: str) -> bool:
    blocks = _extract_repair_blocks(text)
    if blocks:
        return any(OPTION_REF_RE.search(block) for block in blocks)
    return bool(OPTION_REF_RE.search(text))


def _is_structured_concept_patch(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    if not any(start.lower() in lowered and end.lower() in lowered for start, end in SUPPORTED_PATCH_MARKERS):
        return False
    if "concept:" not in lowered:
        return False
    has_instruction = (
        "instruction:" in lowered
        or "actionable instruction:" in lowered
        or "required re-evaluation:" in lowered
    )
    if not has_instruction:
        return False
    has_wrong = (
        "original:" in lowered
        or "original unsupported claim:" in lowered
        or "faulty claim:" in lowered
    )
    has_fix = (
        "corrected:" in lowered
        or "corrected grounded guidance:" in lowered
        or "correction:" in lowered
    )
    return has_wrong and has_fix


def _build_debug_goal(task: str, snippet_logs: str) -> str:
    return (
        "Debug AGDebugger trajectory by repairing hallucinated scientific concepts only.\n"
        "Your role is NOT to solve the multiple-choice question directly.\n"
        "Your role is to:\n"
        "1) inspect the answering agent's trajectory for incorrect scientific concepts or incorrect interactions between concepts;\n"
        "2) verify those concepts or concept interactions with external evidence;\n"
        "3) identify the earliest assistant step where the hallucinated concept or interaction first appears;\n"
        "4) supplement that step with a corrected, actionable concept understanding;\n"
        "5) let the rerun propagate the corrected understanding to the final answer.\n"
        "Never output or imply the final answer choice, option label, <answer> tag, or TERMINATE.\n"
        "Do not rank answer candidates, do not say which answer is correct, and do not restate the final conclusion.\n"
        "Only repair local concept understanding at the affected step.\n"
        "When you produce replacement_text, use exactly this block format:\n"
        "[REASONING REPAIR]\n"
        "Error Type: <fact_error | mapping_error | constraint_error | alignment_error>\n"
        "Concept: <concept, mapping, or constraint name>\n"
        "Faulty Claim: <the incorrect local understanding only>\n"
        "Correction: <fact-checked local understanding — be specific to the subject, not a generic definition>\n"
        "Why: <brief explanation of why the original claim was wrong>\n"
        "Required Re-evaluation: <what the agent should re-examine next>\n"
        "Do-Not-Carry-Forward: <the assumption or mapping that must not be reused>\n"
        "Actionable Instruction: <tell the agent what to re-examine and how to proceed — do not conclude with an answer selection>\n"
        "[END REASONING REPAIR]\n\n"
        "IMPORTANT: The Correction field must be specific to the subject and context of the question, "
        "NOT a generic textbook definition. A good correction explains what is actually true about "
        "the specific system (e.g., 'Chili RNA aptamer uses G-quadruplex stacking, not amino acid interactions') "
        "rather than a dictionary entry (e.g., 'π-π stacking is a non-covalent interaction between aromatic rings').\n\n"
        f"Question:\n{task}\n\n"
        "Current status: the trajectory currently ends in an incorrect final answer, but the explicit answer label is intentionally withheld.\n"
        f"Recent error logs:\n{snippet_logs}\n"
        "Use edit_and_revert actions only for repairs. Never inject <answer>, TERMINATE, direct final-answer choices, or answer-selection language.\n"
        "Prioritize replacing incorrect or unsupported scientific concept descriptions and concept interactions, then let the rerun derive the answer."
    )


def _redact_option_references(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    text = ANSWER_TAG_RE.sub("[answer redacted]", text)
    text = re.sub(r"\bTERMINATE\b", "[terminate redacted]", text, flags=re.IGNORECASE)
    text = OPTION_REF_RE.sub("[option redacted]", text)
    return text


def _action_payload_strings(action: Dict[str, Any]) -> List[str]:
    payload_strings: List[str] = []
    replacement_text = action.get("replacement_text")
    if isinstance(replacement_text, str):
        payload_strings.append(replacement_text)
    for key in ("body", "message"):
        if key in action:
            payload_strings.extend(list(_iter_strings(action.get(key))))
    return payload_strings


def _allow_answer_style_rewrite_fallback() -> bool:
    value = os.environ.get("AGDEBUGGER_ALLOW_ANSWER_STYLE_REWRITE_FALLBACK", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _validate_concept_repair_action(
    action: Dict[str, Any],
    *,
    analysis_context: Dict[str, Any] | None,
    require_repairable_concepts: bool,
) -> str | None:
    concepts = _hallucinated_concepts(analysis_context)
    if require_repairable_concepts and not concepts:
        return "concept-repair mode requires at least one repairable concept"

    action_name = str(action.get("action", "")).strip()
    if action_name != "edit_and_revert":
        return f"concept repair only allows edit_and_revert, got {action_name or '(empty)'}"

    replacement_text = action.get("replacement_text")
    if replacement_text is None and isinstance(action.get("body"), str):
        replacement_text = action.get("body")
    if not isinstance(replacement_text, str) or not replacement_text.strip():
        return "concept repair requires a non-empty replacement_text"

    claim_id = action.get("claim_id")
    strict_local_patch_only = not _allow_answer_style_rewrite_fallback()
    if claim_id is not None:
        matched_concepts = [item for item in concepts if item.get("claim_id") == claim_id]
        if len(matched_concepts) != 1:
            return "concept repair requires claim_id to resolve to exactly one repairable concept"
        corrected_claim_text = str(matched_concepts[0].get("corrected_claim_text", "")).strip()
        if not corrected_claim_text:
            return "concept repair target is missing corrected_claim_text"
        if corrected_claim_text not in replacement_text:
            return "concept repair replacement_text must include corrected_claim_text"
        if _contains_direct_answer_rewrite(corrected_claim_text):
            return "concept repair corrected_claim_text contains direct answer injection"
        if OPTION_REF_RE.search(corrected_claim_text):
            return "concept repair corrected_claim_text contains option identifiers"
        # When claim_id is present and corrected_claim_text passed validation,
        # we only check the patch content itself — not the full replacement_text,
        # because the full text includes the original message which may already
        # contain option/answer content that we are not modifying.
    else:
        if any(_contains_direct_answer_rewrite(text) for text in _action_payload_strings(action)):
            return "concept repair forbids direct answer injection in replacement_text/body"
        if OPTION_REF_RE.search(replacement_text):
            return "concept repair forbids option identifiers in replacement_text"
        if require_repairable_concepts and not _is_structured_concept_patch(replacement_text):
            return "strict concept repair requires either claim_id-driven replacement_text or a structured reasoning repair replacement_text"

    if action.get("target_turn") is None and action.get("timestamp") is None:
        return "concept repair requires target_turn or timestamp"
    return None


def _degraded_fallback_action_error(
    action: Dict[str, Any] | None,
    *,
    analysis_context: Dict[str, Any] | None,
    require_repairable_concepts: bool,
) -> str | None:
    if action is None:
        return "no_deterministic_fallback_action"
    if str(action.get("action", "")).strip() != "edit_and_revert":
        return "degraded fallback requires edit_and_revert"
    if not action.get("claim_id"):
        return "degraded fallback requires claim_id"
    if action.get("anchor_not_found"):
        return "degraded fallback could not localize a safe anchor"
    validation_error = _validate_concept_repair_action(
        action,
        analysis_context=analysis_context,
        require_repairable_concepts=require_repairable_concepts,
    )
    if validation_error is not None:
        return validation_error
    replacement_text = str(action.get("replacement_text") or "")
    lowered = replacement_text.lower()
    if "your_selected_option" in lowered:
        return "degraded fallback replacement_text still contains placeholder answer content"
    if "[system note]" in lowered:
        return "degraded fallback replacement_text still contains system note content"
    return None


def _build_reasoning_repair_block(concept: Dict[str, Any]) -> str:
    concept_name = _redact_option_references(str(concept.get("concept_name", "")).strip()) or "scientific concept"
    incorrect = _redact_option_references(str(concept.get("incorrect_understanding", "")).strip())
    correct = _redact_option_references(str(concept.get("correct_understanding", "")).strip())
    reason = _redact_option_references(str(concept.get("reason", "")).strip())
    repair_type = str(concept.get("repair_type", "")).strip() or "hallucination"

    error_type = str(concept.get("error_type", "")).strip() or {
        "hallucination": "fact_error",
        "verification_error": "constraint_error",
    }.get(repair_type, "fact_error")
    corrected_lines = [
        REASONING_REPAIR_START,
        f"Error Type: {error_type}",
        f"Concept: {concept_name}",
        f"Faulty Claim: {incorrect or '(missing original concept text)'}",
        f"Correction: {correct or '(missing corrected concept guidance)'}",
    ]
    if reason:
        corrected_lines.append(f"Why: {reason}")

    if repair_type == "hallucination":
        if error_type == "mapping_error":
            instruction = (
                "The above reasoning incorrectly connected evidence to an answer candidate. "
                "Re-check whether the evidence supports that answer-level mapping before continuing. "
                "Do not conclude with an answer selection."
            )
        elif error_type == "constraint_error":
            instruction = (
                "The above reasoning violated a subject or domain constraint. "
                "Re-examine the subject's intrinsic components and task constraints before continuing. "
                "Do not conclude with an answer selection."
            )
        elif error_type == "alignment_error":
            instruction = (
                "The above conclusion may not directly address the question's requested target. "
                "Re-evaluate the final reasoning so it answers the question more directly. "
                "Do not conclude with an answer selection."
            )
        else:
            instruction = (
                "The above understanding was incorrect or unsupported by evidence. "
                "Re-examine the subject's actual properties and the available evidence, "
                "then continue reasoning without relying on the faulty claim. "
                "Do not conclude with an answer selection."
            )
    else:
        instruction = (
            "The evidence was insufficient to support this claim. "
            "Remove this unsupported assumption, seek stronger evidence or "
            "reconsider the reasoning chain, then continue. "
            "Do not conclude with an answer selection."
        )
    corrected_lines.extend(
        [
            "Required Re-evaluation: Re-check the local reasoning step against the subject, evidence, and task constraints before continuing.",
            f"Do-Not-Carry-Forward: {incorrect or 'The previously stated unsupported assumption.'}",
            f"Actionable Instruction: {instruction}",
            REASONING_REPAIR_END,
        ]
    )
    return "\n\n".join(corrected_lines)


def _locate_anchor_in_content(content: str, anchor: str) -> tuple[int, int] | None:
    if not isinstance(content, str) or not isinstance(anchor, str):
        return None
    anchor = anchor.strip()
    if not content or not anchor:
        return None
    idx = content.find(anchor)
    if idx != -1:
        return idx, idx + len(anchor)
    normalized_anchor = " ".join(anchor.split())
    if not normalized_anchor:
        return None
    compact_content = " ".join(content.split())
    compact_idx = compact_content.find(normalized_anchor)
    if compact_idx == -1:
        return None

    # Reconstruct the range in the original string by mapping compact chars.
    mapping: List[int] = []
    compact_chars: List[str] = []
    prev_space = False
    for pos, ch in enumerate(content):
        if ch.isspace():
            if compact_chars and not prev_space:
                compact_chars.append(" ")
                mapping.append(pos)
            prev_space = True
            continue
        compact_chars.append(ch)
        mapping.append(pos)
        prev_space = False
    compact_text = "".join(compact_chars)
    compact_idx = compact_text.find(normalized_anchor)
    if compact_idx == -1:
        return None
    start = mapping[compact_idx]
    end = mapping[min(len(mapping) - 1, compact_idx + len(normalized_anchor) - 1)] + 1
    return start, end


def _locate_all_anchor_occurrences(content: str, anchor: str) -> List[tuple[int, int]]:
    """Return all non-overlapping occurrences of ``anchor`` in ``content``.

    Path C (multi-anchor) needs to find every place a specific anchor string
    resolves to — the single-hit ``_locate_anchor_in_content`` only returns
    the first occurrence. This helper iterates by advancing past each hit
    and retrying. Uses the same two-level (strict substring + whitespace
    normalized) matcher as the single-hit variant by delegating to it on
    each residual slice.
    """
    if not isinstance(content, str) or not isinstance(anchor, str):
        return []
    anchor = anchor.strip()
    if not content or not anchor:
        return []
    hits: List[tuple[int, int]] = []
    cursor = 0
    while cursor < len(content):
        hit = _locate_anchor_in_content(content[cursor:], anchor)
        if hit is None:
            break
        abs_start = cursor + hit[0]
        abs_end = cursor + hit[1]
        if abs_end <= abs_start:
            break
        hits.append((abs_start, abs_end))
        cursor = abs_end
    return hits


def _dedup_subsumed_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop spans that are strictly contained in another span.

    ``_concept_anchor_candidates`` produces long/medium/short variants of
    the same phrase. When all three match at the same position, the
    shorter hits are proper subsets of the longer one and should not
    contribute duplicate repair spans. Preserves the first-seen
    ``claim_id`` on the surviving (longest) span.
    """
    if not spans:
        return []
    # Sort by (start asc, span length desc) so that longer spans come first
    # at each position; a later shorter span at the same start is dropped
    # via the "is_subsumed" check below.
    ordered = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])) )
    survivors: List[Dict[str, Any]] = []
    for span in ordered:
        is_subsumed = False
        for kept in survivors:
            if kept["start"] <= span["start"] and span["end"] <= kept["end"]:
                is_subsumed = True
                break
        if not is_subsumed:
            survivors.append(dict(span))
    return survivors


def _fuse_close_spans(
    spans: List[Dict[str, Any]],
    *,
    max_gap: int,
    cross_claim: bool,
) -> List[Dict[str, Any]]:
    """Collapse adjacent spans whose intervening content is ≤ ``max_gap``.

    This is the core of Path C. Two separate anchor hits inside the same
    assistant turn — e.g. ``"Option 2 correctly identifies…"`` at offset A
    and ``"the correct answer is option 2"`` at offset B — get merged into
    a single span ``[min(A_start,B_start), max(A_end,B_end))``, and the
    intervening original text (option enumeration, transition sentences)
    is discarded along with the anchors themselves.

    Args:
        spans: already sorted by (start, end) with no overlaps.
        max_gap: ``span[i+1].start - span[i].end`` above this value keeps
            the spans distinct.
        cross_claim: when False, only fuse spans that share a claim_id.

    The surviving span keeps the first span's ``replacement`` text
    (primary_claim authoritative) and records every fused claim_id under
    ``fused_claim_ids`` for diagnostics.
    """
    if not spans:
        return []
    fused: List[Dict[str, Any]] = [dict(spans[0])]
    fused[0].setdefault("fused_claim_ids", [fused[0].get("claim_id")])
    for span in spans[1:]:
        last = fused[-1]
        same_claim = span.get("claim_id") == last.get("claim_id")
        eligible = cross_claim or same_claim
        gap = span["start"] - last["end"]
        if eligible and 0 <= gap <= max_gap:
            last["end"] = max(last["end"], span["end"])
            last["fused_claim_ids"].append(span.get("claim_id"))
            continue
        new_entry = dict(span)
        new_entry.setdefault("fused_claim_ids", [span.get("claim_id")])
        fused.append(new_entry)
    return fused


def _concept_anchor_candidates(concept: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []

    def _add(value: str) -> None:
        normalized = " ".join(value.split()).strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    def _expand(value: str) -> List[str]:
        text = " ".join(value.split()).strip()
        if not text:
            return []
        variants = [text]
        without_option = re.sub(r"(?i)^option\s*[a-z0-9]+\s*(?:is|:)?\s*", "", text).strip(" .:-")
        if without_option and without_option != text:
            variants.append(without_option)
        without_lead = re.sub(
            r"(?i)^(the agent (?:connects|claims|states|argues) that|the previous claim(?: about)?|it (?:suggests|claims|argues) that)\s+",
            "",
            without_option or text,
        ).strip(" .:-")
        if without_lead and without_lead not in variants:
            variants.append(without_lead)
        # Add a shorter leading clause before comma/period to improve matching when the
        # original assistant message paraphrases only the first part of the sentence.
        for sep in [",", ".", ";"]:
            head = (without_lead or without_option or text).split(sep, 1)[0].strip(" .:-")
            if len(head) >= 24 and head not in variants:
                variants.append(head)
        return variants

    for value in (
        concept.get("faulty_text_anchor"),
        concept.get("original_context"),
        concept.get("incorrect_understanding"),
    ):
        if not isinstance(value, str):
            continue
        for variant in _expand(value):
            _add(variant)
    return candidates


def _target_message_for_concept(
    assistant_messages: List[Dict[str, Any]],
    concept: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    source_timestamp = concept.get("source_timestamp")
    if isinstance(source_timestamp, int):
        for msg in assistant_messages:
            if msg.get("timestamp") == source_timestamp:
                return msg
    original_context = str(concept.get("original_context", "")).strip()
    if original_context:
        for msg in assistant_messages:
            if original_context in _message_content(msg):
                return msg
    for anchor in _concept_anchor_candidates(concept):
        for msg in assistant_messages:
            if _locate_anchor_in_content(_message_content(msg), anchor) is not None:
                return msg
    return assistant_messages[0] if assistant_messages else None


def _resolve_planner_span_cut(
    original_content: str,
    planner_span: Dict[str, Any] | None,
) -> int | None:
    """Resolve the planner-supplied ``wrong_reasoning_span`` into a character
    offset at which to cut the agent message's prefix.

    The planner returns a span as ``{"anchor_start": "...", "anchor_end": "..."}``
    where each anchor is a literal substring that exists in the message. We
    only care about ``anchor_start`` here — it marks where the agent's wrong
    reasoning begins, i.e. the earliest position the prefix cut may fall.

    Returns ``None`` if the span is absent, malformed, or cannot be located
    in ``original_content``. Also rejects anchors that land before the
    minimum prefix length (guards against the planner picking an anchor at
    offset 0 and effectively erasing the whole message).
    """
    if not isinstance(planner_span, dict) or not original_content:
        return None
    anchor_start = planner_span.get("anchor_start")
    if not isinstance(anchor_start, str):
        return None
    anchor_start = anchor_start.strip()
    if len(anchor_start) < 4:
        return None
    idx = original_content.find(anchor_start)
    if idx < 0:
        # Try a loose match on the first line of the anchor (planners
        # sometimes return the full reasoning block as anchor_start).
        first_line = anchor_start.split("\n", 1)[0].strip()
        if len(first_line) >= 4:
            idx = original_content.find(first_line)
        if idx < 0:
            return None
    min_prefix = max(1, int(os.environ.get("AGDEBUGGER_PLANNER_SPAN_MIN_PREFIX", "40")))
    if idx < min_prefix:
        return None
    return idx


_REFLECTION_OPENERS = (
    "Wait — let me re-check my earlier reasoning here.",
    "Hmm, actually, let me re-check this step before I move on.",
    "Hold on — something in what I just said does not line up with the evidence.",
)


def _reflection_enabled_for_strict_mode() -> bool:
    """Whether strict concept repair should use the first-person reflection
    rewrite instead of the legacy ``[REASONING REPAIR]`` directive block.

    Default ON. Flip to ``0`` to fall back to the legacy directive, e.g. for
    quick A/B comparisons during regression runs.
    """
    value = os.environ.get("AGDEBUGGER_STRICT_REFLECTION_REWRITE", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _build_reflection_block(concept: Dict[str, Any]) -> str:
    """Render a short first-person self-correction block in the agent's voice.

    PR-3: the replacement text must read like the agent's own thought stream
    ("Wait — let me re-check…") instead of a third-person ``[REASONING
    REPAIR]`` directive; otherwise the downstream rerun treats the injected
    text as meta commentary and walks straight back to the original wrong
    option (the idx=71 failure mode).

    The block MUST NOT name any option label, mention <answer>, or include
    TERMINATE — those are filtered out by the surrounding sanitizer and the
    final leak-assertion, but we also avoid them here at the template level
    so the happy path never reaches the assertion.
    """
    opener = _REFLECTION_OPENERS[0]
    concept_name = _redact_option_references(str(concept.get("concept_name", "")).strip() or "this step")
    incorrect = _redact_option_references(str(concept.get("incorrect_understanding", "")).strip())
    correct = _redact_option_references(str(concept.get("corrected_claim_text") or concept.get("correct_understanding") or "").strip())
    evidence_basis = concept.get("evidence_basis") or []
    evidence_hint = ""
    if isinstance(evidence_basis, list) and evidence_basis:
        # Use the first grounding URL/paper as a short "checking back at X" hint.
        first = str(evidence_basis[0]).strip()
        if first:
            evidence_hint = first[:120]
    elif isinstance(evidence_basis, str) and evidence_basis.strip():
        evidence_hint = evidence_basis.strip()[:120]

    lines: List[str] = [opener]
    if incorrect:
        lines.append(
            f"Earlier I treated {concept_name} as if {incorrect.rstrip('.')}, "
            "but that is not quite what the evidence actually supports."
        )
    else:
        lines.append(
            f"I want to double-check my earlier handling of {concept_name} before I "
            "commit to a final choice."
        )
    if evidence_hint:
        lines.append(f"Looking back at {evidence_hint}, the actual picture is different:")
    if correct:
        lines.append(correct)
    lines.append(
        "That means the premise I used a moment ago was not on solid ground, "
        "so I need to reconsider the available options one more time from this "
        "corrected understanding before I commit to my final choice."
    )
    return "\n".join(lines).strip()


class InvalidRepairText(ValueError):
    """Raised when a synthesized replacement_text would leak a direct answer.

    The guard is a last-resort safety net: the sanitizer paths already strip
    <answer>/TERMINATE/option labels, but a template change or planner
    regression could still slip material through. We'd rather fail loudly
    here and let the controller fall back to a different claim than silently
    pin the agent to a pre-written answer.
    """


def _assert_replacement_has_no_direct_answer(replacement_text: str) -> None:
    if not isinstance(replacement_text, str) or not replacement_text:
        return
    lowered = replacement_text.lower()
    if "<answer>" in lowered:
        raise InvalidRepairText("replacement_text contains <answer> tag")
    if "terminate" in lowered:
        raise InvalidRepairText("replacement_text contains TERMINATE marker")
    if OPTION_REF_RE.search(replacement_text):
        raise InvalidRepairText("replacement_text contains an option label (option N)")
    if any(pattern.search(replacement_text) for pattern in DIRECT_ANSWER_PATTERNS):
        raise InvalidRepairText("replacement_text contains direct-answer language")


_REWRITE_MODE_VALUES = {"off", "cross_claim_only", "on_fused", "always"}


def _normalize_rewrite_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        return "off"
    value = mode.strip().lower()
    if value in {"0", "false", "no", "none", ""}:
        return "off"
    if value in {"1", "true", "yes", "on"}:
        # Legacy truthy — default to the safest non-off mode.
        return "cross_claim_only"
    return value if value in _REWRITE_MODE_VALUES else "off"


def _rewrite_span_eligible(span: Dict[str, Any], *, mode: str, cross_claim_fuse: bool) -> bool:
    if mode == "off":
        return False
    claim_ids = [cid for cid in (span.get("fused_claim_ids") or []) if cid is not None]
    unique_claim_ids = set(claim_ids)
    is_fused = len(claim_ids) >= 2
    is_cross_claim = len(unique_claim_ids) >= 2
    if mode == "cross_claim_only":
        return cross_claim_fuse and is_cross_claim
    if mode == "on_fused":
        return is_fused
    if mode == "always":
        return True
    return False


def _maybe_apply_llm_rewrite(
    merged: List[Dict[str, Any]],
    *,
    original_content: str,
    concepts: List[Dict[str, Any]],
    config: Dict[str, Any] | None,
    cross_claim_fuse: bool,
    diagnostics: Dict[str, Any] | None,
) -> None:
    """If config enables it, run SpanRewriter on each eligible fused span
    and replace ``repair["replacement"]`` with the LLM's rewrite. Any
    failure leaves the span untouched so the Step-1 corrected_claim_text
    remains as the fallback."""
    env_mode = os.environ.get("AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE", "off")
    config_mode = (config or {}).get("mode") if isinstance(config, dict) else None
    mode = _normalize_rewrite_mode(config_mode if config_mode else env_mode)
    if diagnostics is not None:
        diagnostics["multi_anchor_llm_rewrite_mode"] = mode
    if mode == "off" or not merged or not concepts:
        if diagnostics is not None and mode != "off":
            diagnostics["multi_anchor_llm_rewrite_attempts"] = 0
        return

    model = (config or {}).get("model") if isinstance(config, dict) else None
    api_key = (config or {}).get("api_key") if isinstance(config, dict) else None
    base_url = (config or {}).get("base_url") if isinstance(config, dict) else None
    question_text = (config or {}).get("question_text", "") if isinstance(config, dict) else ""
    target_turn_number = (config or {}).get("target_turn_number") if isinstance(config, dict) else None
    timeout_sec = float((config or {}).get("timeout_sec") or os.environ.get("AGDEBUGGER_REWRITE_TIMEOUT_SEC", "30"))

    attempts = 0
    successes = 0
    per_span_outcomes: List[str] = []
    # Bounded sample of the text the LLM produced when the leak guard rejected
    # it. Without this we know leak_guard fired but cannot see *what* the LLM
    # actually wrote — the fastest way to debug prompt-side failures is to
    # read a few rejected samples directly from the run log.
    leak_samples: List[str] = []
    leak_sample_limit = max(1, int(os.environ.get("AGDEBUGGER_LLM_REWRITE_LEAK_SAMPLES", "3")))
    # Must be large enough to catch the triggering token in practice. Real
    # rewrites are ~400–800 chars; 300 was too short and 8/13 captured leaks
    # had `(none — triggered beyond captured window)` in the case study.
    leak_sample_char_limit = max(200, int(os.environ.get("AGDEBUGGER_LLM_REWRITE_LEAK_CHARS", "800")))
    if not model:
        if diagnostics is not None:
            diagnostics["multi_anchor_llm_rewrite_attempts"] = 0
            diagnostics["multi_anchor_llm_rewrite_skipped_reason"] = "no_model_configured"
        return

    try:
        from external_agent.llm import OpenAICompatibleLLM
        from external_agent.rewriter import SpanRewriter, run_rewrite_sync
    except Exception as exc:  # noqa: BLE001
        if diagnostics is not None:
            diagnostics["multi_anchor_llm_rewrite_attempts"] = 0
            diagnostics["multi_anchor_llm_rewrite_skipped_reason"] = f"import_error:{type(exc).__name__}"
        return

    # Build claim_id → concept lookup once; only concepts that ended up in
    # a fused span will be passed to the rewriter.
    concept_by_id: Dict[Any, Dict[str, Any]] = {}
    for concept in concepts:
        if not isinstance(concept, dict):
            continue
        cid = concept.get("claim_id")
        if cid is not None:
            concept_by_id.setdefault(cid, concept)

    rewriter = None
    for span in merged:
        if not _rewrite_span_eligible(span, mode=mode, cross_claim_fuse=cross_claim_fuse):
            per_span_outcomes.append("not_eligible")
            continue
        fused_ids = [cid for cid in (span.get("fused_claim_ids") or []) if cid is not None]
        contributing_claims = []
        seen_ids: set = set()
        for cid in fused_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            concept = concept_by_id.get(cid)
            if concept is not None:
                contributing_claims.append(concept)
        if not contributing_claims:
            # No known concepts — fall back to the primary span concept.
            if concepts:
                contributing_claims = [concepts[0]]
        original_span_content = original_content[span["start"]: span["end"]]
        prefix_context = original_content[max(0, span["start"] - 800): span["start"]]
        suffix_context = original_content[span["end"]: span["end"] + 400]
        if rewriter is None:
            try:
                llm = OpenAICompatibleLLM(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.0,
                    timeout_sec=timeout_sec,
                )
                rewriter = SpanRewriter(llm)
            except Exception as exc:  # noqa: BLE001
                if diagnostics is not None:
                    diagnostics["multi_anchor_llm_rewrite_skipped_reason"] = f"llm_init_error:{type(exc).__name__}"
                return
        attempts += 1
        result = run_rewrite_sync(
            rewriter,
            question_text=str(question_text or ""),
            target_turn_number=target_turn_number if isinstance(target_turn_number, int) else None,
            original_span_content=original_span_content,
            prefix_context=prefix_context,
            suffix_context=suffix_context,
            contributing_claims=contributing_claims,
            timeout_sec=timeout_sec,
        )
        if result.rewritten_text and result.fallback_reason == "ok":
            # Preserve verbatim corrected_claim_text so the downstream validator
            # (`corrected_claim_text in replacement_text`) still passes even when
            # the LLM rewrote the primary sentence into a first-person paraphrase.
            primary_corrected = ""
            if contributing_claims:
                primary_corrected = str(
                    contributing_claims[0].get("corrected_claim_text")
                    or contributing_claims[0].get("correct_understanding")
                    or ""
                ).strip()
            if primary_corrected and primary_corrected not in result.rewritten_text:
                span["replacement"] = primary_corrected + "\n\n" + result.rewritten_text
                span["llm_rewritten_verbatim_prefixed"] = True
            else:
                span["replacement"] = result.rewritten_text
            span["llm_rewritten"] = True
            successes += 1
            per_span_outcomes.append("ok")
        else:
            per_span_outcomes.append(result.fallback_reason or "unknown")
            # Capture a bounded sample of leak_guard rejections so a
            # post-run case study can see what word tripped the guard.
            if (
                result.fallback_reason == "leak_guard"
                and result.rejected_text
                and len(leak_samples) < leak_sample_limit
            ):
                sample = result.rejected_text.strip()
                if len(sample) > leak_sample_char_limit:
                    sample = sample[:leak_sample_char_limit] + "…"
                leak_samples.append(sample)

    if diagnostics is not None:
        diagnostics["multi_anchor_llm_rewrite_attempts"] = attempts
        diagnostics["multi_anchor_llm_rewrite_successes"] = successes
        diagnostics["multi_anchor_llm_rewrite_outcomes"] = per_span_outcomes
        if leak_samples:
            diagnostics["multi_anchor_llm_rewrite_leak_samples"] = leak_samples


def _build_precise_replacement_text(
    original_content: str,
    concepts: List[Dict[str, Any]],
    *,
    planner_span: Dict[str, Any] | None = None,
    diagnostics: Dict[str, Any] | None = None,
    llm_rewrite_config: Dict[str, Any] | None = None,
) -> tuple[str | None, bool]:
    # ---- Path C: multi-anchor collection (env-gated) ----
    # Default ``on``: each concept contributes ALL non-overlapping anchor-hit
    # spans rather than just the first hit. Subsumed variants are dropped,
    # then adjacent same-claim spans are fused within
    # ``AGDEBUGGER_MULTI_ANCHOR_FUSE_GAP``. Cross-claim fusion is opt-in.
    # ``off`` restores the historical break-on-first-hit behaviour for
    # easy A/B comparison.
    multi_anchor_mode = os.environ.get("AGDEBUGGER_MULTI_ANCHOR_MODE", "on").strip().lower()
    multi_anchor_on = multi_anchor_mode not in {"0", "off", "false", "no"}
    multi_anchor_min_len = max(1, int(os.environ.get("AGDEBUGGER_MULTI_ANCHOR_MIN_LEN", "24")))
    fuse_gap = max(0, int(os.environ.get("AGDEBUGGER_MULTI_ANCHOR_FUSE_GAP", "300")))
    cross_claim_fuse = os.environ.get("AGDEBUGGER_MULTI_ANCHOR_CROSS_CLAIM_FUSE", "0").strip().lower() in {"1", "true", "yes", "on"}

    repairs: List[Dict[str, Any]] = []
    per_claim_hit_count: Dict[str, int] = {}
    for concept in concepts:
        corrected_claim_text = str(concept.get("corrected_claim_text") or concept.get("correct_understanding") or "").strip()
        if not corrected_claim_text:
            continue
        claim_id = concept.get("claim_id")
        hits: List[tuple[int, int]] = []
        for anchor in _concept_anchor_candidates(concept):
            if multi_anchor_on and len(anchor) >= multi_anchor_min_len:
                hits.extend(_locate_all_anchor_occurrences(original_content, anchor))
            else:
                located = _locate_anchor_in_content(original_content, anchor)
                if located is not None:
                    hits.append(located)
                    if not multi_anchor_on:
                        break
        if not hits:
            continue
        # Collapse duplicate hits that the same anchor or its variants
        # produced at the same offset.
        unique_hits = sorted({(s, e) for s, e in hits})
        for start_idx, end_idx in unique_hits:
            repairs.append(
                {
                    "start": start_idx,
                    "end": end_idx,
                    "claim_id": claim_id,
                    "replacement": corrected_claim_text,
                }
            )
        per_claim_hit_count[str(claim_id) if claim_id is not None else "<none>"] = len(unique_hits)

    if not repairs:
        if diagnostics is not None:
            diagnostics.setdefault("multi_anchor_mode_effective", "on" if multi_anchor_on else "off")
            diagnostics["multi_anchor_hit_count_per_claim"] = {}
            diagnostics["multi_anchor_fused_span_count"] = 0
            diagnostics["multi_anchor_cross_claim_fuse"] = cross_claim_fuse
        return None, True

    # Stage 1: drop anchor-variant duplicates that are strictly contained
    # in a longer hit at the same offset.
    repairs = _dedup_subsumed_spans(repairs)
    repairs.sort(key=lambda item: (item["start"], item["end"]))

    merged: List[Dict[str, Any]] = []
    for repair in repairs:
        if not merged or repair["start"] >= merged[-1]["end"]:
            merged.append(dict(repair))
            continue
        # overlapping spans: keep the first-selected repair span stable and extend the range only
        merged[-1]["end"] = max(merged[-1]["end"], repair["end"])

    # Stage 2: fuse adjacent spans whose gap ≤ fuse_gap. With path C enabled,
    # this is the mechanism that eats the prefix's wrong-argument chain —
    # two anchor hits that bracket an option enumeration block merge into
    # one span, and the enumeration between them gets nuked along with the
    # anchors. Disabled (gap=0) when multi_anchor_on is False.
    fused_gap_max = 0
    if multi_anchor_on and len(merged) >= 2 and fuse_gap > 0:
        pre_fuse_count = len(merged)
        # Track max fused gap for diagnostics before fusing destructively.
        for i in range(1, len(merged)):
            g = merged[i]["start"] - merged[i - 1]["end"]
            if g > fused_gap_max and g <= fuse_gap:
                # Only consider gaps that will actually get fused.
                same_claim = merged[i].get("claim_id") == merged[i - 1].get("claim_id")
                if cross_claim_fuse or same_claim:
                    fused_gap_max = g
        merged = _fuse_close_spans(merged, max_gap=fuse_gap, cross_claim=cross_claim_fuse)
        fused_span_reduction = pre_fuse_count - len(merged)
    else:
        fused_span_reduction = 0

    if diagnostics is not None:
        diagnostics.setdefault("multi_anchor_mode_effective", "on" if multi_anchor_on else "off")
        diagnostics["multi_anchor_hit_count_per_claim"] = per_claim_hit_count
        diagnostics["multi_anchor_fused_span_count"] = len(merged)
        diagnostics["multi_anchor_fused_gap_max"] = fused_gap_max
        diagnostics["multi_anchor_fused_span_reduction"] = fused_span_reduction
        diagnostics["multi_anchor_cross_claim_fuse"] = cross_claim_fuse

    # ---- Step 2: optional LLM rewrite for fused spans ----
    # When enabled, replace the static primary-claim ``corrected_claim_text``
    # that currently sits in ``repair["replacement"]`` with an LLM-generated
    # first-person reflection tailored to the full fused span + all
    # contributing claims. This addresses cases where multi-anchor fusion
    # correctly cut the wrong argument prefix but the agent still pivots to
    # the same answer because the injected corrected_text is too short or
    # too generic to shift the choice. Default OFF — any failure (timeout,
    # invalid JSON, leak) falls back to the Step-1 corrected_text, so the
    # rewrite can never break the repair pipeline.
    _maybe_apply_llm_rewrite(
        merged,
        original_content=original_content,
        concepts=concepts,
        config=llm_rewrite_config,
        cross_claim_fuse=cross_claim_fuse,
        diagnostics=diagnostics,
    )

    truncate_after_repair = os.environ.get("AGDEBUGGER_TRUNCATE_AFTER_REPAIR", "1").strip().lower() not in {"0", "false", "no", "off"}
    # Path C default: drop_wrong_prefix is off. Path C's multi-anchor fusion
    # replaces what this regex-based cut used to do, and the regex was
    # heuristic/language-specific. Kept as an explicit opt-in fallback for
    # runs that still need the old behaviour (e.g. when extraction only
    # yields a single claim and fusion has nothing to collapse).
    drop_wrong_prefix = os.environ.get("AGDEBUGGER_DROP_WRONG_PREFIX", "0").strip().lower() not in {"0", "false", "no", "off"}

    # Track whether any span was rewritten by the LLM. When true the legacy
    # `_build_reflection_block` append is suppressed, because the LLM rewrite
    # already includes a first-person self-correction — stacking a second
    # (templated) reflection on top just dilutes the signal and the rerun
    # frequently pivots back to the original wrong answer (idx=? case study).
    any_llm_rewritten = any(bool(span.get("llm_rewritten")) for span in merged)

    parts: List[str] = []
    cursor = 0
    for idx, repair in enumerate(merged):
        prefix_segment = original_content[cursor:repair["start"]] if repair["start"] > cursor else ""
        if prefix_segment:
            # Cat A fix: when the prefix contains the agent's wrong reasoning
            # (e.g. a full enumeration of all options arguing for the wrong
            # answer), keeping it intact means the agent re-reads its own
            # case and outputs the same wrong answer. Cut at the first
            # conclusion marker — but only if one is actually present, so
            # short prefixes that hold only factual setup are kept verbatim.
            if drop_wrong_prefix and idx == 0:
                # Prefer the planner-supplied span if it resolves to a valid
                # cut inside the prefix segment. The planner has full context
                # and can identify option-enumeration / "Let me evaluate" /
                # similar analysis-start markers that the regex does not
                # cover. Fall back to the regex for pre-LLM-decided runs.
                planner_cut_abs = _resolve_planner_span_cut(original_content, planner_span)
                if (
                    planner_cut_abs is not None
                    and planner_cut_abs >= repair["start"] - len(prefix_segment)
                    and planner_cut_abs <= repair["start"]
                ):
                    # Translate absolute offset into prefix_segment coords.
                    rel_cut = planner_cut_abs - (repair["start"] - len(prefix_segment))
                    parts.append(prefix_segment[:rel_cut].rstrip())
                else:
                    marker = _CONCLUSION_MARKER_RE.search(prefix_segment)
                    if marker is not None and marker.start() >= 1:
                        parts.append(prefix_segment[: marker.start()].rstrip())
                    else:
                        parts.append(prefix_segment)
            else:
                parts.append(prefix_segment)
        parts.append(repair["replacement"])
        cursor = repair["end"]
        # In truncate mode, discard everything after the last repair anchor.
        # The old text after the anchor is the reasoning chain that led to the
        # wrong answer — keeping it anchors the agent back to the same conclusion.
        if truncate_after_repair and idx == len(merged) - 1:
            break
    if not truncate_after_repair:
        parts.append(original_content[cursor:])

    replacement_text = "".join(parts).rstrip()
    # Strip trailing system notes and answer placeholders that would prevent
    # the agent from re-deriving its own answer during rerun.
    replacement_text = re.sub(
        r"\[SYSTEM NOTE\].*$",
        "",
        replacement_text,
        flags=re.DOTALL,
    ).rstrip()
    # Strip ALL <answer>...</answer> spans (not just trailing ones), so any
    # answer text embedded in the middle of the rebuilt assistant message
    # cannot anchor the rerun back to the original wrong answer.
    replacement_text = re.sub(
        r"<answer>.*?</answer>",
        "",
        replacement_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Same treatment for TERMINATE markers wherever they appear.
    replacement_text = re.sub(
        r"\bTERMINATE\b",
        "",
        replacement_text,
    )
    # Strip dangling answer-selection sentences left behind after option/answer
    # redaction, e.g. "Therefore, the correct answer is:" with nothing after it.
    replacement_text = re.sub(
        r"(?:Therefore|Thus|Hence|So|In conclusion|Based on[^.\n]{0,60}),?\s*"
        r"(?:the\s+)?(?:correct|best|final|most appropriate|most likely)\s+"
        r"(?:answer|option|choice)\s+(?:is|would be)\s*:?\s*$",
        "",
        replacement_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Collapse the excess whitespace left behind by the deletions.
    replacement_text = re.sub(r"[ \t]+\n", "\n", replacement_text)
    replacement_text = re.sub(r"\n{3,}", "\n\n", replacement_text).rstrip()
    # PR-3: append a first-person self-reflection block in the agent's own
    # voice instead of the legacy third-person ``[REASONING REPAIR]``
    # directive. The directive reads like meta commentary and the rerun
    # frequently ignores it (see idx=71 in comp1/w1 — six identical edits,
    # answer pinned at optionA throughout). A "Wait — let me re-check…"
    # reflection slots in as if the agent authored it and is much more
    # likely to push the next turn toward a different option.
    #
    # CRITICAL: this string MUST NOT contain the literal team-termination
    # keyword (``TERMINATE``) or the literal ``<answer>`` tag — the team's
    # ``TextMentionTermination`` matches on ``TERMINATE`` and, if the
    # injected message contains it, the team stops *immediately* before
    # the agent can re-answer. The leak guard at the end asserts this.
    append_directive = os.environ.get("AGDEBUGGER_APPEND_REPAIR_DIRECTIVE", "1").strip().lower() not in {"0", "false", "no", "off"}
    # Suppress the legacy reflection block when the LLM already produced a
    # tailored first-person rewrite for at least one span. Without this guard
    # the replacement text ends up with three stacked "Wait — let me re-check"
    # segments (LLM rewrite + templated reflection + closing reminder), which
    # dilutes the correction and pulls the agent back to its prior answer.
    # Overridable via AGDEBUGGER_FORCE_REFLECTION_ON_REWRITE for A/B tests.
    force_reflection_on_rewrite = os.environ.get(
        "AGDEBUGGER_FORCE_REFLECTION_ON_REWRITE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    suppress_legacy_reflection = any_llm_rewritten and not force_reflection_on_rewrite
    if diagnostics is not None:
        diagnostics["legacy_reflection_suppressed"] = bool(suppress_legacy_reflection)
    if append_directive and not suppress_legacy_reflection:
        if _reflection_enabled_for_strict_mode() and concepts:
            reflection = _build_reflection_block(concepts[0])
            if reflection:
                replacement_text = replacement_text + "\n\n" + reflection
            # Add a short closing reminder that uses the agent's standard
            # final-answer format without dictating which option to pick.
            replacement_text = replacement_text + (
                "\n\nOnce I have reconsidered each option with this corrected "
                "premise in mind, I will commit to my final choice in my "
                "standard final-answer format."
            )
        else:
            directive = (
                "\n\n[REASONING REPAIR] The earlier inference above is unsupported "
                "by the available evidence. Re-examine each option from the "
                "question's option list with this corrected premise; do not assume "
                "the previous selection. Then commit to your final choice in your "
                "standard final-answer format."
            )
            replacement_text = replacement_text + directive
    elif append_directive and suppress_legacy_reflection:
        # Still append the short closing reminder so the agent knows to
        # re-commit to a final answer — without the templated reflection.
        replacement_text = replacement_text + (
            "\n\nOnce I have reconsidered each option with this corrected "
            "premise in mind, I will commit to my final choice in my "
            "standard final-answer format."
        )

    # PR-3 leak guard: if anything upstream slipped an option label / answer
    # tag / TERMINATE marker past the sanitizers, fail loudly so the caller
    # can fall back to a different claim instead of silently pinning the
    # rerun to a wrong answer. Applied in strict mode only — legacy mode
    # intentionally auto-injects <answer>/TERMINATE via the controller.
    if os.environ.get("AGDEBUGGER_STRICT_CONCEPT_REPAIR_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            _assert_replacement_has_no_direct_answer(replacement_text)
        except InvalidRepairText as leak:
            print(f"      [debug] replacement_text leak guard tripped: {leak}")
            return None, True
    return replacement_text, False


_CONCLUSION_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Therefore|Thus|Hence|So|In conclusion|Based on (?:this|the above|my|our|the) (?:analysis|reasoning|evidence|information))"
    r"[,;:\s]"
    r"|(?:The (?:correct|best|final|most (?:appropriate|likely|plausible)) (?:answer|option|choice))"
    r"|(?:(?:My|Our|The) (?:final )?answer)"
    r"|(?:After (?:careful )?(?:analysis|consideration|review))"
    r"|(?:Given (?:all )?(?:the|this) (?:above|evidence|information|analysis))"
    r")",
    re.IGNORECASE,
)


def _content_prefix_before_conclusion(content: str) -> str:
    """Return the portion of *content* before the first conclusion/answer-selection
    marker, preserving background, evidence, and intermediate reasoning.

    Used as a fallback when anchor-based repair cannot locate the faulty span:
    the prefix provides useful context while the conclusion (which contains the
    wrong answer derivation) is replaced by ``corrected_claim_text``.

    If no conclusion marker is found, returns roughly the first 60% of the
    content (cut at a paragraph or sentence boundary).
    """
    if not content:
        return ""

    match = _CONCLUSION_MARKER_RE.search(content)
    if match:
        cut = match.start()
        # Don't cut too early — keep at least 20% of the content as prefix
        if cut >= len(content) * 0.2:
            return content[:cut].rstrip()

    # No conclusion marker found — keep the first ~60% at a paragraph boundary
    target = int(len(content) * 0.6)
    # Try paragraph break
    para_break = content.rfind("\n\n", 0, target + 200)
    if para_break > len(content) * 0.2:
        return content[:para_break].rstrip()
    # Try sentence break
    sentence_break = max(
        content.rfind(". ", 0, target + 100),
        content.rfind(".\n", 0, target + 100),
    )
    if sentence_break > len(content) * 0.2:
        return content[: sentence_break + 1].rstrip()
    # Last resort — hard cut
    return content[:target].rstrip()


_SYCOPHANT_PARAGRAPH_RE = re.compile(
    r"(?im)^[^\n]*?\b(?:"
    r"i apologize"
    r"|i'?m sorry"
    r"|let me re-?evaluate"
    r"|let me reconsider"
    r"|you'?re absolutely right"
    r"|you'?re right to question"
    r"|on second thought"
    r"|looking at this again"
    r"|i was (?:wrong|incorrect|mistaken)"
    r"|my previous (?:answer|response|conclusion|analysis|reasoning)"
    r"|thank you for pointing"
    r")\b[^\n]*$"
)


def _strip_sycophant_paragraphs(text: str) -> str:
    """Remove paragraphs that begin with sycophantic / apologetic phrasing.

    Cat D fix: when the planner picks an agent message that opens with
    ``"I apologize, but I'm having trouble..."`` and that whole message
    becomes the replacement_text, the agent reads its own apology back
    and randomly flips answers. Drop any paragraph that contains such
    phrasing — but keep the rest of the message intact.
    """
    if not text:
        return text
    paragraphs = re.split(r"\n{2,}", text)
    kept = []
    for para in paragraphs:
        if _SYCOPHANT_PARAGRAPH_RE.search(para):
            continue
        kept.append(para)
    cleaned = "\n\n".join(kept).strip()
    return cleaned if cleaned else text  # never empty out the whole message


def _augment_rewrite_config(
    base: Dict[str, Any] | None,
    *,
    target_turn_number: Any = None,
) -> Dict[str, Any] | None:
    """Merge per-call values (target_turn_number) into the rewrite config.

    The debug_question loop builds a single ``llm_rewrite_config`` up front
    with ``mode / model / api_key / base_url / question_text / timeout_sec``,
    and each repair-action helper adds the per-action ``target_turn_number``
    here before handing the combined dict to ``_build_precise_replacement_text``.
    Returns ``None`` if ``base`` is missing or its mode resolves to ``off``.
    """
    if not isinstance(base, dict):
        return None
    mode = _normalize_rewrite_mode(base.get("mode"))
    if mode == "off":
        return None
    merged = dict(base)
    merged["mode"] = mode
    if isinstance(target_turn_number, int):
        merged["target_turn_number"] = target_turn_number
    return merged


def _complete_claim_repair_action(
    action: Dict[str, Any],
    *,
    snapshot: Dict[str, Any],
    analysis_context: Dict[str, Any] | None,
    force_synthesize_replacement: bool = False,
    llm_rewrite_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized = dict(action)
    if str(normalized.get("action", "")).strip() != "edit_and_revert":
        return normalized
    if force_synthesize_replacement:
        normalized.pop("replacement_text", None)
        normalized.pop("body", None)
        normalized.pop("message", None)
    if normalized.get("replacement_text"):
        return normalized
    claim_id = normalized.get("claim_id")
    concepts = _hallucinated_concepts(analysis_context)
    if not claim_id or not concepts:
        return normalized
    concept = next((item for item in concepts if item.get("claim_id") == claim_id), None)
    if concept is None:
        return normalized

    current_messages = _session_messages(snapshot.get("session_history", {}))
    assistant_messages = _assistant_messages(current_messages)
    if not assistant_messages:
        return normalized
    target_message = _target_message_for_concept(assistant_messages, concept)
    if target_message is None:
        return normalized
    if normalized.get("target_turn") is None:
        target_turn = _assistant_turn_number_for_message(current_messages, target_message)
        if target_turn is not None:
            normalized["target_turn"] = target_turn
    if normalized.get("timestamp") is None and isinstance(target_message.get("timestamp"), int):
        normalized["timestamp"] = int(target_message["timestamp"])

    original_content = _message_content(target_message)
    # Cat D fix: if the agent message we're patching opens with apologetic
    # / sycophantic phrasing, drop those paragraphs before the precise
    # replacement step so they cannot leak back into the trajectory.
    cleaned_original = _strip_sycophant_paragraphs(original_content)
    planner_span = normalized.get("wrong_reasoning_span")
    if not isinstance(planner_span, dict):
        planner_span = None
    repair_diagnostics: Dict[str, Any] = {}
    effective_rewrite_cfg = _augment_rewrite_config(
        llm_rewrite_config, target_turn_number=normalized.get("target_turn")
    )
    replacement_text, anchor_not_found = _build_precise_replacement_text(
        cleaned_original,
        [concept],
        planner_span=planner_span,
        diagnostics=repair_diagnostics,
        llm_rewrite_config=effective_rewrite_cfg,
    )
    if replacement_text is None:
        corrected = str(concept.get("corrected_claim_text") or concept.get("correct_understanding") or "").strip()
        prefix = _content_prefix_before_conclusion(cleaned_original)
        if prefix:
            replacement_text = prefix.rstrip() + "\n\n" + corrected
        else:
            replacement_text = corrected
        anchor_not_found = True
    normalized["replacement_text"] = replacement_text
    normalized["anchor_not_found"] = anchor_not_found
    normalized["critic_followup"] = {
        "concept_name": concept.get("concept_name", ""),
        "incorrect_understanding": concept.get("incorrect_understanding", ""),
        "correct_understanding": concept.get("correct_understanding", ""),
    }
    for key, value in repair_diagnostics.items():
        # Promote multi-anchor diagnostics into the action so they land in
        # ``debug_step.action`` for post-run case study.
        normalized[key] = value
    return normalized


def _sanitize_planner_concept_repair_action(
    action: Dict[str, Any],
    *,
    snapshot: Dict[str, Any],
    analysis_context: Dict[str, Any] | None,
    strict_concept_repair_only: bool,
    llm_rewrite_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not strict_concept_repair_only:
        return action
    if str(action.get("action", "")).strip() != "edit_and_revert":
        return action
    sanitized = dict(action)
    sanitized.pop("replacement_text", None)
    sanitized.pop("body", None)
    sanitized.pop("message", None)
    return _complete_claim_repair_action(
        sanitized,
        snapshot=snapshot,
        analysis_context=analysis_context,
        force_synthesize_replacement=True,
        llm_rewrite_config=llm_rewrite_config,
    )


def _build_deterministic_concept_repair_action(
    snapshot: Dict[str, Any],
    analysis_context: Dict[str, Any] | None,
    *,
    excluded_claim_ids: set[str] | None = None,
    llm_rewrite_config: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    concepts = _hallucinated_concepts(analysis_context)
    if excluded_claim_ids:
        concepts = [item for item in concepts if item.get("claim_id") not in excluded_claim_ids]
    if not concepts:
        return None

    current_messages = _session_messages(snapshot.get("session_history", {}))
    assistant_messages = _assistant_messages(current_messages)
    if not assistant_messages:
        return None

    def _turn_order(item: Dict[str, Any]) -> float:
        turn = item.get("turn_number")
        return float(turn) if isinstance(turn, (int, float)) else float("inf")

    concept = min((item for item in concepts if isinstance(item, dict)), key=_turn_order, default=None)
    if concept is None:
        return None

    primary_claim_id = concept.get("claim_id")
    target_message = _target_message_for_concept(assistant_messages, concept)
    if target_message is None:
        return None

    target_turn = _assistant_turn_number_for_message(current_messages, target_message)
    target_timestamp = target_message.get("timestamp")
    if target_turn is None and not isinstance(target_timestamp, int):
        return None

    original_content = _message_content(target_message)
    target_concepts = []
    for item in concepts:
        if not isinstance(item, dict):
            continue
        if primary_claim_id is not None and item.get("claim_id") != primary_claim_id:
            continue
        same_timestamp = (
            isinstance(item.get("source_timestamp"), int)
            and isinstance(target_timestamp, int)
            and int(item["source_timestamp"]) == int(target_timestamp)
        )
        anchored_here = any(
            _locate_anchor_in_content(original_content, anchor) is not None
            for anchor in _concept_anchor_candidates(item)
        )
        if same_timestamp or anchored_here:
            target_concepts.append(item)
    if not target_concepts:
        target_concepts = [concept]

    repair_diagnostics: Dict[str, Any] = {}
    effective_rewrite_cfg = _augment_rewrite_config(
        llm_rewrite_config, target_turn_number=target_turn
    )
    replacement_text, anchor_not_found = _build_precise_replacement_text(
        original_content,
        target_concepts,
        diagnostics=repair_diagnostics,
        llm_rewrite_config=effective_rewrite_cfg,
    )
    if replacement_text is None:
        replacement_text = str(concept.get("corrected_claim_text") or concept.get("correct_understanding") or "").strip()

    action = {
        "action": "edit_and_revert",
        "claim_id": primary_claim_id,
        "target_turn": target_turn,
        "timestamp": int(target_timestamp) if isinstance(target_timestamp, int) else None,
        "replacement_text": replacement_text,
        "anchor_not_found": anchor_not_found,
    }
    primary_concept = target_concepts[0] if target_concepts else concept
    if isinstance(primary_concept, dict):
        action["critic_followup"] = {
            "concept_name": primary_concept.get("concept_name", ""),
            "incorrect_understanding": primary_concept.get("incorrect_understanding", ""),
            "correct_understanding": primary_concept.get("correct_understanding", ""),
        }
    action.update(repair_diagnostics)
    return action

def _run_analysis_with_timeout(
    *,
    task: str,
    state: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: str,
    evidence_text: str,
    use_websearch: bool,
    search_backend: str,
    search_max_searches: int,
    search_num_results: int,
    search_fetch_top_n: int,
    search_max_output_words: int,
    assistant_only: bool,
    timeout_sec: float,
    question_text: str = "",
    current_answer: str = "",
) -> Dict[str, Any]:
    async def _analyze(use_web: bool) -> Dict[str, Any]:
        return await analyze_session_state(
            task=task,
            state=state,
            model=model,
            api_key=api_key,
            base_url=base_url,
            evidence_text=evidence_text,
            use_websearch=use_web,
            search_backend=search_backend,
            search_max_searches=search_max_searches,
            search_num_results=search_num_results,
            search_fetch_top_n=search_fetch_top_n,
            search_max_output_words=search_max_output_words,
            assistant_only=assistant_only,
            timeout_sec=timeout_sec,
            question_text=question_text,
            current_answer=current_answer,
        )

    try:
        result = asyncio.run(_analyze(use_websearch))
        result["analysis_timeout_sec"] = timeout_sec
        result["analysis_use_websearch_effective"] = use_websearch
        return result
    except Exception as primary_error:
        if not use_websearch:
            raise
        fallback = asyncio.run(_analyze(False))
        fallback["analysis_timeout_sec"] = timeout_sec
        fallback["analysis_use_websearch_effective"] = False
        fallback["analysis_fallback_reason"] = f"{type(primary_error).__name__}: {primary_error}"
        return fallback


class ServerHandle:
    def __init__(
        self,
        process: subprocess.Popen[str],
        log_file_handle,
        *,
        cmd: List[str],
        log_path: Path,
    ):
        self.process = process
        self.log_file_handle = log_file_handle
        self.cmd = cmd
        self.log_path = log_path

    def read_log_tail(self, max_lines: int = 80) -> str:
        self.log_file_handle.flush()
        if not self.log_path.exists():
            return "(server log file does not exist)"

        text = self.log_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return "(server log is empty)"

        lines = text.splitlines()
        return "\n".join(lines[-max_lines:])

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log_file_handle.close()


class QuestionPhaseTimeout(TimeoutError):
    def __init__(self, message: str, *, messages: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.messages = messages or []


class QuestionPhaseRuntimeError(RuntimeError):
    def __init__(self, message: str, *, messages: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.messages = messages or []


class EmptyAgentTextMessageError(QuestionPhaseRuntimeError):
    pass


def start_agdebugger_background(
    repo_dir: Path,
    module_expr: str,
    host: str,
    port: int,
    log_path: Path,
) -> ServerHandle:
    env = os.environ.copy()
    env["AGDEBUGGER_BACKEND_SERVE_UI"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    src_path = str(repo_dir / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "agdebugger.cli",
        module_expr,
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ServerHandle(proc, log_fh, cmd=cmd, log_path=log_path)


def _backend_failure_message(
    handle: Optional[ServerHandle],
    *,
    reason: str,
    last_error: str = "",
    log_lines: int = 80,
) -> str:
    parts = [reason]
    if handle is not None:
        exit_code = handle.process.poll()
        parts.append(f"backend_pid={handle.process.pid}")
        parts.append(f"backend_exit_code={exit_code}")
        parts.append(f"backend_cmd={' '.join(handle.cmd)}")
        parts.append(f"backend_log={handle.log_path}")
        parts.append(f"backend_log_tail:\n{handle.read_log_tail(max_lines=log_lines)}")
    if last_error:
        parts.append(f"last_client_error={last_error}")
    return "\n".join(parts)


def wait_backend_ready(client: AGDebuggerClient, handle: Optional[ServerHandle], timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_err = ""

    # Phase 1: wait for HTTP endpoints to respond.
    while time.time() < deadline:
        if handle is not None and handle.process.poll() is not None:
            time.sleep(0.2)
            raise RuntimeError(
                _backend_failure_message(
                    handle,
                    reason="AGDebugger backend process exited before becoming ready.",
                    last_error=last_err,
                )
            )
        try:
            _ = client.get_topics()
            break
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(0.5)
    else:
        raise TimeoutError(
            _backend_failure_message(
                handle,
                reason="Timed out waiting for AGDebugger backend.",
                last_error=last_err,
            )
        )

    # Phase 2: probe that the agent runtime can actually accept a message.
    # The MCP sub-server may still be initialising after Uvicorn is up,
    # which blocks send() until it finishes.
    topics = client.get_topics()
    probe_recipient = _guess_manager_topic(topics)
    if probe_recipient:
        print(f"[wait_backend_ready] probing agent runtime (send GroupChatReset to {probe_recipient}) ...")
        while time.time() < deadline:
            try:
                # Start the loop first so send() doesn't deadlock —
                # runtime.send_message awaits a future that is only
                # resolved when the runtime loop processes the message.
                client.start_loop()
                client.send(probe_recipient, {"type": "GroupChatReset"})
                client.wait_until_idle(timeout_sec=min(15, deadline - time.time()), warmup_sec=3.0)
                client.stop_loop()
                print("[wait_backend_ready] agent runtime ready.")
                return
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                print(f"[wait_backend_ready] agent runtime not ready yet: {e}")
                try:
                    client.stop_loop()
                except Exception:
                    pass
                time.sleep(2)
        raise TimeoutError(
            _backend_failure_message(
                handle,
                reason="Timed out waiting for agent runtime to accept messages.",
                last_error=last_err,
            )
        )


def _stop_loop_with_fallback(
    client: AGDebuggerClient,
    *,
    stage: str,
    graceful_timeout_sec: float = 30.0,
    force_timeout_sec: float = 10.0,
) -> None:
    try:
        client.stop_loop(timeout_sec=graceful_timeout_sec)
        return
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "Runtime is not started" in msg:
            return
        print(f"  [run_question] stop_loop during {stage} did not finish cleanly: {msg}")

    client.stop_loop(force=True, timeout_sec=force_timeout_sec)


def _wait_for_question_completion(
    client: AGDebuggerClient,
    *,
    after_ts: int,
    timeout_sec: float,
    poll_sec: float = 0.5,
    stall_timeout_sec: float = 20.0,
) -> List[Dict[str, Any]]:
    deadline = time.time() + timeout_sec
    last_new_messages: List[Dict[str, Any]] = []
    saw_activity = False
    last_progress_at = time.time()
    last_seen_ts = after_ts
    last_num_tasks: int | None = None

    while time.time() < deadline:
        state = client.get_session_history()
        current_messages = _session_messages(state)
        new_messages = _messages_after_timestamp(current_messages, after_ts)
        if new_messages:
            last_new_messages = new_messages
            saw_activity = True
            newest_ts = _last_timestamp(new_messages)
            if newest_ts > last_seen_ts:
                last_seen_ts = newest_ts
                last_progress_at = time.time()
            empty_agent_message = _find_empty_agent_text_message(new_messages)
            if empty_agent_message is not None:
                source = _message_source(empty_agent_message) or "assistant"
                timestamp = empty_agent_message.get("timestamp")
                location = f" at timestamp {timestamp}" if timestamp is not None else ""
                raise EmptyAgentTextMessageError(
                    f"Question phase aborted: received empty TextMessage from {source}{location}",
                    messages=new_messages,
                )
            if extract_answer_from_messages(new_messages) is not None:
                return new_messages
            if _has_termination_signal(new_messages):
                return new_messages

        num_tasks = client.get_num_tasks()
        if last_num_tasks is None or num_tasks != last_num_tasks:
            last_num_tasks = num_tasks
            last_progress_at = time.time()
        if num_tasks > 0:
            saw_activity = True

        # Treat a running event loop as a sign of activity — the agent may be
        # blocked on a long MCP tool call (e.g. web search) where neither new
        # messages nor task-count changes are visible.  Only declare a stall
        # when the loop itself has stopped making progress.
        loop_busy = False
        try:
            loop_busy = client.get_loop_status()
        except Exception:
            pass
        if loop_busy:
            last_progress_at = time.time()

        if saw_activity and (time.time() - last_progress_at) >= stall_timeout_sec:
            raise TimeoutError(
                "Question phase stalled: no new messages, no task-state changes, "
                f"and loop not running for {stall_timeout_sec:.1f}s (num_tasks={num_tasks})"
            )

        time.sleep(poll_sec)

    progress = "with partial trajectory" if saw_activity else "without any new trajectory"
    raise TimeoutError(f"Question phase timed out after {timeout_sec:.1f}s {progress}")


def _message_signature(messages: List[Dict[str, Any]]) -> str:
    summary = []
    for msg in messages:
        summary.append(
            {
                "timestamp": msg.get("timestamp"),
                "type": msg.get("type"),
                "source": _message_source(msg),
                "content": _message_content(msg),
            }
        )
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str)


def _wait_for_debug_settlement(
    client: AGDebuggerClient,
    *,
    before_messages: List[Dict[str, Any]],
    timeout_sec: float,
    poll_sec: float = 0.5,
    stall_timeout_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    deadline = time.time() + timeout_sec
    before_signature = _message_signature(before_messages)
    current_signature = before_signature
    last_progress_at = time.time()
    last_message_change_at = last_progress_at
    last_num_tasks: int | None = None
    saw_activity = False
    latest_messages = before_messages
    saw_completion_signal = False
    completion_observed_at: float | None = None
    idle_after_completion_sec = 2.0
    no_progress_deadline: float | None = None

    while time.time() < deadline:
        state = client.get_session_history()
        current_messages = _session_messages(state)
        signature = _message_signature(current_messages)
        if signature != current_signature:
            current_signature = signature
            latest_messages = current_messages
            saw_activity = True
            last_progress_at = time.time()
            last_message_change_at = last_progress_at
            if len(current_messages) > len(before_messages):
                no_progress_deadline = None
            elif no_progress_deadline is None:
                no_progress_deadline = time.time() + min(20.0, stall_timeout_sec)
            if (
                not saw_completion_signal
                and (
                    extract_answer_from_messages(current_messages) is not None
                    or _has_termination_signal(current_messages)
                )
            ):
                saw_completion_signal = True
                completion_observed_at = time.time()

        num_tasks = client.get_num_tasks()
        if last_num_tasks is None or num_tasks != last_num_tasks:
            last_num_tasks = num_tasks
            last_progress_at = time.time()
        if num_tasks > 0:
            saw_activity = True
        loop_busy = False
        try:
            loop_busy = client.get_loop_status()
        except Exception:
            pass
        if loop_busy:
            last_progress_at = time.time()

        if saw_completion_signal and not loop_busy and num_tasks == 0:
            if completion_observed_at is None:
                completion_observed_at = time.time()
            if (time.time() - completion_observed_at) >= idle_after_completion_sec:
                return latest_messages

        # Some runtime states keep loop_status=True even after the rerun has
        # already emitted a final answer/TERMINATE and drained the task queue.
        # Once the post-repair trajectory is stable for a short grace period,
        # let the caller exit and stop the loop in the normal cleanup path.
        if (
            saw_completion_signal
            and num_tasks == 0
            and (time.time() - last_message_change_at) >= idle_after_completion_sec
        ):
            return latest_messages

        if no_progress_deadline is not None and time.time() >= no_progress_deadline:
            raise TimeoutError("rerun_no_progress")

        if saw_activity and not loop_busy and num_tasks == 0 and (time.time() - last_progress_at) >= stall_timeout_sec:
            return latest_messages
        time.sleep(poll_sec)

    raise TimeoutError(f"Debug phase timed out after {timeout_sec:.1f}s without reaching a stable completion state")


def run_question(
    client: AGDebuggerClient,
    manager_topic: str,
    task: str,
    *,
    reset_timeout_sec: float,
    question_timeout_sec: float,
    question_stall_timeout_sec: float,
    question_retry_attempts: int = 1,
) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
    for attempt in range(question_retry_attempts + 1):
        if attempt > 0:
            print(
                "  [run_question] retrying after empty-trajectory stall "
                f"(attempt {attempt + 1}/{question_retry_attempts + 1}) ..."
            )

        # Reset before each question.  Use team_reset() which fans out
        # GroupChatReset to every participant (clearing model_context) and to
        # the manager, then drains the output queue.  This prevents the LLM
        # prompt context from accumulating messages across questions.
        try:
            print("  [run_question] checking loop status ...")
            if client.get_loop_status():
                print("  [run_question] stopping stale loop ...")
                client.stop_loop()
        except Exception:
            # A stale loop state should not block a fresh question from starting.
            pass

        print("  [run_question] starting loop for team reset ...")
        client.start_loop()
        print("  [run_question] performing team_reset ...")
        client.team_reset(timeout_sec=reset_timeout_sec)
        print(f"  [run_question] waiting for reset idle (timeout={reset_timeout_sec}s) ...")
        try:
            client.wait_until_idle(timeout_sec=reset_timeout_sec)
        except TimeoutError as exc:
            try:
                client.stop_loop(force=True, timeout_sec=10.0)
            except Exception:
                pass
            raise TimeoutError(f"Reset phase timed out after {reset_timeout_sec:.1f}s") from exc
        print("  [run_question] reset idle, stopping loop ...")
        _stop_loop_with_fallback(client, stage="reset")

        # Track only new messages for this question attempt.
        before_state = client.get_session_history()
        before_messages = _session_messages(before_state)
        before_ts = _last_timestamp(before_messages)

        start_body = {
            "type": "GroupChatStart",
            "messages": [{"type": "TextMessage", "source": "user", "content": task}],
        }
        print("  [run_question] starting loop for question ...")
        client.start_loop()
        print("  [run_question] sending GroupChatStart ...")
        client.send(manager_topic, start_body)
        print(f"  [run_question] waiting for question completion (timeout={question_timeout_sec}s) ...")
        try:
            new_messages = _wait_for_question_completion(
                client,
                after_ts=before_ts,
                timeout_sec=question_timeout_sec,
                stall_timeout_sec=question_stall_timeout_sec,
            )
        except TimeoutError as exc:
            try:
                client.stop_loop(force=True, timeout_sec=10.0)
            except Exception:
                pass
            after_state = client.get_session_history()
            after_messages = _session_messages(after_state)
            partial_messages = _messages_after_timestamp(after_messages, before_ts)
            timeout_error = QuestionPhaseTimeout(str(exc), messages=partial_messages)
            if not _has_meaningful_agent_activity(partial_messages) and attempt < question_retry_attempts:
                print(
                    "  [run_question] question phase produced no meaningful agent trajectory before timeout; "
                    "forcing reset and retrying once."
                )
                continue
            raise timeout_error from exc
        except QuestionPhaseRuntimeError:
            try:
                client.stop_loop(force=True, timeout_sec=10.0)
            except Exception:
                pass
            raise

        print("  [run_question] question complete, stopping loop ...")
        _stop_loop_with_fallback(client, stage="question")

        after_state = client.get_session_history()
        after_messages = _session_messages(after_state)
        new_messages = _messages_after_timestamp(after_messages, before_ts)
        if not new_messages:
            new_messages = new_messages or after_messages

        answer_raw = extract_answer_from_messages(new_messages)
        return new_messages, answer_raw, before_ts

    raise RuntimeError("run_question reached an unreachable retry state")


def run_llm_debug_loop(
    client: AGDebuggerClient,
    *,
    model: str,
    model_planner: str | None = None,
    model_claim: str | None = None,
    api_key: str,
    api_base: str,
    goal: str,
    expected_normalized: str,
    max_steps: int,
    settle_timeout_sec: float,
    num_options: int,
    log_fh=None,
    question_index: int = 0,
    claim_task: str | None = None,
    claim_evidence_text: str = "",
    claim_use_websearch: bool = False,
    claim_search_backend: str = "bright_data",
    claim_search_max_searches: int = 3,
    claim_search_num_results: int = 5,
    claim_search_fetch_top_n: int = 2,
    claim_search_max_output_words: int = 1500,
    max_edit_attempts: int = 3,
    max_concept_repair_attempts: int = 3,
    enable_deterministic_fallback: bool = True,
    strict_concept_repair_only: bool = False,
    question_baseline_ts: int = -1,
    initial_answer_raw: str | None = None,
) -> Tuple[bool, Optional[str]]:
    _planner_model = model_planner or model
    _claim_model = model_claim or model
    effective_claim_task = _default_debug_claim_task(claim_task)
    effective_claim_use_websearch = claim_use_websearch
    analysis_timeout_sec = float(os.environ.get("AGDEBUGGER_ANALYSIS_TIMEOUT_SEC", "90"))
    intern_api_key = os.environ.get("AGENTDEBUG_INTERN_API_KEY")
    planner_api_key = os.environ.get("AGENTDEBUG_OPENAI_API_KEY_PLANNER") or resolve_value_for_model(
        _planner_model,
        api_key,
        intern_value=intern_api_key,
    )
    claim_api_key = os.environ.get("AGENTDEBUG_OPENAI_API_KEY_CLAIM") or resolve_value_for_model(
        _claim_model,
        api_key,
        intern_value=intern_api_key,
    )
    planner_base_url = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL_PLANNER") or resolve_base_url_for_model(
        _planner_model,
        api_base,
    )
    claim_base_url = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL_CLAIM") or resolve_base_url_for_model(
        _claim_model,
        api_base,
    )
    planner = LLMPlanner(
        model=_planner_model,
        api_key=planner_api_key,
        base_url=planner_base_url,
        user_goal=goal,
        strict_concept_repair_only=strict_concept_repair_only,
    )
    # Multi-anchor LLM rewrite config: reuses the planner model endpoint. The
    # rewrite only runs when AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE is set to a
    # non-off mode; otherwise ``_build_precise_replacement_text`` ignores
    # this dict. Default mode="off" keeps the pipeline behaviour identical
    # to Step 1 unless the rjob script explicitly opts in.
    llm_rewrite_config: Dict[str, Any] = {
        "mode": os.environ.get("AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE", "off"),
        "model": _planner_model,
        "api_key": planner_api_key,
        "base_url": planner_base_url,
        "question_text": goal,
        "timeout_sec": float(os.environ.get("AGDEBUGGER_REWRITE_TIMEOUT_SEC", "30")),
    }
    # -- Save original trajectory & run one-time analysis --
    original_snapshot = _scope_snapshot_after_timestamp(client.snapshot(), question_baseline_ts)
    original_messages = _session_messages(original_snapshot.get("session_history", {}))
    original_analysis: Dict[str, Any] | None = None
    _log_analysis_event(
        log_fh,
        phase="original_start",
        question_index=question_index,
        step=None,
        model=_claim_model,
        timeout_sec=analysis_timeout_sec,
        use_websearch=effective_claim_use_websearch,
        state=original_snapshot.get("session_history", {}),
        status="start",
    )
    original_t0 = time.perf_counter()
    try:
        original_analysis = _run_analysis_with_timeout(
            task=effective_claim_task,
            state=original_snapshot["session_history"],
            model=_claim_model,
            api_key=claim_api_key,
            base_url=claim_base_url,
            evidence_text=claim_evidence_text,
            use_websearch=effective_claim_use_websearch,
            search_backend=claim_search_backend,
            search_max_searches=claim_search_max_searches,
            search_num_results=claim_search_num_results,
            search_fetch_top_n=claim_search_fetch_top_n,
            search_max_output_words=claim_search_max_output_words,
            assistant_only=True,
            timeout_sec=analysis_timeout_sec,
            question_text=goal,
            current_answer=initial_answer_raw or "",
        )
        original_repair = _build_concept_repair_context(original_analysis)
        if original_repair is not None:
            original_analysis = dict(original_analysis)
            original_analysis["concept_repair"] = original_repair
            original_analysis["analysis_task"] = effective_claim_task
            original_analysis["analysis_use_websearch"] = effective_claim_use_websearch
        _log_analysis_event(
            log_fh,
            phase="original_result",
            question_index=question_index,
            step=None,
            model=_claim_model,
            timeout_sec=analysis_timeout_sec,
            use_websearch=effective_claim_use_websearch,
            state=original_snapshot.get("session_history", {}),
            analysis=original_analysis,
            elapsed_sec=time.perf_counter() - original_t0,
            status="ok",
        )
    except Exception as e:  # noqa: BLE001
        original_analysis = {
            "analysis_error": _format_exception(e),
            "analysis_task": effective_claim_task,
            "analysis_use_websearch": effective_claim_use_websearch,
        }
        _log_analysis_event(
            log_fh,
            phase="original_result",
            question_index=question_index,
            step=None,
            model=_claim_model,
            timeout_sec=analysis_timeout_sec,
            use_websearch=effective_claim_use_websearch,
            state=original_snapshot.get("session_history", {}),
            analysis=original_analysis,
            elapsed_sec=time.perf_counter() - original_t0,
            status="error",
            error=original_analysis["analysis_error"],
        )

    prior_repair_signatures: List[str] = []
    failed_claim_ids: set[str] = set()
    concept_repair_attempt_count = 0
    last_action_error: str | None = None
    last_action_signature: str | None = None
    strict_stop_reason: str | None = None
    # PR-2: track the history of claim_ids the planner selected so we can
    # short-circuit when it keeps picking the same claim that previously
    # failed to change the answer. Without this, idx=71-style cases burn the
    # whole max_concept_repair_attempts budget replaying the same edit.
    planner_selected_history: List[str] = []
    # P0: track whether the wide-window analysis retry has already fired for
    # this question. It's a one-shot fallback — when the default single-turn
    # claim extraction finds nothing, we re-run analysis once with a larger
    # assistant-turn window so trajectories whose last turn is TERMINATE / "I
    # could not fetch..." can still surface a repairable concept. This path
    # does NOT touch enable_deterministic_fallback — if the retry also yields
    # zero concepts, we still halt with `no_repairable_concepts`.
    wide_window_retry_done: bool = False
    # Track the most recent <answer> observed during the debug loop so the
    # final ``debug_result`` record can report ``answer_before`` / ``answer_after``
    # explicitly. Previously these fields were absent, forcing downstream
    # triage to cross-reference ``question_final.correct`` to tell whether a
    # repair actually changed the agent's selection.
    last_answer_raw: str | None = initial_answer_raw

    # ---- Path A: LLM Planner driven repair ----
    for i in range(max_steps):
        # Current-state analysis (may differ from original after edits)
        snapshot = _scope_snapshot_after_timestamp(client.snapshot(), question_baseline_ts)
        before_messages = _session_messages(snapshot.get("session_history", {}))
        current_analysis: Dict[str, Any] | None = None
        # Only reuse the one-time original analysis on the very first step,
        # AND only if it actually produced something usable. A failed analysis
        # (timeout / parse error) carries an `analysis_error` and an empty
        # concept_repair, which would silently force the strict-mode loop into
        # a no-repairable-concepts halt on step 1.
        original_concept_repair = (
            original_analysis.get("concept_repair")
            if isinstance(original_analysis, dict)
            else None
        )
        original_has_concepts = (
            isinstance(original_concept_repair, dict)
            and bool(original_concept_repair.get("hallucinated_concepts"))
        )
        original_is_usable = (
            isinstance(original_analysis, dict)
            and not original_analysis.get("analysis_error")
            and original_has_concepts
        )
        if i == 0 and original_is_usable:
            current_analysis = original_analysis
            _log_analysis_event(
                log_fh,
                phase="current_reuse_original",
                question_index=question_index,
                step=i + 1,
                model=_claim_model,
                timeout_sec=analysis_timeout_sec,
                use_websearch=effective_claim_use_websearch,
                state=snapshot.get("session_history", {}),
                analysis=current_analysis,
                status="reused",
            )
        else:
            _log_analysis_event(
                log_fh,
                phase="current_start",
                question_index=question_index,
                step=i + 1,
                model=_claim_model,
                timeout_sec=analysis_timeout_sec,
                use_websearch=effective_claim_use_websearch,
                state=snapshot.get("session_history", {}),
                status="start",
            )
            current_t0 = time.perf_counter()
            try:
                # Pull the latest known answer from the live trajectory so the
                # planner judge can see the wrong answer it is trying to fix.
                live_messages_for_answer = _session_messages(snapshot.get("session_history", {}))
                live_answer_for_judge = (
                    extract_answer_from_messages(live_messages_for_answer)
                    or initial_answer_raw
                    or ""
                )
                current_analysis = _run_analysis_with_timeout(
                    task=effective_claim_task,
                    state=snapshot["session_history"],
                    model=_claim_model,
                    api_key=claim_api_key,
                    base_url=claim_base_url,
                    evidence_text=claim_evidence_text,
                    use_websearch=effective_claim_use_websearch,
                    search_backend=claim_search_backend,
                    search_max_searches=claim_search_max_searches,
                    search_num_results=claim_search_num_results,
                    search_fetch_top_n=claim_search_fetch_top_n,
                    search_max_output_words=claim_search_max_output_words,
                    assistant_only=True,
                    timeout_sec=analysis_timeout_sec,
                    question_text=goal,
                    current_answer=live_answer_for_judge,
                )
                current_repair = _build_concept_repair_context(current_analysis)
                if current_repair is not None:
                    current_analysis = dict(current_analysis)
                    current_analysis["concept_repair"] = current_repair
                    current_analysis["analysis_task"] = effective_claim_task
                    current_analysis["analysis_use_websearch"] = effective_claim_use_websearch
                _log_analysis_event(
                    log_fh,
                    phase="current_result",
                    question_index=question_index,
                    step=i + 1,
                    model=_claim_model,
                    timeout_sec=analysis_timeout_sec,
                    use_websearch=effective_claim_use_websearch,
                    state=snapshot.get("session_history", {}),
                    analysis=current_analysis,
                    elapsed_sec=time.perf_counter() - current_t0,
                    status="ok",
                )
            except Exception as e:  # noqa: BLE001
                current_analysis = {
                    "analysis_error": _format_exception(e),
                    "analysis_task": effective_claim_task,
                    "analysis_use_websearch": effective_claim_use_websearch,
                }
                _log_analysis_event(
                    log_fh,
                    phase="current_result",
                    question_index=question_index,
                    step=i + 1,
                    model=_claim_model,
                    timeout_sec=analysis_timeout_sec,
                    use_websearch=effective_claim_use_websearch,
                    state=snapshot.get("session_history", {}),
                    analysis=current_analysis,
                    elapsed_sec=time.perf_counter() - current_t0,
                    status="error",
                    error=current_analysis["analysis_error"],
                )

        current_hallucinated_concepts = _hallucinated_concepts(current_analysis)
        if (
            strict_concept_repair_only
            and not current_hallucinated_concepts
            and not wide_window_retry_done
        ):
            # P0: the default analysis only looks at the last assistant turn.
            # Retry ONCE with AGDEBUGGER_ANALYSIS_MAX_ASSISTANT_TURNS widened
            # so we don't halt on trajectories whose tail is TERMINATE / "I
            # could not fetch...". This re-runs the real judge pipeline — no
            # deterministic answer rewrite.
            wide_window_retry_done = True
            prev_env = os.environ.get("AGDEBUGGER_ANALYSIS_MAX_ASSISTANT_TURNS")
            retry_window = os.environ.get("AGDEBUGGER_ANALYSIS_RETRY_WINDOW", "6")
            os.environ["AGDEBUGGER_ANALYSIS_MAX_ASSISTANT_TURNS"] = retry_window
            print(
                f"      [debug] zero repairable concepts; retrying analysis with "
                f"AGDEBUGGER_ANALYSIS_MAX_ASSISTANT_TURNS={retry_window}"
            )
            _log_analysis_event(
                log_fh,
                phase="current_retry_wide_window",
                question_index=question_index,
                step=i + 1,
                model=_claim_model,
                timeout_sec=analysis_timeout_sec,
                use_websearch=effective_claim_use_websearch,
                state=snapshot.get("session_history", {}),
                status="start",
            )
            retry_t0 = time.perf_counter()
            try:
                live_messages_for_answer = _session_messages(snapshot.get("session_history", {}))
                live_answer_for_judge = (
                    extract_answer_from_messages(live_messages_for_answer)
                    or initial_answer_raw
                    or ""
                )
                retry_analysis = _run_analysis_with_timeout(
                    task=effective_claim_task,
                    state=snapshot["session_history"],
                    model=_claim_model,
                    api_key=claim_api_key,
                    base_url=claim_base_url,
                    evidence_text=claim_evidence_text,
                    use_websearch=effective_claim_use_websearch,
                    search_backend=claim_search_backend,
                    search_max_searches=claim_search_max_searches,
                    search_num_results=claim_search_num_results,
                    search_fetch_top_n=claim_search_fetch_top_n,
                    search_max_output_words=claim_search_max_output_words,
                    assistant_only=True,
                    timeout_sec=analysis_timeout_sec,
                    question_text=goal,
                    current_answer=live_answer_for_judge,
                )
                retry_repair = _build_concept_repair_context(retry_analysis)
                if retry_repair is not None:
                    retry_analysis = dict(retry_analysis)
                    retry_analysis["concept_repair"] = retry_repair
                    retry_analysis["analysis_task"] = effective_claim_task
                    retry_analysis["analysis_use_websearch"] = effective_claim_use_websearch
                retry_analysis["wide_window_retry"] = True
                current_analysis = retry_analysis
                current_hallucinated_concepts = _hallucinated_concepts(current_analysis)
                _log_analysis_event(
                    log_fh,
                    phase="current_retry_wide_window",
                    question_index=question_index,
                    step=i + 1,
                    model=_claim_model,
                    timeout_sec=analysis_timeout_sec,
                    use_websearch=effective_claim_use_websearch,
                    state=snapshot.get("session_history", {}),
                    analysis=current_analysis,
                    elapsed_sec=time.perf_counter() - retry_t0,
                    status="ok",
                )
            except Exception as e:  # noqa: BLE001
                _log_analysis_event(
                    log_fh,
                    phase="current_retry_wide_window",
                    question_index=question_index,
                    step=i + 1,
                    model=_claim_model,
                    timeout_sec=analysis_timeout_sec,
                    use_websearch=effective_claim_use_websearch,
                    state=snapshot.get("session_history", {}),
                    elapsed_sec=time.perf_counter() - retry_t0,
                    status="error",
                    error=_format_exception(e),
                )
            finally:
                if prev_env is None:
                    os.environ.pop("AGDEBUGGER_ANALYSIS_MAX_ASSISTANT_TURNS", None)
                else:
                    os.environ["AGDEBUGGER_ANALYSIS_MAX_ASSISTANT_TURNS"] = prev_env

        if strict_concept_repair_only and not current_hallucinated_concepts:
            # PR-2: distinguish "planner asked for repair but provided nothing"
            # (contract violation — fixable by prompt hardening / retries)
            # from "genuinely nothing to repair" (judge says all good).
            cr_for_contract = (current_analysis or {}).get("concept_repair") if isinstance(current_analysis, dict) else None
            if isinstance(cr_for_contract, dict) and cr_for_contract.get("planner_repair_contract_violation"):
                strict_stop_reason = "planner_repair_contract_violation"
            else:
                strict_stop_reason = "no_repairable_concepts"
            # Cat E diagnostic: before halting, record exactly what the
            # analysis actually produced for this question so we can tell
            # whether (a) the judge emitted concept_repair but it was empty,
            # (b) concept_repair was missing entirely, (c) analysis errored,
            # or (d) concepts were present but of the wrong shape.
            cr = (current_analysis or {}).get("concept_repair") if isinstance(current_analysis, dict) else None
            raw_concepts = (cr or {}).get("hallucinated_concepts") if isinstance(cr, dict) else None
            concept_shapes: List[Dict[str, Any]] = []
            if isinstance(raw_concepts, list):
                for item in raw_concepts[:10]:
                    if not isinstance(item, dict):
                        concept_shapes.append({"non_dict_type": type(item).__name__})
                        continue
                    corrected = item.get("corrected_claim_text") or item.get("correct_understanding") or ""
                    concept_shapes.append({
                        "claim_id": item.get("claim_id"),
                        "error_type": item.get("error_type"),
                        "repairable": item.get("repairable"),
                        "has_anchor": bool(item.get("faulty_text_anchor") or item.get("original_context")),
                        "corrected_text_len": len(str(corrected)) if corrected else 0,
                    })
            diag = {
                "concept_repair_present": isinstance(cr, dict),
                "concept_repair_keys": list(cr.keys()) if isinstance(cr, dict) else None,
                "hallucinated_concepts_type": type(raw_concepts).__name__ if raw_concepts is not None else None,
                "hallucinated_concepts_count": len(raw_concepts) if isinstance(raw_concepts, list) else 0,
                "analysis_error": (current_analysis or {}).get("analysis_error") if isinstance(current_analysis, dict) else None,
                "concept_shapes": concept_shapes,
            }
            print(f"      [debug] strict concept-repair mode found no repairable concepts; stopping. diag={diag}")
            _log_event(
                log_fh,
                "debug_halt",
                index=question_index,
                step=i + 1,
                reason=strict_stop_reason,
                analysis_context=current_analysis,
                diagnostic=diag,
            )
            break

        # Merge dual-trajectory analysis + dedup constraint
        planner_context: Dict[str, Any] = dict(current_analysis or {})
        planner_context["original_trajectory_analysis"] = original_analysis
        planner_context["prior_repair_attempts"] = prior_repair_signatures
        planner_context["repair_constraint"] = (
            "You MUST choose a DIFFERENT repair strategy from all prior attempts listed above."
        )
        planner_judgment = planner_context.get("planner_judgment") if isinstance(planner_context, dict) else None
        if isinstance(planner_judgment, dict) and planner_judgment.get("selected_claim_id"):
            planner_context["selected_claim_id"] = planner_judgment.get("selected_claim_id")
        if strict_concept_repair_only:
            planner_context["strict_concept_repair_only"] = True
            planner_context["repair_constraint"] = (
                planner_context["repair_constraint"]
                + " Only repair incorrect or unsupported scientific concepts via edit_and_revert. "
                "Do not inject <answer> tags, TERMINATE, publish, send, or insert_after actions."
            )
        if failed_claim_ids:
            planner_context["failed_claim_ids"] = sorted(failed_claim_ids)
            planner_context["repair_constraint"] = (
                planner_context["repair_constraint"]
                + f" Do not choose any previously failed claim_id values: {sorted(failed_claim_ids)}."
            )
        # Hard outer cap on the number of distinct claim_ids we've already
        # tried and failed for this question. This is a belt-and-braces
        # safeguard: the existing concept_repair_attempt_count check is
        # supposed to cap this at max_concept_repair_attempts, but full-bench
        # case study (idx=528) showed 11 distinct claim_ids attempted
        # back-to-back within a single question — the budget check can slip
        # when action_error paths add to failed_claim_ids without incrementing
        # concept_repair_attempt_count. Bail out explicitly here when the
        # distinct-failed-claim count reaches the cap.
        failed_claim_cap = int(os.environ.get("AGDEBUGGER_MAX_FAILED_CLAIM_IDS", "7"))
        if len(failed_claim_ids) >= failed_claim_cap:
            strict_stop_reason = "max_concept_repair_attempts_exhausted"
            print(
                f"      [debug] failed_claim_ids cap reached "
                f"({len(failed_claim_ids)} >= {failed_claim_cap}); halting"
            )
            _log_event(
                log_fh,
                "debug_halt",
                index=question_index,
                step=i + 1,
                reason=strict_stop_reason,
                failed_claim_count=len(failed_claim_ids),
                failed_claim_cap=failed_claim_cap,
            )
            break
        if last_action_error is not None:
            planner_context["debugger_feedback"] = {
                "last_action_error": last_action_error,
                "last_action_signature": last_action_signature,
            }

        step_fallback_action = None
        step_repair_source = "planner"
        if enable_deterministic_fallback:
            step_fallback_action = _build_deterministic_concept_repair_action(
                snapshot,
                current_analysis,
                excluded_claim_ids=failed_claim_ids,
                llm_rewrite_config=llm_rewrite_config,
            )

        degraded_analysis = isinstance(current_analysis, dict) and (
            current_analysis.get("analysis_error") or current_analysis.get("analysis_fallback_reason")
        )
        if degraded_analysis and step_fallback_action is not None:
            degraded_fallback_error = _degraded_fallback_action_error(
                step_fallback_action,
                analysis_context=current_analysis,
                require_repairable_concepts=strict_concept_repair_only,
            )
            if degraded_fallback_error is None:
                action = step_fallback_action
                step_repair_source = "analysis_degraded_fallback"
                print(f"      [debug step {i + 1}] analysis degraded, using deterministic fallback action={action}")
            else:
                strict_stop_reason = "degraded_fallback_unusable"
                print(
                    f"      [debug] analysis degraded and deterministic fallback is unusable: "
                    f"{degraded_fallback_error}"
                )
                _log_event(
                    log_fh,
                    "debug_halt",
                    index=question_index,
                    step=i + 1,
                    reason=strict_stop_reason,
                    analysis_context=current_analysis,
                    action=step_fallback_action,
                    action_error=degraded_fallback_error,
                )
                break
        else:
            try:
                action = planner.plan(snapshot, analysis_context=planner_context)
                action = _complete_claim_repair_action(
                    action,
                    snapshot=snapshot,
                    analysis_context=current_analysis,
                    force_synthesize_replacement=strict_concept_repair_only,
                    llm_rewrite_config=llm_rewrite_config,
                )
                action = _sanitize_planner_concept_repair_action(
                    action,
                    snapshot=snapshot,
                    analysis_context=current_analysis,
                    strict_concept_repair_only=strict_concept_repair_only,
                    llm_rewrite_config=llm_rewrite_config,
                )
                step_repair_source = "planner"
            except Exception as e:  # noqa: BLE001
                print(f"      [debug] planner request failed: {type(e).__name__}: {e}")
                if step_fallback_action is None:
                    _log_event(
                        log_fh,
                        "debug_error",
                        index=question_index,
                        step=i + 1,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    continue
                planner_failed_fallback_error = _degraded_fallback_action_error(
                    step_fallback_action,
                    analysis_context=current_analysis,
                    require_repairable_concepts=strict_concept_repair_only,
                )
                if planner_failed_fallback_error is not None:
                    strict_stop_reason = "planner_failed_fallback_unusable"
                    print(
                        f"      [debug] planner failed and deterministic fallback is unusable: "
                        f"{planner_failed_fallback_error}"
                    )
                    _log_event(
                        log_fh,
                        "debug_halt",
                        index=question_index,
                        step=i + 1,
                        reason=strict_stop_reason,
                        analysis_context=current_analysis,
                        action=step_fallback_action,
                        action_error=planner_failed_fallback_error,
                    )
                    break
                action = step_fallback_action
                step_repair_source = "planner_failed_fallback"
                print(f"      [debug step {i + 1}] planner failed, using deterministic fallback action={action}")

        # PR-2: short-circuit when the planner keeps picking a claim that has
        # already failed. Without this, idx=71-style cases burn the whole
        # max_concept_repair_attempts budget on three variants of the same
        # edit. Track the selection, and if a just-picked claim is already in
        # failed_claim_ids, skip this attempt and mark the loop for halting.
        current_claim_id = action.get("claim_id")
        if isinstance(current_claim_id, str) and current_claim_id:
            if current_claim_id in failed_claim_ids:
                print(
                    f"      [debug step {i + 1}] planner re-selected already-failed "
                    f"claim_id={current_claim_id!r}; skipping"
                )
                planner_selected_history.append(current_claim_id)
                last_action_error = (
                    "Planner picked a claim_id that has already failed this question. "
                    "Choose a DIFFERENT claim_id or downgrade to no_repair_needed."
                )
                last_action_signature = json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)
                # If the planner has now repeated itself twice in a row with
                # failed material, halt — there is no new strategy to try.
                if (
                    len(planner_selected_history) >= 2
                    and planner_selected_history[-1] == planner_selected_history[-2]
                ):
                    strict_stop_reason = "planner_repeated_failed_selection"
                    _log_event(
                        log_fh,
                        "debug_halt",
                        index=question_index,
                        step=i + 1,
                        reason=strict_stop_reason,
                        analysis_context=current_analysis,
                        action=action,
                        action_error=last_action_error,
                    )
                    break
                continue
            planner_selected_history.append(current_claim_id)

        if action.get("anchor_not_found"):
            # Anchor text could not be substring-located in the target message.
            # Instead of skipping, execute the action anyway — replacement_text
            # is already set to corrected_claim_text by _complete_claim_repair_action,
            # which replaces the entire target message with just the correction.
            # The agent re-runs from a clean slate without the old wrong reasoning.
            print(f"      [debug step {i + 1}] anchor not found; executing with full-message replacement")

        validation_error = _validate_concept_repair_action(
            action,
            analysis_context=current_analysis,
            require_repairable_concepts=strict_concept_repair_only,
        )
        if validation_error is not None and step_fallback_action is not None and action != step_fallback_action:
            print(
                f"      [debug step {i + 1}] invalid planner action for concept repair: "
                f"{validation_error}; using concept-repair fallback"
            )
            action = step_fallback_action
            step_repair_source = "concept_repair_fallback"
            validation_error = _validate_concept_repair_action(
                action,
                analysis_context=current_analysis,
                require_repairable_concepts=strict_concept_repair_only,
            )
        if validation_error is not None:
            if action.get("claim_id"):
                failed_claim_ids.add(str(action.get("claim_id")))
            last_action_error = validation_error
            last_action_signature = json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)
            print(f"      [debug] concept-repair validation failed: {validation_error}")
            _log_event(
                log_fh,
                "debug_step",
                index=question_index,
                step=i + 1,
                snapshot=planner._compact_snapshot(snapshot),
                analysis_context=planner_context,
                action=action,
                action_error=validation_error,
                answer_after=None,
                answer_correct=False,
                repair_source=step_repair_source,
            )
            continue

        if action.get("timestamp") is None and action.get("target_turn") is not None:
            bound_timestamp = _assistant_timestamp_for_turn(
                _session_messages(snapshot.get("session_history", {})),
                action.get("target_turn"),
            )
            if bound_timestamp is not None:
                action["timestamp"] = bound_timestamp

        # Dedup check.
        # Cat B fix: also build a content-only signature ``(claim_id,
        # replacement_text)`` so that re-emitting the exact same
        # corrective text under a different timestamp / target_turn is
        # still detected as a duplicate. Without this, the runaway-retry
        # loop on w1/idx=11 (13 identical replacement_texts) goes
        # undetected because the action JSON differs by metadata.
        action_signature = json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)
        content_signature = json.dumps(
            {
                "claim_id": action.get("claim_id"),
                "replacement_text": (action.get("replacement_text") or "")[:2000],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if action_signature in prior_repair_signatures or content_signature in prior_repair_signatures:
            print(f"      [debug step {i + 1}] duplicate action, skipping")
            if action.get("claim_id"):
                failed_claim_ids.add(str(action.get("claim_id")))
            _log_event(log_fh, "duplicate_action", index=question_index, step=i + 1, action=action)
            continue

        action_name = action.get("action", "")
        # edit_and_revert budget check
        if action_name == "edit_and_revert":
            concept_repair_attempt_count += 1
            repair_budget = max_concept_repair_attempts if strict_concept_repair_only else max_edit_attempts
            if concept_repair_attempt_count > repair_budget:
                print(f"      [debug] max concept repair attempts ({repair_budget}) exhausted")
                strict_stop_reason = "max_concept_repair_attempts_exhausted" if strict_concept_repair_only else strict_stop_reason
                break

        print(f"      [debug step {i + 1}] action={action}")

        # Start loop for actions that enqueue messages
        if action_name in ("send", "publish", "insert_after", "edit_and_revert"):
            client.start_loop()

        action_error = None
        try:
            keep_running = execute_action(client, action)
        except Exception as e:  # noqa: BLE001
            keep_running = True
            action_error = f"{type(e).__name__}: {e}"
            if action.get("claim_id"):
                failed_claim_ids.add(str(action.get("claim_id")))
            last_action_error = action_error
            last_action_signature = action_signature
            print(f"      [debug] action failed: {action_error}")
            _log_event(
                log_fh,
                "debug_step",
                index=question_index,
                step=i + 1,
                snapshot=planner._compact_snapshot(snapshot),
                analysis_context=planner_context,
                action=action,
                action_error=action_error,
                answer_after=None,
                answer_correct=False,
                repair_source=step_repair_source,
            )
            try:
                _stop_loop_with_fallback(client, stage=f"debug-step-{i + 1}-error")
            except Exception:
                pass
            continue

        try:
            if action_name in ("send", "publish", "insert_after", "edit_and_revert"):
                _ = _wait_for_debug_settlement(
                    client,
                    before_messages=before_messages,
                    timeout_sec=settle_timeout_sec,
                    stall_timeout_sec=min(60.0, max(15.0, settle_timeout_sec / 5)),
                )
            else:
                client.wait_until_idle(timeout_sec=settle_timeout_sec)
        except Exception as e:  # noqa: BLE001
            print(f"      [debug] wait_until_idle warning: {e}")
        finally:
            try:
                _stop_loop_with_fallback(client, stage=f"debug-step-{i + 1}")
            except Exception:
                pass

        # Record both the full and the content-only signatures so future
        # iterations dedup against either form.
        prior_repair_signatures.append(action_signature)
        prior_repair_signatures.append(content_signature)

        state = _scope_history_state_after_timestamp(client.get_session_history(), question_baseline_ts)
        messages = _session_messages(state)
        answer_raw = extract_answer_from_messages(messages)
        if answer_raw:
            last_answer_raw = answer_raw
        normalized = normalize_answer(answer_raw, num_options=num_options) if answer_raw else ""
        answer_correct = normalized == expected_normalized if answer_raw else False
        initial_normalized = normalize_answer(initial_answer_raw, num_options=num_options) if initial_answer_raw else ""
        answer_unchanged = bool(answer_raw and normalized and normalized == initial_normalized)
        if answer_correct:
            last_action_error = None
            last_action_signature = action_signature
        elif answer_unchanged:
            # Repair ran but answer stayed the same — this claim was ineffective.
            if action.get("claim_id"):
                failed_claim_ids.add(str(action.get("claim_id")))
            last_action_error = (
                "Concept repair did not change the answer. "
                "Choose a DIFFERENT claim_id targeting a different part of the reasoning."
            )
            last_action_signature = action_signature
        elif last_action_signature == action_signature:
            if action.get("claim_id"):
                failed_claim_ids.add(str(action.get("claim_id")))
            last_action_error = "Previous action repeated without fixing the answer. Choose a different repair action."
        else:
            last_action_error = None
            last_action_signature = action_signature

        _log_event(log_fh, "debug_step",
            index=question_index,
            step=i + 1,
            snapshot=planner._compact_snapshot(snapshot),
            analysis_context=planner_context,
            action=action,
            action_error=action_error,
            answer_after=answer_raw,
            answer_correct=answer_correct,
            repair_source=step_repair_source,
        )

        if answer_raw and answer_correct:
            _log_event(log_fh, "debug_result",
                index=question_index, fixed=True,
                fixed_answer=answer_raw, total_steps=i + 1,
                repair_source=step_repair_source,
                answer_before=initial_answer_raw,
                answer_after=answer_raw,
            )
            return True, answer_raw
        if not keep_running:
            break

    # ---- Path B: Deterministic fallback (optional) ----
    if enable_deterministic_fallback:
        fallback_action = _build_deterministic_concept_repair_action(
            original_snapshot,
            original_analysis,
            excluded_claim_ids=failed_claim_ids,
            llm_rewrite_config=llm_rewrite_config,
        )
        if fallback_action is not None:
            print("      [debug] Path A exhausted; trying deterministic concept-repair fallback ...")
            client.start_loop()
            fb_before_messages = _session_messages(
                _scope_snapshot_after_timestamp(client.snapshot(), question_baseline_ts).get("session_history", {})
            )
            fb_error = None
            try:
                execute_action(client, fallback_action)
            except Exception as e:  # noqa: BLE001
                fb_error = f"{type(e).__name__}: {e}"
                print(f"      [debug] deterministic fallback action failed: {fb_error}")

            if fb_error is None:
                try:
                    _ = _wait_for_debug_settlement(
                        client,
                        before_messages=fb_before_messages,
                        timeout_sec=settle_timeout_sec,
                        stall_timeout_sec=min(60.0, max(15.0, settle_timeout_sec / 5)),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"      [debug] deterministic fallback settle warning: {e}")

            try:
                _stop_loop_with_fallback(client, stage="deterministic-fallback")
            except Exception:
                pass

            if fb_error is None:
                state = _scope_history_state_after_timestamp(client.get_session_history(), question_baseline_ts)
                messages = _session_messages(state)
                answer_raw = extract_answer_from_messages(messages)
                if answer_raw:
                    last_answer_raw = answer_raw
                normalized = normalize_answer(answer_raw, num_options=num_options) if answer_raw else ""
                answer_correct = normalized == expected_normalized if answer_raw else False

                _log_event(log_fh, "debug_step",
                    index=question_index,
                    step="fallback",
                    action=fallback_action,
                    action_error=None,
                    answer_after=answer_raw,
                    answer_correct=answer_correct,
                    repair_source="deterministic_fallback",
                )

                if answer_raw and answer_correct:
                    _log_event(log_fh, "debug_result",
                        index=question_index, fixed=True,
                        fixed_answer=answer_raw, total_steps=max_steps,
                        repair_source="deterministic_fallback",
                        answer_before=initial_answer_raw,
                        answer_after=answer_raw,
                    )
                    return True, answer_raw

    _log_event(log_fh, "debug_result",
        index=question_index, fixed=False,
        fixed_answer=None, total_steps=max_steps,
        repair_source=None,
        halt_reason=strict_stop_reason,
        answer_before=initial_answer_raw,
        answer_after=last_answer_raw,
    )
    return False, None


def _load_all_examples(input_path: Path) -> List[Dict[str, Any]]:
    """Load every JSONL line that has parseable edges, without component filtering."""
    import json as _json
    examples: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = _json.loads(raw)
            except _json.JSONDecodeError:
                continue
            from browse_bio_graph_cluster_examples import parse_edges
            edges = parse_edges(obj.get("response", ""))
            if not edges:
                continue
            all_nodes = sorted({src for src, _, _ in edges} | {dst for _, _, dst in edges})
            options = []
            for i in range(1, 7):
                key = f"option{i}"
                val = normalize_text(obj.get(key, ""))
                if val:
                    options.append((key, val))
            examples.append({
                "line_no": line_no,
                "question": normalize_text(obj.get("question", "")),
                "answer": normalize_text(obj.get("answer", "")),
                "options": options,
                "edges": edges,
                "nodes": all_nodes,
            })
    return examples


def load_examples(component_id: int, input_path: Path, graph_dir: Path) -> List[Dict[str, Any]]:
    if component_id == 0:
        examples = _load_all_examples(input_path)
        annotate_focus_nodes(examples)
        print(f"Loaded ALL components: {len(examples)} examples (component_id=0)")
        return examples

    components_csv = graph_dir / "components.csv"
    if not components_csv.exists():
        raise FileNotFoundError(f"components.csv not found: {components_csv}")

    component_nodes = load_component_nodes(components_csv, component_id)
    examples = load_examples_in_component(input_path, component_nodes)
    annotate_focus_nodes(examples)
    print(f"Loaded component {component_id}: {len(examples)} examples, {len(component_nodes)} nodes")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGDebugger in background and auto-debug dataset failures")
    parser.add_argument("--component-id", type=int, required=True)
    parser.add_argument("--input", type=Path, default=DATASETS_DIR / "Protein_professional.jsonl")
    parser.add_argument("--graph-dir", type=Path, default=DATASETS_DIR / "bio_graph_output_professional")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--module", default="test_agent_debug:get_agent_team")
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        help="Legacy fallback timeout used when per-stage timeouts are not specified.",
    )
    parser.add_argument("--reset-timeout", type=float, default=None, help="Timeout for GroupChatReset round.")
    parser.add_argument("--question-timeout", type=float, default=None, help="Timeout for main question round.")
    parser.add_argument(
        "--question-stall-timeout",
        type=float,
        default=None,
        help="If there are no new messages and no pending tasks for this long, treat the question round as stalled.",
    )
    parser.add_argument(
        "--question-retry-attempts",
        type=int,
        default=None,
        help="Retry count when the question round times out without producing any new trajectory.",
    )
    parser.add_argument("--debug-step-timeout", type=float, default=None, help="Timeout for each debug-loop settle round.")
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="Path to backend server log. Auto-generated under dated run directory if not specified.",
    )
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument(
        "--run-log",
        type=Path,
        default=None,
        help="Path to structured JSONL run log. Auto-generated if not specified.",
    )
    parser.add_argument(
        "--analysis-detail-log",
        type=Path,
        default=None,
        help="Path to detailed concept/fact-check/judge JSONL log. Auto-generated if not specified.",
    )

    parser.add_argument(
        "--enable-llm-debug",
        dest="enable_llm_debug",
        action="store_true",
        help="Enable the external LLM trajectory debugger. Enabled by default.",
    )
    parser.add_argument(
        "--disable-llm-debug",
        dest="enable_llm_debug",
        action="store_false",
        help="Disable the external LLM trajectory debugger.",
    )
    parser.set_defaults(enable_llm_debug=True)
    parser.add_argument("--debug-max-steps", type=int, default=12)
    parser.add_argument(
        "--max-edit-attempts",
        type=int,
        default=int(os.environ.get("AGDEBUGGER_MAX_EDIT_ATTEMPTS", "7")),
        help="Max edit_and_revert attempts in Path A before falling back. Default 7.",
    )
    parser.add_argument(
        "--max-concept-repair-attempts",
        type=int,
        default=int(os.environ.get("AGDEBUGGER_MAX_CONCEPT_REPAIR_ATTEMPTS", os.environ.get("AGDEBUGGER_MAX_EDIT_ATTEMPTS", "7"))),
        help="Max concept_repair edit_and_revert attempts before marking the sample wrong. Default 7.",
    )
    parser.add_argument(
        "--enable-deterministic-fallback",
        dest="enable_deterministic_fallback",
        action="store_true",
        help="Enable deterministic concept-repair fallback (Path B). Enabled by default.",
    )
    parser.add_argument(
        "--disable-deterministic-fallback",
        dest="enable_deterministic_fallback",
        action="store_false",
        help="Disable deterministic concept-repair fallback (Path B).",
    )
    parser.set_defaults(enable_deterministic_fallback=True)
    parser.add_argument(
        "--strict-concept-repair-only",
        dest="strict_concept_repair_only",
        action="store_true",
        help="Require at least one repairable concept from analysis before attempting concept-level edit_and_revert repairs.",
    )
    parser.add_argument(
        "--allow-direct-answer-rewrite",
        dest="strict_concept_repair_only",
        action="store_false",
        help="Deprecated compatibility flag. Direct answer rewrites are still rejected; this only disables the no-repairable-concepts halt.",
    )
    parser.set_defaults(
        strict_concept_repair_only=os.environ.get("AGDEBUGGER_STRICT_CONCEPT_REPAIR_ONLY", "0") == "1"
    )
    parser.add_argument(
        "--claim-task",
        choices=["research_questions", "medical_guidelines", "legal_cases", "coding", "scientific_concept_discovery"],
        default=None,
        help="Optional claim extraction/judging task injected into the debugger context.",
    )
    parser.add_argument("--claim-evidence", default="", help="Inline evidence text for the external claim judge.")
    parser.add_argument("--claim-evidence-file", type=Path, default=None, help="Evidence text file for the external claim judge.")
    parser.add_argument(
        "--claim-use-websearch",
        dest="claim_use_websearch",
        action="store_true",
        help="Use the local websearch library to build evidence automatically for each extracted claim.",
    )
    parser.add_argument(
        "--disable-claim-use-websearch",
        dest="claim_use_websearch",
        action="store_false",
        help="Disable automatic websearch evidence gathering for extracted claims.",
    )
    parser.add_argument(
        "--claim-search-backend",
        choices=["bright_data", "serper"],
        default="bright_data",
        help="Search backend for automatic claim evidence. Default forces Bright Data client.",
    )
    parser.add_argument("--claim-search-max-searches", type=int, default=3)
    parser.add_argument("--claim-search-num-results", type=int, default=5)
    parser.add_argument("--claim-search-fetch-top-n", type=int, default=2)
    parser.add_argument("--claim-search-max-output-words", type=int, default=1500)
    parser.add_argument("--model", default=os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument(
        "--model-planner",
        default=None,
        help="Model for LLMPlanner (falls back to --model)",
    )
    parser.add_argument(
        "--model-claim",
        default=None,
        help="Model for ClaimExtractor/ClaimJudge/WebSearchEvidenceProvider (falls back to --model)",
    )
    parser.add_argument("--api-key", default=os.environ.get("AGENTDEBUG_OPENAI_API_KEY", ""))
    parser.add_argument(
        "--api-base",
        default=os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.set_defaults(
        claim_use_websearch=_env_flag("AGDEBUGGER_CLAIM_USE_WEBSEARCH", default=True),
    )
    args = parser.parse_args()
    default_run_timeout = args.run_timeout
    if default_run_timeout is None:
        default_run_timeout = float(os.environ.get("AGDEBUGGER_RUN_TIMEOUT", "300"))
    args.run_timeout = default_run_timeout
    if args.reset_timeout is None:
        args.reset_timeout = float(os.environ.get("AGDEBUGGER_RESET_TIMEOUT", str(default_run_timeout)))
    if args.question_timeout is None:
        args.question_timeout = float(os.environ.get("AGDEBUGGER_QUESTION_TIMEOUT", str(default_run_timeout)))
    if args.question_stall_timeout is None:
        args.question_stall_timeout = float(os.environ.get("AGDEBUGGER_QUESTION_STALL_TIMEOUT", "60"))
    if args.question_retry_attempts is None:
        args.question_retry_attempts = int(os.environ.get("AGDEBUGGER_QUESTION_RETRY_ATTEMPTS", "1"))
    if args.debug_step_timeout is None:
        args.debug_step_timeout = float(os.environ.get("AGDEBUGGER_DEBUG_STEP_TIMEOUT", str(default_run_timeout)))
    model_timeout = float(os.environ.get("AGENTDEBUG_MODEL_TIMEOUT_SEC", "600"))
    min_round_timeout = model_timeout + 30.0
    if args.question_timeout < min_round_timeout:
        args.question_timeout = min_round_timeout
    if args.debug_step_timeout < min_round_timeout:
        args.debug_step_timeout = min_round_timeout
    return args


def _load_claim_evidence_text(args: argparse.Namespace) -> str:
    if args.claim_evidence_file is not None:
        return args.claim_evidence_file.read_text(encoding="utf-8")
    return args.claim_evidence


def main() -> None:
    args = parse_args()
    run_dir, run_log_path, server_log_path, analysis_detail_log_path = _resolve_log_paths(args)
    args.run_log = run_log_path
    args.server_log = server_log_path
    args.analysis_detail_log = analysis_detail_log_path
    os.environ["AGDEBUGGER_ANALYSIS_DETAIL_LOG"] = str(analysis_detail_log_path)
    repo_dir = Path(__file__).resolve().parent
    api_base = f"http://{args.host}:{args.port}/api"
    client_timeout_sec = max(
        args.ready_timeout,
        args.run_timeout,
        args.reset_timeout,
        args.question_timeout,
        args.debug_step_timeout,
        float(os.environ.get("AGENTDEBUG_MODEL_TIMEOUT_SEC", "600")),
    ) + 30.0
    client = AGDebuggerClient(api_base, timeout_sec=client_timeout_sec)
    claim_evidence_text = _load_claim_evidence_text(args)

    # -- Open structured run log --
    log_fh = open(run_log_path, "a", encoding="utf-8", buffering=1)
    print(f"Run log: {run_log_path}")
    print(f"Run dir: {run_dir}")
    print(f"Analysis detail log: {analysis_detail_log_path}")

    handle: Optional[ServerHandle] = None
    if not args.reuse_server:
        print(f"Starting AGDebugger backend in background at {api_base} ...")
        handle = start_agdebugger_background(
            repo_dir=repo_dir,
            module_expr=args.module,
            host=args.host,
            port=args.port,
            log_path=args.server_log,
        )
    else:
        print(f"Reusing existing AGDebugger backend at {api_base}")

    try:
        wait_backend_ready(client, handle, timeout_sec=args.ready_timeout)
        topics = client.get_topics()
        manager_topic = _guess_manager_topic(topics)
        print(f"Backend ready. manager_topic={manager_topic}")

        examples = load_examples(args.component_id, args.input, args.graph_dir)
        if args.start >= len(examples):
            raise ValueError(f"--start {args.start} >= total examples {len(examples)}")
        end = len(examples) if args.limit is None else min(args.start + args.limit, len(examples))
        run_set = examples[args.start : end]
        print(f"Running examples index range [{args.start}, {end}) -> {len(run_set)} questions")

        _log_event(
            log_fh,
            "run_start",
            args=vars(args),
            run_dir=str(run_dir),
            analysis_detail_log=str(analysis_detail_log_path),
        )

        logs_seen = len(client.get_logs())
        total = 0
        correct = 0
        debug_fixed = 0

        for i, ex in enumerate(run_set):
            global_idx = args.start + i
            total += 1
            task = format_task(ex)
            gt_raw = str(ex["answer"])
            gt_norm = normalize_answer(gt_raw, num_options=len(ex["options"]))
            print(f"\n{'#' * 68}")
            print(f"# Example {global_idx + 1} | line={ex.get('line_no')} | GT={gt_raw}")
            print(f"{'#' * 68}")

            _log_event(log_fh, "question_start",
                index=global_idx, line_no=ex.get("line_no"),
                task=task, ground_truth=gt_raw,
            )

            run_error: Optional[str] = None
            try:
                messages, ans_raw, question_baseline_ts = run_question(
                    client,
                    manager_topic,
                    task,
                    reset_timeout_sec=args.reset_timeout,
                    question_timeout_sec=args.question_timeout,
                    question_stall_timeout_sec=args.question_stall_timeout,
                    question_retry_attempts=args.question_retry_attempts,
                )
            except Exception as e:  # noqa: BLE001
                run_error = str(e)
                messages = e.messages if isinstance(e, (QuestionPhaseTimeout, QuestionPhaseRuntimeError)) else []
                ans_raw = None
                question_baseline_ts = -1

            logs = client.get_logs()
            new_logs = logs[logs_seen:]
            logs_seen = len(logs)
            error_logs = [l for l in new_logs if _is_error_log(l)]

            original_ans_raw = ans_raw
            forced_initial_answer = _forced_initial_answer_override()
            answer_override_applied = False
            if forced_initial_answer and ans_raw is not None and run_error is None:
                ans_raw = forced_initial_answer
                answer_override_applied = True

            ans_norm = normalize_answer(ans_raw, len(ex["options"])) if ans_raw else ""
            is_correct = ans_norm == gt_norm
            print(f"  Answer(raw): {ans_raw or '(not found)'}")
            print(f"  Answer(norm): {ans_norm or '(empty)'}")
            print(f"  GroundTruth : {gt_norm}")
            if answer_override_applied:
                print(
                    "  [test-hook] forced initial answer override "
                    f"from {original_ans_raw or '(not found)'} to {ans_raw}"
                )
            if run_error:
                print(f"  RuntimeError: {run_error}")
            if error_logs:
                print(f"  ErrorLogs   : {len(error_logs)}")

            _log_event(log_fh, "question_result",
                index=global_idx, messages=messages,
                answer_raw=ans_raw, answer_norm=ans_norm,
                original_answer_raw=original_ans_raw,
                forced_initial_answer=forced_initial_answer or None,
                answer_override_applied=answer_override_applied,
                is_correct=is_correct, error_logs=error_logs,
                run_error=run_error,
            )

            # Debug eligibility. Historically we required a clean finish
            # (``ans_raw is not None and run_error is None``). In practice the
            # 20-question study showed half of the final-wrong cases timed
            # out (900s) or hit max-message without emitting ``<answer>``,
            # and all of them were being dropped *before* analysis had a
            # chance to look at their partial trajectory. Relax the gate to
            # also include partial trajectories with meaningful agent
            # activity — let the analysis / planner decide whether there is
            # anything to repair.
            has_clean_trajectory = ans_raw is not None and run_error is None
            has_partial_trajectory = (
                not has_clean_trajectory
                and _has_meaningful_agent_activity(messages)
            )
            needs_debug = not is_correct and (has_clean_trajectory or has_partial_trajectory)
            if has_partial_trajectory:
                print(
                    f"  [debug] partial trajectory (ans_raw={ans_raw!r}, "
                    f"run_error={run_error!r}) — still attempting debug"
                )
            fixed = False

            if needs_debug and args.enable_llm_debug:
                if not args.api_key:
                    print("  [debug] skipped: missing --api-key / AGENTDEBUG_OPENAI_API_KEY")
                else:
                    snippet_logs = "\n".join(
                        f"[{entry.get('level')}] {entry.get('message')}" for entry in error_logs[-5:]
                    ) or "(no backend error logs)"
                    debug_goal = _build_debug_goal(task, snippet_logs)
                    print("  [debug] invoking external LLM trajectory debugger ...")
                    fixed, fixed_answer = run_llm_debug_loop(
                        client,
                        model=args.model,
                        model_planner=args.model_planner,
                        model_claim=args.model_claim,
                        api_key=args.api_key,
                        api_base=args.api_base,
                        goal=debug_goal,
                        expected_normalized=gt_norm,
                        max_steps=args.debug_max_steps,
                        settle_timeout_sec=args.debug_step_timeout,
                        num_options=len(ex["options"]),
                        log_fh=log_fh,
                        question_index=global_idx,
                        claim_task=args.claim_task,
                        claim_evidence_text=claim_evidence_text,
                        claim_use_websearch=args.claim_use_websearch,
                        claim_search_backend=args.claim_search_backend,
                        claim_search_max_searches=args.claim_search_max_searches,
                        claim_search_num_results=args.claim_search_num_results,
                        claim_search_fetch_top_n=args.claim_search_fetch_top_n,
                        claim_search_max_output_words=args.claim_search_max_output_words,
                        max_edit_attempts=args.max_edit_attempts,
                        max_concept_repair_attempts=args.max_concept_repair_attempts,
                        enable_deterministic_fallback=args.enable_deterministic_fallback,
                        strict_concept_repair_only=args.strict_concept_repair_only,
                        question_baseline_ts=question_baseline_ts,
                        initial_answer_raw=ans_raw,
                    )
                    print(f"  [debug] fixed={fixed}, answer={fixed_answer}")

            final_correct = is_correct or fixed
            if final_correct:
                correct += 1
                if fixed:
                    debug_fixed += 1
                print("  Result      : CORRECT")
            else:
                print("  Result      : WRONG")

            print(f"  RunningScore: {correct}/{total} ({(correct / total) * 100:.1f}%)")

            _log_event(log_fh, "question_final",
                index=global_idx, correct=final_correct,
                debug_fixed=bool(fixed),
                running_score=f"{correct}/{total}",
            )

        print(f"\nFinal score: {correct}/{total} ({(correct / total * 100) if total else 0:.1f}%)")
        print(f"LLM debug fixed count: {debug_fixed}")
        print(f"Server log file: {args.server_log}")
        print(f"Run log file: {run_log_path}")

        _log_event(log_fh, "run_summary",
            total=total, correct=correct,
            debug_fixed=debug_fixed,
            accuracy=correct / total if total else 0,
        )

    finally:
        log_fh.close()
        if handle is not None and not args.keep_server:
            print("Stopping background AGDebugger backend ...")
            handle.close()
        elif handle is not None:
            print("Keeping background AGDebugger backend alive (--keep-server).")


if __name__ == "__main__":
    main()
