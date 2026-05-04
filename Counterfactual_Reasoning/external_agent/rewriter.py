from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from external_agent.llm import OpenAICompatibleLLM
from external_agent.strategies import (
    REWRITER_SYSTEM_PROMPT,
    build_rewriter_user_prompt,
)


_OPTION_REF_RE = re.compile(
    # Match labels with a space separator: "option 3", "choice A", "candidate 2".
    # Must have at least one whitespace char between the head word and the label,
    # otherwise the plural word "options" would match as option + "s" — a false
    # positive seen in real leak samples (idx=11 step=3 "...in the options
    # matches the paper").
    r"\b(?:option|choice|candidate)\s+(?:[a-z]|\d+)\b"
    # Also match the compact forms: "option#3", "option3" (no space at all).
    r"|\boption\s*#\d+\b|\boption\d+\b",
    flags=re.IGNORECASE,
)
_DIRECT_ANSWER_RE = re.compile(
    r"(?i)(?:therefore|thus|hence|so|in conclusion)[^.\n]{0,40}"
    r"(?:the\s+)?(?:correct|best|final|most (?:appropriate|likely))\s+"
    r"(?:answer|option|choice)\b"
)


@dataclass
class RewriteResult:
    rewritten_text: Optional[str]
    fallback_reason: str  # one of: ok, timeout, empty, leak_guard, json_parse_error, no_claims, llm_error
    elapsed_sec: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)
    # The text the LLM actually produced, even when the leak guard rejected it.
    # Kept so the caller can surface a bounded sample into diagnostics for
    # post-run case study — without it we have no visibility into *why* the
    # rewrite was discarded. Always a string (possibly empty).
    rejected_text: str = ""


def _leak_guard(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    lowered = text.lower()
    if "<answer>" in lowered or "</answer>" in lowered:
        return "leak_guard"
    if re.search(r"\bterminate\b", text, flags=re.IGNORECASE):
        return "leak_guard"
    if _OPTION_REF_RE.search(text):
        return "leak_guard"
    if _DIRECT_ANSWER_RE.search(text):
        return "leak_guard"
    return None


class SpanRewriter:
    """LLM-driven rewrite for a fused span produced by Path C.

    Parallel to ``ClaimJudge`` and ``PlannerJudge``: single prompt → single
    JSON response. The rewritten text replaces the primary claim's
    ``corrected_claim_text`` for that span only. On any failure mode
    (timeout, invalid JSON, empty output, leak guard trip) the caller keeps
    the Step-1 corrected_text — we never block repair because the rewrite
    itself failed.
    """

    def __init__(self, llm: OpenAICompatibleLLM) -> None:
        self.llm = llm

    async def rewrite_fused_span(
        self,
        *,
        question_text: str,
        target_turn_number: int | None,
        original_span_content: str,
        prefix_context: str,
        suffix_context: str,
        contributing_claims: List[Dict[str, Any]],
        timeout_sec: float,
    ) -> RewriteResult:
        if not contributing_claims:
            return RewriteResult(None, "no_claims")
        user_prompt = build_rewriter_user_prompt(
            question_text=question_text,
            target_turn_number=target_turn_number,
            original_span_content=original_span_content,
            prefix_context=prefix_context,
            suffix_context=suffix_context,
            contributing_claims=contributing_claims,
        )
        started = time.perf_counter()
        try:
            payload = await asyncio.wait_for(
                self.llm.complete_json(REWRITER_SYSTEM_PROMPT, user_prompt),
                timeout=max(1.0, float(timeout_sec)),
            )
        except asyncio.TimeoutError:
            return RewriteResult(None, "timeout", time.perf_counter() - started)
        except Exception:  # noqa: BLE001
            return RewriteResult(None, "llm_error", time.perf_counter() - started)
        elapsed = time.perf_counter() - started
        if not isinstance(payload, dict):
            return RewriteResult(None, "json_parse_error", elapsed, raw_response={"raw": payload})
        rewritten = str(payload.get("rewritten_text", "")).strip()
        if not rewritten:
            return RewriteResult(None, "empty", elapsed, raw_response=payload)
        leak = _leak_guard(rewritten)
        if leak:
            return RewriteResult(None, leak, elapsed, raw_response=payload, rejected_text=rewritten)
        return RewriteResult(rewritten, "ok", elapsed, raw_response=payload)


def run_rewrite_sync(
    rewriter: SpanRewriter,
    *,
    question_text: str,
    target_turn_number: int | None,
    original_span_content: str,
    prefix_context: str,
    suffix_context: str,
    contributing_claims: List[Dict[str, Any]],
    timeout_sec: float,
) -> RewriteResult:
    """Synchronous wrapper so the deterministic (non-async) repair helpers
    can invoke the rewriter without the caller having to manage an event
    loop. Safe to call from a thread that does NOT already own a running
    loop (the runner's repair pipeline is synchronous)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # The runner synchronously drives the backend and analysis via
            # asyncio.run; when no loop is active this branch is not taken.
            # If we are unexpectedly inside a running loop (e.g. tests that
            # patch in a loop), fall through to the wait-for version using a
            # fresh loop in a helper thread.
            raise RuntimeError("loop_already_running")
    except RuntimeError:
        pass
    try:
        return asyncio.run(
            rewriter.rewrite_fused_span(
                question_text=question_text,
                target_turn_number=target_turn_number,
                original_span_content=original_span_content,
                prefix_context=prefix_context,
                suffix_context=suffix_context,
                contributing_claims=contributing_claims,
                timeout_sec=timeout_sec,
            )
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                rewriter.rewrite_fused_span(
                    question_text=question_text,
                    target_turn_number=target_turn_number,
                    original_span_content=original_span_content,
                    prefix_context=prefix_context,
                    suffix_context=suffix_context,
                    contributing_claims=contributing_claims,
                    timeout_sec=timeout_sec,
                )
            )
        finally:
            loop.close()
