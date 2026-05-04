from __future__ import annotations

import inspect
import os
import re
from typing import Awaitable, Callable, Iterable, List

from external_agent.llm import OpenAICompatibleLLM
from external_agent.schemas import Claim, EvidenceBundle, Judgment, PlannerJudgment
from external_agent.strategies import ClaimJudgeStrategy


EvidenceProvider = Callable[[Claim], EvidenceBundle | Awaitable[EvidenceBundle]]

_REASON_SUPPORTS_REPAIR_RE = re.compile(
    r"\b(incorrect|unsupported|wrong|mismatch|not supported|contradict(?:ed|s)?|inconsistent|does not support|insufficient evidence)\b",
    re.IGNORECASE,
)
# Negation cues that should suppress the substring-based verification_error
# elevation. The original logic would happily upgrade a judge whose reason said
# "the claim is NOT incorrect" or "no mismatch found" — these phrases trip the
# guard so we leave the LLM's verdict alone.
_REASON_NEGATION_RE = re.compile(
    r"\b(not\s+(?:incorrect|unsupported|wrong|inconsistent)|no\s+(?:mismatch|contradiction)|"
    r"correct|consistent|supported|matches|supports the claim|aligns with)\b",
    re.IGNORECASE,
)


def _normalize_yes_no(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"yes", "true", "1"}:
        return "Yes"
    if lowered in {"no", "false", "0"}:
        return "No"
    return value


def _auto_elevate_verification_error_enabled() -> bool:
    """Whether to allow substring-driven elevation of verification_error.

    Default OFF: trust the judge LLM's explicit verdict. Set
    AGDEBUGGER_AUTO_ELEVATE_VERIFICATION_ERROR=1 to restore the legacy behaviour.
    """
    value = os.environ.get("AGDEBUGGER_AUTO_ELEVATE_VERIFICATION_ERROR", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_judgment_payload(payload: dict) -> dict:
    normalized = dict(payload)
    reason = str(normalized.get("reason", "")).strip()
    hallucination = _normalize_yes_no(str(normalized.get("hallucination", "")))
    verification_error = _normalize_yes_no(str(normalized.get("verification_error", "")))

    # Write the normalized Yes/No values back FIRST so the elevation step below
    # operates on (and overrides) the canonicalized fields. Previously the
    # elevation set normalized["verification_error"] = "Yes" only to be
    # clobbered a few lines later by the unconditional re-assignment from the
    # un-elevated local variable.
    if hallucination:
        normalized["hallucination"] = hallucination
    if verification_error:
        normalized["verification_error"] = verification_error

    if (
        _auto_elevate_verification_error_enabled()
        and reason
        and _REASON_SUPPORTS_REPAIR_RE.search(reason)
        and not _REASON_NEGATION_RE.search(reason)
        and verification_error != "Yes"
        and hallucination != "Yes"
    ):
        normalized["verification_error"] = "Yes"

    return normalized


class ClaimJudge:
    def __init__(self, llm: OpenAICompatibleLLM, strategy: ClaimJudgeStrategy, **kwargs) -> None:
        self.llm = llm
        self.strategy = strategy
        self._judge_kwargs = kwargs  # e.g. question_text for entity-subject checks

    async def judge_claim(self, claim: Claim, evidence: EvidenceBundle | None = None) -> Judgment:
        evidence = evidence or EvidenceBundle()
        payload = await self.llm.complete_json(
            self.strategy.judge_system_prompt,
            self.strategy.build_judge_user_prompt(claim, evidence, **self._judge_kwargs),
        )
        if not isinstance(payload, dict):
            payload = {}
        payload = _normalize_judgment_payload(payload)
        return Judgment(
            claim_id=claim.claim_id,
            conversation_id=claim.conversation_id,
            turn_number=claim.turn_number,
            reference_name=str(payload.get("reference_name", "")),
            reference_grounding=str(payload.get("reference_grounding", "")),
            content_grounding=str(payload.get("content_grounding", "")),
            hallucination=str(payload.get("hallucination", "")),
            abstention=str(payload.get("abstention", "")),
            verification_error=str(payload.get("verification_error", "")),
            concept_true_understanding=str(payload.get("concept_true_understanding", "")),
            reason=str(payload.get("reason", "")),
            raw_response=payload,
        )

    async def judge_claims(
        self,
        claims: Iterable[Claim],
        *,
        evidence: EvidenceBundle | None = None,
        evidence_provider: EvidenceProvider | None = None,
    ) -> List[Judgment]:
        judgments: List[Judgment] = []
        for claim in claims:
            claim_evidence = evidence or EvidenceBundle()
            if evidence_provider is not None:
                provided = evidence_provider(claim)
                claim_evidence = await provided if inspect.isawaitable(provided) else provided
            judgments.append(await self.judge_claim(claim, claim_evidence))
        return judgments


class PlannerJudge:
    def __init__(self, llm: OpenAICompatibleLLM, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    async def judge_plan(self, user_prompt: str) -> PlannerJudgment:
        payload = await self.llm.complete_json(self.system_prompt, user_prompt)
        if not isinstance(payload, dict):
            payload = {}
        raw_target = payload.get("target_turn_number")
        target_turn_number: int | None = None
        if isinstance(raw_target, bool):
            target_turn_number = None
        elif isinstance(raw_target, int):
            target_turn_number = raw_target
        elif isinstance(raw_target, float):
            target_turn_number = int(raw_target)
        elif isinstance(raw_target, str) and raw_target.strip().lstrip("-").isdigit():
            target_turn_number = int(raw_target.strip())
        return PlannerJudgment(
            selected_claim_id=str(payload.get("selected_claim_id", "") or ""),
            decision=str(payload.get("decision", "") or ""),
            selected_claim_reason=str(payload.get("selected_claim_reason", "") or ""),
            answer_grounding_status=str(payload.get("answer_grounding_status", "") or ""),
            mapping_status=str(payload.get("mapping_status", "") or ""),
            alignment_status=str(payload.get("alignment_status", "") or ""),
            confidence=str(payload.get("confidence", "") or ""),
            repair_concept_name=str(payload.get("repair_concept_name", "") or ""),
            incorrect_understanding=str(payload.get("incorrect_understanding", "") or ""),
            correct_understanding=str(payload.get("correct_understanding", "") or ""),
            target_turn_number=target_turn_number,
            raw_response=payload,
        )
