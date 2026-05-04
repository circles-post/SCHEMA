from __future__ import annotations

from typing import Iterable

from external_agent.claim_extractor import ClaimExtractor
from external_agent.judge import ClaimJudge, EvidenceProvider
from external_agent.schemas import ConversationTurn, EvidenceBundle, PipelineResult


class ClaimJudgePipeline:
    def __init__(self, extractor: ClaimExtractor, judge: ClaimJudge) -> None:
        self.extractor = extractor
        self.judge = judge

    async def run_on_turns(
        self,
        turns: Iterable[ConversationTurn],
        *,
        conversation_id: int = 0,
        evidence: EvidenceBundle | None = None,
        evidence_provider: EvidenceProvider | None = None,
        assistant_only: bool = True,
    ) -> PipelineResult:
        claims = await self.extractor.extract_from_conversation(
            turns,
            conversation_id=conversation_id,
            assistant_only=assistant_only,
        )
        judgments = await self.judge.judge_claims(
            claims,
            evidence=evidence,
            evidence_provider=evidence_provider,
        )
        return PipelineResult(
            conversation_id=conversation_id,
            claims=claims,
            judgments=judgments,
        )

    async def run_on_text(
        self,
        text: str,
        *,
        conversation_id: int = 0,
        turn_number: int = 0,
        evidence: EvidenceBundle | None = None,
        evidence_provider: EvidenceProvider | None = None,
    ) -> PipelineResult:
        claims = await self.extractor.extract_from_turn(
            text,
            conversation_id=conversation_id,
            turn_number=turn_number,
        )
        judgments = await self.judge.judge_claims(
            claims,
            evidence=evidence,
            evidence_provider=evidence_provider,
        )
        return PipelineResult(
            conversation_id=conversation_id,
            claims=claims,
            judgments=judgments,
        )
