from __future__ import annotations

import os
import re
from typing import Iterable, List

from external_agent.llm import OpenAICompatibleLLM
from external_agent.schemas import Claim, ConversationTurn, make_short_id
from external_agent.strategies import ClaimJudgeStrategy


# Any claim whose source text matches one of these shapes almost certainly
# originated from a tool_result payload that leaked past the adapter scrubber
# (``literature_search``/``sciverse_fetch_markdown``/etc.). The judge verifies
# these as "grounded" because the sentences are real paper abstracts, which
# leaves the repair pipeline with zero hallucinated concepts to work with.
# Drop them before they reach the judge.
_TOOL_RESULT_SIGNATURE_RE = re.compile(
    r"(?i)(?:"
    r"literature\s+results\s+for:"
    r"|sciverse\s+workflow\s+result"
    r"|\bDOI:\s*(?:N/?A|10\.)"
    r"|\bVenue:\s*N/?A"
    r"|\bAuthors:\s*[A-Z][a-z]"
    r"|use this tool for scholarly papers"
    r"|^\s*\[\{['\"]content['\"]"
    r"|^\s*\{['\"]content['\"]\s*:"
    r")"
)

_MAPPING_CATEGORIES = {"mapping_claim", "answer_alignment_claim", "constraint_claim"}

# Cues for synthesizing a safety-net ``answer_alignment_claim`` when the
# extractor produces zero mapping/alignment claims for a turn. These phrases
# mark the "last hop" from evidence to answer commitment that the planner
# needs to flag when the chosen option is wrong.
_ANSWER_COMMITMENT_RE = re.compile(
    r"(?im)^[^\n]*\b(?:"
    r"final\s+answer"
    r"|the\s+(?:most|best)\s+(?:likely|plausible|appropriate|scientifically)\s+(?:answer|option|choice|explanation)"
    r"|therefore[, ].{0,80}(?:answer|option|choice)"
    r"|thus[, ].{0,80}(?:answer|option|choice)"
    r"|the\s+(?:correct|best)\s+(?:answer|option|choice)"
    r"|i\s+(?:choose|select|pick|conclude)"
    r"|my\s+(?:answer|choice|selection)\s+is"
    r")\b[^\n]*"
)


def _looks_like_tool_result(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if len(text) < 40:
        return False
    return bool(_TOOL_RESULT_SIGNATURE_RE.search(text))


class ClaimExtractor:
    def __init__(
        self,
        llm: OpenAICompatibleLLM,
        strategy: ClaimJudgeStrategy,
        min_content_length: int = 20,
        max_claims_per_turn: int = 1,
        max_total_claims: int = 1,
    ) -> None:
        if strategy.name == "scientific_concept_discovery":
            if max_claims_per_turn == 1:
                max_claims_per_turn = int(os.environ.get("AGDEBUGGER_CONCEPT_MAX_CLAIMS_PER_TURN", "12"))
            if max_total_claims == 1:
                max_total_claims = int(os.environ.get("AGDEBUGGER_CONCEPT_MAX_TOTAL_CLAIMS", "24"))
        self.extraction_mode = os.environ.get("AGDEBUGGER_CONCEPT_EXTRACTION_MODE", "hybrid").strip().lower()
        self.llm = llm
        self.strategy = strategy
        self.min_content_length = min_content_length
        self.max_claims_per_turn = max(1, int(max_claims_per_turn))
        self.max_total_claims = max(1, int(max_total_claims))

    def _dedupe_claims(self, claims: List[Claim]) -> List[Claim]:
        deduped: List[Claim] = []
        seen: set[tuple[str, str, int]] = set()
        for claim in claims:
            signature = (
                claim.source_ref.strip().lower(),
                claim.text.strip().lower(),
                int(claim.turn_number),
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(claim)
        return deduped

    def _heuristic_extract_from_turn(
        self,
        content: str,
        *,
        conversation_id: int,
        turn_number: int,
        turn_metadata: dict | None = None,
    ) -> List[Claim]:
        if self.strategy.name != "scientific_concept_discovery":
            return []

        keywords = (
            # Scientific concept keywords
            "mn2", "mg2", "mn²", "mg²", "radius", "atomic radius", "binding", "affinity",
            "crystal", "diffraction", "stability", "stable", "conformation", "shape",
            "hydrogen bond", "concentration", "lattice", "packing", "riboswitch",
            "previous studies", "coordination",
            # Option mapping keywords
            "option 1", "option 2", "option 3", "option 4", "option 5", "option 6",
            "option1", "option2", "option3", "option4", "option5", "option6",
            "supports", "rules out", "consistent with", "incompatible",
            "correct answer", "most likely", "most plausible",
            # Entity compatibility keywords
            "amino acid", "nucleotide", "protein", "rna", "dna",
            "aptamer", "peptide", "enzyme", "receptor",
            # Answer alignment keywords
            "therefore", "thus", "final answer", "conclusion", "best explanation",
            "best answer", "addresses the question", "does not answer",
        )
        sentences = re.split(r"(?<=[.!?])\s+", content)
        claims: List[Claim] = []
        for sentence in sentences:
            text = " ".join(sentence.split()).strip()
            lowered = text.lower()
            if len(text) < 25:
                continue
            if not any(keyword in lowered for keyword in keywords):
                continue
            # Skip generic preamble, but preserve answer-grounding / alignment reasoning.
            has_option_ref = any(f"option {i}" in lowered or f"option{i}" in lowered for i in range(1, 7))
            has_answer_commitment = any(
                phrase in lowered
                for phrase in (
                    "therefore",
                    "thus",
                    "the answer is",
                    "final answer",
                    "best explanation",
                    "best answer",
                    "most relevant",
                )
            )
            if not has_option_ref and not has_answer_commitment and lowered.startswith((
                "based on my analysis",
                "therefore",
                "thus",
                "so the answer",
                "the question asks",
                "to determine the most plausible factor",
                "**critical evaluation of options",
                "**experimental design",
            )):
                continue
            if "option 1 reflects methodology" in lowered or "option1 reflects methodology" in lowered:
                continue
            category = "scientific_concept"
            source_type = "scientific_concept"
            source_ref = text[:80]
            if has_option_ref:
                category = "mapping_claim"
                source_type = "answer_grounding"
                source_ref = f"Answer grounding: {text[:60]}"
            elif "amino acid" in lowered and "rna" in lowered:
                category = "constraint_claim"
                source_type = "entity_compatibility"
                source_ref = f"Entity compatibility: {text[:60]}"
            elif has_answer_commitment:
                category = "answer_alignment_claim"
                source_type = "answer_alignment"
                source_ref = f"Answer alignment: {text[:60]}"

            claim = Claim(
                claim_id=make_short_id(),
                conversation_id=conversation_id,
                turn_number=turn_number,
                category=category,
                text=text,
                source_ref=source_ref,
                source_type=source_type,
                original_statement=content,
                metadata={},
            )
            if turn_metadata:
                claim.metadata.update(
                    {
                        "source_timestamp": turn_metadata.get("timestamp"),
                        "analysis_assistant_turn": turn_metadata.get("analysis_assistant_turn"),
                        "source_message_type": turn_metadata.get("type"),
                        "extraction_mode": "heuristic_fallback",
                    }
                )
            claims.append(claim)
        return self._dedupe_claims(claims)[: self.max_claims_per_turn]

    async def extract_from_turn(
        self,
        content: str,
        *,
        conversation_id: int = 0,
        turn_number: int = 0,
        turn_metadata: dict | None = None,
    ) -> List[Claim]:
        if len(content.strip()) < self.min_content_length:
            return []

        if self.extraction_mode == "heuristic":
            return self._heuristic_extract_from_turn(
                content,
                conversation_id=conversation_id,
                turn_number=turn_number,
                turn_metadata=turn_metadata,
            )

        payload = await self.llm.complete_json(
            self.strategy.extractor_system_prompt,
            self.strategy.build_extraction_user_prompt(content),
        )
        if isinstance(payload, dict):
            raw_claims = [payload]
        elif isinstance(payload, list):
            raw_claims = payload
        else:
            return []

        claims: List[Claim] = []
        for raw in raw_claims:
            if not isinstance(raw, dict):
                continue
            raw = dict(raw)
            raw["original_statement"] = content
            claim = self.strategy.normalize_claim(raw, conversation_id, turn_number)
            if claim is None:
                continue
            # PR-1: drop tool_result leak claims before they reach the judge.
            # A claim whose concept/understanding/text looks like a literature
            # dump will pass judging as "grounded" (the text is a real paper
            # sentence) but contributes zero signal toward the actual reasoning
            # error, producing the ``no_repairable_concepts`` halts we saw in
            # comp1/comp2 runs.
            if (
                _looks_like_tool_result(claim.text)
                or _looks_like_tool_result(claim.source_ref)
                or _looks_like_tool_result(str(claim.data.get("context_snippet", "")) if isinstance(claim.data, dict) else "")
            ):
                continue
            if turn_metadata:
                metadata = dict(claim.metadata)
                metadata.update(
                    {
                        "source_timestamp": turn_metadata.get("timestamp"),
                        "analysis_assistant_turn": turn_metadata.get("analysis_assistant_turn"),
                        "source_message_type": turn_metadata.get("type"),
                    }
                )
                claim.metadata = metadata
            claims.append(claim)
        claims = self._dedupe_claims(claims)[: self.max_claims_per_turn]
        if claims:
            return claims
        return self._heuristic_extract_from_turn(
            content,
            conversation_id=conversation_id,
            turn_number=turn_number,
            turn_metadata=turn_metadata,
        )

    def _synthesize_answer_alignment_claim(
        self,
        content: str,
        *,
        conversation_id: int,
        turn_number: int,
        turn_metadata: dict | None,
    ) -> Claim | None:
        """Build a placeholder ``answer_alignment_claim`` so the planner always
        has a mapping-class anchor to select, even when the extractor only
        surfaced factual ``scientific_concept`` claims.

        Without this, any turn whose final commitment lives in a sentence the
        extractor did not pick up halts with ``no_repairable_concepts`` — the
        exact failure mode that dominates the comp1/comp2 logs.
        """
        if not content or not content.strip():
            return None
        # Prefer the last explicit "answer commitment" sentence; fall back to
        # the final non-empty line of the turn (which is usually where the
        # agent commits to its choice).
        commitment_text = ""
        match = None
        for candidate in _ANSWER_COMMITMENT_RE.finditer(content):
            match = candidate
        if match is not None:
            commitment_text = " ".join(match.group(0).split()).strip()
        if not commitment_text:
            for line in reversed(content.splitlines()):
                stripped = line.strip()
                if stripped and len(stripped) > 20:
                    commitment_text = stripped
                    break
        if not commitment_text:
            return None
        if _looks_like_tool_result(commitment_text):
            return None
        # Cap to keep downstream anchoring / judging efficient.
        commitment_preview = commitment_text[:240]
        claim = Claim(
            claim_id=make_short_id(),
            conversation_id=conversation_id,
            turn_number=turn_number,
            category="answer_alignment_claim",
            text=(
                "The agent's final commitment above is treated as the 'last hop' "
                "from the assembled evidence to its chosen option; verify that "
                "the commitment actually addresses the question's requested "
                "target rather than a loosely related fact."
            ),
            source_ref=f"Answer alignment: {commitment_preview}",
            source_type="answer_alignment",
            original_statement=content,
            data={
                "scientific_concept": f"Answer alignment: {commitment_preview}",
                "concept_understanding": (
                    "The agent commits to this option based on the preceding reasoning."
                ),
                "corresponding_action": "Final answer commitment",
                "context_snippet": commitment_text,
                "synthesized_by": "extractor_safety_net",
            },
            metadata={},
        )
        if turn_metadata:
            claim.metadata.update(
                {
                    "source_timestamp": turn_metadata.get("timestamp"),
                    "analysis_assistant_turn": turn_metadata.get("analysis_assistant_turn"),
                    "source_message_type": turn_metadata.get("type"),
                    "extraction_mode": "safety_net_synthesis",
                }
            )
        return claim

    async def extract_from_conversation(
        self,
        turns: Iterable[ConversationTurn],
        *,
        conversation_id: int = 0,
        assistant_only: bool = True,
    ) -> List[Claim]:
        claims: List[Claim] = []
        turns_list = list(turns)
        last_assistant_turn: ConversationTurn | None = None
        for turn in turns_list:
            if assistant_only and turn.role != "assistant":
                continue
            if turn.role == "assistant":
                last_assistant_turn = turn
            if len(claims) >= self.max_total_claims:
                continue
            turn_claims = await self.extract_from_turn(
                turn.content,
                conversation_id=conversation_id,
                turn_number=turn.turn_number,
                turn_metadata=turn.metadata,
            )
            remaining = self.max_total_claims - len(claims)
            claims.extend(turn_claims[:remaining])

        # PR-1 safety net: if none of the extracted claims fall into the
        # mapping/alignment/constraint categories, the planner has no
        # "answer-grounding" hook to flag and strict mode will halt with
        # ``no_repairable_concepts`` even when the judge would otherwise catch
        # a real error. Append a synthesized ``answer_alignment_claim`` so the
        # planner_judge always has at least one mapping-class anchor to pick.
        if (
            self.strategy.name == "scientific_concept_discovery"
            and last_assistant_turn is not None
            and not any(claim.category in _MAPPING_CATEGORIES for claim in claims)
        ):
            synthesized = self._synthesize_answer_alignment_claim(
                last_assistant_turn.content,
                conversation_id=conversation_id,
                turn_number=last_assistant_turn.turn_number,
                turn_metadata=last_assistant_turn.metadata,
            )
            if synthesized is not None:
                claims.append(synthesized)

        return claims[: self.max_total_claims]
