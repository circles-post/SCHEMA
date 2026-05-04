from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid


def make_short_id() -> str:
    return str(uuid.uuid4())[:8]


@dataclass
class ConversationTurn:
    role: str
    content: str
    turn_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    claim_id: str
    conversation_id: int
    turn_number: int
    category: str
    text: str
    source_ref: str = ""
    source_type: str = ""
    original_statement: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    search_results: str = ""
    filtered_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Judgment:
    claim_id: str
    conversation_id: int
    turn_number: int
    reference_name: str = ""
    reference_grounding: str = ""
    content_grounding: str = ""
    hallucination: str = ""
    abstention: str = ""
    verification_error: str = ""
    concept_true_understanding: str = ""
    reason: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerJudgment:
    selected_claim_id: str = ""
    decision: str = ""
    selected_claim_reason: str = ""
    answer_grounding_status: str = ""
    mapping_status: str = ""
    alignment_status: str = ""
    confidence: str = ""
    # PR-2: planner must itself describe the repair target when it decides
    # ``repair``. Without these fields the downstream bridge has to guess what
    # the planner actually wanted to fix, which is how we ended up with
    # ``planner_decision == repair and hallucinated_concepts == []`` halts.
    repair_concept_name: str = ""
    incorrect_understanding: str = ""
    correct_understanding: str = ""
    target_turn_number: int | None = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    conversation_id: int
    claims: List[Claim] = field(default_factory=list)
    judgments: List[Judgment] = field(default_factory=list)
