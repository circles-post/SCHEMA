from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation_types import RetrievedEvidenceItem


@dataclass
class SupportingTriple:
    doc_id: str
    chunk_id: str
    head: str
    relation: str
    tail: str
    confidence: float
    evidence: str
    head_type: str = ""
    tail_type: str = ""


@dataclass
class SupportingChunk:
    doc_id: str
    chunk_id: str
    title: str
    section: str
    text: str


@dataclass
class SubgraphNode:
    id: str
    node_type: str


@dataclass
class SubgraphEdge:
    head: str
    relation: str
    tail: str
    aggregated_confidence: float
    support_count: int


@dataclass
class Answer:
    text: str
    canonical_text: str
    answer_type: str


@dataclass
class Option:
    text: str
    is_correct: bool


@dataclass
class Provenance:
    supporting_triples: list[SupportingTriple] = field(default_factory=list)
    supporting_chunks: list[SupportingChunk] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)
    retrieved_evidence_items: list[RetrievedEvidenceItem] = field(default_factory=list)
    corroborating_sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Grounding:
    is_fully_grounded: bool
    answer_supported: bool
    question_entities_supported: bool
    multi_hop_chain_supported: bool
    supporting_evidence_count: int
    doc_support_count: int = 0
    chunk_support_count: int = 0
    double_checked: bool = False
    support_level: str = "single_source"
    validation_mode: str = "rule_only"
    validation_status_detail: str = "unvalidated"
    retrieval_support_count: int = 0
    external_doc_support_count: int = 0
    contradiction_count: int = 0
    support_score: float = 0.0
    model_confidence: str = "low"
    evidence_strength: str = "unknown"
    claim_strength: str = "unknown"
    question_type_allowed_by_evidence: bool = True
    evidence_profile_version: str = "v1"
    corroboration_status: str = "not_requested"
    external_source_count: int = 0
    external_tools_used: list[str] = field(default_factory=list)


@dataclass
class Quality:
    validation_status: str
    difficulty: str
    ambiguity_score: float
    uniqueness_key: str
    rejection_reasons: list[str] = field(default_factory=list)
    validator_version: str = "rule_only_v1"
    model_verdict: str = ""
    model_rejection_reasons: list[str] = field(default_factory=list)
    validation_trace_id: str = ""
    essay_score: float = 0.0
    essay_rationale: str = ""


@dataclass
class SampledSubgraph:
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]
    question_type: str
    target_answer: str
    target_answer_type: str
    prompt_subject: str
    prompt_relation: str
    uniqueness_key: str
    supporting_triples: list[SupportingTriple] = field(default_factory=list)
    supporting_chunks: list[SupportingChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionSample:
    sample_id: str
    question_type: str
    question: str
    answer: Answer
    options: list[Option]
    subgraph: dict[str, Any]
    provenance: Provenance
    grounding: Grounding
    quality: Quality
    metadata: dict[str, Any] = field(default_factory=dict)
