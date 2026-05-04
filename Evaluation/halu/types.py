"""Dataclasses for the hallucination-detection pipeline.

Everything downstream (extractor, evidence, judge, aggregator) passes these
around. Keep them serializable — all consumers eventually flush to JSONL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Verdict = Literal["supported", "refuted", "unverifiable"]


@dataclass
class Step:
    """One agent utterance we want to scan for claims.

    Built from a trajectory.jsonl message after filtering out tool_call_execution
    entries. ``source`` is typically ``"EvalAgent"`` for normal text / thought
    messages, but may also be the agent name on a ``tool_call_request`` whose
    truncated arguments we still inspect.
    """

    sample_id: str
    step_idx: int       # stable id within this sample's trajectory
    source: str
    msg_type: str       # "text" | "tool_call_request" | "tool_call_summary" | ...
    text: str


@dataclass
class Claim:
    """One atomic factual assertion extracted from a Step."""

    sample_id: str
    step_idx: int
    text: str
    concept: str             # surface form, as the model wrote it
    canonical_concept: str   # normalize_keyword()-ed, pre-clustering
    claim_type: str = "factual"  # extractor emits factual|procedural|meta — we keep factual only


@dataclass
class Evidence:
    """One snippet returned by a layer of the evidence chain."""

    source: Literal["supporting_chunk", "graph_1hop", "web", "literature"]
    text: str
    url: str = ""           # non-empty for web/literature layers
    score: float = 0.0      # layer-specific relevance; layer-1 chunks default to 1.0 (gold)


@dataclass
class ConceptBucket:
    """All claims about one canonical concept within a single sample."""

    sample_id: str
    canonical_concept: str
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    evidence_source_used: str = ""   # which chain layer short-circuited
    concept_weight: float = 1.0      # 1 + log(1+deg) from graph; 1.0 on miss
    concept_type: str = ""           # graph node_type (e.g. "Protein", "Disease"); "" on miss


@dataclass
class JudgedClaim:
    """Post-judge verdict for a single Claim in a ConceptBucket.

    ``score`` is derived deterministically from ``verdict``:
      supported→0.0, unverifiable→0.5, refuted→1.0.
    """

    claim: Claim
    verdict: Verdict
    score: float
    rationale: str = ""
    evidence_quote: str = ""
    evidence_source_used: str = ""
    concept_weight: float = 1.0    # copied from ConceptBucket for HS_weighted aggregation
    concept_type: str = ""         # copied from ConceptBucket for by_concept_type slicing


@dataclass
class SampleRecord:
    """Bundle produced by ``halu.io.load_joined`` per (model, sample_id).

    Consumers never load answers.jsonl / scored_results.jsonl / dataset.jsonl
    / trajectory.jsonl directly — they ask io.py for ``SampleRecord`` objects.
    """

    sample_id: str
    model: str
    sample: dict[str, Any]       # raw dataset row (has provenance.supporting_chunks + subgraph + metadata)
    answer: dict[str, Any]       # answers.jsonl row
    scored: dict[str, Any]       # scored_results.jsonl row — is_correct, score, error
    trajectory_messages: list[dict[str, Any]]   # the "messages" list from trajectory.jsonl
    prompt: str = ""

    @property
    def is_correct(self) -> bool:
        return bool(self.scored.get("is_correct"))

    @property
    def question_type(self) -> str:
        return str(self.sample.get("question_type", ""))


@dataclass
class HaluResult:
    """Per-sample output: what ends up in halu_results.jsonl.

    Phase-1 fields: the core HR/HF for overall reporting. Phase-2 adds HS,
    HS_weighted, and the per-slice buckets by consuming ``.judged_claims``.
    """

    sample_id: str
    model: str
    question_type: str = ""
    tier: str = "not_tagged"
    weight: float = 1.0
    corroboration_status: str = "not_requested"
    evidence_strength: str = "unknown"
    is_correct: bool = False

    n_claims: int = 0
    n_refuted: int = 0
    n_unverifiable: int = 0
    n_supported: int = 0

    HR: float = 0.0
    HS: float = 0.0
    HS_weighted: float = 0.0
    HF: int = 0

    judged_claims: list[JudgedClaim] = field(default_factory=list)
    concept_buckets: list[ConceptBucket] = field(default_factory=list)
    error: str = ""
