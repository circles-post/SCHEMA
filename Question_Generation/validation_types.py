from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedEvidenceItem:
    source_type: str
    title: str
    snippet: str
    stance: str = "neutral"
    pmid: str = ""
    doi: str = ""
    url: str = ""
    doc_id: str = ""


@dataclass
class ModelValidationVerdict:
    verdict: str
    confidence_band: str = "low"
    support_score: float = 0.0
    contradiction_score: float = 0.0
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    issue_tags: list[str] = field(default_factory=list)
    short_rationale: str = ""
