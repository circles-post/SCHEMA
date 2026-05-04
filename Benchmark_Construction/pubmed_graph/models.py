from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KeywordRecord:
    term: str
    normalized_term: str
    source: str
    parent_term: str | None = None
    iteration: int = 0
    accepted: bool = True
    notes: str = ""


@dataclass
class PaperRecord:
    pmid: str
    pmcid: str | None
    doi: str | None
    title: str
    abstract: str
    journal: str
    publication_year: str | None
    authors: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    source_queries: list[str] = field(default_factory=list)
    related_pmids: list[str] = field(default_factory=list)
    semantic_score: float = 0.0
    impact_score: float = 0.0
    final_score: float = 0.0
    kept: bool = True
    has_pmc_fulltext: bool = False
    retrieval_source: str = "pubmed"
    preprint_server: str | None = None
    preprint_category: str | None = None
    jats_xml_path: str | None = None
    published_doi: str | None = None
    landing_page_url: str | None = None


@dataclass
class FullTextRecord:
    doc_id: str
    pmid: str | None
    pmcid: str | None
    doi: str | None
    title: str
    journal: str
    publication_year: str | None
    source: str
    has_fulltext: bool
    abstract: str = ""
    text: str = ""
    sections: list[dict[str, str]] = field(default_factory=list)
    raw_path: str | None = None
    parse_status: str = "ok"
    notes: str = ""
    cache_hit: bool = False
    cache_key: str | None = None


@dataclass
class CachedArtifactRef:
    artifact_type: str
    path: str
    source: str
    parse_status: str = "ok"
    parser_backend: str | None = None


@dataclass
class PaperCacheEntry:
    cache_key: str
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    title: str = ""
    source: str = ""
    artifacts: list[CachedArtifactRef] = field(default_factory=list)
    updated_at: str = ""


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    title: str
    section: str
    text: str
    start_offset: int
    end_offset: int


@dataclass
class TripleRecord:
    doc_id: str
    chunk_id: str
    head: str
    head_type: str
    surface_relation: str
    normalized_relation: str
    tail: str
    tail_type: str
    confidence: float
    evidence: str
    source: str = "paper"
    meta: dict[str, Any] = field(default_factory=dict)
