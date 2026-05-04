from __future__ import annotations

from typing import Any

from pubmed_graph.models import PaperRecord
from pubmed_graph.pubmed_client import PubMedClient
from pubmed_graph.utils import normalize_text

from .validation_types import RetrievedEvidenceItem


RELATION_QUERY_HINTS = {
    "associated_with": "association",
    "part_of": "pathway membership",
    "involved_in": "biological role",
    "involved_in_pathway": "pathway involvement",
    "activates": "activation",
    "inhibits": "inhibition",
    "improves": "therapeutic effect",
    "promotes": "promotion",
    "overexpressed_in": "overexpression",
    "upregulated_in": "upregulation",
    "downregulated_in": "downregulation",
}


def _relation_text(relation: str) -> str:
    return normalize_text(relation).replace("_", " ")


def _two_hop_chain_claim(sample) -> str:
    edges = sample.subgraph.get("edges", []) if isinstance(sample.subgraph, dict) else []
    if len(edges) < 2:
        return ""
    first, second = edges[0], edges[1]
    rel1 = _relation_text(str(first.get("relation", "")))
    rel2 = _relation_text(str(second.get("relation", "")))
    intermediate = str(first.get("tail", ""))
    return (
        "The evidence supports a complete two-hop chain: "
        f"{first.get('head', '')} {rel1} {intermediate}, and "
        f"{intermediate} {rel2} {second.get('tail', '')}. "
        f"The unique supported intermediate entity is {intermediate}."
    )


def build_claim_text(sample) -> str:
    if sample.question_type == "claim_choice":
        return sample.answer.text
    if sample.question_type == "two_hop_tail":
        # Construct the chain claim from the actual sample edges/answer so the
        # judge validates both hops and the labelled intermediate entity.
        # Metadata claim text may be conservative prose and may omit uniqueness.
        return _two_hop_chain_claim(sample)
    if sample.subgraph.get("edges"):
        edge = sample.subgraph["edges"][0]
        return f"{edge['head']} {edge['relation']} {edge['tail']}"
    return sample.question


def build_claim_query(sample) -> str:
    query_text = normalize_text(sample.metadata.get("query_text", ""))
    if query_text:
        return query_text
    if sample.question_type == "two_hop_tail":
        edges = sample.subgraph.get("edges", []) if isinstance(sample.subgraph, dict) else []
        if len(edges) >= 2:
            first, second = edges[0], edges[1]
            rel1 = normalize_text(first.get("relation", "")).replace(" ", "_").casefold()
            rel2 = normalize_text(second.get("relation", "")).replace(" ", "_").casefold()
            hint1 = RELATION_QUERY_HINTS.get(rel1, rel1.replace("_", " "))
            hint2 = RELATION_QUERY_HINTS.get(rel2, rel2.replace("_", " "))
            return normalize_text(
                f"{first.get('head', '')} {first.get('tail', '')} {second.get('tail', '')} {hint1} {hint2}"
            )
    if sample.subgraph.get("edges"):
        edge = sample.subgraph["edges"][0]
        relation = normalize_text(edge.get("relation", "")).replace(" ", "_").casefold()
        hint = RELATION_QUERY_HINTS.get(relation, relation.replace("_", " "))
        return normalize_text(f"{edge['head']} {edge['tail']} {hint}")
    return normalize_text(sample.question)


def local_evidence_items(sample) -> list[RetrievedEvidenceItem]:
    items: list[RetrievedEvidenceItem] = []
    for triple in sample.provenance.supporting_triples:
        items.append(
            RetrievedEvidenceItem(
                source_type="local_chunk",
                title=triple.doc_id,
                snippet=triple.evidence,
                stance="support",
                doc_id=triple.doc_id,
                pmid=triple.doc_id.replace("PMID:", "") if str(triple.doc_id).startswith("PMID:") else "",
            )
        )
    for chunk in sample.provenance.supporting_chunks:
        items.append(
            RetrievedEvidenceItem(
                source_type="local_chunk_context",
                title=chunk.title,
                snippet=normalize_text(chunk.text)[:1200],
                stance="support",
                doc_id=chunk.doc_id,
                pmid=chunk.doc_id.replace("PMID:", "") if str(chunk.doc_id).startswith("PMID:") else "",
            )
        )
    return items


def _paper_snippet(paper: PaperRecord) -> str:
    parts = [
        normalize_text(paper.title),
        normalize_text(paper.abstract)[:900],
        normalize_text(paper.journal),
    ]
    return " | ".join(part for part in parts if part)


def pubmed_evidence_items(sample, pubmed_client: PubMedClient, top_k: int = 3) -> list[RetrievedEvidenceItem]:
    query = build_claim_query(sample)
    papers: list[PaperRecord] = pubmed_client.fetch_papers(query=query, retmax=top_k)
    items: list[RetrievedEvidenceItem] = []
    local_doc_ids = set(sample.provenance.source_docs)
    for paper in papers:
        doc_id = f"PMID:{paper.pmid}" if paper.pmid else ""
        if doc_id and doc_id in local_doc_ids:
            continue
        snippet = _paper_snippet(paper)
        items.append(
            RetrievedEvidenceItem(
                source_type="pubmed",
                title=paper.title,
                snippet=snippet,
                stance="neutral",
                pmid=paper.pmid,
                doi=paper.doi or "",
                doc_id=doc_id,
            )
        )
    return items


def corroboration_evidence_items(sample) -> list[RetrievedEvidenceItem]:
    """Wrap agent-found external sources into evidence bundle items.

    ``_validate_corroboration`` runs before this and populates
    ``sample.provenance.corroborating_sources`` with whatever
    ``literature_search`` / ``web_search`` returned (filtered against
    local source_docs). Feeding those same items into the judge's
    evidence bundle means the agent's retrieval is actually READ by the
    LLM, not just counted. Marked ``source_type="external_corroboration"``
    so ``_apply_model_verdict`` includes them in
    ``external_doc_support_count`` (same rule: anything not
    ``local_chunk*``).
    """
    items: list[RetrievedEvidenceItem] = []
    raw = getattr(sample.provenance, "corroborating_sources", None) or []
    for src in raw:
        if not isinstance(src, dict):
            continue
        title = str(src.get("title", "") or "")
        snippet_parts = [
            str(src.get("snippet", "") or ""),
            f"venue: {src.get('venue', '')}" if src.get("venue") else "",
            f"year: {src.get('year', '')}" if src.get("year") else "",
            f"tool: {src.get('tool', '')}" if src.get("tool") else "",
        ]
        snippet = " | ".join(p for p in snippet_parts if p)[:1200]
        doi = str(src.get("doi", "") or "")
        url = str(src.get("url", "") or "")
        # Construct a stable doc_id for evidence dedup downstream.
        doc_id = doi or url or title
        items.append(
            RetrievedEvidenceItem(
                source_type="external_corroboration",
                title=title,
                snippet=snippet,
                stance="neutral",   # judge decides whether it actually supports
                doi=doi,
                url=url,
                doc_id=doc_id,
            )
        )
    return items


def retrieve_evidence_bundle(sample, pubmed_client: PubMedClient | None = None, top_k: int = 3) -> list[RetrievedEvidenceItem]:
    items = local_evidence_items(sample)
    # Option A: include the corroboration agent's external sources FIRST
    # (before PubMed) — they're already fetched and filtered to be
    # independent of local docs, and putting them up-front in the
    # bundle surfaces them to the judge prominently.
    items.extend(corroboration_evidence_items(sample))
    if pubmed_client is not None:
        try:
            items.extend(pubmed_evidence_items(sample, pubmed_client=pubmed_client, top_k=top_k))
        except Exception:
            pass
    return items


def serialize_evidence_bundle(items: list[RetrievedEvidenceItem]) -> list[dict[str, Any]]:
    return [item.__dict__ for item in items]
