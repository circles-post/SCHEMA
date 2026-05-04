from __future__ import annotations

from pubmed_graph.utils import normalize_text


def _first_chunk_title(subgraph) -> str:
    return normalize_text(subgraph.supporting_chunks[0].title) if subgraph.supporting_chunks else ""


def _evidence_phrase(subgraph) -> str:
    if not subgraph.supporting_triples:
        return ""
    return normalize_text(subgraph.supporting_triples[0].evidence)


def _relation_text(relation: str) -> str:
    return normalize_text(relation).replace("_", " ")


def _build_two_hop_claim_texts(subgraph, evidence_profile: dict[str, object]) -> dict[str, str]:
    first, second = subgraph.edges[:2]
    title = _first_chunk_title(subgraph)
    evidence = _evidence_phrase(subgraph)
    rel1 = _relation_text(first.relation)
    rel2 = _relation_text(second.relation)
    intermediate = first.tail
    evidence_strength = str(evidence_profile.get("evidence_strength", "medium"))

    if evidence_strength == "weak":
        claim_text = (
            "The evidence weakly suggests a possible two-hop chain in which "
            f"{first.head} {rel1} {intermediate}, and {intermediate} {rel2} {second.tail}."
        )
        conservative = claim_text
    else:
        claim_text = (
            "The available evidence supports a two-hop chain: "
            f"{first.head} {rel1} {intermediate}, and {intermediate} {rel2} {second.tail}. "
            f"The supported intermediate entity is {intermediate}."
        )
        conservative = (
            "The reported evidence indicates the complete chain "
            f"{first.head} {rel1} {intermediate} and {intermediate} {rel2} {second.tail}; "
            f"{intermediate} is the unique supported intermediate entity."
        )
    if evidence:
        claim_text = f"{claim_text} Evidence snippet: {evidence}"
    query_text = normalize_text(f"{first.head} {intermediate} {second.tail} {rel1} {rel2} {title}")
    chain_claim = (
        "The evidence supports a complete two-hop chain: "
        f"{first.head} {rel1} {intermediate}, and {intermediate} {rel2} {second.tail}. "
        f"{intermediate} is the unique supported intermediate entity."
    )
    return {
        "claim_text": claim_text,
        "claim_text_conservative": conservative,
        "query_text": query_text,
        "chain_claim_text": chain_claim,
    }


def build_claim_texts(subgraph, evidence_profile: dict[str, object]) -> dict[str, str]:
    # Two-hop chains need both edges reflected in both the claim and the query,
    # otherwise the judge only validates hop 1 silently.
    if len(getattr(subgraph, "edges", []) or []) >= 2 or subgraph.question_type == "two_hop_tail":
        return _build_two_hop_claim_texts(subgraph, evidence_profile)

    edge = subgraph.edges[0]
    relation = _relation_text(edge.relation)
    title = _first_chunk_title(subgraph)
    evidence = _evidence_phrase(subgraph)
    evidence_strength = str(evidence_profile.get("evidence_strength", "medium"))

    if evidence_strength == "weak":
        claim_text = f"The evidence suggests a reported association involving {edge.head} and {edge.tail}."
        conservative = claim_text
        query_text = normalize_text(f"{edge.head} {edge.tail} association {title}")
    elif evidence_strength == "strong":
        claim_text = f"The available evidence supports that {edge.head} {relation} {edge.tail}."
        conservative = f"The reported evidence indicates that {edge.head} {relation} {edge.tail}."
        query_text = normalize_text(f"{edge.head} {edge.tail} {relation} {title}")
    else:
        claim_text = f"The evidence supports a contextual relationship in which {edge.head} {relation} {edge.tail}."
        conservative = f"The reported evidence suggests that {edge.head} {relation} {edge.tail}."
        query_text = normalize_text(f"{edge.head} {edge.tail} {relation} {title}")

    if evidence:
        claim_text = f"{claim_text} Evidence snippet: {evidence}"
    return {
        "claim_text": claim_text,
        "claim_text_conservative": conservative,
        "query_text": query_text,
    }
