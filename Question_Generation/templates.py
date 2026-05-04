from __future__ import annotations

from pubmed_graph.utils import normalize_text

from .models import SampledSubgraph


RELATION_PHRASES = {
    "associated_with": "is reported to be associated with",
    "part_of": "is described as part of",
    "involved_in": "is reported to be involved in",
    "involved_in_pathway": "is reported to participate in",
    "activates": "is reported to activate",
    "inhibits": "is reported to inhibit",
    "improves": "is reported to improve",
    "promotes": "is reported to promote",
    "overexpressed_in": "is reported to be overexpressed in",
    "downregulated_in": "is reported to be downregulated in",
    "upregulated_in": "is reported to be upregulated in",
    "upregulates": "is reported to upregulate",
    "downregulates": "is reported to downregulate",
}


def _relation_phrase(relation: str, fallback_subject_first: bool = False) -> str:
    normalized = normalize_text(relation).replace(" ", "_").casefold()
    if normalized in RELATION_PHRASES:
        return RELATION_PHRASES[normalized]
    text = normalized.replace("_", " ")
    return f"is reported to {text}" if fallback_subject_first else text


def _context_prefix(subgraph: SampledSubgraph) -> str:
    title = ""
    if subgraph.supporting_chunks:
        title = normalize_text(subgraph.supporting_chunks[0].title)
    if title:
        return f"Based on the reported evidence from '{title}', "
    return "Based on the supporting scientific evidence, "


def render_question(subgraph: SampledSubgraph) -> str:
    prefix = _context_prefix(subgraph)
    edge = subgraph.edges[0]
    evidence_profile = dict(subgraph.metadata.get("evidence_profile", {}))
    evidence_strength = str(evidence_profile.get("evidence_strength", "medium"))
    claim_text = normalize_text(subgraph.metadata.get("claim_text_conservative") or subgraph.metadata.get("claim_text") or "")
    phrase = _relation_phrase(edge.relation, fallback_subject_first=True)
    if subgraph.question_type == "essay":
        edge = subgraph.edges[0]
        if evidence_strength == "weak":
            return (
                f"{prefix}describe the reported relationship between {edge.head} and {edge.tail}, "
                f"and explain what the available evidence indicates about their interaction."
            )
        if evidence_strength == "strong":
            phrase_verb = _relation_phrase(edge.relation, fallback_subject_first=True).removeprefix("is reported to ")
            return (
                f"{prefix}explain the mechanism by which {edge.head} {phrase_verb} {edge.tail}, "
                f"citing the supporting evidence."
            )
        return (
            f"{prefix}based on the provided evidence, explain the relationship between {edge.head} and {edge.tail}, "
            f"and discuss the strength of the supporting findings."
        )
    if subgraph.question_type == "claim_choice":
        if evidence_strength == "weak":
            return f"{prefix}which candidate statement is most cautiously supported by the provided evidence?"
        return f"{prefix}which candidate claim is most directly supported by the provided evidence?"
    if subgraph.question_type == "boolean_support":
        stem = claim_text or f"{edge.head} {phrase.removeprefix('is reported to ')} {edge.tail}"
        return f"{prefix}is the following scientific claim supported by the provided evidence set: {stem}?"
    if subgraph.question_type == "one_hop_tail":
        target_type = (subgraph.target_answer_type or "entity").strip() or "entity"
        predicate = _relation_phrase(edge.relation, fallback_subject_first=True)
        if evidence_strength == "weak":
            return (
                f"{prefix}identify the {target_type.lower()} that "
                f"{edge.head} {predicate} (name a single entity that best fits)."
            )
        return (
            f"{prefix}identify the {target_type.lower()} that "
            f"{edge.head} {predicate}."
        )
    if subgraph.question_type == "two_hop_tail":
        first, second = subgraph.edges
        first_phrase = _relation_phrase(first.relation, fallback_subject_first=True).removeprefix("is reported to ")
        second_phrase = _relation_phrase(second.relation, fallback_subject_first=True).removeprefix("is reported to ")
        if evidence_strength == "weak":
            return (
                f"{prefix}based on the evidence chain alone, which intermediate {subgraph.target_answer_type.lower()} is most plausibly implicated in the statement that "
                f"{first.head} {first_phrase} an intermediate entity, and that intermediate entity {second_phrase} {second.tail}?"
            )
        return (
            f"{prefix}which intermediate {subgraph.target_answer_type.lower()} is best supported by the evidence chain in which "
            f"{first.head} {first_phrase} an intermediate entity, and that intermediate entity {second_phrase} {second.tail}?"
        )
    raise ValueError(f"Unsupported question type: {subgraph.question_type}")
