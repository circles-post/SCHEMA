from __future__ import annotations

from functools import lru_cache

from pubmed_graph.utils import normalize_text

from .evidence_utils import independent_doc_count, independent_support_count
from .indexing import canonicalize_entity

HEDGE_MARKERS = (
    "suggest",
    "suggested",
    "may",
    "might",
    "potential",
    "possibly",
    "associated",
    "association",
    "correlat",
    "linked",
)

# These two sets are the "fallback / hardcoded core" — they ensure the strength
# heuristic works even if the ontology can't be loaded. Stage 4 makes the
# strength lookup also consult `pubmed_graph.ontology.Ontology` so that
# proposer-added extension relations can declare their own strength via a
# `strength:` field in ontology.yaml.
STRONG_RELATIONS = {
    "activates",
    "inhibits",
    "upregulates",
    "downregulates",
    "overexpressed_in",
    "upregulated_in",
    "downregulated_in",
    "improves",
    "promotes",
}

WEAK_RELATIONS = {
    "associated_with",
    "part_of",
}

_STRENGTH_RANK = {"weak": 0, "medium": 1, "strong": 2}


@lru_cache(maxsize=1)
def _ontology_strength_map() -> dict[str, str]:
    """Build {relation_id -> strength} from the active ontology.

    Stage 4: extension relations can declare strength explicitly via a
    `strength: strong|medium|weak` field on each entry in core_relations.
    Falls back to an empty map (i.e. no overrides) if pubmed_graph is not
    importable, e.g. when this module is reused outside the project.
    """
    try:
        from pubmed_graph.ontology import Ontology  # type: ignore
    except Exception:
        return {}
    try:
        onto = Ontology.default()
    except Exception:
        return {}
    out: dict[str, str] = {}
    for rel in onto._data.get("core_relations") or []:
        rid = str(rel.get("id") or "").strip()
        strength = str(rel.get("strength") or "").strip().lower()
        if rid and strength in {"strong", "medium", "weak"}:
            out[rid.casefold()] = strength
    return out


def _relation_key(relation: str) -> str:
    return normalize_text(relation).replace(" ", "_").casefold()


def _relation_strength(relation: str) -> str:
    normalized = _relation_key(relation)
    # 1. ontology-declared strength wins (lets the OntologyProposerAgent
    #    flag a new mechanistic verb as strong without touching this file)
    onto_map = _ontology_strength_map()
    if normalized in onto_map:
        return onto_map[normalized]
    # 2. fallback to the hardcoded core sets
    if normalized in STRONG_RELATIONS:
        return "strong"
    if normalized in WEAK_RELATIONS:
        return "weak"
    return "medium"


def _hedge_score(texts: list[str]) -> float:
    combined = normalize_text(" ".join(texts)).casefold()
    if not combined:
        return 0.0
    hits = sum(1 for marker in HEDGE_MARKERS if marker in combined)
    return min(hits / 3.0, 1.0)


def _triples_for_edge(subgraph, edge) -> list:
    """Return the supporting_triples that match this specific edge after
    canonicalization. Used to split evidence per-hop on two-hop subgraphs."""
    head = canonicalize_entity(edge.head)
    relation = _relation_key(edge.relation)
    tail = canonicalize_entity(edge.tail)
    out = []
    for triple in subgraph.supporting_triples:
        if canonicalize_entity(triple.head) != head:
            continue
        if _relation_key(triple.relation) != relation:
            continue
        if canonicalize_entity(triple.tail) != tail:
            continue
        out.append(triple)
    return out


def _weakest_strength(strengths: list[str]) -> str:
    if not strengths:
        return "medium"
    return min(strengths, key=lambda value: _STRENGTH_RANK.get(value, 1))


def profile_subgraph_evidence(subgraph, *, corroboration_will_run: bool = False) -> dict[str, object]:
    edges = list(getattr(subgraph, "edges", []) or [])
    hop_profiles: list[dict[str, object]] = []

    if edges:
        for edge in edges:
            triples = _triples_for_edge(subgraph, edge)
            evidence_texts = [triple.evidence for triple in triples if normalize_text(triple.evidence)]
            strength = _relation_strength(edge.relation)
            hop_profiles.append(
                {
                    "head": edge.head,
                    "relation": edge.relation,
                    "tail": edge.tail,
                    "relation_strength": strength,
                    "hedge_score": _hedge_score(evidence_texts),
                    "support_count": independent_support_count(triples),
                    "doc_count": independent_doc_count(triples),
                }
            )
    else:
        # Fallback (e.g. VQA subgraphs with no edges): treat as single virtual hop.
        relation = subgraph.prompt_relation.split(" -> ")[0] if " -> " in subgraph.prompt_relation else subgraph.prompt_relation
        evidence_texts = [triple.evidence for triple in subgraph.supporting_triples if normalize_text(triple.evidence)]
        hop_profiles.append(
            {
                "head": getattr(subgraph, "prompt_subject", ""),
                "relation": relation,
                "tail": getattr(subgraph, "target_answer", ""),
                "relation_strength": _relation_strength(relation),
                "hedge_score": _hedge_score(evidence_texts),
                "support_count": independent_support_count(subgraph.supporting_triples),
                "doc_count": independent_doc_count(subgraph.supporting_triples),
            }
        )

    chunk_texts = [chunk.text for chunk in subgraph.supporting_chunks if normalize_text(chunk.text)]
    if hop_profiles and chunk_texts:
        # Keep the existing chunk-context hedge signal, but apply it as a
        # global maximum rather than letting one strong first hop hide a hedged
        # second hop.
        chunk_hedge = _hedge_score(chunk_texts[:1])
        for profile in hop_profiles:
            profile["hedge_score"] = max(float(profile.get("hedge_score", 0.0)), chunk_hedge)

    relation_strengths = [str(profile["relation_strength"]) for profile in hop_profiles]
    relation_strength = _weakest_strength(relation_strengths)
    hedge_score = max((float(profile.get("hedge_score", 0.0)) for profile in hop_profiles), default=0.0)
    hop_support_counts = [int(profile.get("support_count", 0) or 0) for profile in hop_profiles]
    min_hop_support = min(hop_support_counts) if hop_support_counts else 0
    support_count = independent_support_count(subgraph.supporting_triples)
    doc_count = independent_doc_count(subgraph.supporting_triples)

    all_hops_strong = all(strength == "strong" for strength in relation_strengths) and bool(relation_strengths)
    if all_hops_strong and min_hop_support >= 2 and hedge_score < 0.34:
        evidence_strength = "strong"
    elif relation_strength == "weak" or hedge_score >= 0.67 or min_hop_support == 0:
        evidence_strength = "weak"
    else:
        evidence_strength = "medium"

    if evidence_strength == "strong":
        allowed_question_types = ["claim_choice", "boolean_support", "one_hop_tail", "essay", "experiment_code"]
        # two_hop_tail additionally requires an actual 2-hop subgraph AND the
        # edge count to match the hop profile count (no implicit synthesis).
        if (subgraph.question_type == "two_hop_tail" or len(edges) > 1) and len(edges) == len(hop_profiles):
            allowed_question_types.append("two_hop_tail")
    elif evidence_strength == "medium":
        allowed_question_types = ["claim_choice", "boolean_support", "one_hop_tail", "essay", "experiment_code"]
        if corroboration_will_run and (subgraph.question_type == "two_hop_tail" or len(edges) > 1) \
           and len(edges) == len(hop_profiles):
            allowed_question_types.append("two_hop_tail")
    else:
        # Weak evidence: skip fact-recall (one_hop_tail) since the chunk may not
        # actually commit to a specific tail — stick to hedged claim_choice / essay.
        allowed_question_types = ["claim_choice", "essay"]

    preferred_question_type = "claim_choice" if evidence_strength == "weak" else subgraph.question_type
    claim_strength = {
        "strong": "direct_assertion",
        "medium": "contextual_support",
        "weak": "weak_association",
    }[evidence_strength]

    return {
        "relation_strength": relation_strength,
        "hedge_score": hedge_score,
        "evidence_strength": evidence_strength,
        "claim_strength": claim_strength,
        "allowed_question_types": allowed_question_types,
        "preferred_question_type": preferred_question_type,
        "support_count": support_count,
        "doc_count": doc_count,
        "min_hop_support_count": min_hop_support,
        "hop_support_counts": hop_support_counts,
        "hop_profiles": hop_profiles,
    }
