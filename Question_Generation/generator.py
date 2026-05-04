from __future__ import annotations

import hashlib
import random

from pubmed_graph.utils import normalize_text

from .config import DEFAULT_DISTRACTOR_COUNT
from .evidence_claims import build_claim_texts
from .evidence_profiler import profile_subgraph_evidence
from .evidence_utils import evidence_source_key, independent_chunk_count, independent_doc_count, independent_support_count
from .experiment_generator import build_experiment_sample
from .indexing import QuestionGenerationIndex, canonicalize_entity
from .models import Answer, Option, QuestionSample, Grounding, Provenance, Quality
from .templates import render_question


def _relation_key(relation: str) -> str:
    return normalize_text(relation).replace(" ", "_").casefold()


def _claim_key(head: str, relation: str, tail: str) -> tuple[str, str, str]:
    return (canonicalize_entity(head), _relation_key(relation), canonicalize_entity(tail))


def _current_evidence_source_keys(subgraph) -> set[tuple[str, str, str]]:
    """Evidence sources backing this specific sample (triples + chunks)."""
    keys = {evidence_source_key(triple) for triple in subgraph.supporting_triples}
    for chunk in subgraph.supporting_chunks:
        keys.add((normalize_text(chunk.doc_id), normalize_text(chunk.chunk_id), ""))
    return {key for key in keys if any(key)}


def _claim_supported_by_current_evidence(subgraph, head: str, relation: str, tail: str) -> bool:
    """True if any supporting_triple matches this (h,r,t) after canonicalization."""
    key = _claim_key(head, relation, tail)
    for triple in subgraph.supporting_triples:
        if _claim_key(triple.head, triple.relation, triple.tail) == key:
            return True
    return False


def _shuffle_options(options: list[Option], sample_id: str, salt: str = "gen_v1") -> list[Option]:
    """Permute ``options`` deterministically from ``sample_id``.

    Keeps correctness markers intact — only the visual position changes.
    Without this every multichoice question has ``is_correct=True`` at
    index 0 because :func:`_build_options` always prepends the correct
    option, which lets a constant-``A`` guesser score 100%.
    """
    if len(options) < 2:
        return options
    digest = hashlib.sha1(f"{salt}::{sample_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    perm = list(range(len(options)))
    rng.shuffle(perm)
    return [options[i] for i in perm]


def _claim_text(head: str, relation: str, tail: str) -> str:
    relation_text = relation.replace("_", " ")
    return f"{head} {relation_text} {tail}"


def _claim_option_text(subgraph, head: str, relation: str, tail: str) -> str:
    evidence_strength = str(subgraph.metadata.get("evidence_profile", {}).get("evidence_strength", "medium"))
    if evidence_strength == "weak":
        return f"The evidence suggests a reported association involving {head} and {tail}."
    if evidence_strength == "strong":
        return f"The available evidence supports that {head} {relation.replace('_', ' ')} {tail}."
    return f"The evidence supports a contextual relationship in which {head} {relation.replace('_', ' ')} {tail}."


def _allowed_question_types(subgraph) -> set[str]:
    profile = subgraph.metadata.get("evidence_profile", {})
    return set(profile.get("allowed_question_types", [subgraph.question_type]))


def _build_options(index: QuestionGenerationIndex, subgraph, answer_text: str, answer_type: str, count: int) -> list[Option]:
    if subgraph.question_type == "claim_choice":
        edge = subgraph.edges[0]
        distractors: list[str] = []
        option_claims: list[dict[str, object]] = []
        head_key = canonicalize_entity(edge.head)
        relation = edge.relation
        correct_key = _claim_key(edge.head, relation, answer_text)
        correct_text = _claim_option_text(subgraph, edge.head, relation, answer_text)
        seen_claim_keys = {correct_key}
        seen_text_keys = {normalize_text(correct_text).casefold()}
        current_source_keys = _current_evidence_source_keys(subgraph)

        option_claims.append(
            {
                "text": correct_text,
                "head": edge.head,
                "relation": relation,
                "tail": answer_text,
                "is_correct": True,
                "supported_by_current_evidence": _claim_supported_by_current_evidence(
                    subgraph, edge.head, relation, answer_text
                ),
            }
        )

        def _try_add_distractor(triple) -> None:
            if len(distractors) >= count:
                return
            if _relation_key(triple.normalized_relation) != _relation_key(relation):
                return
            claim_key = _claim_key(triple.head, relation, triple.tail)
            if claim_key in seen_claim_keys:
                return
            # A distractor must not be another claim supported by the same
            # evidence bundle — else the question has multiple correct answers.
            if _claim_supported_by_current_evidence(subgraph, triple.head, relation, triple.tail):
                return
            # Nor extracted from the exact same chunk/doc as the correct answer.
            if evidence_source_key(triple) in current_source_keys:
                return
            claim = _claim_option_text(subgraph, triple.head, relation, triple.tail)
            text_key = normalize_text(claim).casefold()
            if text_key in seen_text_keys:
                return
            seen_claim_keys.add(claim_key)
            seen_text_keys.add(text_key)
            distractors.append(claim)
            option_claims.append(
                {
                    "text": claim,
                    "head": triple.head,
                    "relation": relation,
                    "tail": triple.tail,
                    "is_correct": False,
                    "supported_by_current_evidence": False,
                }
            )

        # Prefer distractors from the same head first (more plausible confusers),
        # then fall back to arbitrary triples with the same relation. Sort by
        # canonical tail so dedupe is stable.
        for triple in sorted(
            index.outgoing_by_head.get(head_key, []),
            key=lambda item: canonicalize_entity(item.tail),
        ):
            _try_add_distractor(triple)
            if len(distractors) >= count:
                break
        for triple in index.triples:
            _try_add_distractor(triple)
            if len(distractors) >= count:
                break

        # Expose the per-option claim map for the validator's
        # ``_claim_choice_options_valid`` cross-check.
        subgraph.metadata["claim_choice_option_claims"] = option_claims
        options = [Option(text=correct_text, is_correct=True)]
        options.extend(Option(text=text, is_correct=False) for text in distractors[:count])
        return options

    distractors: list[str] = []
    seen_entity_keys = {canonicalize_entity(answer_text)}
    type_pool = sorted(index.entities_by_type.get(answer_type, set()))
    for candidate in type_pool:
        candidate_key = canonicalize_entity(candidate)
        if not candidate_key or candidate_key in seen_entity_keys:
            continue
        seen_entity_keys.add(candidate_key)
        distractors.append(candidate)
        if len(distractors) >= count:
            break
    options = [Option(text=answer_text, is_correct=True)]
    options.extend(Option(text=text, is_correct=False) for text in distractors[:count])
    return options


def _build_vqa_sample(subgraph, *, sample_id: str, distractor_count: int = 3):
    """Minimal build path for VQA samples: no distractor mining, no judge,
    no corroboration. All fields come from ``subgraph.metadata``.
    """
    meta = dict(subgraph.metadata or {})
    question_q = str(meta.get("question_q", "")).strip()
    image_path = str(meta.get("image_path", ""))
    vqa_format = str(meta.get("vqa_format", "open"))
    answer_text = str(subgraph.target_answer or "")

    # Question framing: include the entity for a bit of context but keep it
    # short; the bulk of the task is the Q + image.
    question = (
        f"[VQA — image at {image_path}]\n{question_q}"
        if vqa_format == "open"
        else f"[VQA — image at {image_path}]\n{question_q} (Answer Yes or No.)"
    )

    if vqa_format == "yesno":
        is_yes = answer_text.strip().casefold().rstrip(".!?") == "yes"
        options = [
            Option(text="Yes", is_correct=is_yes),
            Option(text="No", is_correct=not is_yes),
        ]
        answer = Answer(
            text=("Yes" if is_yes else "No"),
            canonical_text=("yes" if is_yes else "no"),
            answer_type="Boolean",
        )
    else:
        options = []
        answer = Answer(
            text=answer_text,
            canonical_text=normalize_text(answer_text),
            answer_type="Text",
        )

    provenance = Provenance(
        supporting_triples=[],
        supporting_chunks=[],
        source_docs=[str(meta.get("vqa_doc_id", ""))] if meta.get("vqa_doc_id") else [],
    )
    grounding = Grounding(
        is_fully_grounded=True,
        answer_supported=True,
        question_entities_supported=True,
        multi_hop_chain_supported=False,
        supporting_evidence_count=1,
        doc_support_count=1,
        chunk_support_count=1,
        double_checked=False,
        support_level="single_source",
        validation_mode="rule_only",
        validation_status_detail="vqa",
        evidence_strength="vqa",
        claim_strength="direct_assertion",
        question_type_allowed_by_evidence=True,
        evidence_profile_version="v1",
    )
    quality = Quality(
        validation_status="pending",
        difficulty=("easy" if vqa_format == "yesno" else "medium"),
        ambiguity_score=0.0,
        uniqueness_key=subgraph.uniqueness_key,
        validator_version="vqa_rule_only_v1",
    )
    return QuestionSample(
        sample_id=sample_id,
        question_type="vqa",
        question=question,
        answer=answer,
        options=options,
        subgraph={
            "nodes": [node.__dict__ for node in subgraph.nodes],
            "edges": [],
        },
        provenance=provenance,
        grounding=grounding,
        quality=quality,
        metadata=meta,
    )


def build_question_sample(
    index: QuestionGenerationIndex,
    subgraph,
    sample_id: str,
    distractor_count: int = DEFAULT_DISTRACTOR_COUNT,
    github_search_per_page: int = 3,
    github_search_language: str = "Python",
    llm_code_selection: str = "auto",
    experiment_generation_mode: str = "template",
    corroboration_will_run: bool = False,
) -> QuestionSample:
    # VQA short-circuit: no edges / no supporting triples, so the
    # evidence_profile + claim_text helpers below would crash on [0] access.
    if subgraph.question_type == "vqa":
        return _build_vqa_sample(subgraph, sample_id=sample_id, distractor_count=distractor_count)
    evidence_profile = profile_subgraph_evidence(subgraph, corroboration_will_run=corroboration_will_run)
    claim_bundle = build_claim_texts(subgraph, evidence_profile)
    subgraph.metadata["evidence_profile"] = evidence_profile
    subgraph.metadata.update(claim_bundle)
    if subgraph.question_type not in _allowed_question_types(subgraph):
        subgraph.question_type = str(evidence_profile.get("preferred_question_type", "claim_choice"))
    if subgraph.question_type == "experiment_code":
        return build_experiment_sample(
            subgraph,
            sample_id=sample_id,
            github_search_per_page=github_search_per_page,
            github_search_language=github_search_language,
            llm_code_selection=llm_code_selection,
            generation_mode=experiment_generation_mode,
        )
    question = render_question(subgraph)
    answer_text = subgraph.target_answer
    answer_type = subgraph.target_answer_type
    if subgraph.question_type == "boolean_support":
        options = [Option(text="Supported", is_correct=True), Option(text="Not supported", is_correct=False)]
        options = _shuffle_options(options, sample_id)
        answer = Answer(text="Supported", canonical_text="supported", answer_type="Boolean")
    elif subgraph.question_type == "essay":
        options = []
        evidence_snippets = [triple.evidence for triple in subgraph.supporting_triples if triple.evidence]
        reference_text = claim_bundle.get("claim_text_conservative", "") or claim_bundle.get("claim_text", "")
        if evidence_snippets:
            reference_text = f"{reference_text} Supporting evidence: {' '.join(evidence_snippets[:2])}"
        answer = Answer(text=reference_text, canonical_text=normalize_text(reference_text), answer_type="Essay")
    elif subgraph.question_type == "claim_choice":
        options = _build_options(index, subgraph, answer_text, answer_type, distractor_count)
        options = _shuffle_options(options, sample_id)
        correct_claim = next((option.text for option in options if option.is_correct), "")
        answer = Answer(text=correct_claim, canonical_text=normalize_text(correct_claim), answer_type="Claim")
    else:
        options = _build_options(index, subgraph, answer_text, answer_type, distractor_count)
        options = _shuffle_options(options, sample_id)
        answer = Answer(text=answer_text, canonical_text=normalize_text(answer_text), answer_type=answer_type)
    provenance = Provenance(
        supporting_triples=subgraph.supporting_triples,
        supporting_chunks=subgraph.supporting_chunks,
        source_docs=sorted({triple.doc_id for triple in subgraph.supporting_triples}),
    )
    doc_support_count = independent_doc_count(subgraph.supporting_triples)
    chunk_support_count = independent_chunk_count(subgraph.supporting_triples)
    evidence_support_count = independent_support_count(subgraph.supporting_triples)
    support_level = "multi_doc" if doc_support_count >= 2 else ("multi_chunk" if chunk_support_count >= 2 else "single_source")
    grounding = Grounding(
        is_fully_grounded=True,
        answer_supported=True,
        question_entities_supported=True,
        # two_hop_tail must actually have ≥2 edges; anything else defaults to
        # True. Previous logic was inverted (True for 1-edge) which let
        # two_hop_tail samples with <2 edges pass this flag unreviewed.
        multi_hop_chain_supported=subgraph.question_type != "two_hop_tail" or len(subgraph.edges) >= 2,
        supporting_evidence_count=evidence_support_count,
        doc_support_count=doc_support_count,
        chunk_support_count=chunk_support_count,
        double_checked=False,
        support_level=support_level,
        validation_mode="rule_only",
        validation_status_detail="unvalidated",
        evidence_strength=str(evidence_profile.get("evidence_strength", "unknown")),
        claim_strength=str(evidence_profile.get("claim_strength", "unknown")),
        question_type_allowed_by_evidence=subgraph.question_type in set(evidence_profile.get("allowed_question_types", [])),
        evidence_profile_version="v1",
    )
    support_count = evidence_support_count
    quality = Quality(
        validation_status="pending",
        difficulty="hard" if subgraph.question_type == "two_hop_tail" else ("medium" if support_count > 1 else "easy"),
        ambiguity_score=0.0,
        uniqueness_key=subgraph.uniqueness_key,
        validator_version="rule_only_v1",
    )
    return QuestionSample(
        sample_id=sample_id,
        question_type=subgraph.question_type,
        question=question,
        answer=answer,
        options=options,
        subgraph={
            "nodes": [node.__dict__ for node in subgraph.nodes],
            "edges": [edge.__dict__ for edge in subgraph.edges],
        },
        provenance=provenance,
        grounding=grounding,
        quality=quality,
        metadata=dict(subgraph.metadata) | {
            "claim_text": claim_bundle["claim_text"],
            "claim_text_conservative": claim_bundle["claim_text_conservative"],
            "query_text": claim_bundle["query_text"],
            "evidence_strength": evidence_profile["evidence_strength"],
            "claim_strength": evidence_profile["claim_strength"],
            "relation_strength": evidence_profile["relation_strength"],
            "hedge_score": evidence_profile["hedge_score"],
            "preferred_question_type": evidence_profile["preferred_question_type"],
            "query_subject": subgraph.edges[0].head if subgraph.edges else "",
            "query_relation": subgraph.edges[0].relation if subgraph.edges else "",
            "query_object": subgraph.edges[0].tail if subgraph.edges else "",
        },
    )
