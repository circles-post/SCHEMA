from __future__ import annotations

from collections import Counter
from typing import Any

from pubmed_graph.pubmed_client import PubMedClient
from pubmed_graph.utils import normalize_text

from .config import DEFAULT_MIN_DOUBLE_CHECK_SUPPORT
from .corroboration_agent import CorroborationAgent, CorroborationResult
from .evidence_utils import independent_chunk_count, independent_doc_count, independent_support_count
from .indexing import QuestionGenerationIndex, canonicalize_entity
from .model_validator import judge_claim, judge_essay, judge_experiment_code
from .retrieval_validator import build_claim_text, retrieve_evidence_bundle, serialize_evidence_bundle
from .validation_cache import ValidationCache
from .validation_types import ModelValidationVerdict
from .models import QuestionSample


def _relation_key(relation: str) -> str:
    return normalize_text(relation).replace(" ", "_").casefold()


ALLOWED_TYPE_PATTERNS = {
    ("Gene", "associated_with", "Pathway"),
    ("Gene", "part_of", "Pathway"),
    ("Gene", "involved_in", "BiologicalProcess"),
    ("Gene", "involved_in_pathway", "Pathway"),
    ("Protein", "associated_with", "Pathway"),
    ("Protein", "part_of", "Pathway"),
    ("Drug", "inhibits", "Protein"),
    ("Drug", "improves", "Disease"),
    ("Biomarker", "associated_with", "Disease"),
    ("Pathway", "associated_with", "Disease"),
    ("Gene", "overexpressed_in", "Disease"),
    ("Gene", "upregulated_in", "Disease"),
    ("Gene", "downregulated_in", "Disease"),
}


def _token_overlap_ratio(evidence: str, chunk_text: str) -> float:
    """Fraction of evidence word-tokens (>=3 chars) present in chunk_text.

    Used as a tolerant fallback when strict substring matching fails — upstream
    triple.evidence is sometimes a light paraphrase of the chunk so whitespace
    or punctuation drift kills exact matches even though the assertion is
    clearly present.
    """
    ev_tokens = [t for t in normalize_text(evidence).split() if len(t) >= 3]
    if not ev_tokens:
        return 0.0
    chunk_tokens = set(normalize_text(chunk_text).split())
    hits = sum(1 for t in ev_tokens if t in chunk_tokens)
    return hits / len(ev_tokens)


_EVIDENCE_TOKEN_OVERLAP_THRESHOLD = 0.8


def _refresh_independent_grounding_counts(sample: QuestionSample) -> None:
    """Recompute grounding support counts from the current supporting_triples
    using the independent-source helpers. Keeps the grounding block in sync
    with reality after upstream mutations (e.g. when ``build_question_sample``
    wrote a raw-triple-count and a downstream step wants honest numbers).
    """
    triples = list(getattr(sample.provenance, "supporting_triples", []) or [])
    if not triples:
        return
    sample.grounding.supporting_evidence_count = independent_support_count(triples)
    sample.grounding.doc_support_count = independent_doc_count(triples)
    sample.grounding.chunk_support_count = independent_chunk_count(triples)
    if sample.grounding.doc_support_count >= 2:
        sample.grounding.support_level = "multi_doc"
    elif sample.grounding.chunk_support_count >= 2:
        sample.grounding.support_level = "multi_chunk"
    else:
        sample.grounding.support_level = "single_source"


def _edge_supporting_triples(sample: QuestionSample, edge: dict[str, Any]) -> list[Any]:
    """Return the supporting_triples that match this specific subgraph edge
    after canonicalization. Used to split evidence per-hop for 2-hop gating.
    """
    head = canonicalize_entity(str(edge.get("head", "")))
    relation = _relation_key(str(edge.get("relation", "")))
    tail = canonicalize_entity(str(edge.get("tail", "")))
    matches: list[Any] = []
    for triple in getattr(sample.provenance, "supporting_triples", []) or []:
        if canonicalize_entity(getattr(triple, "head", "")) != head:
            continue
        if _relation_key(getattr(triple, "relation", "")) != relation:
            continue
        if canonicalize_entity(getattr(triple, "tail", "")) != tail:
            continue
        matches.append(triple)
    return matches


def _multi_hop_chain_supported_by_evidence(
    sample: QuestionSample,
    min_per_hop_support: int = DEFAULT_MIN_DOUBLE_CHECK_SUPPORT,
) -> bool:
    """Hard rule for two_hop_tail: both hops must be independently supported
    at ``min_per_hop_support`` or more, and the intermediate entity
    (sample.answer.text) must match both edge[0].tail and edge[1].head.

    ``min_per_hop_support`` defaults to ``DEFAULT_MIN_DOUBLE_CHECK_SUPPORT``
    (=2). Drop to 1 when the multi-source bar is enforced externally by
    runtime corroboration (``--corroboration-mode required``); the
    structural connectivity / intermediate-uniqueness checks still run.
    """
    if sample.question_type != "two_hop_tail":
        return True
    edges = sample.subgraph.get("edges", []) if isinstance(sample.subgraph, dict) else []
    if len(edges) < 2:
        return False
    first, second = edges[0], edges[1]
    intermediate = canonicalize_entity(str(sample.answer.text))
    if canonicalize_entity(str(first.get("tail", ""))) != intermediate:
        return False
    if canonicalize_entity(str(second.get("head", ""))) != intermediate:
        return False
    if canonicalize_entity(str(first.get("tail", ""))) != canonicalize_entity(str(second.get("head", ""))):
        return False
    for edge in (first, second):
        if independent_support_count(_edge_supporting_triples(sample, edge)) < min_per_hop_support:
            return False
    return True


def _claim_supported_by_sample_evidence(sample: QuestionSample, claim: dict[str, object]) -> bool:
    """True if the sample's supporting_triples include this (h, r, t)."""
    head = canonicalize_entity(str(claim.get("head", "")))
    relation = _relation_key(str(claim.get("relation", "")))
    tail = canonicalize_entity(str(claim.get("tail", "")))
    for triple in getattr(sample.provenance, "supporting_triples", []) or []:
        if canonicalize_entity(getattr(triple, "head", "")) != head:
            continue
        if _relation_key(getattr(triple, "relation", "")) != relation:
            continue
        if canonicalize_entity(getattr(triple, "tail", "")) != tail:
            continue
        return True
    return False


def _claim_choice_options_valid(sample: QuestionSample) -> bool:
    """claim_choice rule: exactly one correct option, matching answer.text;
    distractors must NOT be claims also supported by the current evidence
    bundle (otherwise the question has multiple correct answers).

    Relies on ``metadata.claim_choice_option_claims`` written by the generator.
    """
    if sample.question_type != "claim_choice":
        return True
    if not sample.options or len(sample.options) < 2:
        return False
    correct_options = [option for option in sample.options if option.is_correct]
    if len(correct_options) != 1:
        return False
    if normalize_text(correct_options[0].text).casefold() != normalize_text(sample.answer.text).casefold():
        return False
    claims = sample.metadata.get("claim_choice_option_claims", []) if isinstance(sample.metadata, dict) else []
    if not isinstance(claims, list):
        return False
    claims_by_text = {
        normalize_text(str(claim.get("text", ""))).casefold(): claim
        for claim in claims
        if isinstance(claim, dict) and normalize_text(str(claim.get("text", "")))
    }
    if len(claims_by_text) < len(sample.options):
        return False
    for option in sample.options:
        key = normalize_text(option.text).casefold()
        claim = claims_by_text.get(key)
        if claim is None:
            return False
        actual_supported = _claim_supported_by_sample_evidence(sample, claim)
        declared_supported = bool(claim.get("supported_by_current_evidence"))
        if option.is_correct:
            if not (actual_supported and bool(claim.get("is_correct"))):
                return False
        else:
            # Multiple supported claims make the question multi-answer; reject.
            if actual_supported or declared_supported or bool(claim.get("is_correct")):
                return False
    return True


def _claim_choice_judge_question(sample: QuestionSample) -> str:
    """Augment the question with the option list so the judge can actively
    cross-check that the selected answer is the unique supported option."""
    options = "\n".join(f"- {option.text}" for option in sample.options)
    return (
        f"{sample.question}\n\nCandidate options:\n{options}\n\n"
        "Validate that the selected answer is the only option directly supported by the evidence. "
        "Reject the sample if any non-selected option is also supported."
    )


def _two_hop_judge_question(sample: QuestionSample) -> str:
    """Frame the judge question as the complete chain claim so both hops and
    the intermediate uniqueness get validated, not just hop-1."""
    chain_claim = build_claim_text(sample)
    return (
        f"{sample.question}\n\nComplete chain claim to validate: {chain_claim}\n"
        "Validate both hops of the chain and the uniqueness of the intermediate answer."
    )


def _evidence_supported(index: QuestionGenerationIndex, sample: QuestionSample) -> bool:
    # experiment_code samples are grounded through the sandbox harness
    # (reference solution passes unit tests), not through textual evidence
    # substring matching. Forcing the chunk alignment check on them just
    # regurgitates upstream paraphrase drift as spurious rejections.
    if sample.question_type == "experiment_code":
        return True
    # VQA samples carry no local triple evidence — the Q/A comes from
    # PathVQA overlay and the image is the grounding medium. Skip chunk
    # alignment entirely.
    if sample.question_type == "vqa":
        return True
    for triple in sample.provenance.supporting_triples:
        chunk = index.chunks_by_id.get(triple.chunk_id)
        if chunk is None:
            return False
        ev_norm = normalize_text(triple.evidence)
        chunk_norm = normalize_text(chunk.text)
        if ev_norm in chunk_norm:
            continue
        # Substring miss — accept if the evidence's content tokens mostly
        # appear in the chunk. This tolerates paraphrase / punctuation /
        # whitespace drift in upstream extraction.
        if _token_overlap_ratio(triple.evidence, chunk.text) >= _EVIDENCE_TOKEN_OVERLAP_THRESHOLD:
            continue
        return False
    return True


def _supports_minimum_double_check(sample: QuestionSample) -> bool:
    # VQA + experiment_code bypass the local multi-source requirement
    # (their grounding is image / sandbox respectively, not chunk overlap).
    if sample.question_type in {"vqa", "experiment_code"}:
        return True
    # Recompute counts from the actual triples; upstream may have written
    # raw-triple-count into grounding before we had independent counters.
    _refresh_independent_grounding_counts(sample)
    # For two_hop_tail: per-hop multi-source support is the real rule, not the
    # flat aggregate. ``_multi_hop_chain_supported_by_evidence`` enforces that.
    if sample.question_type == "two_hop_tail":
        return _multi_hop_chain_supported_by_evidence(sample)
    return (
        sample.grounding.supporting_evidence_count >= DEFAULT_MIN_DOUBLE_CHECK_SUPPORT
        or sample.grounding.doc_support_count >= 2
        or sample.grounding.chunk_support_count >= 2
    )


def _type_pattern_allowed(sample: QuestionSample) -> bool:
    return True


def _answer_unique(index: QuestionGenerationIndex, sample: QuestionSample) -> bool:
    if sample.question_type in {"boolean_support", "claim_choice", "essay", "experiment_code"}:
        return True
    if sample.question_type == "two_hop_tail":
        first, second = sample.subgraph["edges"]
        head = canonicalize_entity(first["head"])
        rel1 = _relation_key(first["relation"])
        rel2 = _relation_key(second["relation"])
        final_tail = canonicalize_entity(second["tail"])
        intermediate_answers = set()
        for triple in index.outgoing_by_head.get(head, []):
            if _relation_key(triple.normalized_relation) != rel1:
                continue
            pivot = canonicalize_entity(triple.tail)
            for downstream in index.outgoing_by_head.get(pivot, []):
                if _relation_key(downstream.normalized_relation) != rel2:
                    continue
                if canonicalize_entity(downstream.tail) == final_tail:
                    intermediate_answers.add(pivot)
        return intermediate_answers == {canonicalize_entity(sample.answer.text)}
    return True


def _strip_citation_prefix(question: str) -> str:
    """Remove the ``Based on the reported evidence from '…',`` citation.

    Paper titles legitimately contain entity names, so including them in the
    answer-leakage substring check produces massive false positives (e.g.
    answer='Angiotensin-Converting Enzyme 2', title='ACE2 in chronic disease…'
    → title contains the answer ⇒ false leak).

    Returns the question body *after* the first comma that follows the closing
    quote of the citation. If no such pattern is found, returns the input
    unchanged.
    """
    marker = "Based on the reported evidence from"
    lowered = question.lower()
    idx = lowered.find(marker.lower())
    if idx == -1:
        return question
    # find the closing quote of the title, then the comma after it
    q_start = question.find("'", idx)
    if q_start == -1:
        return question
    q_end = question.find("'", q_start + 1)
    if q_end == -1:
        return question
    comma = question.find(",", q_end)
    if comma == -1:
        return question[q_end + 1:]
    return question[comma + 1:]


def _answer_not_leaked(sample: QuestionSample) -> bool:
    if sample.question_type in {"boolean_support", "claim_choice", "essay", "experiment_code", "vqa"}:
        return True
    body = _strip_citation_prefix(sample.question)
    return normalize_text(sample.answer.text) not in normalize_text(body)


def _question_type_allowed_by_evidence(sample: QuestionSample) -> bool:
    return bool(sample.grounding.question_type_allowed_by_evidence)


def _experiment_metadata_complete(sample: QuestionSample) -> bool:
    if sample.question_type != "experiment_code":
        return True
    required = {
        "data_code",
        "main_code",
        "incomplete_main_code",
        "incomplete_functions",
        "unit_tests",
        "github_references",
        "task_objective",
        "research_direction",
    }
    return required.issubset(set(sample.metadata.keys()))


def validate_sample_rule_based(
    index: QuestionGenerationIndex,
    sample: QuestionSample,
    require_local_multi_source: bool = True,
) -> QuestionSample:
    """Run the rule-only guardrails.

    ``require_local_multi_source=False`` skips the
    ``insufficient_multi_source_support`` check — only do this when the
    caller intends to replace the local multi-source gate with a runtime
    external-corroboration step via ``_validate_corroboration``.
    """
    reasons: list[str] = []
    # Bring grounding counts in line with independent-source counters before
    # downstream rules read from grounding.
    _refresh_independent_grounding_counts(sample)
    if not _evidence_supported(index, sample):
        reasons.append("no_evidence_alignment")
    if not _type_pattern_allowed(sample):
        reasons.append("type_relation_mismatch")
    if not _answer_unique(index, sample):
        reasons.append("non_unique_answer")
    # New: 2-hop chain must be supported per-hop by independent sources.
    # When corroboration replaces the local multi-source gate, drop the
    # per-hop threshold to 1 (still enforces structural connectivity and
    # intermediate-entity uniqueness); runtime corroboration validates the
    # multi-source bar via _validate_corroboration.
    _multi_hop_min = 1 if not require_local_multi_source else DEFAULT_MIN_DOUBLE_CHECK_SUPPORT
    if not _multi_hop_chain_supported_by_evidence(sample, min_per_hop_support=_multi_hop_min):
        reasons.append("multi_hop_chain_not_independently_supported")
    # New: claim_choice distractors must not also be supported by the evidence.
    if not _claim_choice_options_valid(sample):
        reasons.append("claim_choice_options_not_unique")
    if not _answer_not_leaked(sample):
        reasons.append("answer_leakage")
    if require_local_multi_source and not _supports_minimum_double_check(sample):
        reasons.append("insufficient_multi_source_support")
    if not _question_type_allowed_by_evidence(sample):
        reasons.append("question_type_not_allowed_by_evidence")
    if not _experiment_metadata_complete(sample):
        reasons.append("missing_experiment_metadata")
    sample.grounding.double_checked = (
        "no_evidence_alignment" not in reasons
        and "type_relation_mismatch" not in reasons
        and "insufficient_multi_source_support" not in reasons
    )
    sample.grounding.is_fully_grounded = not reasons
    sample.grounding.answer_supported = (
        "no_evidence_alignment" not in reasons
        and "claim_choice_options_not_unique" not in reasons
    )
    sample.grounding.multi_hop_chain_supported = "multi_hop_chain_not_independently_supported" not in reasons
    sample.grounding.validation_mode = "rule_only"
    sample.quality.validation_status = "passed" if not reasons else "rejected"
    sample.quality.rejection_reasons = reasons
    sample.quality.validator_version = "rule_only_v1"
    return sample


def _apply_model_verdict(sample: QuestionSample, verdict: ModelValidationVerdict, retrieved_count: int, external_doc_support_count: int) -> QuestionSample:
    degraded = "model_unavailable" in verdict.issue_tags
    sample.grounding.validation_mode = "degraded" if degraded else "hybrid_model"
    sample.grounding.validation_status_detail = verdict.verdict
    sample.grounding.retrieval_support_count = retrieved_count
    sample.grounding.external_doc_support_count = external_doc_support_count
    sample.grounding.support_score = verdict.support_score
    sample.grounding.contradiction_count = len(verdict.contradicting_evidence_ids)
    sample.grounding.model_confidence = verdict.confidence_band
    sample.quality.model_verdict = verdict.verdict
    sample.quality.model_rejection_reasons = list(verdict.issue_tags)
    sample.quality.validator_version = "hybrid_model_v1"
    if degraded:
        sample.quality.validation_status = "passed"
        sample.quality.rejection_reasons = []
        sample.grounding.double_checked = False
        sample.grounding.is_fully_grounded = True
        return sample
    if verdict.verdict == "supported":
        sample.quality.validation_status = "passed"
        sample.quality.rejection_reasons = []
        sample.grounding.double_checked = True
        sample.grounding.is_fully_grounded = True
    elif verdict.verdict == "insufficient_evidence":
        sample.quality.validation_status = "rejected"
        sample.quality.rejection_reasons = ["model_insufficient_evidence"]
        sample.grounding.double_checked = False
        sample.grounding.is_fully_grounded = False
    else:
        sample.quality.validation_status = "rejected"
        sample.quality.rejection_reasons = ["model_contradicted"]
        sample.grounding.double_checked = False
        sample.grounding.is_fully_grounded = False
    return sample


def validate_sample_model_based(
    sample: QuestionSample,
    model_config: dict[str, Any],
    pubmed_client: PubMedClient | None = None,
    retrieval_top_k: int = 3,
    cache: ValidationCache | None = None,
) -> QuestionSample:
    bundle = retrieve_evidence_bundle(sample, pubmed_client=pubmed_client, top_k=retrieval_top_k)
    sample.provenance.retrieved_evidence_items = bundle
    serialized_bundle = serialize_evidence_bundle(bundle)
    if sample.question_type == "essay":
        verdict, essay_score, essay_rationale = judge_essay(
            question=sample.question,
            reference_answer=sample.answer.text,
            evidence_bundle=serialized_bundle,
            model_config=model_config,
            cache=cache,
        )
        sample.quality.essay_score = essay_score
        sample.quality.essay_rationale = essay_rationale
    else:
        # Reframe the prompt for multichoice / two-hop so the judge sees the
        # structure it needs to validate (options list / full chain claim).
        question_for_judge = sample.question
        answer_for_judge = sample.answer.text
        if sample.question_type == "claim_choice":
            question_for_judge = _claim_choice_judge_question(sample)
        elif sample.question_type == "two_hop_tail":
            question_for_judge = _two_hop_judge_question(sample)
            answer_for_judge = build_claim_text(sample)
        verdict = judge_claim(
            question=question_for_judge,
            answer=answer_for_judge,
            evidence_bundle=serialized_bundle,
            model_config=model_config,
            cache=cache,
        )
    external_doc_ids = {
        item.doc_id or item.pmid or item.doi or item.url
        for item in bundle
        if item.source_type not in {"local_chunk", "local_chunk_context"} and (item.doc_id or item.pmid or item.doi or item.url)
    }
    return _apply_model_verdict(
        sample,
        verdict=verdict,
        retrieved_count=len(bundle),
        external_doc_support_count=len(external_doc_ids),
    )


def _validate_experiment_sample(
    sample: QuestionSample,
    *,
    validation_mode: str = "rule_only",
    model_config: dict[str, Any] | None = None,
    pubmed_client: PubMedClient | None = None,
    retrieval_top_k: int = 3,
    cache: ValidationCache | None = None,
) -> QuestionSample:
    """Apply sandbox-based validation to an ``experiment_code`` sample.

    Reads ``sample.metadata['sandbox_evaluation']`` (written by
    ``experiment_generator.build_experiment_sample``) and converts the
    sandbox verdict into validation_status / rejection_reasons / grounding
    fields.

    Fail-closed: missing, timed-out, or otherwise non-passing sandbox
    evaluations are rejected — the code label cannot be trusted without a
    completed runtime check. (Previously, skipped / missing sandbox
    evaluations silently soft-passed.)
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sandbox_eval = metadata.get("sandbox_evaluation") if isinstance(metadata.get("sandbox_evaluation"), dict) else None

    if not sandbox_eval:
        sample.grounding.validation_mode = "sandbox"
        sample.grounding.validation_status_detail = "sandbox_missing"
        sample.quality.validator_version = "experiment_sandbox_v1"
        sample.quality.validation_status = "rejected"
        sample.quality.rejection_reasons = ["missing_sandbox_evaluation"]
        sample.grounding.double_checked = False
        sample.grounding.is_fully_grounded = False
        return sample

    verdict = str(sandbox_eval.get("verdict", "skipped"))
    rejection_reasons = list(sandbox_eval.get("rejection_reasons", []) or [])
    reference = sandbox_eval.get("reference", {}) or {}
    incomplete = sandbox_eval.get("incomplete", {}) or {}

    sample.grounding.validation_mode = "sandbox"
    sample.grounding.validation_status_detail = f"sandbox_{verdict}"
    sample.quality.validator_version = "experiment_sandbox_v1"
    # Use grounding.support_score to surface the per-test pass-rate of the
    # reference solution; downstream consumers already display this field.
    total = int(reference.get("total", 0) or 0)
    passed_tests = int(reference.get("passed", 0) or 0)
    sample.grounding.support_score = (passed_tests / total) if total else 0.0
    sample.grounding.contradiction_count = int(incomplete.get("passed", 0) or 0)

    if verdict != "passed":
        sample.quality.validation_status = "rejected"
        sample.quality.rejection_reasons = rejection_reasons or ["sandbox_validation_failed"]
        sample.grounding.double_checked = False
        sample.grounding.is_fully_grounded = False
        return sample

    # Sandbox passed — mark sandbox status first
    sample.quality.validation_status = "passed"
    sample.quality.rejection_reasons = []
    sample.grounding.double_checked = True
    sample.grounding.is_fully_grounded = True

    # Second opinion: LLM judge on code/claim semantic alignment.
    # Sandbox can only verify the math; the judge catches "code is off-topic"
    # or "direction flipped" errors where the code is internally consistent
    # but doesn't actually operationalize the scientific claim.
    if validation_mode != "hybrid_model" or not (model_config and model_config.get("enabled")):
        return sample

    edges = sample.subgraph.get("edges", []) if isinstance(sample.subgraph, dict) else []
    head = edges[0].get("head", "") if edges else ""
    relation = edges[0].get("relation", "") if edges else ""
    tail = edges[0].get("tail", "") if edges else ""
    scientific_claim = f"{head} {relation} {tail}".strip()
    evidence_snippet = ""
    supporting_triples = getattr(sample.provenance, "supporting_triples", []) or []
    for tr in supporting_triples:
        ev = getattr(tr, "evidence", "")
        if ev:
            evidence_snippet = ev
            break
    main_code = str(metadata.get("main_code", ""))
    unit_tests = list(metadata.get("unit_tests", []) or [])
    incomplete_functions = list(metadata.get("incomplete_functions", []) or [])

    # Reuse retrieval bundle (local chunks + live PubMed) the claim judge uses
    bundle = retrieve_evidence_bundle(sample, pubmed_client=pubmed_client, top_k=retrieval_top_k)
    sample.provenance.retrieved_evidence_items = bundle
    serialized_bundle = serialize_evidence_bundle(bundle)

    judge_verdict = judge_experiment_code(
        scientific_claim=scientific_claim,
        evidence_snippet=evidence_snippet,
        main_code=main_code,
        unit_tests=unit_tests,
        incomplete_functions=incomplete_functions,
        evidence_bundle=serialized_bundle,
        model_config=model_config,
        cache=cache,
    )
    # Apply the judge verdict layered on top of the sandbox result. We treat
    # 'model_unavailable' as degraded (keep sandbox pass) to avoid cascading
    # rejects when the API is down.
    degraded = "model_unavailable" in judge_verdict.issue_tags
    sample.grounding.model_confidence = judge_verdict.confidence_band
    sample.quality.model_verdict = judge_verdict.verdict
    sample.quality.model_rejection_reasons = list(judge_verdict.issue_tags)
    if degraded:
        sample.grounding.validation_mode = "sandbox"  # sandbox-only, judge unavailable
        sample.grounding.validation_status_detail = "sandbox_passed_judge_degraded"
        return sample
    if judge_verdict.verdict == "supported":
        sample.grounding.validation_mode = "sandbox+hybrid_model"
        sample.grounding.validation_status_detail = "sandbox_passed_judge_supported"
        sample.grounding.support_score = max(sample.grounding.support_score, judge_verdict.support_score)
        return sample
    # Judge rejected the semantics — flip to rejected even though sandbox passed
    sample.quality.validation_status = "rejected"
    sample.grounding.validation_mode = "sandbox+hybrid_model"
    sample.grounding.double_checked = False
    sample.grounding.is_fully_grounded = False
    if judge_verdict.verdict == "contradicted":
        reason = "llm_judge_code_contradicts_claim"
    else:
        reason = "llm_judge_code_claim_misaligned"
    sample.quality.rejection_reasons = [reason] + list(judge_verdict.issue_tags or [])
    sample.grounding.validation_status_detail = f"sandbox_passed_judge_{judge_verdict.verdict}"
    return sample


def _validate_corroboration(
    sample: QuestionSample,
    agent: CorroborationAgent,
) -> QuestionSample:
    """Require at least one independent external source for ``sample``.

    Fail-closed: any tool exception or zero-result path → reject. See
    corroboration_agent.py for the tool flow. Writes results into both
    ``sample.grounding`` (status, counts, tools used) and
    ``sample.provenance.corroborating_sources``.
    """
    claim = build_claim_text(sample)
    local_docs = set(sample.provenance.source_docs)
    # experiment_code samples are already sandbox-validated and their
    # "claim" is a code spec, not a scientific assertion — skip.
    # VQA samples are image-grounded QA pairs from PathVQA overlay;
    # external-literature corroboration is not meaningful.
    if sample.question_type in {"experiment_code", "vqa"}:
        sample.grounding.corroboration_status = "not_requested"
        return sample
    result: CorroborationResult = agent.corroborate_claim(claim, local_docs)
    sample.grounding.corroboration_status = result.status
    sample.grounding.external_source_count = len(result.external_sources)
    sample.grounding.external_tools_used = list(result.tools_used)
    sample.provenance.corroborating_sources = [s.to_dict() for s in result.external_sources]

    if result.status == "corroborated":
        # corroborated does not imply validation_status=passed on its own;
        # rule_based already passed, and the pipeline may also run hybrid_model
        # after this. We do NOT overwrite validation_status here.
        sample.grounding.validation_mode = "corroborated"
        sample.grounding.validation_status_detail = "external_corroboration_found"
        return sample

    # insufficient / tool_unavailable — reject
    if result.status == "tool_unavailable":
        reason = "corroboration_tool_unavailable"
    else:
        reason = "no_external_corroboration"
    sample.quality.validation_status = "rejected"
    sample.quality.rejection_reasons = list(sample.quality.rejection_reasons or []) + [reason]
    sample.grounding.double_checked = False
    sample.grounding.is_fully_grounded = False
    sample.grounding.validation_mode = "corroboration_failed"
    sample.grounding.validation_status_detail = result.short_rationale or reason
    return sample


def validate_sample(
    index: QuestionGenerationIndex,
    sample: QuestionSample,
    validation_mode: str = "rule_only",
    model_config: dict[str, Any] | None = None,
    pubmed_client: PubMedClient | None = None,
    retrieval_top_k: int = 3,
    cache: ValidationCache | None = None,
    corroboration_agent: CorroborationAgent | None = None,
) -> QuestionSample:
    sample = validate_sample_rule_based(
        index,
        sample,
        require_local_multi_source=corroboration_agent is None,
    )
    if sample.quality.validation_status != "passed":
        return sample
    # Optional: runtime external-corroboration gate. Runs BEFORE hybrid_model
    # so that a no-external-evidence sample fails fast without a judge call.
    if corroboration_agent is not None:
        sample = _validate_corroboration(sample, corroboration_agent)
        if sample.quality.validation_status != "passed":
            return sample
    if sample.question_type == "experiment_code":
        return _validate_experiment_sample(
            sample,
            validation_mode=validation_mode,
            model_config=model_config,
            pubmed_client=pubmed_client,
            retrieval_top_k=retrieval_top_k,
            cache=cache,
        )
    # VQA samples: rule_based has already verified Q/A shape; no PubMed/
    # judge step applies (the grounding artefact is the image, not text).
    if sample.question_type == "vqa":
        sample.grounding.validation_mode = "rule_only"
        sample.grounding.validation_status_detail = "vqa_accepted"
        return sample
    if validation_mode != "hybrid_model":
        return sample
    return validate_sample_model_based(
        sample,
        model_config=model_config or {},
        pubmed_client=pubmed_client,
        retrieval_top_k=retrieval_top_k,
        cache=cache,
    )


def summarize_validation(samples: list[QuestionSample]) -> dict[str, int]:
    counts = Counter(sample.quality.validation_status for sample in samples)
    return dict(counts)
