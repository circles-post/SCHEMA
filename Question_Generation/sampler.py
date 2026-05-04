from __future__ import annotations

import logging
import random

from pubmed_graph.utils import normalize_text

from .config import DEFAULT_MAX_SAMPLES, DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_SUPPORT
from .evidence_profiler import profile_subgraph_evidence
from .evidence_utils import independent_support_count
from .experiments import DEFAULT_DIFFICULTY
from .indexing import QuestionGenerationIndex, canonicalize_entity
from .subgraph_builder import build_single_edge_subgraph, build_two_hop_subgraph, build_vqa_subgraph


_logger = logging.getLogger("question_generation.sampler")


def sample_single_hop_subgraphs(
    index: QuestionGenerationIndex,
    question_types: tuple[str, ...],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: int = 7,
    experiment_difficulties: tuple[str, ...] = (DEFAULT_DIFFICULTY,),
    corroboration_will_run: bool = False,
):
    """Sample single-edge subgraph candidates.

    ``min_support`` is the number of independent local chunks that must
    support the ``(head, relation, tail)`` group before it is eligible.
    Set to 1 when ``--corroboration-mode required`` replaces the multi-
    source gate with a runtime external-corroboration check.
    """
    candidates = []
    items = list(index.triples_by_key.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    for _, triples in items:
        if not triples:
            continue
        first = triples[0]
        # Use independent-source count (unique doc/chunk), not raw triple row
        # count: LLM extraction can emit multiple triples from the same chunk,
        # which would falsely satisfy support_count >= 2.
        support_count = independent_support_count(triples)
        avg_conf = sum(float(triple.confidence) for triple in triples) / max(len(triples), 1)
        if avg_conf < min_confidence or support_count < min_support:
            continue
        if "claim_choice" in question_types:
            claim_candidate = build_single_edge_subgraph(index, triples, "claim_choice")
            profile = profile_subgraph_evidence(claim_candidate, corroboration_will_run=corroboration_will_run)
            claim_candidate.metadata["evidence_profile"] = profile
            if "claim_choice" in profile.get("allowed_question_types", []):
                candidates.append(claim_candidate)
        # NOTE: one_hop_tail was removed from DEFAULT_SUPPORTED_QUESTION_TYPES;
        # the builder / template / evidence_profiler entries for it are kept
        # so that legacy JSONL outputs remain readable and scorable. This
        # branch is therefore never exercised by the default CLI.
        if "boolean_support" in question_types:
            boolean_candidate = build_single_edge_subgraph(index, triples, "boolean_support")
            profile = profile_subgraph_evidence(boolean_candidate, corroboration_will_run=corroboration_will_run)
            boolean_candidate.metadata["evidence_profile"] = profile
            if "boolean_support" in profile.get("allowed_question_types", []):
                candidates.append(boolean_candidate)
        if "essay" in question_types:
            essay_candidate = build_single_edge_subgraph(index, triples, "essay")
            profile = profile_subgraph_evidence(essay_candidate, corroboration_will_run=corroboration_will_run)
            essay_candidate.metadata["evidence_profile"] = profile
            if "essay" in profile.get("allowed_question_types", []):
                candidates.append(essay_candidate)
        if "experiment_code" in question_types:
            difficulties = experiment_difficulties or (DEFAULT_DIFFICULTY,)
            for difficulty in difficulties:
                experiment_candidate = build_single_edge_subgraph(index, triples, "experiment_code")
                profile = profile_subgraph_evidence(experiment_candidate, corroboration_will_run=corroboration_will_run)
                experiment_candidate.metadata["evidence_profile"] = profile
                experiment_candidate.metadata["experiment_difficulty"] = difficulty
                # Distinguish difficulties so DEFAULT_MAX_PER_UNIQUENESS_KEY=1
                # does not collapse the easy/medium/hard variants of the same edge.
                experiment_candidate.uniqueness_key = (
                    f"{experiment_candidate.uniqueness_key}|difficulty={difficulty}"
                )
                if "experiment_code" in profile.get("allowed_question_types", []):
                    candidates.append(experiment_candidate)
        if len(candidates) >= max_samples:
            break
    return candidates[:max_samples]


def sample_vqa_subgraphs(
    vqa_records,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: int = 7,
):
    """Convert VQA records into SampledSubgraph candidates.

    The ``profile_subgraph_evidence`` helper is skipped because VQA
    samples carry no local triple evidence. We also mark the evidence
    profile with ``allowed_question_types=["vqa"]`` so downstream
    validator checks do not raise.
    """
    import random as _random

    rng = _random.Random(seed)
    records = list(vqa_records)
    rng.shuffle(records)
    candidates = []
    for record in records[:max_samples]:
        subgraph = build_vqa_subgraph(record)
        subgraph.metadata["evidence_profile"] = {
            "relation_strength": "n/a",
            "hedge_score": 0.0,
            "evidence_strength": "vqa",
            "claim_strength": "direct_assertion",
            "allowed_question_types": ["vqa"],
            "preferred_question_type": "vqa",
            "support_count": 1,
            "doc_count": 1,
        }
        candidates.append(subgraph)
    return candidates


def sample_two_hop_subgraphs(
    index: QuestionGenerationIndex,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: int = 7,
    corroboration_will_run: bool = False,
):
    rng = random.Random(seed)
    keys = list(index.triples_by_key.keys())
    rng.shuffle(keys)
    candidates = []
    for key in keys:
        first_hop = index.triples_by_key[key]
        # Consistency with sample_single_hop_subgraphs: require independent
        # multi-source support on each hop. Using independent_support_count
        # (unique doc/chunk) prevents LLM multi-extraction from spoofing
        # support>=2. Callers wanting the prior lax behaviour can pass
        # min_support=1.
        if not first_hop or independent_support_count(first_hop) < min_support:
            continue
        pivot = first_hop[0].tail
        pivot_key = canonicalize_entity(pivot)
        second_hops = index.outgoing_by_head.get(pivot_key, [])
        if not second_hops:
            continue
        grouped: dict[tuple[str, str, str], list] = {}
        for triple in second_hops:
            # Defensive: must actually start at the pivot after canonicalization.
            if canonicalize_entity(triple.head) != pivot_key:
                continue
            if canonicalize_entity(triple.tail) == canonicalize_entity(first_hop[0].head):
                continue
            relation_key = normalize_text(triple.normalized_relation).replace(" ", "_").casefold()
            group_key = (canonicalize_entity(triple.head), relation_key, canonicalize_entity(triple.tail))
            grouped[group_key] = grouped.get(group_key, []) + [triple]
        for second_group in grouped.values():
            if independent_support_count(second_group) < min_support:
                continue
            left_conf = sum(float(t.confidence) for t in first_hop) / len(first_hop)
            right_conf = sum(float(t.confidence) for t in second_group) / len(second_group)
            if min(left_conf, right_conf) < min_confidence:
                continue
            try:
                candidate = build_two_hop_subgraph(index, first_hop, second_group)
            except ValueError as exc:
                # Disconnected after canonicalization — subgraph_builder now
                # refuses to build a non-connected 2-hop. Skip this group.
                _logger.debug("skipping disconnected 2-hop: %s", exc)
                continue
            profile = profile_subgraph_evidence(candidate, corroboration_will_run=corroboration_will_run)
            candidate.metadata["evidence_profile"] = profile
            if "two_hop_tail" not in profile.get("allowed_question_types", []):
                continue
            candidates.append(candidate)
            if len(candidates) >= max_samples:
                return candidates[:max_samples]
    return candidates[:max_samples]
