from __future__ import annotations

from collections import defaultdict

from pubmed_graph.models import TripleRecord

from .evidence_utils import independent_doc_count, independent_support_count
from .indexing import QuestionGenerationIndex, canonicalize_entity
from .models import SampledSubgraph, SubgraphEdge, SubgraphNode, SupportingChunk, SupportingTriple


def aggregate_support(triples: list[TripleRecord], index: QuestionGenerationIndex) -> tuple[list[SupportingTriple], list[SupportingChunk], float]:
    supporting_triples = [
        SupportingTriple(
            doc_id=triple.doc_id,
            chunk_id=triple.chunk_id,
            head=triple.head,
            relation=triple.normalized_relation,
            tail=triple.tail,
            confidence=float(triple.confidence),
            evidence=triple.evidence,
            head_type=triple.head_type,
            tail_type=triple.tail_type,
        )
        for triple in triples
    ]
    # Key by (doc_id, chunk_id) — chunk_id alone can collide across docs
    # (e.g. "chunk_00_0001" appears in many papers). Using just chunk_id
    # silently drops one of two chunks that share the id across docs.
    chunks_by_id: dict[tuple[str, str], SupportingChunk] = {}
    for triple in triples:
        chunk = index.chunks_by_id.get(triple.chunk_id)
        if chunk is None:
            continue
        chunks_by_id[(chunk.doc_id, chunk.chunk_id)] = SupportingChunk(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
        )
    confidence = sum(float(triple.confidence) for triple in triples) / max(len(triples), 1)
    return supporting_triples, list(chunks_by_id.values()), confidence


def build_single_edge_subgraph(
    index: QuestionGenerationIndex,
    triples: list[TripleRecord],
    question_type: str,
) -> SampledSubgraph:
    first = triples[0]
    supporting_triples, supporting_chunks, avg_confidence = aggregate_support(triples, index)
    nodes = [
        SubgraphNode(id=first.head, node_type=first.head_type),
        SubgraphNode(id=first.tail, node_type=first.tail_type),
    ]
    edges = [
        SubgraphEdge(
            head=first.head,
            relation=first.normalized_relation,
            tail=first.tail,
            aggregated_confidence=avg_confidence,
            support_count=independent_support_count(triples),
        )
    ]
    if question_type in {"single_hop_tail", "one_hop_tail"}:
        target_answer = first.tail
        target_type = first.tail_type
        prompt_subject = first.head
    elif question_type == "single_hop_head":
        target_answer = first.head
        target_type = first.head_type
        prompt_subject = first.tail
    elif question_type == "claim_choice":
        target_answer = first.tail
        target_type = first.tail_type
        prompt_subject = first.head
    elif question_type in {"essay", "experiment_code"}:
        target_answer = first.tail
        target_type = first.tail_type
        prompt_subject = first.head
    else:
        target_answer = "yes"
        target_type = "Boolean"
        prompt_subject = first.head
    return SampledSubgraph(
        nodes=nodes,
        edges=edges,
        question_type=question_type,
        target_answer=target_answer,
        target_answer_type=target_type,
        prompt_subject=prompt_subject,
        prompt_relation=first.normalized_relation,
        uniqueness_key="|".join([question_type, canonicalize_entity(first.head), first.normalized_relation, canonicalize_entity(first.tail)]),
        supporting_triples=supporting_triples,
        supporting_chunks=supporting_chunks,
        metadata={
            "support_count": independent_support_count(triples),
            "doc_count": independent_doc_count(triples),
        },
    )


def build_vqa_subgraph(record) -> SampledSubgraph:
    """Build a SampledSubgraph for a VQA record.

    ``record`` is a :class:`question_generation.vqa_source.VqaRecord`. We
    represent the VQA item as a "subgraph" with a single node (the
    self-loop entity) and no edges — the actual Q/A live in
    ``metadata`` and ``supporting_triples`` is empty because VQA samples
    bypass the text-alignment checks in the validator.
    """
    entity = record.entity
    options_hint = "yesno" if record.vqa_format == "yesno" else "open"
    return SampledSubgraph(
        nodes=[SubgraphNode(id=entity, node_type="Image")],
        edges=[],
        question_type="vqa",
        target_answer=record.answer_a,
        target_answer_type="text",
        prompt_subject=entity,
        prompt_relation="vqa",
        uniqueness_key="|".join(["vqa", record.image_key]),
        supporting_triples=[],
        supporting_chunks=[],
        metadata={
            "support_count": 1,
            "doc_count": 1,
            "image_path": record.image_path,
            "image_key": record.image_key,
            "vqa_format": record.vqa_format,
            "vqa_source": record.source,
            "question_q": record.question_q,
            "vqa_doc_id": record.doc_id,
            "vqa_chunk_id": record.chunk_id,
            "vqa_options_hint": options_hint,
        },
    )


def build_two_hop_subgraph(
    index: QuestionGenerationIndex,
    first_hop: list[TripleRecord],
    second_hop: list[TripleRecord],
) -> SampledSubgraph:
    left = first_hop[0]
    right = second_hop[0]
    # Sanity: the two hops must actually share the pivot after canonicalization.
    # Previously the builder silently accepted disconnected "two-hop" chains if
    # raw strings looked close but canonical forms disagreed (casing / whitespace).
    # Callers (sampler.sample_two_hop_subgraphs) now catch ValueError and skip.
    if canonicalize_entity(left.tail) != canonicalize_entity(right.head):
        raise ValueError(
            "two-hop subgraph is not connected after entity canonicalization: "
            f"{left.head!r} -> {left.tail!r} / {right.head!r} -> {right.tail!r}"
        )
    supporting_triples_left, supporting_chunks_left, left_conf = aggregate_support(first_hop, index)
    supporting_triples_right, supporting_chunks_right, right_conf = aggregate_support(second_hop, index)
    chunk_map = {
        (chunk.doc_id, chunk.chunk_id): chunk
        for chunk in [*supporting_chunks_left, *supporting_chunks_right]
    }
    left_support_count = independent_support_count(first_hop)
    right_support_count = independent_support_count(second_hop)
    return SampledSubgraph(
        nodes=[
            SubgraphNode(id=left.head, node_type=left.head_type),
            SubgraphNode(id=left.tail, node_type=left.tail_type),
            SubgraphNode(id=right.tail, node_type=right.tail_type),
        ],
        edges=[
            SubgraphEdge(left.head, left.normalized_relation, left.tail, left_conf, left_support_count),
            SubgraphEdge(right.head, right.normalized_relation, right.tail, right_conf, right_support_count),
        ],
        question_type="two_hop_tail",
        target_answer=left.tail,
        target_answer_type=left.tail_type,
        prompt_subject=left.head,
        prompt_relation=f"{left.normalized_relation} -> {right.normalized_relation}",
        uniqueness_key="|".join([
            "two_hop_tail",
            canonicalize_entity(left.head),
            left.normalized_relation,
            canonicalize_entity(left.tail),
            right.normalized_relation,
            canonicalize_entity(right.tail),
        ]),
        supporting_triples=[*supporting_triples_left, *supporting_triples_right],
        supporting_chunks=list(chunk_map.values()),
        metadata={
            "support_count": independent_support_count([*first_hop, *second_hop]),
            "doc_count": independent_doc_count([*first_hop, *second_hop]),
            "hop_support_counts": [left_support_count, right_support_count],
            "hop_doc_counts": [independent_doc_count(first_hop), independent_doc_count(second_hop)],
        },
    )
