from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pubmed_graph.models import ChunkRecord, TripleRecord
from pubmed_graph.utils import normalize_text


@dataclass
class QuestionGenerationIndex:
    triples: list[TripleRecord]
    chunks: list[ChunkRecord]
    chunks_by_id: dict[str, ChunkRecord]
    triples_by_key: dict[tuple[str, str, str], list[TripleRecord]]
    outgoing_by_head: dict[str, list[TripleRecord]]
    incoming_by_tail: dict[str, list[TripleRecord]]
    entities_by_type: dict[str, set[str]]


def canonicalize_entity(text: str) -> str:
    return normalize_text(text).casefold()


def build_index(triples: list[TripleRecord], chunks: list[ChunkRecord]) -> QuestionGenerationIndex:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    triples_by_key: dict[tuple[str, str, str], list[TripleRecord]] = defaultdict(list)
    outgoing_by_head: dict[str, list[TripleRecord]] = defaultdict(list)
    incoming_by_tail: dict[str, list[TripleRecord]] = defaultdict(list)
    entities_by_type: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        key = (
            canonicalize_entity(triple.head),
            normalize_text(triple.normalized_relation).casefold(),
            canonicalize_entity(triple.tail),
        )
        triples_by_key[key].append(triple)
        outgoing_by_head[canonicalize_entity(triple.head)].append(triple)
        incoming_by_tail[canonicalize_entity(triple.tail)].append(triple)
        if triple.head_type:
            entities_by_type[triple.head_type].add(triple.head)
        if triple.tail_type:
            entities_by_type[triple.tail_type].add(triple.tail)
    return QuestionGenerationIndex(
        triples=triples,
        chunks=chunks,
        chunks_by_id=chunks_by_id,
        triples_by_key=dict(triples_by_key),
        outgoing_by_head=dict(outgoing_by_head),
        incoming_by_tail=dict(incoming_by_tail),
        entities_by_type=dict(entities_by_type),
    )
