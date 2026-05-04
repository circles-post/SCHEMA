from __future__ import annotations

from pathlib import Path

from pubmed_graph.models import ChunkRecord, TripleRecord
from pubmed_graph.utils import read_jsonl


def load_triples(path: str | Path) -> list[TripleRecord]:
    return [TripleRecord(**row) for row in read_jsonl(path)]


def load_chunks(path: str | Path) -> list[ChunkRecord]:
    return [ChunkRecord(**row) for row in read_jsonl(path)]
