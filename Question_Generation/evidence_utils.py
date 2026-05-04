"""Utilities for counting evidence *independently* rather than by raw triple count.

LLM extraction can emit multiple triples from the same chunk or even the same
sentence. Counting raw ``len(triples)`` as "support" over-reports: you really
want to count distinct sources (``doc_id``, ``chunk_id``) — otherwise a single
chunk that happened to yield three triples will falsely satisfy
``support_count >= 2`` gates.

The helpers here normalize the counting surface used by ``sampler.py``,
``generator.py``, ``evidence_profiler.py``, ``validator.py``, etc.
"""

from __future__ import annotations

from typing import Any

from pubmed_graph.utils import normalize_text


def evidence_source_key(record: Any) -> tuple[str, str, str]:
    """Return a stable key for one independent evidence source.

    The primary evidence unit is a local chunk, identified by ``(doc_id,
    chunk_id)``. When either id is missing, fall back to the normalized evidence
    text so duplicate extracted triples from the same span do not inflate
    support counts.
    """
    doc_id = normalize_text(str(getattr(record, "doc_id", "") or ""))
    chunk_id = normalize_text(str(getattr(record, "chunk_id", "") or ""))
    if doc_id or chunk_id:
        return (doc_id, chunk_id, "")
    evidence = normalize_text(str(getattr(record, "evidence", "") or ""))
    return ("", "", evidence[:1000])


def independent_support_keys(records: list[Any]) -> set[tuple[str, str, str]]:
    """Unique evidence-source keys for records with at least one usable key."""
    keys: set[tuple[str, str, str]] = set()
    for record in records:
        key = evidence_source_key(record)
        if any(key):
            keys.add(key)
    return keys


def independent_support_count(records: list[Any]) -> int:
    """Count independent supporting evidence sources, not extracted triple rows."""
    return len(independent_support_keys(records))


def independent_doc_count(records: list[Any]) -> int:
    docs = {
        normalize_text(str(getattr(record, "doc_id", "") or ""))
        for record in records
        if normalize_text(str(getattr(record, "doc_id", "") or ""))
    }
    return len(docs)


def independent_chunk_count(records: list[Any]) -> int:
    return len(independent_support_keys(records))
