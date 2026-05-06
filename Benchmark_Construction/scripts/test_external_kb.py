"""Standalone probe for the three EntityCanonicalizer backends.

Calls each backend in isolation on a list of known biomedical terms so we
can identify which one is silently returning [].

Usage:
    NO_PROXY=eutils.ncbi.nlm.nih.gov,id.nlm.nih.gov \\
    python scripts/test_external_kb.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph.external_kb import (  # noqa: E402
    EntityCanonicalizer,
    MeshGroundingBackend,
    PubMedGroundingBackend,
    SciverseGroundingBackend,
)
from pubmed_graph.pubmed_client import PubMedClient  # noqa: E402

QUERIES = [
    "extracellular acidification rate",
    "CRISPR-Cas12a knockout screen",
    "Orbitrap mass spectrometer",
    "p53",
    "BRCA1",
]


def probe(name: str, backend, queries: list[str]) -> None:
    print(f"\n=== {name} ===")
    if not getattr(backend, "enabled", True):
        print("  (disabled)")
        return
    for q in queries:
        try:
            hits = backend.search(q)
        except Exception as exc:
            print(f"  [{q}]")
            print(f"    EXCEPTION: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            continue
        if not hits:
            print(f"  [{q}] -> 0 hits")
            continue
        print(f"  [{q}] -> {len(hits)} hits")
        for h in hits[:2]:
            print(f"    - source={h.source} title={(h.title or '')[:80]} id={h.identifier}")


def main() -> None:
    print("Probing each EntityCanonicalizer backend in isolation.")
    print("Queries:", QUERIES)

    mesh = MeshGroundingBackend({"enabled": True, "limit": 3, "timeout": 15})
    probe("MeshGroundingBackend", mesh, QUERIES)

    pubmed_client = PubMedClient(api_key=None, email="kb_probe@example.com")
    pm = PubMedGroundingBackend({"enabled": True, "retmax": 3}, pubmed_client=pubmed_client)
    probe("PubMedGroundingBackend", pm, QUERIES)

    sv = SciverseGroundingBackend({"enabled": True, "num_results": 3, "timeout_seconds": 60})
    probe("SciverseGroundingBackend", sv, QUERIES)

    # End-to-end via EntityCanonicalizer
    print()
    print("=== EntityCanonicalizer.resolve() end-to-end ===")
    canon = EntityCanonicalizer(
        {
            "min_hits": 1,
            "min_distinct_sources": 1,
            "cache_dir": "/tmp/pubmed_graph_kb_probe_cache",
            "sciverse": {"enabled": True, "num_results": 3, "timeout_seconds": 60},
            "pubmed": {"enabled": True, "retmax": 3},
            "mesh": {"enabled": True, "limit": 3},
        },
        pubmed_client=pubmed_client,
    )
    for q in QUERIES:
        r = canon.resolve(q)
        print(f"  [{q}] grounded={r.grounded} hits={len(r.hits)} sources={sorted(r.sources_with_hits)}")
        if r.rejected_reason:
            print(f"      reason={r.rejected_reason}")


if __name__ == "__main__":
    main()
