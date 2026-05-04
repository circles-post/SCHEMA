"""End-to-end smoke verifier.

Reads a pipeline output dir, checks that every phase produced its expected
artifact and that the phase_summary.json reports non-trivial counts. Exits
with non-zero status when any check fails so it can be wired into CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[ok]   {msg}")


def main(out_dir: str) -> None:
    out = Path(out_dir)
    if not out.exists():
        fail(f"output dir does not exist: {out}")

    expected_files = [
        "resolved_seed_keywords.json",
        "expanded_keywords.json",
        "accepted_keywords.json",
        "pubmed_candidates.jsonl",
        "pubmed_candidates_scored.jsonl",
        "pmc_fulltext.jsonl",
        "chunks.jsonl",
        "phase_summary.json",
        "normalized_triples.jsonl",
        "raw_triples.jsonl",
    ]
    for name in expected_files:
        path = out / name
        if not path.exists():
            fail(f"missing artifact: {path}")
        ok(f"found {name}")

    summary = json.loads((out / "phase_summary.json").read_text())

    p1 = summary.get("phase_1_keyword_expansion", {})
    if int(p1.get("accepted_terms", 0) or 0) <= 0:
        fail(f"phase_1: no accepted keywords ({p1})")
    ok(f"phase_1 accepted_terms = {p1.get('accepted_terms')}")

    p2 = summary.get("phase_2_retrieval_and_filtering", {})
    if int(p2.get("kept_papers", 0) or 0) <= 0:
        fail(f"phase_2: no kept_papers ({p2})")
    ok(f"phase_2 kept_papers = {p2.get('kept_papers')} / unique = {p2.get('retrieved_unique_papers')}")

    p3 = summary.get("phase_3_fulltext", {})
    total_ft = int(p3.get("fulltext_records", 0) or 0) + int(p3.get("abstract_only_records", 0) or 0)
    if total_ft <= 0:
        fail(f"phase_3: no fulltext or abstract records ({p3})")
    ok(
        f"phase_3 fulltext_records = {p3.get('fulltext_records')}, "
        f"abstract_only = {p3.get('abstract_only_records')}, cache_hits = {p3.get('cache_hits')}"
    )

    chunk_count = sum(1 for _ in (out / "chunks.jsonl").open())
    if chunk_count <= 0:
        fail("phase_3: chunks.jsonl is empty")
    ok(f"chunks.jsonl rows = {chunk_count}")

    p4 = summary.get("phase_4_triple_extraction")
    if not p4:
        fail("phase_4_triple_extraction missing from summary")
    raw_t = int(p4.get("raw_triples", 0) or 0)
    norm_t = int(p4.get("normalized_triples", 0) or 0)
    err = int(p4.get("errors", 0) or 0)
    print(f"[info] phase_4 raw={raw_t} normalized={norm_t} errors={err}")
    if err > 0 and raw_t == 0:
        fail(f"phase_4: extraction yielded zero triples and {err} errors — likely API/credentials issue")
    if raw_t == 0:
        print("[warn] phase_4: zero raw triples — check whether chunks contain real body text and that the LLM key is valid")

    p5 = summary.get("phase_5_graph_export")
    graphml = out / "global_graph.graphml"
    if not p5:
        fail("phase_5_graph_export missing from summary")
    if not graphml.exists():
        fail(f"phase_5: GraphML not found at {graphml}")
    nodes = int(p5.get("global_nodes", 0) or 0)
    edges = int(p5.get("global_edges", 0) or 0)
    print(f"[info] phase_5 nodes={nodes} edges={edges}")
    if norm_t > 0 and (nodes == 0 or edges == 0):
        fail("phase_5: triples present but graph is empty — fusion/prune dropped everything")
    ok(f"GraphML written: {graphml}")

    print()
    print("=" * 60)
    print("SMOKE TEST OK — pipeline reachable end-to-end.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/smoke_check.py <output_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
