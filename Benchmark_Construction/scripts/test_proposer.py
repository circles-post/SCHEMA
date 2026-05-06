"""Standalone test for OntologyProposerAgent.

Bypasses the full pipeline so we can validate the proposer without
needing PubMed/Crossref retrieval to work. Reads an existing
chunks.jsonl, samples N rows, runs the proposer end-to-end, and
prints a summary.

Usage:
    python scripts/test_proposer.py \
        --chunks benchmark_runs/proteinlmbench_full_graph_v1/chunks.jsonl \
        --output /tmp/pubmed_graph_proposer_test \
        --sample-size 10 \
        --no-grounding   # optional, skips sciverse/pubmed/mesh

Environment: must have OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
exported (i.e. `source ./triple_extraction_env.sh`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph.external_kb import EntityCanonicalizer  # noqa: E402
from pubmed_graph.llm import InternChatClient  # noqa: E402
from pubmed_graph.ontology import Ontology  # noqa: E402
from pubmed_graph.ontology_proposer import OntologyProposerAgent  # noqa: E402
from pubmed_graph.pubmed_client import PubMedClient  # noqa: E402
from pubmed_graph.utils import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", default="/tmp/pubmed_graph_proposer_test")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--evidence-threshold", type=int, default=1)
    parser.add_argument("--distinct-doc-threshold", type=int, default=1)
    parser.add_argument("--max-proposals-per-kind", type=int, default=6)
    parser.add_argument("--no-grounding", action="store_true",
                        help="skip sciverse/pubmed/mesh grounding (faster, less safe)")
    parser.add_argument("--max-chunks", type=int, default=200,
                        help="cap on chunks loaded into memory before sampling")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("INTERN_API_KEY"):
        print("[FAIL] no LLM credentials in env. Run: source ./triple_extraction_env.sh", file=sys.stderr)
        sys.exit(2)

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"[FAIL] chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(2)

    rows = read_jsonl(chunks_path)
    print(f"[info] loaded {len(rows)} chunks from {chunks_path}")
    if len(rows) > args.max_chunks:
        rows = rows[: args.max_chunks]
        print(f"[info] capped to first {args.max_chunks} for memory")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Ontology.default()
    print(f"[info] base ontology version = {base.version}, "
          f"core_relations={len(base.allowed_relations)}, "
          f"entity_types={len(base.allowed_entity_types)}")

    llm_cfg = {
        "model": os.environ.get("OPENAI_MODEL", "intern-s1-pro"),
        "base_url": os.environ.get("OPENAI_BASE_URL"),
        "thinking_mode": False,
        "max_tokens": 1500,
    }
    llm_client = InternChatClient(llm_cfg)

    canonicalizer = None
    if not args.no_grounding:
        try:
            canonicalizer = EntityCanonicalizer(
                {
                    "min_hits": 1,
                    "min_distinct_sources": 1,
                    "cache_dir": "/tmp/pubmed_graph_canonicalizer_cache",
                    "sciverse": {"enabled": True, "num_results": 3, "timeout_seconds": 60},
                    "pubmed": {"enabled": True, "retmax": 2},
                    "mesh": {"enabled": True, "limit": 2},
                },
                pubmed_client=PubMedClient(api_key=None, email="proposer_test@example.com"),
            )
            print("[info] grounding enabled (sciverse + pubmed + mesh)")
        except Exception as exc:
            print(f"[warn] canonicalizer init failed, disabling grounding: {exc}")
            canonicalizer = None

    proposer_cfg = {
        "sample_size": args.sample_size,
        "evidence_threshold": args.evidence_threshold,
        "distinct_doc_threshold": args.distinct_doc_threshold,
        "max_proposals_per_kind": args.max_proposals_per_kind,
        "require_grounding": canonicalizer is not None,
        "cache_dir": "/tmp/pubmed_graph_proposer_cache",
    }
    print(f"[info] proposer config = {proposer_cfg}")
    agent = OntologyProposerAgent(proposer_cfg, llm_client=llm_client, canonicalizer=canonicalizer)

    print("[run]  invoking proposer ...")
    run_ontology = agent.propose(chunks=rows, base=base, output_dir=output_dir)

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  run ontology version : {run_ontology.version}")
    meta = run_ontology._data.get("extensions_metadata", {}) or {}
    print(f"  added entity types   : {meta.get('added_entity_types') or []}")
    print(f"  added relations      : {meta.get('added_relations') or []}")
    print(f"  added aliases        : {meta.get('added_aliases') or []}")

    proposer_dir = output_dir / "ontology_proposer"
    decisions = proposer_dir / "ontology_decisions.jsonl"
    rejected = proposer_dir / "ontology.rejected.jsonl"
    print()
    print(f"  ontology.run.yaml    : {proposer_dir/'ontology.run.yaml'}")
    print(f"  decisions log        : {decisions} ({decisions.stat().st_size if decisions.exists() else 0} bytes)")
    print(f"  rejected log         : {rejected}  ({rejected.stat().st_size if rejected.exists() else 0} bytes)")

    if decisions.exists():
        print()
        print("--- last 5 decisions ---")
        lines = decisions.read_text(encoding="utf-8").splitlines()
        for line in lines[-5:]:
            try:
                d = json.loads(line)
                key = d.get("key", "?")
                kind = d.get("kind", "?")
                decision = d.get("decision", "?")
                reason = d.get("reason", "")[:80]
                print(f"  [{decision:7s}] {kind:13s} {key}  {reason}")
            except Exception:
                print(f"  (malformed: {line[:120]})")

    if rejected.exists():
        print()
        print("--- last 5 rejections ---")
        lines = rejected.read_text(encoding="utf-8").splitlines()
        for line in lines[-5:]:
            try:
                d = json.loads(line)
                key = d.get("key", "?")
                kind = d.get("kind", "?")
                reason = d.get("reason", "")[:120]
                print(f"  [{kind:13s}] {key}  {reason}")
            except Exception:
                print(f"  (malformed: {line[:120]})")

    print()
    print("=" * 70)
    print("OK — proposer ran end-to-end. Inspect ontology.run.yaml for details.")
    print("=" * 70)


if __name__ == "__main__":
    main()
