"""Diagnostic for OntologyProposerAgent.

Two-phase test:
  Phase A — feed the proposer a synthetic chunk that contains concepts the
            base ontology cannot represent (a CRISPR screen + a specific
            mass-spec instrument + a non-bio domain term). If the proposer
            returns 0/0/0 here, the prompt or LLM behavior is broken.
  Phase B — sample 30 chunks from a wide window of the real corpus
            (stratified across sections), run the proposer, and report how
            many chunks produced any proposal at all. This measures
            conservatism on real biomedical content.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph.llm import InternChatClient  # noqa: E402
from pubmed_graph.ontology import Ontology  # noqa: E402
from pubmed_graph.ontology_proposer import OntologyProposerAgent  # noqa: E402
from pubmed_graph.utils import read_jsonl  # noqa: E402

CHUNKS_PATH = ROOT / "benchmark_runs/proteinlmbench_full_graph_v1/chunks.jsonl"


SYNTHETIC_CHUNK = {
    "doc_id": "SYNTHETIC:1",
    "chunk_id": "SYNTHETIC:1::chunk_0",
    "title": "CRISPR-Cas12a screen identifies metabolic vulnerabilities in pancreatic cancer",
    "section": "Methods",
    "text": (
        "We performed a genome-wide CRISPR-Cas12a knockout screen using a custom "
        "lentiviral library targeting 19,500 protein-coding genes in MIA PaCa-2 cells. "
        "Proteomics was performed on a Thermo Orbitrap Eclipse Tribrid mass spectrometer "
        "coupled to a Vanquish Neo nano-flow UHPLC system, and proteins were quantified "
        "using TMTpro 16-plex isobaric tagging. Single-cell ATAC-seq libraries were "
        "constructed with the 10x Chromium Next GEM platform. CRISPR sgRNA enrichment "
        "scores were computed using the MAGeCK MLE algorithm. We additionally measured "
        "extracellular acidification rate (ECAR) using a Seahorse XFe96 bioanalyzer to "
        "phenotype glycolytic dependence in the knockout pools."
    ),
}


def run_phase_a(agent: OntologyProposerAgent, base: Ontology, out_root: Path) -> None:
    print("=" * 70)
    print("PHASE A — synthetic chunk that should trigger proposals")
    print("=" * 70)
    out = out_root / "phase_a"
    out.mkdir(parents=True, exist_ok=True)
    agent.propose(chunks=[SYNTHETIC_CHUNK], base=base, output_dir=out)

    summary = json.loads((out / "ontology_proposer" / "summary.json").read_text())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("--- decisions ---")
    for line in (out / "ontology_proposer" / "ontology_decisions.jsonl").read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("kind") == "proposer_call_ok":
            print(f"  call ok: types={d['n_new_entity_types']} rels={d['n_new_relations']} aliases={d['n_new_entity_aliases']}")
            raw = d.get("raw_response")
            if raw:
                print("  ----- raw LLM response -----")
                print("  " + raw.replace("\n", "\n  "))
                print("  -----------------------------")
        else:
            print(f"  [{d.get('decision', d.get('kind'))}] {d.get('key','')}  {d.get('reason','')[:80]}")
            if d.get("payload"):
                pretty = {k: v for k, v in d["payload"].items() if k in {"id", "rationale", "evidence", "surface", "canonical", "type"}}
                print(f"    payload: {json.dumps(pretty, ensure_ascii=False)}")


def run_phase_b(agent: OntologyProposerAgent, base: Ontology, out_root: Path) -> None:
    print()
    print("=" * 70)
    print("PHASE B — diverse sample from real corpus (30 chunks)")
    print("=" * 70)
    rows = read_jsonl(CHUNKS_PATH)
    rng = random.Random(7)
    section_buckets = {"Abstract": [], "Methods": [], "Results": [], "Discussion": [], "Other": []}
    for row in rows:
        section = (row.get("section") or "").strip().split()[0] if row.get("section") else ""
        bucket = section_buckets.get(section, section_buckets["Other"])
        bucket.append(row)
    sample: list[dict] = []
    for name, bucket in section_buckets.items():
        rng.shuffle(bucket)
        take = min(6, len(bucket))
        sample.extend(bucket[:take])
        print(f"  sampled {take} from section '{name}' (pool size {len(bucket)})")
    print(f"  total sample = {len(sample)}")

    out = out_root / "phase_b"
    out.mkdir(parents=True, exist_ok=True)
    agent.propose(chunks=sample, base=base, output_dir=out)

    summary = json.loads((out / "ontology_proposer" / "summary.json").read_text())
    print()
    print("--- summary ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print("--- per-chunk LLM call breakdown ---")
    nonzero = 0
    for line in (out / "ontology_proposer" / "ontology_decisions.jsonl").read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("kind") == "proposer_call_ok":
            n = d["n_new_entity_types"] + d["n_new_relations"] + d["n_new_entity_aliases"]
            if n > 0:
                nonzero += 1
                print(f"  +{n:2d}  section={d.get('section','?'):20s} title={d.get('title','')[:40]}")
    print(f"  ⇒ {nonzero}/{summary['llm_calls_succeeded']} chunks produced ≥1 proposal")

    print()
    print("--- final accepted/rejected breakdown ---")
    print(f"  raw proposals total : {summary['raw_proposals_total']}")
    print(f"  aggregated         : {summary['aggregated_proposals']}")
    print(f"  rejected (evidence): {summary['rejected_evidence_threshold']}")
    print(f"  rejected (dedup)   : {summary['rejected_dedup']}")
    print(f"  rejected (ground)  : {summary['rejected_grounding']}")
    print(f"  accepted           : {summary['accepted']}")


def main() -> None:
    if not CHUNKS_PATH.exists():
        print(f"[FAIL] {CHUNKS_PATH} not found")
        sys.exit(2)

    base = Ontology.default()
    print(f"base ontology version={base.version}, "
          f"core_relations={len(base.allowed_relations)}, "
          f"entity_types={len(base.allowed_entity_types)}")
    print()

    client = InternChatClient({"model": "intern-s1-pro", "max_tokens": 1500, "thinking_mode": False})

    out_root = Path("/tmp/pubmed_graph_proposer_diag")
    out_root.mkdir(parents=True, exist_ok=True)

    agent = OntologyProposerAgent(
        {
            "sample_size": 100,
            "evidence_threshold": 1,
            "distinct_doc_threshold": 1,
            "max_proposals_per_kind": 8,
            "require_grounding": False,
            "cache_dir": "/tmp/pubmed_graph_proposer_diag_cache",
            "debug_dump_raw": True,
        },
        llm_client=client,
        canonicalizer=None,
    )

    run_phase_a(agent, base, out_root)
    run_phase_b(agent, base, out_root)


if __name__ == "__main__":
    main()
