"""Snapshot the current normalize.py / entity_verification.py behavior into a
baseline directory so the ontology refactor can be diffed against it.

Usage:
    python scripts/collect_baseline.py \
        --triples pipeline_outputs_minimal/raw_triples.jsonl \
        --output  benchmark_runs/baseline_pre_refactor

Notes:
    * This script does NOT call any LLM. It re-runs `normalize_triple_rows`
      against an existing raw_triples.jsonl, captures the canonical output,
      and dumps a stats summary (counts per entity_type / relation, drop
      reasons) so the post-refactor diff can prove behavior is unchanged.
    * If --triples is omitted, the script enumerates the most recent
      pipeline_outputs_*/raw_triples.jsonl and uses the largest one.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph.normalize import normalize_triple_rows  # noqa: E402
from pubmed_graph.utils import read_jsonl, write_jsonl  # noqa: E402


def autodetect_raw_triples() -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in ROOT.glob("pipeline_outputs*/raw_triples.jsonl"):
        try:
            candidates.append((path.stat().st_size, path))
        except OSError:
            continue
    for path in (ROOT / "benchmark_runs").glob("*/raw_triples.jsonl"):
        try:
            candidates.append((path.stat().st_size, path))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="", help="raw_triples.jsonl input")
    parser.add_argument("--output", default="benchmark_runs/baseline_pre_refactor")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args()

    raw_path = Path(args.triples) if args.triples else autodetect_raw_triples()
    if raw_path is None or not raw_path.exists():
        print("[FAIL] no raw_triples.jsonl found. Pass --triples PATH explicitly.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(raw_path)
    print(f"[info] raw rows: {len(raw_rows)} from {raw_path}")

    normalized = normalize_triple_rows(raw_rows, confidence_threshold=args.confidence_threshold)
    write_jsonl(out_dir / "normalized_triples.baseline.jsonl", normalized)
    print(f"[info] normalized rows: {len(normalized)}")

    head_types = Counter(row.get("head_type", "") for row in normalized)
    tail_types = Counter(row.get("tail_type", "") for row in normalized)
    relations = Counter(row.get("normalized_relation", "") for row in normalized)
    surface_relations = Counter(row.get("surface_relation", "") for row in normalized)
    confidence_buckets = Counter()
    for row in normalized:
        c = float(row.get("confidence", 0.0) or 0.0)
        confidence_buckets[round(c, 1)] += 1

    summary = {
        "source_raw_triples_path": str(raw_path),
        "raw_row_count": len(raw_rows),
        "normalized_row_count": len(normalized),
        "drop_rate": round(1 - len(normalized) / max(len(raw_rows), 1), 4),
        "head_type_distribution": dict(head_types.most_common()),
        "tail_type_distribution": dict(tail_types.most_common()),
        "normalized_relation_distribution": dict(relations.most_common()),
        "surface_relation_distribution_top20": dict(surface_relations.most_common(20)),
        "confidence_buckets": dict(sorted(confidence_buckets.items())),
    }
    (out_dir / "ontology_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[ok]   wrote {out_dir/'normalized_triples.baseline.jsonl'}")
    print(f"[ok]   wrote {out_dir/'ontology_stats.json'}")
    print()
    print("=" * 60)
    print("baseline captured. Use scripts/diff_against_baseline.py after refactor.")
    print("=" * 60)


if __name__ == "__main__":
    main()
