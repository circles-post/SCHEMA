"""Compare a freshly normalized normalized_triples.jsonl against the baseline
captured by collect_baseline.py. Used as the hard gate for stage 1: any
behavior change in normalize.py / entity_verification.py shows up here.

Usage:
    python scripts/diff_against_baseline.py \
        --baseline benchmark_runs/baseline_pre_refactor/normalized_triples.baseline.jsonl \
        --candidate pipeline_outputs_minimal/normalized_triples.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph.utils import read_jsonl  # noqa: E402

KEY_FIELDS = (
    "doc_id",
    "chunk_id",
    "head",
    "head_type",
    "normalized_relation",
    "tail",
    "tail_type",
    "confidence",
    "evidence",
    "surface_relation",
)


def canonical(row: dict) -> tuple:
    return tuple((k, row.get(k, "")) for k in KEY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--max-show", type=int, default=10)
    args = parser.parse_args()

    base_rows = read_jsonl(args.baseline)
    cand_rows = read_jsonl(args.candidate)

    base_set = {canonical(r) for r in base_rows}
    cand_set = {canonical(r) for r in cand_rows}

    only_base = base_set - cand_set
    only_cand = cand_set - base_set

    print(f"[info] baseline rows  : {len(base_rows)}")
    print(f"[info] candidate rows : {len(cand_rows)}")
    print(f"[info] in baseline only (regressions) : {len(only_base)}")
    print(f"[info] in candidate only (new rows)   : {len(only_cand)}")

    if only_base:
        print()
        print("--- in baseline only (first {} of {}) ---".format(min(args.max_show, len(only_base)), len(only_base)))
        for tup in list(only_base)[: args.max_show]:
            print(json.dumps(dict(tup), ensure_ascii=False))
    if only_cand:
        print()
        print("--- in candidate only (first {} of {}) ---".format(min(args.max_show, len(only_cand)), len(only_cand)))
        for tup in list(only_cand)[: args.max_show]:
            print(json.dumps(dict(tup), ensure_ascii=False))

    if only_base or only_cand:
        print()
        print("[FAIL] baseline diff is non-empty")
        sys.exit(1)
    print()
    print("[ok]   baseline diff is empty — refactor is behavior-equivalent")


if __name__ == "__main__":
    main()
