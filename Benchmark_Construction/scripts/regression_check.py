"""Regression test for normalize.py: compare a candidate normalized_triples.jsonl
against the stage-0 baseline.

Stage 1 required bit-equivalence (`scripts/diff_against_baseline.py`).
Stage 3 will delete single-paper hard rules and we expect SOME drift.
This regression script reports recall/precision against the baseline so
each stage 3 commit can be gated on a configurable threshold.

Usage:
    python scripts/regression_check.py \\
        --baseline benchmark_runs/baseline_pre_refactor/normalized_triples.baseline.jsonl \\
        --candidate /tmp/datasetsa_post_refactor/normalized_triples.baseline.jsonl \\
        --min-recall 0.90

The match key is (head, head_type, normalized_relation, tail, tail_type).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph.utils import read_jsonl  # noqa: E402

MATCH_KEYS = ("head", "head_type", "normalized_relation", "tail", "tail_type")


def make_key(row: dict) -> tuple:
    return tuple(str(row.get(k, "")) for k in MATCH_KEYS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--min-recall", type=float, default=0.90)
    p.add_argument("--max-precision-drop", type=float, default=0.20,
                   help="how much precision (matched/candidate) is allowed to drop")
    p.add_argument("--max-show", type=int, default=10)
    args = p.parse_args()

    baseline_rows = read_jsonl(args.baseline)
    candidate_rows = read_jsonl(args.candidate)

    base_keys = {make_key(r) for r in baseline_rows}
    cand_keys = {make_key(r) for r in candidate_rows}

    matched = base_keys & cand_keys
    only_baseline = base_keys - cand_keys  # regressions: triples we lost
    only_candidate = cand_keys - base_keys  # new triples (could be good or bad)

    recall = len(matched) / max(len(base_keys), 1)
    precision = len(matched) / max(len(cand_keys), 1)

    print(f"baseline rows  : {len(baseline_rows)} ({len(base_keys)} unique keys)")
    print(f"candidate rows : {len(candidate_rows)} ({len(cand_keys)} unique keys)")
    print(f"matched        : {len(matched)}")
    print(f"only_baseline  : {len(only_baseline)}  (recall regressions)")
    print(f"only_candidate : {len(only_candidate)} (newly produced rows)")
    print()
    print(f"recall    = {recall:.4f}")
    print(f"precision = {precision:.4f}")
    print()

    # by-relation breakdown
    base_by_rel = Counter(make_key(r)[2] for r in baseline_rows)
    cand_by_rel = Counter(make_key(r)[2] for r in candidate_rows)
    matched_by_rel: Counter = Counter()
    for k in matched:
        matched_by_rel[k[2]] += 1
    print("per-relation recall:")
    for rel, base_n in base_by_rel.most_common():
        m = matched_by_rel[rel]
        print(f"  {rel:25s} matched={m:4d} / baseline={base_n:4d} "
              f"({m / max(base_n, 1):.2%})  candidate={cand_by_rel[rel]}")

    if only_baseline and args.max_show > 0:
        print()
        print(f"--- regressions (showing first {min(args.max_show, len(only_baseline))} of {len(only_baseline)}) ---")
        for key in list(only_baseline)[: args.max_show]:
            print("  -", " | ".join(key))
    if only_candidate and args.max_show > 0:
        print()
        print(f"--- new in candidate (showing first {min(args.max_show, len(only_candidate))} of {len(only_candidate)}) ---")
        for key in list(only_candidate)[: args.max_show]:
            print("  +", " | ".join(key))

    print()
    if recall < args.min_recall:
        print(f"[FAIL] recall {recall:.4f} < threshold {args.min_recall:.4f}")
        sys.exit(1)
    print(f"[ok]   recall {recall:.4f} >= threshold {args.min_recall:.4f}")


if __name__ == "__main__":
    main()
