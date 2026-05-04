"""Build a graph-coverage-maximizing balanced subset of a v2 bench.

Two-phase greedy:
  1. Stratify by (question_type, tier, evidence_strength).
     Within each stratum, pick up to N samples in order of marginal
     graph-node coverage gain (NOT random).
  2. Coverage tail: globally pick further samples (ignoring stratum caps)
     by marginal gain until either marginal gain == 0 (saturation) or the
     hard `--budget` is reached.

Inputs:
  --dataset      path to samples.jsonl
  --graph        path to global_graph.graphml
  --out          path to write the subset .jsonl
  --summary      path to write a JSON describing what was selected
  --cap          per-stratum cap N (default 25)
  --budget       max total samples after the tail pass (default 800)
  --seed         tie-break random seed (default 42)

Source samples.jsonl is never modified.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


def extract_sample_nodes(record: dict, graph_nodes: set[str]) -> set[str]:
    """Return graph-anchored entities mentioned by this sample."""
    nodes: set[str] = set()
    sg = record.get("subgraph") or {}
    if isinstance(sg, dict):
        for item in sg.get("nodes") or []:
            if isinstance(item, dict) and "id" in item:
                nodes.add(str(item["id"]))
        for item in sg.get("edges") or []:
            if isinstance(item, dict):
                for k in ("head", "tail"):
                    if k in item:
                        nodes.add(str(item[k]))
    md = record.get("metadata") or {}
    for k in ("query_subject", "query_object"):
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            nodes.add(v.strip())
    return nodes & graph_nodes


def stratum_key(record: dict) -> tuple[str, str, str]:
    qt = record.get("question_type", "?")
    bw = (record.get("metadata") or {}).get("benchmark_weight") or {}
    tier = bw.get("tier", "?")
    es = (record.get("grounding") or {}).get("evidence_strength", "?")
    return (qt, tier, es)


def stratified_greedy_with_tail(
    rows: list[dict],
    nodes_per_row: list[set[str]],
    *,
    cap: int,
    budget: int,
    seed: int,
) -> tuple[list[int], dict]:
    rng = random.Random(seed)

    # Build strata (preserve insertion order for determinism, then shuffle each
    # stratum so ties at gain=0 are broken reproducibly).
    by_stratum: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_stratum[stratum_key(r)].append(i)
    for k in by_stratum:
        rng.shuffle(by_stratum[k])

    chosen: list[int] = []
    chosen_set: set[int] = set()
    covered: set[str] = set()

    # ----- Phase 1: per-stratum greedy up to cap -----
    phase1_per_stratum: dict[str, dict] = {}
    for k, idx_list in by_stratum.items():
        local_chosen: list[int] = []
        available = list(idx_list)
        while len(local_chosen) < cap and available:
            best_i, best_gain = None, -1
            for i in available:
                gain = len(nodes_per_row[i] - covered)
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None:
                break
            local_chosen.append(best_i)
            chosen_set.add(best_i)
            covered |= nodes_per_row[best_i]
            available.remove(best_i)
        chosen.extend(local_chosen)
        phase1_per_stratum["|".join(k)] = {
            "stratum_size": len(idx_list),
            "phase1_picked": len(local_chosen),
        }

    # ----- Phase 2: coverage tail (ignore stratum caps) -----
    remaining = [i for i in range(len(rows)) if i not in chosen_set]
    tail_added = 0
    while remaining and len(chosen) < budget:
        best_i, best_gain = None, 0
        for i in remaining:
            gain = len(nodes_per_row[i] - covered)
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None or best_gain <= 0:
            break  # saturation
        chosen.append(best_i)
        chosen_set.add(best_i)
        covered |= nodes_per_row[best_i]
        remaining.remove(best_i)
        tail_added += 1

    summary = {
        "phase1_per_stratum": phase1_per_stratum,
        "phase2_tail_added": tail_added,
        "stop_reason": "budget_reached" if len(chosen) >= budget else "saturation",
    }
    return chosen, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True, help="Output JSONL of selected samples.")
    ap.add_argument("--summary", required=True, help="Output JSON of selection details.")
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    graph_path = Path(args.graph)
    out_path = Path(args.out)
    summary_path = Path(args.summary)

    if not dataset_path.is_file():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    if not graph_path.is_file():
        print(f"ERROR: graph not found: {graph_path}", file=sys.stderr)
        return 2

    print(f"[subset] loading {graph_path}", flush=True)
    g = nx.read_graphml(graph_path)
    graph_nodes = set(g.nodes())
    print(f"[subset]   graph nodes: {len(graph_nodes)}", flush=True)

    print(f"[subset] loading {dataset_path}", flush=True)
    rows: list[dict] = []
    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[subset]   samples: {len(rows)}", flush=True)

    nodes_per_row = [extract_sample_nodes(r, graph_nodes) for r in rows]
    full_coverage = set().union(*nodes_per_row) if nodes_per_row else set()
    print(
        f"[subset]   full-bench covers {len(full_coverage)} graph nodes "
        f"({len(full_coverage) / max(1, len(graph_nodes)) * 100:.1f}%)",
        flush=True,
    )

    print(
        f"[subset] greedy with cap={args.cap}, budget={args.budget}, seed={args.seed}",
        flush=True,
    )
    chosen_idx, sel_summary = stratified_greedy_with_tail(
        rows, nodes_per_row, cap=args.cap, budget=args.budget, seed=args.seed
    )

    chosen_rows = [rows[i] for i in chosen_idx]
    chosen_nodes = set().union(*[nodes_per_row[i] for i in chosen_idx])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve sample order — chosen_idx already reflects pick order. We could
    # alternatively sort by original line index to make diffs friendlier; do that.
    chosen_rows_sorted = [rows[i] for i in sorted(chosen_idx)]
    with out_path.open("w", encoding="utf-8") as fh:
        for r in chosen_rows_sorted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[subset] wrote {len(chosen_rows_sorted)} samples → {out_path}", flush=True)

    # Detailed per-stratum breakdown
    by_stratum_orig: dict[tuple[str, str, str], int] = Counter(stratum_key(r) for r in rows)
    by_stratum_sel: dict[tuple[str, str, str], int] = Counter(stratum_key(r) for r in chosen_rows)
    qt_orig = Counter(r.get("question_type", "?") for r in rows)
    qt_sel = Counter(r.get("question_type", "?") for r in chosen_rows)

    summary = {
        "dataset": str(dataset_path),
        "graph": str(graph_path),
        "out": str(out_path),
        "config": {"cap": args.cap, "budget": args.budget, "seed": args.seed},
        "n_full_bench": len(rows),
        "n_subset": len(chosen_rows),
        "graph_nodes_total": len(graph_nodes),
        "full_bench_covered_nodes": len(full_coverage),
        "subset_covered_nodes": len(chosen_nodes),
        "subset_coverage_of_graph_pct": round(
            len(chosen_nodes) / max(1, len(graph_nodes)) * 100, 2
        ),
        "subset_coverage_of_full_bench_pct": round(
            len(chosen_nodes) / max(1, len(full_coverage)) * 100, 2
        ),
        "phase1_per_stratum": sel_summary["phase1_per_stratum"],
        "phase2_tail_added": sel_summary["phase2_tail_added"],
        "stop_reason": sel_summary["stop_reason"],
        "question_type_orig": dict(qt_orig.most_common()),
        "question_type_subset": dict(qt_sel.most_common()),
        "stratum_orig_vs_subset": {
            "|".join(k): {"orig": by_stratum_orig.get(k, 0), "subset": by_stratum_sel.get(k, 0)}
            for k in sorted(by_stratum_orig.keys())
        },
        "missed_graph_nodes_count": len(graph_nodes - chosen_nodes),
        "missed_relative_to_full_bench_count": len(full_coverage - chosen_nodes),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[subset] wrote summary → {summary_path}", flush=True)

    print()
    print(f"  full bench:   {len(rows):>5d} samples, covers {len(full_coverage):>4d} graph nodes ({summary['full_bench_covered_nodes']/max(1,len(graph_nodes))*100:.1f}% of graph)")
    print(f"  subset:       {len(chosen_rows):>5d} samples, covers {len(chosen_nodes):>4d} graph nodes ({summary['subset_coverage_of_graph_pct']}% of graph, {summary['subset_coverage_of_full_bench_pct']}% of full-bench coverage)")
    print(f"  reduction:    {(1 - len(chosen_rows)/max(1,len(rows)))*100:.1f}%   stop_reason: {summary['stop_reason']}")
    print(f"  question_type orig:   {dict(qt_orig.most_common())}")
    print(f"  question_type subset: {dict(qt_sel.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
