"""Report how many 1-/2-/3-hop chain candidates exist in a triples JSONL.

Mirrors the gating the generator uses: min_confidence=0.7, support>=2, and
`profile_subgraph_evidence` strength bins. Run as a module from the parent:

    PYTHONPATH=/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa \\
    python -m question_generation.scripts.hop_statistics \\
      --triples  /path/to/normalized_triples.jsonl \\
      --chunks   /path/to/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from question_generation.evidence_profiler import _relation_strength
from question_generation.indexing import build_index
from question_generation.io import load_chunks, load_triples


def _key(triple) -> tuple[str, str, str]:
    from question_generation.indexing import canonicalize_entity
    from pubmed_graph.utils import normalize_text

    return (
        canonicalize_entity(triple.head),
        normalize_text(triple.normalized_relation).casefold(),
        canonicalize_entity(triple.tail),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--two-hop-cap", type=int, default=200000)
    parser.add_argument("--three-hop-cap", type=int, default=200000)
    args = parser.parse_args()

    print(f"loading triples from {args.triples}")
    triples = load_triples(args.triples)
    chunks = load_chunks(args.chunks)
    index = build_index(triples, chunks)

    print(f"  {len(triples)} triples, {len(chunks)} chunks, "
          f"{len(index.triples_by_key)} unique (h,r,t) groups")

    # ---------------------------------------------------------------------
    # 1-HOP
    # ---------------------------------------------------------------------
    one_hop_eligible: list[tuple[tuple[str, str, str], list]] = []
    total_one_hop_groups = 0
    for key, group in index.triples_by_key.items():
        total_one_hop_groups += 1
        if len(group) < args.min_support:
            continue
        avg_conf = sum(float(t.confidence) for t in group) / len(group)
        if avg_conf < args.min_confidence:
            continue
        one_hop_eligible.append((key, group))

    strength_1 = Counter()
    for _, group in one_hop_eligible:
        strength_1[_relation_strength(group[0].normalized_relation)] += 1

    print(
        f"\n=== 1-HOP (support>={args.min_support}, avg_conf>={args.min_confidence}) ==="
    )
    print(f"  total distinct (h,r,t) groups           : {total_one_hop_groups}")
    print(f"  eligible after support+confidence gate  : {len(one_hop_eligible)}")
    print(f"  relation-strength breakdown             : {dict(strength_1)}")

    # ---------------------------------------------------------------------
    # 2-HOP
    # ---------------------------------------------------------------------
    two_hop_paths: list[tuple[str, str, str, str, str]] = []  # (h, r1, pivot, r2, t2)
    strength_2 = Counter()  # joint (strength_hop1, strength_hop2)
    chain_strength_2 = Counter()  # weakest-link strength of the 2-hop chain
    # Build an index of eligible outgoing edges keyed by head (canonical)
    eligible_by_head: dict[str, list] = defaultdict(list)
    for key, group in one_hop_eligible:
        head_canon, _, _ = key
        eligible_by_head[head_canon].append((key, group))

    from question_generation.indexing import canonicalize_entity

    for key, group in one_hop_eligible:
        head_canon, _, pivot_canon = key
        # Find eligible 2nd hops starting at pivot, excluding back-edges
        for key2, group2 in eligible_by_head.get(pivot_canon, []):
            _, _, tail2_canon = key2
            if tail2_canon == head_canon:
                continue
            s1 = _relation_strength(group[0].normalized_relation)
            s2 = _relation_strength(group2[0].normalized_relation)
            strength_2[(s1, s2)] += 1
            # weakest-link chain strength
            strongest_order = ["strong", "medium", "weak"]
            weakest = max([s1, s2], key=lambda x: strongest_order.index(x))
            chain_strength_2[weakest] += 1
            two_hop_paths.append(
                (
                    group[0].head,
                    group[0].normalized_relation,
                    group[0].tail,
                    group2[0].normalized_relation,
                    group2[0].tail,
                )
            )
            if len(two_hop_paths) >= args.two_hop_cap:
                break
        if len(two_hop_paths) >= args.two_hop_cap:
            break

    print(
        f"\n=== 2-HOP (both hops eligible, no immediate back-edge, cap {args.two_hop_cap}) ==="
    )
    print(f"  total 2-hop paths                        : {len(two_hop_paths)}")
    print(f"  weakest-link chain strength breakdown    : {dict(chain_strength_2)}")
    print(f"  per-hop strength pair breakdown          :")
    for pair, count in strength_2.most_common():
        print(f"    {pair[0]:7s} -> {pair[1]:7s}  : {count}")

    # ---------------------------------------------------------------------
    # 3-HOP
    # ---------------------------------------------------------------------
    three_hop_count = 0
    chain_strength_3 = Counter()
    per_hop_strength_3 = Counter()  # (s1, s2, s3)
    strongest_order = ["strong", "medium", "weak"]

    for h, r1, t1, r2, t2 in two_hop_paths:
        head_canon = canonicalize_entity(h)
        pivot2_canon = canonicalize_entity(t2)
        first_canon = canonicalize_entity(t1)
        for key3, group3 in eligible_by_head.get(pivot2_canon, []):
            _, _, tail3_canon = key3
            # Skip immediate back-edges at the last hop and revisits of earlier nodes
            if tail3_canon in (head_canon, first_canon):
                continue
            s1 = _relation_strength(r1)
            s2 = _relation_strength(r2)
            s3 = _relation_strength(group3[0].normalized_relation)
            per_hop_strength_3[(s1, s2, s3)] += 1
            weakest = max([s1, s2, s3], key=lambda x: strongest_order.index(x))
            chain_strength_3[weakest] += 1
            three_hop_count += 1
            if three_hop_count >= args.three_hop_cap:
                break
        if three_hop_count >= args.three_hop_cap:
            break

    print(
        f"\n=== 3-HOP (all three hops eligible, no revisits of earlier nodes, cap {args.three_hop_cap}) ==="
    )
    print(f"  total 3-hop paths                        : {three_hop_count}")
    print(f"  weakest-link chain strength breakdown    : {dict(chain_strength_3)}")
    print(f"  top per-hop strength triples             :")
    for triple, count in per_hop_strength_3.most_common(8):
        print(f"    {triple[0]:7s} -> {triple[1]:7s} -> {triple[2]:7s}  : {count}")

    # ---------------------------------------------------------------------
    # JSON dump for programmatic downstream use
    # ---------------------------------------------------------------------
    summary = {
        "min_confidence": args.min_confidence,
        "min_support": args.min_support,
        "one_hop": {
            "total_groups": total_one_hop_groups,
            "eligible": len(one_hop_eligible),
            "strength_breakdown": dict(strength_1),
        },
        "two_hop": {
            "total_paths": len(two_hop_paths),
            "weakest_link_strength": dict(chain_strength_2),
            "per_hop_pairs": {f"{a}->{b}": c for (a, b), c in strength_2.items()},
        },
        "three_hop": {
            "total_paths": three_hop_count,
            "weakest_link_strength": dict(chain_strength_3),
            "per_hop_triples": {f"{a}->{b}->{c}": v for (a, b, c), v in per_hop_strength_3.most_common(20)},
        },
    }
    print("\n=== JSON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
