"""Ratio-driven + coverage-greedy sampling plan.

Three public helpers:

* :func:`parse_ratio_spec` — ``["two_hop_tail=0.3", "experiment_code=0.2"]``
  into a ``{type: weight}`` dict, normalized so the declared entries sum
  to exactly their originals and undeclared-enabled types split the
  remaining mass evenly.

* :func:`load_graph_coverage` — read ``global_graph.graphml`` into
  ``(nodes, edges, alias_map)`` where ``nodes`` is the set of canonical
  node IDs and ``edges`` is the set of ``(h_canon, rel_canon, t_canon)``
  tuples. ``alias_map`` maps pre-fusion entity names to the canonical
  graphml node — so when the sampler hands us a triple whose head was
  fused under a different canonical name, we still register that it
  covered the fused node.

* :func:`allocate_samples` — apply per-type quotas to
  ``candidates_by_type`` and, within each quota, either pick greedily
  for graph coverage or fall back to random. Returns
  ``(selected_list, diag_dict)``.

The allocator is pure: it never touches disk and never imports from
``cli.py`` or ``validator.py``. The diag dict is exactly what
``summary.json`` needs to serialize verbatim under keys
``question_type_quotas`` and ``coverage``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    nx = None
    _NETWORKX_IMPORT_ERROR = exc
else:  # pragma: no cover
    _NETWORKX_IMPORT_ERROR = None

from .indexing import canonicalize_entity


logger = logging.getLogger("question_generation.sampling_plan")


# ---------------------------------------------------------------------------
# Ratio parsing
# ---------------------------------------------------------------------------
def parse_ratio_spec(items: list[str] | None, enabled_types: set[str]) -> dict[str, float]:
    """Turn ``["type=x", ...]`` + the enabled-types set into a final
    ``{type: weight}`` dict that sums to 1.0.

    Rules (chosen by user):
      * declared types in ``items`` get their explicit weight
      * undeclared-but-enabled types share the remainder evenly
      * if declared weights sum > 1, normalize to 1 and log a warning
      * weights < 0 → clamped to 0 + warn
      * items may reference types not in ``enabled_types`` — ignored + warn
    """
    if not items:
        # no ratio spec: every enabled type gets equal weight. This
        # matches the pre-existing "no per-type bias" behaviour as long
        # as the caller follows the same uniform mass conceptually.
        if not enabled_types:
            return {}
        share = 1.0 / len(enabled_types)
        return {t: share for t in enabled_types}

    declared: dict[str, float] = {}
    for raw in items:
        if "=" not in raw:
            logger.warning("ratio item %r missing '=' — skipping", raw)
            continue
        key, val = raw.split("=", 1)
        key = key.strip()
        try:
            w = float(val.strip())
        except ValueError:
            logger.warning("ratio item %r has non-numeric weight — skipping", raw)
            continue
        if key not in enabled_types:
            logger.warning("ratio item %r references disabled type — ignored", raw)
            continue
        if w < 0:
            logger.warning("ratio item %r is negative — clamped to 0", raw)
            w = 0.0
        declared[key] = w

    declared_sum = sum(declared.values())
    if declared_sum > 1.0 + 1e-6:
        logger.warning(
            "declared ratios sum to %.3f > 1.0 — normalizing. Rest split among undeclared types.",
            declared_sum,
        )
        for k in declared:
            declared[k] /= declared_sum
        declared_sum = 1.0

    remaining = max(0.0, 1.0 - declared_sum)
    undeclared = [t for t in enabled_types if t not in declared]
    if undeclared and remaining > 0:
        share = remaining / len(undeclared)
        for t in undeclared:
            declared[t] = share
    elif undeclared and remaining == 0:
        for t in undeclared:
            declared[t] = 0.0

    # Ensure every enabled type has an entry (0 if user explicitly set 0)
    for t in enabled_types:
        declared.setdefault(t, 0.0)
    return declared


def ratios_to_quotas(
    ratios: dict[str, float],
    total_quota: int,
) -> dict[str, int]:
    """Convert ratios to integer quotas with largest-remainder apportionment.

    The returned quotas always sum to ``total_quota`` when at least one ratio
    key is present, avoiding the drift/truncation bug from round-then-nudge
    (which systematically piled drift on a single pivot key, skewing the
    distribution).
    """
    if not ratios:
        return {}
    target = max(0, int(total_quota))
    if target == 0:
        return {t: 0 for t in ratios}

    weights = {t: max(0.0, ratios.get(t, 0.0)) for t in ratios}
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        weights = {t: 1.0 for t in ratios}
        weight_sum = float(len(weights))

    raw = {t: (weights[t] / weight_sum) * target for t in ratios}
    quotas = {t: int(raw[t]) for t in ratios}
    remaining = target - sum(quotas.values())
    if remaining > 0:
        # Hamilton method: assign remaining units to the largest fractional
        # remainders, ties broken by raw value then key name.
        order = sorted(
            ratios,
            key=lambda t: (raw[t] - int(raw[t]), raw[t], t),
            reverse=True,
        )
        for t in order[:remaining]:
            quotas[t] += 1
    elif remaining < 0:
        order = sorted(
            ratios,
            key=lambda t: (raw[t] - int(raw[t]), raw[t], t),
        )
        for t in order[: -remaining]:
            quotas[t] = max(0, quotas[t] - 1)

    assert sum(quotas.values()) == target
    return quotas


# ---------------------------------------------------------------------------
# Coverage universe from graphml
# ---------------------------------------------------------------------------
def load_graph_coverage(
    graph_path: str | Path,
    *,
    with_tiers: bool = False,
) -> tuple[set[str], set[tuple[str, str, str]], dict[str, str]] | tuple[set[str], set[tuple[str, str, str]], dict[str, str], dict[str, str]]:
    """Return ``(nodes, edges, alias_map)`` or, with ``with_tiers=True``,
    also ``tier_map = {canon_node → "T1"|"T2"|"T3"}``.

    Tier assignment mirrors ``benchmark_weight.load_graph_weights``:
    PageRank(α=0.85), log-compress, percentile-rank (strictly-smaller
    convention) → T1 top 20%, T2 middle 60%, T3 bottom 20%. Nodes
    absent from PageRank (shouldn't happen but defensive) fall in T3.
    """
    if nx is None:
        raise ImportError(
            f"networkx required for --coverage greedy: {_NETWORKX_IMPORT_ERROR}"
        )
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"graph file not found: {path}")
    graph = nx.read_graphml(path)
    nodes: set[str] = set()
    edges: set[tuple[str, str, str]] = set()
    alias_map: dict[str, str] = {}
    # raw_canon[raw_node_id] → canonical key used in `nodes` set
    raw_to_canon: dict[str, str] = {}

    for node, data in graph.nodes(data=True):
        canon = canonicalize_entity(str(node))
        if canon:
            nodes.add(canon)
            raw_to_canon[str(node)] = canon
        raw_aliases = data.get("aliases")
        if isinstance(raw_aliases, str) and raw_aliases:
            for alias in raw_aliases.split(" | "):
                alias = alias.strip()
                if not alias or alias == node:
                    continue
                alias_canon = canonicalize_entity(alias)
                if alias_canon:
                    alias_map.setdefault(alias_canon, canon)

    for h, t, data in graph.edges(data=True):
        h_c = canonicalize_entity(str(h))
        t_c = canonicalize_entity(str(t))
        rel_c = str(data.get("relation", "") or data.get("label", "") or "").casefold()
        edges.add((h_c, rel_c, t_c))

    logger.info(
        "coverage universe loaded: %d nodes, %d edges, %d aliases from %s",
        len(nodes), len(edges), len(alias_map), path,
    )

    if not with_tiers:
        return nodes, edges, alias_map

    # Compute tier assignment via PageRank (reuses the same formula as
    # benchmark_weight; we don't import it to avoid the double read).
    import math
    import bisect as _bisect

    try:
        pagerank = nx.pagerank(graph, alpha=0.85)
    except Exception:
        logger.warning("pagerank failed for tier assignment, falling back to degree")
        pagerank = {node: float(deg) for node, deg in graph.degree()}
    eps = 1e-12
    log_scores = {n: math.log(max(pr, 0.0) + eps) for n, pr in pagerank.items()}
    sorted_values = sorted(log_scores.values())
    n_total = len(sorted_values)
    tier_map: dict[str, str] = {}
    for raw_node, raw_score in log_scores.items():
        canon = raw_to_canon.get(str(raw_node)) or canonicalize_entity(str(raw_node))
        if not canon:
            continue
        strictly_smaller = _bisect.bisect_left(sorted_values, raw_score)
        pct = strictly_smaller / (n_total - 1) if n_total > 1 else 1.0
        if pct >= 0.80:
            tier_map[canon] = "T1"
        elif pct >= 0.20:
            tier_map[canon] = "T2"
        else:
            tier_map[canon] = "T3"
    # Any graphml nodes missing from pagerank (shouldn't happen) → T3
    for canon in nodes:
        tier_map.setdefault(canon, "T3")
    return nodes, edges, alias_map, tier_map


def _resolve_to_graph_node(entity: str, graph_nodes: set[str], alias_map: dict[str, str]) -> str | None:
    """Map a pre-fusion entity string to a graphml canonical node ID or
    ``None`` if it isn't in the graph.
    """
    if not entity:
        return None
    canon = canonicalize_entity(entity)
    if not canon:
        return None
    if canon in graph_nodes:
        return canon
    via_alias = alias_map.get(canon)
    if via_alias and via_alias in graph_nodes:
        return via_alias
    return None


def _subgraph_coverage(
    subgraph,
    graph_nodes: set[str] | None,
    graph_edges: set[tuple[str, str, str]] | None,
    alias_map: dict[str, str] | None,
) -> tuple[set[str], set[tuple[str, str, str]]]:
    """Return the ``(nodes_hit, edges_hit)`` this candidate subgraph
    would contribute to the graph-coverage universe. If the graph was
    not loaded, returns empty sets — the caller falls back to random.
    """
    if graph_nodes is None or graph_edges is None:
        return set(), set()
    alias_map = alias_map or {}
    nodes_hit: set[str] = set()
    edges_hit: set[tuple[str, str, str]] = set()
    for n in subgraph.nodes:
        resolved = _resolve_to_graph_node(n.id, graph_nodes, alias_map)
        if resolved:
            nodes_hit.add(resolved)
    for e in subgraph.edges:
        h_res = _resolve_to_graph_node(e.head, graph_nodes, alias_map)
        t_res = _resolve_to_graph_node(e.tail, graph_nodes, alias_map)
        if h_res and t_res:
            rel = str(e.relation or "").casefold()
            tup = (h_res, rel, t_res)
            if tup in graph_edges:
                edges_hit.add(tup)
    return nodes_hit, edges_hit


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def allocate_samples(
    candidates_by_type: dict[str, list],
    total_quota: int,
    ratios: dict[str, float],
    coverage_mode: str = "off",
    graph_nodes: set[str] | None = None,
    graph_edges: set[tuple[str, str, str]] | None = None,
    alias_map: dict[str, str] | None = None,
    coverage_priority: int = 0,
    seed: int = 7,
) -> tuple[list, dict[str, Any]]:
    """Return ``(selected, diag)``.

    ``candidates_by_type`` — ``{type: [SampledSubgraph, ...]}`` already
    filtered for evidence/type eligibility. Each SampledSubgraph carries
    its own ``nodes`` / ``edges`` attributes that coverage reads.

    ``total_quota`` — overall sample budget, typically ``args.max_samples``.

    ``ratios`` — mapping from :func:`parse_ratio_spec`. Sums to 1.

    ``coverage_mode``:
      * ``"off"``: random-shuffle each type's pool and take the first N.
      * ``"greedy"``: pick candidates in order of new-coverage gain.

    When ``coverage_mode == "greedy"`` but the graph isn't loaded (no
    ``graph_nodes``), we log a warning and fall back to random.
    """
    if coverage_mode == "greedy" and (graph_nodes is None or graph_edges is None):
        logger.warning(
            "coverage=greedy but graph_nodes/edges not supplied — falling back to random"
        )
        coverage_mode = "off"
    # coverage_priority requires a graph universe too; silently clamp if absent.
    if coverage_priority > 0 and (graph_nodes is None or graph_edges is None):
        logger.warning(
            "coverage_priority=%d but graph_nodes/edges not supplied — ignoring",
            coverage_priority,
        )
        coverage_priority = 0
    coverage_priority = max(0, min(int(coverage_priority), int(total_quota)))

    per_type_pool: dict[str, int] = {t: len(pool) for t, pool in candidates_by_type.items()}
    per_type_selected: dict[str, int] = {t: 0 for t in candidates_by_type}
    per_type_new_nodes: dict[str, int] = {t: 0 for t in candidates_by_type}
    per_type_new_edges: dict[str, int] = {t: 0 for t in candidates_by_type}

    rng = random.Random(seed)
    selected: list = []
    covered_nodes: set[str] = set()
    covered_edges: set[tuple[str, str, str]] = set()
    # Working copies of each type's pool: we'll pop used candidates from
    # these so the ratio pass operates on the remainder.
    working_pools: dict[str, list] = {t: list(pool) for t, pool in candidates_by_type.items()}

    # ------------------------------------------------------------------
    # Phase 0 (optional): coverage_priority pre-selection
    # Pick up to K samples across ALL types whichever adds the most new
    # graph coverage. This runs before ratio quotas and is deducted from
    # both the total budget and each type's per-type pool.
    # ------------------------------------------------------------------
    coverage_priority_picked = 0
    if coverage_priority > 0:
        picked_seeds, covered_nodes, covered_edges = _greedy_pick_cross_type(
            working_pools, coverage_priority,
            covered_nodes, covered_edges,
            graph_nodes, graph_edges, alias_map,
            per_type_selected=per_type_selected,
            per_type_new_nodes=per_type_new_nodes,
            per_type_new_edges=per_type_new_edges,
        )
        selected.extend(picked_seeds)
        coverage_priority_picked = len(picked_seeds)
        logger.info(
            "coverage_priority: picked %d cross-type seed samples (target=%d)",
            coverage_priority_picked, coverage_priority,
        )

    # Remaining budget after the coverage-priority pass
    remaining_quota = max(0, total_quota - coverage_priority_picked)
    quotas = ratios_to_quotas(ratios, remaining_quota)

    # Process types in a stable deterministic order so the same input
    # always produces the same selection (independent of dict iteration).
    ordered_types = sorted(candidates_by_type.keys())

    for qtype in ordered_types:
        pool = working_pools.get(qtype, [])
        quota = quotas.get(qtype, 0)
        if quota <= 0 or not pool:
            continue
        if quota > len(pool):
            logger.warning(
                "type=%s: quota=%d > pool=%d (after coverage_priority) — capping at pool size",
                qtype, quota, len(pool),
            )
            quota = len(pool)

        # Snapshot BEFORE greedy mutates covered_* in place.
        before_nodes = len(covered_nodes)
        before_edges = len(covered_edges)
        if coverage_mode != "greedy":
            rng.shuffle(pool)
            picked = pool[:quota]
            for cand in picked:
                n_hit, e_hit = _subgraph_coverage(cand, graph_nodes, graph_edges, alias_map)
                covered_nodes |= n_hit
                covered_edges |= e_hit
        else:
            picked = _greedy_pick(
                pool, quota,
                covered_nodes, covered_edges,
                graph_nodes, graph_edges, alias_map,
            )
        # Accumulate on top of any coverage_priority seeds already picked in this type
        per_type_new_nodes[qtype] += len(covered_nodes) - before_nodes
        per_type_new_edges[qtype] += len(covered_edges) - before_edges
        per_type_selected[qtype] += len(picked)
        selected.extend(picked)

    diag = {
        "declared_ratios": dict(ratios),
        "effective_quotas": quotas,
        "per_type_pool": per_type_pool,
        "per_type_selected": per_type_selected,
        "coverage_mode": coverage_mode,
        "coverage_priority_target": coverage_priority,
        "coverage_priority_picked": coverage_priority_picked,
    }
    if graph_nodes is not None and graph_edges is not None:
        diag["coverage"] = {
            "total_graph_nodes": len(graph_nodes),
            "covered_nodes": len(covered_nodes),
            "node_coverage_rate": round(
                len(covered_nodes) / max(len(graph_nodes), 1), 4
            ),
            "total_graph_edges": len(graph_edges),
            "covered_edges": len(covered_edges),
            "edge_coverage_rate": round(
                len(covered_edges) / max(len(graph_edges), 1), 4
            ),
            "per_type_new_nodes": per_type_new_nodes,
            "per_type_new_edges": per_type_new_edges,
        }
    return selected, diag


def allocate_samples_node_based(
    candidates_by_type: dict[str, list],
    total_quota: int,
    graph_nodes: set[str],
    graph_edges: set[tuple[str, str, str]],
    alias_map: dict[str, str],
    tier_map: dict[str, str],
    tier_quota: dict[str, int] | None = None,
    type_ratios: dict[str, float] | None = None,
    seed: int = 7,
) -> tuple[list, dict[str, Any]]:
    """Node-centric allocator.

    Every graphml node gets a per-tier quota (e.g. T1=3, T2=2, T3=1)
    and we pick samples that cover that node, preferring question-type
    diversity. When the sum of per-node quotas exceeds ``total_quota``
    we truncate from T3 up to T1 (i.e. sacrifice niche nodes first).

    Returns ``(selected, diag)`` where ``diag`` carries tier-level
    counts, per-node selection breakdown, and the usual coverage.
    """
    tier_quota = {"T1": 3, "T2": 2, "T3": 1, **(tier_quota or {})}
    # Compute per-type global target from ratios (if given). The allocator
    # uses these as soft constraints: within each node's pick, prefer types
    # whose ``(target - selected_so_far)`` gap is largest.
    #
    # ``type_ratios`` must already be normalized to sum to 1 (callers go
    # through ``parse_ratio_spec`` which enforces this). When None, all
    # enabled types share mass equally, which degenerates to the v1
    # round-robin behaviour.
    type_ratios_norm: dict[str, float]
    if type_ratios:
        type_ratios_norm = dict(type_ratios)
    else:
        enabled = sorted(candidates_by_type.keys())
        eq = 1.0 / len(enabled) if enabled else 0.0
        type_ratios_norm = {t: eq for t in enabled}
    type_target: dict[str, int] = {
        t: int(round(max(0.0, type_ratios_norm.get(t, 0.0)) * total_quota))
        for t in candidates_by_type
    }

    # --- 1) node → tier, count tiers --------------------------------------
    tier_nodes: dict[str, list[str]] = {"T1": [], "T2": [], "T3": []}
    for node in graph_nodes:
        tier_nodes[tier_map.get(node, "T3")].append(node)
    # Deterministic order inside each tier
    for tier in tier_nodes:
        tier_nodes[tier].sort()

    # --- 2) candidate → canonical nodes it touches ------------------------
    # Build (qtype, idx_in_pool) → set_of_canon_nodes_touched and the inverse.
    #
    # Each candidate may touch multiple graph nodes; we don't pre-dedupe
    # across types because the same fact could surface as both a
    # claim_choice and a two_hop_tail — they count as separate samples.
    node_to_candidates: dict[str, list[tuple[str, int]]] = {n: [] for n in graph_nodes}
    cand_touches: dict[tuple[str, int], set[str]] = {}
    for qtype, pool in candidates_by_type.items():
        for idx, cand in enumerate(pool):
            touched = set()
            for node in cand.nodes:
                resolved = _resolve_to_graph_node(node.id, graph_nodes, alias_map or {})
                if resolved:
                    touched.add(resolved)
            for e in cand.edges:
                for ent in (e.head, e.tail):
                    resolved = _resolve_to_graph_node(ent, graph_nodes, alias_map or {})
                    if resolved:
                        touched.add(resolved)
            if not touched:
                continue  # nothing to contribute to graphml coverage
            cand_touches[(qtype, idx)] = touched
            for n in touched:
                node_to_candidates[n].append((qtype, idx))

    # --- 3) initial per-node target (pre-truncation) ----------------------
    per_node_target: dict[str, int] = {
        n: tier_quota.get(tier_map.get(n, "T3"), 0) for n in graph_nodes
    }
    total_target = sum(per_node_target.values())

    # --- 4) truncate from T3 to T1 if total > total_quota -----------------
    truncated_tiers: dict[str, int] = {"T1": 0, "T2": 0, "T3": 0}
    if total_target > total_quota:
        excess = total_target - total_quota
        for tier in ("T3", "T2", "T1"):
            if excess <= 0:
                break
            # Randomly pick nodes in this tier, decrement each by 1 (down to 0)
            # until excess absorbed. Shuffling makes truncation fair across
            # niche nodes of the same tier.
            rng_trunc = random.Random(seed)
            pool_nodes = list(tier_nodes[tier])
            rng_trunc.shuffle(pool_nodes)
            # Repeatedly sweep — each sweep decrements still-positive quotas by 1
            changed = True
            while excess > 0 and changed:
                changed = False
                for n in pool_nodes:
                    if excess <= 0:
                        break
                    if per_node_target[n] > 0:
                        per_node_target[n] -= 1
                        truncated_tiers[tier] += 1
                        excess -= 1
                        changed = True
        total_target = sum(per_node_target.values())
        logger.info(
            "node_based: truncated %d quota-slots (T3=%d, T2=%d, T1=%d) to fit max_samples=%d",
            sum(truncated_tiers.values()),
            truncated_tiers["T3"], truncated_tiers["T2"], truncated_tiers["T1"],
            total_quota,
        )

    # --- 5) iterate T1 → T3, per node pick up to per_node_target[n] -------
    rng = random.Random(seed)
    selected: list = []
    picked_cand_keys: set[tuple[str, int]] = set()  # dedup same (type,idx)
    per_tier_selected: dict[str, int] = {"T1": 0, "T2": 0, "T3": 0}
    per_type_selected: dict[str, int] = {qt: 0 for qt in candidates_by_type}
    per_node_count: dict[str, int] = {n: 0 for n in graph_nodes}
    insufficient_nodes: dict[str, int] = {"T1": 0, "T2": 0, "T3": 0}
    covered_nodes: set[str] = set()
    covered_edges: set[tuple[str, str, str]] = set()

    for tier in ("T1", "T2", "T3"):
        nodes_in_tier = list(tier_nodes[tier])
        rng.shuffle(nodes_in_tier)      # fair order within a tier
        for node in nodes_in_tier:
            want = per_node_target[node]
            if want <= 0:
                continue
            # candidates touching this node that haven't been picked already
            candidates_here = [
                key for key in node_to_candidates.get(node, [])
                if key not in picked_cand_keys
            ]
            if not candidates_here:
                insufficient_nodes[tier] += 1
                continue
            # --- ratio-biased type-diversity pass -------------------------
            # Within this node's pool, at each step pick from the type
            # that's most under-supplied relative to its global target:
            # ``deficit = type_target[t] - per_type_selected[t]``.
            # Ties broken by (1) larger remaining pool at this node,
            # (2) alphabetical. This approximates the declared global
            # ratio while still guaranteeing node coverage.
            by_type: dict[str, list[tuple[str, int]]] = {}
            for key in candidates_here:
                by_type.setdefault(key[0], []).append(key)
            for cands in by_type.values():
                rng.shuffle(cands)
            picked_here: list[tuple[str, int]] = []
            while len(picked_here) < want:
                active_types = [t for t, cs in by_type.items() if cs]
                if not active_types:
                    break
                # Sort by deficit desc, then node-pool size desc, then name
                def _score(t: str):
                    deficit = type_target.get(t, 0) - per_type_selected.get(t, 0)
                    return (-deficit, -len(by_type[t]), t)
                active_types.sort(key=_score)
                chosen_type = active_types[0]
                picked_here.append(by_type[chosen_type].pop(0))
            if len(picked_here) < want:
                insufficient_nodes[tier] += 1
                logger.debug(
                    "node %r (tier %s) short: wanted %d, got %d",
                    node, tier, want, len(picked_here),
                )
            # commit
            for key in picked_here:
                picked_cand_keys.add(key)
                qtype, idx = key
                cand = candidates_by_type[qtype][idx]
                selected.append(cand)
                per_tier_selected[tier] += 1
                per_type_selected[qtype] += 1
                per_node_count[node] += 1
                # track coverage (useful for reporting parity with the
                # old --coverage metric)
                t_nodes, t_edges = _subgraph_coverage(cand, graph_nodes, graph_edges, alias_map or {})
                covered_nodes |= t_nodes
                covered_edges |= t_edges

    # --- 6) diag ---------------------------------------------------------
    per_tier_quota_used = {
        tier: tier_quota.get(tier, 0) for tier in ("T1", "T2", "T3")
    }
    tier_sizes = {tier: len(tier_nodes[tier]) for tier in ("T1", "T2", "T3")}
    diag = {
        "mode": "node_based",
        "tier_quota": per_tier_quota_used,
        "tier_sizes": tier_sizes,
        "per_tier_selected": per_tier_selected,
        "per_type_selected": per_type_selected,
        "per_type_pool": {t: len(p) for t, p in candidates_by_type.items()},
        "per_type_target":  type_target,
        "per_type_ratio":   type_ratios_norm,
        "per_type_deviation": {
            t: per_type_selected.get(t, 0) - type_target.get(t, 0)
            for t in candidates_by_type
        },
        "truncated_tier_slots": truncated_tiers,
        "initial_target_samples": sum(
            tier_quota.get(tier_map.get(n, "T3"), 0) for n in graph_nodes
        ),
        "final_sampled": len(selected),
        "insufficient_nodes": insufficient_nodes,
        "coverage": {
            "coverage_mode": "node_based",
            "total_graph_nodes": len(graph_nodes),
            "covered_nodes": len(covered_nodes),
            "node_coverage_rate": round(
                len(covered_nodes) / max(len(graph_nodes), 1), 4
            ),
            "total_graph_edges": len(graph_edges),
            "covered_edges": len(covered_edges),
            "edge_coverage_rate": round(
                len(covered_edges) / max(len(graph_edges), 1), 4
            ),
            "per_tier_new_nodes": {
                tier: sum(
                    1 for n in tier_nodes[tier] if per_node_count.get(n, 0) > 0
                )
                for tier in ("T1", "T2", "T3")
            },
        },
        "per_node_count_histogram": _histogram(list(per_node_count.values())),
    }
    logger.info(
        "node_based: selected %d samples; tiers picked T1=%d T2=%d T3=%d; "
        "node-coverage %d/%d (%.1f%%); insufficient nodes T1=%d T2=%d T3=%d",
        len(selected),
        per_tier_selected["T1"], per_tier_selected["T2"], per_tier_selected["T3"],
        len(covered_nodes), len(graph_nodes),
        100 * len(covered_nodes) / max(len(graph_nodes), 1),
        insufficient_nodes["T1"], insufficient_nodes["T2"], insufficient_nodes["T3"],
    )
    return selected, diag


def _histogram(values: list[int]) -> dict[str, int]:
    """Small integer histogram used for per_node_count reporting."""
    hist: dict[str, int] = {}
    for v in values:
        key = str(v)
        hist[key] = hist.get(key, 0) + 1
    return hist


def _greedy_pick_cross_type(
    working_pools: dict[str, list],
    quota: int,
    covered_nodes: set[str],
    covered_edges: set[tuple[str, str, str]],
    graph_nodes: set[str] | None,
    graph_edges: set[tuple[str, str, str]] | None,
    alias_map: dict[str, str] | None,
    *,
    per_type_selected: dict[str, int],
    per_type_new_nodes: dict[str, int],
    per_type_new_edges: dict[str, int],
) -> tuple[list, set[str], set[tuple[str, str, str]]]:
    """Greedy set-cover across ALL candidate pools simultaneously.

    Mutates ``working_pools`` by removing picked candidates, so the
    downstream ratio-pass operates on the remainder. Also mutates
    ``per_type_selected / _new_nodes / _new_edges`` to attribute
    coverage gains to the originating type. Returns
    ``(picked, covered_nodes, covered_edges)``.

    Compared to :func:`_greedy_pick` which is per-type, this one
    prioritizes new coverage regardless of question type — that's the
    point of ``--coverage-priority``.
    """
    # Pre-compute coverage for every candidate, tagged with source type
    # so we know which pool to pop from.
    tagged: list[tuple[str, int, set[str], set[tuple[str, str, str]]]] = []
    for qtype, pool in working_pools.items():
        for idx, cand in enumerate(pool):
            n_set, e_set = _subgraph_coverage(cand, graph_nodes, graph_edges, alias_map)
            tagged.append((qtype, idx, n_set, e_set))

    picked: list = []
    picked_positions: dict[str, set[int]] = {t: set() for t in working_pools}

    while tagged and len(picked) < quota:
        best_i = -1
        best_gain = -1
        for i, (qtype, idx, n_set, e_set) in enumerate(tagged):
            if idx in picked_positions.get(qtype, set()):
                continue  # already taken
            gain = len(n_set - covered_nodes) + len(e_set - covered_edges)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break
        qtype, idx, n_set, e_set = tagged.pop(best_i)
        picked_positions[qtype].add(idx)
        cand = working_pools[qtype][idx]
        picked.append(cand)
        before_n, before_e = len(covered_nodes), len(covered_edges)
        covered_nodes |= n_set
        covered_edges |= e_set
        per_type_selected[qtype] = per_type_selected.get(qtype, 0) + 1
        per_type_new_nodes[qtype] = per_type_new_nodes.get(qtype, 0) + (len(covered_nodes) - before_n)
        per_type_new_edges[qtype] = per_type_new_edges.get(qtype, 0) + (len(covered_edges) - before_e)

    # Rebuild working_pools without the picked items so the ratio pass
    # won't re-pick them.
    for qtype, pool in working_pools.items():
        to_drop = picked_positions.get(qtype, set())
        if to_drop:
            working_pools[qtype] = [c for i, c in enumerate(pool) if i not in to_drop]
    return picked, covered_nodes, covered_edges


def _greedy_pick(
    pool: list,
    quota: int,
    covered_nodes: set[str],
    covered_edges: set[tuple[str, str, str]],
    graph_nodes: set[str] | None,
    graph_edges: set[tuple[str, str, str]] | None,
    alias_map: dict[str, str] | None,
) -> list:
    """Greedy set-cover over ``pool`` against the shared coverage sets.

    Pre-computes each candidate's coverage once; repeatedly scans the
    remaining pool for the highest-gain candidate and pops it. On
    ties, the first-encountered (pool order) candidate wins — keeps
    output deterministic when the input is deterministic.
    """
    coverage_cache = [
        _subgraph_coverage(c, graph_nodes, graph_edges, alias_map)
        for c in pool
    ]
    remaining_idx = list(range(len(pool)))
    picked: list = []
    # Work on a local copy of the running covered sets so we can
    # annotate gains correctly, but we mutate the caller-owned
    # ``covered_nodes`` / ``covered_edges`` (the caller passed them in).
    while remaining_idx and len(picked) < quota:
        best_i = -1
        best_gain = -1
        for pos, idx in enumerate(remaining_idx):
            n_set, e_set = coverage_cache[idx]
            gain = len(n_set - covered_nodes) + len(e_set - covered_edges)
            if gain > best_gain:
                best_gain = gain
                best_i = pos
        if best_i < 0:
            break
        chosen_idx = remaining_idx.pop(best_i)
        chosen = pool[chosen_idx]
        picked.append(chosen)
        n_set, e_set = coverage_cache[chosen_idx]
        covered_nodes |= n_set
        covered_edges |= e_set
    return picked
