from __future__ import annotations

import bisect
import logging
import math
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
from .models import QuestionSample


logger = logging.getLogger("question_generation.benchmark_weight")


CENTRALITY_SOURCE_VERSION = "graph_pagerank_log_percentile_v1"

# Percentile cutoffs over the log-PageRank distribution. Anchors in the top
# 20% get T1 (weight 1.5), bottom 20% get T3 (weight 0.5), the middle 60%
# keep the neutral T2 (weight 1.0).
_TIER_CUTOFFS = (
    (0.80, "T1", 1.5),
    (0.20, "T2", 1.0),
    (0.0, "T3", 0.5),
)
# Anchors absent from the pruned graph are almost always entities that
# `pubmed_graph.graph_ops.prune_small_components` dropped as isolates — by
# definition, "niche" in the graph-connectivity sense the user asked about.
# Treat them as T3 (weight 0.5) with a dedicated tier label so the breakdown
# stays readable in the summary.
_NOT_FOUND_TIER = ("T3_not_in_graph", 0.5)


def load_graph_weights(
    graph_path: str | Path,
) -> tuple[dict[str, float], dict[str, str]]:
    """Load ``global_graph.graphml`` and return ``(weight_map, alias_map)``.

    ``weight_map`` is ``{node_id: percentile_rank in [0, 1]}`` computed from
    PageRank(alpha=0.85), log-transformed then percentile-ranked, so values
    sit in [0, 1] regardless of the run's absolute scores.

    ``alias_map`` is ``{alias_name: canonical_node_id}`` harvested from each
    node's ``aliases`` attribute (written by
    ``pubmed_graph.graph_ops.export_graphml``). Used by
    :func:`lookup_with_fallback` so triples whose head/tail strings were
    fusion-merged still resolve to their representative's weight.
    """
    if nx is None:
        raise ImportError(
            f"networkx is required for benchmark weighting: {_NETWORKX_IMPORT_ERROR}"
        )
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"graph file not found: {path}")
    graph = nx.read_graphml(path)
    if graph.number_of_nodes() == 0:
        return {}, {}

    # Harvest fusion aliases. graph_ops.export_graphml joins the list with
    # " | " so we split on that separator.
    alias_map: dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        raw = data.get("aliases")
        if not isinstance(raw, str) or not raw:
            continue
        for alias in raw.split(" | "):
            alias = alias.strip()
            if not alias or alias == node:
                continue
            alias_map.setdefault(alias, node)
            alias_map.setdefault(alias.casefold(), node)
    try:
        pagerank = nx.pagerank(graph, alpha=0.85)
    except Exception:
        # PageRank can fail to converge on pathological graphs — fall back to
        # degree centrality, which has the same coarse "hubs vs niche" signal.
        logger.warning("benchmark_weight: pagerank failed, falling back to degree")
        degree_view = graph.degree()
        pagerank = {node: float(deg) for node, deg in degree_view}
    if not pagerank:
        return {}, alias_map
    # Log-compress, then percentile-rank as
    # ``(# nodes with strictly smaller score) / (n - 1)``.
    # PageRank on real graphs typically has a long flat tail of minimum-score
    # nodes (isolated or dangling). Midrank on that tie bucket would lift
    # everyone to ≈20% and leave T3 empty; the strictly-smaller convention
    # keeps all bottom-tied nodes at percentile 0, i.e. firmly T3.
    eps = 1e-12
    log_scores = {node: math.log(max(score, 0.0) + eps) for node, score in pagerank.items()}
    sorted_values = sorted(log_scores.values())
    n = len(sorted_values)
    percentile_map: dict[str, float] = {}
    for node, value in log_scores.items():
        strictly_smaller = bisect.bisect_left(sorted_values, value)
        percentile_map[node] = strictly_smaller / (n - 1) if n > 1 else 1.0
    return percentile_map, alias_map


def select_anchor_entity(sample: QuestionSample) -> tuple[str, str]:
    """Return ``(entity_raw, anchor_role)`` for the question's answer target.

    - ``two_hop_tail``: the middle node (``edges[0].tail``, same as
      ``edges[1].head``) is the answer.
    - All other types (``claim_choice``, ``boolean_support``, ``essay``,
      ``experiment_code``): the tail of the single edge is the claim's
      subject-of-interest.

    Falls back to an empty string + ``"unknown"`` when the subgraph is
    malformed.
    """
    subgraph = sample.subgraph if isinstance(sample.subgraph, dict) else {}
    edges = subgraph.get("edges") or []
    if not edges:
        return "", "unknown"
    first = edges[0] if isinstance(edges[0], dict) else {}
    if sample.question_type == "two_hop_tail":
        return str(first.get("tail", "") or ""), "two_hop_middle"
    return str(first.get("tail", "") or ""), "tail"


def lookup_with_fallback(
    entity_raw: str,
    weight_map: dict[str, float],
    alias_map: dict[str, str] | None = None,
) -> tuple[float | None, str]:
    """Resolve an entity name against the graph weight map.

    Strategies tried in order:
      1. ``direct``         — exact string match in ``weight_map``
      2. ``canonicalized``  — lowercased/stripped form via ``canonicalize_entity``
      3. ``alias``          — fusion alias → canonical node in ``alias_map``
      4. ``not_found``      — entity pruned or otherwise absent from the graph
    """
    if not entity_raw:
        return None, "not_found"
    direct = weight_map.get(entity_raw)
    if direct is not None:
        return direct, "direct"
    canon = canonicalize_entity(entity_raw)
    if canon and canon != entity_raw:
        via_canon = weight_map.get(canon)
        if via_canon is not None:
            return via_canon, "canonicalized"
    if alias_map:
        canonical_node = alias_map.get(entity_raw) or alias_map.get(entity_raw.casefold())
        if canonical_node is not None:
            via_alias = weight_map.get(canonical_node)
            if via_alias is not None:
                return via_alias, "alias"
    return None, "not_found"


def tier_and_weight(score: float) -> tuple[str, float]:
    """Map a percentile-rank score in [0, 1] to (tier_name, weight)."""
    for cutoff, tier, weight in _TIER_CUTOFFS:
        if score >= cutoff:
            return tier, weight
    tier, weight = _TIER_CUTOFFS[-1][1], _TIER_CUTOFFS[-1][2]
    return tier, weight


def build_benchmark_weight_block(
    sample: QuestionSample,
    weight_map: dict[str, float],
    alias_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the ``metadata.benchmark_weight`` payload for one sample."""
    entity_raw, anchor_role = select_anchor_entity(sample)
    score, match_strategy = lookup_with_fallback(entity_raw, weight_map, alias_map)
    if score is None:
        tier, weight = _NOT_FOUND_TIER
        percentile = None
    else:
        tier, weight = tier_and_weight(score)
        percentile = round(score, 6)
    return {
        "weight": weight,
        "tier": tier,
        "anchor_entity": entity_raw,
        "anchor_role": anchor_role,
        "pagerank_percentile": percentile,
        "match_strategy": match_strategy,
        "centrality_source": CENTRALITY_SOURCE_VERSION,
    }
