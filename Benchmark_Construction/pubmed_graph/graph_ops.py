from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .embeddings import BGELargeEmbedder, NodeFusionEngine
from .models import TripleRecord
from .utils import read_jsonl

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    nx = None
    NETWORKX_IMPORT_ERROR = exc
else:  # pragma: no cover
    NETWORKX_IMPORT_ERROR = None


def _require_networkx() -> None:
    if nx is None:
        raise ImportError(f"networkx is required for graph operations: {NETWORKX_IMPORT_ERROR}")


_TRIPLE_FIELDS = {
    "doc_id", "chunk_id", "head", "head_type", "surface_relation",
    "normalized_relation", "tail", "tail_type", "confidence", "evidence",
    "source", "meta",
}


def load_triples(path: str | Path) -> list[TripleRecord]:
    rows = read_jsonl(path)
    return [TripleRecord(**{k: v for k, v in row.items() if k in _TRIPLE_FIELDS}) for row in rows]


def build_local_graphs(triples: list[TripleRecord]) -> dict[str, Any]:
    _require_networkx()
    graphs: dict[str, Any] = {}
    grouped: dict[str, list[TripleRecord]] = defaultdict(list)
    for triple in triples:
        grouped[triple.doc_id].append(triple)
    for doc_id, doc_triples in grouped.items():
        graph = nx.MultiDiGraph(doc_id=doc_id)
        for triple in doc_triples:
            for node, node_type in ((triple.head, triple.head_type), (triple.tail, triple.tail_type)):
                if graph.has_node(node):
                    existing = graph.nodes[node]
                    if not existing.get("node_type") and node_type:
                        existing["node_type"] = node_type
                    sources = set(existing.get("sources") or [existing.get("source") or "paper"])
                    sources.add(triple.source)
                    existing["sources"] = sorted(sources)
                else:
                    graph.add_node(
                        node,
                        node_type=node_type,
                        aliases=[node],
                        sources=[triple.source],
                    )
            edge_key = triple.normalized_relation or "associated_with"
            if graph.has_edge(triple.head, triple.tail, key=edge_key):
                edge = graph[triple.head][triple.tail][edge_key]
                edge["weight"] += float(triple.confidence)
                edge["max_confidence"] = max(edge["max_confidence"], float(triple.confidence))
                edge["evidence"].append(triple.evidence)
                sources = set(edge.get("sources") or [edge.get("source") or "paper"])
                sources.add(triple.source)
                edge["sources"] = sorted(sources)
                if triple.source != "paper" and triple.meta:
                    edge.setdefault("benchmark_meta", []).append(triple.meta)
            else:
                edge_attrs: dict[str, Any] = {
                    "relation": edge_key,
                    "weight": float(triple.confidence),
                    "max_confidence": float(triple.confidence),
                    "evidence": [triple.evidence],
                    "source": triple.source,
                    "sources": [triple.source],
                }
                if triple.source != "paper" and triple.meta:
                    edge_attrs["benchmark_meta"] = [triple.meta]
                graph.add_edge(triple.head, triple.tail, key=edge_key, **edge_attrs)
        graphs[doc_id] = graph
    return graphs


def compose_global_graph(local_graphs: dict[str, Any]) -> Any:
    _require_networkx()
    global_graph = nx.MultiDiGraph()
    for graph in local_graphs.values():
        global_graph = nx.compose(global_graph, graph)
    return global_graph


def fuse_global_graph(global_graph: Any, embedder: BGELargeEmbedder, threshold: float = 0.9) -> Any:
    _require_networkx()
    nodes = [{"text": node, "type": data.get("node_type")} for node, data in global_graph.nodes(data=True)]
    fusion_engine = NodeFusionEngine(embedder=embedder, threshold=threshold)
    groups = fusion_engine.fuse(nodes)
    mapping = {}
    for group in groups:
        for member in group.member_names:
            mapping[member] = group.canonical_name
    fused = nx.relabel_nodes(global_graph, mapping, copy=True)
    collapsed = nx.MultiDiGraph()
    for node, data in fused.nodes(data=True):
        aliases = sorted({node, *data.get("aliases", [])}) if isinstance(data.get("aliases"), list) else [node]
        if node not in collapsed:
            collapsed.add_node(node, node_type=data.get("node_type"), aliases=aliases)
        else:
            existing = set(collapsed.nodes[node].get("aliases", []))
            collapsed.nodes[node]["aliases"] = sorted(existing | set(aliases))
    for src, dst, key, data in fused.edges(keys=True, data=True):
        relation_key = key or data.get("relation") or "associated_with"
        if collapsed.has_edge(src, dst, key=relation_key):
            edge = collapsed[src][dst][relation_key]
            edge["weight"] += float(data.get("weight", 0.0))
            edge["max_confidence"] = max(
                float(edge.get("max_confidence", 0.0)),
                float(data.get("max_confidence", 0.0)),
            )
            edge["evidence"] = list(edge.get("evidence", [])) + list(data.get("evidence", []))
        else:
            collapsed.add_edge(src, dst, key=relation_key, **data)
    return collapsed


def prune_small_components(graph: Any, min_component_size: int = 3) -> Any:
    _require_networkx()
    keep = set()
    for component in nx.weakly_connected_components(graph):
        if len(component) >= min_component_size:
            keep.update(component)
    return graph.subgraph(keep).copy()


def export_graphml(graph: Any, path: str | Path) -> None:
    _require_networkx()
    import json as _json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned = graph.copy()
    for _, _, _, data in cleaned.edges(keys=True, data=True):
        if isinstance(data.get("evidence"), list):
            data["evidence"] = " || ".join(str(x) for x in data["evidence"])
        if isinstance(data.get("sources"), list):
            data["sources"] = ",".join(data["sources"])
        if isinstance(data.get("benchmark_meta"), list):
            data["benchmark_meta"] = _json.dumps(data["benchmark_meta"], ensure_ascii=False)
    for _, data in cleaned.nodes(data=True):
        if isinstance(data.get("aliases"), list):
            data["aliases"] = " | ".join(str(x) for x in data["aliases"])
        if isinstance(data.get("sources"), list):
            data["sources"] = ",".join(data["sources"])
    nx.write_graphml(cleaned, out)
