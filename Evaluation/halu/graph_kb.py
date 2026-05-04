"""Graph knowledge base for evidence layer 2 (1-hop) and HS_weighted.

Loads a ``global_graph.graphml`` once, embeds all node labels via
``BGELargeEmbedder`` (remote HTTP service by default), and exposes:

  * ``nearest_node(concept)``     → (node_id, cosine) | (None, 0.0)
  * ``one_hop_evidence(node_id)`` → list[Evidence] formatted as short triples
  * ``concept_weight(concept)``   → 1 + log(1 + degree)  (1.0 if miss)

Node embeddings are cached to ``<cache_dir>/bge_nodes.npy`` +
``bge_nodes.labels.txt`` so subsequent runs skip re-embedding (dominant cost
in a cold run — 1112 nodes × 1024-d ≈ 30 s against the remote service).
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EVAL_DIR = Path(__file__).resolve().parents[1]
_PARENT = _EVAL_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from pubmed_graph.embeddings import BGELargeEmbedder
from pubmed_graph.normalize import normalize_keyword

from .types import Evidence


def _fix_no_proxy_for_host(url: str) -> None:
    """Ensure host in ``url`` is in NO_PROXY so httpx/requests don't route via HTTP proxy."""
    if not url:
        return
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    for var in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(var, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        if host not in parts:
            parts.append(host)
            os.environ[var] = ",".join(parts)


class GraphKB:
    """Singleton-ish graph store. Cheap to construct; loads graph on first use."""

    def __init__(
        self,
        graphml_path: Path,
        *,
        cache_dir: Path | None,
        bge_service_url: str | None = None,
        bge_remote_model: str = "bge",
        bge_model_path: str | None = None,
        cosine_floor: float = 0.6,
    ) -> None:
        self.graphml_path = Path(graphml_path)
        self.cache_dir = cache_dir
        self.cosine_floor = float(cosine_floor)

        self._graph = None  # nx.Graph — lazy
        self._node_ids: list[str] = []
        self._node_vecs: np.ndarray | None = None  # shape (N, D), L2-normalized
        self._canonical_to_idx: dict[str, int] = {}

        cfg: dict[str, Any] = {
            "enabled": True,
            "instruction_prefix": "",
            "remote_model": bge_remote_model,
        }
        if bge_service_url:
            cfg["service_url"] = bge_service_url
            _fix_no_proxy_for_host(bge_service_url)
        if bge_model_path:
            cfg["model_path"] = bge_model_path
        self._embedder = BGELargeEmbedder(cfg)

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    def _load_graph(self) -> None:
        if self._graph is not None:
            return
        import networkx as nx

        self._graph = nx.read_graphml(self.graphml_path)
        self._node_ids = list(self._graph.nodes())
        # Quick canonical-equality lookup (no embedding needed if perfect hit).
        for i, nid in enumerate(self._node_ids):
            k = normalize_keyword(nid) or nid.lower().strip()
            # Last-wins on collision is fine — these are already unique node IDs.
            self._canonical_to_idx[k] = i
            # Also index aliases.
            nd = self._graph.nodes[nid]
            aliases = str(nd.get("aliases", "") or "")
            for a in aliases.split("|"):
                a = a.strip()
                if not a:
                    continue
                ak = normalize_keyword(a) or a.lower().strip()
                if ak and ak not in self._canonical_to_idx:
                    self._canonical_to_idx[ak] = i

    def _cache_paths(self) -> tuple[Path, Path] | tuple[None, None]:
        if self.cache_dir is None:
            return None, None
        key = hashlib.sha1(
            f"{self.graphml_path.resolve()}|{self.graphml_path.stat().st_mtime_ns}".encode()
        ).hexdigest()[:12]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"bge_nodes_{key}.npy", self.cache_dir / f"bge_nodes_{key}.labels.txt"

    def _load_or_build_node_vecs(self) -> None:
        if self._node_vecs is not None:
            return
        self._load_graph()
        vec_path, lbl_path = self._cache_paths()
        if vec_path is not None and vec_path.is_file() and lbl_path is not None and lbl_path.is_file():
            labels = lbl_path.read_text(encoding="utf-8").splitlines()
            if labels == self._node_ids:
                self._node_vecs = np.load(vec_path)
                return
            # Stale cache — re-embed.
        # Embed.
        print(f"[halu.graph_kb] embedding {len(self._node_ids)} node labels via BGE...", flush=True)
        vecs = self._embedder.embed_texts(self._node_ids)
        if not vecs:
            raise RuntimeError(
                "BGELargeEmbedder returned no vectors — check service_url reachability and remote_model."
            )
        arr = np.asarray(vecs, dtype=np.float32)
        # Ensure L2-normalized (BGELargeEmbedder already does, but double-guard).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self._node_vecs = arr
        if vec_path is not None:
            np.save(vec_path, arr)
            lbl_path.write_text("\n".join(self._node_ids), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def nearest_node(self, concept: str) -> tuple[str | None, float]:
        """Return (node_id, cosine) or (None, 0.0) if below floor."""
        self._load_graph()
        key = normalize_keyword(concept) or concept.lower().strip()
        if key in self._canonical_to_idx:
            idx = self._canonical_to_idx[key]
            return self._node_ids[idx], 1.0
        # Fallback: embed query + cosine.
        self._load_or_build_node_vecs()
        assert self._node_vecs is not None
        qv = self._embedder.embed_texts([concept])
        if not qv:
            return None, 0.0
        q = np.asarray(qv[0], dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return None, 0.0
        q = q / n
        sims = self._node_vecs @ q
        best = int(np.argmax(sims))
        cos = float(sims[best])
        if cos < self.cosine_floor:
            return None, cos
        return self._node_ids[best], cos

    def one_hop_evidence(
        self,
        node_id: str,
        *,
        max_edges: int = 10,
        max_chars_per_snippet: int = 300,
    ) -> list[Evidence]:
        self._load_graph()
        assert self._graph is not None
        if node_id not in self._graph:
            return []
        g = self._graph
        out: list[Evidence] = []
        # Outgoing edges first (SUBJ = node_id), then incoming.
        edges: list[tuple[str, str, dict]] = []
        for u, v, ed in g.out_edges(node_id, data=True) if g.is_directed() else g.edges(node_id, data=True):
            edges.append((u, v, ed))
        if g.is_directed():
            for u, v, ed in g.in_edges(node_id, data=True):
                edges.append((u, v, ed))
        seen: set[tuple[str, str, str]] = set()
        for u, v, ed in edges:
            rel = str(ed.get("relation", "related_to")).strip() or "related_to"
            sig = (u, rel, v)
            if sig in seen:
                continue
            seen.add(sig)
            weight = ed.get("weight")
            ev_snippet = str(ed.get("evidence", "") or "").strip()
            if len(ev_snippet) > max_chars_per_snippet:
                ev_snippet = ev_snippet[:max_chars_per_snippet] + "..."
            head = f"{u} --{rel}--> {v}"
            if weight is not None:
                head = f"{head}  (w={weight})"
            body = f"{head}\n{ev_snippet}" if ev_snippet else head
            out.append(
                Evidence(
                    source="graph_1hop",
                    text=body,
                    score=float(weight) if isinstance(weight, (int, float)) else 0.0,
                )
            )
            if len(out) >= max_edges:
                break
        return out

    def concept_weight(self, concept: str) -> float:
        """1 + log(1 + degree) for graph-connectivity weighting; 1.0 on miss."""
        nid, _cos = self.nearest_node(concept)
        if nid is None:
            return 1.0
        self._load_graph()
        assert self._graph is not None
        deg = int(self._graph.degree(nid))
        return 1.0 + math.log(1.0 + deg)

    def concept_info(self, concept: str) -> tuple[float, str]:
        """Return (weight, node_type). (1.0, "") on miss."""
        nid, _cos = self.nearest_node(concept)
        if nid is None:
            return 1.0, ""
        self._load_graph()
        assert self._graph is not None
        deg = int(self._graph.degree(nid))
        node_type = str(self._graph.nodes[nid].get("node_type", "") or "")
        return 1.0 + math.log(1.0 + deg), node_type
