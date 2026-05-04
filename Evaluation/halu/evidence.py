"""Evidence chain for hallucination judging.

Layers:
  1. ``supporting_chunk`` — gold chunks from sample.provenance, filtered by
     whole-word concept match.
  2. ``graph`` — 1-hop neighborhood of the concept's nearest node in
     ``global_graph.graphml`` (BGE cosine ≥ floor).
  3. ``web``        — stub (phase-2 todo, wire through agent_workflow_full).
  4. ``literature`` — stub (phase-2 todo).

Each layer returns ``list[Evidence]``. The chain short-circuits on the
FIRST layer that returns a non-empty list.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from .types import ConceptBucket, Evidence

if TYPE_CHECKING:
    from .graph_kb import GraphKB


# ---------------------------------------------------------------------------
# Layer 1: sample.provenance.supporting_chunks
# ---------------------------------------------------------------------------
def _text_mentions_concept(text: str, canonical_concept: str) -> bool:
    """Cheap substring containment check using word-boundary regex.

    ``canonical_concept`` is already normalize_keyword-d (lowercase,
    whitespace-collapsed). We look for it as a whole-word substring inside
    the chunk text (also lowercased) — avoids e.g. "NS3" matching inside
    "ns3-protease-inhibitor" which IS desired, while preventing "p7" from
    matching "dp7k".
    """
    if not canonical_concept or not text:
        return False
    # Token split on canonical — require ALL tokens present (order-free).
    tokens = [t for t in canonical_concept.split() if t]
    if not tokens:
        return False
    lower = text.lower()
    for tok in tokens:
        if not re.search(r"\b" + re.escape(tok) + r"\b", lower):
            return False
    return True


def layer_supporting_chunks(
    sample: dict[str, Any],
    canonical_concept: str,
    *,
    max_chunks: int = 3,
    max_chars_per_chunk: int = 1500,
) -> list[Evidence]:
    """Filter gold supporting_chunks by concept mention.

    Returns at most ``max_chunks`` chunks whose text mentions ``canonical_concept``.
    If NO chunks mention the concept, returns [] — DO NOT fall back to "all
    chunks", because that would make the judge see irrelevant evidence and
    could flip a correct judgement.
    """
    prov = sample.get("provenance") or {}
    chunks = prov.get("supporting_chunks") or []
    out: list[Evidence] = []
    for c in chunks:
        text = str((c or {}).get("text", "")).strip()
        if not text:
            continue
        if not _text_mentions_concept(text, canonical_concept):
            continue
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "...[truncated]"
        out.append(
            Evidence(
                source="supporting_chunk",
                text=text,
                score=1.0,  # gold — perfect relevance for the source question
            )
        )
        if len(out) >= max_chunks:
            break
    return out


# ---------------------------------------------------------------------------
# Layers 2–4: stubs for phase-2 fill-in
# ---------------------------------------------------------------------------
async def layer_graph_1hop(
    sample: dict[str, Any],
    canonical_concept: str,
    *,
    graph_kb: "GraphKB | None" = None,
) -> list[Evidence]:
    """Nearest-node lookup + 1-hop dump from the global graph.

    ``graph_kb`` lazy-loads the graph + node embeddings on first call. Cosine
    below ``graph_kb.cosine_floor`` → treat as miss (return []).
    """
    if graph_kb is None:
        return []
    # Wrap in to_thread: nearest_node may embed on first call (blocking HTTP).
    def _lookup():
        nid, _cos = graph_kb.nearest_node(canonical_concept)
        if nid is None:
            return []
        return graph_kb.one_hop_evidence(nid)

    return await asyncio.to_thread(_lookup)


_WEB_SEARCH_FN = None
_LIT_SEARCH_FN = None
_WEB_DISABLED_REASON = ""
_LIT_DISABLED_REASON = ""


def _lazy_load_search_fns() -> None:
    """Lazy-import web_search / literature_search so missing SDKs don't break the CLI."""
    global _WEB_SEARCH_FN, _LIT_SEARCH_FN, _WEB_DISABLED_REASON, _LIT_DISABLED_REASON
    if _WEB_SEARCH_FN is not None or _WEB_DISABLED_REASON:
        return
    try:
        from evaluation.agent_workflow_full import literature_search as _ls
        from evaluation.agent_workflow_full import web_search as _ws

        _WEB_SEARCH_FN = _ws
        _LIT_SEARCH_FN = _ls
    except Exception as exc:  # noqa: BLE001
        _WEB_DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        _LIT_DISABLED_REASON = _WEB_DISABLED_REASON


def _build_query(canonical_concept: str, claim_texts: list[str]) -> str:
    """Use the most specific claim as query; fall back to concept."""
    if claim_texts:
        head = claim_texts[0].strip()
        return head[:220] if head else canonical_concept
    return canonical_concept


def _format_search_output(
    canonical_concept: str,
    raw: str,
    *,
    source: str,
    max_chars: int = 2500,
) -> list[Evidence]:
    """Wrap a search-tool string response into a single Evidence object."""
    text = (raw or "").strip()
    if not text:
        return []
    low = text.lower()
    if low.startswith("error:") or "no results" in low[:80]:
        return []
    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return [Evidence(source=source, text=f"Query: {canonical_concept}\n\n{text}", score=0.5)]  # type: ignore[arg-type]


async def layer_web(
    canonical_concept: str,
    claim_texts: list[str],
    *,
    num_results: int = 5,
    timeout_s: float = 30.0,
) -> list[Evidence]:
    """Call ``websearch_tools.web_search`` (via agent_workflow_full)."""
    _lazy_load_search_fns()
    if _WEB_SEARCH_FN is None:
        return []
    query = _build_query(canonical_concept, claim_texts)
    try:
        raw = await asyncio.wait_for(_WEB_SEARCH_FN(query=query, num_results=num_results), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        print(f"[halu.evidence.web] error: {type(exc).__name__}: {exc}")
        return []
    return _format_search_output(canonical_concept, raw, source="web")


async def layer_literature(
    canonical_concept: str,
    claim_texts: list[str],
    *,
    num_results: int = 5,
    timeout_s: float = 45.0,
) -> list[Evidence]:
    """Call ``sciverse_tools.literature_search`` (via agent_workflow_full)."""
    _lazy_load_search_fns()
    if _LIT_SEARCH_FN is None:
        return []
    query = _build_query(canonical_concept, claim_texts)
    try:
        raw = await asyncio.wait_for(_LIT_SEARCH_FN(query=query, num_results=num_results), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001
        print(f"[halu.evidence.literature] error: {type(exc).__name__}: {exc}")
        return []
    return _format_search_output(canonical_concept, raw, source="literature")


# ---------------------------------------------------------------------------
# Chain orchestrator
# ---------------------------------------------------------------------------
async def gather_evidence_for_bucket(
    bucket: ConceptBucket,
    sample: dict[str, Any],
    *,
    chain: list[str],
    graph_kb: "GraphKB | None" = None,
) -> tuple[list[Evidence], str]:
    """Walk ``chain`` in order; return the FIRST non-empty hit.

    ``chain`` is a list of layer names: ``["supporting_chunk","graph","web","literature"]``.
    Aliases ``supporting_chunks`` / ``graph_1hop`` are also accepted.

    Returns ``(evidence, source_used)``. If every layer is empty, returns
    ``([], "")`` and the judge will verdict ``unverifiable``.
    """
    claim_texts = [c.text for c in bucket.claims]
    for layer_name in chain:
        lname = layer_name.strip().lower().replace("_1hop", "").replace("_chunks", "").replace("chunks", "")
        if lname in ("supporting_chunk", "support"):
            ev = layer_supporting_chunks(sample, bucket.canonical_concept)
            src = "supporting_chunk"
        elif lname == "graph":
            ev = await layer_graph_1hop(sample, bucket.canonical_concept, graph_kb=graph_kb)
            src = "graph_1hop"
        elif lname == "web":
            ev = await layer_web(bucket.canonical_concept, claim_texts)
            src = "web"
        elif lname == "literature":
            ev = await layer_literature(bucket.canonical_concept, claim_texts)
            src = "literature"
        else:
            continue
        if ev:
            return ev, src
    return [], ""
