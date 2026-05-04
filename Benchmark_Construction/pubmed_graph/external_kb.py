"""External knowledge base wrappers used for entity grounding.

This module is the **fact-reliability layer** for the OntologyProposerAgent.
When the proposer wants to add a new entity type / alias / relation, it
asks an EntityCanonicalizer.resolve() to look up the candidate in real
literature *before* the proposal is allowed into ontology.run.yaml.

Backends:
  - sciverse  — paper search via /mnt/shared-storage-user/fengxinshun/AISci/
                sciverse/sciverse_tools.literature_search (async). Returns
                free-form citation snippets that establish "this concept
                appears in N independent biomedical papers".
  - pubmed    — esearch on the existing PubMedClient. Cheap, no extra deps.
  - mesh      — id.nlm.nih.gov/mesh/lookup/descriptor for canonical names
                of well-known disease/anatomy/chemistry concepts.

The EntityCanonicalizer ranks an entity as "grounded" when at least
`min_hits` independent sources return non-empty results. The default is
2, mirroring the evidence-threshold rule in the OntologyProposer plan.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .pubmed_client import PubMedClient
from .utils import normalize_keyword, normalize_text

# ---------------------------------------------------------------------------
# Sciverse adapter — runs the async tool from /mnt/shared-storage-user/fengxinshun/
# AISci/sciverse/sciverse_tools.py inside a synchronous helper. We do not
# import sciverse at module top because the toolkit is heavy and may not
# be available on every host. Lazy import on first use.
# ---------------------------------------------------------------------------

_DEFAULT_SCIVERSE_TOOLS_PATH = "/mnt/shared-storage-user/fengxinshun/AISci/sciverse"


def _ensure_sciverse_on_path(toolkit_root: str | None = None) -> None:
    root = Path(toolkit_root or _DEFAULT_SCIVERSE_TOOLS_PATH).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"sciverse toolkit root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # if we are already in an event loop, run the coroutine in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Evidence record
# ---------------------------------------------------------------------------

@dataclass
class GroundingEvidence:
    source: str            # "sciverse" | "pubmed" | "mesh"
    title: str
    snippet: str = ""
    identifier: str = ""   # DOI / PMID / MeSH descriptor UI
    confidence: float = 1.0


@dataclass
class GroundingResult:
    query: str
    hits: list[GroundingEvidence] = field(default_factory=list)
    canonical_candidates: list[str] = field(default_factory=list)
    sources_with_hits: set[str] = field(default_factory=set)
    grounded: bool = False
    rejected_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "grounded": self.grounded,
            "sources_with_hits": sorted(self.sources_with_hits),
            "hit_count": len(self.hits),
            "canonical_candidates": list(dict.fromkeys(self.canonical_candidates))[:5],
            "rejected_reason": self.rejected_reason,
            "hits": [
                {
                    "source": h.source,
                    "title": h.title,
                    "snippet": h.snippet[:300],
                    "identifier": h.identifier,
                }
                for h in self.hits[:10]
            ],
        }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class SciverseGroundingBackend:
    """Wraps the async literature_search tool from sciverse_tools.py."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.toolkit_root = str(cfg.get("toolkit_root") or _DEFAULT_SCIVERSE_TOOLS_PATH)
        self.num_results = max(int(cfg.get("num_results", 5)), 1)
        self.language = cfg.get("language") or None
        self.timeout_seconds = float(cfg.get("timeout_seconds", 60.0))
        self._search_fn = None

    def _ensure(self) -> None:
        if self._search_fn is not None:
            return
        _ensure_sciverse_on_path(self.toolkit_root)
        from sciverse_tools import literature_search  # type: ignore

        self._search_fn = literature_search

    def search(self, query: str) -> list[GroundingEvidence]:
        if not self.enabled:
            return []
        try:
            self._ensure()
        except Exception:
            return []
        try:
            text = _run_async(
                asyncio.wait_for(
                    self._search_fn(query=query, num_results=self.num_results, language=self.language),
                    timeout=self.timeout_seconds,
                )
            )
        except Exception:
            return []
        return self._parse_text_response(query, text or "")

    @staticmethod
    def _parse_text_response(query: str, text: str) -> list[GroundingEvidence]:
        if not text or text.lower().startswith("no literature results"):
            return []
        hits: list[GroundingEvidence] = []
        # The tool returns a header + per-paper blocks. Each paper block looks like:
        #   [1] Title here
        #   Authors: ...
        #   Year: ...
        #   DOI: 10.xxxx/...
        #   Venue: ...
        #   Snippet: ...
        block: dict[str, str] = {}

        def flush() -> None:
            if not block.get("title"):
                return
            hits.append(
                GroundingEvidence(
                    source="sciverse",
                    title=block.get("title", ""),
                    snippet=block.get("snippet", ""),
                    identifier=block.get("doi", ""),
                )
            )

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if block:
                    flush()
                    block = {}
                continue
            if line.startswith("[") and "]" in line:
                if block:
                    flush()
                    block = {}
                block["title"] = line.split("]", 1)[1].strip()
                continue
            for prefix, key in (
                ("DOI:", "doi"),
                ("Venue:", "venue"),
                ("Year:", "year"),
                ("Snippet:", "snippet"),
                ("Authors:", "authors"),
            ):
                if line.startswith(prefix):
                    block[key] = line[len(prefix):].strip()
                    break
        if block:
            flush()
        return hits


class PubMedGroundingBackend:
    """Use the existing PubMedClient.esearch as a fast biomedical reality check."""

    def __init__(self, config: dict[str, Any] | None = None, pubmed_client: PubMedClient | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.retmax = max(int(cfg.get("retmax", 3)), 1)
        self.client = pubmed_client or PubMedClient(api_key=cfg.get("api_key"), email=cfg.get("email"))

    def search(self, query: str) -> list[GroundingEvidence]:
        if not self.enabled:
            return []
        try:
            search_result = self.client.esearch(term=query, retmax=self.retmax)
        except Exception:
            return []
        pmids: list[str] = []
        if isinstance(search_result, dict):
            inner = search_result.get("esearchresult") or {}
            if isinstance(inner, dict):
                raw_ids = inner.get("idlist") or []
                pmids = [str(x) for x in raw_ids if x]
            if not pmids:
                # fall back to top-level idlist for older response shapes
                raw_ids = search_result.get("idlist") or []
                pmids = [str(x) for x in raw_ids if x]
        if not pmids:
            return []
        try:
            summaries_raw = self.client.esummary(pmids)
        except Exception:
            summaries_raw = {}
        # esummary returns {"result": {"uids": [...], "<pmid>": {...}}} when retmode=json
        result_block = {}
        if isinstance(summaries_raw, dict):
            result_block = summaries_raw.get("result") or summaries_raw
        hits: list[GroundingEvidence] = []
        for pmid in pmids:
            doc = result_block.get(pmid) if isinstance(result_block, dict) else None
            title = normalize_text((doc or {}).get("title") or (doc or {}).get("Title") or "")
            hits.append(
                GroundingEvidence(
                    source="pubmed",
                    title=title or f"PMID:{pmid}",
                    snippet="",
                    identifier=f"PMID:{pmid}",
                )
            )
        return hits


class MeshGroundingBackend:
    """Use NLM MeSH descriptor lookup for canonical biomedical names."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.base_url = str(cfg.get("base_url") or "https://id.nlm.nih.gov/mesh")
        self.limit = max(int(cfg.get("limit", 3)), 1)
        self.timeout = float(cfg.get("timeout", 15.0))
        self.session = requests.Session()

    def search(self, query: str) -> list[GroundingEvidence]:
        if not self.enabled:
            return []
        url = f"{self.base_url.rstrip('/')}/lookup/descriptor"
        params = {"label": query, "match": "contains", "limit": self.limit}
        try:
            rsp = self.session.get(url, params=params, timeout=self.timeout)
            rsp.raise_for_status()
            data = rsp.json()
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        hits: list[GroundingEvidence] = []
        for entry in data[: self.limit]:
            label = normalize_text(entry.get("label") or "")
            descriptor_uri = normalize_text(entry.get("resource") or "")
            descriptor_id = descriptor_uri.rsplit("/", 1)[-1] if descriptor_uri else ""
            if not label:
                continue
            hits.append(
                GroundingEvidence(
                    source="mesh",
                    title=label,
                    snippet="",
                    identifier=descriptor_id,
                )
            )
        return hits


# ---------------------------------------------------------------------------
# EntityCanonicalizer
# ---------------------------------------------------------------------------

class EntityCanonicalizer:
    """Resolve a candidate entity name against external knowledge bases.

    Used by OntologyProposerAgent.validate_proposal() to enforce the
    "any new entity must be backed by ≥N independent KB hits" rule from
    the refactor plan.

    Returns a GroundingResult; the proposer turns `grounded == True`
    into the gate for accepting a new alias/type into ontology.run.yaml.
    """

    def __init__(self, config: dict[str, Any] | None = None, pubmed_client: PubMedClient | None = None):
        cfg = dict(config or {})
        self.min_hits = max(int(cfg.get("min_hits", 2)), 1)
        self.min_distinct_sources = max(int(cfg.get("min_distinct_sources", 1)), 1)
        self.cache_dir = Path(cfg.get("cache_dir") or "/tmp/datasetsa_canonicalizer_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = float(cfg.get("sleep_seconds", 0.0))

        self.sciverse = SciverseGroundingBackend(cfg.get("sciverse") or {})
        self.pubmed = PubMedGroundingBackend(cfg.get("pubmed") or {}, pubmed_client=pubmed_client)
        self.mesh = MeshGroundingBackend(cfg.get("mesh") or {})

    def _cache_path(self, query: str) -> Path:
        from .utils import sha256_text

        return self.cache_dir / f"{sha256_text(normalize_keyword(query))}.json"

    def _load_cache(self, query: str) -> GroundingResult | None:
        path = self._cache_path(query)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        result = GroundingResult(
            query=data.get("query", query),
            grounded=bool(data.get("grounded")),
            rejected_reason=data.get("rejected_reason", ""),
            sources_with_hits=set(data.get("sources_with_hits") or []),
            canonical_candidates=list(data.get("canonical_candidates") or []),
        )
        for h in data.get("hits") or []:
            result.hits.append(
                GroundingEvidence(
                    source=h.get("source", ""),
                    title=h.get("title", ""),
                    snippet=h.get("snippet", ""),
                    identifier=h.get("identifier", ""),
                )
            )
        return result

    def _save_cache(self, result: GroundingResult) -> None:
        path = self._cache_path(result.query)
        try:
            path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def resolve(self, name: str, hint_type: str = "", evidence: str = "") -> GroundingResult:
        query = normalize_text(name)
        result = GroundingResult(query=query)
        if not query:
            result.rejected_reason = "empty name"
            return result

        cached = self._load_cache(query)
        if cached is not None:
            return cached

        for backend in (self.mesh, self.pubmed, self.sciverse):
            try:
                hits = backend.search(query)
            except Exception:
                hits = []
            if hits:
                result.hits.extend(hits)
                result.sources_with_hits.add(hits[0].source)
                for h in hits[:3]:
                    if h.title and h.title.lower() != query.lower():
                        result.canonical_candidates.append(h.title)
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

        if len(result.hits) >= self.min_hits and len(result.sources_with_hits) >= self.min_distinct_sources:
            result.grounded = True
        else:
            result.rejected_reason = (
                f"insufficient grounding: hits={len(result.hits)} sources={sorted(result.sources_with_hits)}"
            )

        # Only cache successful groundings. Caching failures would let one
        # transient backend outage permanently poison the canonicalizer for
        # legitimate concepts. Failures are cheap to retry next time.
        if result.grounded:
            self._save_cache(result)
        return result
