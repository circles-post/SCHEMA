from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any

_logger = logging.getLogger(__name__)

from external_agent.schemas import Claim, EvidenceBundle
from external_agent.strategies import ClaimJudgeStrategy


_CACHE_MISS = object()


class _EvidenceCache:
    """Bounded LRU cache for evidence-provider API results.

    Two such caches are created per process (one for web_search results,
    one for literature_search results). Both are keyed by a normalized
    string built from the query + relevant config, and return the raw
    object produced by the underlying API so the caller can re-derive
    claim-specific fields (relevance check, truncation, framing) on top
    without re-hitting the network.

    Thread-safe — concurrent judge pipelines in the same worker share
    the cache; inner operations are dict ops so a plain threading.Lock
    is enough. Cache size is env-tunable via
    ``AGDEBUGGER_JUDGE_EVIDENCE_CACHE_SIZE`` (set to 0 to disable).
    """

    def __init__(self, max_size: int = 1000, *, label: str = "") -> None:
        self.max_size = max(0, int(max_size))
        self.label = label
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Any:
        if self.max_size <= 0 or not key:
            return _CACHE_MISS
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
        return _CACHE_MISS

    def set(self, key: str, value: Any) -> None:
        if self.max_size <= 0 or not key:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "label": self.label,
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


def _default_cache_size() -> int:
    try:
        return max(0, int(os.environ.get("AGDEBUGGER_JUDGE_EVIDENCE_CACHE_SIZE", "1000")))
    except ValueError:
        return 1000


_WEB_CACHE = _EvidenceCache(_default_cache_size(), label="web")
_LITERATURE_CACHE = _EvidenceCache(_default_cache_size(), label="literature")
# Path A — cache for sciverse_fetch_markdown (paper full-text via MinerU).
# This path is opt-in and expensive (per-call ~10-60s, 10-50KB output), so
# even a small cache pays off heavily — same query within/across a question's
# debug loop should not re-download.
_LITERATURE_FETCH_CACHE = _EvidenceCache(_default_cache_size(), label="literature_fetch")


def _reset_evidence_caches_for_tests() -> None:
    """Test hook — lets tests monkeypatch the env var and rebuild the caches
    with the new size so size changes actually take effect."""
    global _WEB_CACHE, _LITERATURE_CACHE, _LITERATURE_FETCH_CACHE
    size = _default_cache_size()
    _WEB_CACHE = _EvidenceCache(size, label="web")
    _LITERATURE_CACHE = _EvidenceCache(size, label="literature")
    _LITERATURE_FETCH_CACHE = _EvidenceCache(size, label="literature_fetch")


def get_evidence_cache_stats() -> dict:
    """Expose cache stats for diagnostics (called from run summary logs)."""
    return {
        "web": _WEB_CACHE.stats(),
        "literature": _LITERATURE_CACHE.stats(),
        "literature_fetch": _LITERATURE_FETCH_CACHE.stats(),
    }


_SCIVERSE_LITERATURE_SEARCH = None
_SCIVERSE_LITERATURE_SEARCH_IMPORT_FAILED = False
_SCIVERSE_FETCH_MARKDOWN = None
_SCIVERSE_FETCH_MARKDOWN_IMPORT_FAILED = False
_DEFAULT_SCIVERSE_DIR = "/mnt/shared-storage-user/fengxinshun/AISci/sciverse"


def _load_sciverse_literature_search():
    """Lazy-import sciverse_tools.literature_search so this module still
    loads standalone (e.g. in tests) even when the sibling sciverse repo
    is unavailable. Returns ``None`` if the import cannot be satisfied;
    the caller should silently skip the literature branch in that case.

    Note on the sys.path fallback: the AGDebugger backend process inherits
    its path from ``test_agent_debug.py``, which explicitly adds the
    sibling sciverse repo to ``sys.path``. The runner process
    (``run_dataset_autodebug.py``) does NOT import ``test_agent_debug`` —
    it speaks HTTP to the backend. So the first import attempt here fails
    for the judge pipeline even though the path exists on disk. To fix
    that we retry the import after inserting a candidate sciverse dir
    into ``sys.path`` (override via ``AGDEBUGGER_SCIVERSE_DIR`` env var).
    This was the root cause of 100%-skip on the 2026-04-20 full-bench run.
    """
    global _SCIVERSE_LITERATURE_SEARCH, _SCIVERSE_LITERATURE_SEARCH_IMPORT_FAILED
    if _SCIVERSE_LITERATURE_SEARCH is not None:
        return _SCIVERSE_LITERATURE_SEARCH
    if _SCIVERSE_LITERATURE_SEARCH_IMPORT_FAILED:
        return None

    # First attempt — assume sys.path is already good (backend-process case).
    try:
        from sciverse_tools import literature_search as _ls  # type: ignore
        _SCIVERSE_LITERATURE_SEARCH = _ls
        return _ls
    except Exception as first_exc:  # noqa: BLE001
        _logger.debug("[evidence] first literature_search import failed: %s", first_exc)

    # Second attempt — add the sciverse dir to sys.path and try again.
    candidate_dir = os.environ.get("AGDEBUGGER_SCIVERSE_DIR", _DEFAULT_SCIVERSE_DIR)
    candidate_dir = os.path.expanduser(str(candidate_dir).strip())
    if candidate_dir and os.path.isdir(candidate_dir):
        import sys
        if candidate_dir not in sys.path:
            sys.path.insert(0, candidate_dir)
        try:
            from sciverse_tools import literature_search as _ls  # type: ignore
            _SCIVERSE_LITERATURE_SEARCH = _ls
            _logger.info(
                "[evidence] literature_search loaded after adding %s to sys.path",
                candidate_dir,
            )
            return _ls
        except Exception as second_exc:  # noqa: BLE001
            _logger.info(
                "[evidence] literature_search unavailable even after sys.path fix (%s): %s",
                candidate_dir,
                second_exc,
            )
    else:
        _logger.info(
            "[evidence] literature_search unavailable; AGDEBUGGER_SCIVERSE_DIR=%r not a directory",
            candidate_dir,
        )

    _SCIVERSE_LITERATURE_SEARCH_IMPORT_FAILED = True
    return None


def _reset_sciverse_literature_search_for_tests() -> None:
    """Test hook — let a test monkeypatch the env var and rebuild the
    module-level cache so the next call re-evaluates the import."""
    global _SCIVERSE_LITERATURE_SEARCH, _SCIVERSE_LITERATURE_SEARCH_IMPORT_FAILED
    _SCIVERSE_LITERATURE_SEARCH = None
    _SCIVERSE_LITERATURE_SEARCH_IMPORT_FAILED = False


def _load_sciverse_fetch_markdown():
    """Path A loader — lazy import for ``sciverse_tools.sciverse_fetch_markdown``.

    Same two-stage import + sys.path fallback as
    ``_load_sciverse_literature_search`` (see that docstring). Returning
    ``None`` puts the caller on the no-op path so an unavailable sciverse
    install never blocks the judge pipeline.
    """
    global _SCIVERSE_FETCH_MARKDOWN, _SCIVERSE_FETCH_MARKDOWN_IMPORT_FAILED
    if _SCIVERSE_FETCH_MARKDOWN is not None:
        return _SCIVERSE_FETCH_MARKDOWN
    if _SCIVERSE_FETCH_MARKDOWN_IMPORT_FAILED:
        return None
    try:
        from sciverse_tools import sciverse_fetch_markdown as _fm  # type: ignore
        _SCIVERSE_FETCH_MARKDOWN = _fm
        return _fm
    except Exception as first_exc:  # noqa: BLE001
        _logger.debug("[evidence] first sciverse_fetch_markdown import failed: %s", first_exc)

    candidate_dir = os.environ.get("AGDEBUGGER_SCIVERSE_DIR", _DEFAULT_SCIVERSE_DIR)
    candidate_dir = os.path.expanduser(str(candidate_dir).strip())
    if candidate_dir and os.path.isdir(candidate_dir):
        import sys
        if candidate_dir not in sys.path:
            sys.path.insert(0, candidate_dir)
        try:
            from sciverse_tools import sciverse_fetch_markdown as _fm  # type: ignore
            _SCIVERSE_FETCH_MARKDOWN = _fm
            _logger.info(
                "[evidence] sciverse_fetch_markdown loaded after adding %s to sys.path",
                candidate_dir,
            )
            return _fm
        except Exception as second_exc:  # noqa: BLE001
            _logger.info(
                "[evidence] sciverse_fetch_markdown unavailable even after sys.path fix (%s): %s",
                candidate_dir,
                second_exc,
            )
    else:
        _logger.info(
            "[evidence] sciverse_fetch_markdown unavailable; AGDEBUGGER_SCIVERSE_DIR=%r not a directory",
            candidate_dir,
        )

    _SCIVERSE_FETCH_MARKDOWN_IMPORT_FAILED = True
    return None


def _reset_sciverse_fetch_markdown_for_tests() -> None:
    """Test hook, mirror of the literature_search reset."""
    global _SCIVERSE_FETCH_MARKDOWN, _SCIVERSE_FETCH_MARKDOWN_IMPORT_FAILED
    _SCIVERSE_FETCH_MARKDOWN = None
    _SCIVERSE_FETCH_MARKDOWN_IMPORT_FAILED = False


class WebSearchEvidenceProvider:
    """Evidence provider backed by the local websearch package + optional
    scholarly literature metadata from sciverse.

    Historically this provider only did plain web_search → snippet extraction
    (the sciverse-only refactor removed direct PubMed / Crossref / bioRxiv /
    MinerU calls; those are the agent-under-debug's job via `literature_fetch`).

    With ``AGDEBUGGER_JUDGE_USE_LITERATURE=on`` (default: ``off``) we ALSO
    call ``sciverse_tools.literature_search`` per-claim to get paper
    metadata (title/authors/year/DOI/abstract snippets) and append it to
    the evidence bundle. Only metadata is fetched — full-text download
    (the slow MinerU path) still belongs to the agent, not the judge.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        serper_api_key: str | None = None,
        bright_data_api_key: str | None = None,
        bright_data_zone: str | None = None,
        backend: str = "bright_data",
        max_searches: int = 3,
        num_results: int = 5,
        fetch_top_n: int = 2,
        max_output_words: int = 1500,
    ) -> None:
        from websearch import OpenAIAdapterSampler, WebSearcher

        self._WebSearcher = WebSearcher
        self._OpenAIAdapterSampler = OpenAIAdapterSampler
        self.model = model
        self.api_key = api_key or os.environ.get("AGENTDEBUG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("AGENTDEBUG_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY")
        self.bright_data_api_key = bright_data_api_key or os.environ.get("BRIGHT_DATA_API_KEY")
        self.bright_data_zone = bright_data_zone or os.environ.get("BRIGHT_DATA_ZONE")
        self.backend = backend
        self.max_searches = max_searches
        self.num_results = num_results
        self.fetch_top_n = fetch_top_n
        self.max_output_words = max_output_words
        self.embedding_api_key, self.embedding_base_url = self._resolve_embedding_client_config()
        self._searcher = None
        self._sampler = None
        # Lazy-configured literature knobs (toggle via env var). The max
        # results cap is intentionally small — we only need a few candidate
        # titles + abstract snippets for the judge, not full bibliography.
        self._use_literature = os.environ.get(
            "AGDEBUGGER_JUDGE_USE_LITERATURE", "off"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._literature_num_results = max(
            1, min(8, int(os.environ.get("AGDEBUGGER_JUDGE_LITERATURE_NUM_RESULTS", "3")))
        )
        self._literature_timeout_sec = float(
            os.environ.get("AGDEBUGGER_JUDGE_LITERATURE_TIMEOUT_SEC", "30")
        )
        # Path A — full-text fetch for high-value claims. Disabled by default;
        # enabling it adds a per-claim sciverse_fetch_markdown call that goes
        # through MinerU and is 10-60s (vs <5s for literature_search). We only
        # fire it for claim categories where metadata is known to underspecify
        # the question — mapping_claim / constraint_claim — as measured on the
        # 100-sample 2026-04-21 stuck-case study.
        self._use_literature_fetch = os.environ.get(
            "AGDEBUGGER_JUDGE_USE_LITERATURE_FETCH", "off"
        ).strip().lower() in {"1", "true", "yes", "on"}
        raw_cats = os.environ.get(
            "AGDEBUGGER_JUDGE_LITERATURE_FETCH_CATEGORIES",
            "mapping_claim,constraint_claim",
        )
        self._literature_fetch_categories = {
            c.strip().lower() for c in raw_cats.split(",") if c.strip()
        }
        self._literature_fetch_num_results = max(
            1, min(5, int(os.environ.get("AGDEBUGGER_JUDGE_LITERATURE_FETCH_NUM_RESULTS", "2")))
        )
        self._literature_fetch_max_chars = max(
            1000, int(os.environ.get("AGDEBUGGER_JUDGE_LITERATURE_FETCH_MAX_CHARS", "8000"))
        )
        self._literature_fetch_timeout_sec = float(
            os.environ.get("AGDEBUGGER_JUDGE_LITERATURE_FETCH_TIMEOUT_SEC", "90")
        )

    @staticmethod
    def _is_intern_base_url(base_url: str | None) -> bool:
        if not base_url:
            return False
        lowered = base_url.lower()
        return "intern-ai.org.cn" in lowered

    def _resolve_embedding_client_config(self) -> tuple[str | None, str | None]:
        explicit_key = os.environ.get("AGENTDEBUG_OPENAI_API_KEY_EMBEDDING")
        explicit_base_url = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL_EMBEDDING")
        if explicit_key or explicit_base_url:
            return explicit_key or self.api_key, explicit_base_url or self.base_url

        if self._is_intern_base_url(self.base_url):
            fallback_key = os.environ.get("AGENTDEBUG_NON_INTERN_API_KEY")
            fallback_base_url = os.environ.get("AGENTDEBUG_NON_INTERN_BASE_URL")
            if fallback_key and fallback_base_url:
                return fallback_key, fallback_base_url

        return self.api_key, self.base_url

    @staticmethod
    def _normalize_query_fragment(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"(?i)<answer>.*?</answer>", "", cleaned)
        cleaned = re.sub(r"(?i)\bterminate\b", "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip(" |")

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            normalized = item.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            output.append(item)
        return output

    @staticmethod
    def _relevance_terms(text: str) -> list[str]:
        generic_words = {
            "agent", "understanding", "scientific", "concept", "provided", "correct", "incorrect", "study",
            "studies", "paper", "research", "result", "results", "using", "based", "through", "about",
            "these", "those", "which", "their", "there", "mention", "mentions", "question", "specific",
            "available", "tools", "tool", "directly", "address", "related", "comparison", "comparing",
            "and", "for", "the", "with", "from", "into", "that", "this", "than",
        }
        terms: list[str] = []
        for token in re.findall(r"[A-Za-z0-9+.-]+", text.lower()):
            token = token.strip(" .,+-")
            if not token:
                continue
            if token in generic_words:
                continue
            if len(token) < 3 and not any(ch.isdigit() for ch in token):
                continue
            terms.append(token)
        return WebSearchEvidenceProvider._dedupe_keep_order(terms)

    def _claim_relevance_terms(self, claim: Claim) -> list[str]:
        fragments = [
            self._build_scientific_claim_query(claim),
            str(claim.source_ref or ""),
            str(claim.text or ""),
            str(claim.original_statement or ""),
        ]
        terms: list[str] = []
        for fragment in fragments:
            terms.extend(self._relevance_terms(fragment))
        return self._dedupe_keep_order(terms)[:16]

    def _assess_evidence_relevance(
        self,
        claim: Claim,
        *,
        search_results: str,
        filtered_content: str,
    ) -> tuple[bool, list[str], list[str], str]:
        terms = self._claim_relevance_terms(claim)
        combined = f"{search_results}\n{filtered_content}".lower()
        if not combined.strip():
            return False, terms, [], "evidence search returned no usable content"

        matched_terms = [term for term in terms if term in combined]
        min_matches = max(1, int(os.environ.get("AGDEBUGGER_EVIDENCE_MIN_TERM_MATCHES", "2")))
        min_ratio = float(os.environ.get("AGDEBUGGER_EVIDENCE_MIN_TERM_RATIO", "0.35"))
        if not terms:
            return True, terms, matched_terms, ""

        sufficient = len(matched_terms) >= min_matches or (len(matched_terms) / len(terms)) >= min_ratio
        if sufficient:
            return True, terms, matched_terms, ""
        reason = (
            "retrieved evidence appears topically unrelated to the scientific claim "
            f"(matched relevance terms: {matched_terms or 'none'}; expected terms: {terms[:8]})"
        )
        return False, terms, matched_terms, reason

    def _build_scientific_claim_query(self, claim: Claim) -> str:
        fragments: list[str] = []
        for raw in (
            claim.source_ref,
            claim.text,
            claim.original_statement,
            claim.data.get("context_snippet", "") if isinstance(claim.data, dict) else "",
            claim.data.get("corresponding_action", "") if isinstance(claim.data, dict) else "",
        ):
            fragment = self._normalize_query_fragment(str(raw or ""))
            if not fragment:
                continue
            if any(fragment == existing or fragment in existing or existing in fragment for existing in fragments):
                continue
            fragments.append(fragment)

        technical_terms: list[str] = []
        content_terms: list[str] = []
        generic_words = {
            "agent", "understanding", "scientific", "concept", "option", "options", "evaluation",
            "provided", "correct", "incorrect", "match", "matches", "fact", "facts", "study",
            "studies", "paper", "research", "result", "results", "show", "shows", "shown",
            "using", "based", "through", "about", "these", "those", "which", "their", "there",
            "mention", "mentions", "none", "only", "that", "this", "with", "from", "into",
        }
        for fragment in fragments:
            technical_terms.extend(
                re.findall(r"\b(?:[A-Za-z]+[A-Za-z0-9-]*\d+[A-Za-z0-9-]*|[A-Z]{2,6}|[A-Za-z]+-[A-Za-z]+|\d+-\d+)\b", fragment)
            )
            for token in re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", fragment):
                if token.lower() in generic_words:
                    continue
                content_terms.append(token)

        query_terms = self._dedupe_keep_order(technical_terms)[:10]
        query_terms.extend(term for term in self._dedupe_keep_order(content_terms) if term.lower() not in {t.lower() for t in query_terms})
        query = " ".join(query_terms[:18]).strip()
        if not query:
            query = " ".join(fragments)
        query = re.sub(r"\s+", " ", query).strip()
        if len(query) <= 400:
            return query
        return query[:400].rsplit(" ", 1)[0].strip()

    def _build_search_query(self, claim: Claim, strategy: ClaimJudgeStrategy) -> tuple[str, bool]:
        if strategy.name == "scientific_concept_discovery":
            heuristic_query = self._build_scientific_claim_query(claim)
            if heuristic_query:
                return heuristic_query, False
        return strategy.render_claim(claim), self._sampler is not None

    async def __aenter__(self) -> "WebSearchEvidenceProvider":
        if self.backend == "bright_data":
            if not self.bright_data_api_key:
                raise RuntimeError("Bright Data backend requested but BRIGHT_DATA_API_KEY is not set.")
            if not self.bright_data_zone:
                raise RuntimeError("Bright Data backend requested but BRIGHT_DATA_ZONE is not set.")
            serper_api_key = None
            bright_data_api_key = self.bright_data_api_key
            bright_data_zone = self.bright_data_zone
        elif self.backend == "serper":
            if not self.serper_api_key:
                raise RuntimeError("Serper backend requested but SERPER_API_KEY is not set.")
            serper_api_key = self.serper_api_key
            bright_data_api_key = None
            bright_data_zone = None
        else:
            raise ValueError(f"Unsupported search backend: {self.backend}")

        self._searcher = self._WebSearcher(
            serper_api_key=serper_api_key,
            bright_data_api_key=bright_data_api_key,
            bright_data_zone=bright_data_zone,
            openai_api_key=self.embedding_api_key,
            openai_base_url=self.embedding_base_url,
        )
        await self._searcher.start()
        if self.backend == "bright_data":
            backend_client = getattr(self._searcher, "_serper", None)
            if backend_client is None or backend_client.__class__.__name__ != "BrightDataSearchClient":
                raise RuntimeError(
                    "Bright Data backend was requested, but WebSearcher did not initialize "
                    "BrightDataSearchClient."
                )
        if self.api_key:
            self._sampler = self._OpenAIAdapterSampler(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
            )
        return self

    async def __aexit__(self, *_) -> None:
        if self._searcher is not None:
            await self._searcher.close()
            self._searcher = None
        self._sampler = None

    async def _fetch_literature_metadata(self, query: str) -> tuple[str, dict]:
        """Call sciverse literature_search for paper metadata (title,
        authors, year, DOI, venue, abstract). Returns
        (formatted_block, metadata_dict). Metadata always populated so the
        caller can record WHY the branch was skipped for post-run case study.
        """
        meta: dict = {
            "literature_used": False,
            "literature_query": query,
            "literature_skip_reason": None,
            "literature_results_count": 0,
        }
        if not self._use_literature:
            meta["literature_skip_reason"] = "disabled"
            return "", meta
        if not query:
            meta["literature_skip_reason"] = "empty_query"
            return "", meta
        search_fn = _load_sciverse_literature_search()
        if search_fn is None:
            meta["literature_skip_reason"] = "sciverse_tools_unavailable"
            return "", meta
        cache_key = f"{query.strip().lower()}|n={self._literature_num_results}"
        cached = _LITERATURE_CACHE.get(cache_key)
        if cached is not _CACHE_MISS:
            cached_text, cached_count = cached
            meta["literature_used"] = True
            meta["literature_results_count"] = cached_count
            meta["literature_cache_hit"] = True
            return cached_text, meta

        try:
            raw = await asyncio.wait_for(
                search_fn(query, self._literature_num_results),
                timeout=self._literature_timeout_sec,
            )
        except asyncio.TimeoutError:
            meta["literature_skip_reason"] = "timeout"
            return "", meta
        except Exception as exc:  # noqa: BLE001
            meta["literature_skip_reason"] = f"error:{type(exc).__name__}"
            _logger.info("[evidence] literature_search failed: %s", exc)
            return "", meta
        text = str(raw or "").strip()
        if not text:
            meta["literature_skip_reason"] = "empty_results"
            return "", meta
        # literature_search returns lines like `[1] Title\nAuthors:...\n`.
        # Count result entries by the "[n]" marker for the metadata field.
        result_count = len(re.findall(r"^\[\d+\]", text, flags=re.MULTILINE))
        meta["literature_used"] = True
        meta["literature_results_count"] = result_count
        meta["literature_cache_hit"] = False
        _LITERATURE_CACHE.set(cache_key, (text, result_count))
        return text, meta

    async def _fetch_literature_markdown(
        self, query: str, claim: Claim
    ) -> tuple[str, dict]:
        """Path A — call ``sciverse_fetch_markdown`` to retrieve paper
        full-text (markdown-converted via MinerU) for high-value claim
        categories. Gated three ways:

        1. Master switch ``AGDEBUGGER_JUDGE_USE_LITERATURE_FETCH=on``.
        2. ``claim.category`` must be in
           ``AGDEBUGGER_JUDGE_LITERATURE_FETCH_CATEGORIES`` (default
           mapping_claim / constraint_claim — these are where metadata
           alone underspecifies the question per the 2026-04-21 stuck-case
           study).
        3. Query must be non-empty and sciverse must importable.

        Each paper is truncated to ``max_markdown_chars`` (default 8000) and
        the combined block is capped to ``num_results`` papers. Returns
        ``(formatted_block, meta)``. The meta dict always populated so the
        caller can log WHY the branch was skipped.
        """
        meta: dict = {
            "literature_fetch_used": False,
            "literature_fetch_query": query,
            "literature_fetch_skip_reason": None,
            "literature_fetch_papers_count": 0,
            "literature_fetch_chars_total": 0,
            "literature_fetch_category": claim.category,
        }
        if not self._use_literature_fetch:
            meta["literature_fetch_skip_reason"] = "disabled"
            return "", meta
        category_norm = (claim.category or "").strip().lower()
        if self._literature_fetch_categories and category_norm not in self._literature_fetch_categories:
            meta["literature_fetch_skip_reason"] = f"category_not_in_allowlist:{category_norm}"
            return "", meta
        if not query:
            meta["literature_fetch_skip_reason"] = "empty_query"
            return "", meta
        fetch_fn = _load_sciverse_fetch_markdown()
        if fetch_fn is None:
            meta["literature_fetch_skip_reason"] = "sciverse_fetch_unavailable"
            return "", meta
        cache_key = (
            f"{query.strip().lower()}|n={self._literature_fetch_num_results}"
            f"|mc={self._literature_fetch_max_chars}"
        )
        cached = _LITERATURE_FETCH_CACHE.get(cache_key)
        if cached is not _CACHE_MISS:
            cached_text, cached_count, cached_chars = cached
            meta["literature_fetch_used"] = True
            meta["literature_fetch_papers_count"] = cached_count
            meta["literature_fetch_chars_total"] = cached_chars
            meta["literature_fetch_cache_hit"] = True
            return cached_text, meta
        try:
            raw = await asyncio.wait_for(
                fetch_fn(
                    query,
                    num_results=self._literature_fetch_num_results,
                    max_success=self._literature_fetch_num_results,
                    include_markdown_content=True,
                    max_markdown_chars=self._literature_fetch_max_chars,
                ),
                timeout=self._literature_fetch_timeout_sec,
            )
        except asyncio.TimeoutError:
            meta["literature_fetch_skip_reason"] = "timeout"
            return "", meta
        except Exception as exc:  # noqa: BLE001
            meta["literature_fetch_skip_reason"] = f"error:{type(exc).__name__}"
            _logger.info("[evidence] sciverse_fetch_markdown failed: %s", exc)
            return "", meta
        text = str(raw or "").strip()
        if not text:
            meta["literature_fetch_skip_reason"] = "empty_results"
            return "", meta
        paper_count = len(re.findall(r"^##\s+\[\d+\]|^\[\d+\]", text, flags=re.MULTILINE))
        if paper_count == 0:
            paper_count = 1  # raw markdown with no "[n]" marker still counts
        meta["literature_fetch_used"] = True
        meta["literature_fetch_papers_count"] = paper_count
        meta["literature_fetch_chars_total"] = len(text)
        meta["literature_fetch_cache_hit"] = False
        _LITERATURE_FETCH_CACHE.set(cache_key, (text, paper_count, len(text)))
        return text, meta

    async def build_evidence(self, claim: Claim, strategy: ClaimJudgeStrategy) -> EvidenceBundle:
        if self._searcher is None:
            raise RuntimeError("WebSearchEvidenceProvider not started. Use 'async with ...'.")

        web_query, prefer_sampler = self._build_search_query(claim, strategy)
        use_sampler = self._sampler if prefer_sampler else None
        max_searches = self.max_searches
        if self.fetch_top_n <= 0:
            use_sampler = None
            max_searches = 1

        # Web-only evidence path. Paper-fetching responsibility now lives
        # entirely in the agent under debug via the literature_fetch tool.
        # The web cache key includes every knob that would change the
        # underlying API shape — same query + same knobs ⇒ identical result
        # ⇒ hit. Different claims often produce identical queries (via the
        # scientific_concept_discovery heuristic that strips to technical
        # terms) so this cache pays off within a single question's debug
        # loop and across questions that share vocabulary.
        web_cache_key = "|".join(
            [
                (web_query or "").strip().lower(),
                self.backend,
                f"ms={max_searches}",
                f"nr={self.num_results}",
                f"fn={self.fetch_top_n}",
                f"mw={self.max_output_words}",
                f"sampler={bool(use_sampler)}",
            ]
        )
        web_cache_hit = False
        cached_result = _WEB_CACHE.get(web_cache_key)
        if cached_result is not _CACHE_MISS:
            result = cached_result
            web_cache_hit = True
            _logger.info(
                "[evidence] web_search cache hit claim_id=%s query=%r",
                getattr(claim, "claim_id", ""),
                (web_query or "")[:200],
            )
        else:
            _logger.info(
                "[evidence] web_search for claim_id=%s query=%r",
                getattr(claim, "claim_id", ""),
                (web_query or "")[:200],
            )
            web_t0 = time.perf_counter()
            result = await self._searcher.search_fetch_and_filter(
                claim=web_query,
                sampler=use_sampler,
                max_searches=max_searches,
                num_results=self.num_results,
                fetch_top_n=self.fetch_top_n,
                max_output_words=self.max_output_words,
            )
            _logger.info(
                "[evidence] web_search done claim_id=%s elapsed_sec=%.2f",
                getattr(claim, "claim_id", ""),
                time.perf_counter() - web_t0,
            )
            _WEB_CACHE.set(web_cache_key, result)

        contents_parts = []
        for item in result.contents[: self.fetch_top_n + 1]:
            content = item.get("content", "")
            if not content:
                continue
            words = content.split()
            truncated = " ".join(words[:1200])
            contents_parts.append(
                f"=== {item.get('title') or item.get('url', '')} ===\n"
                f"URL: {item.get('url', '')}\n"
                f"Snippet: {item.get('snippet', '')}\n"
                f"{truncated}"
            )

        filtered_content = result.filtered_content.strip()
        if not filtered_content and contents_parts:
            filtered_content = "\n\n".join(contents_parts)
        if not filtered_content:
            filtered_content = (result.formatted_snippets or "").strip()

        # Literature-metadata branch. When AGDEBUGGER_JUDGE_USE_LITERATURE=on
        # we call sciverse_tools.literature_search with the same query the
        # web path used, then append paper metadata + abstract snippets as
        # an extra block inside ``filtered_content``. The judge LLM reads
        # it with the same <FILTERED_CONTENT>…</FILTERED_CONTENT> framing.
        literature_block, literature_meta = await self._fetch_literature_metadata(web_query)

        if literature_block:
            literature_formatted = (
                "=== Scholarly literature metadata (sciverse) ===\n" + literature_block
            ).strip()
            filtered_content = (
                f"{filtered_content}\n\n{literature_formatted}"
                if filtered_content
                else literature_formatted
            )

        # Path A — full-text fetch for high-value claims. Runs AFTER metadata
        # so the judge always has the cheap branch available as a fallback
        # even if the fetch path is disabled or the claim category is
        # out-of-allowlist.
        fetch_block, fetch_meta = await self._fetch_literature_markdown(web_query, claim)
        if fetch_block:
            fetch_formatted = (
                "=== Scholarly literature full-text (sciverse) ===\n" + fetch_block
            ).strip()
            filtered_content = (
                f"{filtered_content}\n\n{fetch_formatted}"
                if filtered_content
                else fetch_formatted
            )

        literature_mode_parts: list[str] = ["web"]
        if literature_meta.get("literature_used"):
            literature_mode_parts.append("sciverse_metadata")
        if fetch_meta.get("literature_fetch_used"):
            literature_mode_parts.append("sciverse_fulltext")
        literature_mode = "+".join(literature_mode_parts) if len(literature_mode_parts) > 1 else "web_only"

        metadata = {
            "search_backend": self.backend,
            "search_query": web_query,
            "search_query_mode": "planner" if use_sampler is not None else "direct",
            "queries_executed": result.queries_executed,
            "urls_fetched": result.urls_fetched,
            "usage": result.usage,
            "literature_mode": literature_mode,
            "web_cache_hit": web_cache_hit,
            **{f"judge_{k}": v for k, v in literature_meta.items()},
            **{f"judge_{k}": v for k, v in fetch_meta.items()},
        }

        if strategy.name == "scientific_concept_discovery":
            # Include the literature block in the relevance check so papers
            # returned by sciverse can rescue a claim whose plain web_search
            # snippets were off-topic.
            combined_for_relevance = filtered_content
            relevant, relevance_terms, matched_terms, insufficient_reason = self._assess_evidence_relevance(
                claim,
                search_results=result.formatted_snippets,
                filtered_content=combined_for_relevance,
            )
            metadata.update(
                {
                    "evidence_relevance_terms": relevance_terms,
                    "evidence_relevance_matches": matched_terms,
                    "evidence_insufficient": not relevant,
                    "evidence_insufficient_reason": insufficient_reason,
                }
            )
            if not relevant:
                return EvidenceBundle(
                    search_results="",
                    filtered_content="",
                    metadata=metadata,
                )

        return EvidenceBundle(
            search_results=result.formatted_snippets,
            filtered_content=filtered_content,
            metadata=metadata,
        )
