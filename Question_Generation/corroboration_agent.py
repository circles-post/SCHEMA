"""Runtime external-corroboration agent for question_generation.

Takes a claim text (typically "{head} {relation} {tail}") plus the set of
local source doc_ids that already produced the triple, and returns whether
at least one *independent* external source (not in the local set) can be
retrieved via the 4 external tools:

    literature_search  (scholarly metadata, fast, primary)
    web_search         (general web snippets, fallback)
    literature_fetch   (opt-in: full-text PDFs, slow)
    web_fetch          (opt-in: full web pages, slow)

Design notes
------------
1. The 4 tools live in a sibling project at ``/mnt/shared-storage-user/
   fengxinshun/AISci/...`` and are not pip-installed. We inject two
   ``sys.path`` entries at module import time. If import fails, the agent
   degrades to ``TOOLS_AVAILABLE=False`` and every ``corroborate_claim``
   call returns ``tool_unavailable`` immediately.

2. We deliberately do NOT import ``test_agent_debug`` because its
   module-level code sets ``os.environ['http_proxy']`` globally, which
   would pollute pipeline HTTP clients. Instead we implement a local
   ``_literature_fetch`` that calls ``sciverse_tools.sciverse_fetch_markdown``
   directly.

3. Each external call is wrapped in ``_scoped_proxy`` — sets http_proxy /
   https_proxy on entry, restores prior values on exit. Pipeline clients
   outside this block are unaffected.

4. Fail-closed: any exception or zero-result path → ``status='insufficient'``
   or ``status='tool_unavailable'``, never silently passes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .validation_cache import ValidationCache


logger = logging.getLogger("question_generation.corroboration")

# ---------------------------------------------------------------------------
# Tool discovery: inject sibling source dirs into sys.path and import.
# ---------------------------------------------------------------------------
_AGENTDEBUG_PATH = "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger"
_SCIVERSE_PATH = "/mnt/shared-storage-user/fengxinshun/AISci/sciverse"

TOOLS_AVAILABLE = False
_TOOL_IMPORT_ERROR: str | None = None
_web_search = None
_web_fetch = None
_literature_search = None
_sciverse_fetch_markdown = None

for _p in (_AGENTDEBUG_PATH, _SCIVERSE_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Auto-load SCIVERSE_API_TOKEN (and anything else) from the sibling .env
# file so a user who just sourced agentdebug env still gets the sciverse
# creds the toolkit expects. Never overwrites an already-set var.
_SCIVERSE_ENV_FILE = Path(_SCIVERSE_PATH) / ".env"
if _SCIVERSE_ENV_FILE.exists():
    try:
        for _line in _SCIVERSE_ENV_FILE.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip("'").strip('"')
            if _k and _k not in os.environ:
                os.environ[_k] = _v
    except Exception as _exc:  # pragma: no cover
        logger.warning("failed to parse sciverse .env: %s", _exc)

try:
    from websearch_tools import web_search as _web_search  # type: ignore
    from websearch_tools import web_fetch as _web_fetch  # type: ignore
    from sciverse_tools import literature_search as _literature_search  # type: ignore
    from sciverse_tools import sciverse_fetch_markdown as _sciverse_fetch_markdown  # type: ignore
    TOOLS_AVAILABLE = True
    logger.info("corroboration tools imported: web_search, web_fetch, literature_search, sciverse_fetch_markdown")
except Exception as exc:  # pragma: no cover — env-dependent
    _TOOL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    logger.warning("corroboration tools unavailable: %s", _TOOL_IMPORT_ERROR)


# ---------------------------------------------------------------------------
# Proxy scoping
# ---------------------------------------------------------------------------
# Default proxy URL — preference order:
#   1. QG_CORROBORATION_PROXY (explicit override for this feature)
#   2. The shell's existing http_proxy / HTTP_PROXY (what the user set)
#   3. The pjlab internal proxy (legacy fallback for old infra)
# We never mutate proxy env vars at module level — only inside _scoped_proxy.
_DEFAULT_PROXY = (
    os.environ.get("QG_CORROBORATION_PROXY")
    or os.environ.get("http_proxy")
    or os.environ.get("HTTP_PROXY")
    or "http://fengxinshun:xjI8Tv1YQol4j6fKtxVJRwuJ1Rtn1grKpjC4EMKF1GjgGKgDWrZG1hZW6l5O@proxy.h.pjlab.org.cn:23128"
)
_DEFAULT_NO_PROXY = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
_NO_PROXY_ENV_KEYS = ("no_proxy", "NO_PROXY")


@contextlib.contextmanager
def _scoped_proxy(proxy_url: str = _DEFAULT_PROXY, no_proxy: str = _DEFAULT_NO_PROXY):
    """Temporarily install proxy env vars, restoring prior values on exit.

    aiohttp / requests pick up proxy from env at session construction. Any
    session created inside this block will use the proxy; sessions in other
    parts of the pipeline are unaffected. Passing ``proxy_url=""`` forces
    a direct connection for the duration of the block (clears any proxy
    vars). ``no_proxy`` preserves host bypass rules (e.g. intranet
    hosts).
    """
    prev: dict[str, str | None] = {
        k: os.environ.get(k) for k in (*_PROXY_ENV_KEYS, *_NO_PROXY_ENV_KEYS)
    }
    try:
        if proxy_url:
            for k in _PROXY_ENV_KEYS:
                os.environ[k] = proxy_url
        else:
            for k in _PROXY_ENV_KEYS:
                os.environ.pop(k, None)
        if no_proxy:
            for k in _NO_PROXY_ENV_KEYS:
                os.environ[k] = no_proxy
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Local shims (avoid importing test_agent_debug)
# ---------------------------------------------------------------------------
async def _literature_fetch(
    query: str,
    num_results: int = 3,
    max_success: int = 2,
    max_markdown_chars: int = 8000,
) -> str:
    """Local shim of test_agent_debug.literature_fetch that bypasses the
    module's global proxy side effects. Direct wrapper over
    ``sciverse_fetch_markdown``.
    """
    if not TOOLS_AVAILABLE or _sciverse_fetch_markdown is None:
        raise RuntimeError("sciverse_fetch_markdown not available")
    num_results = max(1, min(10, int(num_results)))
    max_success = max(1, min(num_results, int(max_success)))
    max_markdown_chars = max(1000, min(20000, int(max_markdown_chars)))
    return await _sciverse_fetch_markdown(
        query=query,
        num_results=num_results,
        max_success=max_success,
        convert_to_md=True,
        include_markdown_content=True,
        max_markdown_chars=max_markdown_chars,
        max_workers=4,
    )


# ---------------------------------------------------------------------------
# Parsing helpers — each tool returns a formatted string; we parse minimally.
# ---------------------------------------------------------------------------
def _parse_literature_search(text: str) -> list[dict[str, str]]:
    """Extract list of {title, doi, venue, year, snippet} from literature_search output.

    The format is defined in sciverse_tools.py lines 38-70:
        [N] {title}
        Authors: ...
        Year: ...
        DOI: ...
        Venue: ...
        Snippet: ...
    (blank line between entries)
    """
    papers: list[dict[str, str]] = []
    if not text or "No literature results found" in text:
        return papers
    current: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and "]" in line:
            if current.get("title"):
                papers.append(current)
            title_part = line.split("]", 1)[1].strip()
            current = {"title": title_part}
            continue
        for prefix, key in (
            ("Authors:", "authors"),
            ("Year:", "year"),
            ("DOI:", "doi"),
            ("Venue:", "venue"),
            ("Snippet:", "snippet"),
        ):
            if line.startswith(prefix):
                current[key] = line[len(prefix):].strip()
                break
    if current.get("title"):
        papers.append(current)
    return papers


def _parse_web_search(text: str) -> list[dict[str, str]]:
    """Extract list of {url, snippet} from web_search output.

    Format from websearch_tools.py lines 91-101:
        Search results for: "..."
        [1] {url}
        [2] {url}
        ...
        Snippets:
        {formatted_snippets}

    We only need the URLs and the flat snippets blob; snippet-to-URL
    alignment isn't guaranteed so we return a single item per URL with
    the full snippets blob as context.
    """
    results: list[dict[str, str]] = []
    if not text or text.startswith("No results found"):
        return results
    lines = text.splitlines()
    snippets_blob = ""
    in_snippets = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("Snippets:"):
            in_snippets = True
            continue
        if in_snippets:
            if line.startswith("If the snippets above"):
                in_snippets = False
                break
            snippets_blob += raw + "\n"
    for raw in lines:
        line = raw.strip()
        if line.startswith("[") and "] http" in line:
            bracket_end = line.find("]")
            url = line[bracket_end + 1 :].strip()
            results.append({"url": url, "snippet": snippets_blob[:500]})
    return results


_DOI_SENTINELS = {"", "n/a", "na", "none", "null", "unknown"}


def _doi_norm(doi: str) -> str:
    """Normalize a DOI for comparison against local source docs.

    Local source_docs can look like ``10.1002/ctm2.854`` or
    ``DOI:10.1002/...``; we casefold and strip common prefixes. Returns
    ``""`` for sentinel values like ``N/A`` / ``None`` / empty, so that
    the caller's ``if doi_n and doi_n in local_docs_norm`` check does
    NOT accidentally treat two "no DOI" rows as matching.
    """
    s = str(doi or "").strip().casefold()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/", "http://dx.doi.org/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = s.strip()
    if s in _DOI_SENTINELS:
        return ""
    return s


def _extract_host(url: str) -> str:
    """Very loose host extraction (avoid urllib import just for this)."""
    s = str(url or "").strip().casefold()
    if "://" in s:
        s = s.split("://", 1)[1]
    if "/" in s:
        s = s.split("/", 1)[0]
    return s


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CorroborationSource:
    tool: str                          # "literature_search" | "web_search" | "literature_fetch" | "web_fetch"
    title: str = ""
    doi: str = ""
    url: str = ""
    snippet: str = ""
    venue: str = ""
    year: str = ""
    retrieved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorroborationResult:
    status: str                        # "corroborated" | "insufficient" | "tool_unavailable"
    external_sources: list[CorroborationSource] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    issue_tags: list[str] = field(default_factory=list)
    short_rationale: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["external_sources"] = [s for s in d["external_sources"]]
        return d


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
_DEFAULT_LITERATURE_NUM = 5
_DEFAULT_WEB_NUM = 5
_DEFAULT_TIMEOUT = 60
_DEFAULT_SEMAPHORE_SIZE = 4


class CorroborationAgent:
    """Stateful wrapper around the 4 external tools.

    Thread-safe via a lock around the singleton-driven async bridge.
    Each ``corroborate_claim`` call:
      1. Checks cache
      2. Runs ``literature_search`` under ``_scoped_proxy``; filters DOIs
         against ``local_source_docs``; if ≥ min_external_sources survive,
         return corroborated.
      3. Else runs ``web_search`` under scoped proxy; same filter on URL host.
      4. Else returns insufficient.
    Any exception → tool_unavailable with issue_tags.
    """

    def __init__(
        self,
        *,
        min_external_sources: int = 1,
        tool_timeout: float = _DEFAULT_TIMEOUT,
        literature_num: int = _DEFAULT_LITERATURE_NUM,
        web_num: int = _DEFAULT_WEB_NUM,
        cache: ValidationCache | None = None,
        deep_fetch: bool = False,
        max_concurrency: int = _DEFAULT_SEMAPHORE_SIZE,
        proxy_url: str = _DEFAULT_PROXY,
    ) -> None:
        self.min_external_sources = max(1, int(min_external_sources))
        self.tool_timeout = float(tool_timeout)
        self.literature_num = int(literature_num)
        self.web_num = int(web_num)
        self.cache = cache
        self.deep_fetch = bool(deep_fetch)
        self.max_concurrency = max(1, int(max_concurrency))
        self.proxy_url = proxy_url
        self._lock = Lock()
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Public sync entrypoint
    # ------------------------------------------------------------------
    def corroborate_claim(
        self,
        claim_text: str,
        local_source_docs: set[str] | list[str] | None = None,
    ) -> CorroborationResult:
        if not TOOLS_AVAILABLE:
            return CorroborationResult(
                status="tool_unavailable",
                issue_tags=["tools_not_importable"],
                short_rationale=str(_TOOL_IMPORT_ERROR or "tools missing"),
            )
        claim = (claim_text or "").strip()
        if not claim:
            return CorroborationResult(
                status="insufficient",
                issue_tags=["empty_claim"],
                short_rationale="empty claim",
            )
        local_docs_norm = {_doi_norm(d) for d in (local_source_docs or [])}

        # Cache lookup
        cached = self._cache_get(claim)
        if cached is not None:
            return cached

        t0 = time.time()
        try:
            result = asyncio.run(self._run_async(claim, local_docs_norm))
        except Exception as exc:
            logger.error("corroboration agent crash: %s", exc)
            result = CorroborationResult(
                status="tool_unavailable",
                issue_tags=["agent_exception", type(exc).__name__],
                short_rationale=f"{type(exc).__name__}: {exc}",
            )
        result.elapsed_seconds = round(time.time() - t0, 2)
        self._cache_put(claim, result)
        return result

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------
    async def _run_async(self, claim: str, local_docs_norm: set[str]) -> CorroborationResult:
        # Ensure semaphore is bound to this loop
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

        issue_tags: list[str] = []
        tools_used: list[str] = []
        external: list[CorroborationSource] = []

        # ---- literature_search (primary) ----
        async with self._semaphore:
            lit_result = await self._call_tool(
                "literature_search",
                lambda: _literature_search(claim, num_results=self.literature_num, language="en"),
            )
        if lit_result.kind == "ok":
            tools_used.append("literature_search")
            papers = _parse_literature_search(lit_result.payload or "")
            for p in papers:
                doi_n = _doi_norm(p.get("doi", ""))
                if doi_n and doi_n in local_docs_norm:
                    continue
                external.append(
                    CorroborationSource(
                        tool="literature_search",
                        title=p.get("title", ""),
                        doi=p.get("doi", ""),
                        url="",
                        snippet=p.get("snippet", ""),
                        venue=p.get("venue", ""),
                        year=p.get("year", ""),
                        retrieved_at=_now_iso(),
                    )
                )
                if len(external) >= self.min_external_sources:
                    break
            if len(external) >= self.min_external_sources:
                return CorroborationResult(
                    status="corroborated",
                    external_sources=external,
                    tools_used=tools_used,
                    issue_tags=[],
                    short_rationale=f"literature_search returned {len(external)} external papers",
                )
            if not papers:
                issue_tags.append("literature_zero_results")
            else:
                issue_tags.append("literature_all_match_local")
        else:
            issue_tags.append(f"literature_{lit_result.kind}")

        # ---- web_search (fallback) ----
        async with self._semaphore:
            web_result = await self._call_tool(
                "web_search",
                lambda: _web_search(claim, num_results=self.web_num),
            )
        if web_result.kind == "ok":
            tools_used.append("web_search")
            hits = _parse_web_search(web_result.payload or "")
            local_hosts = {_extract_host(d) for d in local_docs_norm if d}
            for h in hits:
                host = _extract_host(h.get("url", ""))
                if host and host in local_hosts:
                    continue
                external.append(
                    CorroborationSource(
                        tool="web_search",
                        title="",
                        url=h.get("url", ""),
                        snippet=h.get("snippet", "")[:500],
                        retrieved_at=_now_iso(),
                    )
                )
                if len(external) >= self.min_external_sources:
                    break
            if len(external) >= self.min_external_sources:
                return CorroborationResult(
                    status="corroborated",
                    external_sources=external,
                    tools_used=tools_used,
                    issue_tags=issue_tags,
                    short_rationale=f"web_search returned {len(external)} external URLs",
                )
            if not hits:
                issue_tags.append("web_zero_results")
            else:
                issue_tags.append("web_all_match_local")
        else:
            issue_tags.append(f"web_{web_result.kind}")

        # Nothing worked
        if any(tag.endswith("_tool_error") or tag.endswith("_timeout") for tag in issue_tags):
            return CorroborationResult(
                status="tool_unavailable",
                external_sources=external,
                tools_used=tools_used,
                issue_tags=issue_tags,
                short_rationale="all corroboration tools failed or timed out",
            )
        return CorroborationResult(
            status="insufficient",
            external_sources=external,
            tools_used=tools_used,
            issue_tags=issue_tags,
            short_rationale="no independent external source found",
        )

    async def _call_tool(self, name: str, coro_factory) -> "_ToolCallResult":
        """Invoke one of the 4 tools inside a scoped_proxy with a timeout.

        We enter the proxy context manager around the AWAIT itself so that
        aiohttp sessions created during the call see the proxy env vars,
        while other parts of the pipeline running concurrently do NOT
        (because they're in separate asyncio tasks / threads and the env
        var change in a sync block is inherently process-global for that
        moment — acceptable because tools typically start their session
        immediately).
        """
        try:
            with _scoped_proxy(self.proxy_url):
                payload = await asyncio.wait_for(coro_factory(), timeout=self.tool_timeout)
            return _ToolCallResult(kind="ok", payload=str(payload))
        except asyncio.TimeoutError:
            logger.warning("corroboration tool %s timed out (%ss)", name, self.tool_timeout)
            return _ToolCallResult(kind="timeout", payload="")
        except Exception as exc:
            logger.warning("corroboration tool %s error: %s: %s", name, type(exc).__name__, exc)
            return _ToolCallResult(kind="tool_error", payload=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    def _cache_payload(self, claim: str) -> dict[str, Any]:
        return {
            "type": "corroboration_agent_v1",
            "claim": claim,
            "min_external_sources": self.min_external_sources,
            "literature_num": self.literature_num,
            "web_num": self.web_num,
        }

    def _cache_get(self, claim: str) -> CorroborationResult | None:
        if not self.cache:
            return None
        raw = self.cache.get(self._cache_payload(claim))
        if not raw:
            return None
        try:
            return CorroborationResult(
                status=str(raw.get("status", "insufficient")),
                external_sources=[CorroborationSource(**s) for s in raw.get("external_sources", [])],
                tools_used=list(raw.get("tools_used", [])),
                issue_tags=list(raw.get("issue_tags", [])),
                short_rationale=str(raw.get("short_rationale", "")),
                elapsed_seconds=float(raw.get("elapsed_seconds", 0.0)),
            )
        except Exception:
            return None

    def _cache_put(self, claim: str, result: CorroborationResult) -> None:
        if not self.cache:
            return
        try:
            payload = self._cache_payload(claim)
            body = {
                "status": result.status,
                "external_sources": [s.to_dict() for s in result.external_sources],
                "tools_used": list(result.tools_used),
                "issue_tags": list(result.issue_tags),
                "short_rationale": result.short_rationale,
                "elapsed_seconds": result.elapsed_seconds,
            }
            self.cache.put(payload, body)
        except Exception as exc:
            logger.warning("corroboration cache write failed: %s", exc)


@dataclass
class _ToolCallResult:
    kind: str                          # "ok" | "timeout" | "tool_error"
    payload: str


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Startup self-check utility
# ---------------------------------------------------------------------------
def self_check(sample_query: str = "SARS-CoV-2 spike protein ACE2") -> dict[str, Any]:
    """One-shot health probe used at CLI startup.

    Runs a tiny literature_search + web_search to confirm tools work end to
    end. Returns a diag dict; does NOT raise.
    """
    diag: dict[str, Any] = {
        "tools_available": TOOLS_AVAILABLE,
        "import_error": _TOOL_IMPORT_ERROR,
    }
    if not TOOLS_AVAILABLE:
        return diag

    async def _probe() -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            with _scoped_proxy(_DEFAULT_PROXY):
                txt = await asyncio.wait_for(
                    _literature_search(sample_query, num_results=3, language="en"),
                    timeout=60,
                )
            parsed = _parse_literature_search(txt)
            out["literature_search"] = {"n_results": len(parsed), "ok": True}
        except Exception as exc:
            out["literature_search"] = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
        try:
            with _scoped_proxy(_DEFAULT_PROXY):
                txt = await asyncio.wait_for(
                    _web_search(sample_query, num_results=3),
                    timeout=60,
                )
            parsed = _parse_web_search(txt)
            out["web_search"] = {"n_results": len(parsed), "ok": True}
        except Exception as exc:
            out["web_search"] = {"ok": False, "err": f"{type(exc).__name__}: {exc}"}
        return out

    diag["probe"] = asyncio.run(_probe())
    return diag
