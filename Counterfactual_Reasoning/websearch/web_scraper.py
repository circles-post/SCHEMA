"""High-level WebSearcher: search → fetch pages/PDFs → optional semantic filter.

This is the main entry point for the websearch package.

Typical usage
-------------
Simple search only (no LLM, no page fetching):

    async with WebSearcher(serper_api_key="...") as ws:
        result = await ws.search("CRISPR cancer therapy", num_results=5)
        print(result.formatted_text)

Search + fetch page content:

    async with WebSearcher(serper_api_key="...") as ws:
        result = await ws.search_and_fetch("APP protein P05067 function", fetch_top_n=2)
        for item in result.contents:
            print(item["url"], item["content"][:300])

Full pipeline with LLM-guided search + semantic filter:

    from websearch import WebSearcher, OpenAIAdapterSampler

    sampler = OpenAIAdapterSampler(api_key="...", base_url="...", model="gpt-4o-mini")
    async with WebSearcher(serper_api_key="...",
                           openai_api_key="...",
                           openai_base_url="...") as ws:
        result = await ws.search_fetch_and_filter(
            claim="APP protein has Kunitz-type protease inhibitor domains",
            sampler=sampler,
            max_searches=3,
            max_output_words=1500,
        )
        print(result.filtered_content)
"""

from __future__ import annotations

import asyncio
import os
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from model_routing import build_openai_extra_body
from .browser_fetcher import fetch_html, extract_text_from_html, close_http_client
from .semantic_filter import extract_relevant_sentences
from .serper_client import SerperSearchClient
from .bright_data_client import BrightDataSearchClient
from .types import SamplerBase, SamplerResponse

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WebSearchResult:
    """Structured result returned by WebSearcher methods."""

    # Always present
    query: str = ""
    queries_executed: List[str] = field(default_factory=list)
    formatted_snippets: str = ""            # Serper snippets only (fast, no HTTP)
    urls_fetched: List[str] = field(default_factory=list)

    # Populated after fetch
    contents: List[Dict[str, Any]] = field(default_factory=list)
    """Each item: {url, title, snippet, content (Markdown)}"""

    # Populated after semantic filter
    filtered_content: str = ""

    # Stats
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter: wrap an AutoGen / OpenAI client as SamplerBase
# ---------------------------------------------------------------------------

class OpenAIAdapterSampler(SamplerBase):
    """Minimal SamplerBase wrapper around openai.AsyncOpenAI.

    Use this to plug the WebSearcher's LLM-guided search planner into
    any OpenAI-compatible endpoint (including the agdebugger's custom one).

    Example::

        sampler = OpenAIAdapterSampler(
            api_key="sk-...",
            base_url="http://34.13.73.248:3888/v1",
            model="gpt-4o-mini",
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        from openai import AsyncOpenAI
        import httpx

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            timeout=httpx.Timeout(60.0, connect=15.0),
            http1=True,
            http2=False,
        )
        kwargs: Dict[str, Any] = dict(api_key=api_key, http_client=http_client)
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model
        self._extra_body = build_openai_extra_body()

    async def __call__(self, message_list) -> SamplerResponse:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=message_list,
            **({"extra_body": self._extra_body} if self._extra_body else {}),
        )
        choice = resp.choices[0]
        usage = resp.usage
        return SamplerResponse(
            response_text=choice.message.content or "",
            actual_queried_message_list=message_list,
            response_metadata={},
            token_usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
            },
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WebSearcher:
    """Search the web, optionally fetch pages/PDFs and apply semantic filtering.

    Args:
        serper_api_key:  Serper API key (falls back to ``SERPER_API_KEY`` env var).
        openai_api_key:  OpenAI-compatible API key for embeddings (semantic filter).
        openai_base_url: Base URL for OpenAI-compatible embedding endpoint.
        max_html_words:  Truncate fetched HTML to this many words (default: 5000).
        log_level:       Logging level for the internal logger.
    """

    def __init__(
        self,
        serper_api_key: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        max_html_words: int = 5000,
        log_level: int = logging.WARNING,
        # Bright Data options (take precedence over serper_api_key when set)
        bright_data_api_key: str | None = None,
        bright_data_zone: str | None = None,
    ):
        self._serper_api_key = serper_api_key
        self._openai_api_key = openai_api_key
        self._openai_base_url = openai_base_url
        self._max_html_words = max_html_words
        self._bright_data_api_key = bright_data_api_key
        self._bright_data_zone = bright_data_zone
        self._serper: SerperSearchClient | None = None

        logging.getLogger("websearch").setLevel(log_level)

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        # Prefer Bright Data when its key is provided or available via env var
        import os
        bd_key = self._bright_data_api_key or os.environ.get("BRIGHT_DATA_API_KEY", "")
        if bd_key:
            self._serper = BrightDataSearchClient(
                api_key=bd_key,
                zone=self._bright_data_zone,
            )
        else:
            self._serper = SerperSearchClient(api_key=self._serper_api_key)
        await self._serper.start()

    async def close(self) -> None:
        if self._serper:
            await self._serper.close()
        await close_http_client()

    async def __aenter__(self) -> "WebSearcher":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> WebSearchResult:
        """Run a single Serper search and return snippets (no page fetching).

        This is the cheapest / fastest method – great for a quick sanity check.
        """
        self._ensure_started()
        raw, _ = await self._serper.search(query, num_results)
        return WebSearchResult(
            query=query,
            queries_executed=[query],
            formatted_snippets=self._serper._format_single_result(raw),
            urls_fetched=[r["link"] for r in raw.get("organic", []) if r.get("link")],
        )

    async def search_and_fetch(
        self,
        query: str,
        num_results: int = 5,
        fetch_top_n: int = 2,
    ) -> WebSearchResult:
        """Search + fetch and parse the top *fetch_top_n* pages.

        Args:
            query:          Search query string.
            num_results:    Number of Serper results to request.
            fetch_top_n:    How many of the top results to actually fetch.

        Returns:
            WebSearchResult with ``contents`` list populated.
        """
        self._ensure_started()

        raw, _ = await self._serper.search(query, num_results)
        organic = raw.get("organic", [])
        urls = [r["link"] for r in organic[:fetch_top_n] if r.get("link")]
        snippets_map = {r["link"]: r.get("snippet", "") for r in organic if r.get("link")}
        titles_map = {r["link"]: r.get("title", "") for r in organic if r.get("link")}

        contents = await self._fetch_urls(urls, snippets_map, titles_map)

        return WebSearchResult(
            query=query,
            queries_executed=[query],
            formatted_snippets=self._serper._format_single_result(raw),
            urls_fetched=urls,
            contents=contents,
        )

    async def search_fetch_and_filter(
        self,
        claim: str,
        sampler: Optional[SamplerBase] = None,
        max_searches: int = 3,
        num_results: int = 5,
        fetch_top_n: int = 2,
        max_output_words: int = 1500,
        embedding_model: str = "text-embedding-3-small",
    ) -> WebSearchResult:
        """Full pipeline: LLM-guided search → fetch pages → semantic filter.

        If *sampler* is None the search falls back to a single non-guided query.

        Args:
            claim:            The claim / question to verify or answer.
            sampler:          LLM sampler for multi-step search planning.
                              Pass ``None`` for a simple single-step search.
            max_searches:     Maximum LLM-guided search iterations.
            num_results:      Serper results per search step.
            fetch_top_n:      Pages to fetch from the final URL selection.
            max_output_words: Word budget for the semantic filter output.
            embedding_model:  OpenAI embedding model for semantic filter.

        Returns:
            WebSearchResult with ``contents`` and ``filtered_content`` populated.
        """
        self._ensure_started()

        _logger.warning(
            "[websearch] web_fallback search_start query=%r sampler=%s num_results=%s fetch_top_n=%s",
            (claim or "")[:200],
            bool(sampler is not None),
            num_results,
            fetch_top_n,
        )
        search_t0 = time.perf_counter()

        # Step 1: search
        if sampler is not None:
            (raw_results, formatted, queries, urls_pos, usage, top_urls) = (
                await self._serper.perform_verification_search(
                    claim_text=claim,
                    sampler=sampler,
                    max_searches=max_searches,
                    num_results=num_results,
                )
            )
        else:
            raw, _ = await self._serper.search(claim, num_results)
            raw_results = [raw]
            formatted = self._serper._format_single_result(raw)
            queries = [claim]
            usage = {}
            top_urls = [r["link"] for r in raw.get("organic", [])[:fetch_top_n] if r.get("link")]

        # Gather URLs to fetch
        fetch_urls = top_urls[:fetch_top_n]

        snippets_map = {
            r["link"]: r.get("snippet", "")
            for step in raw_results
            for r in step.get("organic", [])
            if r.get("link")
        }
        titles_map = {
            r["link"]: r.get("title", "")
            for step in raw_results
            for r in step.get("organic", [])
            if r.get("link")
        }

        # Step 2: fetch
        _logger.warning(
            "[websearch] web_fallback search_done elapsed_sec=%.2f fetch_urls=%s",
            time.perf_counter() - search_t0,
            fetch_urls[:4],
        )
        fetch_t0 = time.perf_counter()
        contents = await self._fetch_urls(fetch_urls, snippets_map, titles_map)
        _logger.warning(
            "[websearch] web_fallback fetch_done elapsed_sec=%.2f contents=%s",
            time.perf_counter() - fetch_t0,
            len(contents),
        )

        # Step 3: semantic filter
        # Wrap with a hard wait_for so a hung embedding endpoint cannot eat
        # the entire per-claim evidence budget. On timeout/failure we fall
        # back to the raw fetched contents (handled in evidence_provider).
        filtered = ""
        filter_t0 = time.perf_counter()
        filter_timed_out = False
        filter_error: str | None = None
        if contents and self._openai_api_key:
            filter_budget = float(os.environ.get("AGDEBUGGER_SEMANTIC_FILTER_TIMEOUT_SEC", "30"))
            try:
                filtered = await asyncio.wait_for(
                    extract_relevant_sentences(
                        websearch_results=contents,
                        claim=claim,
                        max_output_words=max_output_words,
                        embedding_model=embedding_model,
                        openai_api_key=self._openai_api_key,
                        openai_base_url=self._openai_base_url,
                    ),
                    timeout=filter_budget,
                )
            except asyncio.TimeoutError:
                filter_timed_out = True
                filtered = ""
            except Exception as exc:  # noqa: BLE001
                filter_error = f"{type(exc).__name__}: {exc}"
                filtered = ""
        _logger.warning(
            "[websearch] web_fallback filter_done elapsed_sec=%.2f filtered_chars=%s timed_out=%s error=%s",
            time.perf_counter() - filter_t0,
            len(filtered),
            filter_timed_out,
            filter_error,
        )

        return WebSearchResult(
            query=claim,
            queries_executed=queries,
            formatted_snippets=formatted,
            urls_fetched=fetch_urls,
            contents=contents,
            filtered_content=filtered,
            usage=usage,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._serper is None:
            raise RuntimeError("WebSearcher not started. Use 'async with WebSearcher(...):'")

    async def _fetch_single_url(
        self,
        url: str,
        snippet: str,
        title: str,
    ) -> List[Dict[str, Any]]:
        """Fetch one URL and return a list of content dicts (HTML only).

        PDF download / parsing has been removed. If a URL points at a PDF the
        agent should use the literature_fetch tool (sciverse path) instead.
        """
        items: List[Dict[str, Any]] = []

        lowered = url.lower()
        if lowered.endswith(".pdf") or "/pdf/" in lowered:
            _logger.info(
                "[websearch] fetch_single_url skip_pdf url=%s (use literature_fetch tool)",
                url,
            )
            return items

        html = await fetch_html(url)
        if not html:
            _logger.warning("[websearch] fetch_single_url no_html url=%s", url)
            return items

        text = await extract_text_from_html(html, source_url=url, max_words=self._max_html_words)
        if text:
            items.append({"url": url, "title": title, "snippet": snippet, "content": text})
        return items

    async def _fetch_urls(
        self,
        urls: List[str],
        snippets_map: Dict[str, str],
        titles_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Fetch all URLs concurrently and return a flat list of content dicts."""
        tasks = [
            self._fetch_single_url(
                url,
                snippets_map.get(url, ""),
                titles_map.get(url, ""),
            )
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        contents: List[Dict[str, Any]] = []
        for res in results:
            if isinstance(res, Exception):
                _logger.warning(f"Fetch error: {res}")
            else:
                contents.extend(res)
        return contents
