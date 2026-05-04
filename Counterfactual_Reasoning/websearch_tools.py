"""AutoGen-compatible web search tools for the ToolUniverse agent.

Exposes two general web tools:
    web_search(query, num_results)   – returns snippets + URL list (fast, no fetch)
    web_fetch(urls)                  – fetches full text + PDFs for given URLs

The agent calls web_search first; if the snippets are insufficient it calls
web_fetch with the URLs it wants to read in full. Scholarly literature search
is handled separately by the Sciverse-backed literature tool.
"""

from __future__ import annotations

import asyncio
import os
from typing import List

from model_routing import resolve_base_url_for_model, resolve_value_for_model
from websearch import WebSearcher

# ---------------------------------------------------------------------------
# Config (mirrors test_agent_debug.py defaults)
# ---------------------------------------------------------------------------
_BRIGHT_DATA_API_KEY = os.environ.get(
    "BRIGHT_DATA_API_KEY", "cf0ecaca-a28c-49f8-85df-d27e37cd86a8"
)
_BRIGHT_DATA_ZONE = os.environ.get("BRIGHT_DATA_ZONE", "serp_api1")
_DEFAULT_OPENAI_API_KEY = os.environ.get(
    "AGENTDEBUG_OPENAI_API_KEY", "sk-ZnvhxhwyXok91ezpbDBcObLWa8GehlZtMaqnYT3ziVwhnBzC"
)
_INTERN_OPENAI_API_KEY = os.environ.get("AGENTDEBUG_INTERN_API_KEY")
_OPENAI_MODEL_NAME = os.environ.get(
    "AGENTDEBUG_MODEL_AGENT",
    os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini"),
)
_OPENAI_API_KEY = os.environ.get("AGENTDEBUG_OPENAI_API_KEY_AGENT") or resolve_value_for_model(
    _OPENAI_MODEL_NAME,
    _DEFAULT_OPENAI_API_KEY,
    intern_value=_INTERN_OPENAI_API_KEY,
)
_DEFAULT_OPENAI_BASE_URL = os.environ.get(
    "AGENTDEBUG_OPENAI_BASE_URL", "http://34.13.73.248:3888/v1"
)
_OPENAI_BASE_URL = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL_AGENT") or resolve_base_url_for_model(
    _OPENAI_MODEL_NAME,
    _DEFAULT_OPENAI_BASE_URL,
)

# Shared singleton – started on first use, reused across all tool calls
_searcher: WebSearcher | None = None


async def _get_searcher() -> WebSearcher:
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher(
            bright_data_api_key=_BRIGHT_DATA_API_KEY or None,
            bright_data_zone=_BRIGHT_DATA_ZONE or None,
            openai_api_key=_OPENAI_API_KEY or None,
            openai_base_url=_OPENAI_BASE_URL or None,
        )
        await _searcher.start()
    return _searcher


# ---------------------------------------------------------------------------
# Tool 1 – web_search
# ---------------------------------------------------------------------------

async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return result snippets with URLs.

    Use this tool first to get an overview of available sources.
    If the snippets contain enough information, no further fetching is needed.
    If more detail is required, call web_fetch with the URLs you want to read.

    Args:
        query:       The search query string.
        num_results: Number of results to return (default 5, max 10).

    Returns:
        Formatted string with titles, snippets, and numbered URLs.
    """
    ws = await _get_searcher()
    num_results = min(max(1, num_results), 10)
    result = await ws.search(query, num_results=num_results)

    if not result.urls_fetched:
        return f'No results found for query: "{query}"'

    lines = [f'Search results for: "{query}"\n']
    for i, url in enumerate(result.urls_fetched, 1):
        lines.append(f"[{i}] {url}")

    lines.append("")
    lines.append("Snippets:")
    lines.append(result.formatted_snippets)
    lines.append("")
    lines.append(
        "If the snippets above are insufficient, call web_fetch with the URLs you need."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 – web_fetch
# ---------------------------------------------------------------------------

_WEB_FETCH_TIMEOUT_SEC = float(os.environ.get("AGENTDEBUG_WEB_FETCH_TIMEOUT", "120"))
_WEB_FETCH_MAX_WORDS_PER_PAGE = int(os.environ.get("AGENTDEBUG_WEB_FETCH_MAX_WORDS_PER_PAGE", "2000"))
_WEB_FETCH_MAX_TOTAL_WORDS = int(os.environ.get("AGENTDEBUG_WEB_FETCH_MAX_TOTAL_WORDS", "4000"))


async def web_fetch(urls: List[str]) -> str:
    """Fetch the full text content of one or more web pages or PDFs.

    Call this after web_search when the snippets do not contain enough detail.
    Pass the URLs from the search results that are most likely to contain
    the information you need (at most 3 URLs to stay within context limits).

    Automatically detects and extracts PDF content when a URL points to a PDF.

    Args:
        urls: List of URLs to fetch (recommend 1-3 URLs).

    Returns:
        Full page text grouped by source, truncated to ~5000 words per page.
    """
    if not urls:
        return "No URLs provided."

    urls = urls[:3]  # hard cap to avoid flooding context
    ws = await _get_searcher()

    # Fetch each URL independently with per-URL timeout so one slow URL
    # doesn't block the others.
    per_url_timeout = float(os.environ.get("AGENTDEBUG_WEB_FETCH_PER_URL_TIMEOUT", "60"))
    contents: list = []
    timed_out_urls: list = []

    async def _fetch_one(url: str):
        try:
            items = await asyncio.wait_for(
                ws._fetch_single_url(url, "", "", True),
                timeout=per_url_timeout,
            )
            contents.extend(items)
        except asyncio.TimeoutError:
            timed_out_urls.append(url)

    await asyncio.wait_for(
        asyncio.gather(*[_fetch_one(u) for u in urls], return_exceptions=True),
        timeout=_WEB_FETCH_TIMEOUT_SEC,
    )

    parts = []
    if timed_out_urls:
        parts.append(
            f"Note: {len(timed_out_urls)} URL(s) timed out after {per_url_timeout}s "
            f"and were skipped: {', '.join(timed_out_urls)}"
        )

    if not contents:
        if timed_out_urls:
            return parts[0] + "\nNo content was retrieved. Try different URLs."
        return "Could not fetch content from the provided URLs."

    for item in contents:
        title = item.get("title") or item["url"]
        url = item["url"]
        content = item["content"]
        word_count = len(content.split())
        # Truncate very long pages to keep context manageable
        if word_count > _WEB_FETCH_MAX_WORDS_PER_PAGE:
            content = " ".join(content.split()[:_WEB_FETCH_MAX_WORDS_PER_PAGE]) + "\n...[truncated]"
            word_count = _WEB_FETCH_MAX_WORDS_PER_PAGE
        header = f"=== {title} ===\nURL: {url}\nWords: {word_count}\n"
        parts.append(header + content)

    # Enforce a total word budget across all fetched pages.
    result = "\n\n".join(parts)
    total_words = len(result.split())
    if total_words > _WEB_FETCH_MAX_TOTAL_WORDS:
        result = " ".join(result.split()[:_WEB_FETCH_MAX_TOTAL_WORDS]) + "\n...[total output truncated]"

    return result
