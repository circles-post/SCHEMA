"""websearch – self-contained web search, page fetching and semantic filtering.

This package provides ONLY the web-search / web-fetch / semantic-filter
primitives. All literature (paper) downloading and PDF→markdown conversion
has been removed; that responsibility now lives entirely in the
`literature_fetch` agent tool defined in test_agent_debug.py, which
delegates to sciverse_fetch_markdown.

Public API
----------
WebSearcher           High-level orchestrator (search → fetch → filter).
OpenAIAdapterSampler  Wrap an OpenAI-compatible client as a search planner.
WebSearchResult       Structured result dataclass.

Low-level helpers (import directly if needed):
    SerperSearchClient   – raw Serper API client
    BrightDataSearchClient – raw Bright Data search client
    fetch_html           – async httpx page fetcher
    extract_text_from_html – trafilatura text extraction
    extract_relevant_sentences – embedding-based semantic filter
    split_into_blocks    – text chunker
"""

from .web_scraper import WebSearcher, OpenAIAdapterSampler, WebSearchResult
from .serper_client import SerperSearchClient
from .bright_data_client import BrightDataSearchClient
from .browser_fetcher import fetch_html, extract_text_from_html
from .semantic_filter import extract_relevant_sentences, split_into_blocks
from .types import SamplerBase, SamplerResponse

__all__ = [
    # High-level
    "WebSearcher",
    "OpenAIAdapterSampler",
    "WebSearchResult",
    # Mid-level
    "SerperSearchClient",
    "BrightDataSearchClient",
    # Low-level
    "fetch_html",
    "extract_text_from_html",
    "extract_relevant_sentences",
    "split_into_blocks",
    # Base types
    "SamplerBase",
    "SamplerResponse",
]
