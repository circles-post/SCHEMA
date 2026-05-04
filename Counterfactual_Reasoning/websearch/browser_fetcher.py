"""Async webpage fetcher: httpx + trafilatura for content extraction.

Adapted from halluhard/libs/browser_fetcher.py.
Selenium branch (commented out in original) is not included here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import random
from typing import List
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import httpx
import trafilatura

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP Client (connection pool)
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid",
    "_ga", "_gid", "_hsenc", "_hsmi",
    "ref", "referrer", "source",
}


@dataclass
class FetchResult:
    url: str
    html: str | None
    status_code: int | None = None
    content_type: str = ""
    error: str | None = None
    blocked_reason: str | None = None


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(20.0, connect=10.0),
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
            headers=_DEFAULT_HEADERS,
            verify=True,
            http1=True,
            http2=False,
        )
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client. Call at program shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def infer_blocked_reason(url: str, status_code: int | None, html: str = "", error: str | None = None) -> str | None:
    url_lower = url.lower()
    html_lower = html.lower()

    if error:
        return f"request failed: {error}"
    if "just a moment" in html_lower or "cf-browser-verification" in html_lower or "cloudflare" in html_lower:
        return "blocked by anti-bot challenge"
    if "preparing to download" in html_lower and "pow" in html_lower:
        return "blocked by PMC proof-of-work download interstitial"
    if "tdm-reservation" in html_lower or ("elsevier" in html_lower and status_code == 403):
        return "blocked by Elsevier access control or anti-bot checks"
    if status_code == 403 and ("cell.com" in url_lower or "sciencedirect.com" in url_lower):
        return "HTTP 403 from Elsevier site (likely access control or anti-bot)"
    if status_code == 401:
        return "HTTP 401 unauthorized"
    if status_code == 403:
        return "HTTP 403 forbidden"
    if status_code == 404:
        return "HTTP 404 not found"
    if status_code is not None and status_code >= 400:
        return f"HTTP {status_code}"
    return None


def describe_fetch_result(result: FetchResult, *, expected: str = "html") -> str:
    if result.blocked_reason:
        return result.blocked_reason
    if result.status_code is not None and result.status_code >= 400:
        return f"HTTP {result.status_code}"
    if expected == "pdf" and result.content_type:
        return f"expected PDF but received {result.content_type}"
    if result.error:
        return result.error
    return "no usable content extracted"

def normalize_url(url: str) -> str:
    """Remove common tracking parameters from a URL."""
    try:
        parsed = urlparse(url)
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            filtered = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
            new_query = urlencode(filtered, doseq=True) if filtered else ""
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ""))
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, "", ""))
    except Exception:
        return url


async def fetch_html(url: str, timeout: int = 20, max_retries: int = 2) -> str | None:
    """Fetch raw HTML from *url* asynchronously.

    Returns the HTML string or None on failure.
    """
    result = await fetch_html_result(url, timeout=timeout, max_retries=max_retries)
    return result.html


async def fetch_html_result(url: str, timeout: int = 20, max_retries: int = 2) -> FetchResult:
    """Fetch raw HTML plus response metadata for diagnostics."""
    clean_url = normalize_url(url)

    for attempt in range(max_retries + 1):
        try:
            await asyncio.sleep(random.uniform(0, 0.2))
            client = await _get_http_client()
            response = await client.get(clean_url, timeout=timeout)
            html = response.text or ""
            content_type = response.headers.get("Content-Type", "")
            blocked_reason = infer_blocked_reason(str(response.url), response.status_code, html)
            if response.status_code >= 400:
                return FetchResult(
                    url=str(response.url),
                    html=html or None,
                    status_code=response.status_code,
                    content_type=content_type,
                    blocked_reason=blocked_reason,
                )
            if not html or len(html) < 500:
                return FetchResult(
                    url=str(response.url),
                    html=None,
                    status_code=response.status_code,
                    content_type=content_type,
                    blocked_reason=blocked_reason,
                    error="response too short",
                )
            return FetchResult(
                url=str(response.url),
                html=html,
                status_code=response.status_code,
                content_type=content_type,
                blocked_reason=blocked_reason,
            )
        except (httpx.TimeoutException, httpx.ConnectError, OSError) as e:
            if attempt < max_retries:
                wait = 2 ** attempt + random.uniform(0, 1)
                _logger.debug(f"Retrying in {wait:.1f}s ({type(e).__name__})")
                await asyncio.sleep(wait)
                continue
            _logger.debug(f"Failed after {max_retries} retries: {type(e).__name__}")
            return FetchResult(
                url=clean_url,
                html=None,
                error=type(e).__name__,
                blocked_reason=infer_blocked_reason(clean_url, None, error=type(e).__name__),
            )
        except Exception as e:
            err = str(e)
            if "SSL" not in err and "certificate" not in err:
                _logger.debug(f"Unexpected error for {clean_url[:80]}: {type(e).__name__}")
            return FetchResult(
                url=clean_url,
                html=None,
                error=type(e).__name__,
                blocked_reason=infer_blocked_reason(clean_url, None, error=type(e).__name__),
            )

    return FetchResult(url=clean_url, html=None, error="unknown error", blocked_reason="unknown error")


async def extract_text_from_html(html: str, source_url: str = "", max_words: int | None = None) -> str:
    """Extract main content from HTML using trafilatura, returning Markdown.

    Runs trafilatura in the thread-pool (CPU-bound).
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=True,
                output_format="markdown",
                url=source_url or None,
                favor_recall=True,
            ),
        )
        text = result or ""
        if max_words:
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words]) + f"\n\n[Content truncated at {max_words} words]"
        return text
    except Exception as e:
        _logger.debug(f"trafilatura failed: {e}")
        return ""


