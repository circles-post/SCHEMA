"""Bright Data SERP API client – drop-in replacement for SerperSearchClient.

Uses Bright Data's Web Unlocker / SERP zone to execute Google searches and
returns results in the same format as SerperSearchClient so that the rest of
the websearch package (web_scraper.py, etc.) requires no changes.

Environment variables:
    BRIGHT_DATA_API_KEY  – Bright Data API bearer token (required)
    BRIGHT_DATA_ZONE     – Bright Data zone name (required)

Example::

    async with BrightDataSearchClient() as client:
        raw, _ = await client.search("CRISPR cancer therapy", num_results=5)
        print(raw["organic"][0]["title"])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any, Dict, Tuple
from urllib.parse import quote_plus

import httpx


class _BrightDataTransient(Exception):
    """Transient Bright Data failure that should be retried (empty 200, parse error, 5xx)."""

from .serper_client import SerperSearchClient

_logger = logging.getLogger(__name__)

_BRIGHT_DATA_API_URL = "https://api.brightdata.com/request"


def _normalize_organic(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map Bright Data parsed_light fields to Serper-compatible field names.

    Bright Data               →  Serper
    ----------------------------------------
    item["url"]               →  item["link"]
    item["description"]       →  item["snippet"]
    item["global_rank"]       →  item["position"]
    """
    normalized = dict(item)  # shallow copy

    # url → link
    if "url" in normalized and "link" not in normalized:
        normalized["link"] = normalized.pop("url")
    elif "url" in normalized:
        normalized.pop("url")

    # description → snippet
    if "description" in normalized and "snippet" not in normalized:
        normalized["snippet"] = normalized.pop("description")
    elif "description" in normalized:
        normalized.pop("description")

    # global_rank → position
    if "global_rank" in normalized and "position" not in normalized:
        normalized["position"] = normalized.pop("global_rank")
    elif "global_rank" in normalized:
        normalized.pop("global_rank")

    return normalized


def _normalize_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a full Bright Data parsed_light response to Serper format."""
    result = dict(raw)
    if "organic" in result and isinstance(result["organic"], list):
        result["organic"] = [_normalize_organic(item) for item in result["organic"]]
    return result


class BrightDataSearchClient(SerperSearchClient):
    """Bright Data SERP API client with the same interface as SerperSearchClient.

    Overrides ``start()`` and ``search()`` to hit the Bright Data endpoint.
    All LLM-guided multi-step search logic (``perform_verification_search``,
    ``_plan_next_step``, etc.) is inherited unchanged.

    Args:
        api_key:   Bright Data API key. Falls back to ``BRIGHT_DATA_API_KEY`` env var.
        zone:      Bright Data zone name. Falls back to ``BRIGHT_DATA_ZONE`` env var.
        logger:    Optional logger instance.
        conn_limit: Max concurrent HTTP connections (default: 50).
    """

    def __init__(
        self,
        api_key: str | None = None,
        zone: str | None = None,
        logger: logging.Logger | None = None,
        conn_limit: int | None = None,
    ):
        self._bd_api_key = api_key or os.environ.get("BRIGHT_DATA_API_KEY", "")
        if not self._bd_api_key:
            raise ValueError("Provide api_key or set BRIGHT_DATA_API_KEY env var")

        self._bd_zone = zone or os.environ.get("BRIGHT_DATA_ZONE", "YOUR_ZONE")
        self.logger = logger or logging.getLogger(__name__)
        self.conn_limit = conn_limit or self.DEFAULT_CONN_LIMIT
        self._client: httpx.AsyncClient | None = None
        self._owns_client: bool = True

        # Satisfy parent's api_key attribute (used nowhere critical, but present)
        self.api_key = self._bd_api_key
        self.log_file = None

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:  # type: ignore[override]
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.DEFAULT_TIMEOUT,
                limits=httpx.Limits(
                    max_connections=self.conn_limit,
                    max_keepalive_connections=self.conn_limit // 2,
                ),
                headers={
                    "Authorization": f"Bearer {self._bd_api_key}",
                    "Content-Type": "application/json",
                },
                http1=True,
                http2=False,
            )
            self._owns_client = True

    # -- Core Search ---------------------------------------------------------

    async def search(  # type: ignore[override]
        self,
        query: str,
        num_results: int = 5,
        max_retries: int = 3,
        context: str | None = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Execute a single Bright Data SERP search.

        Builds a Google search URL and sends it to Bright Data's request
        endpoint using ``parsed_light`` format.  The response is normalized to
        match the Serper API response shape so downstream code is unaffected.

        Returns:
            (raw_results_dict, total_requests_made)
        """
        client = self._ensure_client()
        prefix = f"[{context}] " if context else ""

        google_url = (
            f"https://www.google.com/search?q={quote_plus(query)}"
            f"&num={num_results}&hl=en&gl=us"
        )
        payload: Dict[str, Any] = {
            "zone": self._bd_zone,
            "url": google_url,
            "format": "raw",
            "data_format": "parsed_light",
        }

        last_diag: str = ""
        for attempt in range(max_retries + 1):
            try:
                response = await client.post(_BRIGHT_DATA_API_URL, json=payload)
                status = response.status_code
                body = response.content or b""
                ctype = response.headers.get("content-type", "")

                if status == 200:
                    # Defensive: Bright Data occasionally returns HTTP 200 with
                    # an empty (or whitespace-only) body under load/quota/zone
                    # misconfig. Treat that as transient and retry.
                    if not body.strip():
                        last_diag = (
                            f"status=200 empty_body content_type={ctype!r} "
                            f"req_id={response.headers.get('x-brd-req-id')!r}"
                        )
                        raise _BrightDataTransient(last_diag)
                    try:
                        raw = json.loads(body)
                    except json.JSONDecodeError as je:
                        snippet = body[:200].decode("utf-8", errors="replace")
                        last_diag = (
                            f"status=200 non_json content_type={ctype!r} "
                            f"body_len={len(body)} body_head={snippet!r} err={je}"
                        )
                        raise _BrightDataTransient(last_diag) from je
                    normalized = _normalize_response(raw)
                    return normalized, attempt + 1

                # Retryable HTTP error codes
                if status in (408, 429, 500, 502, 503, 504):
                    last_diag = f"status={status} body_head={response.text[:200]!r}"
                    if attempt < max_retries:
                        wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                        self.logger.debug(
                            f"{prefix}Bright Data retry {attempt+1}: {last_diag}"
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise Exception(
                        f"Bright Data API error after {max_retries} retries: {last_diag}"
                    )

                # Non-retryable HTTP error (4xx auth/zone/validation etc.) — fail fast
                raise Exception(
                    f"Bright Data API error {status}: {response.text[:200]}"
                )

            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.ProxyError,
                _BrightDataTransient,
                OSError,
            ) as e:
                last_diag = last_diag or f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    self.logger.debug(
                        f"{prefix}Bright Data retry {attempt+1}: {last_diag}"
                    )
                    await asyncio.sleep(wait)
                    continue
                # Final failure: surface the diagnostic upward so the caller
                # logs a meaningful `evidence_error` instead of a bare
                # "Expecting value: line 1 column 1 (char 0)".
                raise Exception(
                    f"Bright Data failed after {max_retries} retries: {last_diag}"
                ) from e

        raise Exception(
            f"Bright Data search failed after all retries: {last_diag or 'unknown'}"
        )
