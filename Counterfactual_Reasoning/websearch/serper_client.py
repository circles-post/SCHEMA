"""Serper API client – web search with optional LLM-guided multi-step planning.

Adapted from halluhard/libs/serper/client.py.
All inter-package imports have been converted to relative imports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx

from .types import SamplerBase, UsageStats
from .json_utils import extract_json_from_response, sanitize_json_string

MAX_URLS_TO_FETCH = 2
MAX_PDFS_TO_FETCH = 1

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SerperUsageStats(UsageStats):
    total_serper_requests: int = 0


@dataclass
class SearchResult:
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    formatted_text: str = ""
    queries: List[str] = field(default_factory=list)
    urls_with_positions: List[Tuple[int, str]] = field(default_factory=list)
    top_urls: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

_STATEMENT_PLACEHOLDER = "[STATEMENT]"
_KNOWLEDGE_PLACEHOLDER = "[KNOWLEDGE]"
_PREVIOUS_QUERIES_PLACEHOLDER = "[PREVIOUS_QUERIES]"

_NEXT_SEARCH_PROMPT = f"""Instructions:

1. You are given a **STATEMENT** and some **KNOWLEDGE** gathered from **PREVIOUS QUERIES**.
2. The STATEMENT references a **source** (paper, book, report, dataset, author, etc.). Your goal is to **identify that source** and verify whether the STATEMENT accurately reflects it.
3. Analyze the KNOWLEDGE collected so far (indexed as [Step.Item]).
4. Decide if you have enough information to verify the statement AND identify the source.
   - If YES:
     - Set "continue_searching" to false.
     - Select the **two most relevant URLs** from the KNOWLEDGE that best support your verification (prioritize sources that are freely accessible and not protected by aggressive anti-bot measures, such as arXiv rather than researchgate or harvard).
     - **IMPORTANT**: You MUST refer to them by their index (e.g. "[1.1]", "[2.3]"). Do NOT output the full URL string, to avoid transcription errors.
   - If NO:
     - Set "continue_searching" to true.
     - Generate a new Google Search query to fill the gaps.
     - The query must be clearly **different** from PREVIOUS QUERIES.

STRICT RULES FOR THE QUERY:
- Do **NOT** copy any numbers, equations, variables, mathematical symbols, quoted passages, or units from the STATEMENT or KNOWLEDGE.
- Do **NOT** include any content in quotation marks.
- Do **NOT** use `site:` filters.
- Use only **broad textual identifiers**: reference title, author names, year, topic keywords.
- If the target appears to be an **academic paper**, prefer a **compact, title-like scholarly query** built from core entities (gene/protein/compound/disease/method) plus one mechanism or phenotype term.
- For academic-paper searches, avoid conversational or evaluation phrasing such as `answer`, `option`, `correct`, `incorrect`, `provided choices`, or long claim restatements.
- Prefer scholarly-source-oriented query terms that are likely to appear in paper titles/abstracts, and prioritize sources like PubMed, PMC, DOI landing pages, journal pages, bioRxiv, medRxiv, or arXiv when selecting URLs.

OUTPUT FORMAT:
You must output a valid JSON object in the following format:

```json
{{
  "reasoning": "Brief explanation of your decision...",
  "continue_searching": boolean,
  "search_query": "Your query string here (required if continue_searching is true, else null)",
  "relevant_urls": ["[1.1]", "[2.3]"] (list of indices as strings, required if continue_searching is false)
}}
```

PREVIOUS QUERIES:
{_PREVIOUS_QUERIES_PLACEHOLDER}

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

_FINAL_SELECTION_PROMPT = """You have completed the search process or reached the limit.
Based on the collected KNOWLEDGE below, identify the **two most relevant URLs** that verify the STATEMENT.
Prioritize sources that are freely accessible and not protected by aggressive anti-bot measures, such as arXiv rather than researchgate or harvard.

STATEMENT:
{statement}

KNOWLEDGE:
{knowledge}

Output ONLY a JSON object:
{{
  "relevant_urls": ["[1.1]", "[2.3]"]
}}
"""


# ---------------------------------------------------------------------------
# Serper Client
# ---------------------------------------------------------------------------

class SerperSearchClient:
    """Async client for Serper API web search.

    Usage (context manager — recommended):
        async with SerperSearchClient() as client:
            result, _ = await client.search("CRISPR cancer therapy")

    Usage (manual lifecycle):
        client = SerperSearchClient()
        await client.start()
        try:
            result, _ = await client.search("query")
        finally:
            await client.close()
    """

    BASE_URL = "https://google.serper.dev/search"
    DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=30.0)
    DEFAULT_CONN_LIMIT = 50

    def __init__(
        self,
        api_key: str | None = None,
        logger: logging.Logger | None = None,
        log_file: str | None = None,
        conn_limit: int | None = None,
    ):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("Provide api_key or set SERPER_API_KEY env var")
        self.logger = logger or logging.getLogger(__name__)
        self.log_file = log_file
        self.conn_limit = conn_limit or self.DEFAULT_CONN_LIMIT
        self._client: httpx.AsyncClient | None = None
        self._owns_client: bool = True

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.DEFAULT_TIMEOUT,
                limits=httpx.Limits(
                    max_connections=self.conn_limit,
                    max_keepalive_connections=self.conn_limit // 2,
                ),
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                http1=True,
                http2=False,
            )
            self._owns_client = True

    async def close(self) -> None:
        if self._client and not self._client.is_closed and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "SerperSearchClient":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            raise RuntimeError("Client not started. Use 'async with client:' or call 'await client.start()'")
        return self._client

    # -- Core Search ---------------------------------------------------------

    async def search(
        self,
        query: str,
        num_results: int = 5,
        max_retries: int = 3,
        context: str | None = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Execute a single Serper search.

        Returns:
            (raw_results_dict, total_requests_made)
        """
        client = self._ensure_client()
        payload = {"q": query, "num": num_results}
        prefix = f"[{context}] " if context else ""

        for attempt in range(max_retries + 1):
            try:
                response = await client.post(self.BASE_URL, json=payload)
                if response.status_code == 200:
                    return response.json(), attempt + 1
                if response.status_code == 429 and attempt < max_retries:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                raise Exception(f"Serper API error {response.status_code}: {response.text}")
            except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
                if attempt < max_retries:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    self.logger.debug(f"{prefix}Serper retry {attempt+1}: {e}")
                    await asyncio.sleep(wait)
                    continue
                raise Exception(f"Serper failed after {max_retries} retries: {e}")

        raise Exception("Serper search failed after all retries")

    # -- Multi-Step Verification Search --------------------------------------

    async def perform_verification_search(
        self,
        claim_text: str,
        sampler: SamplerBase,
        max_searches: int = 5,
        num_results: int = 5,
        search_semaphore: asyncio.Semaphore | None = None,
        context: str | None = None,
        custom_planner_prompt: str | None = None,
    ) -> Tuple[List[Dict], str, List[str], List[Tuple], Dict, List[str]]:
        """LLM-guided multi-step search to verify a claim.

        Returns:
            (raw_results, formatted_text, queries, urls_with_positions, usage_dict, top_urls)
        """
        raw_results: List[Dict[str, Any]] = []
        queries: List[str] = []
        final_urls: List[str] = []
        usage = SerperUsageStats()

        for step in range(max_searches):
            decision = await self._plan_next_step(
                claim_text=claim_text,
                raw_results=raw_results,
                queries=queries,
                sampler=sampler,
                step=step,
                usage=usage,
                custom_planner_prompt=custom_planner_prompt,
            )
            if decision is None:
                break

            if not decision.get("continue_searching", True):
                url_refs = decision.get("relevant_urls", [])
                final_urls = self._resolve_url_references(url_refs, raw_results)
                break

            query = decision.get("search_query")
            if not query:
                break
            queries.append(query)

            try:
                await asyncio.sleep(random.uniform(0, 0.3))
                if search_semaphore:
                    async with search_semaphore:
                        results, reqs = await self.search(query, num_results, context=context)
                else:
                    results, reqs = await self.search(query, num_results, context=context)
                usage.total_serper_requests += reqs
                raw_results.append(results)
            except Exception as e:
                self.logger.debug(f"Search step {step+1} failed: {e}")
                break

        if not final_urls and raw_results:
            final_urls = await self._select_final_urls(claim_text, raw_results, sampler, usage)

        return (
            raw_results,
            self._format_all_results(raw_results),
            queries,
            self._extract_urls_with_positions(raw_results),
            usage.to_dict(),
            final_urls[:MAX_URLS_TO_FETCH],
        )

    # -- Planning Helpers ----------------------------------------------------

    async def _plan_next_step(
        self,
        claim_text: str,
        raw_results: List[Dict],
        queries: List[str],
        sampler: SamplerBase,
        step: int,
        usage: SerperUsageStats,
        custom_planner_prompt: str | None = None,
    ) -> Dict[str, Any] | None:
        knowledge = self._format_knowledge_with_indices(raw_results)
        past_queries = self._format_past_queries(queries)
        base_prompt = custom_planner_prompt or _NEXT_SEARCH_PROMPT
        prompt = (
            base_prompt
            .replace(_STATEMENT_PLACEHOLDER, claim_text)
            .replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
            .replace(_PREVIOUS_QUERIES_PLACEHOLDER, past_queries)
        )
        try:
            response = await sampler([{"role": "user", "content": prompt}])
            if response.token_usage:
                usage.accumulate(response.token_usage)
            return self._parse_planner_response(response.response_text.strip(), step)
        except Exception as e:
            self.logger.debug(f"Planning step {step+1} failed: {e}")
            return None

    async def _select_final_urls(
        self,
        claim_text: str,
        raw_results: List[Dict],
        sampler: SamplerBase,
        usage: SerperUsageStats,
    ) -> List[str]:
        prompt = _FINAL_SELECTION_PROMPT.format(
            statement=claim_text,
            knowledge=self._format_knowledge_with_indices(raw_results),
        )
        try:
            response = await sampler([{"role": "user", "content": prompt}])
            if response.token_usage:
                usage.accumulate(response.token_usage)
            decision = json.loads(sanitize_json_string(
                extract_json_from_response(response.response_text.strip())
            ))
            return self._resolve_url_references(decision.get("relevant_urls", []), raw_results)
        except Exception as e:
            self.logger.debug(f"Final URL selection failed: {e}")
            return []

    def _parse_planner_response(self, response_text: str, step: int) -> Dict[str, Any]:
        try:
            return json.loads(sanitize_json_string(extract_json_from_response(response_text)))
        except Exception:
            match = re.search(r"```(?:\w+)?\s*(.*?)```", response_text, re.DOTALL)
            if match:
                return {"continue_searching": True, "search_query": match.group(1).strip()}
            return {"continue_searching": False}

    # -- URL Resolution ------------------------------------------------------

    def _resolve_url_references(self, url_refs: List[str], raw_results: List[Dict]) -> List[str]:
        valid_urls = {
            res.get("link")
            for step in raw_results
            for res in step.get("organic", [])
            if res.get("link")
        }
        resolved = []
        for ref in url_refs:
            ref = str(ref).strip()
            m = re.match(r"\[?(\d+)\.(\d+)\]?", ref)
            if m:
                si, ii = int(m.group(1)) - 1, int(m.group(2)) - 1
                if 0 <= si < len(raw_results):
                    organic = raw_results[si].get("organic", [])
                    if 0 <= ii < len(organic):
                        link = organic[ii].get("link")
                        if link:
                            resolved.append(link)
                            continue
            if ref in valid_urls or ref.startswith("http"):
                resolved.append(ref)
        return resolved

    # -- Formatting Helpers --------------------------------------------------

    def _format_past_queries(self, queries: List[str]) -> str:
        return "N/A" if not queries else "- " + "\n- ".join(queries)

    def _format_knowledge_with_indices(self, raw_results: List[Dict]) -> str:
        if not raw_results:
            return "N/A"
        sections = []
        for i, step in enumerate(raw_results, 1):
            parts = [f"## Search Step {i}"]
            if step.get("answerBox"):
                ab = step["answerBox"]
                parts.append(f"[Step {i} AnswerBox] {ab.get('snippet') or ab.get('answer') or ''}")
            for j, r in enumerate(step.get("organic", []), 1):
                parts.append(
                    f"[{i}.{j}] Title: {r.get('title','')}\nLink: {r.get('link','')}\nSnippet: {r.get('snippet','')}"
                )
            sections.append("\n\n".join(parts))
        return "\n\n".join(sections)

    def _format_all_results(self, raw_results: List[Dict]) -> str:
        if not raw_results:
            return "No search results found."
        return "\n\n".join(self._format_single_result(r) for r in raw_results)

    def _format_single_result(self, results: Dict) -> str:
        snippets = []
        if results.get("answerBox"):
            ab = results["answerBox"]
            for key in ("answer", "snippet", "snippetHighlighted"):
                val = ab.get(key)
                if val:
                    snippets.append(" ".join(val) if isinstance(val, list) else str(val).replace("\n", " "))
        for r in results.get("organic", []):
            for key in ("title", "date", "link", "snippet"):
                if r.get(key):
                    snippets.append(str(r[key]))
        return " -- ".join(snippets) if snippets else "No results"

    def _extract_urls_with_positions(self, raw_results: List[Dict]) -> List[Tuple[int, str]]:
        urls = []
        for step in raw_results:
            for r in step.get("organic", []):
                if r.get("link"):
                    urls.append((r.get("position", 999), r["link"]))
        return urls
