"""
Test the websearch package at three levels.

Usage:
    # Run all stages
    python test_websearch.py

    # Run specific stages only
    python test_websearch.py --stages 1 2
    python test_websearch.py --stages 4

Environment variables required (one of the two search backends):
    SERPER_API_KEY          – Serper.dev API key
    BRIGHT_DATA_API_KEY     – Bright Data API key  (takes precedence)
    BRIGHT_DATA_ZONE        – Bright Data zone name (required with BRIGHT_DATA_API_KEY)

Environment variables optional (for LLM-guided search and semantic filter):
    AGENTDEBUG_OPENAI_API_KEY   – OpenAI-compatible API key
    AGENTDEBUG_OPENAI_BASE_URL  – Custom base URL of an OpenAI-compatible gateway
    AGENTDEBUG_MODEL_NAME       – Model name (default: gpt-4o-mini)
"""

import argparse
import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Allow running from the agdebugger directory
sys.path.insert(0, os.path.dirname(__file__))

from websearch import (
    WebSearcher,
    OpenAIAdapterSampler,
)

# ---------------------------------------------------------------------------
# Config (reuse agdebugger defaults)
# ---------------------------------------------------------------------------
SERPER_API_KEY      = os.environ.get("SERPER_API_KEY", "")
BRIGHT_DATA_API_KEY = os.environ.get("BRIGHT_DATA_API_KEY", "")
BRIGHT_DATA_ZONE    = os.environ.get("BRIGHT_DATA_ZONE", "")
OPENAI_API_KEY      = os.environ.get("AGENTDEBUG_OPENAI_API_KEY", "")
OPENAI_BASE_URL     = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "")
MODEL_NAME          = os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini")

TEST_QUERY = "APP protein P05067 biological functions Alzheimer"
TEST_CLAIM = "The human APP protein (UniProt P05067) contains a Kunitz-type serine protease inhibitor domain."


# ---------------------------------------------------------------------------
# Stage 1: simple search (no page fetch)
# ---------------------------------------------------------------------------
async def stage1_simple_search(ws: WebSearcher) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1: Simple search (snippets only)")
    print("=" * 60)

    result = await ws.search(TEST_QUERY, num_results=5)
    print(f"URLs returned: {len(result.urls_fetched)}")
    for u in result.urls_fetched[:5]:
        print(f"  - {u}")
    print("\nFormatted snippets (first 500 chars):")
    print(result.formatted_snippets[:500])
    print("\n[PASS] Stage 1 done.")


# ---------------------------------------------------------------------------
# Stage 2: search + fetch pages
# ---------------------------------------------------------------------------
async def stage2_fetch_pages(ws: WebSearcher) -> None:
    print("\n" + "=" * 60)
    print("STAGE 2: Search + fetch top 2 pages")
    print("=" * 60)

    result = await ws.search_and_fetch(TEST_QUERY, num_results=5, fetch_top_n=2)
    print(f"Pages fetched: {len(result.contents)}")
    for item in result.contents:
        wc = len(item["content"].split())
        print(f"  [{wc} words] {item['url'][:80]}")
        print(f"  Preview: {item['content'][:200].replace(chr(10),' ')}")
    print("\n[PASS] Stage 2 done.")


# ---------------------------------------------------------------------------
# Stage 3: full pipeline with LLM-guided search + semantic filter (disabled)
# ---------------------------------------------------------------------------
# async def stage3_full_pipeline(ws: WebSearcher) -> None:
#     print("\n" + "=" * 60)
#     print("STAGE 3: LLM-guided search + fetch + semantic filter")
#     print("=" * 60)
#     print(f"Claim: {TEST_CLAIM}\n")
#
#     sampler = OpenAIAdapterSampler(
#         api_key=OPENAI_API_KEY,
#         base_url=OPENAI_BASE_URL,
#         model=MODEL_NAME,
#     )
#
#     result = await ws.search_fetch_and_filter(
#         claim=TEST_CLAIM,
#         sampler=sampler,
#         max_searches=3,
#         num_results=5,
#         fetch_top_n=2,
#         max_output_words=800,
#     )
#
#     print(f"Queries executed: {result.queries_executed}")
#     print(f"URLs fetched:     {result.urls_fetched}")
#     print(f"Content items:    {len(result.contents)}")
#     print(f"\nFiltered content ({len(result.filtered_content.split())} words):")
#     print(result.filtered_content[:600] or "(no filtered content — OPENAI_API_KEY may not be set)")
#     print("\n[PASS] Stage 3 done.")


# Stage 4 (direct PDF fetch + MinerU) was removed when the in-tree
# pdf_extractor was deleted. PDF/full-text retrieval now belongs to the
# `literature_fetch` agent tool (sciverse path) — see test_agent_debug.py.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(stages: list) -> None:
    using_bright_data = bool(BRIGHT_DATA_API_KEY)
    needs_search = bool({1, 2} & set(stages))

    if needs_search and not using_bright_data and not SERPER_API_KEY:
        print("ERROR: Set BRIGHT_DATA_API_KEY (+ BRIGHT_DATA_ZONE) or SERPER_API_KEY.")
        sys.exit(1)

    backend = "Bright Data" if using_bright_data else "Serper.dev"
    print(f"Stages to run : {stages}")
    if needs_search:
        print(f"Search backend: {backend}")
        if using_bright_data:
            print(f"BD zone       : {BRIGHT_DATA_ZONE}")
        else:
            print(f"Serper key    : {'set' if SERPER_API_KEY else 'MISSING'}")
    print(f"OpenAI key    : {'set' if OPENAI_API_KEY else 'not set (stage 3 filter disabled)'}")
    print(f"Model         : {MODEL_NAME}  @  {OPENAI_BASE_URL}")

    if 4 in stages:
        print("\n[SKIP] Stage 4 (direct PDF fetch) was removed; use literature_fetch tool instead.")

    if needs_search:
        async with WebSearcher(
            serper_api_key=SERPER_API_KEY or None,
            openai_api_key=OPENAI_API_KEY,
            openai_base_url=OPENAI_BASE_URL,
            bright_data_api_key=BRIGHT_DATA_API_KEY or None,
            bright_data_zone=BRIGHT_DATA_ZONE or None,
        ) as ws:
            if 1 in stages:
                await stage1_simple_search(ws)
            if 2 in stages:
                await stage2_fetch_pages(ws)

    print("\nAll requested stages complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the websearch package")
    parser.add_argument(
        "--stages", nargs="+", type=int, choices=[1, 2, 4], default=[1, 2, 4],
        metavar="N", help="Stages to run (1=search, 2=fetch, 4=pdf). Default: 1 2 4.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.stages))


if __name__ == "__main__":
    main()

