"""Semantic relevance filter for web content using OpenAI embeddings.

Adapted from halluhard/libs/information_extraction.py (embedding portion only).
Splits content into overlapping character blocks and ranks them by cosine
similarity to the query/claim, returning only the most relevant portions.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List

import httpx
import numpy as np
from openai import AsyncOpenAI

_logger = logging.getLogger(__name__)

_openai_clients: dict[tuple[str | None, str | None], AsyncOpenAI] = {}


def get_openai_client(api_key: str | None = None, base_url: str | None = None) -> AsyncOpenAI:
    """Return the shared AsyncOpenAI client, creating it on first call.

    Args:
        api_key:  OpenAI (or compatible) API key.  Falls back to OPENAI_API_KEY env var.
        base_url: Optional base URL for OpenAI-compatible endpoints.
    """
    config = (api_key, base_url)
    client = _openai_clients.get(config)
    if client is None:
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
            timeout=httpx.Timeout(15.0, connect=5.0),
            http1=True,
            http2=False,
        )
        kwargs: Dict[str, Any] = dict(timeout=15.0, max_retries=0, http_client=http_client)
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncOpenAI(**kwargs)
        _openai_clients[config] = client
    return client


def split_into_blocks(text: str, max_chars: int = 10000, overlap: int = 200) -> List[str]:
    """Split *text* into overlapping character blocks at natural boundaries.

    Args:
        text:      Input text.
        max_chars: Maximum characters per block.
        overlap:   Characters of overlap between adjacent blocks.

    Returns:
        List of text blocks.
    """
    text = text.strip()
    if not text:
        return []
    blocks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = max(start + int(0.1 * max_chars), start)
            candidates = [
                text.rfind("\n\n", window, end),
                text.rfind(". ", window, end),
                text.rfind("? ", window, end),
                text.rfind("! ", window, end),
            ]
            best = max(candidates)
            if best != -1:
                end = best + 1 if text[best] in ".?!" else best + 2
        block = text[start:end].strip()
        if block:
            blocks.append(block)
        if end >= n:
            break
        start = max(0, end - overlap)
    return blocks


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / np.linalg.norm(query)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m = matrix / (norms + 1e-10)
    return np.dot(m, q)


async def extract_relevant_sentences(
    websearch_results: List[Dict[str, Any]],
    claim: str,
    max_output_words: int = 1500,
    embedding_semaphore: asyncio.Semaphore | None = None,
    embedding_model: str = "text-embedding-3-small",
    block_size: int = 10000,
    overlap: int = 200,
    deduplication_threshold: float = 0.85,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Return the most semantically relevant content blocks for *claim*.

    Args:
        websearch_results:        List of dicts with keys ``title``, ``url``,
                                  ``snippet``, ``content``.
        claim:                    The query/claim to match against.
        max_output_words:         Word budget for the returned text.
        embedding_semaphore:      Optional semaphore to cap concurrent API calls.
        embedding_model:          OpenAI embedding model name.
        block_size:               Max characters per block.
        overlap:                  Overlap between adjacent blocks.
        deduplication_threshold:  Cosine-similarity threshold for dedup.
        openai_api_key:           API key (falls back to env var).
        openai_base_url:          Base URL for OpenAI-compatible endpoints.

    Returns:
        Formatted string of relevant blocks grouped by source, or "" on failure.
    """
    if not websearch_results:
        return ""

    client = get_openai_client(api_key=openai_api_key, base_url=openai_base_url)

    # Split each source into blocks
    all_blocks: List[str] = []
    block_sources: List[int] = []
    for src_idx, source in enumerate(websearch_results):
        for block in split_into_blocks(source.get("content", ""), block_size, overlap):
            if len(block.strip()) > 20:
                all_blocks.append(block)
                block_sources.append(src_idx)

    if not all_blocks:
        return ""

    async def _embed(texts: List[str]) -> List[np.ndarray]:
        for attempt in range(4):
            try:
                await asyncio.sleep(random.uniform(0, 0.2))
                resp = await client.embeddings.create(model=embedding_model, input=texts)
                return [np.array(item.embedding) for item in resp.data]
            except Exception as e:
                if attempt < 3:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    _logger.warning(f"Embedding retry {attempt+1}: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise

    async def _embed_guarded(texts: List[str]) -> List[np.ndarray]:
        if embedding_semaphore:
            async with embedding_semaphore:
                return await _embed(texts)
        return await _embed(texts)

    # Embed claim
    try:
        claim_emb = (await _embed_guarded([claim]))[0]
    except Exception as e:
        _logger.warning(f"Claim embedding failed: {e}")
        return ""

    # Embed all blocks in batches of 50
    batch_size = 50
    tasks = [
        _embed_guarded(all_blocks[i : i + batch_size])
        for i in range(0, len(all_blocks), batch_size)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    block_embs: List[np.ndarray] = []
    for r in results:
        if isinstance(r, Exception):
            _logger.warning(f"Batch embedding failed: {r}")
        else:
            block_embs.extend(r)

    if not block_embs:
        return ""

    emb_matrix = np.array(block_embs)
    block_sources = block_sources[: len(block_embs)]
    all_blocks = all_blocks[: len(block_embs)]

    sims = _cosine_similarity(claim_emb, emb_matrix)
    ranked = np.argsort(sims)[::-1]

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_norm = emb_matrix / (norms + 1e-10)

    selected_blocks: List[tuple] = []  # (src_idx, text)
    word_count = 0
    sel_indices: List[int] = []

    for idx in ranked:
        block = all_blocks[idx]
        src_idx = block_sources[idx]

        # Deduplication
        if sel_indices:
            sel_embs = emb_norm[sel_indices]
            pair_sims = np.dot(sel_embs, emb_norm[idx])
            if np.any(pair_sims > deduplication_threshold):
                continue

        wc = len(block.split())
        if word_count + wc > max_output_words:
            remaining = max_output_words - word_count
            if remaining > 10:
                selected_blocks.append((src_idx, " ".join(block.split()[:remaining]) + "..."))
            break

        selected_blocks.append((src_idx, block))
        word_count += wc
        sel_indices.append(idx)

    # Group by source and format
    by_source: defaultdict = defaultdict(list)
    for src_idx, text in selected_blocks:
        by_source[src_idx].append(text)

    parts = []
    for src_idx in sorted(by_source.keys()):
        src = websearch_results[src_idx]
        header = f"-- Source: {src.get('title','Unknown')}, url: {src.get('url','')} --"
        snippet = src.get("snippet", "")
        body = "\n\n".join(by_source[src_idx])
        section = header + ("\n" + snippet + "\n\n" if snippet else "\n") + body
        parts.append(section)

    return "\n\n".join(parts)
