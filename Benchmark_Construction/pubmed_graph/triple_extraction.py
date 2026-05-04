from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from .entity_verification import ExternalFactEntityVerifier
from .llm import InternChatClient
from .normalize import normalize_triple_rows
from .ontology import Ontology
from .pubmed_client import PubMedClient
from .utils import ensure_dir, load_env_file, read_jsonl, write_jsonl

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_PATH_J2 = PROMPTS_DIR / "triple_extraction.j2"
PROMPT_PATH = PROMPTS_DIR / "triple_extraction.txt"


def _render_extraction_prompt(ontology: Ontology | None = None) -> str:
    """Render the triple-extraction system prompt.

    Prefers the Jinja template if present so the entity-type / relation lists
    come from ontology.yaml. Falls back to the legacy plain-text prompt for
    safety during the stage 1 transition.
    """
    if PROMPT_PATH_J2.exists():
        from jinja2 import Template

        onto = ontology or Ontology.default()
        template = Template(PROMPT_PATH_J2.read_text(encoding="utf-8"))
        return template.render(
            entity_types_block=onto.render_prompt_section_entity_types(),
            relations_block=onto.render_prompt_section_relations(),
            # Intentionally empty during stage 1 to keep the rendered prompt
            # textually equivalent to the legacy hand-written prompt. Stage 2
            # may inject onto.render_prompt_mapping_rules() here.
            relation_alias_block="",
        )
    return PROMPT_PATH.read_text(encoding="utf-8")


def extract_for_chunk(
    client: InternChatClient,
    prompt_text: str,
    chunk: dict[str, Any],
    model: str,
    confidence_threshold: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    user_content = "\n".join(
        [
            f"doc_id: {chunk.get('doc_id', '')}",
            f"chunk_id: {chunk.get('chunk_id', '')}",
            f"title: {chunk.get('title', '')}",
            f"section: {chunk.get('section', '')}",
            "text:",
            str(chunk.get("text", "")),
        ]
    )
    payload = client.chat_json(
        [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": user_content},
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=0,
    )
    rows = payload if isinstance(payload, list) else payload.get("triples", [])
    extracted = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if confidence <= confidence_threshold:
            continue
        extracted.append(
            {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "head": str(row.get("head", "")).strip(),
                "head_type": str(row.get("head_type", "")).strip(),
                "surface_relation": str(row.get("surface_relation", "")).strip(),
                "normalized_relation": str(row.get("normalized_relation", "")).strip(),
                "tail": str(row.get("tail", "")).strip(),
                "tail_type": str(row.get("tail_type", "")).strip(),
                "confidence": confidence,
                "evidence": str(row.get("evidence", "")).strip(),
            }
        )
    return [row for row in extracted if row["head"] and row["tail"] and row["normalized_relation"]]


def _select_chunks_for_extraction(
    chunks: list[dict[str, Any]],
    limit: int,
    strategy: str = "round_robin_by_doc",
) -> list[dict[str, Any]]:
    if limit <= 0 or len(chunks) <= limit:
        return chunks
    if strategy != "round_robin_by_doc":
        return chunks[:limit]
    grouped: dict[str, deque[dict[str, Any]]] = {}
    doc_order: list[str] = []
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", "")) or "__unknown__"
        if doc_id not in grouped:
            grouped[doc_id] = deque()
            doc_order.append(doc_id)
        grouped[doc_id].append(chunk)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        made_progress = False
        for doc_id in doc_order:
            bucket = grouped[doc_id]
            if not bucket:
                continue
            selected.append(bucket.popleft())
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected


def run_triple_extraction(
    config: dict[str, Any],
    chunks_path: str | Path,
    output_path: str | Path,
    limit: int = 0,
    raw_output_path: str | Path | None = None,
) -> dict[str, Any]:
    load_env_file(config.get("env_file"))
    extraction_cfg = dict(config.get("openai_extraction", {}))
    extraction_cfg.setdefault("model", "intern-s1-pro")
    extraction_cfg.setdefault("thinking_mode", False)
    extraction_cfg.setdefault("max_tokens", 1200)
    client = InternChatClient(extraction_cfg)
    prompt_text = _render_extraction_prompt()
    chunks = read_jsonl(chunks_path)
    chunk_selection_strategy = str(extraction_cfg.get("chunk_selection_strategy", "round_robin_by_doc"))
    if limit > 0:
        chunks = _select_chunks_for_extraction(chunks, limit, strategy=chunk_selection_strategy)

    raw_rows: list[dict[str, Any]] = []
    error_count = 0
    confidence_threshold = float(extraction_cfg.get("confidence_threshold", 0.5))
    max_tokens = int(extraction_cfg.get("max_tokens", 1200))

    # Stream raw rows to disk as they are produced so a mid-run crash
    # (OOM, API outage, Ctrl+C, node preemption) doesn't lose hours of
    # LLM work. The final write_jsonl pass at the bottom is still the
    # authoritative `raw_triples.jsonl` / `normalized_triples.jsonl`
    # output — this checkpoint just makes sure we can always recover
    # the raw rows even if we never reach that line.
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if raw_output_path is not None:
        checkpoint_path = Path(raw_output_path)
    else:
        checkpoint_path = output_path.parent / "raw_triples.checkpoint.jsonl"
    ensure_dir(checkpoint_path.parent)
    progress_every = max(int(extraction_cfg.get("progress_every", 25)), 1)
    total_chunks = len(chunks)
    start_ts = time.time()
    with open(checkpoint_path, "w", encoding="utf-8") as checkpoint_handle:
        for idx, chunk in enumerate(chunks, start=1):
            try:
                produced = extract_for_chunk(
                    client,
                    prompt_text,
                    chunk,
                    extraction_cfg.get("model", "intern-s1-pro"),
                    confidence_threshold,
                    max_tokens,
                )
            except Exception as exc:
                error_count += 1
                produced = [
                    {
                        "doc_id": chunk.get("doc_id", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "head": "",
                        "head_type": "",
                        "surface_relation": "",
                        "normalized_relation": "",
                        "tail": "",
                        "tail_type": "",
                        "confidence": 0.0,
                        "evidence": f"ERROR: {exc}",
                    }
                ]
            raw_rows.extend(produced)
            for row in produced:
                checkpoint_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint_handle.flush()
            if idx % progress_every == 0 or idx == total_chunks:
                elapsed = time.time() - start_ts
                rate = idx / elapsed if elapsed > 0 else 0.0
                eta = (total_chunks - idx) / rate if rate > 0 else 0.0
                print(
                    f"[triple_extraction] {idx}/{total_chunks} chunks "
                    f"({rate:.2f}/s, eta {eta/60:.1f}m) "
                    f"raw={len([r for r in raw_rows if r.get('head')])} errors={error_count}",
                    file=sys.stderr,
                    flush=True,
                )

    verifier = None
    verification_cfg = dict(config.get("entity_verification", {}))
    if verification_cfg.get("enabled", False):
        pubmed_cfg = dict(config.get("pubmed", {}))
        verifier = ExternalFactEntityVerifier(
            verification_cfg,
            pubmed_client=PubMedClient(api_key=pubmed_cfg.get("api_key"), email=pubmed_cfg.get("email")),
            default_llm_config=extraction_cfg,
        )
    normalized_rows = normalize_triple_rows(
        raw_rows,
        confidence_threshold=confidence_threshold,
        entity_verifier=verifier,
    )
    write_jsonl(output_path, normalized_rows)

    # checkpoint_path already holds the streamed raw rows. If the caller
    # requested an explicit raw_output_path it IS the checkpoint, so we're
    # done. Otherwise the checkpoint sits next to the normalized output as
    # `raw_triples.checkpoint.jsonl` for post-mortem inspection.
    final_raw_path = checkpoint_path if raw_output_path is not None else None

    return {
        "status": "ok",
        "chunks": len(chunks),
        "raw_triples": len([row for row in raw_rows if row.get("head")]),
        "normalized_triples": len(normalized_rows),
        "errors": error_count,
        "output": str(output_path),
        "raw_output": str(final_raw_path) if final_raw_path else str(checkpoint_path),
        "checkpoint": str(checkpoint_path),
    }
