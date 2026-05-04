"""Benchmark-to-triples overlay.

For each benchmark QA item, produce either:
  - 1 self-loop when exactly one biomedical concept is extracted
  - real-semantic triples when 2+ concepts are extracted and the LLM can
    identify high-confidence relations between them
  - nothing (and log to a skipped file) when 0 concepts pass the filter

Concept extraction reuses benchmark_seeds._extract_from_question (regex
path) and the existing BENCHMARK_SEED_SYSTEM_PROMPT (LLM path) so the
same signal-based biomedical filter that produced the 158 ProteinLMBench
seeds is applied here too. No ontology-type whitelist is enforced.
"""
from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_seeds import (
    BENCHMARK_SEED_SYSTEM_PROMPT,
    _canonical_seed_key,
    _canonicalize_seed_candidate,
    _extract_from_question,
    _is_useful_candidate,
    _split_compound_candidate,
)
from .llm import InternChatClient
from .models import TripleRecord


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
BENCHMARK_TRIPLE_PROMPT_PATH = PROMPTS_DIR / "benchmark_triple_extraction.txt"

SELF_LOOP_RELATION = "benchmark_evidence"


@dataclass
class BenchmarkItem:
    dataset: str
    split: str
    question_id: str
    question: str
    answer: str = ""
    question_type: str = ""
    image_ref: dict[str, Any] | None = None


@dataclass
class BenchmarkExtractionResult:
    item: BenchmarkItem
    entities: list[str] = field(default_factory=list)
    triples: list[TripleRecord] = field(default_factory=list)
    skip_reason: str = ""


class BenchmarkTripleExtractor:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_client: InternChatClient | None = None,
    ):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.llm_entity_extraction = bool(cfg.get("llm_entity_extraction", True))
        self.max_entities_per_item = max(int(cfg.get("max_entities_per_item", 12)), 1)
        self.relation_confidence_threshold = float(cfg.get("relation_confidence_threshold", 0.6))
        self.self_loop_confidence = float(cfg.get("self_loop_confidence", 0.9))
        self.evidence_char_limit = max(int(cfg.get("evidence_char_limit", 600)), 80)
        self.verbose = bool(cfg.get("verbose", False))

        llm_cfg = dict(cfg.get("llm", {}))
        llm_cfg.setdefault("model", "intern-s1-pro")
        llm_cfg.setdefault("thinking_mode", False)
        llm_cfg.setdefault("temperature", 0.0)
        llm_cfg.setdefault("max_tokens", 900)
        self.llm = llm_client or InternChatClient(llm_cfg)

        self._triple_prompt = BENCHMARK_TRIPLE_PROMPT_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Entity extraction
    # ------------------------------------------------------------------ #

    def _regex_entities(self, text: str) -> list[str]:
        return _extract_from_question(text)

    def _llm_entities(self, item: BenchmarkItem) -> list[str]:
        if not self.llm_entity_extraction:
            return []
        user = (
            f"Benchmark: {item.dataset}  question_type={item.question_type or 'n/a'}\n\n"
            f"QUESTION:\n{item.question}\n\nANSWER:\n{item.answer}\n\n"
            f"Return at most {self.max_entities_per_item} seed keywords."
        )
        try:
            payload = self.llm.chat_json(
                [
                    {"role": "system", "content": BENCHMARK_SEED_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ]
            )
        except Exception as exc:
            if self.verbose:
                print(f"  [llm-entity] {item.question_id}: {exc}")
            return []
        items = payload.get("seed_keywords") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        return [str(x).strip() for x in items if str(x).strip()]

    def _collect_entities(self, item: BenchmarkItem) -> list[str]:
        qa_text = (item.question + "\n" + item.answer).strip()
        regex_cands = self._regex_entities(qa_text)
        llm_cands = self._llm_entities(item)

        merged: list[str] = []
        seen: set[str] = set()
        for candidate in list(regex_cands) + list(llm_cands):
            for split in _split_compound_candidate(candidate):
                canonical = _canonicalize_seed_candidate(split)
                if not _is_useful_candidate(canonical):
                    continue
                key = _canonical_seed_key(canonical)
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(canonical)
                if len(merged) >= self.max_entities_per_item:
                    return merged
        return merged

    # ------------------------------------------------------------------ #
    # Triple extraction (LLM)
    # ------------------------------------------------------------------ #

    def _llm_triples(self, item: BenchmarkItem, entities: list[str]) -> list[dict[str, Any]]:
        user = (
            f"Benchmark: {item.dataset}  question_type={item.question_type or 'n/a'}\n"
            f"QUESTION:\n{item.question}\n\nANSWER:\n{item.answer}\n\n"
            f"CANDIDATE ENTITIES (pick head/tail only from this list):\n"
            + "\n".join(f"  - {e}" for e in entities)
        )
        try:
            payload = self.llm.chat_json(
                [
                    {"role": "system", "content": self._triple_prompt},
                    {"role": "user", "content": user},
                ]
            )
        except Exception as exc:
            if self.verbose:
                print(f"  [llm-triple] {item.question_id}: {exc}")
            return []
        triples = payload.get("triples") if isinstance(payload, dict) else []
        if not isinstance(triples, list):
            return []
        return [t for t in triples if isinstance(t, dict)]

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def extract(self, item: BenchmarkItem) -> BenchmarkExtractionResult:
        result = BenchmarkExtractionResult(item=item)
        if not self.enabled:
            result.skip_reason = "extractor_disabled"
            return result

        entities = self._collect_entities(item)
        result.entities = entities

        if len(entities) == 0:
            result.skip_reason = "no_biomedical_entity"
            return result

        evidence_text = _truncate(_combine_qa(item), self.evidence_char_limit)
        meta = _build_meta(item)
        doc_id = f"benchmark::{item.dataset}::{item.split}::{item.question_id}"
        chunk_id = f"{doc_id}::q"

        if len(entities) == 1:
            e = entities[0]
            result.triples.append(
                TripleRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    head=e,
                    head_type="",
                    surface_relation="benchmark_evidence",
                    normalized_relation=SELF_LOOP_RELATION,
                    tail=e,
                    tail_type="",
                    confidence=self.self_loop_confidence,
                    evidence=evidence_text,
                    source="benchmark",
                    meta=meta,
                )
            )
            return result

        entity_set = {_canonical_seed_key(e): e for e in entities}

        raw_triples = self._llm_triples(item, entities)
        accepted: list[TripleRecord] = []
        for rt in raw_triples:
            head = str(rt.get("head", "")).strip()
            tail = str(rt.get("tail", "")).strip()
            relation = str(rt.get("relation", "")).strip().lower().replace(" ", "_")
            try:
                confidence = float(rt.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not head or not tail or not relation:
                continue
            if confidence < self.relation_confidence_threshold:
                continue
            head_canon = entity_set.get(_canonical_seed_key(head))
            tail_canon = entity_set.get(_canonical_seed_key(tail))
            if head_canon is None or tail_canon is None:
                continue
            if head_canon == tail_canon:
                continue
            evidence_span = str(rt.get("evidence_span", "")).strip()
            accepted.append(
                TripleRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    head=head_canon,
                    head_type="",
                    surface_relation=relation,
                    normalized_relation=relation,
                    tail=tail_canon,
                    tail_type="",
                    confidence=confidence,
                    evidence=evidence_span or evidence_text,
                    source="benchmark",
                    meta=meta,
                )
            )

        if accepted:
            result.triples.extend(accepted)
            return result

        # 2+ entities but LLM found no high-confidence relation:
        # preserve evidence as self-loops on each entity so concepts stay linked
        # back to the QA, without inventing cross-entity semantics.
        for e in entities:
            result.triples.append(
                TripleRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    head=e,
                    head_type="",
                    surface_relation="benchmark_evidence",
                    normalized_relation=SELF_LOOP_RELATION,
                    tail=e,
                    tail_type="",
                    confidence=self.self_loop_confidence,
                    evidence=evidence_text,
                    source="benchmark",
                    meta=meta,
                )
            )
        result.skip_reason = "no_high_confidence_relation;fallback_to_self_loops"
        return result


def _combine_qa(item: BenchmarkItem) -> str:
    parts = []
    if item.question:
        parts.append(f"Q: {item.question}")
    if item.answer:
        parts.append(f"A: {item.answer}")
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_meta(item: BenchmarkItem) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "dataset": item.dataset,
        "split": item.split,
        "question_id": item.question_id,
        "question_type": item.question_type or "",
    }
    if item.image_ref:
        meta["image_ref"] = item.image_ref
    return meta


def write_triples_jsonl(triples: list[TripleRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in triples:
            fh.write(json.dumps(dataclasses.asdict(t), ensure_ascii=False) + "\n")
    return len(triples)


def run_benchmark_extraction(
    items: list[BenchmarkItem],
    config: dict[str, Any],
    output_dir: Path,
    progress_every: int = 20,
    max_workers: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_cfg = dict(config.get("benchmark_overlay", {}))
    extractor = BenchmarkTripleExtractor(overlay_cfg)

    if max_workers is None:
        max_workers = int(overlay_cfg.get("max_workers", 6))
    max_workers = max(1, min(int(max_workers), 32))

    results: dict[int, BenchmarkExtractionResult] = {}
    t0 = time.time()

    # Per-item LLM calls are independent and httpx-backed; concurrency is
    # bounded by Intern API rate limits (InternChatClient already retries
    # with exponential backoff on 429/5xx).
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(extractor.extract, item): idx
                      for idx, item in enumerate(items)}
        completed = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = BenchmarkExtractionResult(
                    item=items[idx], skip_reason=f"error:{type(exc).__name__}:{exc}"
                )
            completed += 1
            if completed % progress_every == 0 or completed == len(items):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                triple_count = sum(len(r.triples) for r in results.values())
                print(f"  [benchmark] {completed}/{len(items)} items, "
                      f"{triple_count} triples so far, "
                      f"{rate:.2f} items/s (workers={max_workers})", flush=True)

    # reassemble in submission order → deterministic output regardless of
    # future completion order
    all_triples: list[TripleRecord] = []
    used_items: list[BenchmarkItem] = []
    skipped: list[dict[str, Any]] = []
    for idx in range(len(items)):
        res = results[idx]
        if res.triples:
            all_triples.extend(res.triples)
            used_items.append(res.item)
        else:
            skipped.append({
                "question_id": res.item.question_id,
                "reason": res.skip_reason or "unknown",
                "entities": res.entities,
            })

    triples_path = output_dir / "benchmark_triples.jsonl"
    write_triples_jsonl(all_triples, triples_path)

    skipped_path = output_dir / "benchmark_skipped.jsonl"
    with skipped_path.open("w", encoding="utf-8") as fh:
        for row in skipped:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "items_total": len(items),
        "items_contributing": len(used_items),
        "items_skipped": len(skipped),
        "triples_emitted": len(all_triples),
        "self_loops": sum(1 for t in all_triples if t.normalized_relation == SELF_LOOP_RELATION),
        "semantic_edges": sum(1 for t in all_triples if t.normalized_relation != SELF_LOOP_RELATION),
        "triples_path": str(triples_path),
        "skipped_path": str(skipped_path),
        "used_items": used_items,
    }
