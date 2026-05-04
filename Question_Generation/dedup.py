from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pubmed_graph.utils import normalize_text

from .indexing import canonicalize_entity


logger = logging.getLogger("question_generation.dedup")


def normalize_question_text(text: str) -> str:
    return normalize_text(text).casefold()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer", "")
    return str(_field(answer, "canonical_text", "") or _field(answer, "text", "") or answer or "")


def _option_texts(row: dict[str, Any]) -> tuple[str, ...]:
    options = row.get("options", []) or []
    out: list[str] = []
    if not isinstance(options, list):
        return ()
    for option in options:
        text = _field(option, "text", "")
        if text:
            out.append(str(text))
    return tuple(out)


def _canonical_option_set(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(canonicalize_entity(text) for text in _option_texts(row) if canonicalize_entity(text)))


def dedup_key(row: dict[str, Any]) -> str:
    """Return a collision-resistant key for algorithmically distinct samples.

    Multiple-choice questions, especially claim_choice, can share the same
    stem while having different option sets and different correct answers. For
    those rows the key must include options and answer; for open-ended rows the
    normalized question text remains the conservative de-duplication key.
    """
    question = normalize_question_text(str(row.get("question", "")))
    qtype = str(row.get("question_type", "")).casefold()
    if qtype in {"claim_choice", "one_hop_tail", "two_hop_tail", "boolean_support"}:
        answer = canonicalize_entity(_answer_text(row))
        options = _canonical_option_set(row)
        return json.dumps(
            {
                "question_type": qtype,
                "question": question,
                "answer": answer,
                "options": options,
            },
            sort_keys=True,
        )
    return question


def deduplicate_by_question(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        key = dedup_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def load_seen_questions(paths: list[str | Path]) -> set[str]:
    """Read one or more samples.jsonl files and return previously seen keys.

    Used by ``--dedup-against`` in the CLI to skip samples that already exist
    in previous outputs. Multiple-choice keys include options and answers so
    distinct candidate-claim questions with the same stem are not collapsed.
    """
    seen: set[str] = set()
    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            logger.warning("--dedup-against path not found: %s — skipping", p)
            continue
        count_before = len(seen)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = dedup_key(row)
                if key:
                    seen.add(key)
        logger.info(
            "--dedup-against %s: loaded %d new sample keys (set size: %d)",
            p, len(seen) - count_before, len(seen),
        )
    return seen


def deduplicate_against(rows: list[dict], seen: set[str]) -> tuple[list[dict], int]:
    """Drop rows whose de-duplication key is already in ``seen``.

    Returns ``(kept, dropped_count)``.
    """
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        key = dedup_key(row)
        if key and key in seen:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped
