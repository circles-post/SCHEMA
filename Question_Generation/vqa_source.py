"""Load PathVQA-style benchmark overlay triples + image index into
VqaRecord objects the sampler can consume.

Expected input shape (current ``protein_plus_pathvqa_500_v3`` layout):

* ``benchmark_triples.jsonl`` — rows have ``head == tail`` (self-loop),
  ``normalized_relation == "benchmark_evidence"``, and ``evidence`` field
  like ``"Q: what are positively charged ... ?\\nA: the histone subunits"``.

* ``benchmark_image_index.json`` — flat dict of
  ``{short_doc_id: absolute_image_path}`` where ``short_doc_id`` is the
  ``PathVQA::test::<file>::<N>`` portion of the triple's full ``doc_id``.

We:

1. Strip the ``benchmark::`` and leading ``PathVQA::test::`` prefix chain
   on each triple's ``doc_id`` until it matches an image_index key.
2. Split ``evidence`` on the first ``\\nA:`` (tolerant of spacing) into
   ``question_q`` / ``answer_a``.
3. Detect ``yesno`` vs ``open`` by looking at the normalized answer.
4. Drop rows with no image match, malformed Q/A, empty answer, or a
   ``head != tail`` (those are the 3 real-relation rows in PathVQA — not
   VQA candidates).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("question_generation.vqa_source")


_YESNO_TOKENS = {"yes", "no"}
_QA_SPLIT_RE = re.compile(r"\n\s*A\s*:\s*", flags=re.IGNORECASE)
_Q_PREFIX_RE = re.compile(r"^\s*Q\s*:\s*", flags=re.IGNORECASE)
# Strip the redundant benchmark chain prefixes when joining with image_index.
# doc_id format:   benchmark::PathVQA::test::PathVQA::test::<file>::<n>
# image_index key: PathVQA::test::<file>::<n>
# We progressively strip any number of ``benchmark::`` and the initial
# ``PathVQA::test::`` that duplicates the later one.
_BENCHMARK_PREFIX = "benchmark::"


@dataclass
class VqaRecord:
    """One VQA question extracted from a benchmark overlay self-loop."""

    doc_id: str                 # original full doc_id (for provenance)
    image_key: str              # the short key used to join with image_index
    image_path: str             # absolute path to the jpg/png
    entity: str                 # the self-loop head (== tail)
    question_q: str             # text of the Q portion of evidence
    answer_a: str               # text of the A portion of evidence
    vqa_format: str             # "yesno" | "open"
    raw_evidence: str           # the original "Q:... A:..." combined text
    chunk_id: str = ""
    source: str = "PathVQA"


def _derive_image_key(doc_id: str, index_keys: set[str]) -> str | None:
    """Return the subsection of ``doc_id`` that matches a key in
    ``benchmark_image_index.json``, or ``None`` if no match.

    Progressively strips ``benchmark::`` prefixes and tests each
    candidate; the first one that lives in ``index_keys`` wins.
    """
    s = doc_id
    # strip repeated benchmark:: prefix
    while s.startswith(_BENCHMARK_PREFIX):
        s = s[len(_BENCHMARK_PREFIX):]
    if s in index_keys:
        return s
    # Some triples duplicate "PathVQA::test::" inside the id; walk
    # past successive ``PathVQA::`` segments until a match.
    while "::" in s:
        if s in index_keys:
            return s
        # drop one leading segment (e.g. "PathVQA::")
        s = s.split("::", 1)[1]
    if s in index_keys:
        return s
    return None


def _parse_qa(evidence: str) -> tuple[str, str] | None:
    """Split ``"Q: ...\\nA: ..."`` into (question, answer). Returns
    ``None`` on malformed input.
    """
    if not evidence:
        return None
    parts = _QA_SPLIT_RE.split(evidence, maxsplit=1)
    if len(parts) != 2:
        return None
    q_raw, a_raw = parts[0], parts[1]
    q_clean = _Q_PREFIX_RE.sub("", q_raw).strip()
    a_clean = a_raw.strip()
    if not q_clean or not a_clean:
        return None
    return q_clean, a_clean


def _detect_format(answer: str) -> str:
    return "yesno" if answer.strip().casefold().rstrip(".!?") in _YESNO_TOKENS else "open"


def load_vqa_source(
    benchmark_triples_path: str | Path,
    image_index_path: str | Path,
) -> list[VqaRecord]:
    """Emit VqaRecord objects ready for the sampler.

    Silently drops malformed rows, rows without images, and non-self-loop
    rows (those 3 real-relation rows in PathVQA which aren't VQA).
    """
    triples_path = Path(benchmark_triples_path)
    index_path = Path(image_index_path)
    if not triples_path.exists():
        logger.warning("vqa benchmark triples not found: %s", triples_path)
        return []
    if not index_path.exists():
        logger.warning("vqa image index not found: %s", index_path)
        return []

    image_index: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))
    index_keys = set(image_index.keys())

    records: list[VqaRecord] = []
    skipped_non_selfloop = 0
    skipped_no_image = 0
    skipped_malformed = 0
    seen_keys: set[str] = set()

    for line in triples_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            skipped_malformed += 1
            continue
        head = str(t.get("head", "") or "")
        tail = str(t.get("tail", "") or "")
        if head.casefold() != tail.casefold():
            skipped_non_selfloop += 1
            continue
        doc_id = str(t.get("doc_id", "") or "")
        image_key = _derive_image_key(doc_id, index_keys)
        if not image_key:
            skipped_no_image += 1
            continue
        parsed = _parse_qa(str(t.get("evidence", "") or ""))
        if not parsed:
            skipped_malformed += 1
            continue
        question_q, answer_a = parsed

        # De-dup: multiple self-loop triples per doc_id (several head entities
        # from the same QA) all share the same Q/A; keep only the first.
        if image_key in seen_keys:
            continue
        seen_keys.add(image_key)

        records.append(
            VqaRecord(
                doc_id=doc_id,
                image_key=image_key,
                image_path=image_index[image_key],
                entity=head,
                question_q=question_q,
                answer_a=answer_a,
                vqa_format=_detect_format(answer_a),
                raw_evidence=str(t.get("evidence", "") or ""),
                chunk_id=str(t.get("chunk_id", "") or ""),
                source="PathVQA",
            )
        )

    logger.info(
        "vqa_source: loaded %d records (dropped: non_selfloop=%d, no_image=%d, malformed=%d)",
        len(records), skipped_non_selfloop, skipped_no_image, skipped_malformed,
    )
    return records
