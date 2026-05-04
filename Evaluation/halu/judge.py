"""Concept-bucket judge.

One JSON-mode LLM call per ``ConceptBucket``: judge receives the concept, all
numbered claims about it, and the evidence blob (from layers 1-4). Returns
one verdict per claim. Score is computed deterministically in Python.

Caching:
  * key = (judge_model, canonical_concept, sha1(concat claims), sha1(concat evidence))
  * file = <output-dir>/cache/judgements.jsonl

Interface mirrors the extractor: sync ``judge_sync`` + async ``judge`` wrapper
around ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

_JUDGE_INNER_RETRIES = int(os.environ.get("HALU_JUDGE_INNER_RETRIES", "4"))
_JUDGE_RETRY_BASE_SLEEP = float(os.environ.get("HALU_JUDGE_RETRY_BASE_SLEEP", "1.5"))

_EVAL_DIR = Path(__file__).resolve().parents[1]
_PARENT = _EVAL_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from pubmed_graph.llm import InternChatClient

from .types import Claim, ConceptBucket, Evidence, JudgedClaim


_SYSTEM_PROMPT = """You are a biomedical fact-checking judge.

Given:
  * ONE concept.
  * A numbered list of CLAIMS a model made about that concept.
  * An EVIDENCE blob assembled from gold supporting chunks and/or retrieval.

For EACH claim, decide:
  * "supported"     — evidence explicitly or strongly implies the claim.
  * "refuted"       — evidence directly contradicts the claim.
  * "unverifiable"  — evidence is silent or only tangentially related.

Rules:
  * Use ONLY the evidence provided. Do not rely on your own training knowledge.
  * If the evidence does not explicitly address a claim, return "unverifiable",
    NOT "supported".
  * If the evidence contradicts PART of a claim but supports another part,
    return "refuted".

Output a JSON LIST, one object per claim in the input order:
  [{"claim_idx": 0, "verdict": "supported"|"refuted"|"unverifiable",
    "rationale": "<=40 words", "evidence_quote": "<short span from evidence or empty>"}, ...]

Return JSON ONLY. No markdown fences, no prose."""


_VERDICT_SCORE = {
    "supported": 0.0,
    "unverifiable": 0.5,
    "refuted": 1.0,
}


def _format_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    parts: list[str] = []
    for ev in evidence:
        tag = f"[{ev.source.upper()}]"
        if ev.url:
            tag = f"{tag}({ev.url})"
        parts.append(f"{tag}\n{ev.text}")
    return "\n\n".join(parts)


def _format_claims(claims: list[Claim]) -> str:
    return "\n".join(f"{i}. {c.text}" for i, c in enumerate(claims))


class BucketJudge:
    def __init__(
        self,
        client: InternChatClient,
        *,
        cache_dir: Path | None,
        use_cache: bool = True,
    ) -> None:
        self.client = client
        self.use_cache = use_cache and cache_dir is not None
        self._cache: dict[str, list[JudgedClaim]] = {}
        self._cache_path: Path | None = None
        if self.use_cache and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / "judgements.jsonl"
            if self._cache_path.is_file():
                self._load_cache()

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _cache_key(
        self,
        model: str,
        concept: str,
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> str:
        claim_blob = "|".join(c.text for c in claims)
        ev_blob = "|".join(f"{e.source}:{e.text[:500]}" for e in evidence)
        h = hashlib.sha1((claim_blob + "||" + ev_blob).encode("utf-8")).hexdigest()[:16]
        return f"{model}|{concept}|{h}"

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        for line in open(self._cache_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("key")
            jcs = rec.get("judged", [])
            if not key:
                continue
            out: list[JudgedClaim] = []
            for jc in jcs:
                try:
                    claim = Claim(**jc["claim"])
                    out.append(
                        JudgedClaim(
                            claim=claim,
                            verdict=jc.get("verdict", "unverifiable"),
                            score=float(jc.get("score", 0.5)),
                            rationale=jc.get("rationale", ""),
                            evidence_quote=jc.get("evidence_quote", ""),
                            evidence_source_used=jc.get("evidence_source_used", ""),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
            if out:
                self._cache[key] = out

    def _append_cache(self, key: str, judged: list[JudgedClaim]) -> None:
        if not self.use_cache or self._cache_path is None:
            return
        rec = {
            "key": key,
            "judged": [
                {
                    "claim": asdict(jc.claim),
                    "verdict": jc.verdict,
                    "score": jc.score,
                    "rationale": jc.rationale,
                    "evidence_quote": jc.evidence_quote,
                    "evidence_source_used": jc.evidence_source_used,
                }
                for jc in judged
            ],
        }
        with open(self._cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Judgement
    # ------------------------------------------------------------------
    def judge_sync(self, bucket: ConceptBucket, judge_model: str) -> list[JudgedClaim]:
        if not bucket.claims:
            return []

        # Empty evidence → short-circuit to all unverifiable; skip API call.
        if not bucket.evidence:
            return [
                JudgedClaim(
                    claim=c,
                    verdict="unverifiable",
                    score=_VERDICT_SCORE["unverifiable"],
                    rationale="No evidence retrieved from any chain layer.",
                    evidence_source_used="",
                )
                for c in bucket.claims
            ]

        key = self._cache_key(judge_model, bucket.canonical_concept, bucket.claims, bucket.evidence)
        if self.use_cache and key in self._cache:
            return self._cache[key]

        base_user_content = (
            f"CONCEPT: {bucket.canonical_concept}\n\n"
            f"CLAIMS (numbered):\n{_format_claims(bucket.claims)}\n\n"
            f"EVIDENCE:\n{_format_evidence(bucket.evidence)}"
        )

        # Inner retry with per-attempt nonce. The nonce is appended as a trailing
        # comment so the model ignores it semantically, but the request hash
        # changes every attempt — defeats gateway-level "same request failed
        # before, please modify" dedupe (e.g., boyue's new_api_error 400).
        last_exc: Exception | None = None
        raw: Any = None
        for attempt in range(_JUDGE_INNER_RETRIES + 1):
            nonce = f"{uuid.uuid4().hex[:12]}-{attempt}-{int(time.time()*1000)}"
            user_content = (
                f"{base_user_content}\n\n"
                f"# request_nonce: {nonce}  (ignore; for request-uniqueness only)"
            )
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            try:
                raw = self.client.chat_json(messages, temperature=0.0, max_tokens=1500)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= _JUDGE_INNER_RETRIES:
                    break
                sleep_s = _JUDGE_RETRY_BASE_SLEEP * (2 ** attempt)
                print(
                    f"[halu.judge] retry {attempt + 1}/{_JUDGE_INNER_RETRIES} on "
                    f"concept={bucket.canonical_concept} after {type(exc).__name__}; "
                    f"sleeping {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)

        if last_exc is not None:
            print(
                f"[halu.judge] LLM error on concept={bucket.canonical_concept} "
                f"after {_JUDGE_INNER_RETRIES + 1} attempts: "
                f"{type(last_exc).__name__}: {last_exc}"
            )
            # Fail-closed: unverifiable on LLM error so we don't falsely mark a hallucination.
            return [
                JudgedClaim(
                    claim=c,
                    verdict="unverifiable",
                    score=_VERDICT_SCORE["unverifiable"],
                    rationale=f"Judge LLM error ({type(last_exc).__name__}); treated as unverifiable.",
                    evidence_source_used=bucket.evidence_source_used,
                )
                for c in bucket.claims
            ]

        judged = _parse_judge_output(raw, bucket)
        if self.use_cache:
            self._cache[key] = judged
            self._append_cache(key, judged)
        return judged

    async def judge(self, bucket: ConceptBucket, judge_model: str) -> list[JudgedClaim]:
        return await asyncio.to_thread(self.judge_sync, bucket, judge_model)


def _parse_judge_output(raw: Any, bucket: ConceptBucket) -> list[JudgedClaim]:
    """Align LLM output list → bucket.claims by index. Robust to missing entries."""
    out: list[JudgedClaim] = []
    lookup: dict[int, dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "claim_idx" in item:
                try:
                    lookup[int(item["claim_idx"])] = item
                except (TypeError, ValueError):
                    continue

    for i, claim in enumerate(bucket.claims):
        item = lookup.get(i) or {}
        verdict = str(item.get("verdict", "unverifiable")).strip().lower()
        if verdict not in _VERDICT_SCORE:
            verdict = "unverifiable"
        out.append(
            JudgedClaim(
                claim=claim,
                verdict=verdict,   # type: ignore[arg-type]
                score=_VERDICT_SCORE[verdict],
                rationale=str(item.get("rationale", ""))[:400],
                evidence_quote=str(item.get("evidence_quote", ""))[:400],
                evidence_source_used=bucket.evidence_source_used,
            )
        )
    return out
