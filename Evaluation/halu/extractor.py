"""Claim extractor.

Per step, prompt a small LLM with JSON mode to emit atomic factual claims
paired with the concept/entity they are about. Phase-1 uses the sync
``pubmed_graph.llm.InternChatClient.chat_json`` (OpenAI-compatible under the
hood) — ``asyncio.to_thread`` wraps it from the async CLI.

Normalization strategy (phase-1):
  * ``normalize_keyword(concept)`` from pubmed_graph.normalize → canonical bucket key
  * BGE-Large fallback is a stub; phase-2 will plug in ``BGELargeEmbedder``.

Cache:
  * keyed by (extractor_model, sample_id, step_idx, sha1(step_text))
  * file: <output-dir>/cache/extractions.jsonl
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# pubmed_graph lives one directory up (datasetsa/) — make sure it's on sys.path
_EVAL_DIR = Path(__file__).resolve().parents[1]
_PARENT = _EVAL_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from pubmed_graph.llm import InternChatClient
from pubmed_graph.normalize import normalize_keyword

from .types import Claim, ConceptBucket, Step


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You extract atomic FACTUAL claims about entities from a
biomedical AI agent's internal reasoning.

A "factual claim" is a concrete assertion about the world that could in
principle be checked against evidence: a gene/protein's function, a
drug-target interaction, a disease association, a pathway, a numerical
parameter, a mechanism of action, etc.

EXTRACT (this is the most common failure mode — don't skip these):
  * Hedged factual assertions. "Evidence suggests that X inhibits Y",
    "The literature supports X's role in Y", "Based on the snippets, X
    appears to modulate Y" all contain the extractable claim "X inhibits Y"
    / "X has a role in Y" / "X modulates Y". Strip the hedge, keep the claim.
  * Option-evaluation reasoning in multiple-choice answers. If the agent
    writes "Option B says TAF2 is associated with microcephaly, which
    matches the evidence", extract "TAF2 is associated with microcephaly"
    as a factual claim (regardless of which option the agent picked).
  * Negative claims. "H1N1 is NOT directly inhibited by X" is a factual
    claim about the mechanism's absence — extract it with the negation
    preserved in the claim text.
  * Claims attributed to sources. "The paper states that X causes Y" →
    extract "X causes Y" (the attribution itself is not a fact about the
    world, but the content it attributes is).

Do NOT extract:
  * Pure procedural plans. "I should search for X", "Let me try
    literature_search", "I will verify this next".
  * Pure hedges with no factual content. "This is unclear", "I'm not sure".
  * Tool-call stubs like "[tool_call:web_search]" with only a query
    string — these are plans, not claims.
  * Verbatim restatements of the question itself.
  * Meta-commentary about the answer format ("I will wrap my answer in
    <answer> tags").

For each factual claim emit a JSON object:
  {
    "concept":           <the ONE most-salient entity/concept it's about, surface form as written>,
    "canonical_concept": <lowercased+punct-stripped version of concept>,
    "claim":             <one-sentence paraphrase stripped of hedge language; self-contained (no "it"/"this")>,
    "claim_type":        "factual"
  }

Return a JSON LIST. Empty list only if the step truly contains zero factual
content. Do NOT wrap in markdown. Do NOT add explanatory prose."""

_FEWSHOT = [
    # Example 1 — direct factual assertions
    {
        "role": "user",
        "content": (
            "QUESTION: What does TAF1 do in neurodevelopment?\n"
            "STEP: Mutations in TAF1, a TFIID subunit, cause intellectual disability and microcephaly in XLID patients. "
            "TAF2 mutations have a similar phenotype."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            [
                {
                    "concept": "TAF1",
                    "canonical_concept": "taf1",
                    "claim": "TAF1 is a subunit of the TFIID transcription factor complex.",
                    "claim_type": "factual",
                },
                {
                    "concept": "TAF1",
                    "canonical_concept": "taf1",
                    "claim": "Mutations in TAF1 cause intellectual disability and microcephaly in XLID.",
                    "claim_type": "factual",
                },
                {
                    "concept": "TAF2",
                    "canonical_concept": "taf2",
                    "claim": "TAF2 mutations produce a phenotype similar to TAF1 mutations (intellectual disability, microcephaly).",
                    "claim_type": "factual",
                },
            ]
        ),
    },
    # Example 2 — pure plan, no facts
    {
        "role": "user",
        "content": (
            "QUESTION: Does NITD-448 inhibit DENV E protein?\n"
            "STEP: I should look up NITD-448 in literature_search to see if there's a direct mechanism described."
        ),
    },
    {"role": "assistant", "content": "[]"},
    # Example 3 — HEDGED essay reasoning (must extract despite hedge)
    {
        "role": "user",
        "content": (
            "QUESTION: How does isovaleric acid affect influenza?\n"
            "STEP: The literature on 'Gut microbiota-derived isovaleric acid ameliorates influenza virus infection' "
            "examines effects of isovaleric acid during influenza infections. "
            "The direct support can be inferred for claims related to inflammatory responses. "
            "While H1N1 was not specifically mentioned as being inhibited, the mechanism whereby "
            "isovaleric acid helps in reducing inflammatory responses is emphasized."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            [
                {
                    "concept": "isovaleric acid",
                    "canonical_concept": "isovaleric acid",
                    "claim": "Isovaleric acid affects inflammatory responses during influenza virus infections.",
                    "claim_type": "factual",
                },
                {
                    "concept": "isovaleric acid",
                    "canonical_concept": "isovaleric acid",
                    "claim": "Isovaleric acid reduces inflammatory responses via gut microbiota.",
                    "claim_type": "factual",
                },
                {
                    "concept": "H1N1",
                    "canonical_concept": "h1n1",
                    "claim": "The literature does not specifically mention H1N1 being inhibited by isovaleric acid.",
                    "claim_type": "factual",
                },
            ]
        ),
    },
    # Example 4 — MCQ option evaluation (must extract claim from option text)
    {
        "role": "user",
        "content": (
            "QUESTION: Which compound targets the SARS-CoV-2 spike receptor binding domain?\n"
            "STEP: Option A says remdesivir, which targets RdRp not the spike, so not correct. "
            "Option B says a monoclonal antibody cocktail that binds the RBD of SARS-CoV-2 spike "
            "protein — this matches the known mechanism. Option C says paxlovid, which targets "
            "the 3CL protease. I will pick B."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            [
                {
                    "concept": "remdesivir",
                    "canonical_concept": "remdesivir",
                    "claim": "Remdesivir targets the RNA-dependent RNA polymerase (RdRp), not the SARS-CoV-2 spike protein.",
                    "claim_type": "factual",
                },
                {
                    "concept": "monoclonal antibody cocktail",
                    "canonical_concept": "monoclonal antibody cocktail",
                    "claim": "A monoclonal antibody cocktail can bind the receptor binding domain (RBD) of the SARS-CoV-2 spike protein.",
                    "claim_type": "factual",
                },
                {
                    "concept": "paxlovid",
                    "canonical_concept": "paxlovid",
                    "claim": "Paxlovid targets the 3CL protease of SARS-CoV-2.",
                    "claim_type": "factual",
                },
            ]
        ),
    },
]


# ---------------------------------------------------------------------------
# LLM-backed extraction
# ---------------------------------------------------------------------------
class ClaimExtractor:
    """Wraps an ``InternChatClient`` for claim extraction with disk caching."""

    def __init__(
        self,
        client: InternChatClient,
        *,
        cache_dir: Path | None,
        use_cache: bool = True,
    ) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None
        self._cache: dict[str, list[Claim]] = {}
        self._cache_path: Path | None = None
        if self.use_cache and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / "extractions.jsonl"
            if self._cache_path.is_file():
                self._load_cache()

    def _cache_key(self, model: str, sample_id: str, step_idx: int, step_text: str) -> str:
        h = hashlib.sha1(step_text.encode("utf-8")).hexdigest()[:16]
        return f"{model}|{sample_id}|{step_idx}|{h}"

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
            claims = rec.get("claims", [])
            if key:
                self._cache[key] = [
                    Claim(**c) for c in claims if isinstance(c, dict)
                ]

    def _append_cache(self, key: str, claims: list[Claim]) -> None:
        if not self.use_cache or self._cache_path is None:
            return
        with open(self._cache_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"key": key, "claims": [asdict(c) for c in claims]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def extract_sync(
        self,
        sample_id: str,
        step: Step,
        *,
        question: str,
        model_being_tested: str,
    ) -> list[Claim]:
        key = self._cache_key(model_being_tested, sample_id, step.step_idx, step.text)
        if self.use_cache and key in self._cache:
            return self._cache[key]

        user_content = (
            f"QUESTION: {question.strip()[:1000]}\n"
            f"STEP: {step.text.strip()[:4000]}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEWSHOT,
            {"role": "user", "content": user_content},
        ]
        try:
            raw = self.client.chat_json(messages, temperature=0.0, max_tokens=1200)
        except Exception as exc:  # noqa: BLE001
            # Extractor failures must not kill the pipeline — log and return [].
            print(f"[halu.extractor] LLM error on sample={sample_id} step={step.step_idx}: {type(exc).__name__}: {exc}")
            return []

        claims = _parse_claim_list(raw, sample_id=sample_id, step_idx=step.step_idx)
        if self.use_cache:
            self._cache[key] = claims
            self._append_cache(key, claims)
        return claims

    async def extract(
        self,
        sample_id: str,
        step: Step,
        *,
        question: str,
        model_being_tested: str,
    ) -> list[Claim]:
        return await asyncio.to_thread(
            self.extract_sync,
            sample_id,
            step,
            question=question,
            model_being_tested=model_being_tested,
        )


def _parse_claim_list(raw: Any, *, sample_id: str, step_idx: int) -> list[Claim]:
    if not isinstance(raw, list):
        return []
    out: list[Claim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ct = str(item.get("claim_type", "factual")).strip().lower() or "factual"
        if ct != "factual":
            continue
        concept = str(item.get("concept", "")).strip()
        claim_text = str(item.get("claim", "")).strip()
        if not concept or not claim_text:
            continue
        # Re-normalize in code — trust normalize_keyword over the model's canonical_concept.
        canonical = normalize_keyword(concept) or concept.lower().strip()
        out.append(
            Claim(
                sample_id=sample_id,
                step_idx=step_idx,
                text=claim_text,
                concept=concept,
                canonical_concept=canonical,
                claim_type=ct,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Concept clustering
# ---------------------------------------------------------------------------
class ConceptClusterer:
    """Groups claims into buckets by canonical_concept.

    Phase-1: pure string equality on ``canonical_concept`` (= normalize_keyword
    output). Phase-2 will plug in BGE-Large cosine clustering for buckets that
    would otherwise be singletons.
    """

    def __init__(self, *, cluster_threshold: float = 0.85) -> None:
        # threshold reserved for phase-2 BGE fallback; unused in phase-1.
        self.cluster_threshold = cluster_threshold

    def bucket(self, sample_id: str, claims: list[Claim]) -> list[ConceptBucket]:
        by_canonical: dict[str, ConceptBucket] = {}
        for c in claims:
            key = c.canonical_concept
            if not key:
                continue
            if key not in by_canonical:
                by_canonical[key] = ConceptBucket(
                    sample_id=sample_id,
                    canonical_concept=key,
                    claims=[],
                )
            by_canonical[key].claims.append(c)
        return list(by_canonical.values())
