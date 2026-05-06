"""OntologyProposerAgent: corpus-level extension of the base ontology.

This is stage 2.1 of the ontology refactor. The agent runs ONCE per
pipeline run, between phase 3 (chunks) and phase 4 (LLM triple extraction).
It samples chunks, asks the LLM to propose extensions to the base
ontology, deduplicates against the base via a second LLM call, and grounds
each surviving proposal against external KBs (sciverse / pubmed / mesh)
via EntityCanonicalizer.

Outputs (all written to <output_dir>/ontology_proposer/):
  - ontology.run.yaml          — merged base + accepted extensions
  - ontology.extensions.yaml   — accepted extensions only
  - ontology_decisions.jsonl   — append-only audit log of every decision
  - ontology.rejected.jsonl    — rejected proposals with reasons

Cache:
  Keyed by hash(base_version + sample_corpus_hash). Reusing the same
  base ontology + same chunks sample skips all LLM calls.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .external_kb import EntityCanonicalizer, GroundingResult
from .llm import InternChatClient
from .ontology import Ontology
from .pubmed_client import PubMedClient
from .utils import normalize_keyword, normalize_text, sha256_text

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROPOSER_PROMPT_PATH = PROMPTS_DIR / "ontology_proposer_user.txt"
DEDUP_PROMPT_PATH = PROMPTS_DIR / "ontology_dedup_user.txt"

PROPOSER_SYSTEM = (
    "You are a biomedical knowledge-graph ontology auditor. "
    "Return strict JSON only. Be specific and evidence-driven: only "
    "propose entries that the chunk explicitly supports AND that the "
    "existing ontology cannot already represent. When you do find a "
    "genuine gap, you must propose it."
)
DEDUP_SYSTEM = (
    "You are a biomedical ontology validator. Return strict JSON only. "
    "When in doubt between accept and merge, prefer merge."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class _RawProposal:
    kind: str                # "entity_type" | "relation" | "entity_alias"
    payload: dict            # the per-item dict from the LLM response
    chunk_id: str
    doc_id: str


@dataclass
class _AggregatedProposal:
    kind: str                # entity_type | relation | entity_alias
    key: str                 # canonical key for grouping (id or surface form)
    payload: dict            # representative payload
    evidences: list[dict] = field(default_factory=list)  # {chunk_id, doc_id, quote}
    docs: set[str] = field(default_factory=set)
    grounding: GroundingResult | None = None
    decision: str = "pending"  # accept|merge|reject|pending
    merge_target: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# OntologyProposerAgent
# ---------------------------------------------------------------------------

class OntologyProposerAgent:
    def __init__(
        self,
        config: dict[str, Any],
        llm_client: InternChatClient,
        canonicalizer: EntityCanonicalizer | None = None,
    ):
        cfg = dict(config or {})
        self.sample_size = max(int(cfg.get("sample_size", 30)), 1)
        self.evidence_threshold = max(int(cfg.get("evidence_threshold", 2)), 1)
        self.distinct_doc_threshold = max(int(cfg.get("distinct_doc_threshold", 2)), 1)
        self.cache_dir = Path(cfg.get("cache_dir") or "/tmp/pubmed_graph_proposer_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_proposals_per_kind = int(cfg.get("max_proposals_per_kind", 8))
        self.require_grounding = bool(cfg.get("require_grounding", True))
        self.skip_when_disabled = bool(cfg.get("skip_when_disabled", True))
        self.random_seed = int(cfg.get("random_seed", 42))
        self.debug_dump_raw = bool(cfg.get("debug_dump_raw", False))
        self.client = llm_client
        self.canonicalizer = canonicalizer

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------

    def propose(
        self,
        chunks: list[dict],
        base: Ontology,
        output_dir: Path,
    ) -> Ontology:
        out = Path(output_dir) / "ontology_proposer"
        out.mkdir(parents=True, exist_ok=True)

        sample = self._sample_chunks(chunks)
        cache_key = self._cache_key(base, sample)
        run_yaml_path = out / "ontology.run.yaml"
        cache_path = self.cache_dir / f"run_{cache_key}.yaml"
        summary_path = out / "summary.json"

        # cache hit
        if cache_path.exists():
            run_yaml_path.write_text(cache_path.read_text(encoding="utf-8"), encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "cache_hit",
                        "cache_path": str(cache_path),
                        "cache_key": cache_key,
                        "sample_chunks": len(sample),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return Ontology.load(run_yaml_path)

        decisions_path = out / "ontology_decisions.jsonl"
        rejected_path = out / "ontology.rejected.jsonl"
        decisions_handle = decisions_path.open("a", encoding="utf-8")
        rejected_handle = rejected_path.open("a", encoding="utf-8")

        # mutable counters for the always-written summary
        self._stats: dict[str, Any] = {
            "status": "ok",
            "cache_key": cache_key,
            "cache_hit": False,
            "sample_chunks": len(sample),
            "llm_calls_attempted": 0,
            "llm_calls_succeeded": 0,
            "llm_calls_failed": 0,
            "raw_proposals_total": 0,
            "raw_proposals_entity_type": 0,
            "raw_proposals_relation": 0,
            "raw_proposals_alias": 0,
            "aggregated_proposals": 0,
            "rejected_evidence_threshold": 0,
            "rejected_dedup": 0,
            "rejected_grounding": 0,
            "accepted": 0,
            "first_failure": "",
        }

        try:
            raw_proposals = self._collect_raw_proposals(sample, base, decisions_handle)
            aggregated = self._aggregate(raw_proposals)
            self._apply_evidence_threshold(aggregated, rejected_handle)
            survived = [p for p in aggregated if p.decision == "pending"]
            self._dedup_against_base(survived, base, rejected_handle)
            survived = [p for p in aggregated if p.decision == "accept"]
            if self.canonicalizer is not None and self.require_grounding:
                self._ground_with_canonicalizer(survived, rejected_handle)
            survived = [p for p in aggregated if p.decision == "accept"]

            extensions_yaml = self._materialize_extensions(survived)
            (out / "ontology.extensions.yaml").write_text(
                yaml.safe_dump(extensions_yaml, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            merged_yaml = self._merge_yaml(base, extensions_yaml)
            run_yaml_path.write_text(
                yaml.safe_dump(merged_yaml, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            cache_path.write_text(run_yaml_path.read_text(encoding="utf-8"), encoding="utf-8")

            for prop in aggregated:
                decisions_handle.write(
                    json.dumps(
                        {
                            "kind": prop.kind,
                            "key": prop.key,
                            "decision": prop.decision,
                            "merge_target": prop.merge_target,
                            "reason": prop.reason,
                            "evidence_count": len(prop.evidences),
                            "distinct_docs": sorted(prop.docs),
                            "grounding": prop.grounding.to_dict() if prop.grounding else None,
                            "payload": prop.payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            self._stats["accepted"] = sum(1 for p in aggregated if p.decision == "accept")
            self._stats["aggregated_proposals"] = len(aggregated)
            self._stats["rejected_evidence_threshold"] = sum(
                1 for p in aggregated if p.decision == "reject" and p.reason.startswith("evidence_threshold")
            )
            self._stats["rejected_grounding"] = sum(
                1
                for p in aggregated
                if p.decision == "reject" and (p.reason.startswith("ungrounded") or p.reason.startswith("canonicalizer_error"))
            )
            self._stats["rejected_dedup"] = sum(
                1
                for p in aggregated
                if p.decision in {"merge", "reject"}
                and not p.reason.startswith("evidence_threshold")
                and not p.reason.startswith("ungrounded")
                and not p.reason.startswith("canonicalizer_error")
            )
        except Exception as exc:
            self._stats["status"] = "error"
            self._stats["first_failure"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            decisions_handle.close()
            rejected_handle.close()
            try:
                summary_path.write_text(
                    json.dumps(self._stats, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return Ontology.load(run_yaml_path)

    # -------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------

    def _sample_chunks(self, chunks: list[dict]) -> list[dict]:
        if len(chunks) <= self.sample_size:
            return list(chunks)
        rng = random.Random(self.random_seed)
        # stratified by doc: round-robin one chunk from each doc until budget filled
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            by_doc[str(c.get("doc_id", "?"))].append(c)
        for bucket in by_doc.values():
            rng.shuffle(bucket)
        doc_order = list(by_doc.keys())
        rng.shuffle(doc_order)
        out: list[dict] = []
        while len(out) < self.sample_size and any(by_doc.values()):
            for doc in doc_order:
                if len(out) >= self.sample_size:
                    break
                if by_doc[doc]:
                    out.append(by_doc[doc].pop())
        return out

    @staticmethod
    def _cache_key(base: Ontology, sample: list[dict]) -> str:
        sig = base.version + "|" + str(len(sample)) + "|"
        sig += sha256_text(
            "|".join(f"{c.get('doc_id','')}::{c.get('chunk_id','')}" for c in sample)
        )
        return hashlib.sha256(sig.encode()).hexdigest()[:24]

    # -------------------------------------------------------------------
    # LLM proposal collection
    # -------------------------------------------------------------------

    def _collect_raw_proposals(
        self,
        sample: list[dict],
        base: Ontology,
        decisions_handle,
    ) -> list[_RawProposal]:
        prompt_template = PROPOSER_PROMPT_PATH.read_text(encoding="utf-8")
        entity_types_block = base.render_prompt_section_entity_types()
        relations_block = base.render_prompt_section_relations()
        relation_aliases_block = base.render_prompt_mapping_rules() or "(none)"
        entity_aliases_block = "\n".join(
            f"- {entry['surface']} -> {entry['canonical']} ({entry.get('type') or 'any'})"
            for entry in base._data.get("aliases") or []
        ) or "(none)"

        proposals: list[_RawProposal] = []
        for chunk in sample:
            self._stats["llm_calls_attempted"] += 1
            chunk_block = (
                f"doc_id: {chunk.get('doc_id','')}\n"
                f"chunk_id: {chunk.get('chunk_id','')}\n"
                f"title: {chunk.get('title','')}\n"
                f"section: {chunk.get('section','')}\n"
                f"text:\n{chunk.get('text','')}"
            )
            prompt = prompt_template.format(
                entity_types_block=entity_types_block,
                relations_block=relations_block,
                relation_aliases_block=relation_aliases_block,
                entity_aliases_block=entity_aliases_block,
                chunk_block=chunk_block,
            )
            raw_text = ""
            try:
                if self.debug_dump_raw:
                    raw_text = self.client.chat(
                        [
                            {"role": "system", "content": PROPOSER_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                        max_tokens=1500,
                    )
                    json_text = self.client._extract_json_text(raw_text) if raw_text else ""
                    payload = json.loads(json_text) if json_text else {}
                else:
                    payload = self.client.chat_json(
                        [
                            {"role": "system", "content": PROPOSER_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                        max_tokens=1500,
                    )
                self._stats["llm_calls_succeeded"] += 1
            except Exception as exc:
                self._stats["llm_calls_failed"] += 1
                if not self._stats["first_failure"]:
                    self._stats["first_failure"] = f"{type(exc).__name__}: {exc}"
                decisions_handle.write(
                    json.dumps(
                        {
                            "kind": "proposer_error",
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_id": chunk.get("doc_id"),
                            "error": str(exc),
                        }
                    )
                    + "\n"
                )
                continue
            if not isinstance(payload, dict):
                decisions_handle.write(
                    json.dumps(
                        {
                            "kind": "proposer_non_dict_response",
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_id": chunk.get("doc_id"),
                            "payload_type": type(payload).__name__,
                            "payload_repr": repr(payload)[:300],
                        }
                    )
                    + "\n"
                )
                continue
            n_types = len(payload.get("new_entity_types") or [])
            n_rels = len(payload.get("new_relations") or [])
            n_aliases = len(payload.get("new_entity_aliases") or [])
            call_record: dict[str, Any] = {
                "kind": "proposer_call_ok",
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "section": chunk.get("section", ""),
                "title": (chunk.get("title", "") or "")[:80],
                "n_new_entity_types": n_types,
                "n_new_relations": n_rels,
                "n_new_entity_aliases": n_aliases,
            }
            if self.debug_dump_raw and raw_text:
                call_record["raw_response"] = raw_text[:2000]
            decisions_handle.write(json.dumps(call_record, ensure_ascii=False) + "\n")
            for item in payload.get("new_entity_types") or []:
                if isinstance(item, dict) and item.get("id"):
                    self._stats["raw_proposals_total"] += 1
                    self._stats["raw_proposals_entity_type"] += 1
                    proposals.append(
                        _RawProposal(
                            kind="entity_type",
                            payload=item,
                            chunk_id=str(chunk.get("chunk_id", "")),
                            doc_id=str(chunk.get("doc_id", "")),
                        )
                    )
            for item in payload.get("new_relations") or []:
                if isinstance(item, dict) and item.get("id"):
                    self._stats["raw_proposals_total"] += 1
                    self._stats["raw_proposals_relation"] += 1
                    proposals.append(
                        _RawProposal(
                            kind="relation",
                            payload=item,
                            chunk_id=str(chunk.get("chunk_id", "")),
                            doc_id=str(chunk.get("doc_id", "")),
                        )
                    )
            for item in payload.get("new_entity_aliases") or []:
                if isinstance(item, dict) and item.get("surface"):
                    self._stats["raw_proposals_total"] += 1
                    self._stats["raw_proposals_alias"] += 1
                    proposals.append(
                        _RawProposal(
                            kind="entity_alias",
                            payload=item,
                            chunk_id=str(chunk.get("chunk_id", "")),
                            doc_id=str(chunk.get("doc_id", "")),
                        )
                    )
        return proposals

    # -------------------------------------------------------------------
    # Aggregation + thresholds
    # -------------------------------------------------------------------

    @staticmethod
    def _proposal_key(prop: _RawProposal) -> str:
        if prop.kind == "entity_type":
            return f"type::{normalize_keyword(prop.payload.get('id',''))}"
        if prop.kind == "relation":
            return f"rel::{normalize_keyword(prop.payload.get('id',''))}"
        return f"alias::{normalize_keyword(prop.payload.get('surface',''))}"

    def _aggregate(self, raw: list[_RawProposal]) -> list[_AggregatedProposal]:
        groups: dict[str, _AggregatedProposal] = {}
        for r in raw:
            key = self._proposal_key(r)
            if key not in groups:
                groups[key] = _AggregatedProposal(
                    kind=r.kind,
                    key=key,
                    payload=dict(r.payload),
                )
            agg = groups[key]
            agg.evidences.append(
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "quote": str(r.payload.get("evidence", "")),
                }
            )
            agg.docs.add(r.doc_id)
        return list(groups.values())

    def _apply_evidence_threshold(
        self,
        aggregated: list[_AggregatedProposal],
        rejected_handle,
    ) -> None:
        for prop in aggregated:
            if len(prop.evidences) < self.evidence_threshold or len(prop.docs) < self.distinct_doc_threshold:
                prop.decision = "reject"
                prop.reason = (
                    f"evidence_threshold: hits={len(prop.evidences)} docs={len(prop.docs)} "
                    f"required hits>={self.evidence_threshold} docs>={self.distinct_doc_threshold}"
                )
                rejected_handle.write(
                    json.dumps(
                        {
                            "kind": prop.kind,
                            "key": prop.key,
                            "reason": prop.reason,
                            "evidences": prop.evidences[:3],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # -------------------------------------------------------------------
    # Second-pass dedup against base
    # -------------------------------------------------------------------

    def _dedup_against_base(
        self,
        candidates: list[_AggregatedProposal],
        base: Ontology,
        rejected_handle,
    ) -> None:
        if not candidates:
            return
        prompt_template = DEDUP_PROMPT_PATH.read_text(encoding="utf-8")
        entity_types_block = base.render_prompt_section_entity_types()
        relations_block = base.render_prompt_section_relations()
        entity_aliases_block = "\n".join(
            f"- {entry['surface']} -> {entry['canonical']}"
            for entry in base._data.get("aliases") or []
        ) or "(none)"

        candidates_payload = {
            "entity_types": [
                {"id": p.payload.get("id"), "rationale": p.payload.get("rationale", ""), "evidence": p.evidences[:2]}
                for p in candidates if p.kind == "entity_type"
            ][: self.max_proposals_per_kind],
            "relations": [
                {"id": p.payload.get("id"), "rationale": p.payload.get("rationale", ""), "evidence": p.evidences[:2]}
                for p in candidates if p.kind == "relation"
            ][: self.max_proposals_per_kind],
            "entity_aliases": [
                {
                    "surface": p.payload.get("surface"),
                    "canonical": p.payload.get("canonical", ""),
                    "type": p.payload.get("type", ""),
                    "evidence": p.evidences[:2],
                }
                for p in candidates if p.kind == "entity_alias"
            ][: self.max_proposals_per_kind],
        }

        prompt = prompt_template.format(
            entity_types_block=entity_types_block,
            relations_block=relations_block,
            entity_aliases_block=entity_aliases_block,
            candidates_json=json.dumps(candidates_payload, ensure_ascii=False, indent=2),
        )

        try:
            payload = self.client.chat_json(
                [
                    {"role": "system", "content": DEDUP_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1500,
            )
        except Exception as exc:
            for prop in candidates:
                prop.decision = "reject"
                prop.reason = f"dedup llm error: {exc}"
                rejected_handle.write(json.dumps({"kind": prop.kind, "key": prop.key, "reason": prop.reason}) + "\n")
            return

        if not isinstance(payload, dict):
            return

        verdict_index: dict[str, dict] = {}
        for entry in payload.get("entity_types") or []:
            if isinstance(entry, dict) and entry.get("id"):
                verdict_index[f"type::{normalize_keyword(entry['id'])}"] = entry
        for entry in payload.get("relations") or []:
            if isinstance(entry, dict) and entry.get("id"):
                verdict_index[f"rel::{normalize_keyword(entry['id'])}"] = entry
        for entry in payload.get("entity_aliases") or []:
            if isinstance(entry, dict) and entry.get("surface"):
                verdict_index[f"alias::{normalize_keyword(entry['surface'])}"] = entry

        for prop in candidates:
            verdict = verdict_index.get(prop.key)
            if not verdict:
                prop.decision = "reject"
                prop.reason = "no verdict from dedup pass"
                rejected_handle.write(json.dumps({"kind": prop.kind, "key": prop.key, "reason": prop.reason}) + "\n")
                continue
            decision = str(verdict.get("decision") or "reject").lower()
            prop.decision = decision if decision in {"accept", "merge", "reject"} else "reject"
            prop.merge_target = str(verdict.get("merge_target") or "")
            prop.reason = str(verdict.get("reason") or "")
            if prop.decision != "accept":
                rejected_handle.write(
                    json.dumps(
                        {
                            "kind": prop.kind,
                            "key": prop.key,
                            "decision": prop.decision,
                            "merge_target": prop.merge_target,
                            "reason": prop.reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # -------------------------------------------------------------------
    # External KB grounding
    # -------------------------------------------------------------------

    def _ground_with_canonicalizer(
        self,
        candidates: list[_AggregatedProposal],
        rejected_handle,
    ) -> None:
        if self.canonicalizer is None:
            return
        for prop in candidates:
            queries = self._grounding_queries(prop)
            grounding = None
            for q in queries:
                try:
                    grounding = self.canonicalizer.resolve(q, hint_type=prop.payload.get("type", ""))
                except Exception as exc:
                    prop.decision = "reject"
                    prop.reason = f"canonicalizer_error: {exc}"
                    rejected_handle.write(
                        json.dumps({"kind": prop.kind, "key": prop.key, "reason": prop.reason})
                        + "\n"
                    )
                    grounding = None
                    break
                if grounding.grounded:
                    break
            if grounding is None:
                continue
            prop.grounding = grounding
            if not grounding.grounded:
                prop.decision = "reject"
                prop.reason = "ungrounded: " + (grounding.rejected_reason or "no kb hits")
                rejected_handle.write(
                    json.dumps(
                        {
                            "kind": prop.kind,
                            "key": prop.key,
                            "reason": prop.reason,
                            "tried_queries": queries,
                            "grounding": grounding.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @staticmethod
    def _grounding_queries(prop: _AggregatedProposal) -> list[str]:
        """Generate up to 3 query strings to try against external KBs.

        For entity types we try the type id itself + a phrase pulled from the
        first evidence quote. For aliases we try canonical name + surface form.
        For relations we try the relation id stripped of underscores.
        """
        queries: list[str] = []
        payload = prop.payload or {}
        evidences = prop.evidences or []
        first_quote = (evidences[0].get("quote") if evidences else "") or ""

        if prop.kind == "entity_type":
            tid = str(payload.get("id") or "").strip()
            if tid:
                queries.append(tid)
            # try a meaningful phrase from the evidence quote (first ~10 words)
            short = " ".join(first_quote.split()[:12])
            if short:
                queries.append(short)
        elif prop.kind == "entity_alias":
            canonical = str(payload.get("canonical") or "").strip()
            surface = str(payload.get("surface") or "").strip()
            if canonical:
                queries.append(canonical)
            if surface and surface.lower() != canonical.lower():
                queries.append(surface)
        else:  # relation
            rid = str(payload.get("id") or "").replace("_", " ").strip()
            if rid:
                queries.append(rid)
            sf = str(payload.get("surface_form") or "").strip()
            if sf and sf.lower() != rid.lower():
                queries.append(sf)
        return [q[:200] for q in queries if q]

    # -------------------------------------------------------------------
    # YAML materialization
    # -------------------------------------------------------------------

    @staticmethod
    def _materialize_extensions(accepted: list[_AggregatedProposal]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "extensions": {
                "version": "1",
                "entity_types": [],
                "core_relations": [],
                "aliases": [],
            }
        }
        for prop in accepted:
            if prop.kind == "entity_type":
                out["extensions"]["entity_types"].append(
                    {
                        "id": prop.payload.get("id"),
                        "description": prop.payload.get("rationale", ""),
                        "examples": [],
                        "surface_aliases": [],
                        "_provenance": {
                            "evidence_count": len(prop.evidences),
                            "distinct_docs": sorted(prop.docs),
                        },
                    }
                )
            elif prop.kind == "relation":
                out["extensions"]["core_relations"].append(
                    {
                        "id": prop.payload.get("id"),
                        "description": prop.payload.get("rationale", ""),
                        "directionality": "directed",
                        "head_types": ["*"],
                        "tail_types": ["*"],
                        "surface_aliases": [prop.payload.get("surface_form")] if prop.payload.get("surface_form") else [],
                        "_provenance": {
                            "evidence_count": len(prop.evidences),
                            "distinct_docs": sorted(prop.docs),
                        },
                    }
                )
            elif prop.kind == "entity_alias":
                out["extensions"]["aliases"].append(
                    {
                        "surface": prop.payload.get("surface"),
                        "canonical": prop.payload.get("canonical"),
                        "type": prop.payload.get("type"),
                        "_provenance": {
                            "evidence_count": len(prop.evidences),
                            "distinct_docs": sorted(prop.docs),
                        },
                    }
                )
        return out

    @staticmethod
    def _merge_yaml(base: Ontology, extensions: dict[str, Any]) -> dict[str, Any]:
        merged = json.loads(json.dumps(base._data))  # deep copy via json round-trip
        ext = (extensions or {}).get("extensions") or {}
        if ext.get("entity_types"):
            merged.setdefault("entity_types", []).extend(ext["entity_types"])
        if ext.get("core_relations"):
            merged.setdefault("core_relations", []).extend(ext["core_relations"])
        if ext.get("aliases"):
            merged.setdefault("aliases", []).extend(ext["aliases"])
        merged["version"] = f"{merged.get('version', '0.0.0')}+ext"
        merged["extensions_metadata"] = {
            "added_entity_types": [e.get("id") for e in ext.get("entity_types") or []],
            "added_relations": [r.get("id") for r in ext.get("core_relations") or []],
            "added_aliases": [a.get("surface") for a in ext.get("aliases") or []],
        }
        return merged
