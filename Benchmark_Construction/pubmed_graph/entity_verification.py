from __future__ import annotations

import json
from typing import Any

import requests

from .llm import InternChatClient
from .pubmed_client import PubMedClient
from .utils import normalize_keyword, normalize_text

ENTITY_VERIFICATION_SYSTEM_PROMPT = """You verify biomedical concept typing using external facts.

Return strict JSON only with keys:
- canonical_name
- entity_type
- confidence
- keep

Rules:
- rely on external facts first, then use the local evidence snippet
- use concise canonical biomedical names
- expand abbreviations only when supported by the evidence or external facts
- if the concept is a drug code or compound identifier, do not classify it as CellLine
- if the concept is a phospho-protein form such as P-P65 or phospho-p65, classify it as Protein
- if the concept is a process fragment like "physical association between A and B", convert it to a compact interaction/process name
- if the concept is too ambiguous and cannot be grounded by external facts, set keep=false

Allowed entity_type values are loaded dynamically from the active ontology
(see pubmed_graph/ontology.yaml). When in doubt about typing, use the most
specific type that the external evidence supports.
"""

# Stage 3.2: removed LOCAL_ABBREVIATION_OVERRIDES, LOCAL_CONTEXTUAL_OVERRIDES,
# and _apply_post_verification_rules. The dd / id / oa abbreviations are
# already in ontology.yaml `aliases:`. Per-paper hard rules (NSE, SMCs,
# m6A, Cullin-3, IGF2BP1-3) are now resolved at runtime via Ontology +
# the sciverse/pubmed/mesh-backed EntityCanonicalizer.


class ExternalFactEntityVerifier:
    def __init__(
        self,
        config: dict[str, Any],
        pubmed_client: PubMedClient,
        default_llm_config: dict[str, Any] | None = None,
    ):
        self.enabled = bool(config.get("enabled", False))
        self.pubmed_client = pubmed_client
        self.pubmed_retmax = max(int(config.get("pubmed_retmax", 2)), 1)
        self.timeout = float(config.get("timeout", 20))
        self.mesh_enabled = bool(config.get("mesh_enabled", True))
        self.mesh_base_url = config.get("mesh_base_url", "https://id.nlm.nih.gov/mesh")
        self.mesh_limit = max(int(config.get("mesh_limit", 2)), 1)
        self.session = requests.Session()
        llm_cfg = dict(default_llm_config or {})
        llm_cfg.update(config.get("llm", {}))
        llm_cfg.setdefault("model", "intern-s1-pro")
        llm_cfg.setdefault("thinking_mode", False)
        llm_cfg.setdefault("temperature", 0.0)
        llm_cfg.setdefault("max_tokens", 600)
        self.client = InternChatClient(llm_cfg) if self.enabled else None
        self.cache: dict[str, dict[str, Any]] = {}

    def verify(self, entity_text: str, entity_type: str = "", evidence: str = "") -> dict[str, Any]:
        cleaned = normalize_text(entity_text).strip(" \t\n\r\"'`.,;:()[]{}")
        if not cleaned:
            return {"canonical_name": "", "entity_type": entity_type, "confidence": 0.0, "keep": False}
        cache_key = normalize_keyword(cleaned)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if not self.enabled or self.client is None:
            result = {
                "canonical_name": cleaned,
                "entity_type": entity_type,
                "confidence": 0.0,
                "keep": True,
            }
            self.cache[cache_key] = result
            return result

        # Stage 3.2: short-circuit on aliases that are already in the active
        # ontology. This replaces the old NSE/SMC/m6a/Cullin-3/IGF2BP1-3
        # hard-coded branches.
        from .ontology import Ontology

        ontology = Ontology.default()
        if cache_key in ontology.static_alias_map:
            canonical = ontology.static_alias_map[cache_key]
            type_hint = ontology._alias_type_map.get(cache_key) or entity_type
            result = {
                "canonical_name": canonical,
                "entity_type": type_hint,
                "confidence": 0.99,
                "keep": True,
            }
            self.cache[cache_key] = result
            return result

        facts = self._gather_facts(cleaned)
        prompt_user = {
            "concept": cleaned,
            "local_entity_type": entity_type,
            "local_evidence": normalize_text(evidence)[:1000],
            "external_facts": facts,
        }
        try:
            payload = self.client.chat_json(
                [
                    {"role": "system", "content": ENTITY_VERIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt_user, ensure_ascii=False)},
                ],
            )
        except Exception:
            payload = {}
        result = {
            "canonical_name": normalize_text(payload.get("canonical_name", cleaned)) if isinstance(payload, dict) else cleaned,
            "entity_type": normalize_text(payload.get("entity_type", entity_type)) if isinstance(payload, dict) else entity_type,
            "confidence": self._coerce_confidence(payload.get("confidence", 0.0)) if isinstance(payload, dict) else 0.0,
            "keep": bool(payload.get("keep", True)) if isinstance(payload, dict) else True,
        }
        if not result["canonical_name"]:
            result["canonical_name"] = cleaned
        if not result["entity_type"]:
            result["entity_type"] = entity_type
        self.cache[cache_key] = result
        return result

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            normalized = normalize_keyword(value)
            if normalized in {"high", "strong"}:
                return 0.9
            if normalized in {"medium", "moderate"}:
                return 0.7
            if normalized in {"low", "weak"}:
                return 0.4
            return 0.0

    def _gather_facts(self, concept: str) -> dict[str, Any]:
        return {
            "pubmed": self._lookup_pubmed(concept),
            "mesh": self._lookup_mesh(concept) if self.mesh_enabled else [],
        }

    def _lookup_pubmed(self, concept: str) -> list[dict[str, Any]]:
        query = f"(\"{concept}\"[Title/Abstract] OR \"{concept}\"[MeSH Terms])"
        try:
            papers = self.pubmed_client.fetch_papers(query=query, retmax=self.pubmed_retmax)
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for paper in papers[: self.pubmed_retmax]:
            rows.append(
                {
                    "title": normalize_text(paper.title),
                    "journal": normalize_text(paper.journal),
                    "mesh_terms": paper.mesh_terms[:8],
                    "abstract_snippet": normalize_text(paper.abstract)[:400],
                }
            )
        return rows

    def _lookup_mesh(self, concept: str) -> list[str]:
        labels: list[str] = []
        for endpoint in ["lookup/descriptor", "lookup/term"]:
            try:
                response = self.session.get(
                    f"{self.mesh_base_url}/{endpoint}",
                    params={"label": concept, "match": "contains", "limit": str(self.mesh_limit)},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                for item in response.json():
                    label = normalize_text(item.get("label", ""))
                    if label and label not in labels:
                        labels.append(label)
            except Exception:
                continue
            if labels:
                break
        return labels[: self.mesh_limit]
