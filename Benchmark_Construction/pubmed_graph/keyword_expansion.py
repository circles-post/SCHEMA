from __future__ import annotations

import re
from typing import Any

import requests

from .llm import InternChatClient
from .models import KeywordRecord
from .utils import normalize_keyword

GENERIC_EXPANSION_BLOCKLIST = {
    "protein",
    "gene",
    "domain",
    "complex",
    "enzyme",
    "mutation",
    "variant",
    "interaction",
    "binding",
    "mechanism",
    "function",
    "structure",
    "pathway",
    "assay",
    "cell",
    "cells",
}
TOPIC_EXPANSION_HINTS = {
    "cancer",
    "tumor",
    "tumour",
    "disease",
    "syndrome",
    "infection",
    "immunity",
    "inflammation",
    "resistance",
    "pathway",
    "metabolism",
}


def _supports_generic_suffix_expansion(term: str) -> bool:
    normalized = normalize_keyword(term)
    tokens = [token for token in re.findall(r"[A-Za-z0-9+-]+", normalized) if token]
    if not normalized or len(tokens) > 5:
        return False
    if any(ch.isdigit() for ch in term) or any(sym in term for sym in "/:-+"):
        return False
    if tokens and tokens[-1] in GENERIC_EXPANSION_BLOCKLIST:
        return False
    return any(token in TOPIC_EXPANSION_HINTS for token in tokens)


def _supports_acronym_expansion(term: str) -> bool:
    normalized = normalize_keyword(term)
    tokens = [token for token in re.findall(r"[A-Za-z0-9+-]+", term) if token]
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    if any(token.lower() in GENERIC_EXPANSION_BLOCKLIST for token in tokens):
        return False
    if any(any(ch.isdigit() for ch in token) for token in tokens):
        return False
    if any(any(sym in token for sym in "/:-+") for token in tokens):
        return False
    return normalized.replace(" ", "") != normalized


class LocalKeywordExpander:
    def __init__(self, config: dict[str, Any]):
        self.manual_synonyms = {
            normalize_keyword(k): list(v)
            for k, v in config.get("manual_synonyms", {}).items()
        }
        self.enable_acronyms = bool(config.get("enable_acronyms", True))
        self.default_suffixes = config.get(
            "heuristic_suffixes",
            ["diagnosis", "treatment", "therapy", "biomarker", "screening", "prognosis"],
        )
        self.generic_replacements = config.get(
            "generic_replacements",
            {"cancer": ["neoplasm", "carcinoma", "oncology"], "tumor": ["tumour", "neoplasm"]},
        )

    def expand(self, term: str) -> list[KeywordRecord]:
        normalized = normalize_keyword(term)
        candidates: list[KeywordRecord] = []
        for synonym in self.manual_synonyms.get(normalized, []):
            candidates.append(
                KeywordRecord(
                    term=synonym,
                    normalized_term=normalize_keyword(synonym),
                    source="local_manual_synonym",
                    parent_term=term,
                    notes="manual_synonyms",
                )
            )
        for token, replacements in self.generic_replacements.items():
            if token in normalized:
                for replacement in replacements:
                    variant = re.sub(token, replacement, term, flags=re.IGNORECASE)
                    candidates.append(
                        KeywordRecord(
                            term=variant,
                            normalized_term=normalize_keyword(variant),
                            source="local_generic_replacement",
                            parent_term=term,
                            notes=f"{token}->{replacement}",
                        )
                    )
        if _supports_generic_suffix_expansion(term):
            for suffix in self.default_suffixes:
                variant = f"{term} {suffix}"
                candidates.append(
                    KeywordRecord(
                        term=variant,
                        normalized_term=normalize_keyword(variant),
                        source="local_heuristic_suffix",
                        parent_term=term,
                        notes="heuristic_suffix",
                    )
                )
        acronym = "".join(part[0].upper() for part in re.findall(r"[A-Za-z]+", term) if part)
        if self.enable_acronyms and len(acronym) >= 3 and acronym != term and _supports_acronym_expansion(term):
            candidates.append(
                KeywordRecord(
                    term=acronym,
                    normalized_term=normalize_keyword(acronym),
                    source="local_acronym",
                    parent_term=term,
                    notes="heuristic_acronym",
                )
            )
        return candidates


class MeshKeywordExpander:
    def __init__(self, config: dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.base_url = config.get("base_url", "https://id.nlm.nih.gov/mesh")
        self.limit = max(int(config.get("limit", 5)), 1)
        self.term_match_modes = config.get("term_match_modes", ["exact"])
        self.match_modes = config.get("match_modes", ["exact", "contains"])
        self.include_terms = bool(config.get("include_terms", True))
        self.include_seealso = bool(config.get("include_seealso", True))
        self.include_qualifiers = bool(config.get("include_qualifiers", False))
        self.timeout = float(config.get("timeout", 20))
        self.session = requests.Session()

    def expand(self, term: str) -> list[KeywordRecord]:
        if not self.enabled:
            return []
        try:
            descriptors = self._search_descriptors_from_terms(term)
            if not descriptors:
                descriptors = self._search_descriptors(term)
        except Exception as exc:
            return [
                KeywordRecord(
                    term=term,
                    normalized_term=normalize_keyword(term),
                    source="mesh_error",
                    parent_term=term,
                    accepted=False,
                    notes=f"mesh_lookup_failed={exc}",
                )
            ]

        records: list[KeywordRecord] = []
        seen: set[str] = set()
        for descriptor in descriptors:
            descriptor_id = self._descriptor_id(descriptor.get("resource", ""))
            descriptor_label = (descriptor.get("label") or "").strip()
            if descriptor_label:
                self._append_if_new(
                    records,
                    seen,
                    KeywordRecord(
                        term=descriptor_label,
                        normalized_term=normalize_keyword(descriptor_label),
                        source="mesh_descriptor",
                        parent_term=term,
                        notes=f"descriptor_id={descriptor_id}",
                    ),
                )
            try:
                details = self._lookup_details(descriptor_id)
            except Exception as exc:
                self._append_if_new(
                    records,
                    seen,
                    KeywordRecord(
                        term=descriptor_label or term,
                        normalized_term=normalize_keyword(descriptor_label or term),
                        source="mesh_details_error",
                        parent_term=term,
                        accepted=False,
                        notes=f"descriptor_id={descriptor_id}; details_failed={exc}",
                    ),
                )
                details = {}
            for item in details.get("terms", []) if isinstance(details, dict) else []:
                label = (item.get("label") or "").strip()
                preferred = bool(item.get("preferred"))
                if label:
                    self._append_if_new(
                        records,
                        seen,
                        KeywordRecord(
                            term=label,
                            normalized_term=normalize_keyword(label),
                            source="mesh_preferred_term" if preferred else "mesh_term",
                            parent_term=term,
                            notes=f"descriptor_id={descriptor_id}; preferred={preferred}",
                        ),
                    )
            if self.include_seealso:
                for item in details.get("seealso", []) if isinstance(details, dict) else []:
                    label = (item.get("label") or "").strip()
                    if label:
                        self._append_if_new(
                            records,
                            seen,
                            KeywordRecord(
                                term=label,
                                normalized_term=normalize_keyword(label),
                                source="mesh_seealso",
                                parent_term=term,
                                notes=f"descriptor_id={descriptor_id}",
                            ),
                        )
            if self.include_qualifiers:
                for item in details.get("qualifiers", []) if isinstance(details, dict) else []:
                    label = (item.get("label") or "").strip()
                    if label and len(label.split()) <= 4:
                        combined = f"{descriptor_label} {label}".strip()
                        self._append_if_new(
                            records,
                            seen,
                            KeywordRecord(
                                term=combined,
                                normalized_term=normalize_keyword(combined),
                                source="mesh_qualifier_combo",
                                parent_term=term,
                                notes=f"descriptor_id={descriptor_id}; qualifier={label}",
                            ),
                        )
        return records

    def _search_descriptors(self, term: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_resources: set[str] = set()
        for match_mode in self.match_modes:
            response = self.session.get(
                f"{self.base_url}/lookup/descriptor",
                params={"label": term, "match": match_mode, "limit": str(self.limit)},
                timeout=self.timeout,
            )
            response.raise_for_status()
            for item in response.json():
                resource = item.get("resource", "")
                if resource and resource not in seen_resources:
                    seen_resources.add(resource)
                    results.append(item)
        return results

    def _search_terms(self, term: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_resources: set[str] = set()
        for match_mode in self.term_match_modes:
            response = self.session.get(
                f"{self.base_url}/lookup/term",
                params={"label": term, "match": match_mode, "limit": str(self.limit)},
                timeout=self.timeout,
            )
            response.raise_for_status()
            for item in response.json():
                resource = item.get("resource", "")
                if resource and resource not in seen_resources:
                    seen_resources.add(resource)
                    results.append(item)
        return results

    def _search_descriptors_from_terms(self, term: str) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        seen_resources: set[str] = set()
        for term_item in self._search_terms(term):
            term_resource = term_item.get("resource", "")
            if not term_resource:
                continue
            for descriptor in self._map_term_resource_to_descriptors(term_resource):
                resource = descriptor.get("resource", "")
                if resource and resource not in seen_resources:
                    seen_resources.add(resource)
                    descriptors.append(descriptor)
        return descriptors

    def _lookup_details(self, descriptor: str) -> dict[str, Any]:
        includes = []
        if self.include_terms:
            includes.append("terms")
        if self.include_seealso:
            includes.append("seealso")
        if self.include_qualifiers:
            includes.append("qualifiers")
        params = {"descriptor": descriptor}
        if includes:
            params["includes"] = ",".join(includes)
        response = self.session.get(
            f"{self.base_url}/lookup/details",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _map_term_resource_to_descriptors(self, term_resource: str) -> list[dict[str, Any]]:
        sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
SELECT DISTINCT ?d ?dName WHERE {{
  ?c meshv:preferredTerm <{term_resource}> .
  ?d meshv:concept ?c .
  ?d rdfs:label ?dName .
}}
LIMIT {self.limit}
""".strip()
        response = self.session.get(
            f"{self.base_url}/sparql",
            params={"query": sparql, "format": "JSON"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        bindings = response.json().get("results", {}).get("bindings", [])
        descriptors: list[dict[str, Any]] = []
        for item in bindings:
            resource = item.get("d", {}).get("value", "")
            label = item.get("dName", {}).get("value", "")
            if resource and label:
                descriptors.append({"resource": resource, "label": label})
        return descriptors

    @staticmethod
    def _descriptor_id(resource: str) -> str:
        return resource.rstrip("/").split("/")[-1]

    @staticmethod
    def _append_if_new(records: list[KeywordRecord], seen: set[str], record: KeywordRecord) -> None:
        if record.normalized_term in seen:
            return
        seen.add(record.normalized_term)
        records.append(record)


class OpenAIKeywordExpander:
    def __init__(self, config: dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self.model = config.get("model", "intern-latest")
        self.max_terms = max(int(config.get("max_terms_per_seed", 8)), 1)
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 400))
        self.client = None
        if self.enabled:
            try:
                self.client = InternChatClient(config)
            except Exception:
                self.client = None

    def expand(self, term: str) -> list[KeywordRecord]:
        if not self.enabled or self.client is None:
            return []
        system_prompt = (
            "You expand biomedical search keywords. Return strict JSON with one key expansions, "
            "whose value is a short list of highly relevant biomedical synonyms or refined query phrases."
        )
        user_prompt = (
            f"Seed term: {term}\n"
            f"Return up to {self.max_terms} items. Avoid generic filler terms. "
            "Use canonical biomedical names when possible. Format: {\"expansions\": [\"...\"]}"
        )
        try:
            payload = self.client.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            return [
                KeywordRecord(
                    term=term,
                    normalized_term=normalize_keyword(term),
                    source="openai_error",
                    parent_term=term,
                    accepted=False,
                    notes=f"intern_expand_failed={exc}",
                )
            ]
        expansions = payload.get("expansions", []) if isinstance(payload, dict) else []
        records: list[KeywordRecord] = []
        seen: set[str] = set()
        for item in expansions:
            label = str(item).strip()
            normalized = normalize_keyword(label)
            if not label or normalized in seen or normalized == normalize_keyword(term):
                continue
            seen.add(normalized)
            records.append(
                KeywordRecord(
                    term=label,
                    normalized_term=normalized,
                    source="openai_intern",
                    parent_term=term,
                    notes=f"model={self.model}",
                )
            )
        return records


class KeywordExpansionEngine:
    def __init__(self, config: dict[str, Any]):
        self.iterations = max(int(config.get("iterations", 2)), 1)
        self.max_terms = max(int(config.get("max_terms", 500)), 1)
        self.min_term_length = max(int(config.get("min_term_length", 3)), 1)
        self.local_expander = LocalKeywordExpander(config)
        self.mesh_expander = MeshKeywordExpander(config.get("mesh", {}))
        self.openai_expander = OpenAIKeywordExpander(config.get("openai", {}))

    def expand(self, seed_keywords: list[str]) -> tuple[list[KeywordRecord], dict[str, Any]]:
        all_records: list[KeywordRecord] = []
        seen_terms: set[str] = set()
        frontier: list[str] = list(seed_keywords)
        for seed in seed_keywords:
            normalized = normalize_keyword(seed)
            if normalized not in seen_terms:
                seen_terms.add(normalized)
                all_records.append(
                    KeywordRecord(
                        term=seed,
                        normalized_term=normalized,
                        source="seed",
                        iteration=0,
                        notes="user_provided",
                    )
                )
        for iteration in range(1, self.iterations + 1):
            next_frontier: list[str] = []
            for term in frontier:
                provider_records = (
                    self.local_expander.expand(term)
                    + self.mesh_expander.expand(term)
                    + self.openai_expander.expand(term)
                )
                for record in provider_records:
                    record.iteration = iteration
                    if len(record.normalized_term) < self.min_term_length:
                        record.accepted = False
                        record.notes = f"{record.notes}; rejected:min_length"
                    if record.normalized_term in seen_terms:
                        record.accepted = False
                        record.notes = f"{record.notes}; rejected:duplicate"
                    if len(all_records) >= self.max_terms:
                        record.accepted = False
                        record.notes = f"{record.notes}; rejected:max_terms"
                    all_records.append(record)
                    if record.accepted:
                        seen_terms.add(record.normalized_term)
                        next_frontier.append(record.term)
            if not next_frontier:
                break
            frontier = next_frontier
        accepted_records = [record for record in all_records if record.accepted]
        stats = {
            "seed_count": len(seed_keywords),
            "accepted_count": len(accepted_records),
            "candidate_count": len(all_records),
            "iterations": self.iterations,
        }
        return all_records, stats
