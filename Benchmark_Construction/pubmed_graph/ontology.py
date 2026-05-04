"""Ontology loader and runtime resolver for the pubmed_graph pipeline.

This module is the runtime façade over `pubmed_graph/ontology.yaml`. It is
the single source of truth for entity types, relation types, alias maps,
and the heuristic regexes that previously lived as Python constants in
`normalize.py` and `entity_verification.py`.

Stage 1 of the ontology refactor only requires that loading this module
and calling its methods produces output bit-for-bit equivalent to the
old hardcoded `normalize_triple_rows` pipeline. Therefore the rule
ordering and edge cases here mirror `normalize.py` exactly. Do not
"clean up" any of the conditions below without re-running the baseline
diff in `scripts/diff_against_baseline.py`.

Stage 2 introduces `OntologyProposerAgent`, which can produce a
`run_ontology = Ontology.merge(base, extensions)` instance. Extensions
only ever ADD entries (new entity types, new aliases, new relation
surface forms). They never remove core_relations or core entity_types.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

from .utils import normalize_keyword, normalize_text

DEFAULT_ONTOLOGY_PATH = Path(__file__).resolve().parent / "ontology.yaml"


# ---------------------------------------------------------------------------
# Helpers shared with normalize.py (kept here so Ontology is self-contained)
# ---------------------------------------------------------------------------

def _snake_case(text: str) -> str:
    text = normalize_text(text).replace("-", "_").replace("/", " ")
    text = re.sub(r"[^A-Za-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text.lower()


def _compile_flags(names: Iterable[str]) -> int:
    flags = 0
    for name in names or ():
        if name == "IGNORECASE":
            flags |= re.IGNORECASE
        elif name == "MULTILINE":
            flags |= re.MULTILINE
        elif name == "DOTALL":
            flags |= re.DOTALL
    return flags


@dataclass
class _TypeRegexRule:
    id: str
    pattern: re.Pattern
    type: str
    use_search: bool = False  # True for hint-style regexes, False for ^...$ matchers


@dataclass
class _ContextualOverride:
    key: str
    when: tuple[str, ...]
    canonical: str
    type: str | None


@dataclass
class _AliasEntry:
    surface: str
    canonical: str
    type: str | None


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

class Ontology:
    """Runtime ontology view backed by ontology.yaml.

    All methods that mirror legacy normalize.py functions retain the same
    name where possible to make the stage 1.3 migration mechanical.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

        # ---- entity types ----
        self._entity_types: list[dict[str, Any]] = list(data.get("entity_types") or [])
        self._entity_type_ids: set[str] = {et["id"] for et in self._entity_types}
        # ENTITY_TYPE_MAP equivalent: surface_alias → canonical id.
        # We register BOTH the literal lowercased form AND the snake_case form
        # so legacy normalize.py lookups (which used both spellings as separate
        # keys, e.g. "cell line" and "cell_line") remain bit-for-bit equivalent.
        self._entity_type_map: dict[str, str] = {}
        for et in self._entity_types:
            for alias in et.get("surface_aliases") or []:
                raw_lower = (alias or "").strip().lower()
                if raw_lower:
                    self._entity_type_map[raw_lower] = et["id"]
                snake = _snake_case(alias)
                if snake:
                    self._entity_type_map[snake] = et["id"]

        # ---- relations ----
        self._core_relations: list[dict[str, Any]] = list(data.get("core_relations") or [])
        self._allowed_relations: set[str] = {r["id"] for r in self._core_relations}
        # RELATION_MAP equivalent: each surface alias → core relation id.
        # Register both raw lowered form and snake_case form (mirrors the way
        # normalize.RELATION_MAP held both "member of" and "member_of" as keys).
        self._relation_map: dict[str, str] = {}
        for rel in self._core_relations:
            self._relation_map[rel["id"]] = rel["id"]
            for alias in rel.get("surface_aliases") or []:
                raw_lower = (alias or "").strip().lower()
                if raw_lower:
                    self._relation_map[raw_lower] = rel["id"]
                snake = _snake_case(alias)
                if snake:
                    self._relation_map[snake] = rel["id"]
        # extra resolution table (recruits → associated_with, etc.)
        for surface, target in (data.get("relation_resolution") or {}).items():
            raw_lower = (surface or "").strip().lower()
            if raw_lower:
                self._relation_map[raw_lower] = str(target)
            snake = _snake_case(surface)
            if snake:
                self._relation_map[snake] = str(target)

        self._swap_surface_relations: dict[str, str] = {
            _snake_case(k): v for k, v in (data.get("swap_surface_relations") or {}).items()
        }

        hints = data.get("relation_hints") or {}
        self._association_hints: set[str] = set(hints.get("association_hints") or [])
        self._improvement_hints: set[str] = set(hints.get("improvement_hints") or [])

        # ---- aliases (STATIC_ALIAS_MAP equivalent) ----
        self._aliases: list[_AliasEntry] = []
        self._static_alias_map: dict[str, str] = {}
        self._alias_type_map: dict[str, str | None] = {}
        for entry in data.get("aliases") or []:
            ae = _AliasEntry(
                surface=normalize_keyword(entry["surface"]),
                canonical=str(entry["canonical"]),
                type=entry.get("type"),
            )
            self._aliases.append(ae)
            self._static_alias_map[ae.surface] = ae.canonical
            self._alias_type_map[ae.surface] = ae.type

        # ---- contextual overrides (NSE/SMC/definitive endoderm) ----
        self._contextual_overrides: list[_ContextualOverride] = [
            _ContextualOverride(
                key=normalize_keyword(o["key"]),
                when=tuple(o.get("when") or ()),
                canonical=str(o["canonical"]),
                type=o.get("type"),
            )
            for o in data.get("contextual_overrides") or []
        ]

        # ---- regex rules ----
        self._type_regex_rules: list[_TypeRegexRule] = []
        for r in data.get("type_regex_rules") or []:
            rid = r["id"]
            pattern = re.compile(r["pattern"], _compile_flags(r.get("flags") or []))
            # protein_name_hint_re and survival_re use re.search; the rest use re.match.
            use_search = rid in {"protein_name_hint_re", "survival_re"}
            self._type_regex_rules.append(
                _TypeRegexRule(id=rid, pattern=pattern, type=r["type"], use_search=use_search)
            )
        # CELL_LINE_RE and PHOSPHO_PROTEIN_RE are referenced from inside cell-line detection.
        # Keep dedicated handles for them so canonicalize_entity_type can replicate the
        # CellLine tie-break.
        self._cell_line_re = re.compile(r"^(?=.*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+){1,5}$")
        self._phospho_protein_re = re.compile(r"^(?:p|P)-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
        self._drug_code_re = re.compile(r"^[A-Z]{2,6}-\d{2,6}[A-Z]?$")

        # ---- substring & suffix rules ----
        self._type_substring_rules: list[dict[str, Any]] = list(data.get("type_substring_rules") or [])
        self._type_suffix_rules: list[dict[str, Any]] = list(data.get("type_suffix_rules") or [])

        # ---- cell-type heuristics ----
        ct = data.get("cell_type_hints") or {}
        self._cell_type_exact: set[str] = set(ct.get("exact_lower") or [])
        self._cell_type_suffixes: tuple[str, ...] = tuple(ct.get("suffix_lower") or ())
        self._cell_type_excludes: tuple[str, ...] = tuple(ct.get("exclude_substring_lower") or ())

        # ---- complex exact-match (NSE) ----
        self._complex_exact: set[str] = set(data.get("complex_exact_lower") or [])

        # ---- generic blocklist & sentence rules ----
        self._generic_blocklist: set[str] = set(data.get("generic_entity_blocklist") or [])
        self._sentence_prefixes: tuple[str, ...] = tuple(data.get("sentence_entity_prefixes") or ())
        self._sentence_max_words: int = int(data.get("sentence_entity_max_words", 10))
        self._sentence_conjunction_words: tuple[str, ...] = tuple(
            data.get("sentence_conjunction_words") or ()
        )
        self._sentence_conjunction_max_words: int = int(
            data.get("sentence_conjunction_max_words", 5)
        )
        self._short_upper_re = re.compile(data.get("short_upper_acronym_pattern") or "^[A-Z]{2,4}$")
        self._phospho_prefixes: tuple[str, ...] = tuple(data.get("phospho_protein_prefixes") or ())

        # ---- type synonym fallback (stage 5) ----
        # Normalise keys so ENTITY_TYPE_MAP-style lookup works:
        # store both the raw lowercased form and the _snake_case form.
        self._type_synonyms: dict[str, str] = {}
        for raw_key, target in (data.get("type_synonyms") or {}).items():
            if not target or target not in self._entity_type_ids:
                continue
            raw_lower = str(raw_key).strip().lower()
            snake = _snake_case(raw_key)
            if raw_lower:
                self._type_synonyms[raw_lower] = target
            if snake:
                self._type_synonyms[snake] = target
        self._strict_entity_types: bool = bool(data.get("strict_entity_types", False))

        # ---- forbidden direction drops ----
        self._forbidden_drops: list[dict[str, Any]] = list(data.get("forbidden_direction_drops") or [])

        # ---- canonical name replacements ----
        self._canonical_replacements: list[dict[str, Any]] = list(data.get("canonical_name_replacements") or [])

        self.version: str = str(data.get("version", "0.0.0"))

    # -------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Ontology":
        path = Path(path) if path else DEFAULT_ONTOLOGY_PATH
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Ontology file must be a YAML mapping: {path}")
        return cls(data)

    @classmethod
    def default(cls) -> "Ontology":
        return _default_ontology()

    # -------------------------------------------------------------------
    # Read-only views (for backwards-compat with normalize.py constants)
    # -------------------------------------------------------------------

    @property
    def allowed_relations(self) -> set[str]:
        return set(self._allowed_relations)

    @property
    def allowed_entity_types(self) -> set[str]:
        return set(self._entity_type_ids)

    @property
    def entity_type_map(self) -> dict[str, str]:
        return dict(self._entity_type_map)

    @property
    def relation_map(self) -> dict[str, str]:
        return dict(self._relation_map)

    @property
    def static_alias_map(self) -> dict[str, str]:
        return dict(self._static_alias_map)

    @property
    def association_hints(self) -> set[str]:
        return set(self._association_hints)

    @property
    def improvement_hints(self) -> set[str]:
        return set(self._improvement_hints)

    @property
    def generic_entity_names(self) -> set[str]:
        return set(self._generic_blocklist)

    @property
    def swap_surface_relations(self) -> dict[str, str]:
        return dict(self._swap_surface_relations)

    @property
    def type_synonyms(self) -> dict[str, str]:
        return dict(self._type_synonyms)

    @property
    def strict_entity_types(self) -> bool:
        return self._strict_entity_types

    def resolve_type_synonym(self, raw_type: str) -> str:
        """Map an LLM-invented entity type string to a core ontology type id.

        Consulted by normalize.canonical_entity_type as a fallback before
        the regex/substring rules. Returns "" if no synonym is registered
        for raw_type — in that case, the caller's downstream logic decides
        whether to drop or keep the triple.
        """
        if not raw_type:
            return ""
        raw_lower = str(raw_type).strip().lower()
        if raw_lower in self._type_synonyms:
            return self._type_synonyms[raw_lower]
        snake = _snake_case(raw_type)
        return self._type_synonyms.get(snake, "")

    # -------------------------------------------------------------------
    # Entity type canonicalization (mirrors canonical_entity_type)
    # -------------------------------------------------------------------

    def canonical_entity_type(self, raw_type: str, entity_text: str) -> str:
        from .normalize import _clean_entity_text  # local import to avoid cycle

        entity_text = _clean_entity_text(entity_text)
        lower = normalize_keyword(entity_text)
        mapped = self._entity_type_map.get(_snake_case(raw_type), "")

        # 1. complex exact set (nse / nse subunits / smcs / smc proteins)
        if lower in self._complex_exact:
            return "Complex"

        # 2. regex rules in declared order
        for rule in self._type_regex_rules:
            if rule.id == "protein_name_hint_re" or rule.id == "survival_re":
                # search-style: must come AFTER specific structural matches
                continue
            if rule.pattern.match(entity_text):
                if rule.id == "phospho_protein_re":
                    return rule.type
                return rule.type

        # 3. suffix-based BiologicalProcess heuristics
        for srule in self._type_suffix_rules:
            suffix = srule.get("suffix") or ""
            if not suffix:
                continue
            if lower.endswith(suffix.strip().lower() if not suffix.startswith(" ") else suffix.lower()):
                if srule.get("require_dash_in_text") and "-" not in entity_text:
                    continue
                return srule["type"]

        # 4. NSE-with-digit + dash → Complex
        if lower.startswith("nse") and any(ch.isdigit() for ch in lower) and ("-" in entity_text or "–" in entity_text):
            return "Complex"

        # 5. protein hint regex (search) and other search-style regexes
        for rule in self._type_regex_rules:
            if rule.id == "protein_family_re":
                if rule.pattern.match(entity_text):
                    return rule.type
        for rule in self._type_regex_rules:
            if rule.id == "protein_name_hint_re":
                if rule.pattern.search(entity_text):
                    return rule.type

        # 6. phospho prefix
        if any(lower.startswith(prefix) for prefix in self._phospho_prefixes):
            return "Protein"

        # 7. survival regex
        for rule in self._type_regex_rules:
            if rule.id == "survival_re":
                if rule.pattern.search(entity_text):
                    return rule.type

        # 8. substring rules in declared order — BUT replicate the original
        # handling of ClinicalEndpoint / Pathway / Complex / Cell-type / etc.
        for rule in self._type_substring_rules:
            if any(term in lower for term in rule.get("any_of") or ()):
                # special case: CellLine substring rule must defer to phospho/drug-code checks
                if rule["type"] == "CellLine":
                    if not self._drug_code_re.match(entity_text) and not self._phospho_protein_re.match(entity_text):
                        return "CellLine"
                    continue
                return rule["type"]
            if any(lower == term for term in rule.get("exact_lower") or ()):
                return rule["type"]

        # 9. _looks_like_cell_type
        if self._looks_like_cell_type(entity_text):
            return "CellType"

        # 10. CellLine via CELL_LINE_RE structural form
        if self._cell_line_re.match(entity_text) and not self._drug_code_re.match(entity_text) and not self._phospho_protein_re.match(entity_text):
            return "CellLine"

        # 11. fall back to surface-alias mapped value if it lands on a known canonical type
        if mapped in self._entity_type_ids:
            return mapped

        # 12. last resort fallback
        return mapped or (raw_type.strip() if raw_type.strip() else "Entity")

    def _looks_like_cell_type(self, text: str) -> bool:
        lower = normalize_keyword(text)
        if not lower:
            return False
        if any(excl in lower for excl in self._cell_type_excludes):
            return False
        for suffix in self._cell_type_suffixes:
            if lower.endswith(suffix):
                return True
        return lower in self._cell_type_exact

    # -------------------------------------------------------------------
    # Entity name canonicalization (mirrors canonical_entity_name)
    # -------------------------------------------------------------------

    def canonical_entity_name(self, text: str, entity_type: str, evidence: str = "") -> str:
        from .normalize import _normalize_entity_fragment  # avoid cycle

        cleaned = _normalize_entity_fragment(text, evidence=evidence)
        if not cleaned:
            return ""
        evidence_key = normalize_keyword(evidence)
        cleaned_key = normalize_keyword(cleaned)

        # contextual overrides — match key against cleaned name + evidence substrings
        for override in self._contextual_overrides:
            if cleaned_key != override.key:
                continue
            if not override.when:
                cleaned = override.canonical
                cleaned_key = normalize_keyword(cleaned)
                continue
            if any(token in evidence_key for token in override.when):
                cleaned = override.canonical
                cleaned_key = normalize_keyword(cleaned)

        # static alias lookup (post-context override)
        alias = self._static_alias_map.get(normalize_keyword(cleaned))
        if alias:
            cleaned = alias

        # type-specific text replacements (e.g. "positive survival rate" → "overall survival")
        for repl in self._canonical_replacements:
            if repl.get("when_type") and repl["when_type"] != entity_type:
                continue
            for pair in repl.get("replacements") or []:
                cleaned = cleaned.replace(pair["from"], pair["to"])

        return cleaned

    # -------------------------------------------------------------------
    # Relation canonicalization (mirrors canonical_relation)
    # -------------------------------------------------------------------

    def canonical_relation(
        self,
        normalized_relation: str,
        surface_relation: str,
        head: str,
        head_type: str,
        tail: str,
        tail_type: str,
        evidence: str = "",
    ) -> str:
        evidence_key = normalize_keyword(evidence)
        surface_key = _snake_case(surface_relation)
        normalized_key = _snake_case(normalized_relation)

        # special demotions to associated_with / involved_in (mirrors normalize.py)
        if surface_key == "recruits" and normalized_key == "part_of":
            return "associated_with"
        if surface_key in {
            "binding",
            "binds",
            "bound to",
            "bound_to",
            "through_the",
            "through",
            "forms",
            "interaction_with",
            "interaction_between",
            "association_between",
        } and normalized_key == "part_of":
            return "associated_with"
        if surface_key in {"dna_binding", "mediated"} and normalized_key == "part_of":
            return "involved_in"

        for candidate in (surface_relation, normalized_relation):
            key = _snake_case(candidate)
            if not key:
                continue
            relation = self._relation_map.get(key, key)
            if relation == "activated_by":
                return "activates"
            if relation == "inhibited_by":
                return "inhibits"
            if relation == "regulates_expression_of":
                surf = _snake_case(surface_relation)
                if surf in {"silences", "suppresses", "downregulates", "inhibits"}:
                    return "downregulates"
                if surf in {"upregulates", "induces", "activates"}:
                    return "upregulates"
                return relation
            if relation in {"involved_in", "promotes"} and (
                tail_type == "Pathway" or "pathway" in normalize_keyword(tail)
            ):
                return "involved_in_pathway"
            if relation in {"improves", "associated_with"} and tail_type == "ClinicalEndpoint":
                if relation == "associated_with" and any(hint in evidence_key for hint in self._improvement_hints):
                    return "improves"
                return relation
            if relation in self._allowed_relations:
                return relation

        if any(hint in evidence_key for hint in self._association_hints):
            return "associated_with"
        return ""

    # -------------------------------------------------------------------
    # Drop logic (mirrors _should_drop)
    # -------------------------------------------------------------------

    def is_generic_entity(self, text: str) -> bool:
        key = normalize_keyword(text)
        return key in self._generic_blocklist or key.endswith(" et al")

    def looks_like_sentence_entity(self, text: str) -> bool:
        lower = normalize_keyword(text)
        for prefix in self._sentence_prefixes:
            if lower.startswith(prefix):
                return True
        return len(text.split()) > self._sentence_max_words

    def is_short_upper_acronym(self, text: str) -> bool:
        return bool(self._short_upper_re.match(text))

    def is_forbidden_direction(self, head_type: str, relation: str, tail_type: str) -> bool:
        for rule in self._forbidden_drops:
            if relation not in (rule.get("relation_in") or []):
                continue
            if head_type != rule.get("head_type"):
                continue
            if tail_type in (rule.get("tail_type_in") or []):
                return True
        return False

    # -------------------------------------------------------------------
    # Prompt rendering
    # -------------------------------------------------------------------

    def render_prompt_section_entity_types(self) -> str:
        return "\n".join(f"- {et['id']}" for et in self._entity_types)

    def render_prompt_section_relations(self) -> str:
        return "\n".join(f"- {rel['id']}" for rel in self._core_relations)

    def render_prompt_mapping_rules(self) -> str:
        # The legacy prompt has a hand-written mapping rules block. We synthesize a
        # similar block from relation_resolution + relation surface aliases so the
        # downstream prompt template can include it without changing meaning.
        lines: list[str] = []
        for rel in self._core_relations:
            extra = [a for a in (rel.get("surface_aliases") or []) if a != rel["id"]]
            if not extra:
                continue
            lines.append(f"- {', '.join(repr(x) for x in extra)} -> {rel['id']}")
        return "\n".join(lines)


@lru_cache(maxsize=4)
def _default_ontology() -> Ontology:
    return Ontology.load(DEFAULT_ONTOLOGY_PATH)
