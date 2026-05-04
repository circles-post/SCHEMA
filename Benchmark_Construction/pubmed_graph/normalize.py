from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from .models import TripleRecord
from .ontology import Ontology
from .utils import normalize_keyword, normalize_text

# ---------------------------------------------------------------------------
# Ontology singleton.
#
# As of stage 1.3, the entity-type / relation / alias tables that used to live
# as Python literals in this file now come from `pubmed_graph/ontology.yaml`.
# The module-level constants below are kept as backwards-compatible read-only
# views: any code that historically did `from pubmed_graph.normalize import
# STATIC_ALIAS_MAP` continues to work, but the source of truth is the
# Ontology singleton. Stage 2 lets the OntologyProposerAgent replace the
# default singleton with a per-run extended ontology before pipeline phase 4.
# ---------------------------------------------------------------------------
_ONTOLOGY = Ontology.default()


def _set_active_ontology(ontology: Ontology) -> None:
    """Used by stage 2 OntologyProposerAgent to swap in the per-run ontology.

    Reassigns the module-level views so that anything still importing the
    legacy constants picks up the new tables.
    """
    global _ONTOLOGY, GENERIC_ENTITY_NAMES, STATIC_ALIAS_MAP, ENTITY_TYPE_MAP
    global RELATION_MAP, ALLOWED_RELATIONS, ASSOCIATION_HINTS, IMPROVEMENT_HINTS
    _ONTOLOGY = ontology
    GENERIC_ENTITY_NAMES = ontology.generic_entity_names
    STATIC_ALIAS_MAP = ontology.static_alias_map
    ENTITY_TYPE_MAP = ontology.entity_type_map
    RELATION_MAP = ontology.relation_map
    ALLOWED_RELATIONS = ontology.allowed_relations
    ASSOCIATION_HINTS = ontology.association_hints
    IMPROVEMENT_HINTS = ontology.improvement_hints

ABBREVIATION_RE = re.compile(
    r"(?P<long>[A-Za-z][A-Za-z0-9][A-Za-z0-9 /,\-]{3,120}?)\s*\((?P<short>[A-Z][A-Z0-9\-]{1,15})\)"
)
CELL_LINE_RE = re.compile(r"^(?=.*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+){1,5}$")
DRUG_CODE_RE = re.compile(r"^[A-Z]{2,6}-\d{2,6}[A-Z]?$")
PHOSPHO_PROTEIN_RE = re.compile(r"^(?:p|P)-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
SHORT_UPPER_ENTITY_RE = re.compile(r"^[A-Z]{2,4}$")
PROTEIN_FAMILY_RE = re.compile(r"^(?:IGF2BP|HNRNP|hnRNP|BRD|LATS|MST|TMPRSS|FZD|SMC|NSE|CUL)\w*(?:[-/–]\w+)*$")
INTERACTION_FRAGMENT_RE = re.compile(r"^(?:physical\s+)?association\s+between\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE)
BINDING_FRAGMENT_RE = re.compile(r"^(?:physical\s+)?binding\s+of\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE)
INTERACTION_BETWEEN_RE = re.compile(r"^(?:physical\s+)?interaction\s+between\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE)
INTERACTION_OF_RE = re.compile(r"^(.+?)\s+interaction\s+with\s+(.+)$", re.IGNORECASE)
PROCESS_NODE_RE = re.compile(r"^(?:physical\s+)?(?:association|interaction|binding)\b", re.IGNORECASE)
MOLECULAR_ENTITY_RE = re.compile(r"^(?:m6a|n6[- ]methyladenosine|n1[- ]methyladenosine|m1a|n6,2'-o-dimethyladenosine)$", re.IGNORECASE)
PROTEIN_NAME_HINT_RE = re.compile(r"\b(?:cullin(?:[- ]\d+)?|cul\d+|kinase|phosphatase|ligase|receptor|histone|actin|tubulin|integrin|cyclin|protease|transporter|subunit)\b", re.IGNORECASE)
SURVIVAL_RE = re.compile(
    r"\b(overall survival|progression-free survival|disease-free survival|relapse-free survival|survival rate)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# The eight tables below used to be Python literals here. Stage 1.3 moved them
# to pubmed_graph/ontology.yaml. They are exposed as module-level views so any
# legacy `from pubmed_graph.normalize import STATIC_ALIAS_MAP` keeps working.
#
# IMPORTANT: never mutate these dicts/sets in place. Always copy
# (`dict(STATIC_ALIAS_MAP)`) before extending — the underlying objects are
# owned by the Ontology singleton and stage 2 may swap the singleton via
# `_set_active_ontology` to install a per-run extended ontology.
# ---------------------------------------------------------------------------
GENERIC_ENTITY_NAMES = _ONTOLOGY.generic_entity_names
STATIC_ALIAS_MAP = _ONTOLOGY.static_alias_map
ENTITY_TYPE_MAP = _ONTOLOGY.entity_type_map
RELATION_MAP = _ONTOLOGY.relation_map
ALLOWED_RELATIONS = _ONTOLOGY.allowed_relations
ASSOCIATION_HINTS = _ONTOLOGY.association_hints
IMPROVEMENT_HINTS = _ONTOLOGY.improvement_hints


def _to_row(item: TripleRecord | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    return dict(item)


def _snake_case(text: str) -> str:
    text = normalize_text(text).replace("-", "_").replace("/", " ")
    text = re.sub(r"[^A-Za-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text.lower()


def _clean_entity_text(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = cleaned.strip(" \t\n\r\"'`.,;:()[]{}")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\([^)]*$", "", cleaned).strip()
    return cleaned.strip()


def _looks_like_abbreviation(short_form: str, long_form: str) -> bool:
    compact_short = re.sub(r"[^A-Z0-9]", "", short_form.upper())
    initials = "".join(ch.upper() for ch in re.findall(r"\b[A-Za-z]", long_form))
    return bool(compact_short) and (initials.startswith(compact_short) or compact_short in initials)


def _extract_alias_map(evidence: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for match in ABBREVIATION_RE.finditer(evidence or ""):
        long_form = _clean_entity_text(match.group("long"))
        short_form = _clean_entity_text(match.group("short"))
        if not long_form or not short_form:
            continue
        if not _looks_like_abbreviation(short_form, long_form):
            continue
        aliases[normalize_keyword(short_form)] = long_form
    return aliases


def _compact_binary_process(left: str, right: str, evidence: str = "") -> str:
    norm_left = _normalize_entity_fragment(left, evidence=evidence)
    norm_right = _normalize_entity_fragment(right, evidence=evidence)
    if norm_left and norm_right:
        return f"{norm_left}-{norm_right} interaction"
    return ""


def _normalize_entity_fragment(text: str, evidence: str = "") -> str:
    cleaned = _clean_entity_text(text)
    if not cleaned:
        return ""
    alias_map = dict(STATIC_ALIAS_MAP)
    alias_map.update(_extract_alias_map(evidence))
    lowered = normalize_keyword(cleaned)
    if lowered in alias_map:
        return alias_map[lowered]
    for pattern in (INTERACTION_FRAGMENT_RE, BINDING_FRAGMENT_RE, INTERACTION_BETWEEN_RE, INTERACTION_OF_RE):
        match = pattern.match(cleaned)
        if match:
            compact = _compact_binary_process(match.group(1), match.group(2), evidence=evidence)
            if compact:
                return compact
    cleaned = re.sub(r"^(?:physical\s+)?association\s+between\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:physical\s+)?interaction\s+between\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:physical\s+)?binding\s+of\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:expression of|levels of)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:!?")
    lowered = normalize_keyword(cleaned)
    return alias_map.get(lowered, cleaned)


def _looks_like_cell_type(text: str) -> bool:
    lower = normalize_keyword(text)
    if not lower or "cell line" in lower:
        return False
    if lower.endswith(" cells") or lower.endswith(" cell"):
        return True
    return lower in {
        "pbmcs",
        "escs",
        "embryonic stem cells",
        "definitive endoderm",
        "definitive endoderm cells",
        "mesendoderm",
        "mesendoderm cells",
    }


def canonical_entity_type(raw_type: str, entity_text: str) -> str:
    entity_text = _clean_entity_text(entity_text)
    lower = normalize_keyword(entity_text)
    mapped = ENTITY_TYPE_MAP.get(_snake_case(raw_type), "")
    # Stage 3.1: deleted single-paper NSE/SMC special-cases. The ontology
    # proposer + EntityCanonicalizer now handle paper-specific terminology
    # at runtime.
    if MOLECULAR_ENTITY_RE.match(entity_text):
        return "MolecularEntity"
    if lower.endswith(" modification"):
        return "BiologicalProcess"
    if lower.endswith(" interaction") and "-" in entity_text:
        return "BiologicalProcess"
    if PROTEIN_FAMILY_RE.match(entity_text):
        return "Protein"
    if PROTEIN_NAME_HINT_RE.search(entity_text):
        return "Protein"
    if PHOSPHO_PROTEIN_RE.match(entity_text) or lower.startswith("phospho-"):
        return "Protein"
    if DRUG_CODE_RE.match(entity_text):
        return "Drug"
    if SURVIVAL_RE.search(entity_text) or any(
        term in lower
        for term in [
            "prognosis",
            "progression",
            "recurrence",
            "mortality",
            "treatment response",
            "response rate",
            "resistance",
        ]
    ):
        return "ClinicalEndpoint"
    if "pathway" in lower or "pathways" in lower or "signaling" in lower or "signalling" in lower or "cascade" in lower:
        return "Pathway"
    if "complex" in lower:
        return "Complex"
    if _looks_like_cell_type(entity_text):
        return "CellType"
    if "immunohistochemistry" in lower or "staining" in lower or lower == "ihc":
        return "StainingMethod"
    if "algorithm" in lower or "classifier" in lower or "random forest" in lower or lower == "svm":
        return "Algorithm"
    if "cell line" in lower or (
        CELL_LINE_RE.match(entity_text)
        and not DRUG_CODE_RE.match(entity_text)
        and not PHOSPHO_PROTEIN_RE.match(entity_text)
    ):
        return "CellLine"
    if mapped in {
        "Gene",
        "Protein",
        "Drug",
        "Biomarker",
        "Algorithm",
        "StainingMethod",
        "TissueRegion",
        "Pathway",
        "CellLine",
        "CellType",
        "Complex",
        "RNA",
        "MolecularEntity",
    }:
        return mapped
    if any(
        term in lower
        for term in [
            "apoptosis",
            "carcinogenesis",
            "cell survival",
            "cell death",
            "proliferation",
            "angiogenesis",
            "migration",
            "invasion",
        ]
    ):
        return "BiologicalProcess"
    if any(
        term in lower
        for term in ["cancer", "carcinoma", "neoplasm", "tumor", "tumour", "disease", "syndrome"]
    ):
        return "Disease"

    # Stage 5: final-chance synonym lookup for LLM-invented types. This
    # catches "SmallMolecule", "lncRNA", "ProteinDomain", etc. and routes
    # them to one of the 16 core types. Anything still unmapped falls
    # through to the legacy "keep raw" fallback — but _should_drop will
    # reject it when `strict_entity_types` is enabled in ontology.yaml.
    synonym = _ONTOLOGY.resolve_type_synonym(raw_type)
    if synonym:
        return synonym
    return mapped or (raw_type.strip() if raw_type.strip() else "Entity")


def canonical_entity_name(text: str, entity_type: str, evidence: str = "") -> str:
    cleaned = _normalize_entity_fragment(text, evidence=evidence)
    if not cleaned:
        return ""
    # Stage 3.1: deleted NSE/SMC/definitive-endoderm single-paper overrides.
    # Per-paper terminology is now resolved at runtime via OntologyProposerAgent
    # and EntityCanonicalizer (which checks MeSH/PubMed/sciverse).
    alias_map = dict(STATIC_ALIAS_MAP)
    alias_map.update(_extract_alias_map(evidence))
    alias = alias_map.get(normalize_keyword(cleaned))
    if alias:
        cleaned = alias
    if entity_type == "ClinicalEndpoint":
        cleaned = cleaned.replace("positive survival rate", "overall survival")
    return cleaned


def canonical_relation(
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
    for candidate in [surface_relation, normalized_relation]:
        key = _snake_case(candidate)
        if not key:
            continue
        relation = RELATION_MAP.get(key, key)
        if relation == "activated_by":
            return "activates"
        if relation == "inhibited_by":
            return "inhibits"
        if relation == "regulates_expression_of":
            surface_key = _snake_case(surface_relation)
            if surface_key in {"silences", "suppresses", "downregulates", "inhibits"}:
                return "downregulates"
            if surface_key in {"upregulates", "induces", "activates"}:
                return "upregulates"
            return relation
        if relation in {"involved_in", "promotes"} and (
            tail_type == "Pathway" or "pathway" in normalize_keyword(tail)
        ):
            return "involved_in_pathway"
        if relation in {"improves", "associated_with"} and tail_type == "ClinicalEndpoint":
            if relation == "associated_with" and any(hint in evidence_key for hint in IMPROVEMENT_HINTS):
                return "improves"
            return relation
        if relation in ALLOWED_RELATIONS:
            return relation
    if any(hint in evidence_key for hint in ASSOCIATION_HINTS):
        return "associated_with"
    return ""


def _is_generic_entity(text: str) -> bool:
    key = normalize_keyword(text)
    return key in GENERIC_ENTITY_NAMES or key.endswith(" et al")


def _looks_like_sentence_entity(text: str) -> bool:
    lower = normalize_keyword(text)
    # legacy prefix heuristics — kept as-is so nothing that passed under the
    # stage-1 baseline starts failing.
    if lower.startswith(("interaction between ", "binding of ", "association between ")):
        return True
    words = text.split()
    # stage 5: lowered 10 -> 8 because the full run produced a 9-word
    # "Programmed Cell Death 1 and Programmed Cell Death Ligand 1" hub.
    if len(words) > _ONTOLOGY._sentence_max_words:
        return True
    # stage 5: short phrases joined by "and" / "or" are almost always
    # conjunctions of two entities that should have been split upstream.
    for conj in _ONTOLOGY._sentence_conjunction_words:
        if conj in (" " + lower + " "):  # pad so the substring check is robust
            if len(words) > _ONTOLOGY._sentence_conjunction_max_words:
                return True
            break
    return False


def _should_drop(row: dict[str, Any]) -> bool:
    head = row["head"]
    tail = row["tail"]
    relation = row["normalized_relation"]
    if not head or not tail or not relation:
        return True
    if relation not in ALLOWED_RELATIONS:
        return True
    # Stage 5: strict entity-type gate. When ontology.yaml sets
    # `strict_entity_types: true`, any triple whose head_type or tail_type
    # is empty, or not in the active ontology's allowed set, is dropped.
    # This closes the leak where LLM-invented types ("SmallMolecule",
    # "ProteinDomain", "lncRNA", ...) were propagating into the graph.
    if _ONTOLOGY.strict_entity_types:
        allowed = _ONTOLOGY.allowed_entity_types
        head_type = row.get("head_type") or ""
        tail_type = row.get("tail_type") or ""
        if not head_type or not tail_type:
            return True
        if head_type not in allowed or tail_type not in allowed:
            return True
    if row["head_type"] == "CellLine" and PROTEIN_FAMILY_RE.match(head):
        return True
    if row["tail_type"] == "CellLine" and PROTEIN_FAMILY_RE.match(tail):
        return True
    if normalize_keyword(head) == normalize_keyword(tail):
        return True
    if _looks_like_sentence_entity(head) or _looks_like_sentence_entity(tail):
        return True
    if _is_generic_entity(head) or _is_generic_entity(tail):
        return True
    if row["head_type"] in {"Disease", "ClinicalEndpoint"} and SHORT_UPPER_ENTITY_RE.match(head) and normalize_keyword(head) not in STATIC_ALIAS_MAP:
        return True
    if row["tail_type"] in {"Disease", "ClinicalEndpoint"} and SHORT_UPPER_ENTITY_RE.match(tail) and normalize_keyword(tail) not in STATIC_ALIAS_MAP:
        return True
    if relation in {"downregulates", "upregulates", "regulates_expression_of"}:
        if row["head_type"] == "Disease" and row["tail_type"] in {"Gene", "Protein"}:
            return True
    return False


def normalize_triple_row(
    item: TripleRecord | dict[str, Any],
    confidence_threshold: float = 0.5,
    entity_verifier: Any | None = None,
) -> dict[str, Any] | None:
    row = _to_row(item)
    confidence = float(row.get("confidence", 0.0) or 0.0)
    if confidence <= confidence_threshold:
        return None
    evidence = normalize_text(row.get("evidence", ""))
    surface_relation = normalize_text(row.get("surface_relation", ""))
    normalized_relation = normalize_text(row.get("normalized_relation", ""))
    surface_key = _snake_case(surface_relation)
    if surface_key in {"activated_by", "inhibited_by", "upregulated_by", "downregulated_by"}:
        row["head"], row["tail"] = row.get("tail", ""), row.get("head", "")
        row["head_type"], row["tail_type"] = row.get("tail_type", ""), row.get("head_type", "")
        surface_relation = {
            "activated_by": "activates",
            "inhibited_by": "inhibits",
            "upregulated_by": "upregulates",
            "downregulated_by": "downregulates",
        }[surface_key]
        normalized_relation = surface_relation
        row["surface_relation"] = surface_relation
        row["normalized_relation"] = normalized_relation
    head_type = canonical_entity_type(str(row.get("head_type", "")), str(row.get("head", "")))
    tail_type = canonical_entity_type(str(row.get("tail_type", "")), str(row.get("tail", "")))
    head = canonical_entity_name(str(row.get("head", "")), head_type, evidence=evidence)
    tail = canonical_entity_name(str(row.get("tail", "")), tail_type, evidence=evidence)
    if entity_verifier is not None:
        head_verdict = entity_verifier.verify(head, entity_type=head_type, evidence=evidence)
        tail_verdict = entity_verifier.verify(tail, entity_type=tail_type, evidence=evidence)
        if not head_verdict.get("keep", True) or not tail_verdict.get("keep", True):
            return None
        head = canonical_entity_name(str(head_verdict.get("canonical_name", head)), head_type, evidence=evidence)
        tail = canonical_entity_name(str(tail_verdict.get("canonical_name", tail)), tail_type, evidence=evidence)
        head_type = normalize_text(head_verdict.get("entity_type", head_type)) or head_type
        tail_type = normalize_text(tail_verdict.get("entity_type", tail_type)) or tail_type
    head_type = canonical_entity_type(head_type, head)
    tail_type = canonical_entity_type(tail_type, tail)
    normalized = {
        "doc_id": str(row.get("doc_id", "")).strip(),
        "chunk_id": str(row.get("chunk_id", "")).strip(),
        "head": head,
        "head_type": head_type,
        "surface_relation": surface_relation,
        "normalized_relation": canonical_relation(
            normalized_relation,
            surface_relation,
            head,
            head_type,
            tail,
            tail_type,
            evidence=evidence,
        ),
        "tail": tail,
        "tail_type": tail_type,
        "confidence": confidence,
        "evidence": evidence,
    }
    if _should_drop(normalized):
        return None
    return normalized


def deduplicate_triples(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["doc_id"],
            row["head"],
            row["head_type"],
            row["normalized_relation"],
            row["tail"],
            row["tail_type"],
        )
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = dict(row)
            continue
        if float(row.get("confidence", 0.0)) > float(previous.get("confidence", 0.0)):
            previous["confidence"] = float(row["confidence"])
            previous["surface_relation"] = row["surface_relation"]
            previous["evidence"] = row["evidence"]
        elif len(str(row.get("evidence", ""))) > len(str(previous.get("evidence", ""))):
            previous["evidence"] = row["evidence"]
    return list(deduped.values())


def normalize_triple_rows(
    rows: Iterable[TripleRecord | dict[str, Any]],
    confidence_threshold: float = 0.5,
    entity_verifier: Any | None = None,
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = normalize_triple_row(row, confidence_threshold=confidence_threshold, entity_verifier=entity_verifier)
        if item is not None:
            normalized.append(item)
    return deduplicate_triples(normalized)


def normalize_triple_records(
    rows: Iterable[TripleRecord | dict[str, Any]],
    confidence_threshold: float = 0.5,
    entity_verifier: Any | None = None,
) -> list[TripleRecord]:
    normalized_rows = normalize_triple_rows(rows, confidence_threshold=confidence_threshold, entity_verifier=entity_verifier)
    return [TripleRecord(**row) for row in normalized_rows]
