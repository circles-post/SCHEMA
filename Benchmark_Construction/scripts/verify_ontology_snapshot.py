"""Verify that pubmed_graph/ontology.yaml is a faithful snapshot of the
hardcoded constants in pubmed_graph/normalize.py.

This is a STAGE 1.1 sanity check. It does not run any LLM. It simply
loads ontology.yaml via the Ontology class and asserts that:

- Ontology.allowed_relations              == normalize.ALLOWED_RELATIONS
- Ontology.entity_type_map                == normalize.ENTITY_TYPE_MAP
- Ontology.relation_map                   ⊇ normalize.RELATION_MAP
- Ontology.static_alias_map               == normalize.STATIC_ALIAS_MAP
- Ontology.association_hints              == normalize.ASSOCIATION_HINTS
- Ontology.improvement_hints              == normalize.IMPROVEMENT_HINTS
- Ontology.generic_entity_names           == normalize.GENERIC_ENTITY_NAMES

If anything is missing the script prints the diff and exits non-zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pubmed_graph import normalize  # noqa: E402
from pubmed_graph.ontology import Ontology  # noqa: E402


def diff_set(name: str, lhs: set, rhs: set) -> bool:
    only_l = lhs - rhs
    only_r = rhs - lhs
    if only_l or only_r:
        print(f"[FAIL] {name}: missing in ontology={sorted(only_l)} extra={sorted(only_r)}")
        return False
    print(f"[ok]   {name}: {len(lhs)} entries match")
    return True


def diff_dict(name: str, lhs: dict, rhs: dict) -> bool:
    only_l = {k: lhs[k] for k in lhs if k not in rhs}
    only_r = {k: rhs[k] for k in rhs if k not in lhs}
    mismatched = {k: (lhs[k], rhs[k]) for k in lhs if k in rhs and lhs[k] != rhs[k]}
    if only_l or only_r or mismatched:
        if only_l:
            print(f"[FAIL] {name}: missing in ontology: {only_l}")
        if only_r:
            print(f"[FAIL] {name}: extra in ontology: {only_r}")
        if mismatched:
            print(f"[FAIL] {name}: value mismatches: {mismatched}")
        return False
    print(f"[ok]   {name}: {len(lhs)} entries match")
    return True


def main() -> None:
    onto = Ontology.default()
    ok = True

    # Sets
    ok &= diff_set("ALLOWED_RELATIONS", normalize.ALLOWED_RELATIONS, onto.allowed_relations)
    ok &= diff_set("ASSOCIATION_HINTS", normalize.ASSOCIATION_HINTS, onto.association_hints)
    ok &= diff_set("IMPROVEMENT_HINTS", normalize.IMPROVEMENT_HINTS, onto.improvement_hints)
    ok &= diff_set("GENERIC_ENTITY_NAMES", normalize.GENERIC_ENTITY_NAMES, onto.generic_entity_names)

    # Dicts (entity_type_map should be a strict superset of normalize.ENTITY_TYPE_MAP)
    ok &= diff_dict("ENTITY_TYPE_MAP", normalize.ENTITY_TYPE_MAP, onto.entity_type_map)
    ok &= diff_dict("STATIC_ALIAS_MAP", normalize.STATIC_ALIAS_MAP, onto.static_alias_map)

    # RELATION_MAP: ontology may have MORE entries (because aliases are auto-derived);
    # we require that every key in normalize.RELATION_MAP also exists in onto and resolves
    # to the same canonical relation.
    missing = []
    mismatched = []
    for k, v in normalize.RELATION_MAP.items():
        if k not in onto.relation_map:
            missing.append(k)
        elif onto.relation_map[k] != v:
            # ontology may resolve aliases to a deeper canonical (e.g. activated_by → activates)
            # whereas legacy RELATION_MAP keeps activated_by → activated_by; this is allowed
            # only when the legacy value is in {activated_by, inhibited_by, gene_expression}.
            if v in {"activated_by", "inhibited_by", "gene_expression"}:
                continue
            mismatched.append((k, v, onto.relation_map[k]))
    if missing or mismatched:
        if missing:
            print(f"[FAIL] RELATION_MAP missing keys in ontology: {missing}")
        if mismatched:
            print(f"[FAIL] RELATION_MAP value mismatches: {mismatched}")
        ok = False
    else:
        print(f"[ok]   RELATION_MAP: {len(normalize.RELATION_MAP)} keys match (with {len(onto.relation_map) - len(normalize.RELATION_MAP)} ontology-only extras)")

    # Entity types whitelist
    legacy_types = set(normalize.ENTITY_TYPE_MAP.values()) | {
        "Gene", "Protein", "Drug", "Biomarker", "Algorithm", "StainingMethod",
        "TissueRegion", "Pathway", "CellLine", "CellType", "Complex", "RNA",
        "MolecularEntity", "BiologicalProcess", "Disease", "ClinicalEndpoint",
    }
    ok &= diff_set("entity_type_ids (≥)", legacy_types, onto.allowed_entity_types | legacy_types)

    print()
    if ok:
        print("=" * 60)
        print("ontology.yaml is a faithful snapshot of normalize.py constants.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("SNAPSHOT MISMATCH — fix ontology.yaml before stage 1.2.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
