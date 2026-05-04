from __future__ import annotations

from pubmed_graph.utils import normalize_text

from .._helpers import csv_rows
from ..registry import REGISTRY, BlueprintContext, ExperimentBlueprint, normalize_relation


_RELATIONS = {"upregulated_in", "downregulated_in", "overexpressed_in"}
_GENE_HEAD_TYPES = {"gene", "protein", "transcript", "mrna", "lncrna", "mirna"}
_DISEASE_TAIL_TYPES = {"disease", "condition", "syndrome", "cancer", "tumor", "phenotype"}


def _factory(context: BlueprintContext) -> ExperimentBlueprint:
    gene = normalize_text(context.head)
    disease = normalize_text(context.tail)
    relation = normalize_relation(context.relation)
    relation_label = relation.replace("_", " ")
    # Match synthetic expression direction to the relation so the unit test
    # expectation lines up with the ground-truth claim: "downregulated_in"
    # produces a NEGATIVE log2 fold change, anything else positive.
    downregulated = "down" in relation
    if downregulated:
        disease_values = (4.0, 5.0, 4.5, 5.5)
        control_values = (13.0, 11.0, 12.5, 10.5)
        expected_log2_fc = -1.3068
        fold_change_test_name = "fold_change_negative"
    else:
        disease_values = (13.0, 11.0, 12.5, 10.5)
        control_values = (4.0, 5.0, 4.5, 5.5)
        expected_log2_fc = 1.3068
        fold_change_test_name = "fold_change_positive"
    csv_text = csv_rows(
        [
            ("sample_id", "group", "expression"),
            ("D1", "disease", disease_values[0]),
            ("D2", "disease", disease_values[1]),
            ("D3", "disease", disease_values[2]),
            ("D4", "disease", disease_values[3]),
            ("C1", "control", control_values[0]),
            ("C2", "control", control_values[1]),
            ("C3", "control", control_values[2]),
            ("C4", "control", control_values[3]),
        ]
    )
    data_code = f'''#!/usr/bin/env python3
"""Synthetic expression table for {gene} in {disease}."""

from io import StringIO

import pandas as pd


CSV_TEXT = """{csv_text}"""


def load_expression_table() -> pd.DataFrame:
    return pd.read_csv(StringIO(CSV_TEXT))
'''
    main_code = f'''#!/usr/bin/env python3
"""Quantify differential expression for {gene} in {disease}."""

import numpy as np

from data_en import load_expression_table


def compute_log2_fold_change(case_values: np.ndarray, control_values: np.ndarray) -> float:
    """
    Compute log2 fold change with a small pseudocount for stability.
    """
    pseudocount = 1e-6
    case_mean = float(np.mean(case_values))
    control_mean = float(np.mean(control_values))
    return float(np.log2((case_mean + pseudocount) / (control_mean + pseudocount)))


def classify_expression_shift(log2_fc: float, relation_label: str) -> str:
    """
    Convert the computed fold change into an evidence-aware qualitative label.
    """
    relation_key = relation_label.casefold()
    if "down" in relation_key:
        return "supported" if log2_fc < -0.5 else "not_supported"
    return "supported" if log2_fc > 0.5 else "not_supported"


def summarize_expression_shift() -> dict[str, float | str]:
    table = load_expression_table()
    case_values = table.loc[table["group"] == "disease", "expression"].to_numpy(dtype=float)
    control_values = table.loc[table["group"] == "control", "expression"].to_numpy(dtype=float)
    log2_fc = compute_log2_fold_change(case_values, control_values)
    verdict = classify_expression_shift(log2_fc, "{relation_label}")
    return {{
        "gene": "{gene}",
        "disease": "{disease}",
        "log2_fold_change": round(log2_fc, 4),
        "relation": "{relation_label}",
        "verdict": verdict,
    }}


if __name__ == "__main__":
    print(summarize_expression_shift())
'''
    return ExperimentBlueprint(
        name="differential_expression",
        task_family="differential_expression",
        relation=relation,
        direction=f"differential_expression_{gene.lower().replace(' ', '_')}",
        discipline="life",
        function_type="Numerical calculation",
        task_objective=f"Implement the fold-change analysis used to test whether {gene} is reported to be {relation_label} in {disease}.",
        research_focus=(
            f"The evidence states that {gene} is reported to be {relation_label} in {disease}. "
            "Turn that claim into a small differential-expression coding task with explicit quantitative checks."
        ),
        data_code_template=data_code,
        main_code_template=main_code,
        incomplete_functions=("compute_log2_fold_change", "classify_expression_shift"),
        hard_extra_blanks=("summarize_expression_shift",),
        github_repo_query=f"{gene} differential expression {disease} python",
        github_code_query=f"{gene} {disease} log2 fold change differential expression",
        unit_tests=(
            # The synthetic expression direction matches the relation:
            # up/overexpressed claims expect positive log2FC; down claims expect negative log2FC.
            {"name": fold_change_test_name, "input": {}, "expected_output": {"log2_fold_change": expected_log2_fc}},
            {"name": "verdict_supported", "input": {}, "expected_output": {"verdict": "supported"}},
        ),
    )


def _predicate(context: BlueprintContext) -> bool:
    if normalize_relation(context.relation) not in _RELATIONS:
        return False
    head_t = (context.head_type or "").strip().lower().replace(" ", "")
    tail_t = (context.tail_type or "").strip().lower().replace(" ", "")
    return head_t in _GENE_HEAD_TYPES and tail_t in _DISEASE_TAIL_TYPES


REGISTRY.register(name="differential_expression", predicate=_predicate, factory=_factory, priority=20)
