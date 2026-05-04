from __future__ import annotations

from pubmed_graph.utils import normalize_text

from .._helpers import csv_rows
from ..registry import REGISTRY, BlueprintContext, ExperimentBlueprint, normalize_relation


# Only match triples that are semantically "pathway activity" claims.
# Previously this blueprint was registered as the unconditional fallback,
# which meant "Drug improves Disease" or "Gene→Protein" triples got
# pathway-activity code attached — semantically meaningless but sandbox-
# passing, which pollutes the experiment_code benchmark.
_PATHWAY_RELATIONS = {
    "associated_with",
    "part_of",
    "involved_in",
    "involved_in_pathway",
    "activates",
    "inhibits",
}
_SUBJECT_TYPES = {"gene", "protein", "enzyme", "transcript", "mrna", "lncrna", "mirna"}
_PATHWAY_TYPES = {"pathway", "biologicalprocess", "biological_process", "process"}


def _type_key(node_type: str) -> str:
    return (node_type or "").strip().lower().replace(" ", "").replace("_", "")


def _predicate(context: BlueprintContext) -> bool:
    relation = normalize_relation(context.relation)
    if relation not in _PATHWAY_RELATIONS:
        return False
    return _type_key(context.head_type) in _SUBJECT_TYPES and _type_key(context.tail_type) in _PATHWAY_TYPES


def _factory(context: BlueprintContext) -> ExperimentBlueprint:
    subject = normalize_text(context.head)
    pathway = normalize_text(context.tail)
    relation = normalize_relation(context.relation)
    csv_text = csv_rows(
        [
            ("gene", "weight", "signal"),
            (subject, 1.0, 2.5),
            (f"{subject}_partner", 0.7, 1.8),
            (f"{subject}_regulator", 0.5, 1.2),
        ]
    )
    data_code = f'''#!/usr/bin/env python3
"""Synthetic pathway activity inputs for {subject} and {pathway}."""

from io import StringIO

import pandas as pd


CSV_TEXT = """{csv_text}"""


def load_pathway_activity_inputs() -> pd.DataFrame:
    return pd.read_csv(StringIO(CSV_TEXT))
'''
    main_code = f'''#!/usr/bin/env python3
"""Score pathway activity linked to {subject} and {pathway}."""

import numpy as np

from data_en import load_pathway_activity_inputs


def calculate_weighted_signal(weights: np.ndarray, signals: np.ndarray) -> float:
    """
    Compute a weighted pathway signal score.
    """
    return float(np.sum(weights * signals) / max(np.sum(weights), 1e-8))


def classify_pathway_support(score: float) -> str:
    """
    Convert the pathway score into a simple support label.
    """
    return "supported" if score >= 1.8 else "not_supported"


def summarize_pathway_support() -> dict[str, float | str]:
    table = load_pathway_activity_inputs()
    weights = table["weight"].to_numpy(dtype=float)
    signals = table["signal"].to_numpy(dtype=float)
    score = calculate_weighted_signal(weights, signals)
    verdict = classify_pathway_support(score)
    return {{
        "subject": "{subject}",
        "pathway": "{pathway}",
        "score": round(score, 4),
        "verdict": verdict,
    }}


if __name__ == "__main__":
    print(summarize_pathway_support())
'''
    return ExperimentBlueprint(
        name="pathway_activity",
        task_family="pathway_activity",
        relation=relation,
        direction=f"pathway_activity_{subject.lower().replace(' ', '_')}",
        discipline="life",
        function_type="Data processing",
        task_objective=f"Implement a pathway activity score that reflects whether {subject} is connected to {pathway}.",
        research_focus=(
            f"The evidence places {subject} in a reported relationship with pathway context {pathway}. "
            "Turn that observation into a minimal activity-scoring task."
        ),
        data_code_template=data_code,
        main_code_template=main_code,
        incomplete_functions=("calculate_weighted_signal", "classify_pathway_support"),
        hard_extra_blanks=("summarize_pathway_support",),
        github_repo_query=f"{pathway} pathway activity scoring python",
        github_code_query="pathway activity score python",
        unit_tests=(
            # (1.0*2.5 + 0.7*1.8 + 0.5*1.2) / (1.0+0.7+0.5) = 4.36/2.2 ≈ 1.9818
            # verified by sandbox self-test.
            {"name": "weighted_score", "input": {}, "expected_output": {"score": 1.9818}},
            {"name": "verdict", "input": {}, "expected_output": {"verdict": "supported"}},
        ),
    )


# Pathway activity is now a predicate-gated blueprint so non-pathway triples
# (e.g. Drug→Disease) aren't forced through the pathway harness. Triples that
# don't match any blueprint will hit dispatch_blueprint's RuntimeError, which
# experiment_generator now catches and turns into a rejected sample.
REGISTRY.register(name="pathway_activity", predicate=_predicate, factory=_factory, priority=40)
