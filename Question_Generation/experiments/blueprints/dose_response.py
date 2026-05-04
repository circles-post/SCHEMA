from __future__ import annotations

from pubmed_graph.utils import normalize_text

from .._helpers import csv_rows
from ..registry import REGISTRY, BlueprintContext, ExperimentBlueprint, normalize_relation


_RELATIONS = {"inhibits"}
# dose_response models a pharmacological titration where a small molecule
# is titrated against a molecular target and a response curve is fit. That
# only makes sense for (Drug|Compound, inhibits, Protein|Enzyme|Receptor|...)
# triples. Firing on (Protein, promotes, Disease) or (Complex, promotes,
# Process) produces nonsense code that has nothing to do with the claim.
_DRUG_HEAD_TYPES = {
    "drug", "compound", "smallmolecule", "small_molecule", "chemical",
    "inhibitor", "molecule",
}
_TARGET_TAIL_TYPES = {
    "protein", "enzyme", "kinase", "receptor", "gene", "transcriptionfactor",
    "transcription_factor", "channel", "transporter", "target",
}


def _factory(context: BlueprintContext) -> ExperimentBlueprint:
    drug = normalize_text(context.head)
    target = normalize_text(context.tail)
    relation = normalize_relation(context.relation)
    csv_text = csv_rows(
        [
            ("dose_um", "response"),
            (0.01, 0.95),
            (0.10, 0.82),
            (1.00, 0.51),
            (10.0, 0.19),
            (50.0, 0.08),
        ]
    )
    data_code = f'''#!/usr/bin/env python3
"""Synthetic dose-response table for {drug} against {target}."""

from io import StringIO

import pandas as pd


CSV_TEXT = """{csv_text}"""


def load_dose_response() -> pd.DataFrame:
    return pd.read_csv(StringIO(CSV_TEXT))
'''
    main_code = f'''#!/usr/bin/env python3
"""Estimate inhibitory response metrics for {drug} and {target}."""

import numpy as np

from data_en import load_dose_response


def compute_activity_drop(responses: np.ndarray) -> float:
    """
    Measure the fractional activity drop from the first to the last dose.
    """
    baseline = float(responses[0])
    final = float(responses[-1])
    return float((baseline - final) / max(baseline, 1e-8))


def estimate_ic50(doses: np.ndarray, responses: np.ndarray) -> float:
    """
    Estimate the IC50 by selecting the dose whose response is closest to 50% activity.
    """
    idx = int(np.argmin(np.abs(responses - 0.5)))
    return float(doses[idx])


def summarize_drug_response() -> dict[str, float]:
    table = load_dose_response()
    doses = table["dose_um"].to_numpy(dtype=float)
    responses = table["response"].to_numpy(dtype=float)
    activity_drop = compute_activity_drop(responses)
    ic50 = estimate_ic50(doses, responses)
    return {{
        "drug": "{drug}",
        "target": "{target}",
        "activity_drop": round(activity_drop, 4),
        "ic50_um": round(ic50, 4),
    }}


if __name__ == "__main__":
    print(summarize_drug_response())
'''
    return ExperimentBlueprint(
        name="dose_response",
        task_family="dose_response",
        relation=relation,
        direction=f"{drug.lower().replace(' ', '_')}_{relation}_{target.lower().replace(' ', '_')}",
        discipline="life",
        function_type="Numerical calculation",
        task_objective=f"Implement the core response metrics for a dose-response study involving {drug} and {target}.",
        research_focus=(
            f"The evidence suggests that {drug} is reported to {relation.replace('_', ' ')} {target}. "
            "Construct a compact experimental analysis task around inhibitory response estimation."
        ),
        data_code_template=data_code,
        main_code_template=main_code,
        incomplete_functions=("compute_activity_drop", "estimate_ic50"),
        hard_extra_blanks=("summarize_drug_response",),
        github_repo_query=f"{drug} {target} dose response python",
        github_code_query=f"{drug} {target} ic50 dose response",
        unit_tests=(
            {"name": "activity_drop", "input": {}, "expected_output": {"activity_drop": 0.9158}},
            {"name": "ic50", "input": {}, "expected_output": {"ic50_um": 1.0}},
        ),
    )


def _predicate(context: BlueprintContext) -> bool:
    if normalize_relation(context.relation) not in _RELATIONS:
        return False
    head_t = (context.head_type or "").strip().lower().replace(" ", "")
    tail_t = (context.tail_type or "").strip().lower().replace(" ", "")
    return head_t in _DRUG_HEAD_TYPES and tail_t in _TARGET_TAIL_TYPES


REGISTRY.register(name="dose_response", predicate=_predicate, factory=_factory, priority=30)
