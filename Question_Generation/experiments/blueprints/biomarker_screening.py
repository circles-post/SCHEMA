from __future__ import annotations

from pubmed_graph.utils import normalize_text

from .._helpers import csv_rows
from ..registry import REGISTRY, BlueprintContext, ExperimentBlueprint, normalize_relation


def _factory(context: BlueprintContext) -> ExperimentBlueprint:
    biomarker = normalize_text(context.head)
    disease = normalize_text(context.tail)
    csv_text = csv_rows(
        [
            ("patient_id", "risk_score", "label"),
            ("P001", 0.91, 1),
            ("P002", 0.83, 1),
            ("P003", 0.71, 1),
            ("P004", 0.64, 1),
            ("P005", 0.40, 0),
            ("P006", 0.28, 0),
            ("P007", 0.19, 0),
            ("P008", 0.58, 1),
            ("P009", 0.35, 0),
        ]
    )
    data_code = f'''#!/usr/bin/env python3
"""Synthetic screening cohort for {biomarker} and {disease}."""

from io import StringIO

import pandas as pd


CSV_TEXT = """{csv_text}"""


def load_screening_cohort() -> pd.DataFrame:
    """Load a small biomarker screening cohort."""
    return pd.read_csv(StringIO(CSV_TEXT))
'''
    main_code = f'''#!/usr/bin/env python3
"""Evaluate biomarker screening performance for {biomarker} in {disease}."""

import numpy as np
from sklearn.metrics import confusion_matrix

from data_en import load_screening_cohort


def calculate_sensitivity_specificity(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> tuple[float, float]:
    """
    Compute sensitivity and specificity for a binary screening threshold.
    """
    epsilon = 1e-8
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    return float(sensitivity), float(specificity)


def calculate_ppv(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
    """
    Compute positive predictive value for the same screening threshold.
    """
    epsilon = 1e-8
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tp / (tp + fp + epsilon))


def summarize_screening_performance(threshold: float = 0.6) -> dict[str, float]:
    cohort = load_screening_cohort()
    y_true = cohort["label"].to_numpy(dtype=int)
    y_scores = cohort["risk_score"].to_numpy(dtype=float)
    sensitivity, specificity = calculate_sensitivity_specificity(y_true, y_scores, threshold)
    ppv = calculate_ppv(y_true, y_scores, threshold)
    prevalence = float(np.mean(y_true))
    return {{
        "threshold": float(threshold),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "prevalence": round(prevalence, 4),
    }}


if __name__ == "__main__":
    print(summarize_screening_performance())
'''
    return ExperimentBlueprint(
        name="biomarker_screening",
        task_family="biomarker_screening",
        relation="associated_with",
        direction=f"{disease.lower().replace(' ', '_')}_biomarker_screening",
        discipline="life",
        function_type="Metric calculation",
        task_objective=f"Implement the screening metrics used to evaluate whether {biomarker} can stratify {disease} risk.",
        research_focus=(
            f"The evidence links {biomarker} to {disease}. Build the missing metric functions for a "
            "minimal early-screening analysis so the agent reasons about model operating points instead of "
            "answering the claim directly."
        ),
        data_code_template=data_code,
        main_code_template=main_code,
        incomplete_functions=("calculate_sensitivity_specificity", "calculate_ppv"),
        hard_extra_blanks=("summarize_screening_performance",),
        github_repo_query=f"{disease} biomarker screening python",
        github_code_query=f"{biomarker} {disease} sensitivity specificity ppv",
        unit_tests=(
            # Values verified by sandbox self-test against the embedded CSV
            # (5 positive / 4 negative; perfect separation at threshold 0.6).
            {"name": "threshold_0_60", "input": {"threshold": 0.6}, "expected_output": {"sensitivity": 0.8, "specificity": 1.0, "ppv": 1.0}},
            {"name": "threshold_0_50", "input": {"threshold": 0.5}, "expected_output": {"sensitivity": 1.0, "specificity": 1.0, "ppv": 1.0}},
            {"name": "prevalence_check", "input": {"threshold": 0.6}, "expected_output": {"prevalence": 0.5556}},
        ),
    )


_BIOMARKER_HEAD_TYPES = {"biomarker", "gene", "protein", "metabolite", "marker"}
_BIOMARKER_TAIL_TYPES = {"disease", "condition", "syndrome", "cancer", "tumor", "phenotype"}


def _predicate(context: BlueprintContext) -> bool:
    if normalize_relation(context.relation) != "associated_with":
        return False
    head_t = (context.head_type or "").strip().lower().replace(" ", "")
    tail_t = (context.tail_type or "").strip().lower().replace(" ", "")
    return head_t in _BIOMARKER_HEAD_TYPES and tail_t in _BIOMARKER_TAIL_TYPES


REGISTRY.register(name="biomarker_screening", predicate=_predicate, factory=_factory, priority=10)
