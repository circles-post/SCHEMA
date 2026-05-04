from __future__ import annotations

VALIDATION_SYSTEM_PROMPT = """You are validating whether a scientific benchmark item is supported by evidence.\nReturn JSON only.\nUse only the provided evidence bundle.\nDo not use outside knowledge.\nAllowed verdicts: supported, insufficient_evidence, contradicted.\nIf evidence is weak, partial, or only loosely related, choose insufficient_evidence.\n"""

ESSAY_JUDGE_SYSTEM_PROMPT = """You are a scientific expert evaluating whether a reference answer to an open-ended scientific question is accurate and well-supported by the provided evidence.

Evaluate the reference answer on these criteria:
1. Scientific accuracy: Is the answer factually correct based on the evidence?
2. Evidence support: Is every claim in the answer backed by the provided evidence?
3. Completeness: Does the answer adequately address the question?

Return JSON only with these keys:
- verdict: "supported" | "insufficient_evidence" | "contradicted"
- score: float 0.0-1.0 (overall quality score)
- rationale: brief explanation of your judgment
- issue_tags: list of any issues found (e.g. "unsupported_claim", "missing_relevant_evidence", "overstated")
"""


def build_validation_messages(question: str, answer: str, evidence_bundle: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Validate this benchmark item.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Evidence bundle:\n"
                f"{evidence_bundle}\n\n"
                "Return JSON with keys: verdict, confidence_band, support_score, contradiction_score, "
                "supporting_evidence_ids, contradicting_evidence_ids, issue_tags, short_rationale."
            ),
        },
    ]


EXPERIMENT_JUDGE_SYSTEM_PROMPT = """You are a computational biologist reviewing whether a programming exercise actually operationalizes a specific scientific claim.

The sandbox has already verified that the reference code passes its own unit tests and the masked version fails them. You do NOT need to re-check math correctness. Your job is to judge the SCIENTIFIC SEMANTICS:

1. Does the reference code genuinely model/measure the phenomenon in the claim?
   e.g. for "OTUB2 inhibits ischemic stroke" the code should compare
   wild-type vs knockout infarct volume, NOT pathway enrichment or IC50.
2. Are the unit test ``expected_output`` values scientifically reasonable given the synthetic data?
3. Does blanking out the ``incomplete_functions`` actually test the key conceptual knowledge the claim implies?
4. Is the evidence in the bundle consistent with the direction/sign the code encodes?

Return JSON ONLY with keys:
- verdict: "supported" | "insufficient_evidence" | "contradicted"
- confidence_band: "low" | "medium" | "high"
- support_score: float 0.0-1.0
- contradiction_score: float 0.0-1.0
- short_rationale: 1-2 sentences
- issue_tags: list of short tags like "code_off_topic", "expected_output_unphysical", "blank_too_trivial", "direction_flipped"
- supporting_evidence_ids: list of evidence ids you found supportive
- contradicting_evidence_ids: list of evidence ids you found contradictory

Be strict. Choose "insufficient_evidence" when the code is generic/boilerplate and not specific to the claim. Choose "contradicted" only when the code encodes the OPPOSITE direction (e.g. code says ``supported = fold < 1`` but the claim is an inhibition).
"""


def build_experiment_judge_messages(
    *,
    scientific_claim: str,
    evidence_snippet: str,
    main_code: str,
    unit_tests: list[dict],
    incomplete_functions: list[str],
    evidence_bundle: list[dict],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EXPERIMENT_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate this experiment_code benchmark item.\n\n"
                f"Scientific claim: {scientific_claim}\n"
                f"Evidence snippet: {evidence_snippet[:800]}\n\n"
                f"Blanked functions (what the student must implement): {incomplete_functions}\n\n"
                "Reference main_code (truncated):\n"
                "```python\n"
                f"{main_code[:3500]}\n"
                "```\n\n"
                f"Unit tests: {unit_tests}\n\n"
                "Extra evidence bundle (may include additional literature hits):\n"
                f"{evidence_bundle}\n\n"
                "Return the JSON described in the system prompt, nothing else."
            ),
        },
    ]


def build_essay_judge_messages(question: str, reference_answer: str, evidence_bundle: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ESSAY_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate this scientific essay question and its reference answer.\n\n"
                f"Question: {question}\n\n"
                f"Reference Answer: {reference_answer}\n\n"
                "Evidence bundle:\n"
                f"{evidence_bundle}\n\n"
                "Return JSON with keys: verdict, score, rationale, issue_tags."
            ),
        },
    ]
