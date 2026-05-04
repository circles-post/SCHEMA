"""score_one / score_many / aggregate — orchestration layer."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .scorers import (
    score_boolean,
    score_essay,
    score_experiment_code,
    score_multichoice,
    score_vqa,
)
from .types import EvalResult


_MULTICHOICE_TYPES = {"claim_choice", "one_hop_tail", "two_hop_tail"}
_DEFAULT_PASS_THRESHOLD = 0.5


def _load_samples(samples: list[dict] | str | Path) -> list[dict]:
    if isinstance(samples, (str, Path)):
        return [json.loads(line) for line in open(samples, encoding="utf-8")]
    return list(samples)


def _extract_tier(sample: dict) -> tuple[str, float]:
    bw = (sample.get("metadata") or {}).get("benchmark_weight") or {}
    return str(bw.get("tier") or "not_tagged"), float(bw.get("weight") or 1.0)


def _extract_corroboration(sample: dict) -> str:
    return str((sample.get("grounding") or {}).get("corroboration_status") or "not_requested")


def _extract_evidence_strength(sample: dict) -> str:
    return str((sample.get("grounding") or {}).get("evidence_strength") or "unknown")


def score_one(
    sample: dict[str, Any],
    model_answer: Any,
    *,
    judge_model_config: dict[str, Any] | None = None,
    sandbox_timeout: float | None = None,
    pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
) -> EvalResult:
    """Score a single sample against its ``model_answer``.

    ``model_answer`` shape per question_type:
      * multichoice (claim_choice / one_hop_tail / two_hop_tail):
            int index | str letter "A" | str full option text
      * boolean_support:   bool | str ("Supported" / "yes" / ...)
      * essay:             str (needs ``judge_model_config``)
      * experiment_code:   str (the completed main_code)
    """
    qtype = sample.get("question_type", "")
    sample_id = sample.get("sample_id", "")
    tier, weight = _extract_tier(sample)
    base = dict(
        sample_id=sample_id,
        question_type=qtype,
        tier=tier,
        weight=weight,
        corroboration_status=_extract_corroboration(sample),
        evidence_strength=_extract_evidence_strength(sample),
    )

    try:
        if qtype in _MULTICHOICE_TYPES:
            picked, expected, score, detail, err = score_multichoice(sample, model_answer)
        elif qtype == "boolean_support":
            picked, expected, score, detail, err = score_boolean(sample, model_answer)
        elif qtype == "essay":
            picked, expected, score, detail, err = score_essay(sample, model_answer, judge_model_config)
        elif qtype == "vqa":
            picked, expected, score, detail, err = score_vqa(sample, model_answer, judge_model_config)
        elif qtype == "experiment_code":
            picked, expected, score, detail, err = score_experiment_code(sample, model_answer, timeout=sandbox_timeout)
        else:
            return EvalResult(
                **base,
                score=0.0,
                is_correct=False,
                picked=model_answer,
                expected=None,
                error=f"unsupported_question_type:{qtype}",
                detail={},
            )
    except Exception as exc:
        return EvalResult(
            **base,
            score=0.0,
            is_correct=False,
            picked=model_answer,
            expected=None,
            error=f"scorer_exception:{type(exc).__name__}:{exc}",
            detail={},
        )

    score = max(0.0, min(1.0, float(score)))
    return EvalResult(
        **base,
        score=score,
        is_correct=(score >= pass_threshold and not err),
        picked=picked,
        expected=expected,
        error=err,
        detail=detail,
    )


def score_many(
    samples: list[dict] | str | Path,
    answers: dict[str, Any] | Iterable[tuple[str, Any]],
    *,
    judge_model_config: dict[str, Any] | None = None,
    sandbox_timeout: float | None = None,
    pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
    skip_missing: bool = True,
) -> list[EvalResult]:
    """Bulk-score samples. ``samples`` may be a path or a list.

    ``answers`` maps ``sample_id -> model_answer``. Samples with no answer
    are skipped (``skip_missing=True``) or scored as ``missing_answer``
    (``skip_missing=False``).
    """
    sample_list = _load_samples(samples)
    if isinstance(answers, dict):
        answer_map = answers
    else:
        answer_map = dict(answers)

    results: list[EvalResult] = []
    for sample in sample_list:
        sid = sample.get("sample_id", "")
        if sid not in answer_map:
            if skip_missing:
                continue
            tier, weight = _extract_tier(sample)
            results.append(
                EvalResult(
                    sample_id=sid,
                    question_type=sample.get("question_type", ""),
                    tier=tier,
                    weight=weight,
                    corroboration_status=_extract_corroboration(sample),
                    evidence_strength=_extract_evidence_strength(sample),
                    error="missing_answer",
                )
            )
            continue
        results.append(
            score_one(
                sample,
                answer_map[sid],
                judge_model_config=judge_model_config,
                sandbox_timeout=sandbox_timeout,
                pass_threshold=pass_threshold,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _empty_slice() -> dict[str, float]:
    return {"n": 0, "sum_score": 0.0, "sum_weighted_score": 0.0, "total_weight": 0.0, "errors": 0}


def _bump(slot: dict[str, float], r: EvalResult) -> None:
    slot["n"] += 1
    slot["sum_score"] += r.score
    slot["sum_weighted_score"] += r.score * r.weight
    slot["total_weight"] += r.weight
    if r.error:
        slot["errors"] += 1


def _finalize(slot: dict[str, float]) -> dict[str, Any]:
    n = int(slot["n"])
    total_w = slot["total_weight"]
    return {
        "n": n,
        "acc": (slot["sum_score"] / n) if n else 0.0,
        "weighted_acc": (slot["sum_weighted_score"] / total_w) if total_w else 0.0,
        "total_weight": round(total_w, 3),
        "errors": int(slot["errors"]),
    }


def aggregate(results: list[EvalResult]) -> dict[str, Any]:
    """Return a nested dict of aggregate statistics.

    Shape::

        {
          "overall":                  {n, acc, weighted_acc, total_weight, errors},
          "per_question_type":        {qt: {...}, ...},
          "per_tier":                 {tier: {...}, ...},
          "per_corroboration_status": {status: {...}, ...},
          "per_evidence_strength":    {strength: {...}, ...},
          "per_tier_x_type":          {tier: {qt: {...}, ...}, ...},
        }
    """
    overall = _empty_slice()
    per_type: dict[str, dict] = defaultdict(_empty_slice)
    per_tier: dict[str, dict] = defaultdict(_empty_slice)
    per_corrob: dict[str, dict] = defaultdict(_empty_slice)
    per_strength: dict[str, dict] = defaultdict(_empty_slice)
    per_tier_x_type: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(_empty_slice))

    for r in results:
        _bump(overall, r)
        _bump(per_type[r.question_type], r)
        _bump(per_tier[r.tier], r)
        _bump(per_corrob[r.corroboration_status], r)
        _bump(per_strength[r.evidence_strength], r)
        _bump(per_tier_x_type[r.tier][r.question_type], r)

    return {
        "overall": _finalize(overall),
        "per_question_type": {k: _finalize(v) for k, v in per_type.items()},
        "per_tier": {k: _finalize(v) for k, v in per_tier.items()},
        "per_corroboration_status": {k: _finalize(v) for k, v in per_corrob.items()},
        "per_evidence_strength": {k: _finalize(v) for k, v in per_strength.items()},
        "per_tier_x_type": {
            tier: {qt: _finalize(v) for qt, v in qts.items()}
            for tier, qts in per_tier_x_type.items()
        },
    }


class Evaluator:
    """Convenience wrapper bundling samples + scoring + aggregation."""

    def __init__(
        self,
        samples: list[dict] | str | Path,
        *,
        judge_model_config: dict[str, Any] | None = None,
        sandbox_timeout: float | None = None,
        pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
    ) -> None:
        self.samples = _load_samples(samples)
        self._by_id = {s.get("sample_id", ""): s for s in self.samples}
        self.judge_model_config = judge_model_config
        self.sandbox_timeout = sandbox_timeout
        self.pass_threshold = pass_threshold

    def score_one(self, sample_id: str, model_answer: Any) -> EvalResult:
        sample = self._by_id.get(sample_id)
        if sample is None:
            return EvalResult(sample_id=sample_id, question_type="", error="sample_not_found")
        return score_one(
            sample,
            model_answer,
            judge_model_config=self.judge_model_config,
            sandbox_timeout=self.sandbox_timeout,
            pass_threshold=self.pass_threshold,
        )

    def score_many(
        self,
        answers: dict[str, Any],
        *,
        skip_missing: bool = True,
    ) -> list[EvalResult]:
        return score_many(
            self.samples,
            answers,
            judge_model_config=self.judge_model_config,
            sandbox_timeout=self.sandbox_timeout,
            pass_threshold=self.pass_threshold,
            skip_missing=skip_missing,
        )

    @staticmethod
    def aggregate(results: list[EvalResult]) -> dict[str, Any]:
        return aggregate(results)
