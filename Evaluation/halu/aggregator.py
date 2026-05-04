"""Aggregate per-sample HaluResult into overall + sliced summary metrics.

Phase-1 metrics (what the pilot needs):
  * ``HR`` (hallucination rate) = #refuted_claims / #total_claims
  * ``HS`` (hallucination severity) = mean(claim.score)  where score ∈ {0, 0.5, 1}
  * ``HF`` (hallucination flag)    = 1 if any claim is refuted else 0  (per sample → mean fraction overall)

Slices mirror ``evaluation.core.aggregate``:
  ``overall``, ``per_question_type``, ``per_tier``, ``per_corroboration_status``,
  ``per_evidence_strength``, ``per_tier_x_type``.

Phase-2 additions:
  * ``HS_weighted`` with concept-graph-connectivity weights (needs graph_kb.py).
  * ``by_concept_type`` slice.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import HaluResult


def compute_sample_metrics(result: HaluResult) -> None:
    """Mutate ``result`` in-place to fill n_*, HR, HS, HS_weighted, HF."""
    claims = result.judged_claims
    n = len(claims)
    n_ref = sum(1 for c in claims if c.verdict == "refuted")
    n_unv = sum(1 for c in claims if c.verdict == "unverifiable")
    n_sup = sum(1 for c in claims if c.verdict == "supported")
    result.n_claims = n
    result.n_refuted = n_ref
    result.n_unverifiable = n_unv
    result.n_supported = n_sup
    result.HR = (n_ref / n) if n else 0.0
    result.HS = (sum(c.score for c in claims) / n) if n else 0.0
    result.HF = 1 if n_ref > 0 else 0
    # HS_weighted: graph-connectivity-weighted severity.
    # Falls back to plain HS if no claim got a graph match (all weights == 1.0).
    w_sum = sum(c.concept_weight for c in claims)
    if w_sum > 0:
        result.HS_weighted = sum(c.score * c.concept_weight for c in claims) / w_sum
    else:
        result.HS_weighted = result.HS


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _empty_slice() -> dict[str, float]:
    return {
        "n_samples": 0,
        "n_claims": 0,
        "n_refuted": 0,
        "n_unverifiable": 0,
        "n_supported": 0,
        "HR_sum": 0.0,   # sum of per-sample HR (for mean-of-means report)
        "HS_sum": 0.0,
        "HS_weighted_sum": 0.0,
        "HF_sum": 0,
        "w_HR_numer": 0.0,   # claim-weighted — Σ refuted / Σ claims
        "w_HS_numer": 0.0,
        "cw_HS_numer": 0.0,  # concept-weighted score numerator
        "cw_HS_denom": 0.0,  # concept-weighted score denominator
    }


def _bump(slot: dict[str, float], r: HaluResult) -> None:
    slot["n_samples"] += 1
    slot["n_claims"] += r.n_claims
    slot["n_refuted"] += r.n_refuted
    slot["n_unverifiable"] += r.n_unverifiable
    slot["n_supported"] += r.n_supported
    slot["HR_sum"] += r.HR
    slot["HS_sum"] += r.HS
    slot["HS_weighted_sum"] += r.HS_weighted
    slot["HF_sum"] += r.HF
    slot["w_HR_numer"] += r.n_refuted
    slot["w_HS_numer"] += r.HS * r.n_claims
    for jc in r.judged_claims:
        slot["cw_HS_numer"] += jc.score * jc.concept_weight
        slot["cw_HS_denom"] += jc.concept_weight


def _finalize(slot: dict[str, float]) -> dict[str, Any]:
    n = int(slot["n_samples"])
    n_claims = int(slot["n_claims"])
    cw_denom = slot["cw_HS_denom"]
    return {
        "n_samples": n,
        "n_claims": n_claims,
        "n_refuted": int(slot["n_refuted"]),
        "n_unverifiable": int(slot["n_unverifiable"]),
        "n_supported": int(slot["n_supported"]),
        # Macro-averages: mean across samples (each sample weighted equally).
        "HR_macro": (slot["HR_sum"] / n) if n else 0.0,
        "HS_macro": (slot["HS_sum"] / n) if n else 0.0,
        "HS_weighted_macro": (slot["HS_weighted_sum"] / n) if n else 0.0,
        "HF_rate": (slot["HF_sum"] / n) if n else 0.0,
        # Micro-averages: claim-weighted.
        "HR_micro": (slot["w_HR_numer"] / n_claims) if n_claims else 0.0,
        "HS_micro": (slot["w_HS_numer"] / n_claims) if n_claims else 0.0,
        # Concept-weighted micro: each claim weighted by 1 + log(1+deg(concept)).
        "HS_weighted_micro": (slot["cw_HS_numer"] / cw_denom) if cw_denom else 0.0,
    }


def _bump_claim_only(slot: dict[str, float], jc, sample_claims_n: int) -> None:
    """Aggregate at claim-granularity (for by_concept_type slice — no sample-level fields)."""
    slot["n_claims"] += 1
    if jc.verdict == "refuted":
        slot["n_refuted"] += 1
    elif jc.verdict == "unverifiable":
        slot["n_unverifiable"] += 1
    elif jc.verdict == "supported":
        slot["n_supported"] += 1
    slot["w_HR_numer"] += 1 if jc.verdict == "refuted" else 0
    slot["w_HS_numer"] += jc.score
    slot["cw_HS_numer"] += jc.score * jc.concept_weight
    slot["cw_HS_denom"] += jc.concept_weight


def aggregate_halu(results: list[HaluResult]) -> dict[str, Any]:
    overall = _empty_slice()
    per_type: dict[str, dict] = defaultdict(_empty_slice)
    per_tier: dict[str, dict] = defaultdict(_empty_slice)
    per_corrob: dict[str, dict] = defaultdict(_empty_slice)
    per_strength: dict[str, dict] = defaultdict(_empty_slice)
    per_tier_x_type: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(_empty_slice))
    by_concept_type: dict[str, dict] = defaultdict(_empty_slice)

    for r in results:
        _bump(overall, r)
        _bump(per_type[r.question_type or "unknown"], r)
        _bump(per_tier[r.tier or "not_tagged"], r)
        _bump(per_corrob[r.corroboration_status or "not_requested"], r)
        _bump(per_strength[r.evidence_strength or "unknown"], r)
        _bump(per_tier_x_type[r.tier or "not_tagged"][r.question_type or "unknown"], r)
        for jc in r.judged_claims:
            _bump_claim_only(by_concept_type[jc.concept_type or "unknown"], jc, r.n_claims)

    # by_concept_type: claim-level only; drop per-sample macro keys by zeroing then
    # re-finalizing — the _finalize output keeps the meaningful micro metrics while
    # macro/HF/n_samples show 0 (no sample identity here).
    def _finalize_claim_only(slot: dict[str, float]) -> dict[str, Any]:
        out = _finalize(slot)
        for k in ("HR_macro", "HS_macro", "HS_weighted_macro", "HF_rate", "n_samples"):
            out[k] = 0.0 if k != "n_samples" else 0
        return out

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
        "by_concept_type": {k: _finalize_claim_only(v) for k, v in by_concept_type.items()},
    }
