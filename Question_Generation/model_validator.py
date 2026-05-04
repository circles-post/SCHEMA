from __future__ import annotations

import hashlib
import json
from typing import Any

from pubmed_graph.llm import InternChatClient

from .validation_cache import ValidationCache
from .validation_prompts import (
    build_essay_judge_messages,
    build_experiment_judge_messages,
    build_validation_messages,
)
from .validation_types import ModelValidationVerdict


def _fallback_verdict() -> ModelValidationVerdict:
    return ModelValidationVerdict(
        verdict="insufficient_evidence",
        confidence_band="low",
        issue_tags=["model_unavailable"],
        short_rationale="Model validation unavailable.",
    )


def judge_claim(
    question: str,
    answer: str,
    evidence_bundle: list[dict[str, Any]],
    model_config: dict[str, Any],
    cache: ValidationCache | None = None,
) -> ModelValidationVerdict:
    payload = {
        "question": question,
        "answer": answer,
        "evidence_bundle": evidence_bundle,
        "model": model_config.get("model", ""),
    }
    if cache:
        cached = cache.get(payload)
        if cached:
            return ModelValidationVerdict(**cached)
    enabled = bool(model_config.get("enabled", False))
    if not enabled or not model_config.get("api_key") or not model_config.get("base_url") or not model_config.get("model"):
        result = _fallback_verdict()
        if cache:
            cache.put(payload, result.__dict__)
        return result
    client = InternChatClient(model_config)
    messages = build_validation_messages(question=question, answer=answer, evidence_bundle=evidence_bundle)
    try:
        response = client.chat_json(
            messages,
            model=model_config.get("model"),
            temperature=float(model_config.get("temperature", 0.0) or 0.0),
            max_tokens=int(model_config.get("max_tokens", 800) or 800),
        )
    except Exception:
        result = _fallback_verdict()
        if cache:
            cache.put(payload, result.__dict__)
        return result
    try:
        result = ModelValidationVerdict(
            verdict=str(response.get("verdict", "insufficient_evidence")),
            confidence_band=str(response.get("confidence_band", "low")),
            support_score=float(response.get("support_score", 0.0) or 0.0),
            contradiction_score=float(response.get("contradiction_score", 0.0) or 0.0),
            supporting_evidence_ids=list(response.get("supporting_evidence_ids", []) or []),
            contradicting_evidence_ids=list(response.get("contradicting_evidence_ids", []) or []),
            issue_tags=list(response.get("issue_tags", []) or []),
            short_rationale=str(response.get("short_rationale", "")),
        )
    except Exception:
        result = _fallback_verdict()
    if cache:
        cache.put(payload, result.__dict__)
    return result


def judge_essay(
    question: str,
    reference_answer: str,
    evidence_bundle: list[dict[str, Any]],
    model_config: dict[str, Any],
    cache: ValidationCache | None = None,
) -> tuple[ModelValidationVerdict, float, str]:
    payload = {
        "type": "essay_judge",
        "question": question,
        "reference_answer": reference_answer,
        "evidence_bundle": evidence_bundle,
        "model": model_config.get("model", ""),
    }
    if cache:
        cached = cache.get(payload)
        if cached:
            verdict = ModelValidationVerdict(**{k: cached[k] for k in ModelValidationVerdict.__dataclass_fields__ if k in cached})
            return verdict, float(cached.get("essay_score", 0.0)), str(cached.get("essay_rationale", ""))
    enabled = bool(model_config.get("enabled", False))
    if not enabled or not model_config.get("api_key") or not model_config.get("base_url") or not model_config.get("model"):
        result = _fallback_verdict()
        if cache:
            cache.put(payload, {**result.__dict__, "essay_score": 0.0, "essay_rationale": ""})
        return result, 0.0, ""
    client = InternChatClient(model_config)
    messages = build_essay_judge_messages(question=question, reference_answer=reference_answer, evidence_bundle=evidence_bundle)
    try:
        response = client.chat_json(
            messages,
            model=model_config.get("model"),
            temperature=float(model_config.get("temperature", 0.0) or 0.0),
            max_tokens=int(model_config.get("max_tokens", 800) or 800),
        )
    except Exception:
        result = _fallback_verdict()
        if cache:
            cache.put(payload, {**result.__dict__, "essay_score": 0.0, "essay_rationale": ""})
        return result, 0.0, ""
    try:
        verdict_str = str(response.get("verdict", "insufficient_evidence"))
        essay_score = float(response.get("score", 0.0) or 0.0)
        essay_rationale = str(response.get("rationale", ""))
        issue_tags = list(response.get("issue_tags", []) or [])
        result = ModelValidationVerdict(
            verdict=verdict_str,
            confidence_band="high" if essay_score >= 0.7 else ("medium" if essay_score >= 0.4 else "low"),
            support_score=essay_score,
            issue_tags=issue_tags,
            short_rationale=essay_rationale,
        )
    except Exception:
        result = _fallback_verdict()
        essay_score = 0.0
        essay_rationale = ""
    if cache:
        cache.put(payload, {**result.__dict__, "essay_score": essay_score, "essay_rationale": essay_rationale})
    return result, essay_score, essay_rationale


def judge_experiment_code(
    *,
    scientific_claim: str,
    evidence_snippet: str,
    main_code: str,
    unit_tests: list[dict[str, Any]],
    incomplete_functions: list[str],
    evidence_bundle: list[dict[str, Any]],
    model_config: dict[str, Any],
    cache: ValidationCache | None = None,
) -> ModelValidationVerdict:
    """Second-opinion LLM judge on ``experiment_code`` samples.

    Sandbox already proved the reference solution compiles and passes its
    own unit tests — this judge checks the SEMANTIC alignment between the
    code and the scientific claim, which sandbox cannot see. Rejects with
    ``issue_tags`` like ``code_off_topic`` / ``direction_flipped`` /
    ``blank_too_trivial``.
    """
    # Include an evidence digest in the cache key. The judge explicitly
    # decides whether the code's direction / sign matches the evidence, so two
    # runs with identical code but different evidence must not share a cache
    # entry (otherwise the second verdict is stale).
    evidence_payload = {
        "evidence_snippet": evidence_snippet,
        "evidence_bundle": evidence_bundle,
    }
    evidence_digest = hashlib.sha1(
        json.dumps(evidence_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    payload = {
        "type": "experiment_judge",
        "scientific_claim": scientific_claim,
        # Only hash what affects the verdict — truncated so paraphrases hit cache.
        "main_code_head": (main_code or "")[:3500],
        "unit_tests": unit_tests,
        "incomplete_functions": list(incomplete_functions),
        "evidence_digest": evidence_digest,
        "model": model_config.get("model", ""),
    }
    if cache:
        cached = cache.get(payload)
        if cached:
            try:
                return ModelValidationVerdict(**{
                    k: cached[k]
                    for k in ModelValidationVerdict.__dataclass_fields__
                    if k in cached
                })
            except Exception:
                pass
    enabled = bool(model_config.get("enabled", False))
    if not enabled or not model_config.get("api_key") or not model_config.get("base_url") or not model_config.get("model"):
        result = _fallback_verdict()
        if cache:
            cache.put(payload, result.__dict__)
        return result
    client = InternChatClient(model_config)
    messages = build_experiment_judge_messages(
        scientific_claim=scientific_claim,
        evidence_snippet=evidence_snippet,
        main_code=main_code,
        unit_tests=unit_tests,
        incomplete_functions=incomplete_functions,
        evidence_bundle=evidence_bundle,
    )
    try:
        response = client.chat_json(
            messages,
            model=model_config.get("model"),
            temperature=float(model_config.get("temperature", 0.0) or 0.0),
            max_tokens=int(model_config.get("max_tokens", 900) or 900),
        )
    except Exception:
        result = _fallback_verdict()
        if cache:
            cache.put(payload, result.__dict__)
        return result
    try:
        result = ModelValidationVerdict(
            verdict=str(response.get("verdict", "insufficient_evidence")),
            confidence_band=str(response.get("confidence_band", "low")),
            support_score=float(response.get("support_score", 0.0) or 0.0),
            contradiction_score=float(response.get("contradiction_score", 0.0) or 0.0),
            supporting_evidence_ids=list(response.get("supporting_evidence_ids", []) or []),
            contradicting_evidence_ids=list(response.get("contradicting_evidence_ids", []) or []),
            issue_tags=list(response.get("issue_tags", []) or []),
            short_rationale=str(response.get("short_rationale", response.get("rationale", ""))),
        )
    except Exception:
        result = _fallback_verdict()
    if cache:
        cache.put(payload, result.__dict__)
    return result
