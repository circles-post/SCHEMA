from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import time
from typing import Any, Dict

from external_agent.agdebugger_adapter import load_turns_from_agdebugger_state
from external_agent.claim_extractor import ClaimExtractor
from external_agent.evidence_provider import WebSearchEvidenceProvider
from external_agent.judge import ClaimJudge, PlannerJudge
from external_agent.llm import OpenAICompatibleLLM
from external_agent.pipeline import ClaimJudgePipeline
from external_agent.schemas import EvidenceBundle, Judgment, PipelineResult
from external_agent.strategies import get_strategy, PLANNER_JUDGE_SYSTEM_PROMPT, build_planner_judge_user_prompt


def _analysis_detail_log_path() -> str:
    return os.environ.get("AGDEBUGGER_ANALYSIS_DETAIL_LOG", "").strip()


def _log_analysis_detail(event: str, **data: Any) -> None:
    path = _analysis_detail_log_path()
    if not path:
        return
    entry = {
        "event": event,
        "ts": datetime.datetime.now().isoformat(),
        **data,
    }
    log_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _claim_summary(claim) -> Dict[str, Any]:
    metadata = claim.metadata if isinstance(claim.metadata, dict) else {}
    data = claim.data if isinstance(getattr(claim, "data", None), dict) else {}
    return {
        "claim_id": claim.claim_id,
        "conversation_id": claim.conversation_id,
        "turn_number": claim.turn_number,
        "category": claim.category,
        "source_ref": claim.source_ref,
        "source_type": claim.source_type,
        "text": claim.text,
        "original_statement": getattr(claim, "original_statement", ""),
        "context_snippet": data.get("context_snippet", ""),
        "source_timestamp": metadata.get("source_timestamp"),
        "analysis_assistant_turn": metadata.get("analysis_assistant_turn"),
        "source_message_type": metadata.get("source_message_type"),
    }


def _summarize_result(result) -> Dict[str, Any]:
    # Surface every extracted claim/judgment so the planner_judge and the
    # downstream concept_repair builder can see the full set. Previously this
    # was capped at the first 8, silently dropping the long tail when the
    # extractor produced more (default AGDEBUGGER_CONCEPT_MAX_TOTAL_CLAIMS=24).
    claim_summaries = [_claim_summary(claim) for claim in result.claims]

    judgment_summaries = []
    for judgment in result.judgments:
        judgment_summaries.append(
            {
                "claim_id": judgment.claim_id,
                "turn_number": judgment.turn_number,
                "reference_name": judgment.reference_name,
                "reference_grounding": judgment.reference_grounding,
                "content_grounding": judgment.content_grounding,
                "hallucination": judgment.hallucination,
                "verification_error": judgment.verification_error,
                "concept_true_understanding": judgment.concept_true_understanding,
                "reason": judgment.reason,
            }
        )

    return {
        "conversation_id": result.conversation_id,
        "claim_count": len(result.claims),
        "judgment_count": len(result.judgments),
        "hallucination_yes_count": sum(
            1 for judgment in result.judgments if judgment.hallucination.strip().lower() == "yes"
        ),
        "verification_error_yes_count": sum(
            1 for judgment in result.judgments if judgment.verification_error.strip().lower() == "yes"
        ),
        "claims": claim_summaries,
        "judgments": judgment_summaries,
    }


async def _run_planner_judge(*, llm: OpenAICompatibleLLM, question_text: str, current_answer: str, summary: Dict[str, Any], failed_claim_ids: list[str] | None = None) -> Dict[str, Any]:
    planner_judge = PlannerJudge(llm=llm, system_prompt=PLANNER_JUDGE_SYSTEM_PROMPT)
    payload = await planner_judge.judge_plan(
        build_planner_judge_user_prompt(
            question_text=question_text,
            current_answer=current_answer,
            analysis_summary=summary,
            failed_claim_ids=failed_claim_ids or [],
        )
    )
    return {
        "selected_claim_id": payload.selected_claim_id,
        "decision": payload.decision,
        "selected_claim_reason": payload.selected_claim_reason,
        "answer_grounding_status": payload.answer_grounding_status,
        "mapping_status": payload.mapping_status,
        "alignment_status": payload.alignment_status,
        "confidence": payload.confidence,
        "repair_concept_name": payload.repair_concept_name,
        "incorrect_understanding": payload.incorrect_understanding,
        "correct_understanding": payload.correct_understanding,
        "target_turn_number": payload.target_turn_number,
        "raw_response": payload.raw_response,
    }


def _turn_summaries(turns) -> list[Dict[str, Any]]:
    summaries = []
    for turn in turns:
        summaries.append(
            {
                "turn_number": turn.turn_number,
                "role": turn.role,
                "content_length": len(turn.content),
                "content_preview": " ".join(turn.content.split())[:240],
            }
        )
    return summaries


def _make_verification_error_judgment(
    claim,
    *,
    reason: str,
    reference_name: str | None = None,
) -> Judgment:
    return Judgment(
        claim_id=claim.claim_id,
        conversation_id=claim.conversation_id,
        turn_number=claim.turn_number,
        reference_name=reference_name if reference_name is not None else claim.source_ref,
        reference_grounding="",
        content_grounding="N/A",
        hallucination="N/A",
        abstention="Yes",
        verification_error="Yes",
        concept_true_understanding="",
        reason=reason,
        raw_response={"reason": reason, "verification_error": "Yes"},
    )


def _make_hallucination_judgment(
    claim,
    *,
    reason: str,
    concept_true_understanding: str,
    reference_name: str | None = None,
    reference_grounding: str = "Deterministic subject-entity compatibility pre-check",
) -> Judgment:
    return Judgment(
        claim_id=claim.claim_id,
        conversation_id=claim.conversation_id,
        turn_number=claim.turn_number,
        reference_name=reference_name if reference_name is not None else claim.source_ref,
        reference_grounding=reference_grounding,
        content_grounding="False",
        hallucination="Yes",
        abstention="No",
        verification_error="No",
        concept_true_understanding=concept_true_understanding,
        reason=reason,
        raw_response={
            "reason": reason,
            "hallucination": "Yes",
            "verification_error": "No",
            "concept_true_understanding": concept_true_understanding,
            "reference_grounding": reference_grounding,
        },
    )


_NUCLEIC_ACID_SUBJECT_RE = re.compile(
    r"\b(rna|dna|aptamer|riboswitch|nucleic acid|nucleotide|oligonucleotide|mrna|trna|rrna)\b",
    re.IGNORECASE,
)
_PROTEIN_SUBJECT_RE = re.compile(
    r"\b(protein|enzyme|peptide|polypeptide|antibody|receptor|kinase|transporter)\b",
    re.IGNORECASE,
)
_PROTEIN_COMPONENT_RE = re.compile(
    r"\b(amino acid|amino acids|residue|residues|side chain|side chains|peptide backbone)\b",
    re.IGNORECASE,
)
_NUCLEIC_ACID_COMPONENT_RE = re.compile(
    r"\b(nucleotide|nucleotides|nucleobase|nucleobases|base pair|base pairs|phosphodiester)\b",
    re.IGNORECASE,
)


def _infer_subject_domain(question_text: str) -> str | None:
    if _NUCLEIC_ACID_SUBJECT_RE.search(question_text):
        return "nucleic_acid"
    if _PROTEIN_SUBJECT_RE.search(question_text):
        return "protein"
    return None


def _infer_claim_entity_domain(claim_text: str) -> str | None:
    if _PROTEIN_COMPONENT_RE.search(claim_text):
        return "protein_component"
    if _NUCLEIC_ACID_COMPONENT_RE.search(claim_text):
        return "nucleic_acid_component"
    return None


def _deterministic_precheck_enabled() -> bool:
    value = os.environ.get("AGDEBUGGER_ENABLE_DETERMINISTIC_PRECHECK", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _run_deterministic_claim_precheck(claim, *, question_text: str = "") -> Judgment | None:
    if not _deterministic_precheck_enabled():
        return None
    if not question_text.strip():
        return None
    if claim.category != "constraint_claim" and claim.source_type != "entity_compatibility":
        return None

    subject_domain = _infer_subject_domain(question_text)
    claim_text = " ".join(part for part in (claim.source_ref, claim.text) if part).strip()
    entity_domain = _infer_claim_entity_domain(claim_text)

    if subject_domain == "nucleic_acid" and entity_domain == "protein_component":
        return _make_hallucination_judgment(
            claim,
            reason=(
                "Deterministic compatibility check failed: the question subject is a nucleic-acid system, "
                "but the claim grounds it in protein-specific components such as amino acids or residues."
            ),
            concept_true_understanding=(
                "A nucleic-acid subject such as RNA or DNA is composed of nucleotides rather than amino acids; "
                "any answer grounding must stay within nucleic-acid entities unless an external binding partner is explicitly stated."
            ),
        )

    if subject_domain == "protein" and entity_domain == "nucleic_acid_component":
        return _make_hallucination_judgment(
            claim,
            reason=(
                "Deterministic compatibility check failed: the question subject is a protein system, "
                "but the claim treats nucleic-acid-specific components as intrinsic parts of that subject."
            ),
            concept_true_understanding=(
                "A protein subject is built from amino-acid residues rather than nucleotides; "
                "any answer grounding must respect that entity-type distinction unless nucleic acids are explicitly introduced as separate molecules."
            ),
        )

    return None


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError))


async def analyze_text(
    *,
    task: str,
    text: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    evidence_text: str = "",
    use_websearch: bool = False,
    search_backend: str = "bright_data",
    search_max_searches: int = 3,
    search_num_results: int = 5,
    search_fetch_top_n: int = 2,
    search_max_output_words: int = 1500,
    conversation_id: int = 0,
    turn_number: int = 0,
) -> Dict[str, Any]:
    strategy = get_strategy(task)
    _log_analysis_detail(
        "analysis_text_start",
        task=task,
        model=model,
        conversation_id=conversation_id,
        turn_number=turn_number,
        use_websearch=use_websearch,
        text_preview=" ".join(text.split())[:400],
    )
    llm = OpenAICompatibleLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    extractor = ClaimExtractor(llm=llm, strategy=strategy)
    judge = ClaimJudge(llm=llm, strategy=strategy)
    pipeline = ClaimJudgePipeline(extractor=extractor, judge=judge)
    try:
        if use_websearch:
            async with WebSearchEvidenceProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                backend=search_backend,
                max_searches=search_max_searches,
                num_results=search_num_results,
                fetch_top_n=search_fetch_top_n,
                max_output_words=search_max_output_words,
            ) as provider:
                result = await pipeline.run_on_text(
                    text,
                    conversation_id=conversation_id,
                    turn_number=turn_number,
                    evidence=None,
                    evidence_provider=lambda claim: provider.build_evidence(claim, strategy),
                )
        else:
            evidence = EvidenceBundle(
                search_results=evidence_text,
                filtered_content=evidence_text,
            )
            result = await pipeline.run_on_text(
                text,
                conversation_id=conversation_id,
                turn_number=turn_number,
                evidence=evidence,
            )
        summary = _summarize_result(result)
        _log_analysis_detail(
            "analysis_text_result",
            task=task,
            model=model,
            conversation_id=conversation_id,
            turn_number=turn_number,
            use_websearch=use_websearch,
            summary=summary,
        )
        return summary
    finally:
        await llm.close()


async def analyze_session_state(
    *,
    task: str,
    state: Dict[str, Any],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    evidence_text: str = "",
    use_websearch: bool = False,
    search_backend: str = "bright_data",
    search_max_searches: int = 3,
    search_num_results: int = 5,
    search_fetch_top_n: int = 2,
    search_max_output_words: int = 1500,
    conversation_id: int = 0,
    session_id: int | None = None,
    assistant_only: bool = True,
    timeout_sec: float | None = None,
    question_text: str = "",
    current_answer: str = "",
) -> Dict[str, Any]:
    strategy = get_strategy(task)
    llm = OpenAICompatibleLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    extractor = ClaimExtractor(llm=llm, strategy=strategy)
    judge_kwargs: Dict[str, Any] = {}
    if question_text:
        judge_kwargs["question_text"] = question_text
    judge = ClaimJudge(llm=llm, strategy=strategy, **judge_kwargs)
    pipeline = ClaimJudgePipeline(extractor=extractor, judge=judge)
    turns = load_turns_from_agdebugger_state(
        state,
        session_id=session_id,
        assistant_only=assistant_only,
    )
    _log_analysis_detail(
        "analysis_session_start",
        task=task,
        model=model,
        conversation_id=conversation_id,
        session_id=session_id,
        use_websearch=use_websearch,
        assistant_only=assistant_only,
        analyzed_turn_count=len(turns),
        analyzed_turns=_turn_summaries(turns),
    )
    start_t0 = time.perf_counter()

    def _remaining_time() -> float | None:
        if timeout_sec is None:
            return None
        remaining = timeout_sec - (time.perf_counter() - start_t0)
        return max(0.0, remaining)

    def _per_claim_timeout(remaining: float | None) -> float | None:
        configured = float(os.environ.get("AGDEBUGGER_ANALYSIS_PER_CLAIM_TIMEOUT_SEC", "120"))
        if remaining is None:
            return configured
        return max(1.0, min(configured, remaining))

    async def _await_with_remaining(awaitable):
        remaining = _remaining_time()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("analysis overall deadline reached")
        if remaining is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=remaining)

    partial_reason: str | None = None
    claim_concurrency = max(1, int(os.environ.get("AGDEBUGGER_ANALYSIS_CLAIM_CONCURRENCY", "1")))
    try:
        if use_websearch:
            async with WebSearchEvidenceProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                backend=search_backend,
                max_searches=search_max_searches,
                num_results=search_num_results,
                fetch_top_n=search_fetch_top_n,
                max_output_words=search_max_output_words,
            ) as provider:
                extract_t0 = time.perf_counter()
                _log_analysis_detail(
                    "extract_start",
                    task=task,
                    model=model,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    analyzed_turn_count=len(turns),
                )
                try:
                    claims = await _await_with_remaining(
                        extractor.extract_from_conversation(
                            turns,
                            conversation_id=conversation_id,
                            assistant_only=assistant_only,
                        )
                    )
                except TimeoutError:
                    claims = []
                    partial_reason = "extract_timeout"
                    _log_analysis_detail(
                        "extract_error",
                        task=task,
                        model=model,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        error_type="TimeoutError",
                        error="analysis overall deadline reached during extract",
                    )
                extract_elapsed_sec = time.perf_counter() - extract_t0
                _log_analysis_detail(
                    "extract_result",
                    task=task,
                    model=model,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    elapsed_sec=extract_elapsed_sec,
                    claim_count=len(claims),
                    claims=[_claim_summary(claim) for claim in claims],
                )
                judge_t0 = time.perf_counter()
                judgments = []

                semaphore = asyncio.Semaphore(claim_concurrency)

                async def _process_claim(claim):
                    nonlocal partial_reason
                    async with semaphore:
                        remaining = _remaining_time()
                        if remaining is not None and remaining <= 0:
                            partial_reason = partial_reason or "judge_timeout"
                            return _make_verification_error_judgment(
                                claim,
                                reason="Skipped because analysis timed out before this concept could be judged.",
                            )
                        precheck_judgment = _run_deterministic_claim_precheck(claim, question_text=question_text)
                        if precheck_judgment is not None:
                            _log_analysis_detail(
                                "precheck_result",
                                task=task,
                                model=model,
                                conversation_id=conversation_id,
                                session_id=session_id,
                                claim=_claim_summary(claim),
                                hallucination=precheck_judgment.hallucination,
                                verification_error=precheck_judgment.verification_error,
                                concept_true_understanding=precheck_judgment.concept_true_understanding,
                                reason=precheck_judgment.reason,
                            )
                            return precheck_judgment
                        _log_analysis_detail(
                            "evidence_start",
                            task=task,
                            model=model,
                            conversation_id=conversation_id,
                            session_id=session_id,
                            claim=_claim_summary(claim),
                        )
                        evidence_t0 = time.perf_counter()
                        try:
                            evidence = await asyncio.wait_for(
                                provider.build_evidence(claim, strategy),
                                timeout=_per_claim_timeout(remaining),
                            )
                        except Exception as exc:
                            _log_analysis_detail(
                                "evidence_error",
                                task=task,
                                model=model,
                                conversation_id=conversation_id,
                                session_id=session_id,
                                claim=_claim_summary(claim),
                                elapsed_sec=time.perf_counter() - evidence_t0,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                            if _is_timeout_exception(exc):
                                partial_reason = partial_reason or "evidence_timeout"
                                return _make_verification_error_judgment(
                                    claim,
                                    reason="Evidence gathering timed out for this concept.",
                                )
                            raise
                        evidence_elapsed_sec = time.perf_counter() - evidence_t0
                        _log_analysis_detail(
                            "evidence_result",
                            task=task,
                            model=model,
                            conversation_id=conversation_id,
                            session_id=session_id,
                            claim=_claim_summary(claim),
                            elapsed_sec=evidence_elapsed_sec,
                            search_results_chars=len(evidence.search_results or ""),
                            filtered_content_chars=len(evidence.filtered_content or ""),
                            evidence_metadata=evidence.metadata,
                            literature_mode=(evidence.metadata or {}).get("literature_mode") if isinstance(evidence.metadata, dict) else None,
                            literature_used=(evidence.metadata or {}).get("literature_used") if isinstance(evidence.metadata, dict) else None,
                            urls_fetched=(evidence.metadata or {}).get("urls_fetched") if isinstance(evidence.metadata, dict) else None,
                            search_preview=(evidence.search_results or "")[:500],
                            filtered_preview=(evidence.filtered_content or "")[:500],
                        )
                        evidence_metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
                        if evidence_metadata.get("evidence_insufficient"):
                            reason = str(
                                evidence_metadata.get("evidence_insufficient_reason")
                                or "Evidence gathering returned no relevant support for this concept."
                            ).strip()
                            judgment = _make_verification_error_judgment(claim, reason=reason)
                            _log_analysis_detail(
                                "judge_result",
                                task=task,
                                model=model,
                                conversation_id=conversation_id,
                                session_id=session_id,
                                claim=_claim_summary(claim),
                                elapsed_sec=0.0,
                                hallucination=judgment.hallucination,
                                verification_error=judgment.verification_error,
                                concept_true_understanding=judgment.concept_true_understanding,
                                reason=judgment.reason,
                                skipped_due_to_evidence_insufficient=True,
                            )
                            return judgment
                        _log_analysis_detail(
                            "judge_start",
                            task=task,
                            model=model,
                            conversation_id=conversation_id,
                            session_id=session_id,
                            claim=_claim_summary(claim),
                        )
                        claim_judge_t0 = time.perf_counter()
                        try:
                            judgment = await asyncio.wait_for(
                                judge.judge_claim(claim, evidence),
                                timeout=_per_claim_timeout(_remaining_time()),
                            )
                        except Exception as exc:
                            _log_analysis_detail(
                                "judge_error",
                                task=task,
                                model=model,
                                conversation_id=conversation_id,
                                session_id=session_id,
                                claim=_claim_summary(claim),
                                elapsed_sec=time.perf_counter() - claim_judge_t0,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                            if _is_timeout_exception(exc):
                                partial_reason = partial_reason or "judge_timeout"
                                return _make_verification_error_judgment(
                                    claim,
                                    reason="Judgment timed out for this concept.",
                                )
                            raise
                        claim_judge_elapsed_sec = time.perf_counter() - claim_judge_t0
                        _log_analysis_detail(
                            "judge_result",
                            task=task,
                            model=model,
                            conversation_id=conversation_id,
                            session_id=session_id,
                            claim=_claim_summary(claim),
                            elapsed_sec=claim_judge_elapsed_sec,
                            hallucination=judgment.hallucination,
                            verification_error=judgment.verification_error,
                            concept_true_understanding=judgment.concept_true_understanding,
                            reason=judgment.reason,
                        )
                        return judgment

                judgments = list(await asyncio.gather(*[_process_claim(claim) for claim in claims]))
                judge_elapsed_sec = time.perf_counter() - judge_t0
        else:
            evidence = EvidenceBundle(search_results=evidence_text, filtered_content=evidence_text)
            extract_t0 = time.perf_counter()
            _log_analysis_detail(
                "extract_start",
                task=task,
                model=model,
                conversation_id=conversation_id,
                session_id=session_id,
                analyzed_turn_count=len(turns),
            )
            try:
                claims = await _await_with_remaining(
                    extractor.extract_from_conversation(
                        turns,
                        conversation_id=conversation_id,
                        assistant_only=assistant_only,
                    )
                )
            except TimeoutError:
                claims = []
                partial_reason = "extract_timeout"
                _log_analysis_detail(
                    "extract_error",
                    task=task,
                    model=model,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    error_type="TimeoutError",
                    error="analysis overall deadline reached during extract",
                )
            extract_elapsed_sec = time.perf_counter() - extract_t0
            _log_analysis_detail(
                "extract_result",
                task=task,
                model=model,
                conversation_id=conversation_id,
                session_id=session_id,
                elapsed_sec=extract_elapsed_sec,
                claim_count=len(claims),
                claims=[_claim_summary(claim) for claim in claims],
            )
            judge_t0 = time.perf_counter()
            judgments = []
            for claim in claims:
                remaining = _remaining_time()
                if remaining is not None and remaining <= 0:
                    partial_reason = partial_reason or "judge_timeout"
                    judgments.append(
                        _make_verification_error_judgment(
                            claim,
                            reason="Skipped because analysis timed out before this concept could be judged.",
                        )
                    )
                    continue
                precheck_judgment = _run_deterministic_claim_precheck(claim, question_text=question_text)
                if precheck_judgment is not None:
                    judgments.append(precheck_judgment)
                    _log_analysis_detail(
                        "precheck_result",
                        task=task,
                        model=model,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        claim=_claim_summary(claim),
                        hallucination=precheck_judgment.hallucination,
                        verification_error=precheck_judgment.verification_error,
                        concept_true_understanding=precheck_judgment.concept_true_understanding,
                        reason=precheck_judgment.reason,
                    )
                    continue
                _log_analysis_detail(
                    "judge_start",
                    task=task,
                    model=model,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    claim=_claim_summary(claim),
                    evidence_preview=(evidence.filtered_content or evidence.search_results or "")[:500],
                )
                claim_judge_t0 = time.perf_counter()
                try:
                    judgment = await asyncio.wait_for(
                        judge.judge_claim(claim, evidence),
                        timeout=_per_claim_timeout(remaining),
                    )
                except Exception as exc:
                    _log_analysis_detail(
                        "judge_error",
                        task=task,
                        model=model,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        claim=_claim_summary(claim),
                        elapsed_sec=time.perf_counter() - claim_judge_t0,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if _is_timeout_exception(exc):
                        partial_reason = partial_reason or "judge_timeout"
                        judgments.append(
                            _make_verification_error_judgment(
                                claim,
                                reason="Judgment timed out for this concept.",
                            )
                        )
                        continue
                    raise
                claim_judge_elapsed_sec = time.perf_counter() - claim_judge_t0
                judgments.append(judgment)
                _log_analysis_detail(
                    "judge_result",
                    task=task,
                    model=model,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    claim=_claim_summary(claim),
                    elapsed_sec=claim_judge_elapsed_sec,
                    hallucination=judgment.hallucination,
                    verification_error=judgment.verification_error,
                    concept_true_understanding=judgment.concept_true_understanding,
                    reason=judgment.reason,
                )
            judge_elapsed_sec = time.perf_counter() - judge_t0
        pipeline_result = PipelineResult(
            conversation_id=conversation_id,
            claims=claims,
            judgments=judgments,
        )
        summary = _summarize_result(pipeline_result)
        summary["analyzed_turn_count"] = len(turns)
        summary["analyzed_turns"] = _turn_summaries(turns)
        summary["extract_elapsed_sec"] = extract_elapsed_sec
        summary["judge_elapsed_sec"] = judge_elapsed_sec
        summary["analysis_elapsed_sec"] = extract_elapsed_sec + judge_elapsed_sec
        if timeout_sec is not None:
            summary["analysis_timeout_sec"] = timeout_sec
        if partial_reason is not None:
            summary["analysis_fallback_reason"] = partial_reason
            summary["analysis_partial"] = True

        planner_judgment_t0 = time.perf_counter()
        _log_analysis_detail(
            "planner_judge_start",
            task=task,
            model=model,
            conversation_id=conversation_id,
            session_id=session_id,
            current_answer=current_answer,
        )
        planner_judgment = await _run_planner_judge(
            llm=llm,
            question_text=question_text,
            current_answer=current_answer,
            summary=summary,
            failed_claim_ids=[],
        )
        summary["planner_judgment"] = planner_judgment
        summary["selected_claim_id"] = planner_judgment.get("selected_claim_id")
        summary["planner_decision"] = planner_judgment.get("decision")
        _log_analysis_detail(
            "planner_judge_result",
            task=task,
            model=model,
            conversation_id=conversation_id,
            session_id=session_id,
            elapsed_sec=time.perf_counter() - planner_judgment_t0,
            planner_judgment=planner_judgment,
        )

        _log_analysis_detail(
            "analysis_session_result",
            task=task,
            model=model,
            conversation_id=conversation_id,
            session_id=session_id,
            use_websearch=use_websearch,
            summary=summary,
        )
        return summary
    except Exception as exc:
        _log_analysis_detail(
            "analysis_session_error",
            task=task,
            model=model,
            conversation_id=conversation_id,
            session_id=session_id,
            use_websearch=use_websearch,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        await llm.close()
