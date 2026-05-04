from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from external_agent.schemas import Claim, EvidenceBundle, make_short_id


def _coerce_snippet_to_text(value: Any) -> str:
    """Turn whatever the extractor LLM returned for ``context_snippet`` into a
    plain string suitable for downstream substring anchoring.

    The extractor prompt asks for a string, but when the agent's response is
    dominated by structured tool output (e.g., literature_fetch result lists,
    nested JSON blocks) the LLM occasionally returns a list or dict instead.
    ``str(value)`` would then yield a Python repr like ``"[{'content': '...'}]"``
    which can never be substring-located in the real trajectory text. This
    helper extracts the textual content from common shapes and falls back to
    ``""`` when the payload is genuinely non-textual so anchor resolution can
    degrade to other signals (original_statement, claim text) instead.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in ("content", "text", "snippet", "value"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        return ""
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            chunk = _coerce_snippet_to_text(item)
            if chunk:
                parts.append(chunk)
        return "\n".join(parts).strip()
    return ""


PLANNER_JUDGE_SYSTEM_PROMPT = (
    "You are a planner-level reasoning judge for AGDebugger. Return JSON only.\n"
    "\n"
    "Your job is to identify the single most important repair target from the provided claim/judgment "
    "summaries. Prefer claims that most directly cause wrong answer grounding, mapping errors, or answer "
    "misalignment.\n"
    "\n"
    "Use these fields exactly:\n"
    "- selected_claim_id\n"
    "- decision              (one of: repair, no_repair_needed)\n"
    "- selected_claim_reason\n"
    "- answer_grounding_status\n"
    "- mapping_status\n"
    "- alignment_status\n"
    "- confidence\n"
    "- repair_concept_name       (a short, human-readable handle for the concept you want fixed; "
    "do NOT put any option label or answer identifier here)\n"
    "- incorrect_understanding   (one sentence describing what the agent currently believes that is wrong)\n"
    "- correct_understanding     (one sentence describing the corrected local understanding; DO NOT "
    "state which option is correct, DO NOT name any option number/letter, DO NOT include <answer> tags "
    "or the token TERMINATE; limit to ~60 words)\n"
    "- target_turn_number        (integer assistant turn number where the faulty reasoning lives; use the "
    "claim's turn_number if unsure)\n"
    "\n"
    "HARD CONTRACT — if decision==\"repair\":\n"
    "  * repair_concept_name, incorrect_understanding, correct_understanding and target_turn_number are ALL "
    "required and must be non-empty.\n"
    "  * correct_understanding must NOT directly write the answer (no option labels, no 'the answer is…').\n"
    "  * If you cannot supply those four fields, you MUST set decision=\"no_repair_needed\" instead.\n"
    "\n"
    "The downstream repair pipeline will paste correct_understanding into the agent's own turn as a "
    "first-person self-correction, so keep the tone concrete and local to a single reasoning step."
)


REWRITER_SYSTEM_PROMPT = (
    "You rewrite a contiguous span of an assistant agent's own reasoning, replacing "
    "a flawed argument with a short first-person reflection that reopens the decision.\n"
    "\n"
    "Return JSON only. The object must contain exactly:\n"
    "- rewritten_text  (string, 150-400 characters, 1-2 short paragraphs)\n"
    "\n"
    "=====================================================================\n"
    "HARD CONSTRAINTS — any violation causes the rewrite to be DISCARDED.\n"
    "A discarded rewrite wastes this call; write carefully.\n"
    "=====================================================================\n"
    "  (C1) First-person voice, as if the agent paused mid-reasoning.\n"
    "  (C2) ZERO option labels. Ban list (any case, any number/letter):\n"
    "         option 1 / option 2 / option 3 / option 4 / option 5 / option 6\n"
    "         option A / option B / option C / option D / option E / option F\n"
    "         option #1 / optionN / choice 1 / choice 2 / candidate 1\n"
    "       Do not refer to alternatives by index at all. Describe them\n"
    "       by their mechanism or subject instead.\n"
    "  (C3) ZERO tokens: <answer>, </answer>, TERMINATE (even in passing).\n"
    "  (C4) NO conclusion sentence that names or selects an answer.\n"
    "       Ban: \"Therefore the answer is …\", \"Thus the correct option is …\",\n"
    "            \"So the final answer is …\", \"the correct answer should be …\",\n"
    "            \"the best choice is …\", \"my conclusion is …\".\n"
    "       The span must REOPEN the choice, not close it.\n"
    "  (C5) Do NOT fabricate citations or paper titles. Use only the claims below.\n"
    "  (C6) No meta commentary like \"[REASONING REPAIR]\" or \"the agent should …\".\n"
    "\n"
    "=====================================================================\n"
    "CONTRASTIVE EXAMPLES (study these — most leak_guard failures are\n"
    "variations of the BAD pattern)\n"
    "=====================================================================\n"
    "BAD : \"Wait — I realize now that option 3 is more aligned with the evidence, so the correct answer is option 3.\"\n"
    "       (violates C2: names option 3 twice; violates C4: names the answer)\n"
    "GOOD: \"Wait — I jumped too fast. The evidence doesn't actually establish the inducible-expression mechanism I invoked; I need to look again at which of the stated choices describes the design-and-construct approach the paper actually reports.\"\n"
    "\n"
    "BAD : \"Hold on, option 1 and option 4 both mention protein engineering, and on re-reading I think option 1 is wrong.\"\n"
    "       (violates C2 three times; violates C4 by ruling an option out)\n"
    "GOOD: \"Hold on — I conflated a general protein-engineering method with the specific significance of the crystal structure the question asks about. Those are different attributes; I should re-read each listed method and check which one matches the specific structure this study reports.\"\n"
    "\n"
    "BAD : \"Let me re-check. Therefore the answer is likely choice 2 because …\"\n"
    "       (violates C2 + C4; also closes the choice)\n"
    "GOOD: \"Let me re-check. My earlier inference that the data support a super-resolution approach isn't actually what the cited tomography paper demonstrates — it reports 3D reconstructions from tilt series. I should go back to the listed methods and pick the one that describes that reconstruction workflow.\"\n"
    "\n"
    "BAD : \"TERMINATE reasoning — I need to re-evaluate the options.\"\n"
    "       (violates C3; ends the agent's turn)\n"
    "GOOD: \"I need to re-evaluate my earlier mapping. The evidence only shows X, not Y, so I should go back to the listed choices and compare each against X specifically.\"\n"
    "\n"
    "=====================================================================\n"
    "REQUIRED STRUCTURE (use in this order)\n"
    "=====================================================================\n"
    "  (S1) One sentence acknowledging the earlier inference was not actually\n"
    "       supported by the cited evidence.\n"
    "  (S2) One or two sentences restating the corrected local understanding\n"
    "       drawn verbatim-in-spirit from the supplied claims (NEVER copy\n"
    "       any option label or answer id that happened to appear in them).\n"
    "  (S3) One sentence inviting yourself to re-examine the listed methods /\n"
    "       choices / alternatives from scratch, WITHOUT naming any of them.\n"
    "\n"
    "Remember: describe alternatives by mechanism (\"the inducible-expression\n"
    "method\", \"the 3D-reconstruction method\", \"the super-resolution method\"),\n"
    "NEVER by their label (\"option 3\", \"choice A\")."
)


def build_rewriter_user_prompt(
    *,
    question_text: str,
    target_turn_number: int | None,
    original_span_content: str,
    prefix_context: str,
    suffix_context: str,
    contributing_claims: list[Dict[str, Any]],
) -> str:
    turn_suffix = f" (assistant turn #{target_turn_number})" if target_turn_number is not None else ""
    prefix = (prefix_context or "").strip()
    suffix = (suffix_context or "").strip()
    prefix_block = prefix[-800:] if len(prefix) > 800 else prefix
    suffix_block = suffix[:400] if len(suffix) > 400 else suffix
    claim_lines: list[str] = []
    for idx, claim in enumerate(contributing_claims or [], start=1):
        if not isinstance(claim, dict):
            continue
        concept = str(claim.get("concept_name") or claim.get("repair_concept_name") or "").strip()
        incorrect = str(claim.get("incorrect_understanding", "")).strip()
        correct = str(
            claim.get("corrected_claim_text")
            or claim.get("correct_understanding")
            or ""
        ).strip()
        evidence = claim.get("evidence_basis") or []
        if isinstance(evidence, list):
            evidence_str = "; ".join(str(e).strip() for e in evidence if str(e).strip())[:300]
        else:
            evidence_str = str(evidence).strip()[:300]
        claim_lines.append(
            f"  Claim {idx}:\n"
            f"    concept: {concept or '(unspecified)'}\n"
            f"    agent previously believed: {incorrect or '(not stated)'}\n"
            f"    corrected understanding: {correct or '(not stated)'}\n"
            f"    evidence basis: {evidence_str or '(none)'}"
        )
    claims_block = "\n".join(claim_lines) if claim_lines else "  (no contributing claims supplied)"
    return (
        f"QUESTION:\n{question_text or '(not provided)'}\n\n"
        f"TARGET TURN{turn_suffix}:\n"
        f"  [prefix context — the text that comes BEFORE the span being rewritten]:\n"
        f"{prefix_block or '(empty)'}\n\n"
        f"  [SPAN TO REWRITE — this is the flawed reasoning you must replace]:\n"
        f"{(original_span_content or '').strip() or '(empty)'}\n\n"
        f"  [suffix context — the text that comes AFTER the span]:\n"
        f"{suffix_block or '(empty)'}\n\n"
        f"CONTRIBUTING CLAIMS (use these as the factual basis — do not invent others):\n"
        f"{claims_block}\n\n"
        "Produce the replacement as JSON: {\"rewritten_text\": \"...\"}.\n"
        "Remember: first-person, no option labels, no <answer>, no TERMINATE, no "
        "conclusion that names an answer."
    )


def build_planner_judge_user_prompt(*, question_text: str, current_answer: str, analysis_summary: Dict[str, Any], failed_claim_ids: list[str] | None = None) -> str:
    failed_claim_ids = failed_claim_ids or []
    return (
        f"Question:\n{question_text}\n\n"
        f"Current answer:\n{current_answer or '(none)'}\n\n"
        f"Analysis summary JSON:\n{analysis_summary}\n\n"
        f"Failed claim ids (do not prefer these — if every remaining option is in this list, "
        f"downgrade to decision=no_repair_needed rather than recycling a failed claim): "
        f"{failed_claim_ids}\n\n"
        "Return one JSON object following the schema in the system prompt. "
        "If decision==\"repair\", also include repair_concept_name, incorrect_understanding, "
        "correct_understanding, and target_turn_number; otherwise set decision=\"no_repair_needed\"."
    )


class ClaimJudgeStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def extractor_system_prompt(self) -> str:
        raise NotImplementedError

    @property
    def judge_system_prompt(self) -> str:
        return (
            "You are a hallucination judge.\n"
            "Return JSON only.\n"
            "Judge whether the claim is grounded by the provided evidence.\n"
            "Use these fields exactly:\n"
            "- reference_name\n"
            "- reference_grounding\n"
            "- content_grounding\n"
            "- hallucination\n"
            "- abstention\n"
            "- verification_error\n"
            "- reason\n"
            "Use Yes/No/N/A where appropriate."
        )

    @abstractmethod
    def build_extraction_user_prompt(self, content: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def normalize_claim(self, raw: Dict[str, Any], conversation_id: int, turn_number: int) -> Claim | None:
        raise NotImplementedError

    @abstractmethod
    def render_claim(self, claim: Claim) -> str:
        raise NotImplementedError

    def build_judge_user_prompt(self, claim: Claim, evidence: EvidenceBundle, **kwargs: Any) -> str:
        return (
            "Claim:\n"
            f"{self.render_claim(claim)}\n\n"
            "Evidence:\n"
            "<SEARCH_RESULTS>\n"
            f"{evidence.search_results}\n"
            "</SEARCH_RESULTS>\n\n"
            "<FILTERED_CONTENT>\n"
            f"{evidence.filtered_content}\n"
            "</FILTERED_CONTENT>\n\n"
            "Judge whether the claim is supported by the evidence."
        )

    def _make_claim(
        self,
        *,
        conversation_id: int,
        turn_number: int,
        category: str,
        text: str,
        source_ref: str = "",
        source_type: str = "",
        original_statement: str = "",
        data: Dict[str, Any] | None = None,
    ) -> Claim:
        return Claim(
            claim_id=make_short_id(),
            conversation_id=conversation_id,
            turn_number=turn_number,
            category=category,
            text=text,
            source_ref=source_ref,
            source_type=source_type,
            original_statement=original_statement,
            data=data or {},
        )


class ResearchStrategy(ClaimJudgeStrategy):
    name = "research_questions"
    extractor_system_prompt = (
        "Extract every atomic claim explicitly attributed to a source.\n"
        "Return a JSON array.\n"
        "Each object must contain:\n"
        "- inferred_source_type\n"
        "- claimed_content\n"
        "- full_citation\n"
        "- claimed_title\n"
        "- claimed_authors\n"
        "- claimed_year\n"
        "- claimed_url\n"
        "If nothing is sourced, return []."
    )

    def build_extraction_user_prompt(self, content: str) -> str:
        return f"Text:\n<START>\n{content}\n<END>\n\nExtract sourced claims."

    def normalize_claim(self, raw: Dict[str, Any], conversation_id: int, turn_number: int) -> Claim | None:
        text = str(raw.get("claimed_content", "")).strip()
        if not text or text == "Unknown":
            return None
        return self._make_claim(
            conversation_id=conversation_id,
            turn_number=turn_number,
            category="research_claim",
            text=text,
            source_ref=str(raw.get("full_citation") or raw.get("claimed_title") or raw.get("claimed_url") or "").strip(),
            source_type=str(raw.get("inferred_source_type", "")).strip(),
            original_statement=str(raw.get("original_statement", "")).strip(),
            data=raw,
        )

    def render_claim(self, claim: Claim) -> str:
        data = claim.data
        return (
            f"Type: {claim.source_type}\n"
            f"Claim: {claim.text}\n"
            f"Citation: {data.get('full_citation', '')}\n"
            f"Title: {data.get('claimed_title', '')}\n"
            f"Authors: {data.get('claimed_authors', '')}\n"
            f"Year: {data.get('claimed_year', '')}\n"
            f"URL: {data.get('claimed_url', '')}"
        )


class MedicalStrategy(ClaimJudgeStrategy):
    name = "medical_guidelines"
    extractor_system_prompt = (
        "Extract every explicitly sourced medical guideline claim.\n"
        "Return a JSON array with:\n"
        "- authority\n"
        "- full_citation\n"
        "- claimed_content\n"
        "- claimed_url\n"
        "If nothing qualifies, return []."
    )

    def build_extraction_user_prompt(self, content: str) -> str:
        return f"Text:\n<START>\n{content}\n<END>\n\nExtract explicitly sourced guideline claims."

    def normalize_claim(self, raw: Dict[str, Any], conversation_id: int, turn_number: int) -> Claim | None:
        text = str(raw.get("claimed_content", "")).strip()
        authority = str(raw.get("authority", "")).strip()
        if not text or not authority or text == "Unknown" or authority == "Unknown":
            return None
        return self._make_claim(
            conversation_id=conversation_id,
            turn_number=turn_number,
            category="medical_claim",
            text=text,
            source_ref=authority,
            source_type="guideline",
            original_statement=str(raw.get("original_statement", "")).strip(),
            data=raw,
        )

    def render_claim(self, claim: Claim) -> str:
        data = claim.data
        return (
            f"Authority: {claim.source_ref}\n"
            f"Claim: {claim.text}\n"
            f"Full Citation: {data.get('full_citation', '')}\n"
            f"URL/DOI: {data.get('claimed_url', '')}"
        )


class LegalStrategy(ClaimJudgeStrategy):
    name = "legal_cases"
    extractor_system_prompt = (
        "Extract every cited legal reference used to support a proposition.\n"
        "Return a JSON array with:\n"
        "- type\n"
        "- content\n"
        "- reference_name\n"
        "- holding_or_description\n"
        "If nothing qualifies, return []."
    )

    def build_extraction_user_prompt(self, content: str) -> str:
        return f"Text:\n<START>\n{content}\n<END>\n\nExtract cited legal claims."

    def normalize_claim(self, raw: Dict[str, Any], conversation_id: int, turn_number: int) -> Claim | None:
        text = str(raw.get("content", "")).strip()
        ref = str(raw.get("reference_name", "")).strip()
        if not text or not ref or text == "Unknown" or ref == "Unknown":
            return None
        return self._make_claim(
            conversation_id=conversation_id,
            turn_number=turn_number,
            category="legal_claim",
            text=text,
            source_ref=ref,
            source_type=str(raw.get("type", "")).strip(),
            original_statement=str(raw.get("original_statement", "")).strip(),
            data=raw,
        )

    def render_claim(self, claim: Claim) -> str:
        data = claim.data
        return (
            f"Reference Type: {claim.source_type}\n"
            f"Reference: {claim.source_ref}\n"
            f"Holding/Description: {data.get('holding_or_description', '')}\n"
            f"Claim: {claim.text}"
        )


class CodingStrategy(ClaimJudgeStrategy):
    name = "coding"
    extractor_system_prompt = (
        "Extract atomic code elements that can be checked for hallucination.\n"
        "Return a JSON array with:\n"
        "- element_type (import|install|function_call)\n"
        "- package_name\n"
        "- code_snippet\n"
        "- language\n"
        "- function_name (optional)\n"
        "Skip standard library elements. If none exist, return []."
    )

    def build_extraction_user_prompt(self, content: str) -> str:
        return f"Code or assistant response:\n<START>\n{content}\n<END>\n\nExtract verifiable code elements."

    def normalize_claim(self, raw: Dict[str, Any], conversation_id: int, turn_number: int) -> Claim | None:
        element_type = str(raw.get("element_type", "")).strip()
        package_name = str(raw.get("package_name", "")).strip()
        code_snippet = str(raw.get("code_snippet", "")).strip()
        if element_type not in {"import", "install", "function_call"} or not package_name or not code_snippet:
            return None
        function_name = str(raw.get("function_name", "")).strip()
        return self._make_claim(
            conversation_id=conversation_id,
            turn_number=turn_number,
            category=element_type,
            text=code_snippet,
            source_ref=package_name,
            source_type=str(raw.get("language", "")).strip(),
            original_statement=str(raw.get("original_statement", "")).strip(),
            data={**raw, "function_name": function_name},
        )

    def render_claim(self, claim: Claim) -> str:
        data = claim.data
        return (
            f"Element Type: {claim.category}\n"
            f"Package: {claim.source_ref}\n"
            f"Language: {claim.source_type}\n"
            f"Function: {data.get('function_name', '')}\n"
            f"Code: {claim.text}"
        )

    @property
    def judge_system_prompt(self) -> str:
        return (
            "You are a code hallucination judge.\n"
            "Return JSON only.\n"
            "Judge whether the package or API usage is real given the evidence.\n"
            "Use these fields exactly:\n"
            "- reference_name\n"
            "- reference_grounding\n"
            "- content_grounding\n"
            "- hallucination\n"
            "- abstention\n"
            "- verification_error\n"
            "- reason"
        )


class ScientificConceptDiscoveryStrategy(ClaimJudgeStrategy):
    name = "scientific_concept_discovery"
    extractor_system_prompt = (
        "You extract claims from an *agent's* reasoning, not from the literature it cites.\n"
        "\n"
        "CRITICAL — tool_result / literature_search payloads are READ-ONLY EVIDENCE.\n"
        "If a chunk of the input looks like a tool result (a paper title + Authors: + "
        "DOI: + Venue:, a numbered list of PubMed abstracts, a block starting with "
        "'Literature results for:' / 'Sciverse workflow result for:', or a Python/JSON "
        "list of dicts with a 'content' and 'name' field), DO NOT turn any of its "
        "sentences into a scientific_concept claim. Those sentences are real paper "
        "abstracts and the judge will only confirm them as grounded — which wastes the "
        "entire repair budget. Only extract claims from sentences that are the agent's "
        "OWN reasoning, interpretation, or answer commitment.\n"
        "\n"
        "Extract all atomic scientific concepts that materially drive the agent's reasoning.\n"
        "Prioritize concepts the agent uses to choose its final answer, reject competing options, or justify an intermediate inference.\n"
        "Ignore repeated paraphrases, transport wrappers, control tokens, and purely stylistic text.\n"
        "Split bundled reasoning into separate concepts whenever they can be fact-checked independently.\n"
        "Return a JSON array.\n"
        "Each object must contain:\n"
        "- scientific_concept: the scientific concept the agent's reasoning depends on\n"
        "- concept_understanding: the agent's specific understanding/interpretation of this concept\n"
        "- corresponding_action: the next reasoning step or decision the agent takes based on this concept (can be empty if implicit)\n"
        "- context_snippet: the relevant snippet from the original text\n"
        "Return all non-duplicate concepts, with the most error-prone concepts first.\n"
        "If no scientific concepts are used, return [].\n\n"
        "ADDITIONAL — you MUST also extract the following mapping claims when present:\n\n"
        "MAPPING TYPE 1 — Answer Grounding:\n"
        "When the agent connects evidence or a structural feature to a specific answer candidate or final answer, extract it as:\n"
        "- scientific_concept: 'Answer grounding: [answer text or candidate text]'\n"
        "- concept_understanding: the agent's reasoning connecting the evidence to this option\n"
        "- corresponding_action: 'Selected answer candidate', 'Rejected answer candidate', or equivalent answer-grounding action\n"
        "- context_snippet: the relevant snippet\n\n"
        "MAPPING TYPE 2 — Entity-Option Compatibility:\n"
        "When the agent attributes a property to the subject that may be biologically/chemically incompatible "
        "(e.g., attributing amino acid properties to an RNA system), extract it as:\n"
        "- scientific_concept: 'Entity compatibility: [entity] in [subject context]'\n"
        "- concept_understanding: the agent's assumption about why this entity is relevant\n"
        "- corresponding_action: how this assumption drives the option selection\n"
        "- context_snippet: the relevant snippet\n\n"
        "MAPPING TYPE 3 — Answer Alignment:\n"
        "When the agent's final answer or concluding statement does not actually align with the question's requested target "
        "(for example, it answers with a generally related fact but not the asked mechanism / entity / relationship), extract it as:\n"
        "- scientific_concept: 'Answer alignment: [final answer or conclusion]'\n"
        "- concept_understanding: the agent's final answer commitment and why it believes this addresses the question\n"
        "- corresponding_action: the final answer statement or commitment it makes\n"
        "- context_snippet: the relevant snippet\n\n"
        "These mapping claims are HIGH PRIORITY — they capture the 'last hop' from evidence to answer.\n"
        "ALWAYS emit at least one 'Answer alignment: ...' or 'Answer grounding: ...' claim for any turn that "
        "commits to an answer or option, even if you cannot identify any scientific_concept claim. Returning a "
        "JSON array with only scientific_concept entries and no mapping/alignment entry is a failure mode that "
        "blocks downstream repair."
    )

    def build_extraction_user_prompt(self, content: str) -> str:
        return (
            f"Text:\n<START>\n{content}\n<END>\n\n"
            "Extract all atomic scientific concepts AND answer-grounding reasoning that materially drive the reasoning or answer.\n"
            "For EACH answer candidate, final answer statement, or equivalent answer commitment the agent explicitly evaluates, endorses, or rejects, extract one mapping claim capturing "
            "WHY the agent connects its evidence to that answer.\n"
            "If the agent's chosen answer involves an entity class (e.g., amino acids, nucleotides, metal ions) "
            "that may not exist in the subject (e.g., RNA aptamer vs protein), extract this as an entity compatibility claim.\n"
            "If the final answer or conclusion is only loosely related and may not directly answer the question's requested target, "
            "extract this as an answer alignment claim.\n"
            "Include concepts used to support the chosen answer, dismiss competing answer candidates, or justify intermediate reasoning. "
            "If one paragraph contains multiple independently checkable concepts, output them separately. "
            "Do not collapse different concepts into a single item."
        )

    def normalize_claim(self, raw: Dict[str, Any], conversation_id: int, turn_number: int) -> Claim | None:
        concept = str(raw.get("scientific_concept", "")).strip()
        understanding = str(raw.get("concept_understanding", "")).strip()
        action = str(raw.get("corresponding_action", "")).strip()
        context_snippet = _coerce_snippet_to_text(raw.get("context_snippet"))
        if not concept or concept == "Unknown":
            return None
        if not understanding:
            return None
        # PR-1: reject raw claim payloads whose scientific_concept/context field
        # is clearly a tool_result (literature_search dump, Python-repr of a
        # list-of-dicts, or a block starting with "Literature results for:").
        # These slip through even with the strengthened system prompt because
        # some extractor models happily echo the input back as the concept
        # field when they cannot find real reasoning to summarize.
        lowered_concept_for_filter = concept.lower()
        if (
            lowered_concept_for_filter.startswith("[{'content'")
            or lowered_concept_for_filter.startswith('[{"content"')
            or lowered_concept_for_filter.startswith("{'content'")
            or lowered_concept_for_filter.startswith('{"content"')
            or "literature results for" in lowered_concept_for_filter
            or "sciverse workflow result" in lowered_concept_for_filter
        ):
            return None
        if (
            "literature results for" in understanding.lower()
            or "sciverse workflow result" in understanding.lower()
            or understanding.lower().startswith(("[{'", '[{"'))
        ):
            return None
        lowered_concept = concept.lower()
        category = "scientific_concept"
        source_type = "scientific_concept"
        if lowered_concept.startswith("answer grounding:"):
            category = "mapping_claim"
            source_type = "answer_grounding"
        elif lowered_concept.startswith("entity compatibility:"):
            category = "constraint_claim"
            source_type = "entity_compatibility"
        elif lowered_concept.startswith("answer alignment:"):
            category = "answer_alignment_claim"
            source_type = "answer_alignment"
        return self._make_claim(
            conversation_id=conversation_id,
            turn_number=turn_number,
            category=category,
            text=understanding,
            source_ref=concept,
            source_type=source_type,
            original_statement=context_snippet,
            data={**raw, "corresponding_action": action},
        )

    def render_claim(self, claim: Claim) -> str:
        data = claim.data
        return (
            f"Scientific Concept: {claim.source_ref}\n"
            f"Agent's Understanding: {claim.text}\n"
            f"Corresponding Action: {data.get('corresponding_action', '')}\n"
            f"Original Context: {data.get('context_snippet', '')}"
        )

    @property
    def judge_system_prompt(self) -> str:
        return (
            "You are a scientific concept verification judge.\n"
            "Return JSON only.\n"
            "Use these fields exactly:\n"
            "- reference_name: the concept name\n"
            "- reference_grounding: what sources you based your judgment on "
            "(list multiple sources)\n"
            "- content_grounding: whether the concept content is correct\n"
            "- hallucination: Yes / No / N/A\n"
            "- concept_true_understanding: if hallucination=Yes, describe what "
            "is wrong with the agent's claim — what assumption failed, what "
            "entity/attribute was mismatched, or what logical gap exists. Do NOT "
            "state the correct answer or which option should be selected; only "
            "describe the flaw so the agent can re-derive the answer independently. "
            "If hallucination=No, leave empty\n"
            "- abstention: Yes / No / N/A\n"
            "- verification_error: Yes / No / N/A\n"
            "- premise_verdict: true / false / unverified (for mapping/alignment/constraint claims)\n"
            "- inference_verdict: true / false / unverified (for mapping/alignment/constraint claims)\n"
            "- reason: your judgment rationale, MUST explicitly cite both the premise verdict "
            "and the inference verdict for mapping/alignment/constraint claims\n\n"
            "====================================================================\n"
            "EVALUATION PROTOCOL — READ CAREFULLY\n"
            "====================================================================\n"
            "Different claim categories require different evaluation protocols.\n"
            "You MUST use the protocol that matches the claim's category and source_type.\n\n"
            "--- PROTOCOL A: pure factual claims (category=scientific_concept) ---\n"
            "Cross-validate using MULTIPLE independent evidence sources. If the agent's\n"
            "stated fact is supported by evidence, set hallucination=No. If contradicted\n"
            "or fabricated, set hallucination=Yes.\n\n"
            "--- PROTOCOL B: two-stage evaluation (REQUIRED for category ∈ {mapping_claim, "
            "answer_alignment_claim, constraint_claim}) ---\n"
            "A mapping/alignment/constraint claim implicitly contains TWO assertions that\n"
            "MUST be verified SEPARATELY. Failing either one is a hallucination.\n\n"
            "STAGE 1 — PREMISE check (premise_verdict):\n"
            "  Is the cited evidence factually accurate and actually present in the\n"
            "  referenced source? If evidence is missing or contradicted, premise_verdict=false.\n"
            "  Otherwise premise_verdict=true.\n\n"
            "STAGE 2 — INFERENCE check (inference_verdict):\n"
            "  Does the evidence SEMANTICALLY answer the SPECIFIC question that was asked,\n"
            "  or only a related-but-different question? Check all of:\n"
            "    (a) Entity match: does the evidence's subject entity (organism, protein,\n"
            "        molecular species, experimental condition) match the question's?\n"
            "    (b) Attribute match: does the evidence address the attribute the question\n"
            "        is asking about (e.g. 'significance of crystal structure' vs.\n"
            "        'protein engineering for crystallizability' are DIFFERENT attributes)?\n"
            "    (c) Coverage match: if the question asks about interaction / interplay /\n"
            "        mechanism between two things, the evidence must actually address THAT\n"
            "        interaction — not each side in isolation.\n"
            "    (d) Option specificity: if the claim maps evidence to a specific option\n"
            "        text, does the evidence match THAT specific option phrasing, not a\n"
            "        neighbouring option that is also generally true?\n"
            "  If any of (a)-(d) fails, inference_verdict=false.\n"
            "  Only when all of (a)-(d) pass, inference_verdict=true.\n\n"
            "FINAL VERDICT RULES for PROTOCOL B:\n"
            "  premise_verdict=false              → hallucination=Yes, error_type=fact_error\n"
            "  premise_verdict=true, inference_verdict=false\n"
            "                                     → hallucination=Yes, error_type=mapping_error\n"
            "                                       (the facts are true but don't justify the\n"
            "                                        agent's jump to this specific option)\n"
            "  BOTH premise_verdict=true AND inference_verdict=true → hallucination=No\n"
            "  premise_verdict=unverified OR inference_verdict=unverified\n"
            "                                     → abstention=Yes, verification_error=Yes\n\n"
            "In the `reason` field for PROTOCOL B, you MUST write two explicit sentences:\n"
            "  1. \"Premise: <verdict> — <why>.\"\n"
            "  2. \"Inference: <verdict> — <why, referring to (a)/(b)/(c)/(d) as needed>.\"\n\n"
            "--- ENTITY-SUBJECT COMPATIBILITY OVERRIDE ---\n"
            "Before either protocol, check: does the entity class in the claim naturally\n"
            "exist as an intrinsic component of the question's subject?\n"
            "  - amino acids are protein components, NOT RNA components.\n"
            "  - nucleotides are RNA/DNA components, NOT protein components.\n"
            "If there is a fundamental biological/chemical incompatibility, set\n"
            "hallucination=Yes immediately and state it in `reason`, regardless of evidence.\n\n"
            "====================================================================\n"
            "WORKED EXAMPLE — mapping_error (premise true, inference false)\n"
            "====================================================================\n"
            "Question: \"What is the significance of the 2.5 Å crystal structure of the\n"
            "mouse PD-L1 PD-1 binding domain?\"\n"
            "Claim (category=mapping_claim, source_type=answer_grounding):\n"
            "  \"The PMC article confirms that protein-engineering methods are designed\n"
            "   to enhance crystallizability, supporting option 1 (Engineering novel\n"
            "   types of proteins for crystallization).\"\n"
            "Evidence: PMC article does state protein-engineering enhances crystallizability.\n\n"
            "Correct judgment:\n"
            "  premise_verdict=true  (the PMC article really says this)\n"
            "  inference_verdict=false\n"
            "    (b) Attribute mismatch: the question asks about the SIGNIFICANCE of a\n"
            "        specific crystal structure, not about general crystallization methods.\n"
            "    (a) Entity mismatch: the evidence discusses generic protein engineering,\n"
            "        not mouse PD-L1 PD-1 binding specifically.\n"
            "  → hallucination=Yes, error_type=mapping_error\n"
            "  reason: \"Premise: true — PMC source confirms protein-engineering is used\n"
            "    to enhance crystallizability. Inference: false — (a)(b): the question\n"
            "    asks about the significance of a specific mouse PD-L1 crystal structure,\n"
            "    but the cited evidence is about generic protein-engineering methods, a\n"
            "    different subject and a different attribute. The jump to option 1 is\n"
            "    therefore unjustified by the cited evidence.\""
        )

    def build_judge_user_prompt(self, claim: Claim, evidence: EvidenceBundle, **kwargs: Any) -> str:
        question_text = kwargs.get("question_text", "")
        two_stage_categories = {"mapping_claim", "answer_alignment_claim", "constraint_claim"}
        use_two_stage = claim.category in two_stage_categories

        preamble = (
            "IMPORTANT: Cross-validate using MULTIPLE independent sources from the evidence.\n"
            "If only one source supports the claim, mark hallucination as N/A.\n"
            "If the evidence is empty, topically unrelated to the claim, or too weak to support or refute the claim, "
            "return abstention=Yes and verification_error=Yes instead of forcing a hallucination judgment.\n"
            "If the agent's understanding is incorrect, provide the correct understanding "
            "in concept_true_understanding.\n\n"
        )

        if use_two_stage:
            preamble += (
                "THIS CLAIM REQUIRES PROTOCOL B (two-stage evaluation).\n"
                f"Claim category: {claim.category}\n"
                f"Claim source_type: {claim.source_type}\n"
                "You MUST evaluate both premise_verdict AND inference_verdict and explain them\n"
                "separately in `reason`. The premise being true is NOT sufficient — the inference\n"
                "from evidence to the specific question/option must ALSO hold, or this is a\n"
                "mapping_error hallucination.\n\n"
            )

        if question_text:
            preamble += (
                f"Question context:\n{question_text}\n\n"
                "Use the question text above to perform STAGE 2 (inference check):\n"
                "- Does the cited evidence's subject entity match the question's subject?\n"
                "- Does the cited evidence address the SPECIFIC attribute the question asks about?\n"
                "- If the question asks about an interaction/mechanism/relationship, does the\n"
                "  evidence actually address that specific interaction (not just each side)?\n"
                "If any of these fail, inference_verdict=false → hallucination=Yes\n"
                "(error_type=mapping_error), regardless of whether the premise is true.\n\n"
            )

        return (
            preamble
            + "Claim:\n"
            f"Category: {claim.category}\n"
            f"{self.render_claim(claim)}\n\n"
            "Evidence:\n"
            "<SEARCH_RESULTS>\n"
            f"{evidence.search_results}\n"
            "</SEARCH_RESULTS>\n\n"
            "<FILTERED_CONTENT>\n"
            f"{evidence.filtered_content}\n"
            "</FILTERED_CONTENT>\n\n"
            + (
                "Apply PROTOCOL B. Output premise_verdict, inference_verdict, and a `reason`\n"
                "containing two explicit sentences: \"Premise: <verdict> — <why>.\" and\n"
                "\"Inference: <verdict> — <why>.\"\n"
                "If premise_verdict=true and inference_verdict=false, set hallucination=Yes\n"
                "and put the correct scope/mapping in concept_true_understanding."
                if use_two_stage
                else "Judge whether the agent's understanding of the scientific concept is correct based on the evidence."
            )
        )


STRATEGY_MAP = {
    "research_questions": ResearchStrategy,
    "medical_guidelines": MedicalStrategy,
    "legal_cases": LegalStrategy,
    "coding": CodingStrategy,
    "scientific_concept_discovery": ScientificConceptDiscoveryStrategy,
}


def get_strategy(name: str) -> ClaimJudgeStrategy:
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_MAP)}")
    return STRATEGY_MAP[name]()
