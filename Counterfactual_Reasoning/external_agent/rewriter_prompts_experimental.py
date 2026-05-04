"""Experimental system-prompt variants for the Path B stuck-case study.

All variants share the same HARD CONSTRAINTS (C1-C6) as the production
``REWRITER_SYSTEM_PROMPT`` in ``strategies.py`` — no option labels, no
``<answer>`` tags, no ``TERMINATE``, no conclusion sentence that names
an answer. They differ ONLY in the REQUIRED STRUCTURE and CONTRASTIVE
EXAMPLES sections, in order to isolate the effect of epistemic framing.

The constraint the user set for Path B: "do not inject the answer; the
rewrite must change the original agent's reasoning trajectory so it
realizes its own hallucination or wrong inference and self-corrects."
Each variant pushes on a different mechanism:

  V1  EVIDENCE_GAP_ENUMERATION
      Force the rewrite to enumerate the gap between what the agent
      assumed and what the evidence actually supports. Hypothesis: a
      visible gap is harder to hand-wave than a generic "let me
      reconsider."

  V2  COMMITMENT_INVALIDATION
      Rewrite declares the prior inference INVALID and commits to
      re-deriving the mapping from scratch. Hypothesis: strong
      invalidation language breaks the anchoring that keeps 24/26
      stuck cases returning to the same answer.

  V3  EVIDENCE_ANCHORED_CONTRADICTION
      Rewrite paraphrases one specific evidence fragment and contrasts
      it verbatim-in-spirit with the agent's incorrect premise.
      Hypothesis: concrete contradiction is harder to dismiss than
      abstract reflection.
"""
from __future__ import annotations


_HARD_CONSTRAINTS_BLOCK = (
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
)

_JSON_CONTRACT = (
    "You rewrite a contiguous span of an assistant agent's own reasoning, replacing\n"
    "a flawed argument with a short first-person reflection that reopens the decision.\n"
    "\n"
    "Return JSON only. The object must contain exactly:\n"
    "- rewritten_text  (string, 150-400 characters, 1-2 short paragraphs)\n"
    "\n"
)


# ---------------------------------------------------------------------------
# V1 — EVIDENCE_GAP_ENUMERATION
# ---------------------------------------------------------------------------
REWRITER_SYSTEM_PROMPT_V1_EVIDENCE_GAP = (
    _JSON_CONTRACT
    + _HARD_CONSTRAINTS_BLOCK
    + "\n"
    "=====================================================================\n"
    "REQUIRED STRUCTURE — enumerate the evidence gap explicitly.\n"
    "The rewrite must make three things visible to me, the agent who is\n"
    "reading this turn back:\n"
    "  (S1) The SPECIFIC factual assumption my earlier reasoning relied on\n"
    "       (one clause, concrete — NOT a generic 'I may have misjudged').\n"
    "  (S2) What the evidence actually supports (one or two sentences,\n"
    "       paraphrased from the supplied claims, no fabricated details).\n"
    "  (S3) The GAP — one sentence starting with 'My assumption required X\n"
    "       but the evidence establishes only Y; it does NOT establish Z.'\n"
    "  (S4) A redirection that names NO option: 'I should therefore\n"
    "       re-match each listed mechanism against what the evidence\n"
    "       actually establishes, not against my assumption.'\n"
    "=====================================================================\n"
    "\n"
    "CONTRASTIVE EXAMPLE (GOOD shape):\n"
    "  GOOD: \"Wait — my reasoning assumed that the cited tomography paper\n"
    "  uses Fourier-ring-correlation to judge resolution (S1). The evidence\n"
    "  only says the paper reports 3D reconstructions built from tilt-series\n"
    "  back-projection (S2). My assumption required a resolution-metric\n"
    "  claim but the evidence establishes only a reconstruction-method\n"
    "  claim; it does not establish the resolution metric I invoked (S3).\n"
    "  I should therefore re-match each listed method against the\n"
    "  reconstruction-workflow claim the evidence actually supports (S4).\"\n"
    "\n"
    "Remember: describe alternatives by mechanism\n"
    "(\"the tilt-series reconstruction method\", \"the single-particle\n"
    "averaging method\"), NEVER by their label (\"option 3\", \"choice A\")."
)


# ---------------------------------------------------------------------------
# V2 — COMMITMENT_INVALIDATION
# ---------------------------------------------------------------------------
REWRITER_SYSTEM_PROMPT_V2_INVALIDATION = (
    _JSON_CONTRACT
    + _HARD_CONSTRAINTS_BLOCK
    + "\n"
    "=====================================================================\n"
    "REQUIRED STRUCTURE — invalidate the prior inference and restart.\n"
    "The rewrite is a deliberate epistemic reset. You must actively\n"
    "RETRACT the prior inference, not merely question it.\n"
    "  (S1) Open with an explicit invalidation sentence:\n"
    "       'The inference that led me to my earlier position is not\n"
    "       supported by the cited evidence and I am retracting it.'\n"
    "  (S2) One or two sentences stating what the evidence actually\n"
    "       establishes (paraphrased from the supplied claims).\n"
    "  (S3) Commit to a full re-derivation:\n"
    "       'I will treat my earlier mapping as void and re-derive the\n"
    "       mapping from the evidence alone, without anchoring on any\n"
    "       choice I previously made.'\n"
    "  (S4) One sentence naming the concrete subject-level criterion to\n"
    "       use, without naming any option label: e.g. 'I will compare\n"
    "       each listed mechanism strictly against the evidence-supported\n"
    "       mechanism above.'\n"
    "=====================================================================\n"
    "\n"
    "CONTRASTIVE EXAMPLE (GOOD shape):\n"
    "  GOOD: \"The inference that led me to my earlier position is not\n"
    "  supported by the cited evidence and I am retracting it (S1). The\n"
    "  cited work actually characterizes an inhibitory interaction at the\n"
    "  RNA-exit channel, not the antitermination role I had invoked (S2).\n"
    "  I will treat my earlier mapping as void and re-derive it from the\n"
    "  evidence alone, without anchoring on any choice I previously made\n"
    "  (S3). I will compare each listed mechanism strictly against the\n"
    "  RNA-exit-channel inhibition the evidence supports (S4).\"\n"
    "\n"
    "Remember: 'retracting', 'void', 'not supported' — use direct\n"
    "invalidation language, not softening phrases like 'might have' or\n"
    "'perhaps reconsider'."
)


# ---------------------------------------------------------------------------
# V3 — EVIDENCE_ANCHORED_CONTRADICTION
# ---------------------------------------------------------------------------
REWRITER_SYSTEM_PROMPT_V3_CONTRADICTION = (
    _JSON_CONTRACT
    + _HARD_CONSTRAINTS_BLOCK
    + "\n"
    "=====================================================================\n"
    "REQUIRED STRUCTURE — contrast an evidence paraphrase with the\n"
    "agent's wrong premise, concretely.\n"
    "  (S1) 'The cited evidence states that <short paraphrase of the\n"
    "       correct_understanding / evidence_basis supplied below>.'\n"
    "       The paraphrase must be concrete, specific and mechanism-level —\n"
    "       NOT a generic 'the evidence shows the claim is wrong.'\n"
    "  (S2) 'But my earlier reasoning instead relied on the premise that\n"
    "       <paraphrase of the incorrect_understanding supplied below>,\n"
    "       which the evidence does not support.'\n"
    "  (S3) 'Because those two are different mechanisms / properties /\n"
    "       objects, any choice I made by leaning on the wrong premise is\n"
    "       not justified.'\n"
    "  (S4) 'I should re-examine each listed mechanism against what the\n"
    "       evidence actually says, not against my earlier premise.' —\n"
    "       NO option labels.\n"
    "=====================================================================\n"
    "\n"
    "CONTRASTIVE EXAMPLE (GOOD shape):\n"
    "  GOOD: \"The cited evidence states that ClfA residues H6 H7 G10 Q13\n"
    "  A14 G15 bind the fibrinogen gamma chain (S1). But my earlier\n"
    "  reasoning instead relied on the premise that residues 398-411 are\n"
    "  the binding residues, which the evidence does not support (S2).\n"
    "  Because those are different residue sets from different regions of\n"
    "  the protein, any choice I made by leaning on the 398-411 premise is\n"
    "  not justified (S3). I should re-examine each listed candidate\n"
    "  against the H6-G15 binding region the evidence actually identifies,\n"
    "  not against my earlier premise (S4).\"\n"
    "\n"
    "Remember: the paraphrase in (S1) must carry concrete, mechanism-level\n"
    "detail from the supplied claims — do NOT paraphrase it into 'the\n"
    "evidence shows the agent is wrong' or any abstract equivalent.\n"
    "Describe alternatives by mechanism, never by label."
)


# Variant registry for the harness.
VARIANTS: dict[str, str] = {
    "V1_evidence_gap": REWRITER_SYSTEM_PROMPT_V1_EVIDENCE_GAP,
    "V2_invalidation": REWRITER_SYSTEM_PROMPT_V2_INVALIDATION,
    "V3_contradiction": REWRITER_SYSTEM_PROMPT_V3_CONTRADICTION,
}
