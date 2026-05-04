# Methods Outline — Evidence-Grounded Benchmark Construction

> Drop-in outline for the Methods section. Numbered subsections map to the
> conventional structure (Overview → Inputs → Sampling → Generation →
> Validation → Statistics). Each bullet is a sentence-or-paragraph slot;
> fill in citations and dataset-specific numbers from
> `samples_balanced.summary.json` and `current_workflow_2026_04.md`.

---

## 3. Methods

### 3.1 Overview

- We construct an **evidence-grounded scientific benchmark** by lifting
  normalized knowledge-graph triples emitted by an upstream literature
  pipeline (`pubmed_graph`) into five complementary question formats:
  `claim_choice`, `boolean_support`, `two_hop_tail`, `essay`, and
  `experiment_code`.
- Unlike relation-label-driven benchmarks, our generator is **evidence-
  first**: every question is conditioned on (i) an explicit evidence
  profile of the underlying triple and (ii) a sandbox- and LLM-verified
  validation stack. This couples question wording, gating, and
  acceptance to the strength of the supporting literature rather than to
  the surface form of the relation.
- The pipeline is a single linear flow:
  *load → index → sample → profile → generate → validate → dedup →
  export*, executed in parallel with per-sample timeouts.
  (Figure: pipeline diagram from `current_workflow_2026_04.md`, lines 26–47.)

### 3.2 Inputs

- **Triples** (`normalized_triples.jsonl`): canonicalized
  `(head, relation, tail, evidence, doc_id, chunk_id)` tuples produced
  by an LLM-based extractor over biomedical chunks.
- **Chunks** (`chunks.jsonl`): word-window splits of paper full-text
  (180 words, 40-word overlap) used both as evidence carriers and as
  retrieval contexts for validation.
- **Live retrieval**: at validation time we additionally query PubMed
  E-utilities (`top_k = 3`) to obtain external evidence beyond the
  cached chunk pool.

### 3.3 Subgraph Sampling

- `sample_single_hop_subgraphs` enumerates one-hop subgraphs
  `(head, relation, tail)` keyed on a `uniqueness_key` to control
  duplication.
- `sample_two_hop_subgraphs` chains two single hops `A→B→C` whose middle
  node `B` is graph-unique, providing the substrate for `two_hop_tail`
  questions.
- A pre-filter caps how many candidates share the same uniqueness key
  (`DEFAULT_MAX_PER_UNIQUENESS_KEY = 3` in production).
- For `experiment_code`, the sampler emits **one candidate per
  difficulty level** (`easy`, `medium`, `hard`); their uniqueness keys
  are suffixed with `|difficulty=...` so deduplication preserves all three.

### 3.4 Evidence Profiling and Gating

- For each sampled subgraph, `evidence_profiler` computes
  - `relation_strength`: prior weight by relation label,
  - `hedge_score`: lexical hedge density in the evidence string,
  - and a combined **`evidence_strength ∈ {weak, medium, strong}`**.
- The strength **gates** the legal question types:

  | type | gating |
  |---|---|
  | `claim_choice` | any |
  | `boolean_support` | medium+ |
  | `two_hop_tail` | strong only (both hops independently supported) |
  | `essay` | any (wording adapts to strength) |
  | `experiment_code` | any (validated via sandbox) |

- The same strength **drives claim wording** in `evidence_claims.py`
  (e.g. *"suggests a reported association"* / *"supports a contextual
  relationship"* / *"supports that X relation Y"*), keeping all
  surface forms anchored to the actual evidence.

### 3.5 Question Generation

#### 3.5.1 Template-based types

- Four types (`claim_choice`, `boolean_support`, `two_hop_tail`,
  `essay`) are produced by `generator.build_question_sample`, which
  combines per-type templates (`templates.py`), evidence-strength-aware
  claim phrasings (`evidence_claims.py`), and graph-derived distractors.
- Distractors for `claim_choice` are sampled from same-type, same-
  context entities; the acceptance filter requires **≥ 2 viable
  options** so that single-distractor subgraphs are silently dropped.

#### 3.5.2 Programmatic `experiment_code` (Plan C)

- For `experiment_code`, `experiment_llm_generator.generate_experiment_via_llm`
  prompts an LLM (`intern-s1-pro`, OpenAI-compatible) to synthesize a
  bespoke spec per triple containing
  `{task_family, research_direction, data_code, main_code,
   incomplete_main_code, incomplete_functions, unit_tests}`.
- The spec is shape-validated and gated by a sandbox (Section 3.6.2);
  on rejection, the failure modes (compile error, expected vs. actual)
  are appended as feedback and the LLM is re-queried, up to **3 retries**.
- Difficulty is enforced by `experiments.difficulty.select_blank_targets`:
  `easy` blanks one helper, `medium` blanks all helpers, `hard`
  additionally blanks the orchestration `summarize_*` function.
- A `hybrid` mode falls back to four hardcoded blueprints
  (`biomarker_screening`, `differential_expression`, `dose_response`,
  with `pathway_activity` as a catch-all) when the LLM path is
  exhausted.

### 3.6 Validation Stack

The validator is two-stage: hard rule guardrails first, then optional
LLM-as-Judge cross-validation. Failure of either stage rejects the
sample.

#### 3.6.1 Rule-based guardrails (`validator.py`)

- `_evidence_supported`: triple-evidence string must be present in the
  cited chunk via strict substring **or** ≥ 80 % token overlap;
  skipped for `experiment_code` (grounded by sandbox).
- `_answer_unique`, `_answer_not_leaked` (with title-prefix stripping
  to avoid e.g. ACE2-style false positives), `_supports_minimum_double_check`
  (≥ 2 supporting evidences / 2 docs / 2 chunks),
  `_question_type_allowed_by_evidence`, `_experiment_metadata_complete`.

#### 3.6.2 Sandbox execution gate (`experiment_code` only)

- `sandbox_runner.evaluate_experiment_sample` bundles `data_code +
  main_code + unit_tests` into a single harness shipped to a remote
  Python sandbox via the `sandbox_fusion` library.
- We require the sample to satisfy **two** properties simultaneously:
  1. the **reference solution** passes *all* unit tests, and
  2. the **incomplete (blanked) solution** fails *at least one* unit
     test, so that the blanks are non-trivial.
- `sandbox_client.py` adds a TCP probe and a 20-s hard timeout
  (`sandbox_fusion` itself has none) so that a stuck server cannot hang
  the run; `worker_timeout`, `internal_error`, and `sandbox_disabled`
  all degrade gracefully to rule-only acceptance.

#### 3.6.3 LLM-as-Judge cross-validation

- For `claim_choice`, `boolean_support`, and `two_hop_tail`,
  `model_validator.judge_claim` evaluates the question against an
  evidence bundle assembled by `retrieval_validator` (local chunks +
  live PubMed top-`k`).
- For `essay`, `judge_essay` scores the reference answer.
- For `experiment_code`, `judge_experiment_code` is a **second-opinion
  judge** that reads the synthesized code together with the original
  scientific claim; this catches sandbox-passing samples that are
  off-topic or direction-flipped (e.g. inhibits ↔ activates) — failure
  modes that unit tests alone cannot detect.
- Verdicts: `supported` ⇒ accept; `insufficient_evidence` /
  `contradicted` ⇒ reject with the corresponding tag;
  `model_unavailable` ⇒ the run **silently degrades** to rule-based
  acceptance, with `validation_mode = degraded` recorded in the sample
  for downstream auditability.
- All judge calls are cached on a content-hash key
  (`ValidationCache`) to make re-runs cheap and prompt edits explicit.

### 3.7 Deduplication and Export

- After validation, `deduplicate_by_question` collapses samples sharing
  the same question text (so distractor ordering is part of identity).
- `exporter.py` writes (i) `question_samples.jsonl`, (ii) a
  `summary.json` reporting per-type counts, evidence-strength
  distribution, validation modes, and rejection reasons, and (iii) a
  full `run.log` for observability.

### 3.8 Implementation and Reproducibility

- **Parallelism**: non-experiment samples run with `workers = 8`;
  `experiment_code` is capped at `workers = 2` to respect LLM,
  sandbox, and GitHub rate limits.
- **Per-sample timeout** (`--sample-timeout`, 600 s in production):
  running futures are abandoned, queued ones cancelled; **three
  consecutive timeouts abort the batch** to prevent service-outage
  avalanches.
- **Robustness**: `PYTHONUNBUFFERED=1`, `setsid --wait`, an early-boot
  heartbeat, and per-phase logging in `build_experiment_sample` close
  out a class of silent failures observed in earlier versions.
- Code, prompts, and configuration files are released under the
  `question_generation` repository; full reproduction requires the
  upstream `pubmed_graph` outputs and an `intern-s1-pro`-compatible
  endpoint for both generation (Plan C) and judging.

### 3.9 Released Benchmark Statistics

> Numbers below are illustrative; replace with the final shipped subset
> (e.g. the 800-question `samples_balanced.jsonl` over `proteinlmbench_full_graphbench`).

- Sampled subgraphs: 2,000; accepted questions after full validation:
  **1,368** (pass rate 68.6 %).
- Per-type composition (post-dedup): `two_hop_tail` 1,225 /
  `experiment_code` 73 / `claim_choice` 32 / `essay` 26 /
  `boolean_support` 12.
- `experiment_code` generation source: LLM (Plan C) **68.5 %**,
  template fallback 31.5 %; difficulty mix easy/medium/hard ≈ 28/21/24;
  54 unique blueprint keys.
- Rejection breakdown (top reasons): `no_evidence_alignment` 403
  (rule, upstream paraphrase drift), `reference_solution_failed_unit_tests`
  179 (sandbox; most self-heal via retry),
  `model_insufficient_evidence` 99 (LLM judge),
  `model_contradicted` 91 (LLM judge),
  `llm_judge_code_claim_misaligned/contradicts_claim` 18
  (experiment-code judge — would have entered the benchmark silently
  in the pre-Plan-C era).
- Support-score distribution among accepted samples: 88.8 % in
  [0.9, 1.0], 11.2 % in [0.7, 0.9), none below 0.7.

### 3.10 Limitations

- `no_evidence_alignment` accounts for ~64 % of rejections and traces
  to LLM paraphrase drift in the upstream extractor's
  `triple.evidence` field — addressed in `pubmed_graph`, not here.
- The graph topology of the current dataset over-produces two-hop
  paths, biasing the unbalanced benchmark toward `two_hop_tail`; the
  released benchmark is therefore **stratified-balanced** by
  `(question_type × hop_class × evidence_strength)` to a fixed budget
  (Section 3.7).
- LLM judging incurs a non-trivial latency and cost; degraded mode is
  recorded in `validation_mode` so downstream analyses can stratify.

---

## Suggested figures and tables

1. **Figure 1** — Pipeline diagram (lift the ASCII flow from
   `current_workflow_2026_04.md` lines 26–47 into a vector schematic).
2. **Figure 2** — Validator stack as a swim-lane (rule → sandbox →
   LLM judge, with reject paths annotated by tag).
3. **Table 1** — Question types × gating × validator stack
   (Section 3.4 + 3.6).
4. **Table 2** — Released-benchmark composition: per-type, per-evidence-
   strength, per-hop-class counts (from `samples_balanced.summary.json`).
5. **Table 3** — Rejection-reason breakdown by validation layer (rule /
   sandbox / LLM judge), as in Section 3.9.
6. **Figure 3** — Distribution of LLM-judge `support_score` across
   accepted samples; histogram from `summary.json`.

## Drop-in citations / references

- Upstream pipeline & GraphML construction: cite the parent
  `pubmed_graph` system (workflow notes:
  `../workflow_mds/pubmed_graph_workflow.md`).
- Knowledge-graph extraction LLM: `intern-s1-pro` via the OpenAI-
  compatible Intern Chat API.
- Evidence retrieval: PubMed E-utilities + local chunk index.
- Sandbox execution: `sandbox_fusion` library.
- LLM-as-Judge methodology: standard reference; clarify in the paper
  that we use **two judges** for `experiment_code` (the unit-test
  sandbox + the cross-checking LLM) precisely because either alone
  admits a documented failure mode.
