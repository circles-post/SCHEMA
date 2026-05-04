# question_generation — current workflow (2026-04)

This document is the **current ground truth** for how `question_generation`
builds benchmark questions on top of `pubmed_graph` output. It supersedes
any claims in `question_generation_workflow.md` or
`question_generation_per_type_flow.md` that contradict it — those older
docs describe the template-only era. The big recent changes:

1. **Plan C (LLM per-triple generation)** — `experiment_code` is no longer
   bound to four hardcoded blueprints. `intern-s1-pro` synthesizes a
   bespoke `(data_code, main_code, unit_tests)` per triple, then a sandbox
   gate validates it. `hybrid` mode keeps the blueprint path as a fallback.
2. **LLM cross-validation for experiment_code** — `judge_experiment_code`
   (new) is a second-opinion judge that reads the code + claim and rejects
   code that passes sandbox but is off-topic / direction-flipped.
3. **Parallel execution with per-sample timeout** — non-experiment samples
   run with `workers=8`; experiment_code caps at `workers=2` due to LLM/
   sandbox/GitHub rate limits. A stuck sample can't hang the pool.
4. **Silent-failure fixes** — `PYTHONUNBUFFERED=1`, `setsid --wait`,
   early-boot heartbeat in `cli.py`, TCP probe before `sandbox_fusion`,
   hard timeout wrapper inside `run_code_sync`, and per-phase logging in
   `build_experiment_sample`.

## High-level flow

```
pubmed_graph/
  normalized_triples.jsonl + chunks.jsonl
         │
         ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ question_generation.cli                                        │
 │   ├─ phase 1: load_triples                                     │
 │   ├─ phase 2: load_chunks                                      │
 │   ├─ phase 3: build_index (entities_by_type, chunks_by_id, …)  │
 │   ├─ phase 4: sample_single_hop + sample_two_hop_subgraphs     │
 │   ├─ phase 5: pre-filter by uniqueness_key                     │
 │   │              ├─ non-experiment batch (workers=8, parallel) │
 │   │              │     └─ build_question_sample + validate     │
 │   │              └─ experiment_code batch (workers=2)          │
 │   │                    └─ build_experiment_sample + validate   │
 │   └─ phase 6: deduplicate_by_question → export_samples         │
 └────────────────────────────────────────────────────────────────┘
         │
         ▼
  question_samples.jsonl + summary.json + run.log
```

## Question types (5)

| type | form | gating (evidence strength) | validator stack |
|---|---|---|---|
| `claim_choice` | 4 options, pick the best-supported claim | any | rule → `judge_claim` |
| `boolean_support` | Supported / Not supported | medium+ | rule → `judge_claim` |
| `two_hop_tail` | fill in the middle entity of A→B→C | strong only | rule → `judge_claim` |
| `essay` | open-ended, LLM judges reference answer | any | rule → `judge_essay` |
| `experiment_code` | Python code + unit tests, sandbox-gated | any | rule → sandbox → `judge_experiment_code` |

## The validator stack in detail

```
validate_sample(sample, validation_mode, model_config, …)
│
├─ validate_sample_rule_based(index, sample)
│     ├─ _evidence_supported      (strict substring OR ≥80% token overlap)
│     │        └─ SKIPPED for experiment_code (grounded via sandbox)
│     ├─ _type_pattern_allowed
│     ├─ _answer_unique
│     ├─ _answer_not_leaked        (strips "Based on the reported evidence from '<title>', "
│     │                             prefix before substring check — fixes ACE2-style
│     │                             false positives where the title contains the answer)
│     ├─ _supports_minimum_double_check
│     ├─ _question_type_allowed_by_evidence
│     └─ _experiment_metadata_complete
│
├─ experiment_code → _validate_experiment_sample
│     ├─ read metadata.sandbox_evaluation (written by build_experiment_sample)
│     ├─ sandbox verdict == "passed"?
│     │        NO → reject with sandbox rejection_reasons
│     │        YES →
│     │              support_score = ref.passed / ref.total
│     │              contradiction_count = inc.passed
│     │
│     └─ if validation_mode == hybrid_model AND model_config.enabled:
│              judge_experiment_code(
│                  scientific_claim = f"{head} {relation} {tail}",
│                  main_code, unit_tests, incomplete_functions,
│                  evidence_bundle = retrieve_evidence_bundle(local+pubmed)
│              )
│              ├─ verdict == supported
│              │     → pass, vmode = sandbox+hybrid_model
│              ├─ verdict == insufficient_evidence
│              │     → reject, reason = llm_judge_code_claim_misaligned
│              ├─ verdict == contradicted
│              │     → reject, reason = llm_judge_code_contradicts_claim
│              └─ model_unavailable (API down)
│                    → keep sandbox pass, vmode = sandbox, graceful degradation
│
├─ non-experiment types, hybrid_model mode →
│     validate_sample_model_based
│     ├─ retrieve_evidence_bundle (local chunks + live PubMed top_k)
│     ├─ serialize_evidence_bundle
│     ├─ judge_essay (for essay) OR judge_claim (all others)
│     ├─ cached via ValidationCache keyed on (question, answer, bundle hash)
│     └─ _apply_model_verdict
│              supported        → pass, vmode = hybrid_model
│              insufficient     → reject, reason = model_insufficient_evidence
│              contradicted     → reject, reason = model_contradicted
│              model_unavailable → vmode = degraded, NOT rejected (soft pass)
│
└─ non-experiment types, rule_only mode →
      return rule result unchanged
```

## experiment_code generation pipeline (Plan C)

```
build_experiment_sample(subgraph, mode ∈ {template, llm, hybrid}):
│
├─ if mode in {llm, hybrid}:
│     generate_experiment_via_llm(head, rel, tail, types, evidence, difficulty)
│     │
│     │  [retry loop, up to 3 attempts]
│     │  └─ InternChatClient.chat_json(
│     │         build_system_message() + build_user_message(...)
│     │      )
│     │       → spec = {task_family, research_direction, data_code,
│     │                 main_code, incomplete_main_code,
│     │                 incomplete_functions, unit_tests, ...}
│     │
│     │  _validate_shape(spec)     # must have load_*, summarize_*, etc.
│     │
│     │  sandbox gate:
│     │      evaluate_experiment_sample(
│     │          data_code, main_code, incomplete_main_code, unit_tests
│     │      )
│     │      ├─ reference run: must pass ALL unit_tests
│     │      ├─ incomplete run: must FAIL at least one unit_test
│     │      └─ verdict = passed | rejected | skipped
│     │
│     │  on sandbox rejection:
│     │      build detailed feedback_lines from test_results
│     │        (expected vs actual, compile errors, stderr)
│     │      retry with:
│     │          {"role": "assistant", "content": <prior spec JSON>},
│     │          {"role": "user", "content": <feedback>}
│     │
│     ├─ success → use LLM spec, blueprint_name = f"llm::{task_family}"
│     └─ exhausted → llm_spec = None
│
│  if mode == "llm" and llm_spec is None:
│      → _build_rejected_llm_sample("llm_generation_failed")
│  else (template or hybrid fallback):
│      dispatch_blueprint(context)  # predicate-driven, type-aware
│      build_incomplete_code(main_code, blank_targets)
│      evaluate_experiment_sample(...) on blueprint
│
├─ _build_github_reference_pack (optional, rate-limited)
│     └─ github_tools.search_github_code + code_relevance.select_relevant_functions_cached
│
└─ render question + assemble QuestionSample
        metadata.generation_source ∈ {llm, template}
        metadata.generation_mode ∈ {template, llm, hybrid}
        metadata.generation_attempts
        metadata.experiment_blueprint  (e.g. "llm::otub2_stroke_inhibition")
        metadata.sandbox_evaluation.verdict
```

### Blueprint predicates (post-refactor)

Hardcoded blueprints now use stricter `(relation, head_type, tail_type)`
predicates, so nothing gets force-routed to `dose_response`:

| blueprint | relation(s) | head type | tail type | fallback? |
|---|---|---|---|---|
| `biomarker_screening` | `associated_with` | biomarker / gene / protein / metabolite | disease / condition / phenotype | no |
| `differential_expression` | `upregulated_in`, `downregulated_in`, `overexpressed_in` | gene / protein / mRNA / ncRNA | disease / condition / phenotype | no |
| `dose_response` | `inhibits` | drug / compound / inhibitor / smallmolecule | protein / kinase / enzyme / receptor | no |
| `pathway_activity` | — | — | — | **YES (fallback)** |

Most real triples now fall through to `pathway_activity`; the old "force
dose_response on `OTUB2 inhibits stroke`" misrouting is gone. Plan C LLM
generation is what actually produces bespoke code per triple.

## Sandbox layer

- `sandbox_client.py` wraps `sandbox_fusion.run_code()` with:
  - TCP probe (5 s) before binding endpoint
  - Eager `NO_PROXY` injection at module load
  - **Hard timeout** via a `ThreadPoolExecutor(1).submit(...).result(timeout=20s)`
    — `sandbox_fusion` has no internal timeout, so a hung server would
    block forever without this. On timeout we return `reason=worker_timeout`.
- `sandbox_runner.py` bundles `data_code + main_code + unit_tests` into a
  harness, imports it as `main_en` under a synthetic `data_en` module,
  runs each unit test through `summarize_*` by default (or a named helper),
  and returns per-test `{passed, actual, error}`.
- **Host**: default `100.99.239.71:8080` (old `100.99.100.95` is dead).
- `worker_timeout` / `internal_error` / `sandbox_disabled` are all treated
  as "sandbox unavailable" → validator falls back to rule-only instead of
  marking every sample as `reference_solution_failed_unit_tests`.

## LLM clients used

| purpose | function | client | prompt |
|---|---|---|---|
| Plan-C spec generation | `experiment_llm_generator.generate_experiment_via_llm` | `InternChatClient` | `_build_system_message / _build_user_message` |
| experiment_code cross-validation | `model_validator.judge_experiment_code` | `InternChatClient` | `validation_prompts.build_experiment_judge_messages` |
| claim / boolean / two-hop judge | `model_validator.judge_claim` | `InternChatClient` | `validation_prompts.build_validation_messages` |
| essay judge | `model_validator.judge_essay` | `InternChatClient` | `validation_prompts.build_essay_judge_messages` |
| GitHub code-function relevance | `code_relevance.select_relevant_functions_cached` | `InternChatClient` | inline prompt in `code_relevance.py` |

All of the above resolve credentials from (in order): `cfg["api_key"]`,
`INTERN_API_KEY`, `OPENAI_API_KEY`. The base URL defaults to
`https://chat.intern-ai.org.cn/api/v1/`. The model is
`intern-s1-pro` unless overridden. **The Intern API is OpenAI-compatible**,
so `OPENAI_API_KEY` here points at the Intern key — it does NOT reach
`api.openai.com`.

## Parallelism

```
work_items = [(idx, subgraph), ...]
non_experiment = [... where type != experiment_code]     # len ≈ 1900
experiment     = [... where type == experiment_code]     # len ≈ 100

_run_batch(non_experiment, workers=args.workers)         # typ. 8
_run_batch(experiment,     workers=min(args.workers, 2)) # GitHub rate-limit cap
```

Inside `_run_batch` there is a per-sample hard timeout
(`--sample-timeout`, default 180 s, production uses 600 s). On timeout:

- futures that are `running()` → marked `TIMEOUT-ABANDONED-RUNNING`,
  counted as reject, the background thread is abandoned (`pool.shutdown(wait=False)`)
- futures that are not running yet → `future.cancel()`, counted as reject
  (no work wasted)
- **three consecutive timeouts with no progress → abort the whole batch**
  and mark everything remaining as rejected. Prevents "96 experiments ×
  180 s each" avalanches when a downstream service is catatonic.

## CLI + shell invocation

```
./scripts/run_graphbench_questions_full.sh
```

This is the one-shot production wrapper. It:

1. `unset QG_SANDBOX_HOST` (prevents stale env leaking old IP)
2. `source triple_extraction_env.sh` (loads `OPENAI_API_KEY` / `INTERN_API_KEY`)
3. Fails fast if no API key present
4. Wipes `__pycache__` so recent edits take effect
5. Calls `./scripts/run_graphbench_questions.sh` with:
   ```
   --max-samples 2000
   --max-per-uniqueness-key 3
   --question-types claim_choice boolean_support two_hop_tail essay experiment_code
   --experiment-difficulty mixed
   --experiment-generation-mode hybrid
   --llm-code-selection auto
   --validation-mode hybrid_model
   --validator-enabled
   --validation-cache-dir benchmark_runs/.../.qg_val_cache
   --retrieval-top-k 3
   --workers 8
   --sample-timeout 600
   --force
   ```
6. Dumps post-run stats (summary.json, per-type counts, vmode breakdown,
   experiment_code `generation_source` distribution).

The nested shell script (`run_graphbench_questions.sh`) handles:

- proxy policy (`NO_PROXY` append for sandbox/embedding hosts)
- `PYTHONUNBUFFERED=1` + `python -u` (line-buffered stderr)
- `setsid --wait` + `trap '' HUP` (detach from TTY → SIGHUP-proof)
- Passes `--log-file` so logs stream to both stderr and a file

## Observability

Every run writes to
`benchmark_runs/<run>/question_generation/run.log`. Key grep patterns:

```bash
# phase progress
grep -E '\[phase [0-9]/6\]' run.log

# per-sample outcomes
grep -E '  \[[0-9]+/[0-9]+\]' run.log | head

# LLM generation outcomes
grep -E 'llm_generate (succeeded|failed)|sandbox rejected LLM spec|feedback:' run.log

# experiment phase timing
grep -E 'build_experiment_sample|sandbox_eval|github_refs' run.log | head

# validation-mode breakdown across accepted samples
grep -oP 'vmode=\w+' run.log | sort | uniq -c

# rejection-reason breakdown
grep -oP "reasons=\[[^\]]*\]" run.log | sort | uniq -c | sort -rn
```

## Full-run statistics (2026-04-15 00:39 → 01:58, 79 min)

| metric | value |
|---|---|
| triples | 4488 |
| chunks | 11224 |
| sampled_subgraphs | 2000 |
| accepted_questions | **1368** |
| pass rate | 68.6% |
| total wall clock | 79 min 9 s |

### per-type counts (after dedup)

| type | count | validation_mode | all pass via LLM judge? |
|---|---|---|---|
| two_hop_tail | 1225 | hybrid_model | ✅ |
| experiment_code | 73 | sandbox+hybrid_model | ✅ (sandbox + judge) |
| claim_choice | 32 | hybrid_model | ✅ |
| essay | 26 | hybrid_model | ✅ |
| boolean_support | 12 | hybrid_model | ✅ |

### experiment_code breakdown

- `generation_source = llm`: **50** (68.5%)
- `generation_source = template` (hybrid fallback): 23
- Difficulty mix: easy 28 / medium 21 / hard 24
- Unique blueprint keys: **54** (e.g. `llm::otub2_stroke_inhibition`,
  `llm::smc56_rloop_degradation`, `llm::wnt_hiden_pathway`, ...)

### rejection reasons

| reason | count | layer |
|---|---|---|
| `no_evidence_alignment` | 403 | rule (upstream paraphrase drift — not a bug in QG) |
| `reference_solution_failed_unit_tests` | 179 | sandbox (LLM retry loop, most self-heal) |
| `model_insufficient_evidence` | 99 | LLM judge (claim/essay) |
| `model_contradicted` | 91 | LLM judge (claim/essay) |
| `llm_judge_code_claim_misaligned` (+ tags) | 14 | **LLM judge (experiment)** ⭐ |
| `llm_judge_code_contradicts_claim` (+ tags) | 4 | **LLM judge (experiment)** ⭐ |
| `non_unique_answer` / `answer_leakage` / misc | 12 | rule |

The 18 `llm_judge_code_*` rejects are samples that passed sandbox but
were caught by the new cross-validation — off-topic code, direction-
flipped code, trivial blanks. These would have silently entered the
benchmark in the pre-Plan-C era.

### support_score distribution

| bucket | count |
|---|---|
| 0.9–1.0 | 1215 (88.8%) |
| 0.7–0.9 | 153 (11.2%) |
| < 0.7 | 0 |

### no timeouts / stuck futures in this run

## Known gaps & next steps

1. **`no_evidence_alignment` accounts for 64 % of all rejects** — root cause
   is in `pubmed_graph/triple_extraction.py`, where `triple.evidence` is
   an LLM paraphrase of the chunk rather than a verbatim substring. Fix
   is out of scope for `question_generation` (cross-project change).
2. **`two_hop_tail` dominates** (89.5 %) because the graph sampler yields
   many more two-hop paths than one-hop candidates for this dataset.
   Consider capping two_hop_tail with a per-type cap in `cli.py` if the
   benchmark mix needs to be more balanced.
3. **experiment_code LLM first-attempt pass rate ≈ 56 %**, recovered to
   76 % after 3 retries and hybrid template fallback. Could be improved
   by (a) giving the LLM the harness signature explicitly up-front so the
   `kwargs` mismatch on attempt 1 disappears, or (b) raising retry count
   to 4–5 at the cost of latency.
4. **Experiment judge prompt tuning**. Currently rejects 18 samples for
   `code_off_topic` / `direction_flipped`. Manual review will tell
   whether this is the right sensitivity or too strict.
5. **Hybrid fallback to template + judge rejection** — when LLM generation
   fails 3× and we fall back to `pathway_activity` template, the judge
   almost always rejects it as `code_off_topic`. In `llm`-only mode we'd
   save the template build + sandbox + judge cost by rejecting
   immediately. Consider making `hybrid` mode skip the template fallback
   entirely once LLM is known-failed (or count `hybrid` template results
   as pre-rejected before running judge).

## File map

```
question_generation/
├── cli.py                      # entrypoint, phases 1–6, parallel workers
├── config.py                   # DEFAULT_* knobs (max_samples, vmode, …)
├── generator.py                # build_question_sample (routes by type)
├── experiment_generator.py     # build_experiment_sample (Plan C + template)
├── experiment_llm_generator.py # NEW: generate_experiment_via_llm + retry
├── sampler.py                  # sample_single_hop / sample_two_hop
├── indexing.py                 # QuestionGenerationIndex
├── templates.py                # non-experiment question templates
├── evidence_claims.py          # evidence-strength-aware claim wording
├── evidence_profiler.py        # relation_strength + hedge_score + gating
├── validator.py                # validate_sample_rule_based + experiment + model
├── model_validator.py          # judge_claim / judge_essay / judge_experiment_code
├── validation_prompts.py       # intern-s1-pro system/user prompts
├── validation_cache.py         # disk cache keyed on prompt payload
├── sandbox_client.py           # sandbox_fusion wrapper + TCP probe + hard timeout
├── sandbox_runner.py           # harness builder + evaluate_experiment_sample
├── retrieval_validator.py      # retrieve_evidence_bundle (local + live PubMed)
├── code_relevance.py           # GitHub function-level LLM filter
├── github_tools.py             # search_github_code / rate-limit handling
├── experiments/
│   ├── registry.py             # predicate-driven BlueprintRegistry
│   ├── difficulty.py           # select_blank_targets (easy/medium/hard)
│   └── blueprints/
│       ├── biomarker_screening.py
│       ├── differential_expression.py
│       ├── dose_response.py
│       └── pathway_activity.py     # fallback blueprint
├── dedup.py                    # deduplicate_by_question
├── exporter.py                 # JSONL + summary writer
└── models.py                   # QuestionSample dataclass
```
