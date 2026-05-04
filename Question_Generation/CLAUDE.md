# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Scope note: this directory is a standalone Python package, separate from the parent `pubmed_graph` pipeline. The parent `../CLAUDE.md` explicitly excludes it. It does, however, *consume* the parent pipeline's outputs (`normalized_triples.jsonl` + `chunks.jsonl`) and imports `pubmed_graph.pubmed_client.PubMedClient` for live evidence retrieval, so it is not fully decoupled — `PYTHONPATH` must include the parent dir.

## Purpose

`question_generation` is an **evidence-first scientific benchmark generator + hybrid model validation framework**. Given normalized KG triples and source chunks produced by `pubmed_graph`, it emits fact-grounded benchmark questions (`question_samples.jsonl`) with provenance, designed to test LLMs on scientific fact understanding.

Five question types are supported (`config.DEFAULT_SUPPORTED_QUESTION_TYPES`):

| type | answer form | gating |
|---|---|---|
| `claim_choice` | 4 options, claim text | all evidence strengths |
| `boolean_support` | Supported / Not supported | evidence_strength ≥ medium |
| `two_hop_tail` | middle entity of A→B→C | strong only, both hops independently supported |
| `essay` | free-text reference answer, judged by LLM-as-Judge | all (essay adapts to strength) |
| `experiment_code` | `data_code` + `incomplete_main_code` + `unit_tests`, GitHub-assisted | rule_only validation |

## Architecture (big picture)

The pipeline is a single linear flow assembled in `cli.py:main`:

```
load_triples + load_chunks  →  build_index
        ↓
sample_single_hop_subgraphs  (+ sample_two_hop_subgraphs if two_hop_tail requested)
        ↓
for each subgraph:
    build_question_sample   (generator.py)   ← uses evidence_profiler + evidence_claims + templates
    validate_sample          (validator.py)   ← rule guardrails, then optional model verdict
        ↓
deduplicate_by_question  →  export_samples + export_summary
```

Key architectural ideas that span multiple files:

1. **Evidence-first, not relation-first.** Relation labels like `associated_with` are too weak to drive question wording. Instead, `evidence_profiler.py` computes `relation_strength`, `hedge_score`, and a combined `evidence_strength ∈ {weak, medium, strong}` for every sampled subgraph. That strength then **gates** which question types are legal (see table above) and **drives** claim wording in `evidence_claims.py` (`"suggests a reported association"` → `"supports a contextual relationship"` → `"supports that X relation Y"`). The `templates.py` and `generator.py` consume the same strength to phrase questions cautiously vs. assertively. If you change profiler thresholds, downstream wording shifts silently — re-check sample outputs.

2. **Two-stage validation.** `validator.py` first runs rule-based hard guardrails: evidence string must be findable in the chunk text; ≥ 2 supporting evidence / 2 docs / 2 chunks; no answer leakage in the question (skipped for essay/boolean/claim_choice); two-hop middle-node uniqueness; question_type must be allowed by the evidence profile. If those pass and `--validation-mode hybrid_model --validator-enabled`, it then calls `model_validator.judge_claim()` (for claim_choice/boolean/two_hop) or `judge_essay()` (for essay) using `intern-s1-pro` over an evidence bundle assembled by `retrieval_validator.py` (local chunks + live PubMed via `PubMedClient`). Cache results via `validation_cache.ValidationCache` keyed by `--validation-cache-dir`.

3. **Degraded mode is silent.** If model creds are missing/empty or the API call fails, validation does **not** raise — it keeps the rule-based result, sets `validation_mode = degraded`, and `double_checked = False`. Inspect `quality.validation_status` and `grounding.validation_mode` in the output to detect this.

4. **`experiment_code` is a different beast.** `experiment_generator.py` + the `experiments/` package + `github_tools.py` rewrite a triple into a small biology programming task. Blueprints live in `experiments/blueprints/*.py` and self-register against `experiments.registry.REGISTRY` on import; dispatch is predicate-based on a `BlueprintContext(head, head_type, relation, tail, tail_type, evidence, difficulty)`, with `pathway_activity` as the explicit fallback. Difficulty (`easy` / `medium` / `hard`) is resolved by `experiments.difficulty.select_blank_targets` — `easy` blanks one helper, `medium` blanks all listed helpers, `hard` additionally blanks the orchestration `summarize_*` function declared in each blueprint's `hard_extra_blanks`. The CLI flag `--experiment-difficulty {easy,medium,hard,mixed}` controls this; `mixed` causes `sampler.sample_single_hop_subgraphs` to emit one candidate per difficulty (uniqueness keys are suffixed with `|difficulty=...` so dedup keeps all three). Reference complete code lives in `answer.text`; the masked version + `unit_tests` + `github_references` + `experiment_blueprint` + `experiment_difficulty` + `sandbox_evaluation` live in `metadata`. `summary.json` reports `experiment_blueprint_breakdown` keyed by `"<blueprint>:<difficulty>"`.

5. **Sandbox-backed validation for `experiment_code`.** `sandbox_runner.evaluate_experiment_sample` is invoked at generation time inside `build_experiment_sample`. It bundles the blueprint's `data_code` + the candidate `main_code` + `unit_tests` into a single Python script and ships it to a remote sandbox via `sandbox_client.run_code_sync` (interface mirrored from `RL-Factory/envs/tools/python.py`, uses the `sandbox_fusion` library). It runs **two** checks: the reference solution must pass *all* unit tests, and the masked prompt must fail at least one (otherwise the blanks are pointless — the question is trivial). The result lives at `metadata.sandbox_evaluation` with shape `{reference, incomplete, verdict, rejection_reasons}`. `validator._validate_experiment_sample` consumes that field instead of short-circuiting: `verdict='passed'` upgrades the sample to `validation_mode=sandbox`, `verdict='rejected'` rejects with `reference_solution_failed_unit_tests` / `incomplete_code_already_passes_unit_tests`. **Crucially, when SANDBOX_HOST is empty (the default) every sandbox call returns `sandbox_disabled` and the validator silently falls back to the prior rule-only behaviour** — so existing runs are not broken until you fill in `SANDBOX_HOST` in `sandbox_client.py` (or set `QG_SANDBOX_HOST=...`). The sandbox host needs the standard scientific Python stack (pandas / numpy / sklearn) to actually run the blueprint code.

6. **Output schema is in `models.py`.** `QuestionSample`, `SupportingTriple`, `Option`, etc. are plain dataclasses. `exporter.py` serializes via `__dict__`, and `dedup.py` dedupes on question text *after* validation but *before* export — so `len(samples)` in the summary reflects the post-dedup count.

## Running

The package is **not** installed; run it as a module with `PYTHONPATH` pointing at the parent so `pubmed_graph` resolves:

```bash
PYTHONPATH="/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa" \
python -m question_generation.cli \
  --triples  /path/to/normalized_triples.jsonl \
  --chunks   /path/to/chunks.jsonl \
  --output   out/question_samples.jsonl \
  --summary-output out/summary.json \
  --question-types claim_choice boolean_support two_hop_tail essay \
  --validation-mode rule_only
```

Hybrid model validation (adds essay LLM-as-Judge):

```bash
PYTHONPATH=".." python -m question_generation.cli \
  --triples ... --chunks ... --output ... \
  --validation-mode hybrid_model \
  --validator-enabled \
  --validator-model    "intern-s1-pro" \
  --validator-base-url "https://chat.intern-ai.org.cn/api/v1/" \
  --validator-api-key  "$INTERN_API_KEY" \
  --retrieval-top-k 3 \
  --validation-cache-dir out/.val_cache
```

`experiment_code` runs in `rule_only` mode and additionally uses GitHub search (`--github-search-language Python`, `--github-search-per-page 3`).

There is no test suite, linter, or build step. Smoke checks:

```bash
python -m compileall .                  # parse check
# then a CLI run against a small chunks/triples fixture and inspect summary.json
```

## Important gotchas

- **`PYTHONPATH` must include the parent dir.** `cli.py` does `from pubmed_graph.pubmed_client import PubMedClient` at import time — even `--validation-mode rule_only` runs will `ImportError` without it.
- **Rule guardrail "evidence in chunk text" is a substring match.** If `pubmed_graph` upstream changes how it stores `triple.evidence`, samples will be silently rejected here. Check `summary.json → validation` counts before debugging anywhere else.
- **`DEFAULT_MAX_PER_UNIQUENESS_KEY = 1`** in `config.py` caps how many samples share the same `subgraph.uniqueness_key`. Raising `--max-samples` alone won't yield more questions if the underlying graph is small — bump this constant.
- **Acceptance filter in `cli.py:main` requires `len(options) >= 2`** for non-essay/non-experiment_code samples. A subgraph that produces only one viable distractor will be silently dropped post-validation.
- **Post-validation dedup** (`deduplicate_by_question`) runs on question text. Two samples differing only in distractor ordering can collapse to one — ordering is therefore part of the identity.
- **Model validation results are cached by `ValidationCache`** keyed by claim+evidence — if you change `validation_prompts.py` or the judging logic, blow away `--validation-cache-dir` or you'll keep getting stale verdicts.

## Reference docs

- `workflow_mds/question_generation_workflow.md` — most current source of truth for evidence profiling rules, gating thresholds, prompt design, and verified intern-s1-pro results on `proteinlmbench_full_graph_v1`.
- `workflow_mds/experiment_code_agent_workflow.md` — design notes for the `experiment_code` GitHub-assisted path.
- `AGENTS.md` — short style/contribution guide (4-space indent, snake_case, dataclasses, no formatter configured).
