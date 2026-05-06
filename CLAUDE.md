# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`SCHEMA_NIPS/` is the umbrella project that holds **four loosely-coupled subprojects**, each its own Python codebase with its own `CLAUDE.md`. They form a single research pipeline — KG construction → question generation → model evaluation → counterfactual / repair analysis — but each step writes JSONL artifacts that the next step reads, so they are normally run independently.

```
Benchmark_Construction/   biomedical literature → normalized KG (.graphml + triples + chunks)
        │ normalized_triples.jsonl + chunks.jsonl
        ▼
Question_Generation/      KG triples → benchmark question samples (claim_choice, boolean,
        │                              two_hop_tail, essay, experiment_code)
        │ question_samples.jsonl
        ▼
Evaluation/               run LLMs (Boyue / Intern / OpenAI-compatible) over the bench,
        │                 score answers, optional halu (hallucination) post-hoc analysis
        │ scored_results.jsonl + trajectory.jsonl
        ▼
Counterfactual_Reasoning/ AGDebugger fork: rerun wrong answers with an LLM-driven
                          claim-extract / judge / edit-and-revert repair loop
```

The top-level directory is just a container. There is no top-level package, no shared `pyproject.toml`, no shared test suite. Each subproject is independent — you almost always `cd` into one of them before doing anything.

## Per-subproject entry points

When working in a subproject, **read its `CLAUDE.md` first** — they hold the architecture details and gotchas that this file deliberately does not duplicate.

| Directory | Per-project guide | Run as |
|---|---|---|
| `Benchmark_Construction/` | `Benchmark_Construction/CLAUDE.md` | `python literature_pipeline.py --config pipeline_config.*.json ...` |
| `Question_Generation/` | `Question_Generation/CLAUDE.md` (also `AGENTS.md`) | `python -m question_generation.cli ...` (run from parent dir, see below) |
| `Evaluation/` | (no project CLAUDE.md — see `scripts/run_eval.sh`, `halu/cli.py`) | `python -m evaluation.runner ...` then optionally `python -m evaluation.halu.cli ...` |
| `Counterfactual_Reasoning/` | `Counterfactual_Reasoning/CLAUDE.md` (also `DEV.md`, `README.md`) | `bash run_with_models.sh ...` / `bash run_parallel.sh ...` |

`Evaluation/halu/` is a hallucination-detection post-processor that reads the runner's `trajectory.jsonl` + `scored_results.jsonl`, extracts claims per agent step, gathers evidence (currently only via `supporting_chunk` matching — layers 2–4 are stubbed), and judges per-bucket with `intern-s1-pro`. It writes `halu_results.jsonl` + `halu_summary.json` next to the eval run.

## Cross-subproject `PYTHONPATH` coupling (important)

Several subprojects import each other at runtime. The repo root is **not** on `sys.path` automatically; runners either insert it themselves or the user has to set `PYTHONPATH`.

- `Question_Generation/cli.py` does `from pubmed_graph.pubmed_client import PubMedClient` at import time. Even rule-only runs `ImportError` without `PYTHONPATH` including `Benchmark_Construction/`.
- `Evaluation/runner.py` and `Evaluation/halu/cli.py` insert their parent dir into `sys.path` so `evaluation.*`, `pubmed_graph.*`, `question_generation.*` all resolve regardless of CWD — this requires `pubmed_graph/` and `question_generation/` to be siblings under the same parent dir as `evaluation/` (i.e. all four subprojects sit side-by-side under `SCHEMA_NIPS/`). If you `mv` directories, those `sys.path.insert` lines stop resolving.
- `Counterfactual_Reasoning/run_dataset_autodebug.py` hardcodes external sibling repos (`AISci/datasets`, `AISci/ToolUniverse`, `AISci/sciverse`) that are **not** in this tree — it cannot run end-to-end without those.

## Configuration / secrets

Each subproject loads its own `.env` next to its code (see `.env.example` files). They are not unified — `Benchmark_Construction/.env` uses `OPENAI_*` aliases, `Evaluation/.env` uses `BOYUE_*` + `INTERN_*` + `JUDGE_*` + `SCIVERSE_*` + `BRIGHT_DATA_*`, `Question_Generation/.env` uses `GITHUB_TOKEN` (for the `experiment_code` GitHub-search path), `Counterfactual_Reasoning/` reads its credentials from `run_with_models.sh` env vars routed through `model_routing.py`.

`.env` files are gitignored at the root (`**/.env`); `.env.example` and `.env.template` are committed as templates. All previously-hardcoded absolute paths (conda envs, model weights, sibling repos, embedding service IPs) have been moved into env vars (`PYTHON`, `CONDA_BASE`, `AGDEBUGGER_CONDA_ENV`, `TOOLUNIVERSE_DIR`, `SCIVERSE_DIR`, `AGDEBUGGER_DATASETS_DIR`, etc.) — set them in your shell or in the matching `.env` before launching.

## Tests / lint / build

There is no repo-wide test suite, linter, or CI. Per subproject:

- `Benchmark_Construction/`, `Question_Generation/`, `Evaluation/` — no test suite, no formatter; "tests" are smoke runs validated by inspecting the generated `phase_summary.json` / `summary.json` / `eval_summary.json`. `python -m compileall .` is the closest thing to a parse check.
- `Counterfactual_Reasoning/` — has a real `pyproject.toml` (`pip install -e ".[dev]"`), runs `pytest`, `ruff`, `pyright`, `mypy`. But the dev tooling is configured for the upstream `src/agdebugger/` package only; the fork's runner code at the repo root is not type-checked / linted by default.

## Working notes

- Most directories named `tmp_*_smoke/`, `pipeline_outputs*/`, `experiments/`, `logs/`, `paper_cache/`, `cache/`, `*.val_cache/` are run artifacts, not source. They are large and self-contained — each `phase_summary.json` / `summary.json` is the canonical entrypoint for inspecting a run, not the surrounding files.
- Workflow design specs (Chinese) live in each subproject under `workflow_mds/` or `agentworkflow_md/`. They are the de-facto design docs and are usually more current than the READMEs.
