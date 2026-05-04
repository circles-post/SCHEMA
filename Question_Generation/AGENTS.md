# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python package for evidence-first question generation over scientific knowledge-graph triples. Core pipeline modules live at the repository root: `cli.py` orchestrates the run, `generator.py` builds samples, `sampler.py` and `subgraph_builder.py` choose candidate graphs, and `validator.py`, `retrieval_validator.py`, and `model_validator.py` enforce rule-based and model-based checks. Shared schemas are in `models.py`, defaults are in `config.py`, and prompt text lives in `templates.py` plus `validation_prompts.py`. Workflow notes are kept in `workflow_mds/question_generation_workflow.md`.

## Build, Test, and Development Commands
There is no build system in this snapshot; use direct Python commands.

- `python -m compileall .` checks that all modules parse.
- `PYTHONPATH=.. python -m question_generation.cli --triples data/normalized_triples.jsonl --chunks data/chunks.jsonl --output out/question_samples.jsonl` runs the generator from the parent directory.
- Add `--summary-output out/summary.json` to export aggregate stats.
- Add `--validation-mode hybrid_model --validator-enabled` only when model credentials and endpoint settings are available.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and small focused modules consistent with the existing codebase. Prefer `snake_case` for functions, variables, and module names; use `PascalCase` for dataclasses such as `QuestionSample` and `SupportingTriple`. Keep imports explicit, avoid script-style side effects outside `main()`, and preserve the current pattern of plain dataclasses plus helper functions. No formatter or linter is configured here, so match surrounding style closely.

## Testing Guidelines
This checkout does not include a `tests/` directory yet. At minimum, run `python -m compileall .` and a CLI smoke test against small `normalized_triples.jsonl` and `chunks.jsonl` fixtures before submitting changes. New automated tests should go under `tests/` and follow `test_*.py` naming. Prioritize coverage for sampling, validation rejection paths, and export formatting.

## Commit & Pull Request Guidelines
Git history is not present in this exported snapshot, so no local commit convention can be derived. Use short imperative subjects such as `validator: tighten evidence alignment checks`. PRs should describe the pipeline stage changed, note any config or dependency assumptions, and include a small sample command plus expected output files when behavior changes.

## Security & Configuration Tips
Do not hard-code API keys or model endpoints. Pass validator settings through CLI flags or local environment management, and keep generated JSONL outputs outside the package source tree.
