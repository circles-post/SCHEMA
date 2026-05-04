# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a fork of the upstream **AGDebugger** (CHI 2025) — an interactive UI for debugging AutoGen agent teams via send/edit/revert/insert operations on a message timeline. On top of the upstream UI/backend (`src/agdebugger/`), the fork layers a much larger **automated dataset evaluation + LLM-driven trajectory repair** system used for life-science benchmarks. Most active development is in that layer, not in the upstream UI.

The system has two stages per question:
1. A primary AutoGen agent team answers the question (default team in `test_agent_debug.py:get_agent_team`).
2. If the answer is wrong, an **external agent** extracts claims from the trajectory, judges them against web/literature evidence, picks a faulty turn, and uses the AGDebugger HTTP API to `edit_and_revert` (or insert) and rerun from that point.

Authoritative architecture write-ups live in `agentworkflow_md/` (Chinese). The most current ones are `latest_workflow.md`, `agdebugger_workflow_latest_20260317.md`, `external_agent_repair_timeline.md`, and `prompt_reference.md` — read these before making non-trivial changes to the runner or external agent.

## Big-picture architecture

```
run_parallel.sh ──► run_with_models.sh ──► (1) ToolUniverse MCP server (shared, port 7000)
                                           (2) agdebugger backend (uvicorn, port 8081 + worker idx)
                                           (3) run_dataset_autodebug.py
                                                  │
                                                  ├─ feeds dataset questions to backend via HTTP
                                                  ├─ waits for the AutoGen team (test_agent_debug.get_agent_team) to go idle
                                                  ├─ on wrong answer ─► external_agent_controller.py (LLMPlanner)
                                                  │                       │
                                                  │                       └─ external_agent.integration.analyze_session_state
                                                  │                              ├─ ClaimExtractor (external_agent/claim_extractor.py)
                                                  │                              ├─ WebSearchEvidenceProvider (websearch/* + serper/bright-data)
                                                  │                              ├─ ClaimJudge / PlannerJudge (external_agent/judge.py)
                                                  │                              └─ ClaimJudgePipeline (external_agent/pipeline.py)
                                                  └─ executes planner action (edit_and_revert / insert_after / send / step)
                                                     via AGDebuggerClient HTTP calls back to the backend
```

Key seams:

- **`src/agdebugger/`** — upstream FastAPI backend + frontend bundle. `cli.py` is the entrypoint (`agdebugger` script / `python -m agdebugger.cli`). `backend.py` exposes the `/api/*` HTTP routes that everything else (frontend, controller, dataset runner) talks to. Don't change route shapes without updating both `external_agent_controller.AGDebuggerClient` and `frontend/`.
- **`test_agent_debug.py`** — defines `get_agent_team()`, the AutoGen team loaded by the backend. Wires up MCP tools (ToolUniverse via `StreamableHttpServerParams`), `websearch_tools`, `sciverse_tools.literature_search`, and the OpenAI-compatible model clients. This is the *agent under debug*, not the controller.
- **`model_routing.py`** — single source of truth for "which base_url / api_key / extra_body for which model name". Both `test_agent_debug.py` and `external_agent_controller.py` route through it. Intern (`intern-s1*`) vs non-intern endpoints, plus reasoning/`thinking_mode` toggle, are decided here.
- **`external_agent/`** — the claim/judge framework. Has its own `cli.py` and `README.md` and is *also* runnable standalone on plain text. `integration.analyze_session_state` is the entrypoint used in-process by the dataset runner. `strategies.py` holds per-domain prompts (`research_questions`, `medical_guidelines`, `legal_cases`, `coding`).
- **`external_agent_controller.py`** — turns AGDebugger session state + analysis result into a planner JSON action and pushes it back through `AGDebuggerClient`. Contains both a `rule` planner (deterministic bootstrap/step) and an `llm` planner.
- **`run_dataset_autodebug.py`** — orchestrator. Loads dataset components from the sibling `datasets/` repo (`/mnt/shared-storage-user/fengxinshun/AISci/datasets`, added to `sys.path` at runtime — this path is hardcoded), iterates questions, drives the backend, invokes analysis, applies repairs, scores answers, writes JSONL run logs.
- **`websearch/`** + `websearch_tools.py` — evidence providers (Serper, Bright Data, PubMed, Crossref, bioRxiv, PDF extraction via MinerU). Used both by the agent under debug (as MCP/function tools) and by the external agent's `WebSearchEvidenceProvider`.

External dependencies the runner expects to find on disk:
- `/mnt/shared-storage-user/fengxinshun/AISci/ToolUniverse/` — launched as the shared MCP server.
- `/mnt/shared-storage-user/fengxinshun/AISci/datasets/` — `browse_bio_graph_cluster_examples` is imported from here.
- `/mnt/shared-storage-user/fengxinshun/AISci/sciverse/` — `sciverse_tools.literature_search` is imported by `test_agent_debug.py`.

If any of those are missing the team will fail to construct. They are not pip-installable.

## Common commands

### Upstream UI (interactive, original AGDebugger)

```bash
# First-time install (frontend + python package)
cd frontend && npm install && npm run build && cd ..
pip install -e .

# Launch UI on a custom AutoGen team factory `module:func`
agdebugger test_agent_debug:get_agent_team --host 127.0.0.1 --port 8081 --launch
# Backend-only (frontend dev server hits this):
AGDEBUGGER_BACKEND_SERVE_UI=FALSE agdebugger test_agent_debug:get_agent_team --port 8123
# Frontend dev server
cd frontend && npm run dev   # needs .env.development.local with VITE_AGDEBUGGER_FRONTEND_API_URL
```

### Dataset auto-debug runs (the main workflow in this fork)

```bash
# Single slice — sets up models, ToolUniverse, backend, then runs the dataset.
bash run_with_models.sh --component-id 1 --start 0 --limit 5

# Parallel — N workers, non-overlapping slices, shared ToolUniverse MCP.
bash run_parallel.sh --workers 5 --total-examples 116 -- --component-id 1
```

Both scripts source conda env `agentdebug` from `/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3` by default (override via `AGDEBUGGER_CONDA_ENV`, `CONDA_BASE`). They write per-run artifacts under `logs/<YYYYMMDD>/run_<stamp>/` (`server.log`, `run.jsonl`, `analysis_detail.jsonl`).

Model routing is configured by editing the five env vars at the top of `run_with_models.sh` (`AGENTDEBUG_MODEL_NAME`, `AGENTDEBUG_MODEL_AGENT`, `AGENTDEBUG_MODEL_MCP`, `MODEL_PLANNER`, `MODEL_CLAIM`) — `model_routing.py` then resolves base_url + api_key based on whether the name is an `intern-s1*` model.

### External agent standalone (claim extract + judge on plain text or saved session state)

```bash
python -m external_agent.cli --task research_questions \
  --text-file response.txt --evidence-file evidence.txt --model gpt-4o-mini

python -m external_agent.cli --task research_questions \
  --history-state-file session_state.json --assistant-only \
  --evidence-file evidence.txt --model gpt-4o-mini
```

### Tests, lint, types

```bash
pip install -e ".[dev]"
pytest                                  # all tests under tests/
pytest tests/test_external_agent_integration.py::TestName::test_case   # single test
ruff check . && ruff format .           # ruff config in pyproject.toml; line-length 120
pyright                                 # basic mode, configured in pyproject.toml
mypy                                    # strict; targets src/, examples/, tests/
```

Note: `pyproject.toml` only declares the upstream `agdebugger` package under `src/`. The fork's runner code (`run_dataset_autodebug.py`, `external_agent_controller.py`, `external_agent/`, `websearch/`, `model_routing.py`, `test_agent_debug.py`, `websearch_tools.py`) lives at the repo root and is *not* part of the installed package. `run_with_models.sh` sets `PYTHONPATH=${SCRIPT_DIR}:${SRC_DIR}` so they import; do the same if running them outside the launcher. Their tests live in `tests/test_external_agent_*.py`, `tests/test_run_dataset_autodebug.py`, `tests/test_pdf_extractor.py`.

## Things to be careful about

- **Hardcoded absolute paths and credentials.** `run_with_models.sh`, `run_parallel.sh`, `test_agent_debug.py`, `run_dataset_autodebug.py`, and `external_agent_controller.py` contain hardcoded `/mnt/shared-storage-user/fengxinshun/...` paths plus inline default API keys (Bright Data, MinerU, Intern, the non-intern proxy endpoint, and an http proxy URL with credentials). These are intentional defaults so that `bash run_with_models.sh` "just works" on the author's box. Don't accidentally exfiltrate them in commits to public branches, and prefer overriding via env vars rather than editing in place.
- **Proxy hack in `test_agent_debug.py` and `external_agent_controller.py`.** They mutate `http_proxy` / `no_proxy` at import time because httpx doesn't accept CIDR in `no_proxy` and the LLM endpoint must *not* be bypassed. Don't "clean up" this block without re-testing connectivity to both the LLM endpoint and the MCP server.
- **`autogen-ext` monkey-patch.** `test_agent_debug.py` patches `autogen_ext.models.openai._openai_client._add_usage` to tolerate `None` token counts from non-OpenAI providers. Re-apply if you upgrade `autogen-ext`.
- **ToolUniverse coordination.** Multiple workers share one MCP server via a lock dir + pid file under `logs/`. If you see `tooluniverse_shared.lock` left behind, the launchers will auto-clean it after 30s; don't `rm -rf logs/` mid-run.
- **Two stages, two purposes for the controller.** `external_agent_controller.py` is invoked by `run_dataset_autodebug.py` *as a library* (via `LLMPlanner` + `execute_action`) but it also has a CLI `main()` for manual driving against a running backend. Changing function signatures breaks the in-process path even if the CLI still works.
- **Language for design docs.** `agentworkflow_md/*.md` are maintained in Chinese and are the de-facto design spec. When changing the analysis/repair pipeline, update the relevant doc (usually `latest_workflow.md` or a dated `*_plan_*.md`) so the next session has accurate context.
