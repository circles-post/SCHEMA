#!/usr/bin/env bash
# Full bench A + B run for the 7 models confirmed to work in tool mode.
# This script runs ALL 7 models with the 4 retrieval tools only (web_search,
# web_fetch, literature_search, literature_fetch); ToolUniverse MCP is OFF for
# every model so comparisons are apples-to-apples (gemini/kimi/grok wouldn't
# work with TU anyway, and TU adds non-trivial latency + contention).
#
# Models in tool mode (4 retrieval tools, no ToolUniverse):
#   - gpt-4o
#   - gpt-5.4-mini
#   - gemini-3-flash-preview-thinking
#   - qwen3.6-plus
#   - doubao-seed-2-0-pro-260215
#   - llama-4-scout
#   - grok-4-1-fast-reasoning
#
# Models pending tool-mode support (skipped here, can be added once patched):
#   - glm-5.1                             # thinking model: reasoning_content propagation TODO
#   - deepseek-v4-flash                   # thinking model: same as glm-5.1
#   - kimi-k2.5                           # thinking model + strict-schema (TU auto-disabled)
#   - baidu/ERNIE-4.5-300B-A47B           # platform-disabled by Boyue
#
# Trajectory dump for halu phase 2: ALWAYS ON. The runner writes
# eval_runs/<bench>__<run-name>/<model>/trajectory.jsonl (see runner.py
# --no-emit-trajectory if you ever want to skip it; default is on).
#
# Usage:
#   bash evaluation/scripts/run_full_tool_models.sh                # both benches, full
#   bash evaluation/scripts/run_full_tool_models.sh A              # bench A only
#   bash evaluation/scripts/run_full_tool_models.sh B              # bench B only
#   bash evaluation/scripts/run_full_tool_models.sh both my_run    # custom RUN_NAME
#
# Optional env overrides:
#   WORKERS=2                # per-model sample concurrency. Default 2 is safe under
#                            # external API rate limits; >=4 risks Bright Data /
#                            # sciverse / MinerU contention (see run_eval.sh notes).
#   SUBSET=balanced          # use samples_balanced.jsonl instead of full bench
#   LIMIT=N                  # truncate to N samples (smoke tests only)
#   PER_SAMPLE_TIMEOUT=600   # per-sample wall clock cap (seconds)
#   MODELS=...               # override the model list

set -euo pipefail

ROOT="<repo-root>"

: "${WORKERS:=2}"
# All 7 tool-mode models run with the 4 retrieval tools but WITHOUT the
# ToolUniverse MCP workbench. Reasons: gemini/kimi/grok already need TU off;
# disabling for the others (gpt-*, qwen, doubao, llama) keeps results
# apples-to-apples and removes a contention source under WORKERS>=2.
# Override with USE_TOOLUNIVERSE=1 if you want TU back.
: "${USE_TOOLUNIVERSE:=0}"
export USE_TOOLUNIVERSE
# Full 7-model list (uncomment when ready to run all):
# : "${MODELS:=gpt-4o,gpt-5.4-mini,gemini-3-flash-preview-thinking,qwen3.6-plus,doubao-seed-2-0-pro-260215,llama-4-scout,grok-4-1-fast-reasoning}"
: "${MODELS:=gemini-3-flash-preview-thinking,qwen3.6-plus}"
: "${RUN_NAME:=full_tool_models_$(date +%Y%m%d)}"
: "${PER_SAMPLE_TIMEOUT:=600}"

WHICH="${1:-both}"
[[ -n "${2:-}" ]] && RUN_NAME="$2"

echo "================================================================="
echo "Full tool-mode multi-model run"
echo "  bench:        $WHICH"
echo "  run_name:     $RUN_NAME"
echo "  models:       $MODELS"
echo "  workers:      $WORKERS  (per model)"
echo "  timeout:      ${PER_SAMPLE_TIMEOUT}s per sample"
[[ -n "${SUBSET:-}" ]] && echo "  subset:       $SUBSET"
[[ -n "${LIMIT:-}" ]] && echo "  limit:        $LIMIT"
echo "  trajectory:   ENABLED (runner default; needed for halu phase 2)"
echo "================================================================="

export MODELS WORKERS RUN_NAME PER_SAMPLE_TIMEOUT
USE_TOOLS=1 \
  bash "$ROOT/evaluation/scripts/run_eval.sh" "$WHICH" "$RUN_NAME"

echo
echo "================================================================="
echo "Done. Outputs:"
echo "  Per-model answers + scoring + trajectory:"
echo "    evaluation/eval_runs/<bench>__${RUN_NAME}/<model>/{answers.jsonl, scored_results.jsonl, summary.json, trajectory.jsonl}"
echo
echo "Next step (hallucination analysis on the trajectories):"
echo "  bash evaluation/scripts/run_bench.sh $WHICH ${RUN_NAME}_halu  # or call halu.cli directly"
echo "================================================================="
