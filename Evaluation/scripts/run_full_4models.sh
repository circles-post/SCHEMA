#!/usr/bin/env bash
# Run all 4 target models on the full bench A and bench B.
#
# Important: 2 of the 4 models can NOT use the agent retrieval tools, so this
# script invokes the runner TWICE per bench (different --use-tools flags).
# All 4 model subdirs end up under the same eval_runs/<bench>__<run-name>/ tree.
#
# Models split:
#   tool-mode (full agent + 4 retrieval tools + ToolUniverse MCP):
#     - gpt-4o
#     - gpt-5.4-mini
#   closed-book-only (model API can't accept our function schemas):
#     - gemini-3-flash-preview-thinking   # Gemini rejects schemas missing `type` fields
#     - deepseek-v4-flash                 # thinking-mode requires reasoning_content echoed
#
# Comparing tool-mode vs closed-book scores is apples-to-oranges; the
# combined_summary.json keeps each model's mode in the record so you can see it.
#
# Usage:
#   bash evaluation/scripts/run_full_4models.sh                # both benches
#   bash evaluation/scripts/run_full_4models.sh A              # bench A only
#   bash evaluation/scripts/run_full_4models.sh B              # bench B only
#   bash evaluation/scripts/run_full_4models.sh both <run-name>
#
# Optional env overrides:
#   WORKERS=2                # per-model sample concurrency (default 2; safe under rate limits)
#   SUBSET=balanced          # use samples_balanced.jsonl instead of full bench
#   LIMIT=N                  # truncate to N samples (smoke tests only)
#   TOOL_MODELS=...          # override which models go through tool-mode pass
#   CLOSED_MODELS=...        # override which models go through closed-book pass

set -euo pipefail

ROOT="<repo-root>"

: "${WORKERS:=2}"
: "${RUN_NAME:=full_4models_$(date +%Y%m%d)}"
: "${TOOL_MODELS:=gpt-4o,gpt-5.4-mini}"
: "${CLOSED_MODELS:=gemini-3-flash-preview-thinking,deepseek-v4-flash}"

WHICH="${1:-both}"
[[ -n "${2:-}" ]] && RUN_NAME="$2"

echo "================================================================="
echo "Full multi-model run"
echo "  bench:       $WHICH        run_name: $RUN_NAME"
echo "  workers:     $WORKERS  (per model)"
echo "  tool-mode:   $TOOL_MODELS"
echo "  closed-book: $CLOSED_MODELS"
[[ -n "${SUBSET:-}" ]] && echo "  subset:      $SUBSET"
[[ -n "${LIMIT:-}" ]] && echo "  limit:       $LIMIT"
echo "================================================================="

# Phase 1: tool-mode models (gpt-4o, gpt-5.4-mini)
if [[ -n "$TOOL_MODELS" ]]; then
  echo
  echo ">>> Phase 1: tool-mode models ($TOOL_MODELS)"
  MODELS="$TOOL_MODELS" USE_TOOLS=1 \
    bash "$ROOT/evaluation/scripts/run_eval.sh" "$WHICH" "$RUN_NAME"
fi

# Phase 2: closed-book-only models (gemini, deepseek)
if [[ -n "$CLOSED_MODELS" ]]; then
  echo
  echo ">>> Phase 2: closed-book models ($CLOSED_MODELS)"
  # Closed-book is faster; bump per-sample timeout down to 180s
  MODELS="$CLOSED_MODELS" USE_TOOLS=0 PER_SAMPLE_TIMEOUT=180 \
    bash "$ROOT/evaluation/scripts/run_eval.sh" "$WHICH" "$RUN_NAME"
fi

echo
echo "All phases complete. Per-bench output dir:"
echo "  eval_runs/<bench>__${RUN_NAME}/{gpt-4o,gpt-5.4-mini,gemini-3-flash-preview-thinking,deepseek-v4-flash}/"
