#!/usr/bin/env bash
# Run the 5 remaining tool-mode models on bench A balanced (800 samples),
# one model per runner invocation so a single model's failure does not kill
# the rest. Output goes alongside the already-DONE gpt-4o / gpt-5.4-mini
# under paired_protein_v2_balanced__<RUN_NAME>/.
#
# Models (in execution order):
#   1. gemini-3-flash-preview-thinking
#   2. qwen3.6-plus
#   3. doubao-seed-2-0-pro-260215
#   4. llama-4-scout
#   5. grok-4-1-fast-reasoning
#
# Behavior:
#   * Cleans any partial state under <model>/ before launching that model.
#   * Skips a model if its scored_results.jsonl already has 800 lines (DONE).
#   * Prints a final scoreboard.
#
# Optional env:
#   RUN_NAME           default "full_tool_models_20260427" — keeps writes in
#                      the same dir as the existing gpt-4o / gpt-5.4-mini runs.
#   WORKERS            per-model sample concurrency (default 2)
#   TIMEOUT            per-sample timeout in seconds (default 600)
#   PY                 python interpreter (default = agentdebug env)

# Note: deliberately NOT using `set -e` — we want to continue to the next
# model when one fails. We do use set -u to catch typos.
set -uo pipefail

ROOT="<repo-root>"
: "${PY:=python}"
: "${RUN_NAME:=full_tool_models_20260427}"
: "${WORKERS:=2}"
: "${TIMEOUT:=600}"

DATASET="$ROOT/benchmark_runs/proteinlmbench_full_graphbench/qgen_paired_protein_v2/samples_balanced.jsonl"
EVAL_OUT="$ROOT/evaluation/eval_runs/paired_protein_v2_balanced__${RUN_NAME}"

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET" >&2
  exit 2
fi
if [[ ! -d "$EVAL_OUT" ]]; then
  echo "WARN: $EVAL_OUT does not exist yet — creating fresh." >&2
  mkdir -p "$EVAL_OUT"
fi

# Match literature_fetch global semaphore to WORKERS so 2 concurrent samples
# don't queue up behind one PDF parse.
export EVAL_LITERATURE_FETCH_CONCURRENCY="$WORKERS"

MODELS=(
  "gemini-3-flash-preview-thinking"
  "qwen3.6-plus"
  "doubao-seed-2-0-pro-260215"
  "llama-4-scout"
  "grok-4-1-fast-reasoning"
)

EXPECTED=$(wc -l < "$DATASET")

echo "================================================================="
echo "Bench A balanced — sequential 5-model run"
echo "  dataset:    $DATASET ($EXPECTED samples)"
echo "  output dir: $EVAL_OUT"
echo "  workers:    $WORKERS    timeout: ${TIMEOUT}s"
echo "  models (in order):"
for m in "${MODELS[@]}"; do echo "    - $m"; done
echo "================================================================="

cd "$ROOT"

for m in "${MODELS[@]}"; do
  scored_file="$EVAL_OUT/$m/scored_results.jsonl"
  echo
  echo "=================================================================="
  echo ">>> $(date +'%Y-%m-%d %H:%M:%S')  Model: $m"
  echo "=================================================================="

  # No pre-skip / no rm-rf: let runner's built-in resume + fast-path handle it.
  # Runner will:
  #   * skip if summary.json reports n_samples == EXPECTED (fast-path),
  #   * else read answers.jsonl, keep good rows, drop errored rows, append new.
  # Use --force-rerun on the runner CLI if you ever need a true clean redo.

  start=$(date +%s)
  "$PY" -m evaluation.runner \
    --dataset "$DATASET" \
    --models "$m" \
    --output-dir "$EVAL_OUT" \
    --use-tools \
    --no-tooluniverse \
    --per-sample-timeout "$TIMEOUT" \
    --sample-concurrency "$WORKERS"
  status=$?
  dur=$(( $(date +%s) - start ))

  if [[ $status -eq 0 ]]; then
    n_scored=0
    [[ -f "$scored_file" ]] && n_scored=$(wc -l < "$scored_file")
    echo ">>> $m FINISHED in ${dur}s — $n_scored scored"
  else
    echo "!!! $m exited with status $status after ${dur}s — moving on to next model"
  fi
done

echo
echo "================================================================="
echo "Final scoreboard ($EVAL_OUT):"
for d in "$EVAL_OUT"/*/; do
  m=$(basename "$d")
  scored="$d/scored_results.jsonl"
  if [[ -f "$scored" ]]; then
    n=$(wc -l < "$scored")
    [[ "$n" -eq "$EXPECTED" ]] && tag="DONE" || tag="PARTIAL"
    printf "  %-46s %5d scored  (%s)\n" "$m" "$n" "$tag"
  else
    a=0
    [[ -f "$d/answers.jsonl" ]] && a=$(wc -l < "$d/answers.jsonl")
    printf "  %-46s %5d answers (no scored — failed)\n" "$m" "$a"
  fi
done
echo "================================================================="
