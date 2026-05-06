#!/usr/bin/env bash
# Sequentially run the 3 thinking models on bench A balanced (800 samples).
# These needed the thinking_model_patch (reasoning_content propagation) before
# they could complete a tool-call multi-turn:
#
#   1. glm-5.1
#   2. kimi-k2.5            (also gets ToolUniverse auto-disabled by runner)
#   3. deepseek-v4-flash
#
# Each model runs in its own runner invocation so a crash on one does not kill
# the others. Output goes alongside the existing tool-model runs under
# eval_runs/paired_protein_v2_balanced__<RUN_NAME>/.
#
# Behavior:
#   * Runner's built-in fast-path skips a model whose summary.json reports
#     n_samples == 800 (already DONE).
#   * Runner's resume keeps already-good rows from a partial run; only errored
#     / missing samples are re-attempted.
#   * Use --force-rerun on the runner CLI for a true clean redo.
#
# Optional env overrides:
#   RUN_NAME   default "full_tool_models_20260427" (same dir as other runs)
#   WORKERS    per-model sample concurrency (default 2)
#   TIMEOUT    per-sample timeout seconds (default 600)
#   PY         python interpreter (default = agentdebug env)

# Don't `set -e`: keep going to the next model when one fails.
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
mkdir -p "$EVAL_OUT"

# Match literature_fetch global semaphore to WORKERS so concurrent samples
# don't queue up behind one PDF parse.
export EVAL_LITERATURE_FETCH_CONCURRENCY="$WORKERS"

MODELS=(
  "glm-5.1"
  "kimi-k2.5"
  "deepseek-v4-flash"
)

EXPECTED=$(wc -l < "$DATASET")

echo "================================================================="
echo "Bench A balanced — thinking-model 3-pack (sequential)"
echo "  dataset:    $DATASET ($EXPECTED samples)"
echo "  output dir: $EVAL_OUT"
echo "  workers:    $WORKERS    timeout: ${TIMEOUT}s"
echo "  models (in order):"
for m in "${MODELS[@]}"; do echo "    - $m"; done
echo "================================================================="

cd "$ROOT"

for m in "${MODELS[@]}"; do
  echo
  echo "=================================================================="
  echo ">>> $(date +'%Y-%m-%d %H:%M:%S')  Model: $m"
  echo "=================================================================="

  # No pre-skip / no rm-rf: let runner's built-in resume + fast-path handle it.
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

  scored_file="$EVAL_OUT/$m/scored_results.jsonl"
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
for m in "${MODELS[@]}"; do
  d="$EVAL_OUT/$m"
  scored="$d/scored_results.jsonl"
  if [[ -f "$scored" ]]; then
    n=$(wc -l < "$scored")
    [[ "$n" -eq "$EXPECTED" ]] && tag="DONE" || tag="PARTIAL"
    printf "  %-30s %5d scored  (%s)\n" "$m" "$n" "$tag"
  else
    a=0
    [[ -f "$d/answers.jsonl" ]] && a=$(wc -l < "$d/answers.jsonl")
    printf "  %-30s %5d answers (no scored — failed)\n" "$m" "$a"
  fi
done
echo "================================================================="
