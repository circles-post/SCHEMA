#!/usr/bin/env bash
# Run all currently-tested models on bench B balanced (531 samples).
# Each model runs in its own runner invocation so a crash on one doesn't kill
# the rest. Output goes to eval_runs/paired_enhanced_v2_balanced__<RUN_NAME>/.
#
# Models (in execution order):
#   tool-mode (4 retrieval tools, no ToolUniverse):
#     1. gpt-4o
#     2. gpt-5.4-mini
#     3. gemini-3-flash-preview-thinking      (auto-no-TU by runner)
#     4. qwen3.6-plus
#     5. doubao-seed-2-0-pro-260215           (auto-routes to HTTPS gateway)
#     6. llama-4-scout                        (auto-routes to HTTPS gateway)
#     7. grok-4-1-fast-reasoning              (auto-no-TU + auto-routes)
#     8. glm-5.1                              (thinking_model_patch)
#     9. kimi-k2.5                            (thinking_model_patch + auto-no-TU)
#    10. deepseek-v4-flash                    (thinking_model_patch)
#    11. intern-s1-pro                        (auto-routes to chat.intern-ai.org.cn + INTERN_API_KEY)
#
# Behavior:
#   * Runner's built-in fast-path skips any model whose summary.json reports
#     n_samples == 531 (already DONE).
#   * Runner's resume keeps already-good rows from a partial run; only errored
#     / missing samples are re-attempted.
#   * Use --force-rerun on the runner CLI for a true clean redo.
#
# Optional env overrides:
#   RUN_NAME      default "full_models_$(date +%Y%m%d)"
#   WORKERS       per-model sample concurrency (default 2)
#   TIMEOUT       per-sample timeout seconds (default 600)
#   MODELS        override the model list (comma-separated)
#   PY            python interpreter (default = agentdebug env)

# Don't `set -e`: keep going to the next model when one fails.
set -uo pipefail

ROOT="/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa"
: "${PY:=/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/agentdebug/bin/python}"
: "${RUN_NAME:=full_models_$(date +%Y%m%d)}"
: "${WORKERS:=2}"
: "${TIMEOUT:=600}"

DATASET_FULL="$ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2/samples_balanced.jsonl"
DATASET_NO_VQA="$ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2/samples_balanced_no_vqa.jsonl"
EVAL_OUT="$ROOT/evaluation/eval_runs/paired_enhanced_v2_balanced__${RUN_NAME}"

if [[ ! -f "$DATASET_FULL" ]]; then
  echo "ERROR: dataset not found: $DATASET_FULL" >&2
  echo "  Generate it first via evaluation/scripts/build_balanced_subset.py" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Auto-generate the no-VQA filtered subset.
#
# The text-only agent pipeline cannot see images, so VQA samples would be
# answered by guessing yes/no — meaningless. We filter them out here for the
# main run and keep the original samples_balanced.jsonl untouched so a
# multimodal-equipped follow-up run can target VQA only.
# ---------------------------------------------------------------------------
if [[ ! -f "$DATASET_NO_VQA" ]] || [[ "$DATASET_FULL" -nt "$DATASET_NO_VQA" ]]; then
  echo ">>> generating no-VQA subset → $DATASET_NO_VQA"
  "$PY" -c "
import json
src = '$DATASET_FULL'
dst = '$DATASET_NO_VQA'
kept = vqa = 0
with open(src) as fh, open(dst,'w') as out:
    for line in fh:
        r = json.loads(line)
        if r.get('question_type') == 'vqa':
            vqa += 1
            continue
        out.write(line); kept += 1
print(f'  kept={kept}  vqa_excluded={vqa}')
"
fi

DATASET="$DATASET_NO_VQA"
mkdir -p "$EVAL_OUT"

# Match literature_fetch global semaphore to WORKERS so concurrent samples
# don't queue up behind one PDF parse.
export EVAL_LITERATURE_FETCH_CONCURRENCY="$WORKERS"

DEFAULT_MODELS=(
  "gpt-4o"
  "gpt-5.4-mini"
  "gemini-3-flash-preview-thinking"

)

# If MODELS env override given, parse it; else use defaults
if [[ -n "${MODELS:-}" ]]; then
  IFS=',' read -ra MODELS_ARR <<< "$MODELS"
else
  MODELS_ARR=("${DEFAULT_MODELS[@]}")
fi

EXPECTED=$(wc -l < "$DATASET")

echo "================================================================="
echo "Bench B balanced — sequential multi-model run"
echo "  dataset:    $DATASET ($EXPECTED samples)"
echo "  output dir: $EVAL_OUT"
echo "  workers:    $WORKERS    timeout: ${TIMEOUT}s"
echo "  models (${#MODELS_ARR[@]}, in order):"
for m in "${MODELS_ARR[@]}"; do echo "    - $m"; done
echo "================================================================="

cd "$ROOT"

for m in "${MODELS_ARR[@]}"; do
  echo
  echo "=================================================================="
  echo ">>> $(date +'%Y-%m-%d %H:%M:%S')  Model: $m"
  echo "=================================================================="

  # No pre-skip / no rm-rf: let runner's fast-path + resume handle it.
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
    echo "!!! $m exited with status $status after ${dur}s — continuing to next model"
  fi
done

# ---------------------------------------------------------------------------
# Final scoreboard
# ---------------------------------------------------------------------------
echo
echo "================================================================="
echo "Final scoreboard ($EVAL_OUT):"
for m in "${MODELS_ARR[@]}"; do
  d="$EVAL_OUT/$m"
  scored="$d/scored_results.jsonl"
  if [[ -f "$scored" ]]; then
    n=$(wc -l < "$scored")
    [[ "$n" -eq "$EXPECTED" ]] && tag="DONE" || tag="PARTIAL"
    if [[ -f "$d/summary.json" ]]; then
      acc=$("$PY" -c "import json; print(f'{json.load(open(\"$d/summary.json\"))[\"aggregate\"][\"overall\"][\"acc\"]:.3f}')" 2>/dev/null || echo "?")
    else
      acc="?"
    fi
    printf "  %-40s %5d scored  acc=%-6s (%s)\n" "$m" "$n" "$acc" "$tag"
  else
    a=0
    [[ -f "$d/answers.jsonl" ]] && a=$(wc -l < "$d/answers.jsonl")
    printf "  %-40s %5d answers (no scored — failed/incomplete)\n" "$m" "$a"
  fi
done
echo "================================================================="
echo
echo "Next step (hallucination analysis on these trajectories):"
echo "  bash evaluation/scripts/run_halu_pending.sh B ${RUN_NAME}"
