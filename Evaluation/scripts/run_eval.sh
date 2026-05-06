#!/usr/bin/env bash
# Phase-1 only: run evaluation.runner on bench A or B and dump trajectories.
# No halu (hallucination detection) — use run_bench.sh for the full pipeline.
#
# Usage:
#   bash evaluation/scripts/run_eval.sh <which> [<run-name-suffix>]
#
# <which> ∈ { A | B | both }
#   A    = paired_protein_v2   (proteinlmbench_full_graphbench)
#   B    = paired_enhanced_v2  (protein_plus_pathvqa_500_v3)
#   both = run A then B
#
# Edit MODELS= below, or override via env vars.
#
# Optional env overrides:
#   MODELS           comma-list of model ids (default = gpt-4o)
#   LIMIT            sample cap (0 = full bench)
#   USE_TOOLS        1=full agent w/ web+literature+tooluniverse, 0=closed-book
#   USE_TOOLUNIVERSE 1=attach ToolUniverse MCP (default), 0=skip (4 retrieval tools only).
#                    runner.py also auto-disables TU for known incompatible models
#                    (gemini, kimi, moonshot, grok); this flag forces it OFF universally.
#   PER_SAMPLE_TIMEOUT  per-sample wall-clock cap (seconds, default 600)
#   WORKERS          per-model sample concurrency (default 1 = strict serial)
#                    >1 builds N independent agent teams; samples are dispatched
#                    in parallel. Reasonable values: 4-8 for full agent mode.
#   SUBSET           "" (default) = samples.jsonl;  "balanced" = samples_balanced.jsonl;
#                    any other token <X> = samples_<X>.jsonl (must exist next to samples.jsonl).
#   RUN_NAME         suffix appended to output dir (default = today's date)
#   PY               python interpreter (default = agentdebug env)

set -euo pipefail

# --------------------------------------------------------------------------- #
# CHANGE ME — comma-separated list of model ids to evaluate.                  #
# --------------------------------------------------------------------------- #
: "${MODELS:=gpt-4o}"

: "${LIMIT:=0}"
: "${USE_TOOLS:=1}"
: "${USE_TOOLUNIVERSE:=1}"
: "${PER_SAMPLE_TIMEOUT:=600}"
: "${WORKERS:=1}"
# literature_fetch is process-global Semaphore-bounded so MinerU/PDF parses
# don't herd. Default to WORKERS so concurrency scales together; override via
# LIT_FETCH_WORKERS=2 if you're memory-constrained.
: "${LIT_FETCH_WORKERS:=$WORKERS}"
export EVAL_LITERATURE_FETCH_CONCURRENCY="$LIT_FETCH_WORKERS"
: "${SUBSET:=}"
: "${RUN_NAME:=$(date +%Y%m%d)}"
: "${PY:=python}"

ROOT="<repo-root>"

# SUBSET="" → samples.jsonl;  SUBSET="balanced" → samples_balanced.jsonl;
# any other <X>           → samples_<X>.jsonl (must exist next to samples.jsonl).
SAMPLES_FILE="samples.jsonl"
TAG_SUFFIX=""
if [[ -n "$SUBSET" ]]; then
  SAMPLES_FILE="samples_${SUBSET}.jsonl"
  TAG_SUFFIX="_${SUBSET}"
fi

BENCH_A_DATASET="$ROOT/benchmark_runs/proteinlmbench_full_graphbench/qgen_paired_protein_v2/$SAMPLES_FILE"
BENCH_A_TAG="paired_protein_v2${TAG_SUFFIX}"

BENCH_B_DATASET="$ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2/$SAMPLES_FILE"
BENCH_B_TAG="paired_enhanced_v2${TAG_SUFFIX}"

WHICH="${1:-}"
if [[ -n "${2:-}" ]]; then RUN_NAME="$2"; fi

case "$WHICH" in
  A|a)     BENCHES=("A") ;;
  B|b)     BENCHES=("B") ;;
  both|"") BENCHES=("A" "B") ;;
  *)
    echo "ERROR: first arg must be 'A', 'B', or 'both' (got: $WHICH)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$ROOT/evaluation/.env" ]]; then
  echo "ERROR: $ROOT/evaluation/.env not found." >&2
  exit 2
fi
set +u
# shellcheck disable=SC1091
source <(grep -E '^BOYUE_API_KEY=' "$ROOT/evaluation/.env" | sed 's/^/export /')
set -u
if [[ -z "${BOYUE_API_KEY:-}" ]]; then
  echo "ERROR: BOYUE_API_KEY missing in evaluation/.env." >&2
  exit 2
fi

_USE_TOOLS_FLAG="--use-tools"
[[ "$USE_TOOLS" = "0" ]] && _USE_TOOLS_FLAG="--no-tools"
_USE_TU_FLAG=""
[[ "$USE_TOOLUNIVERSE" = "0" ]] && _USE_TU_FLAG="--no-tooluniverse"
_LIMIT_FLAG=""
[[ "$LIMIT" -gt 0 ]] && _LIMIT_FLAG="--limit $LIMIT"

run_one_bench() {
  local tag="$1" dataset="$2"
  local eval_out="$ROOT/evaluation/eval_runs/${tag}__${RUN_NAME}"

  if [[ ! -f "$dataset" ]]; then
    echo "ERROR: dataset not found: $dataset" >&2
    if [[ -n "$SUBSET" ]]; then
      echo "       (SUBSET=$SUBSET expects samples_${SUBSET}.jsonl next to samples.jsonl;" >&2
      echo "        run evaluation/scripts/build_balanced_subset.py first if missing.)" >&2
    fi
    exit 2
  fi

  echo
  echo "=================================================================="
  echo "BENCH: $tag    RUN_NAME: $RUN_NAME"
  echo "  dataset:    $dataset  ($(wc -l < "$dataset") samples)"
  echo "  models:     $MODELS"
  echo "  limit:      ${LIMIT:-0}    use_tools: $USE_TOOLS    use_tooluniverse: $USE_TOOLUNIVERSE    timeout: ${PER_SAMPLE_TIMEOUT}s"
  echo "  workers:    $WORKERS    lit_fetch_workers: $LIT_FETCH_WORKERS"
  echo "  eval_out:   $eval_out"
  echo "=================================================================="

  cd "$ROOT"
  # shellcheck disable=SC2086
  "$PY" -m evaluation.runner \
    --dataset "$dataset" \
    --models "$MODELS" \
    --output-dir "$eval_out" \
    --per-sample-timeout "$PER_SAMPLE_TIMEOUT" \
    --sample-concurrency "$WORKERS" \
    $_USE_TOOLS_FLAG \
    $_USE_TU_FLAG \
    $_LIMIT_FLAG

  echo
  echo "DONE: $tag"
  for d in "$eval_out"/*/; do
    if [[ -f "$d/trajectory.jsonl" ]]; then
      n=$(wc -l < "$d/trajectory.jsonl")
      m=$(basename "$d")
      echo "  - $m: $n trajectory lines  ($d)"
    fi
  done
}

for b in "${BENCHES[@]}"; do
  if [[ "$b" = "A" ]]; then
    run_one_bench "$BENCH_A_TAG" "$BENCH_A_DATASET"
  else
    run_one_bench "$BENCH_B_TAG" "$BENCH_B_DATASET"
  fi
done

echo
echo "All requested benches complete."
