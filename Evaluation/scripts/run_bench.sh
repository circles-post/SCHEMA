#!/usr/bin/env bash
# Run evaluation.runner + evaluation.halu.cli on the v2 benches.
#
# Usage:
#   bash evaluation/scripts/run_bench.sh <which> [<run-name-suffix>]
#
# <which> ∈ { A | B | both }
#   A    = paired_protein_v2   (proteinlmbench_full_graphbench)
#   B    = paired_enhanced_v2  (protein_plus_pathvqa_500_v3)
#   both = run A then B
#
# REQUIRED: edit the MODELS= line below to the models you want to test.
# Models must exist on the Boyue endpoint; see `curl $BOYUE_BASE_URL/models`.
#
# Optional env overrides (all have defaults):
#   MODELS           comma-list of model ids to test
#   LIMIT            sample cap (0 = full bench)
#   USE_TOOLS        1=full agent w/ web+literature+tooluniverse, 0=closed-book
#   EVIDENCE_CHAIN   layers tried after supporting_chunk, e.g. "graph,web,literature"
#   INCLUDE_CORRECT  1=include correct trajectories in halu, 0=errors-only
#   WORKERS          per-model sample concurrency in agent eval (default 1 = strict serial)
#                    >1 builds N independent agent teams; samples dispatched in parallel.
#   HALU_EXTRACTOR_CONCURRENCY   halu claim extraction concurrency (default 4)
#   HALU_JUDGE_CONCURRENCY       halu bucket judging concurrency (default 10)
#   SUBSET           "" (default) = samples.jsonl;  "balanced" = samples_balanced.jsonl;
#                    any other token <X> = samples_<X>.jsonl (must exist next to samples.jsonl).
#   RUN_NAME         suffix appended to output dir (default = today's date)
#   PY               python interpreter (default = agentdebug env)
#
# Outputs:
#   eval_runs/<bench>__<run-name>/<model>/{answers,scored_results,trajectory}.jsonl
#   halu_runs/<bench>__<run-name>/<model>/{halu_results.jsonl,halu_summary.json}

set -euo pipefail

# --------------------------------------------------------------------------- #
# CHANGE ME — comma-separated list of model ids to evaluate.                  #
# --------------------------------------------------------------------------- #
: "${MODELS:=gpt-4o-mini}"

# --------------------------------------------------------------------------- #
# Defaults — override by `LIMIT=20 USE_TOOLS=1 bash run_bench.sh A`.          #
# --------------------------------------------------------------------------- #
: "${LIMIT:=0}"
: "${USE_TOOLS:=1}"
: "${EVIDENCE_CHAIN:=graph,web,literature}"
: "${INCLUDE_CORRECT:=1}"
: "${WORKERS:=1}"
# literature_fetch is process-global Semaphore-bounded so MinerU/PDF parses
# don't herd. Default to WORKERS so concurrency scales together; override via
# LIT_FETCH_WORKERS=2 if you're memory-constrained.
: "${LIT_FETCH_WORKERS:=$WORKERS}"
export EVAL_LITERATURE_FETCH_CONCURRENCY="$LIT_FETCH_WORKERS"
: "${HALU_EXTRACTOR_CONCURRENCY:=4}"
: "${HALU_JUDGE_CONCURRENCY:=10}"
: "${SUBSET:=}"
: "${RUN_NAME:=$(date +%Y%m%d)}"
: "${PY:=python}"

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
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
BENCH_A_GRAPH="$ROOT/benchmark_runs/proteinlmbench_full_graphbench/global_graph.graphml"
BENCH_A_TAG="paired_protein_v2${TAG_SUFFIX}"

BENCH_B_DATASET="$ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2/$SAMPLES_FILE"
BENCH_B_GRAPH="$ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/global_graph.graphml"
BENCH_B_TAG="paired_enhanced_v2${TAG_SUFFIX}"

# --------------------------------------------------------------------------- #
# Argparse                                                                    #
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Sanity checks                                                               #
# --------------------------------------------------------------------------- #
if [[ ! -f "$ROOT/evaluation/.env" ]]; then
  echo "ERROR: $ROOT/evaluation/.env not found." >&2
  exit 2
fi
# Pull keys from .env to fail loud BEFORE the slow runner kicks off.
set +u
# shellcheck disable=SC1091
source <(grep -E '^(BOYUE_API_KEY|INTERN_API_KEY)=' "$ROOT/evaluation/.env" | sed 's/^/export /')
set -u
if [[ -z "${BOYUE_API_KEY:-}" ]]; then
  echo "ERROR: BOYUE_API_KEY missing in evaluation/.env (needed by the agent runner)." >&2
  exit 2
fi
if [[ -z "${INTERN_API_KEY:-}" ]]; then
  echo "ERROR: INTERN_API_KEY missing in evaluation/.env (needed by the halu pipeline)." >&2
  exit 2
fi

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
_USE_TOOLS_FLAG="--use-tools"
[[ "$USE_TOOLS" = "0" ]] && _USE_TOOLS_FLAG="--no-tools"

_INCLUDE_CORRECT_FLAG=""
[[ "$INCLUDE_CORRECT" = "1" ]] && _INCLUDE_CORRECT_FLAG="--include-correct"

_LIMIT_FLAG=""
[[ "$LIMIT" -gt 0 ]] && _LIMIT_FLAG="--limit $LIMIT"

run_one_bench() {
  local tag="$1" dataset="$2" graph="$3"
  local eval_out="$ROOT/evaluation/eval_runs/${tag}__${RUN_NAME}"
  local halu_out="$ROOT/evaluation/halu_runs/${tag}__${RUN_NAME}"

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
  echo "  graph:      $graph"
  echo "  models:     $MODELS"
  echo "  limit:      ${LIMIT:-0}    use_tools: $USE_TOOLS"
  echo "  workers:    $WORKERS    lit_fetch_workers: $LIT_FETCH_WORKERS"
  echo "  evidence:   $EVIDENCE_CHAIN"
  echo "  eval_out:   $eval_out"
  echo "  halu_out:   $halu_out"
  echo "=================================================================="

  # ---- Phase 1: agent evaluation (runner) ----
  cd "$ROOT"
  # shellcheck disable=SC2086
  "$PY" -m evaluation.runner \
    --dataset "$dataset" \
    --models "$MODELS" \
    --output-dir "$eval_out" \
    --per-sample-timeout 600 \
    --sample-concurrency "$WORKERS" \
    $_USE_TOOLS_FLAG \
    $_LIMIT_FLAG

  # ---- Phase 2: hallucination detection (halu.cli) ----
  # shellcheck disable=SC2086
  "$PY" -m evaluation.halu.cli \
    --runs-dir "$eval_out" \
    --dataset "$dataset" \
    --graph "$graph" \
    --output-dir "$halu_out" \
    --evidence-chain "$EVIDENCE_CHAIN" \
    --extractor-concurrency "$HALU_EXTRACTOR_CONCURRENCY" \
    --judge-concurrency "$HALU_JUDGE_CONCURRENCY" \
    $_INCLUDE_CORRECT_FLAG \
    $_LIMIT_FLAG

  echo
  echo "DONE: $tag"
  echo "  Per-model halu summary:"
  for d in "$halu_out"/*/; do
    if [[ -f "$d/halu_summary.json" ]]; then
      "$PY" -c "
import json, sys
s = json.load(open(sys.argv[1]))
ov = s['aggregate'].get('overall', {})
print(f\"  - {s['model']:30s}  n_samples={ov.get('n_samples',0):3d}  n_claims={ov.get('n_claims',0):4d}  HR_macro={ov.get('HR_macro',0):.3f}  HS_macro={ov.get('HS_macro',0):.3f}  HS_w_micro={ov.get('HS_weighted_micro',0):.3f}  HF_rate={ov.get('HF_rate',0):.3f}  refuted={ov.get('n_refuted',0)}/{ov.get('n_claims',0)}\")
" "$d/halu_summary.json"
    fi
  done
}

# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
for b in "${BENCHES[@]}"; do
  if [[ "$b" = "A" ]]; then
    run_one_bench "$BENCH_A_TAG" "$BENCH_A_DATASET" "$BENCH_A_GRAPH"
  else
    run_one_bench "$BENCH_B_TAG" "$BENCH_B_DATASET" "$BENCH_B_GRAPH"
  fi
done

echo
echo "All requested benches complete."
