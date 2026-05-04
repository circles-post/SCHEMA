#!/usr/bin/env bash
# Run halu phase 2 (hallucination analysis) on every model that has finished
# evaluation but has NOT been halu'd yet. Idempotent: models whose
# halu_summary.json already exists are skipped.
#
# Usage:
#   bash evaluation/scripts/run_halu_pending.sh [bench]
#     bench: A (default) | B
#   bash evaluation/scripts/run_halu_pending.sh A custom_run_name
#
# Optional env overrides:
#   RUN_NAME                default "full_tool_models_20260427"
#   EVIDENCE_CHAIN          default "graph,web,literature"
#   EXTRACTOR_CONCURRENCY   default 4
#   JUDGE_CONCURRENCY       default 10
#   FORCE                   1 = redo halu even for already-done models
#   PY                      python interpreter
#
# Behavior:
#   * Eligible model = has BOTH trajectory.jsonl AND scored_results.jsonl
#     non-empty under eval_runs/.../<model>/.
#   * Done = halu_summary.json present under halu_runs/.../<model>/.
#   * Each pending model runs in its own halu.cli invocation so a crash on
#     one doesn't kill the rest.
#   * halu's extractor + judge caches are shared across models (keyed by
#     content hash) so retries are token-free.

set -uo pipefail

ROOT="/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa"
: "${PY:=/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/agentdebug/bin/python}"
: "${RUN_NAME:=full_tool_models_20260427}"
: "${EVIDENCE_CHAIN:=graph,web,literature}"
: "${EXTRACTOR_CONCURRENCY:=4}"
: "${JUDGE_CONCURRENCY:=10}"
: "${FORCE:=0}"

WHICH="${1:-A}"
[[ -n "${2:-}" ]] && RUN_NAME="$2"

case "$WHICH" in
  A|a)
    EVAL_DIR="$ROOT/evaluation/eval_runs/paired_protein_v2_balanced__${RUN_NAME}"
    HALU_DIR="$ROOT/evaluation/halu_runs/paired_protein_v2_balanced__${RUN_NAME}"
    DATASET="$ROOT/benchmark_runs/proteinlmbench_full_graphbench/qgen_paired_protein_v2/samples_balanced.jsonl"
    GRAPH="$ROOT/benchmark_runs/proteinlmbench_full_graphbench/global_graph.graphml"
    ;;
  B|b)
    EVAL_DIR="$ROOT/evaluation/eval_runs/paired_enhanced_v2_balanced__${RUN_NAME}"
    HALU_DIR="$ROOT/evaluation/halu_runs/paired_enhanced_v2_balanced__${RUN_NAME}"
    DATASET="$ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2/samples_balanced.jsonl"
    GRAPH="$ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/global_graph.graphml"
    ;;
  *)
    echo "ERROR: first arg must be 'A' or 'B' (got: $WHICH)" >&2
    exit 2
    ;;
esac

if [[ ! -d "$EVAL_DIR" ]]; then
  echo "ERROR: eval dir not found: $EVAL_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Build (eligible, pending) sets
# ---------------------------------------------------------------------------
ELIGIBLE=()
INELIGIBLE=()
for d in "$EVAL_DIR"/*/; do
  [[ -d "$d" ]] || continue
  m=$(basename "$d")
  has_traj=0; has_scored=0
  [[ -f "$d/trajectory.jsonl"     && -s "$d/trajectory.jsonl"     ]] && has_traj=1
  [[ -f "$d/scored_results.jsonl" && -s "$d/scored_results.jsonl" ]] && has_scored=1
  if [[ $has_traj -eq 1 && $has_scored -eq 1 ]]; then
    ELIGIBLE+=("$m")
  else
    INELIGIBLE+=("$m (traj=$has_traj scored=$has_scored)")
  fi
done

PENDING=()
SKIPPED=()
for m in "${ELIGIBLE[@]}"; do
  if [[ "$FORCE" != "1" && -f "$HALU_DIR/$m/halu_summary.json" ]]; then
    SKIPPED+=("$m")
  else
    PENDING+=("$m")
  fi
done

echo "================================================================="
echo "Halu phase 2 — pending-model scan"
echo "  bench:      $WHICH       run_name:  $RUN_NAME"
echo "  eval_dir:   $EVAL_DIR"
echo "  halu_dir:   $HALU_DIR"
echo "  dataset:    $DATASET ($(wc -l < "$DATASET") samples)"
echo "  graph:      $GRAPH"
echo "  evidence:   $EVIDENCE_CHAIN     extractor=$EXTRACTOR_CONCURRENCY  judge=$JUDGE_CONCURRENCY"
echo
echo "  eligible (trajectory+scored both present): ${#ELIGIBLE[@]}"
for m in "${ELIGIBLE[@]}"; do echo "    + $m"; done
if [[ ${#INELIGIBLE[@]} -gt 0 ]]; then
  echo "  ineligible (missing trajectory or scored): ${#INELIGIBLE[@]}"
  for x in "${INELIGIBLE[@]}"; do echo "    - $x"; done
fi
echo "  already DONE (halu_summary.json present): ${#SKIPPED[@]}"
for m in "${SKIPPED[@]}"; do echo "    ✓ $m"; done
echo "  PENDING: ${#PENDING[@]}"
for m in "${PENDING[@]}"; do echo "    ► $m"; done
echo "================================================================="

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "Nothing to do. Set FORCE=1 to redo halu on already-done models."
  exit 0
fi

# ---------------------------------------------------------------------------
# Run halu per pending model
# ---------------------------------------------------------------------------
mkdir -p "$HALU_DIR"
cd "$ROOT"

for m in "${PENDING[@]}"; do
  echo
  echo "=================================================================="
  echo ">>> $(date +'%Y-%m-%d %H:%M:%S')  Halu: $m"
  echo "=================================================================="
  start=$(date +%s)

  "$PY" -m evaluation.halu.cli \
    --runs-dir   "$EVAL_DIR" \
    --dataset    "$DATASET" \
    --graph      "$GRAPH" \
    --output-dir "$HALU_DIR" \
    --models     "$m" \
    --include-correct \
    --evidence-chain "$EVIDENCE_CHAIN" \
    --extractor-concurrency "$EXTRACTOR_CONCURRENCY" \
    --judge-concurrency "$JUDGE_CONCURRENCY"
  status=$?
  dur=$(( $(date +%s) - start ))

  if [[ $status -eq 0 ]]; then
    echo ">>> $m halu FINISHED in ${dur}s"
  else
    echo "!!! $m halu exited with status $status after ${dur}s — continuing"
  fi
done

# ---------------------------------------------------------------------------
# Final scoreboard
# ---------------------------------------------------------------------------
echo
echo "================================================================="
echo "Final halu scoreboard ($HALU_DIR):"
for d in "$HALU_DIR"/*/; do
  [[ -d "$d" ]] || continue
  m=$(basename "$d")
  s="$d/halu_summary.json"
  if [[ -f "$s" ]]; then
    line=$("$PY" -c "
import json, sys
try:
    s = json.load(open(sys.argv[1]))
    ov = s.get('aggregate', {}).get('overall', {})
    n = ov.get('n_samples', 0)
    nc = ov.get('n_claims', 0)
    hr = ov.get('HR_macro', 0.0)
    hs = ov.get('HS_macro', 0.0)
    hf = ov.get('HF_rate', 0.0)
    nr = ov.get('n_refuted', 0)
    print(f'n={n:>3d} n_claims={nc:>4d} HR={hr:.3f} HS={hs:.3f} HF_rate={hf:.3f} refuted={nr}/{nc}')
except Exception as e:
    print(f'(could not parse: {e})')
" "$s")
    printf "  %-46s %s\n" "$m" "$line"
  else
    printf "  %-46s (no halu_summary.json — incomplete)\n" "$m"
  fi
done
echo "================================================================="
