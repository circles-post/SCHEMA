#!/usr/bin/env bash
# run_graphbench_questions_full.sh
#
# One-shot production run for question_generation on top of the
# proteinlmbench graphbench output. Produces ~1000-1500 benchmark
# questions across all 5 types with full LLM cross-validation.
#
# Pipeline:
#   1. claim_choice / boolean_support / two_hop_tail / essay
#      → rule layer + intern-s1-pro judge_claim / judge_essay
#   2. experiment_code
#      → LLM per-triple spec generation (plan C, hybrid mode)
#      → sandbox gate (reference passes + incomplete fails)
#      → intern-s1-pro judge_experiment_code (semantic alignment)
#
# Before running: triple_extraction_env.sh must be sourced so
# OPENAI_API_KEY / INTERN_API_KEY are exported.
#
# Monitor with:
#   tail -f benchmark_runs/proteinlmbench_full_graphbench/question_generation/run.log \
#     | grep -E 'phase|PASS|REJECT|llm_generate|vmode=|build_experiment_sample TOTAL'
#
# Expected wall clock: ~60-90 minutes (4 non-experiment types run in
# parallel via workers=8; experiment_code runs with workers=2 due to
# GitHub / LLM rate limits).

set -euo pipefail

REPO_ROOT="/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa"
cd "$REPO_ROOT"

# --- 1. ensure sandbox host env does not leak from a prior session ----
unset QG_SANDBOX_HOST

# --- 2. source LLM credentials (INTERN_API_KEY / OPENAI_API_KEY) ------
if [[ -f "$REPO_ROOT/triple_extraction_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/triple_extraction_env.sh"
else
  echo "[FATAL] triple_extraction_env.sh not found at $REPO_ROOT" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${INTERN_API_KEY:-}" ]]; then
  echo "[FATAL] no LLM API key in environment — judge will be degraded" >&2
  exit 1
fi

# --- 3. wipe stale bytecode so any recent edits take effect -----------
find "$REPO_ROOT/question_generation" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# --- 4. launch the run ------------------------------------------------
echo "============================================================"
echo "  full graphbench question_generation run"
echo "  started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

./scripts/run_graphbench_questions.sh \
  --max-samples 2000 \
  --max-per-uniqueness-key 3 \
  --question-types claim_choice boolean_support two_hop_tail essay experiment_code \
  --experiment-difficulty mixed \
  --experiment-generation-mode hybrid \
  --llm-code-selection auto \
  --validation-mode hybrid_model \
  --validator-enabled \
  --validation-cache-dir "benchmark_runs/proteinlmbench_full_graphbench/.qg_val_cache" \
  --retrieval-top-k 3 \
  --workers 8 \
  --sample-timeout 600 \
  --force

echo
echo "============================================================"
echo "  run finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- 5. quick post-run summary ----------------------------------------
OUT_DIR="$REPO_ROOT/benchmark_runs/proteinlmbench_full_graphbench/question_generation"
SUMMARY="$OUT_DIR/summary.json"
SAMPLES="$OUT_DIR/question_samples.jsonl"

if [[ -f "$SUMMARY" ]]; then
  echo
  echo "=== summary.json ==="
  python3 -c "
import json
from collections import Counter
s = json.load(open('$SUMMARY'))
print(f'  triples            : {s.get(\"triples\")}')
print(f'  chunks             : {s.get(\"chunks\")}')
print(f'  sampled_subgraphs  : {s.get(\"sampled_subgraphs\")}')
print(f'  accepted_questions : {s.get(\"accepted_questions\")}')
print(f'  validation         : {s.get(\"validation\")}')
print(f'  experiment_blueprint_breakdown:')
for k, v in sorted((s.get('experiment_blueprint_breakdown') or {}).items()):
    print(f'      {k:40s} {v}')
"
fi

if [[ -f "$SAMPLES" ]]; then
  echo
  echo "=== per-type counts on disk ==="
  python3 -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('$SAMPLES')]
types = Counter(r['question_type'] for r in rows)
for k, v in sorted(types.items()):
    print(f'  {k:20s} {v}')

vmodes = Counter((r['question_type'], r['grounding']['validation_mode']) for r in rows)
print()
print('=== validation_mode breakdown ===')
for (qtype, vmode), v in sorted(vmodes.items()):
    print(f'  {qtype:20s} {vmode:30s} {v}')

sources = Counter(r.get('metadata', {}).get('generation_source') for r in rows if r['question_type'] == 'experiment_code')
if sources:
    print()
    print('=== experiment_code generation_source ===')
    for k, v in sorted(sources.items(), key=lambda x: str(x[0])):
        print(f'  {str(k):20s} {v}')
"
fi

echo
echo "output dir: $OUT_DIR"
echo "samples  :  $SAMPLES"
echo "summary  :  $SUMMARY"
echo "run.log  :  $OUT_DIR/run.log"
