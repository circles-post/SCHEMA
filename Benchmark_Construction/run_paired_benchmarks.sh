#!/usr/bin/env bash
# Re-run the paired benchmark generation after the expert-review patch.
# Parameters are identical to the previous qgen_paired_* runs, only the
# output dirs are renamed so the old outputs stay intact for comparison.
#
# Usage:
#   bash run_paired_benchmarks.sh
#
# Prerequisites:
#   - $INTERN_API_KEY must be set (via triple_extraction_env.sh below)
#   - sandbox 100.99.239.71:8080 is reachable
#   - conda env "agentdebug" available

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
REPO_ROOT="/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa"
PY="/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/agentdebug/bin/python"

cd "$REPO_ROOT"

# Pulls INTERN_API_KEY / OPENAI_* etc.
# shellcheck disable=SC1091
source triple_extraction_env.sh

if [[ -z "${INTERN_API_KEY:-}" ]]; then
  echo "ERROR: INTERN_API_KEY is not set after sourcing triple_extraction_env.sh" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT"

# ---------------------------------------------------------------------------
# Run A — Protein Bench
# ---------------------------------------------------------------------------
RUN_A="$REPO_ROOT/benchmark_runs/proteinlmbench_full_graphbench/qgen_paired_protein_v2"
echo "====================================================================="
echo "[A] Protein Bench  →  $RUN_A"
echo "====================================================================="

rm -rf   "$RUN_A"
mkdir -p "$RUN_A/.corrob_cache" "$RUN_A/.val_cache"

"$PY" -m question_generation.cli \
  --triples benchmark_runs/proteinlmbench_full_graphbench/normalized_triples.jsonl \
  --chunks  benchmark_runs/proteinlmbench_full_graphbench/chunks.jsonl \
  --graph   benchmark_runs/proteinlmbench_full_graphbench/global_graph.graphml \
  --output         "$RUN_A/samples.jsonl" \
  --summary-output "$RUN_A/summary.json" \
  --log-file       "$RUN_A/run.log" \
  \
  --question-types claim_choice boolean_support two_hop_tail essay experiment_code \
  --node-based auto \
  --node-quota T1=3 T2=2 T3=1 \
  --ratio two_hop_tail=0.25 experiment_code=0.15 essay=0.20 claim_choice=0.30 boolean_support=0.10 \
  --max-samples 3000 \
  \
  --min-local-support 1 \
  --min-confidence 0.7 \
  \
  --corroboration-mode required \
  --min-external-sources 1 \
  --corroboration-tool-timeout 60 \
  --corroboration-cache-dir "$RUN_A/.corrob_cache" \
  \
  --validation-mode hybrid_model \
  --validator-enabled \
  --validator-model    intern-s1-pro \
  --validator-base-url https://chat.intern-ai.org.cn/api/v1/ \
  --validator-api-key  "$INTERN_API_KEY" \
  --validation-cache-dir "$RUN_A/.val_cache" \
  --retrieval-top-k 3 \
  \
  --experiment-difficulty medium \
  --experiment-generation-mode hybrid \
  --llm-code-selection auto \
  \
  --workers 4 \
  --sample-timeout 240

# Quick headline stats for A
"$PY" -c "
import json
s = json.load(open('$RUN_A/summary.json'))
print('[A] accepted:', s.get('accepted_questions'))
print('[A] validation:', s.get('validation'))
a = s.get('allocation') or {}
print('[A] per_tier_selected:', a.get('per_tier_selected'))
print('[A] per_type_selected:', a.get('per_type_selected'))
c = s.get('coverage') or {}
if c:
    print('[A] nodes: %s/%s = %.1f%%' % (
        c.get('covered_nodes'), c.get('total_graph_nodes'), (c.get('node_coverage_rate',0)*100)))
"

# ---------------------------------------------------------------------------
# Run B — Enhanced (pathvqa), dedup-against Run A
# ---------------------------------------------------------------------------
RUN_B="$REPO_ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2"
echo ""
echo "====================================================================="
echo "[B] Enhanced (pathvqa)  →  $RUN_B"
echo "    --dedup-against  ← $RUN_A/samples.jsonl"
echo "====================================================================="

rm -rf   "$RUN_B"
mkdir -p "$RUN_B/.corrob_cache" "$RUN_B/.val_cache"

"$PY" -m question_generation.cli \
  --triples benchmark_runs/protein_plus_pathvqa_500_v3/merged_triples.jsonl \
  --chunks  benchmark_runs/protein_plus_pathvqa_500_v3/_incremental/chunks.jsonl \
  --graph   benchmark_runs/protein_plus_pathvqa_500_v3/global_graph.graphml \
  --vqa-triples     benchmark_runs/protein_plus_pathvqa_500_v3/benchmark_triples.jsonl \
  --vqa-image-index benchmark_runs/protein_plus_pathvqa_500_v3/benchmark_image_index.json \
  --output         "$RUN_B/samples.jsonl" \
  --summary-output "$RUN_B/summary.json" \
  --log-file       "$RUN_B/run.log" \
  \
  --question-types claim_choice boolean_support two_hop_tail essay experiment_code vqa \
  --node-based auto \
  --node-quota T1=3 T2=2 T3=1 \
  --ratio two_hop_tail=0.20 experiment_code=0.10 essay=0.15 claim_choice=0.20 boolean_support=0.05 vqa=0.30 \
  --max-samples 3000 \
  \
  --dedup-against "$RUN_A/samples.jsonl" \
  \
  --min-local-support 1 \
  --min-confidence 0.7 \
  \
  --corroboration-mode required \
  --min-external-sources 1 \
  --corroboration-tool-timeout 60 \
  --corroboration-cache-dir "$RUN_B/.corrob_cache" \
  \
  --validation-mode hybrid_model \
  --validator-enabled \
  --validator-model    intern-s1-pro \
  --validator-base-url https://chat.intern-ai.org.cn/api/v1/ \
  --validator-api-key  "$INTERN_API_KEY" \
  --validation-cache-dir "$RUN_B/.val_cache" \
  --retrieval-top-k 3 \
  \
  --experiment-difficulty medium \
  --experiment-generation-mode hybrid \
  --llm-code-selection auto \
  \
  --workers 4 \
  --sample-timeout 240

# Quick headline stats for B
"$PY" -c "
import json
s = json.load(open('$RUN_B/summary.json'))
print('[B] accepted:', s.get('accepted_questions'))
print('[B] validation:', s.get('validation'))
a = s.get('allocation') or {}
print('[B] per_tier_selected:', a.get('per_tier_selected'))
print('[B] per_type_selected:', a.get('per_type_selected'))
c = s.get('coverage') or {}
if c:
    print('[B] nodes: %s/%s = %.1f%%' % (
        c.get('covered_nodes'), c.get('total_graph_nodes'), (c.get('node_coverage_rate',0)*100)))
"

# ---------------------------------------------------------------------------
# Final cross-check: overlap between A and B must be 0 (dedup-against verified)
# ---------------------------------------------------------------------------
echo ""
echo "====================================================================="
echo "Cross-run dedup check"
echo "====================================================================="
"$PY" -c "
import json
from question_generation.dedup import dedup_key
a_keys = {dedup_key(json.loads(l)) for l in open('$RUN_A/samples.jsonl')}
b_keys = {dedup_key(json.loads(l)) for l in open('$RUN_B/samples.jsonl')}
overlap = a_keys & b_keys
print(f'A unique keys: {len(a_keys)}   B unique keys: {len(b_keys)}   overlap: {len(overlap)}')
if overlap:
    print('WARNING: overlap > 0 — --dedup-against did not catch everything')
"

echo ""
echo "Done."
echo "  A → $RUN_A/samples.jsonl"
echo "  B → $RUN_B/samples.jsonl"
