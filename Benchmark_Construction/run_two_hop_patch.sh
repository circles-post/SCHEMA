#!/usr/bin/env bash
# Append-only re-run for two_hop_tail on the two paired graph benches.
# Uses the patched evidence_profiler (corroboration_will_run=True relaxes
# the static min_hop_support>=2 gate when --corroboration-mode required).
#
# Order matters: A merges first because B's --dedup-against points at A's
# merged samples.jsonl.
#
# Usage:
#   bash run_two_hop_patch.sh

set -euo pipefail

REPO_ROOT="/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa"
PY="/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/agentdebug/bin/python"

cd "$REPO_ROOT"

# shellcheck disable=SC1091
source triple_extraction_env.sh
if [[ -z "${INTERN_API_KEY:-}" ]]; then
  echo "ERROR: INTERN_API_KEY is not set after sourcing triple_extraction_env.sh" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT"

RUN_A="$REPO_ROOT/benchmark_runs/proteinlmbench_full_graphbench/qgen_paired_protein_v2"
RUN_B="$REPO_ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2"
PATCH_A="$RUN_A/two_hop_patch"
PATCH_B="$RUN_B/two_hop_patch"

# Sanity: existing samples must be there (we append to them)
for f in "$RUN_A/samples.jsonl" "$RUN_B/samples.jsonl"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing or empty $f — refuse to proceed" >&2
    exit 1
  fi
done

# Sanity: trafilatura present (corroboration tools needed)
if ! "$PY" -c "import trafilatura" >/dev/null 2>&1; then
  echo "ERROR: trafilatura not importable in $PY — corroboration would fail closed" >&2
  exit 1
fi

merge_two_hop () {
  local main="$1"
  local new="$2"
  local backup="$3"
  "$PY" - <<PYEOF
import json, shutil
from pathlib import Path
main   = Path("$main")
new    = Path("$new")
backup = Path("$backup")

if not new.exists() or new.stat().st_size == 0:
    print(f"ABORT: patch produced no rows at {new}")
    raise SystemExit(1)

shutil.copy(main, backup)
existing = main.read_text().splitlines()
offset   = len(existing)
new_rows = []
for i, line in enumerate(new.read_text().splitlines()):
    r = json.loads(line)
    r["sample_id"] = f"qg_{offset + i + 1:06d}"
    new_rows.append(json.dumps(r, ensure_ascii=False))

with main.open("w") as f:
    f.write("\n".join(existing + new_rows) + "\n")

print(f"merged into {main.name}: {len(existing)} existing + {len(new_rows)} two_hop = {len(existing)+len(new_rows)} total")
PYEOF
}

# ---------------------------------------------------------------------------
# Run A — two_hop_tail patch on protein graph
# ---------------------------------------------------------------------------
echo "====================================================================="
echo "[A] Protein bench  →  two_hop_tail patch"
echo "  output: $PATCH_A"
echo "====================================================================="

rm -rf "$PATCH_A"
mkdir -p "$PATCH_A"

"$PY" -m question_generation.cli \
  --triples benchmark_runs/proteinlmbench_full_graphbench/normalized_triples.jsonl \
  --chunks  benchmark_runs/proteinlmbench_full_graphbench/chunks.jsonl \
  --graph   benchmark_runs/proteinlmbench_full_graphbench/global_graph.graphml \
  --output         "$PATCH_A/samples.jsonl" \
  --summary-output "$PATCH_A/summary.json" \
  --log-file       "$PATCH_A/run.log" \
  \
  --question-types two_hop_tail \
  --max-samples 800 \
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
  --workers 4 \
  --sample-timeout 240

merge_two_hop \
  "$RUN_A/samples.jsonl" \
  "$PATCH_A/samples.jsonl" \
  "$RUN_A/samples.jsonl.bak_pre_two_hop"

# ---------------------------------------------------------------------------
# Run B — two_hop_tail patch on protein+pathvqa graph (dedup against merged A)
# ---------------------------------------------------------------------------
echo ""
echo "====================================================================="
echo "[B] Enhanced (pathvqa)  →  two_hop_tail patch"
echo "  output:        $PATCH_B"
echo "  dedup-against: $RUN_A/samples.jsonl  (merged)"
echo "====================================================================="

rm -rf "$PATCH_B"
mkdir -p "$PATCH_B"

"$PY" -m question_generation.cli \
  --triples benchmark_runs/protein_plus_pathvqa_500_v3/merged_triples.jsonl \
  --chunks  benchmark_runs/protein_plus_pathvqa_500_v3/_incremental/chunks.jsonl \
  --graph   benchmark_runs/protein_plus_pathvqa_500_v3/global_graph.graphml \
  --output         "$PATCH_B/samples.jsonl" \
  --summary-output "$PATCH_B/summary.json" \
  --log-file       "$PATCH_B/run.log" \
  \
  --question-types two_hop_tail \
  --max-samples 800 \
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
  --workers 4 \
  --sample-timeout 240

merge_two_hop \
  "$RUN_B/samples.jsonl" \
  "$PATCH_B/samples.jsonl" \
  "$RUN_B/samples.jsonl.bak_pre_two_hop"

# ---------------------------------------------------------------------------
# Final cross-check + headline stats
# ---------------------------------------------------------------------------
echo ""
echo "====================================================================="
echo "Cross-run check"
echo "====================================================================="
"$PY" - <<PYEOF
import json, collections
from question_generation.dedup import dedup_key

for label, path in [("A", "$RUN_A/samples.jsonl"), ("B", "$RUN_B/samples.jsonl")]:
    rows = [json.loads(l) for l in open(path)]
    types = collections.Counter(r["question_type"] for r in rows)
    print(f"[{label}] total={len(rows)}  types={dict(types)}")

ka = {dedup_key(json.loads(l)) for l in open("$RUN_A/samples.jsonl")}
kb = {dedup_key(json.loads(l)) for l in open("$RUN_B/samples.jsonl")}
print(f"A keys={len(ka)}  B keys={len(kb)}  overlap={len(ka & kb)}")
if ka & kb:
    print("WARNING: A∩B != 0 — --dedup-against did not catch everything")
PYEOF

echo ""
echo "Done."
echo "  A → $RUN_A/samples.jsonl   (backup: samples.jsonl.bak_pre_two_hop)"
echo "  B → $RUN_B/samples.jsonl   (backup: samples.jsonl.bak_pre_two_hop)"
