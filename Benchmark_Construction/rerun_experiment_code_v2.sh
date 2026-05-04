#!/usr/bin/env bash
# Re-generate the experiment_code question slice that was lost in the
# qgen_paired_*_v2 runs because of the sandbox-harness ``name_hint`` bug.
#
# Per-bench cap: 100 experiment_code samples (per user request).
# After each bench finishes, new samples are appended into the existing
# qgen_paired_*_v2/samples.jsonl (backup left at samples.jsonl.bak_pre_expcode)
# and IDs are renumbered to avoid collisions with the original ids.

set -euo pipefail

REPO_ROOT="/mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa"
PY="/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/agentdebug/bin/python"

cd "$REPO_ROOT"

# shellcheck disable=SC1091
source triple_extraction_env.sh

if [[ -z "${INTERN_API_KEY:-}" ]]; then
  echo "ERROR: INTERN_API_KEY missing after sourcing triple_extraction_env.sh" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT"

RUN_A_DIR="$REPO_ROOT/benchmark_runs/proteinlmbench_full_graphbench/qgen_paired_protein_v2"
RUN_B_DIR="$REPO_ROOT/benchmark_runs/protein_plus_pathvqa_500_v3/qgen_paired_enhanced_v2"
RERUN_A="$RUN_A_DIR/expcode_rerun"
RERUN_B="$RUN_B_DIR/expcode_rerun"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
common_args=(
  --question-types experiment_code
  --ratio          experiment_code=1.0
  --max-samples    100
  --node-based     auto
  --node-quota     T1=3 T2=2 T3=1

  --min-local-support 1
  --min-confidence    0.7

  --corroboration-mode required
  --min-external-sources 1
  --corroboration-tool-timeout 60

  --validation-mode hybrid_model
  --validator-enabled
  --validator-model    intern-s1-pro
  --validator-base-url https://chat.intern-ai.org.cn/api/v1/
  --validator-api-key  "$INTERN_API_KEY"
  --retrieval-top-k 3

  --experiment-difficulty       medium
  --experiment-generation-mode  hybrid
  --llm-code-selection          auto

  --workers        4
  --sample-timeout 240
)

merge_into_main () {
  local main_path="$1"
  local rerun_path="$2"
  local id_start="$3"

  if [[ ! -s "$rerun_path" ]]; then
    echo "  WARN: $rerun_path is empty — nothing to merge"
    return 0
  fi

  local backup="${main_path}.bak_pre_expcode"
  if [[ ! -e "$backup" ]]; then
    cp "$main_path" "$backup"
    echo "  backup: $backup"
  else
    echo "  backup already exists at $backup (left untouched)"
  fi

  "$PY" - "$main_path" "$rerun_path" "$id_start" <<'PYEOF'
import json, sys, shutil
main_path, rerun_path, id_start = sys.argv[1], sys.argv[2], int(sys.argv[3])

existing_ids = set()
existing_lines = []
with open(main_path) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        existing_lines.append(line)
        try:
            existing_ids.add(json.loads(line).get("sample_id"))
        except Exception:
            pass

new_rows = []
with open(rerun_path) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        new_rows.append(json.loads(line))

# Renumber the new samples so they cannot collide with existing ids.
idx = id_start
appended = 0
out_lines = list(existing_lines)
for row in new_rows:
    while f"qg_{idx:06d}" in existing_ids:
        idx += 1
    new_id = f"qg_{idx:06d}"
    row["sample_id"] = new_id
    existing_ids.add(new_id)
    out_lines.append(json.dumps(row, ensure_ascii=False))
    idx += 1
    appended += 1

with open(main_path, "w") as f:
    for line in out_lines:
        f.write(line + "\n")

print(f"    merged: existing={len(existing_lines)}  appended={appended}  total={len(out_lines)}")
PYEOF
}

# ---------------------------------------------------------------------------
# Run A — Protein Bench
# ---------------------------------------------------------------------------
echo "====================================================================="
echo "[A] re-generating experiment_code (n=100)  →  $RERUN_A"
echo "====================================================================="

rm -rf "$RERUN_A"
mkdir -p "$RERUN_A"

"$PY" -m question_generation.cli \
  --triples benchmark_runs/proteinlmbench_full_graphbench/normalized_triples.jsonl \
  --chunks  benchmark_runs/proteinlmbench_full_graphbench/chunks.jsonl \
  --graph   benchmark_runs/proteinlmbench_full_graphbench/global_graph.graphml \
  --output         "$RERUN_A/samples.jsonl" \
  --summary-output "$RERUN_A/summary.json" \
  --log-file       "$RERUN_A/run.log" \
  --corroboration-cache-dir "$RUN_A_DIR/.corrob_cache" \
  --validation-cache-dir    "$RUN_A_DIR/.val_cache" \
  "${common_args[@]}"

"$PY" -c "
import json
s = json.load(open('$RERUN_A/summary.json'))
print('[A] expcode_rerun accepted:', s.get('accepted_questions'))
print('[A] expcode_rerun validation:', s.get('validation'))
print('[A] expcode_rerun blueprint_breakdown:', s.get('experiment_blueprint_breakdown'))
"

echo
echo "[A] merging $RERUN_A/samples.jsonl  →  $RUN_A_DIR/samples.jsonl"
merge_into_main "$RUN_A_DIR/samples.jsonl" "$RERUN_A/samples.jsonl" 999000

# ---------------------------------------------------------------------------
# Run B — Enhanced (pathvqa)  — dedup against the *updated* A
# ---------------------------------------------------------------------------
echo
echo "====================================================================="
echo "[B] re-generating experiment_code (n=100)  →  $RERUN_B"
echo "    --dedup-against  ← $RUN_A_DIR/samples.jsonl  (now incl. A's new expcode)"
echo "====================================================================="

rm -rf "$RERUN_B"
mkdir -p "$RERUN_B"

"$PY" -m question_generation.cli \
  --triples benchmark_runs/protein_plus_pathvqa_500_v3/merged_triples.jsonl \
  --chunks  benchmark_runs/protein_plus_pathvqa_500_v3/_incremental/chunks.jsonl \
  --graph   benchmark_runs/protein_plus_pathvqa_500_v3/global_graph.graphml \
  --output         "$RERUN_B/samples.jsonl" \
  --summary-output "$RERUN_B/summary.json" \
  --log-file       "$RERUN_B/run.log" \
  --dedup-against  "$RUN_A_DIR/samples.jsonl" \
  --corroboration-cache-dir "$RUN_B_DIR/.corrob_cache" \
  --validation-cache-dir    "$RUN_B_DIR/.val_cache" \
  "${common_args[@]}"

"$PY" -c "
import json
s = json.load(open('$RERUN_B/summary.json'))
print('[B] expcode_rerun accepted:', s.get('accepted_questions'))
print('[B] expcode_rerun validation:', s.get('validation'))
print('[B] expcode_rerun blueprint_breakdown:', s.get('experiment_blueprint_breakdown'))
"

echo
echo "[B] merging $RERUN_B/samples.jsonl  →  $RUN_B_DIR/samples.jsonl"
merge_into_main "$RUN_B_DIR/samples.jsonl" "$RERUN_B/samples.jsonl" 999000

# ---------------------------------------------------------------------------
# Final headline stats
# ---------------------------------------------------------------------------
echo
echo "====================================================================="
echo "Final per-type counts after merge"
echo "====================================================================="
"$PY" - <<PYEOF
import json, collections
for tag, path in [
    ("A_protein",  "$RUN_A_DIR/samples.jsonl"),
    ("B_enhanced", "$RUN_B_DIR/samples.jsonl"),
]:
    types = collections.Counter()
    n = 0
    for line in open(path):
        n += 1
        types[json.loads(line).get("question_type","?")] += 1
    print(f"{tag} total={n}  per_type={dict(types)}")
PYEOF

echo
echo "Done."
