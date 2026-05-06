#!/usr/bin/env bash
# Minimal real run of the pubmed_graph pipeline.
# 1 seed keyword -> 3 PubMed papers -> full text + abstract fallback ->
# chunking -> Intern LLM triple extraction (8 chunks) -> GraphML.
#
# Usage:
#   ./scripts/run_minimal.sh                  # default output dir
#   ./scripts/run_minimal.sh /tmp/my_run      # custom output dir

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT_DIR/pipeline_config.minimal.json"
ENV_SCRIPT="$ROOT_DIR/triple_extraction_env.sh"
OUT_DIR="${1:-$ROOT_DIR/pipeline_outputs_minimal}"

cd "$ROOT_DIR"

# shellcheck disable=SC1090
source "$ENV_SCRIPT"

echo "============================================================"
echo "minimal pipeline run"
echo "  seed:    FGFR3 urothelial carcinoma (mechanistic, 1 seed, retmax=3)"
echo "  chunks:  limit=24"
echo "  output:  $OUT_DIR"
echo "============================================================"

rm -rf "$OUT_DIR"

"$PYTHON_BIN" literature_pipeline.py \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --extract-triples \
  --chunk-limit 24 \
  --export-graph \
  --graph-output "$OUT_DIR/global_graph.graphml"

echo
echo "============================================================"
echo "results"
echo "============================================================"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

out = Path("$OUT_DIR")
summary = json.loads((out / "phase_summary.json").read_text())

def get(d, *keys):
    cur = d
    for k in keys:
        cur = (cur or {}).get(k)
    return cur

print(f"  phase 1  expanded keywords     : {get(summary,'phase_1_keyword_expansion','accepted_terms')}")
print(f"  phase 2  retrieved unique      : {get(summary,'phase_2_retrieval_and_filtering','retrieved_unique_papers')}")
print(f"  phase 2  kept after scoring    : {get(summary,'phase_2_retrieval_and_filtering','kept_papers')}")
print(f"  phase 3  fulltext records      : {get(summary,'phase_3_fulltext','fulltext_records')}")
print(f"  phase 3  abstract-only records : {get(summary,'phase_3_fulltext','abstract_only_records')}")
print(f"  phase 3  cache hits            : {get(summary,'phase_3_fulltext','cache_hits')}")
print(f"  phase 3  chunks (file)         : {sum(1 for _ in (out/'chunks.jsonl').open())}")
print(f"  phase 4  raw triples           : {get(summary,'phase_4_triple_extraction','raw_triples')}")
print(f"  phase 4  normalized triples    : {get(summary,'phase_4_triple_extraction','normalized_triples')}")
print(f"  phase 4  errors                : {get(summary,'phase_4_triple_extraction','errors')}")
print(f"  phase 5  graph nodes           : {get(summary,'phase_5_graph_export','global_nodes')}")
print(f"  phase 5  graph edges           : {get(summary,'phase_5_graph_export','global_edges')}")

triples_path = out / "normalized_triples.jsonl"
if triples_path.exists():
    print()
    print("  sample triples:")
    for i, line in enumerate(triples_path.open()):
        if i >= 3:
            break
        t = json.loads(line)
        print(f"    [{t.get('head_type','?')}] {t.get('head','?')}  --{t.get('normalized_relation','?')}-->  [{t.get('tail_type','?')}] {t.get('tail','?')}")

print()
print(f"  full summary : {out}/phase_summary.json")
print(f"  graphml      : {out}/global_graph.graphml")
PY
