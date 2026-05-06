#!/usr/bin/env bash
# End-to-end smoke test for the pubmed_graph pipeline.
# Runs the smallest viable input through every phase and verifies the output.
#
# Usage:
#   ./scripts/run_smoke_test.sh                  # default output dir
#   ./scripts/run_smoke_test.sh /tmp/my_smoke    # custom output dir
#   KEEP_OUTPUT=1 ./scripts/run_smoke_test.sh    # keep output dir on success

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT_DIR/pipeline_config.smoke.json"
ENV_SCRIPT="$ROOT_DIR/triple_extraction_env.sh"
OUT_DIR="${1:-/tmp/pubmed_graph_smoke_run}"

cd "$ROOT_DIR"

echo "============================================================"
echo "pubmed_graph pipeline smoke test"
echo "  python:    $PYTHON_BIN"
echo "  config:    $CONFIG"
echo "  output:    $OUT_DIR"
echo "============================================================"

# --- preflight ---
[[ -x "$PYTHON_BIN" ]] || { echo "[FAIL] python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$CONFIG"     ]] || { echo "[FAIL] missing config: $CONFIG"           >&2; exit 1; }
[[ -f "$ENV_SCRIPT" ]] || { echo "[FAIL] missing env script: $ENV_SCRIPT"   >&2; exit 1; }

# --- step 0: load extraction credentials ---
echo
echo "[step 0] loading extraction credentials"
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
for var in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL; do
  if [[ -z "${!var:-}" ]]; then
    echo "[FAIL] env var $var is empty after sourcing $ENV_SCRIPT" >&2
    exit 1
  fi
  echo "  $var = set"
done

# --- step 1: static import + bug-fix sanity check ---
echo
echo "[step 1] static import + bug-fix sanity check"
"$PYTHON_BIN" -m compileall -q pubmed_graph literature_pipeline.py
"$PYTHON_BIN" - <<'PY'
import pubmed_graph.workflow, pubmed_graph.utils, pubmed_graph.graph_ops
import pubmed_graph.normalize, pubmed_graph.triple_extraction
import pubmed_graph.fulltext, pubmed_graph.embeddings, pubmed_graph.retrieval
from pubmed_graph.utils import extract_html_text, clean_crossref_landing_text
assert "dataLayer" not in extract_html_text("<script>window.dataLayer=1</script>Hello"), \
    "extract_html_text bug regressed: <script> body not stripped"
sample = "<p>Abstract This is a long enough body sentence about cancer signaling pathways and gene regulation in tumor cells.</p>"
assert isinstance(clean_crossref_landing_text(sample), str), \
    "clean_crossref_landing_text bug regressed: still raises"
print("  all imports OK + utils bugfixes verified")
PY

# --- step 2: end-to-end pipeline run ---
echo
echo "[step 2] running pipeline (1 seed, retmax=3, chunk-limit=6)"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" literature_pipeline.py \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --extract-triples \
  --chunk-limit 6 \
  --export-graph \
  --graph-output "$OUT_DIR/global_graph.graphml"

# --- step 3: verify outputs ---
echo
echo "[step 3] verifying outputs against phase_summary.json"
"$PYTHON_BIN" "$ROOT_DIR/scripts/smoke_check.py" "$OUT_DIR"

# --- step 4: GraphML readability cross-check ---
echo
echo "[step 4] GraphML readability cross-check"
"$PYTHON_BIN" - <<PY
import networkx as nx
G = nx.read_graphml("$OUT_DIR/global_graph.graphml")
print(f"  GraphML loaded: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
PY

echo
echo "============================================================"
echo "SMOKE TEST PASSED"
echo "  output dir: $OUT_DIR"
echo "  inspect:    $OUT_DIR/phase_summary.json"
echo "============================================================"

if [[ -z "${KEEP_OUTPUT:-}" ]]; then
  echo "(set KEEP_OUTPUT=1 to keep $OUT_DIR after success)"
fi
