#!/usr/bin/env bash
set -euo pipefail
source /mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa/triple_extraction_env.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/new_rl/bin/python"
BASE_CONFIG="$ROOT_DIR/pipeline_config.benchmark.json"
PIPELINE_ENTRY="$ROOT_DIR/literature_pipeline.py"
MODE="full"
OUTPUT_DIR=""
QUESTION_LIMIT=""
CHUNK_LIMIT=""
MAX_SEEDS=""
RETMAX_PER_KEYWORD=""
RELATED_EXPAND_LIMIT=""
RELATED_PER_SEED=""

usage() {
  cat <<'USAGE'
Usage:
  ./run_proteinlmbench_full_graph.sh [--sample] [--output-dir DIR] [--question-limit N] [--chunk-limit N] [--max-seeds N]

Examples:
  ./run_proteinlmbench_full_graph.sh --sample
  ./run_proteinlmbench_full_graph.sh --output-dir benchmark_runs/proteinlmbench_full_graph_v1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample)
      MODE="sample"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --question-limit)
      QUESTION_LIMIT="$2"
      shift 2
      ;;
    --chunk-limit)
      CHUNK_LIMIT="$2"
      shift 2
      ;;
    --max-seeds)
      MAX_SEEDS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" == "sample" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_runs/proteinlmbench_full_graph_sample_smoke}"
  QUESTION_LIMIT="${QUESTION_LIMIT:-5}"
  CHUNK_LIMIT="${CHUNK_LIMIT:-12}"
  MAX_SEEDS="${MAX_SEEDS:-12}"
  RETMAX_PER_KEYWORD="2"
  RELATED_EXPAND_LIMIT="1"
  RELATED_PER_SEED="1"
else
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_runs/proteinlmbench_full_graph_v1}"
  QUESTION_LIMIT="${QUESTION_LIMIT:-944}"
  CHUNK_LIMIT="${CHUNK_LIMIT:-0}"
  MAX_SEEDS="${MAX_SEEDS:-120}"
  RETMAX_PER_KEYWORD="5"
  RELATED_EXPAND_LIMIT="5"
  RELATED_PER_SEED="2"
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing base config: $BASE_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$PIPELINE_ENTRY" ]]; then
  echo "Missing pipeline entry: $PIPELINE_ENTRY" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python env: $PYTHON_BIN" >&2
  exit 1
fi

TMP_CONFIG="$(mktemp /tmp/proteinlmbench_full_config.XXXXXX.json)"
cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

cd "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY' "$BASE_CONFIG" "$TMP_CONFIG" "$MODE" "$QUESTION_LIMIT" "$MAX_SEEDS" "$RETMAX_PER_KEYWORD" "$RELATED_EXPAND_LIMIT" "$RELATED_PER_SEED"
import json
import sys
from pathlib import Path

base_config = Path(sys.argv[1])
out_config = Path(sys.argv[2])
mode = sys.argv[3]
question_limit = int(sys.argv[4])
max_seeds = int(sys.argv[5])
retmax = int(sys.argv[6])
related_expand_limit = int(sys.argv[7])
related_per_seed = int(sys.argv[8])

data = json.loads(base_config.read_text())
env_file = str(data.get("env_file", ".env"))
if env_file and not Path(env_file).is_absolute():
    data["env_file"] = str((base_config.parent / env_file).resolve())
data["project_name"] = f"proteinlmbench_pubmed_graph_{mode}"
data.setdefault("benchmark_seed_source", {})["question_limit"] = question_limit
data["benchmark_seed_source"]["max_seed_keywords"] = max_seeds
data.setdefault("retrieval", {})["retmax_per_keyword"] = retmax
data["retrieval"]["related_expand_limit"] = related_expand_limit
data["retrieval"]["related_per_seed"] = related_per_seed
out_config.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(out_config)
PY

REMOTE_URL="$($PYTHON_BIN - <<'PY' "$TMP_CONFIG"
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("scoring", {}).get("sapbert", {}).get("service_url", ""))
PY
)"

if [[ -n "$REMOTE_URL" ]]; then
  echo "[check] remote embedding service: $REMOTE_URL"
  "$PYTHON_BIN" - <<'PY' "$REMOTE_URL"
import json
import sys
import requests
base = sys.argv[1].rstrip('/')
rsp = requests.get(base + '/health', timeout=20)
rsp.raise_for_status()
print(rsp.text)
PY
fi

echo "[run] mode=$MODE"
echo "[run] config=$TMP_CONFIG"
echo "[run] output_dir=$OUTPUT_DIR"
echo "[run] question_limit=$QUESTION_LIMIT"
echo "[run] chunk_limit=$CHUNK_LIMIT"
echo "[run] max_seed_keywords=$MAX_SEEDS"

CMD=(
  "$PYTHON_BIN" "$PIPELINE_ENTRY"
  --config "$TMP_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --extract-triples
  --chunk-limit "$CHUNK_LIMIT"
  --export-graph
  --graph-output "$OUTPUT_DIR/global_graph.graphml"
)

"${CMD[@]}"

echo "[done] outputs written to $OUTPUT_DIR"
echo "[done] phase summary: $OUTPUT_DIR/phase_summary.json"
