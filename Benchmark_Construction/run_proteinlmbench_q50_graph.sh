#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/new_graphcompass/bin/python"
CONFIG_PATH="${ROOT_DIR}/pipeline_config.benchmark.q50.json"
OUTPUT_DIR="${1:-${ROOT_DIR}/benchmark_runs/proteinlmbench_q50_graph}"
CHUNK_LIMIT="${2:-60}"
GRAPH_PATH="${OUTPUT_DIR}/global_graph.graphml"

cd "${ROOT_DIR}"

echo "[run] config=${CONFIG_PATH}"
echo "[run] output_dir=${OUTPUT_DIR}"
echo "[run] chunk_limit=${CHUNK_LIMIT}"

"${PYTHON_BIN}" literature_pipeline.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --extract-triples \
  --chunk-limit "${CHUNK_LIMIT}" \
  --export-graph \
  --graph-output "${GRAPH_PATH}"

echo
echo "[done] phase summary: ${OUTPUT_DIR}/phase_summary.json"
echo "[done] triples: ${OUTPUT_DIR}/normalized_triples.jsonl"
echo "[done] graph: ${GRAPH_PATH}"
