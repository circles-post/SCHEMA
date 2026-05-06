#!/usr/bin/env bash
# download_benchmarks.sh — wrapper around scripts/download_benchmarks.py
#
# Why this wrapper:
#   - pins the `new_rl` conda interpreter that has huggingface_hub + datasets
#   - optionally switches HF_ENDPOINT to the hf-mirror.com mirror for pjlab
#     network (flip with --mirror or HF_ENDPOINT env)
#   - forwards the rest of argv verbatim to the python script
#
# Usage:
#   ./scripts/download_benchmarks.sh --list
#   ./scripts/download_benchmarks.sh                       # download all to default dir
#   ./scripts/download_benchmarks.sh --only path-vqa
#   ./scripts/download_benchmarks.sh --mirror --only slake-en
#   HF_TOKEN=hf_xxxx ./scripts/download_benchmarks.sh --only omnimedvqa
#   ./scripts/download_benchmarks.sh --output-dir /data/my_benchmarks --dry-run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-python}"
SCRIPT="$ROOT_DIR/scripts/download_benchmarks.py"

MIRROR="${USE_HF_MIRROR:-false}"
FORWARD_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --mirror)     MIRROR="true" ;;
    --no-mirror)  MIRROR="false" ;;
    *)            FORWARD_ARGS+=("$arg") ;;
  esac
done

if [[ "$MIRROR" == "true" && -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi

# resumable downloads benefit from a stable HF cache; pin it under the
# project root unless the user set their own HF_HOME.
if [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${ROOT_DIR}/.hf_cache"
fi
mkdir -p "$HF_HOME"

[[ -x "$PY_BIN" ]]  || { echo "[FAIL] python not found: $PY_BIN" >&2; exit 1; }
[[ -f "$SCRIPT" ]]  || { echo "[FAIL] missing $SCRIPT" >&2; exit 1; }

echo "python      : $PY_BIN"
echo "HF_ENDPOINT : ${HF_ENDPOINT:-<default huggingface.co>}"
echo "HF_HOME     : $HF_HOME"
echo "HF_TOKEN    : $([[ -n "${HF_TOKEN:-}" ]] && echo "set" || echo "unset")"
echo

exec "$PY_BIN" "$SCRIPT" "${FORWARD_ARGS[@]}"
