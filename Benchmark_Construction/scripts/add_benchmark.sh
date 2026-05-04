#!/usr/bin/env bash
# add_benchmark.sh — wrapper for scripts/add_benchmark.py
#
# Pins the new_rl conda python, sources extraction credentials, and sets
# NO_PROXY so the internal embedding service is reachable. All remaining
# args are forwarded to the Python entrypoint verbatim.
#
# Example:
#   ./scripts/add_benchmark.sh --add SLAKE_EN \
#       --base-run benchmark_runs/proteinlmbench_full_sciverse \
#       --output-dir benchmark_runs/proteinlmbench_plus_slake_smoke \
#       --question-limit 50
#
#   ./scripts/add_benchmark.sh --add MedXpertQA_MM \
#       --base-run benchmark_runs/proteinlmbench_full_sciverse \
#       --output-dir benchmark_runs/proteinlmbench_plus_medxpert_smoke \
#       --question-limit 100 --skip-retrieval

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/new_rl/bin/python}"
ENV_SCRIPT="$ROOT_DIR/triple_extraction_env.sh"
SCRIPT="$ROOT_DIR/scripts/add_benchmark.py"

EMBEDDING_HOST="${EMBEDDING_HOST:-100.99.247.97}"

[[ -x "$PYTHON_BIN" ]] || { echo "[FAIL] python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$SCRIPT"     ]] || { echo "[FAIL] missing $SCRIPT" >&2; exit 1; }
[[ -f "$ENV_SCRIPT" ]] || { echo "[FAIL] missing $ENV_SCRIPT" >&2; exit 1; }

# shellcheck disable=SC1090
source "$ENV_SCRIPT"

if [[ -n "${NO_PROXY:-}" || -n "${no_proxy:-}" ]]; then
  EXISTING_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
  case ",$EXISTING_NO_PROXY," in
    *,$EMBEDDING_HOST,*) export NO_PROXY="$EXISTING_NO_PROXY" ;;
    *)                   export NO_PROXY="$EMBEDDING_HOST,$EXISTING_NO_PROXY" ;;
  esac
else
  export NO_PROXY="$EMBEDDING_HOST"
fi
export no_proxy="$NO_PROXY"

echo "python : $PYTHON_BIN"
echo "model  : ${OPENAI_MODEL:-<unset>}"
echo

exec "$PYTHON_BIN" "$SCRIPT" "$@"
