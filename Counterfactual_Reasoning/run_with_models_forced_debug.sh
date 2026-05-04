#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AGDEBUGGER_FORCE_INITIAL_ANSWER="${AGDEBUGGER_FORCE_INITIAL_ANSWER:-option4}"

exec bash "${SCRIPT_DIR}/run_with_models_debug.sh" "$@"
