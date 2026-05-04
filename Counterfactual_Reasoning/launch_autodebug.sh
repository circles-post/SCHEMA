#!/usr/bin/env bash
# Compatibility wrapper. The real launcher logic now lives in run_with_models.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_with_models.sh" "$@"
