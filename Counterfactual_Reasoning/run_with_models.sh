#!/usr/bin/env bash
# Unified AGDebugger launcher:
# 1) configure models + API routing
# 2) start backend
# 3) run dataset autodebug
# 4) stop backend on exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
LOG_DIR="${SCRIPT_DIR}/logs"
RUN_STAMP="${AGDEBUGGER_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DAY="${RUN_STAMP:0:8}"
DAY_LOG_DIR="${LOG_DIR}/${RUN_DAY}"
RUN_ARTIFACT_DIR="${DAY_LOG_DIR}/run_${RUN_STAMP}"
UV_CACHE_DIR="${AGDEBUGGER_UV_CACHE_DIR:-${LOG_DIR}/.uv-cache}"
XDG_CACHE_HOME="${AGDEBUGGER_XDG_CACHE_HOME:-${LOG_DIR}/.cache}"

# ---------------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------------
CONDA_ENV="${AGDEBUGGER_CONDA_ENV:-}"
CONDA_BASE="${CONDA_BASE:-}"

HOST="${AGDEBUGGER_HOST:-127.0.0.1}"
PORT="${AGDEBUGGER_PORT:-8081}"
MODULE="${AGDEBUGGER_MODULE:-test_agent_debug:get_agent_team}"
READY_TIMEOUT="${AGDEBUGGER_READY_TIMEOUT:-120}"
RUN_TIMEOUT="${AGDEBUGGER_RUN_TIMEOUT:-300}"
RESET_TIMEOUT="${AGDEBUGGER_RESET_TIMEOUT:-120}"
QUESTION_TIMEOUT="${AGDEBUGGER_QUESTION_TIMEOUT:-900}"
QUESTION_STALL_TIMEOUT="${AGDEBUGGER_QUESTION_STALL_TIMEOUT:-120}"
QUESTION_RETRY_ATTEMPTS="${AGDEBUGGER_QUESTION_RETRY_ATTEMPTS:-1}"
DEBUG_STEP_TIMEOUT="${AGDEBUGGER_DEBUG_STEP_TIMEOUT:-300}"
ANALYSIS_TIMEOUT="${AGDEBUGGER_ANALYSIS_TIMEOUT_SEC:-600}"
MAX_CONCEPT_REPAIR_ATTEMPTS="${AGDEBUGGER_MAX_CONCEPT_REPAIR_ATTEMPTS:-3}"
STRICT_CONCEPT_REPAIR_ONLY="${AGDEBUGGER_STRICT_CONCEPT_REPAIR_ONLY:-1}"
KILL_STALE_BACKEND="${AGDEBUGGER_KILL_STALE_BACKEND:-1}"
DEFAULT_COMPONENT_ID="${DEFAULT_COMPONENT_ID:-0}"
SERVER_LOG="${SERVER_LOG:-${RUN_ARTIFACT_DIR}/server.log}"
RUN_LOG="${RUN_LOG:-${RUN_ARTIFACT_DIR}/run.jsonl}"
ANALYSIS_DETAIL_LOG="${ANALYSIS_DETAIL_LOG:-${RUN_ARTIFACT_DIR}/analysis_detail.jsonl}"
CLAIM_USE_WEBSEARCH="${AGDEBUGGER_CLAIM_USE_WEBSEARCH:-1}"
CLAIM_SEARCH_MAX_SEARCHES="${AGDEBUGGER_CLAIM_SEARCH_MAX_SEARCHES:-3}"
CLAIM_SEARCH_NUM_RESULTS="${AGDEBUGGER_CLAIM_SEARCH_NUM_RESULTS:-5}"
CLAIM_SEARCH_FETCH_TOP_N="${AGDEBUGGER_CLAIM_SEARCH_FETCH_TOP_N:-2}"
CLAIM_SEARCH_MAX_OUTPUT_WORDS="${AGDEBUGGER_CLAIM_SEARCH_MAX_OUTPUT_WORDS:-1500}"
ANALYSIS_CLAIM_CONCURRENCY="${AGDEBUGGER_ANALYSIS_CLAIM_CONCURRENCY:-2}"
# Cap on concurrent literature_fetch calls (sciverse + local mineru). 1 is
# safest for shared machines; bump if you have plenty of CPU + RAM headroom.
LITERATURE_FETCH_CONCURRENCY="${AGDEBUGGER_LITERATURE_FETCH_CONCURRENCY:-1}"

# ---------------------------------------------------------------------------
# Model config
# Edit these five variables only for normal use.
# ---------------------------------------------------------------------------
export AGENTDEBUG_MODEL_NAME="${AGENTDEBUG_MODEL_NAME:-intern-s1}"
export AGENTDEBUG_MODEL_AGENT="${AGENTDEBUG_MODEL_AGENT:-${AGENTDEBUG_MODEL_NAME}}"
export AGENTDEBUG_MODEL_MCP="${AGENTDEBUG_MODEL_MCP:-${AGENTDEBUG_MODEL_NAME}}"
export MODEL_PLANNER="${MODEL_PLANNER:-intern-s1-pro}"
export MODEL_CLAIM="${MODEL_CLAIM:-intern-s1}"
# Set to false to disable provider reasoning/thinking mode for faster responses.
# Accepted values: true | false | auto
export AGENTDEBUG_THINKING_MODE="${AGENTDEBUG_THINKING_MODE:-false}"

# Non-Intern API (set via env or shell before launching).
export AGENTDEBUG_NON_INTERN_API_KEY="${AGENTDEBUG_NON_INTERN_API_KEY:-}"
export AGENTDEBUG_NON_INTERN_BASE_URL="${AGENTDEBUG_NON_INTERN_BASE_URL:-}"

# Intern API
export AGENTDEBUG_INTERN_API_KEY="${AGENTDEBUG_INTERN_API_KEY:-}"
export AGENTDEBUG_INTERN_BASE_URL="${AGENTDEBUG_INTERN_BASE_URL:-https://chat.intern-ai.org.cn/api/v1/}"

# Bright Data config for external-agent claim websearch.
# Keep this aligned with run_with_models_debug.sh so both launchers use the
# same evidence backend settings by default.
export BRIGHT_DATA_API_KEY="${BRIGHT_DATA_API_KEY:-}"
export BRIGHT_DATA_ZONE="${BRIGHT_DATA_ZONE:-}"

export TOOLUNIVERSE_DIR="${TOOLUNIVERSE_DIR:-}"
TOOLUNIVERSE_MODE="${AGDEBUGGER_TOOLUNIVERSE_MODE:-shared_http}"
TOOLUNIVERSE_HOST="${AGDEBUGGER_TOOLUNIVERSE_HOST:-127.0.0.1}"
TOOLUNIVERSE_PORT="${AGDEBUGGER_TOOLUNIVERSE_PORT:-7000}"
TOOLUNIVERSE_URL="${AGDEBUGGER_TOOLUNIVERSE_URL:-http://${TOOLUNIVERSE_HOST}:${TOOLUNIVERSE_PORT}/mcp}"
TOOLUNIVERSE_READY_TIMEOUT="${AGDEBUGGER_TOOLUNIVERSE_READY_TIMEOUT:-240}"
TOOLUNIVERSE_MAX_WORKERS="${AGDEBUGGER_TOOLUNIVERSE_MAX_WORKERS:-16}"
TOOLUNIVERSE_LOG="${AGDEBUGGER_TOOLUNIVERSE_LOG:-${LOG_DIR}/tooluniverse_shared.log}"
TOOLUNIVERSE_PID_FILE="${AGDEBUGGER_TOOLUNIVERSE_PID_FILE:-${LOG_DIR}/tooluniverse_shared.pid}"
TOOLUNIVERSE_LOCK_DIR="${AGDEBUGGER_TOOLUNIVERSE_LOCK_DIR:-${LOG_DIR}/tooluniverse_shared.lock}"
TOOLUNIVERSE_SHARED_CACHE_DIR="${AGDEBUGGER_TOOLUNIVERSE_SHARED_CACHE_DIR:-${LOG_DIR}/tooluniverse_shared_cache}"
TOOLUNIVERSE_LOCAL_CACHE_DIR="${AGDEBUGGER_TOOLUNIVERSE_LOCAL_CACHE_DIR:-${RUN_ARTIFACT_DIR}/tooluniverse_cache}"

is_intern_model() {
    case "${1:-}" in
        intern-s1-pro|intern-s1|intern-s1-mini) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_base_url() {
    local model="${1:-}"
    if is_intern_model "${model}"; then
        printf '%s\n' "${AGENTDEBUG_INTERN_BASE_URL}"
    else
        printf '%s\n' "${AGENTDEBUG_NON_INTERN_BASE_URL}"
    fi
}

resolve_api_key() {
    local model="${1:-}"
    if is_intern_model "${model}"; then
        printf '%s\n' "${AGENTDEBUG_INTERN_API_KEY}"
    else
        printf '%s\n' "${AGENTDEBUG_NON_INTERN_API_KEY}"
    fi
}

export AGENTDEBUG_OPENAI_API_KEY="$(resolve_api_key "${AGENTDEBUG_MODEL_NAME}")"
export AGENTDEBUG_OPENAI_BASE_URL="$(resolve_base_url "${AGENTDEBUG_MODEL_NAME}")"
export AGENTDEBUG_OPENAI_API_KEY_AGENT="$(resolve_api_key "${AGENTDEBUG_MODEL_AGENT}")"
export AGENTDEBUG_OPENAI_BASE_URL_AGENT="$(resolve_base_url "${AGENTDEBUG_MODEL_AGENT}")"
export AGENTDEBUG_OPENAI_API_KEY_MCP="$(resolve_api_key "${AGENTDEBUG_MODEL_MCP}")"
export AGENTDEBUG_OPENAI_BASE_URL_MCP="$(resolve_base_url "${AGENTDEBUG_MODEL_MCP}")"
export AGENTDEBUG_OPENAI_API_KEY_PLANNER="$(resolve_api_key "${MODEL_PLANNER}")"
export AGENTDEBUG_OPENAI_BASE_URL_PLANNER="$(resolve_base_url "${MODEL_PLANNER}")"
export AGENTDEBUG_OPENAI_API_KEY_CLAIM="$(resolve_api_key "${MODEL_CLAIM}")"
export AGENTDEBUG_OPENAI_BASE_URL_CLAIM="$(resolve_base_url "${MODEL_CLAIM}")"
export AGENTDEBUG_OPENAI_API_KEY_EMBEDDING="${AGENTDEBUG_OPENAI_API_KEY_EMBEDDING:-${AGENTDEBUG_NON_INTERN_API_KEY}}"
export AGENTDEBUG_OPENAI_BASE_URL_EMBEDDING="${AGENTDEBUG_OPENAI_BASE_URL_EMBEDDING:-${AGENTDEBUG_NON_INTERN_BASE_URL}}"

export AGDEBUGGER_BACKEND_SERVE_UI=FALSE
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="${SCRIPT_DIR}:${SRC_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${no_proxy}"
export AGENTDEBUG_MODEL_TIMEOUT_SEC="${AGENTDEBUG_MODEL_TIMEOUT_SEC:-600}"
export AGENTDEBUG_MODEL_MAX_RETRIES="${AGENTDEBUG_MODEL_MAX_RETRIES:-2}"
export AGDEBUGGER_ANALYSIS_TIMEOUT_SEC="${ANALYSIS_TIMEOUT}"
export AGDEBUGGER_ANALYSIS_DETAIL_LOG="${ANALYSIS_DETAIL_LOG}"
export AGDEBUGGER_ANALYSIS_CLAIM_CONCURRENCY="${ANALYSIS_CLAIM_CONCURRENCY}"
export AGDEBUGGER_LITERATURE_FETCH_CONCURRENCY="${LITERATURE_FETCH_CONCURRENCY}"
export AGDEBUGGER_TOOLUNIVERSE_MODE="${TOOLUNIVERSE_MODE}"
export AGDEBUGGER_TOOLUNIVERSE_URL="${TOOLUNIVERSE_URL}"
export TOOLUNIVERSE_CACHE_DIR="${TOOLUNIVERSE_LOCAL_CACHE_DIR}"
export UV_CACHE_DIR
export XDG_CACHE_HOME

# Keep outer round timeouts longer than the inner model timeout, otherwise
# the dataset runner can kill a still-valid long LLM call prematurely.
MIN_ROUND_TIMEOUT="$(( ${AGENTDEBUG_MODEL_TIMEOUT_SEC%.*} + 30 ))"
if (( QUESTION_TIMEOUT < MIN_ROUND_TIMEOUT )); then
    QUESTION_TIMEOUT="${MIN_ROUND_TIMEOUT}"
fi
if (( DEBUG_STEP_TIMEOUT < MIN_ROUND_TIMEOUT )); then
    DEBUG_STEP_TIMEOUT="${MIN_ROUND_TIMEOUT}"
fi

activate_conda() {
    if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
        echo "[launcher] Activated conda env: ${CONDA_ENV} (python=$(which python))"
    else
        echo "[launcher] WARNING: conda.sh not found at ${CONDA_BASE}, using current python."
    fi
}

has_component_id() {
    local arg
    for arg in "$@"; do
        [[ "${arg}" == "--component-id" || "${arg}" == --component-id=* ]] && return 0
    done
    return 1
}

API_BASE="http://${HOST}:${PORT}/api"
BACKEND_PID=""
BACKEND_PID_FILE="${AGDEBUGGER_BACKEND_PID_FILE:-${LOG_DIR}/backend_${HOST}_${PORT}.pid}"

pid_matches_backend() {
    local pid="${1:-}"
    local cmdline=""
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    cmdline="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [[ "${cmdline}" == *"python -m agdebugger.cli ${MODULE}"* ]] || return 1
    [[ "${cmdline}" == *"--host ${HOST}"* ]] || return 1
    [[ "${cmdline}" == *"--port ${PORT}"* ]] || return 1
    return 0
}

stop_backend_pid_if_matches() {
    local pid="${1:-}"
    pid_matches_backend "${pid}" || return 1
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    return 0
}

cleanup() {
    local recorded_pid=""
    if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo ""
        echo "[launcher] Stopping AGDebugger backend (PID ${BACKEND_PID}) ..."
        stop_backend_pid_if_matches "${BACKEND_PID}" || true
        echo "[launcher] Backend stopped."
    fi
    if [[ -f "${BACKEND_PID_FILE}" ]]; then
        recorded_pid="$(cat "${BACKEND_PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${BACKEND_PID}" && "${recorded_pid}" == "${BACKEND_PID}" ]]; then
            rm -f "${BACKEND_PID_FILE}"
        elif ! pid_matches_backend "${recorded_pid}"; then
            rm -f "${BACKEND_PID_FILE}"
        fi
    fi
}
trap cleanup EXIT INT TERM

wait_ready() {
    local deadline=$((SECONDS + READY_TIMEOUT))
    echo "[launcher] Waiting for backend at ${API_BASE} (timeout ${READY_TIMEOUT}s) ..."
    while (( SECONDS < deadline )); do
        if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
            echo "[launcher] ERROR: backend exited unexpectedly."
            tail -30 "${SERVER_LOG}" 2>/dev/null || true
            return 1
        fi
        if curl -sf "${API_BASE}/agents" >/dev/null 2>&1; then
            echo "[launcher] Backend is ready."
            return 0
        fi
        sleep 1
    done
    echo "[launcher] ERROR: backend did not become ready within ${READY_TIMEOUT}s."
    tail -30 "${SERVER_LOG}" 2>/dev/null || true
    return 1
}

tooluniverse_ready() {
    TOOLUNIVERSE_URL="${TOOLUNIVERSE_URL}" NO_PROXY="${NO_PROXY}" no_proxy="${no_proxy}" python - <<'PY' >/dev/null 2>&1
import asyncio
import os

from autogen_ext.tools.mcp import StreamableHttpServerParams
from autogen_ext.tools.mcp._session import create_mcp_server_session


async def main() -> None:
    params = StreamableHttpServerParams(
        url=os.environ["TOOLUNIVERSE_URL"],
        timeout=10.0,
        sse_read_timeout=30.0,
        terminate_on_close=False,
    )
    async with create_mcp_server_session(params) as session:
        await session.initialize()
        await session.list_tools()


asyncio.run(main())
PY
}

tooluniverse_pid_running() {
    local pid="${1:-}"
    local cmdline=""
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    cmdline="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [[ "${cmdline}" == *"tooluniverse-smcp"* ]] || return 1
    [[ "${cmdline}" == *"--port ${TOOLUNIVERSE_PORT}"* ]] || return 1
    return 0
}

wait_for_tooluniverse() {
    local deadline=$((SECONDS + TOOLUNIVERSE_READY_TIMEOUT))
    local starter_pid="${1:-}"
    echo "[launcher] Waiting for ToolUniverse MCP at ${TOOLUNIVERSE_URL} (timeout ${TOOLUNIVERSE_READY_TIMEOUT}s) ..."
    while (( SECONDS < deadline )); do
        if tooluniverse_ready; then
            echo "[launcher] ToolUniverse MCP is ready."
            return 0
        fi
        if [[ -n "${starter_pid}" ]] && ! kill -0 "${starter_pid}" 2>/dev/null; then
            echo "[launcher] ERROR: ToolUniverse MCP exited unexpectedly."
            tail -30 "${TOOLUNIVERSE_LOG}" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done
    echo "[launcher] ERROR: ToolUniverse MCP did not become ready within ${TOOLUNIVERSE_READY_TIMEOUT}s."
    tail -30 "${TOOLUNIVERSE_LOG}" 2>/dev/null || true
    return 1
}

ensure_tooluniverse_ready() {
    local started_pid=""
    local recorded_pid=""

    if [[ "${TOOLUNIVERSE_MODE}" != "shared_http" ]]; then
        return 0
    fi

    mkdir -p "${TOOLUNIVERSE_SHARED_CACHE_DIR}" "${TOOLUNIVERSE_LOCAL_CACHE_DIR}"

    if tooluniverse_ready; then
        echo "[launcher] Reusing ToolUniverse MCP: ${TOOLUNIVERSE_URL}"
        return 0
    fi

    while ! mkdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null; do
        if tooluniverse_ready; then
            echo "[launcher] Reusing ToolUniverse MCP: ${TOOLUNIVERSE_URL}"
            return 0
        fi
        sleep 1
    done
    if [[ -f "${TOOLUNIVERSE_PID_FILE}" ]]; then
        recorded_pid="$(cat "${TOOLUNIVERSE_PID_FILE}" 2>/dev/null || true)"
        if ! tooluniverse_pid_running "${recorded_pid}"; then
            rm -f "${TOOLUNIVERSE_PID_FILE}"
            recorded_pid=""
        fi
    fi

    if tooluniverse_ready; then
        rmdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || true
        echo "[launcher] Reusing ToolUniverse MCP: ${TOOLUNIVERSE_URL}"
        return 0
    fi

    if [[ -n "${recorded_pid}" ]]; then
        echo "[launcher] ToolUniverse MCP is already starting (PID ${recorded_pid}) ..."
        rmdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || true
        wait_for_tooluniverse "${recorded_pid}"
        return $?
    fi

    echo "[launcher] Starting shared ToolUniverse MCP server ..."
    TOOLUNIVERSE_CACHE_DIR="${TOOLUNIVERSE_SHARED_CACHE_DIR}" \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    uv --directory "${TOOLUNIVERSE_DIR}" run tooluniverse-smcp \
        --transport http \
        --host "${TOOLUNIVERSE_HOST}" \
        --port "${TOOLUNIVERSE_PORT}" \
        --exclude-tool-types PackageTool \
        --compact-mode \
        --max-workers "${TOOLUNIVERSE_MAX_WORKERS}" \
        >> "${TOOLUNIVERSE_LOG}" 2>&1 &
    started_pid=$!
    printf '%s\n' "${started_pid}" > "${TOOLUNIVERSE_PID_FILE}"
    echo "[launcher] ToolUniverse MCP PID: ${started_pid}"
    rmdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || true
    wait_for_tooluniverse "${started_pid}"
}

main() {
    activate_conda
    mkdir -p "${RUN_ARTIFACT_DIR}"
    mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TOOLUNIVERSE_LOCAL_CACHE_DIR}"

    if [[ "${KILL_STALE_BACKEND}" == "1" ]]; then
        if [[ -f "${BACKEND_PID_FILE}" ]]; then
            local stale_pid=""
            stale_pid="$(cat "${BACKEND_PID_FILE}" 2>/dev/null || true)"
            if stop_backend_pid_if_matches "${stale_pid}"; then
                echo "[launcher] Stopped stale backend from pid file: ${stale_pid}"
            else
                rm -f "${BACKEND_PID_FILE}"
            fi
        fi
        sleep 1
    fi

    local -a forwarded_args=("$@")
    if ! has_component_id "${forwarded_args[@]}"; then
        forwarded_args=(--component-id "${DEFAULT_COMPONENT_ID}" "${forwarded_args[@]}")
    fi

    echo "============================================================"
    echo "[launcher] AGDebugger Unified Launcher"
    echo "============================================================"
    echo "[launcher] Backend: ${API_BASE}"
    echo "[launcher] Module:  ${MODULE}"
    echo "[launcher] Main model: ${AGENTDEBUG_MODEL_NAME} @ ${AGENTDEBUG_OPENAI_BASE_URL}"
    echo "[launcher] Agent: ${AGENTDEBUG_MODEL_AGENT} @ ${AGENTDEBUG_OPENAI_BASE_URL_AGENT}"
    echo "[launcher] MCP:   ${AGENTDEBUG_MODEL_MCP} @ ${AGENTDEBUG_OPENAI_BASE_URL_MCP}"
    echo "[launcher] ToolUniverse mode: ${TOOLUNIVERSE_MODE}"
    echo "[launcher] ToolUniverse URL:  ${TOOLUNIVERSE_URL}"
    echo "[launcher] Plan:  ${MODEL_PLANNER} @ ${AGENTDEBUG_OPENAI_BASE_URL_PLANNER}"
    echo "[launcher] Claim: ${MODEL_CLAIM} @ ${AGENTDEBUG_OPENAI_BASE_URL_CLAIM}"
    echo "[launcher] Embed: text-embedding-3-small @ ${AGENTDEBUG_OPENAI_BASE_URL_EMBEDDING}"
    echo "[launcher] Claim websearch: ${CLAIM_USE_WEBSEARCH}"
    echo "[launcher] literature_fetch concurrency: ${LITERATURE_FETCH_CONCURRENCY}"
    echo "[launcher] thinking_mode: ${AGENTDEBUG_THINKING_MODE}"
    echo "[launcher] Run log: ${RUN_LOG}"
    echo "[launcher] Analysis detail log: ${ANALYSIS_DETAIL_LOG}"
    echo "[launcher] Run dir: ${RUN_ARTIFACT_DIR}"
    echo "[launcher] Timeouts: ready=${READY_TIMEOUT}s reset=${RESET_TIMEOUT}s question=${QUESTION_TIMEOUT}s stall=${QUESTION_STALL_TIMEOUT}s debug_step=${DEBUG_STEP_TIMEOUT}s analysis=${AGDEBUGGER_ANALYSIS_TIMEOUT_SEC}s model=${AGENTDEBUG_MODEL_TIMEOUT_SEC}s"
    echo "[launcher] Repair mode: strict=${STRICT_CONCEPT_REPAIR_ONLY} max_concept_repair_attempts=${MAX_CONCEPT_REPAIR_ATTEMPTS}"
    echo "[launcher] Extra args: ${forwarded_args[*]}"
    echo ""

    ensure_tooluniverse_ready

    echo "[launcher] Starting AGDebugger backend ..."
    python -m agdebugger.cli "${MODULE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        >> "${SERVER_LOG}" 2>&1 &
    BACKEND_PID=$!
    printf '%s\n' "${BACKEND_PID}" > "${BACKEND_PID_FILE}"
    echo "[launcher] Backend PID: ${BACKEND_PID}"

    wait_ready

    python "${SCRIPT_DIR}/run_dataset_autodebug.py" \
        --reuse-server \
        --host "${HOST}" \
        --port "${PORT}" \
        --server-log "${SERVER_LOG}" \
        --run-log "${RUN_LOG}" \
        --analysis-detail-log "${ANALYSIS_DETAIL_LOG}" \
        --ready-timeout "${READY_TIMEOUT}" \
        --run-timeout "${RUN_TIMEOUT}" \
        --reset-timeout "${RESET_TIMEOUT}" \
      --question-timeout "${QUESTION_TIMEOUT}" \
      --question-stall-timeout "${QUESTION_STALL_TIMEOUT}" \
      --question-retry-attempts "${QUESTION_RETRY_ATTEMPTS}" \
        --debug-step-timeout "${DEBUG_STEP_TIMEOUT}" \
        --max-concept-repair-attempts "${MAX_CONCEPT_REPAIR_ATTEMPTS}" \
        --model "${AGENTDEBUG_MODEL_NAME}" \
        --model-planner "${MODEL_PLANNER}" \
        --model-claim "${MODEL_CLAIM}" \
        --api-key "${AGENTDEBUG_OPENAI_API_KEY}" \
        --api-base "${AGENTDEBUG_OPENAI_BASE_URL}" \
        --claim-search-max-searches "${CLAIM_SEARCH_MAX_SEARCHES}" \
        --claim-search-num-results "${CLAIM_SEARCH_NUM_RESULTS}" \
        --claim-search-fetch-top-n "${CLAIM_SEARCH_FETCH_TOP_N}" \
        --claim-search-max-output-words "${CLAIM_SEARCH_MAX_OUTPUT_WORDS}" \
        $([[ "${CLAIM_USE_WEBSEARCH}" == "1" ]] && printf '%s' "--claim-use-websearch") \
        $([[ "${STRICT_CONCEPT_REPAIR_ONLY}" == "1" ]] && printf '%s' "--strict-concept-repair-only") \
        "${forwarded_args[@]}"
}

main "$@"
