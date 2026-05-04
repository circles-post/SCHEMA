#!/usr/bin/env bash
# Parallel launcher: run N concurrent instances of run_with_models.sh
# on different port / data slices.
#
# Usage:
#   bash run_parallel.sh [--workers N] [--total-examples M] [--start S] [-- extra_args...]
#
# Example – 5 workers, 116 examples starting from 0:
#   bash run_parallel.sh --workers 5 --total-examples 116 -- --component-id 1
#
# Example – 2 workers, 2 examples starting from example 4:
#   bash run_parallel.sh --workers 2 --total-examples 2 --start 4 -- --component-id 1
#
# Each worker gets a unique PORT (base 8081 + worker index) and a
# non-overlapping --start / --limit slice of the dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
BASE_PORT="${AGDEBUGGER_BASE_PORT:-8081}"
CONDA_ENV="${AGDEBUGGER_CONDA_ENV:-agentdebug}"
CONDA_BASE="${CONDA_BASE:-/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3}"
UV_CACHE_DIR="${AGDEBUGGER_UV_CACHE_DIR:-${LOG_DIR}/.uv-cache}"
XDG_CACHE_HOME="${AGDEBUGGER_XDG_CACHE_HOME:-${LOG_DIR}/.cache}"
TOOLUNIVERSE_DIR="${TOOLUNIVERSE_DIR:-/mnt/shared-storage-user/fengxinshun/AISci/ToolUniverse/}"
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
CLEAN_STALE_PROCESSES="${AGDEBUGGER_CLEAN_STALE_PROCESSES:-1}"
export BRIGHT_DATA_API_KEY="${BRIGHT_DATA_API_KEY:-cf0ecaca-a28c-49f8-85df-d27e37cd86a8}"
export BRIGHT_DATA_ZONE="${BRIGHT_DATA_ZONE:-serp_api1}"
# Cap on concurrent literature_fetch (sciverse + local mineru) calls across
# all parallel workers. Defaults to 1 because mineru is CPU/IO heavy.
export AGDEBUGGER_LITERATURE_FETCH_CONCURRENCY="${AGDEBUGGER_LITERATURE_FETCH_CONCURRENCY:-1}"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${no_proxy}"
WORKERS=2
TOTAL_EXAMPLES=116
BASE_START=0
EXTRA_ARGS=()

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)          WORKERS="$2"; shift 2 ;;
        --total-examples)   TOTAL_EXAMPLES="$2"; shift 2 ;;
        --start)            BASE_START="$2"; shift 2 ;;
        --)                 shift; EXTRA_ARGS=("$@"); break ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Strip --start / --limit from EXTRA_ARGS to avoid conflicts
CLEAN_EXTRA_ARGS=()
skip_next=0
for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if (( skip_next )); then
        skip_next=0
        continue
    fi
    case "${arg}" in
        --start|--limit) skip_next=1; continue ;;
        --start=*|--limit=*) continue ;;
        *) CLEAN_EXTRA_ARGS+=("${arg}") ;;
    esac
done

PER_WORKER=$(( TOTAL_EXAMPLES / WORKERS ))
REMAINDER=$(( TOTAL_EXAMPLES % WORKERS ))

mkdir -p "${LOG_DIR}"
mkdir -p "${UV_CACHE_DIR}"
mkdir -p "${XDG_CACHE_HOME}"

activate_conda() {
    if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
        echo "[parallel] Activated conda env: ${CONDA_ENV} (python=$(which python))"
    else
        echo "[parallel] WARNING: conda.sh not found at ${CONDA_BASE}, using current python."
    fi
}

cleanup_current_tty_residual_processes() {
    local tty_path=""
    local tty_name=""
    local pids=()
    local pid=""
    local cmd=""

    if [[ "${CLEAN_STALE_PROCESSES}" != "1" ]]; then
        echo "[parallel] Skipping stale-process cleanup (AGDEBUGGER_CLEAN_STALE_PROCESSES=${CLEAN_STALE_PROCESSES})."
        return 0
    fi

    tty_path="$(tty 2>/dev/null || true)"
    if [[ -z "${tty_path}" || "${tty_path}" == "not a tty" ]]; then
        echo "[parallel] No controlling TTY detected; skipping stale-process cleanup."
        return 0
    fi
    tty_name="${tty_path#/dev/}"

    while IFS= read -r pid; do
        [[ -n "${pid}" ]] || continue
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        [[ "${pid}" == "$$" ]] && continue
        [[ "${pid}" == "${PPID}" ]] && continue

        cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
        [[ -n "${cmd}" ]] || continue

        if [[ "${cmd}" == *"${SCRIPT_DIR}/run_parallel.sh"* ]] || \
           [[ "${cmd}" == *"${SCRIPT_DIR}/run_with_models.sh"* ]] || \
           [[ "${cmd}" == *"python -m agdebugger.cli ${MODULE}"* ]]; then
            pids+=("${pid}")
        fi
    done < <(ps -t "${tty_name}" -o pid= 2>/dev/null || true)

    if (( ${#pids[@]} == 0 )); then
        echo "[parallel] No residual agdebugger processes found on tty ${tty_name}."
        return 0
    fi

    echo "[parallel] Cleaning residual agdebugger processes on tty ${tty_name}: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
    sleep 1

    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done
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
    echo "[parallel] Waiting for shared ToolUniverse MCP at ${TOOLUNIVERSE_URL} (timeout ${TOOLUNIVERSE_READY_TIMEOUT}s) ..."
    while (( SECONDS < deadline )); do
        if tooluniverse_ready; then
            echo "[parallel] Shared ToolUniverse MCP is ready."
            return 0
        fi
        if [[ -n "${starter_pid}" ]] && ! kill -0 "${starter_pid}" 2>/dev/null; then
            echo "[parallel] ERROR: shared ToolUniverse MCP exited unexpectedly."
            tail -30 "${TOOLUNIVERSE_LOG}" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done
    echo "[parallel] ERROR: shared ToolUniverse MCP did not become ready within ${TOOLUNIVERSE_READY_TIMEOUT}s."
    tail -30 "${TOOLUNIVERSE_LOG}" 2>/dev/null || true
    return 1
}

ensure_shared_tooluniverse() {
    local started_pid=""
    local recorded_pid=""
    local lock_mtime=""
    local now_ts=""

    if [[ "${TOOLUNIVERSE_MODE}" != "shared_http" ]]; then
        echo "[parallel] ToolUniverse mode=${TOOLUNIVERSE_MODE}; workers will manage MCP individually."
        return 0
    fi

    mkdir -p "${TOOLUNIVERSE_SHARED_CACHE_DIR}"

    if tooluniverse_ready; then
        echo "[parallel] Reusing shared ToolUniverse MCP: ${TOOLUNIVERSE_URL}"
        return 0
    fi

    if [[ -d "${TOOLUNIVERSE_LOCK_DIR}" && ! -f "${TOOLUNIVERSE_PID_FILE}" ]]; then
        lock_mtime="$(stat -c %Y "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || echo 0)"
        now_ts="$(date +%s)"
        if [[ "${lock_mtime}" =~ ^[0-9]+$ ]] && [[ "${now_ts}" =~ ^[0-9]+$ ]]; then
            if (( now_ts - lock_mtime > 30 )); then
                echo "[parallel] Removing stale ToolUniverse lock: ${TOOLUNIVERSE_LOCK_DIR}"
                rmdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || true
            fi
        fi
    fi

    while ! mkdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null; do
        if tooluniverse_ready; then
            echo "[parallel] Reusing shared ToolUniverse MCP: ${TOOLUNIVERSE_URL}"
            return 0
        fi
        if [[ ! -f "${TOOLUNIVERSE_PID_FILE}" ]]; then
            lock_mtime="$(stat -c %Y "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || echo 0)"
            now_ts="$(date +%s)"
            if [[ "${lock_mtime}" =~ ^[0-9]+$ ]] && [[ "${now_ts}" =~ ^[0-9]+$ ]]; then
                if (( now_ts - lock_mtime > 30 )); then
                    echo "[parallel] Clearing stale ToolUniverse lock while waiting: ${TOOLUNIVERSE_LOCK_DIR}"
                    rmdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || true
                    sleep 1
                    continue
                fi
            fi
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
        echo "[parallel] Reusing shared ToolUniverse MCP: ${TOOLUNIVERSE_URL}"
        return 0
    fi

    if [[ -n "${recorded_pid}" ]]; then
        echo "[parallel] Shared ToolUniverse MCP is already starting (PID ${recorded_pid}) ..."
        rmdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || true
        wait_for_tooluniverse "${recorded_pid}"
        return $?
    fi

    echo "[parallel] Starting shared ToolUniverse MCP server ..."
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
    echo "[parallel] Shared ToolUniverse MCP PID: ${started_pid}"
    rmdir "${TOOLUNIVERSE_LOCK_DIR}" 2>/dev/null || true
    wait_for_tooluniverse "${started_pid}"
}

echo "============================================================"
echo "[parallel] Launching ${WORKERS} workers over ${TOTAL_EXAMPLES} examples (base_start=${BASE_START})"
echo "[parallel] Base port: ${BASE_PORT}"
echo "[parallel] Shared ToolUniverse: mode=${TOOLUNIVERSE_MODE} url=${TOOLUNIVERSE_URL}"
echo "[parallel] Extra args: ${CLEAN_EXTRA_ARGS[*]:-}"
echo "============================================================"

PIDS=()
OFFSET=0
STAGGER_SEC="${AGDEBUGGER_STAGGER_SEC:-5}"

activate_conda
cleanup_current_tty_residual_processes
ensure_shared_tooluniverse

for (( i=0; i<WORKERS; i++ )); do
    LIMIT=${PER_WORKER}
    if (( i < REMAINDER )); then
        LIMIT=$(( LIMIT + 1 ))
    fi
    if (( LIMIT == 0 )); then
        break
    fi

    # Stagger worker launches to avoid simultaneous API bursts
    if (( i > 0 && STAGGER_SEC > 0 )); then
        echo "[parallel] Waiting ${STAGGER_SEC}s before launching worker ${i} ..."
        sleep "${STAGGER_SEC}"
    fi

    WORKER_START=$(( BASE_START + OFFSET ))
    WORKER_PORT=$(( BASE_PORT + i ))
    WORKER_STAMP="$(date +%Y%m%d_%H%M%S)_w${i}"

    echo "[parallel] Worker ${i}: port=${WORKER_PORT} start=${WORKER_START} limit=${LIMIT}"

    WORKER_TOOLUNIVERSE_LOCAL_CACHE_DIR="${LOG_DIR}/tooluniverse_worker_${i}_cache"

    mkdir -p "${WORKER_TOOLUNIVERSE_LOCAL_CACHE_DIR}"

    AGDEBUGGER_TOOLUNIVERSE_MODE="${TOOLUNIVERSE_MODE}" \
    AGDEBUGGER_TOOLUNIVERSE_HOST="${TOOLUNIVERSE_HOST}" \
    AGDEBUGGER_TOOLUNIVERSE_PORT="${TOOLUNIVERSE_PORT}" \
    AGDEBUGGER_TOOLUNIVERSE_URL="${TOOLUNIVERSE_URL}" \
    AGDEBUGGER_TOOLUNIVERSE_LOG="${TOOLUNIVERSE_LOG}" \
    AGDEBUGGER_TOOLUNIVERSE_PID_FILE="${TOOLUNIVERSE_PID_FILE}" \
    AGDEBUGGER_TOOLUNIVERSE_LOCK_DIR="${TOOLUNIVERSE_LOCK_DIR}" \
    AGDEBUGGER_TOOLUNIVERSE_SHARED_CACHE_DIR="${TOOLUNIVERSE_SHARED_CACHE_DIR}" \
    AGDEBUGGER_TOOLUNIVERSE_LOCAL_CACHE_DIR="${WORKER_TOOLUNIVERSE_LOCAL_CACHE_DIR}" \
    AGDEBUGGER_PORT="${WORKER_PORT}" \
    AGDEBUGGER_RUN_STAMP="${WORKER_STAMP}" \
    bash "${SCRIPT_DIR}/run_with_models.sh" \
        --start "${WORKER_START}" \
        --limit "${LIMIT}" \
        "${CLEAN_EXTRA_ARGS[@]}" \
        > "${SCRIPT_DIR}/logs/parallel_worker_${i}.log" 2>&1 &

    PIDS+=($!)
    OFFSET=$(( OFFSET + LIMIT ))
done

echo "[parallel] All ${#PIDS[@]} workers launched. PIDs: ${PIDS[*]}"
echo "[parallel] Logs: logs/parallel_worker_*.log"
echo "[parallel] Waiting for all workers to finish ..."

FAILED=0
for (( i=0; i<${#PIDS[@]}; i++ )); do
    if wait "${PIDS[$i]}"; then
        echo "[parallel] Worker ${i} (PID ${PIDS[$i]}) finished OK"
    else
        echo "[parallel] Worker ${i} (PID ${PIDS[$i]}) FAILED (exit $?)"
        FAILED=$(( FAILED + 1 ))
    fi
done

echo "============================================================"
echo "[parallel] Done. ${#PIDS[@]} workers, ${FAILED} failed."
echo "============================================================"

exit ${FAILED}
