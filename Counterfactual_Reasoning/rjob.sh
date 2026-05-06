#!/usr/bin/env bash
# Full-benchmark driver. Runs `run_parallel.sh` once over the complete
# Protein dataset (component_id=0 loads all 703 examples from
# Protein.jsonl), then archives per-worker stdout logs and aggregates
# initial/debug/final correctness across all workers.
#
# Usage:
#   bash rjob.sh                           # full bench, RJOB_WORKERS workers
#   RJOB_WORKERS=8 bash rjob.sh            # override worker count
#   RJOB_TOTAL=200 bash rjob.sh            # run only first 200 examples
#   RJOB_COMPONENT_ID=1 RJOB_TOTAL=118 bash rjob.sh   # single component
#   RJOB_START=300 RJOB_TOTAL=200 bash rjob.sh        # slice [300, 500)
#
# NOTE: do NOT add the non-intern LLM/embedding endpoint host to no_proxy.
# In some clusters that endpoint is a public IP only reachable via an
# upstream proxy; bypassing the proxy causes every embedding call to time
# out after 30 s and kills the websearch semantic filter (returns 0 bytes),
# starving the judge of evidence and inflating `no_repairable_concepts` halts.
#
# OPTIONAL — Path C Step 2 (LLM rewrite) knobs (default OFF):
#   AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE   off|cross_claim_only|on_fused|always
#   AGDEBUGGER_REWRITE_TIMEOUT_SEC         per-span LLM timeout (default 30)
#   AGDEBUGGER_MULTI_ANCHOR_CROSS_CLAIM_FUSE  1 to fuse across claim_ids
# Enable explicitly, e.g.:
#   AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE=on_fused bash rjob.sh
#
# Intern LLM rate limiting (applies to judge + planner + rewriter calls;
# ToolUniverse MCP calls are NOT governed here — set ToolUniverse's own
# max_workers for those):
#   AGDEBUGGER_LLM_RATE_LIMITER_DISABLED  1 to skip client-side queueing
#                                         entirely and rely ONLY on retry
#                                         after 429/`请求过于频繁`. Recommended
#                                         when analysis pipelines time out
#                                         while queued (observed on full bench:
#                                         median analysis elapsed 120s vs 90s
#                                         default timeout because of per-call
#                                         queue waits).
#   AGDEBUGGER_LLM_RPM            per-worker request budget / minute (default 100)
#   AGDEBUGGER_LLM_TPM            per-worker token budget / minute   (default 50000)
#   AGDEBUGGER_LLM_RETRY_MAX_ATTEMPTS  retries on 429/`请求过于频繁` (default 6)
#   AGDEBUGGER_LLM_RETRY_BASE_DELAY_SEC  initial backoff (default 2)
#   AGDEBUGGER_LLM_RETRY_MAX_DELAY_SEC   cap for exponential backoff (default 120)
#
# IMPORTANT: the limiter is PROCESS-LOCAL, so when you run N workers against a
# shared account you MUST divide the account budget by N. Example for the
# 100 RPM / 50000 TPM Intern account with 4 workers:
#   AGDEBUGGER_LLM_RPM=25 AGDEBUGGER_LLM_TPM=12500 bash rjob.sh
#
# Recommended full-bench recipe (disable client-side queue, retry 3×):
#   AGDEBUGGER_LLM_RATE_LIMITER_DISABLED=1 \
#   AGDEBUGGER_LLM_RETRY_MAX_ATTEMPTS=3 \
#   AGDEBUGGER_DROP_WRONG_PREFIX=1 \
#   AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE=always \
#   AGDEBUGGER_MULTI_ANCHOR_CROSS_CLAIM_FUSE=1 \
#   AGDEBUGGER_JUDGE_USE_LITERATURE=on \
#   bash rjob.sh
#
# Judge-side literature metadata (Fix #1): when
# ``AGDEBUGGER_JUDGE_USE_LITERATURE=on``, WebSearchEvidenceProvider calls
# sciverse_tools.literature_search per-claim alongside web_search. Lets the
# judge see paper titles/authors/DOIs/abstract snippets — helps when the
# primary agent's literature_fetch failed and plain web snippets are too
# shallow to verify a claim. Tuning:
#   AGDEBUGGER_JUDGE_USE_LITERATURE           off (default) | on
#   AGDEBUGGER_JUDGE_LITERATURE_NUM_RESULTS   candidate papers per claim (1-8, default 3)
#   AGDEBUGGER_JUDGE_LITERATURE_TIMEOUT_SEC   per-claim timeout (default 30)
#
# Path A — high-value-claim full-text fetch (2026-04-22):
# on top of the metadata branch above, when
# ``AGDEBUGGER_JUDGE_USE_LITERATURE_FETCH=on``, the judge ALSO calls
# ``sciverse_tools.sciverse_fetch_markdown`` to pull paper markdown bodies
# for high-value claim categories (default ``mapping_claim,constraint_claim``).
# Use this when the 100-sample stuck-case study shows the judge has metadata
# but still can't distinguish method-level options (e.g. 120kV vs 300kV
# cryo-EM, 3D vs 2D tomography). This path is slower (10-60s per call), so
# it is cached per-query and gated by claim category.
#   AGDEBUGGER_JUDGE_USE_LITERATURE_FETCH           off (default) | on
#   AGDEBUGGER_JUDGE_LITERATURE_FETCH_CATEGORIES    CSV (default mapping_claim,constraint_claim)
#   AGDEBUGGER_JUDGE_LITERATURE_FETCH_NUM_RESULTS   papers to fetch (1-5, default 2)
#   AGDEBUGGER_JUDGE_LITERATURE_FETCH_MAX_CHARS     per-paper body cap (default 8000)
#   AGDEBUGGER_JUDGE_LITERATURE_FETCH_TIMEOUT_SEC   per-claim timeout (default 90)
# Requires SCIVERSE_API_TOKEN to be set for the sciverse REST API.
#
# Judge evidence cache (in-process LRU, shared across claims + debug steps
# within the same worker). Two caches: web_search results and
# literature_search results, each keyed by the normalized query + relevant
# config. Size is total entry count per cache (not bytes); 0 disables.
#   AGDEBUGGER_JUDGE_EVIDENCE_CACHE_SIZE     LRU capacity per cache (default 1000, 0 = off)
#
# Claim-retry budget (Fix #2): the full-bench case study observed a runaway
# where a single question tried 11 distinct claim_ids back-to-back. Both the
# existing concept-repair budget default and a new hard outer cap have been
# raised/added:
#   --max-concept-repair-attempts DEFAULT  3 → 7
#   --max-edit-attempts           DEFAULT  3 → 7
#   AGDEBUGGER_MAX_FAILED_CLAIM_IDS        hard outer cap (default 7)

set -uo pipefail   # intentionally NOT -e: aggregation must still run even if the sweep errors

cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -n "${CONDA_BASE:-}" && -n "${AGDEBUGGER_CONDA_ENV:-}" ]]; then
    source "${CONDA_BASE}/bin/activate" "${AGDEBUGGER_CONDA_ENV}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Tunables (env-overridable)
# ----------------------------------------------------------------------------
# component_id=0 → load ALL examples from Protein.jsonl (the full bench, 703
# questions). Set a positive id to restrict to a single component.
COMPONENT_ID="${RJOB_COMPONENT_ID:-0}"

# Total number of examples to include in this run.  If larger than the actual
# dataset slice the runner auto-clamps.  Default 703 == full Protein bench.
TOTAL_SAMPLES="${RJOB_TOTAL:-703}"

# Where to start inside the dataset.  Useful for resuming a prior crash.
START="${RJOB_START:-0}"

# Concurrent workers.  Each worker owns its own AGDebugger backend on a
# distinct port.  They share the single ToolUniverse MCP (max_workers=16
# inside that server), so 4–8 is a sensible upper bound here.
WORKERS="${RJOB_WORKERS:-4}"

# ----------------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${SCRIPT_DIR}/logs/full_bench_${STAMP}"
mkdir -p "${ARCHIVE_DIR}"

echo "============================================================"
echo "[rjob] Full-benchmark sweep"
echo "[rjob] stamp:         ${STAMP}"
echo "[rjob] archive:       ${ARCHIVE_DIR}"
echo "[rjob] component_id:  ${COMPONENT_ID}   (0 = all components)"
echo "[rjob] total_samples: ${TOTAL_SAMPLES}"
echo "[rjob] start offset:  ${START}"
echo "[rjob] workers:       ${WORKERS}"
echo "============================================================"

# Track which run_* directories belong to *this* sweep so aggregation
# doesn't pick up historical runs. The per-worker dir name includes
# WORKER_STAMP=<date>_w<i>, which is generated fresh inside run_parallel.sh;
# the START_TS below brackets everything created during this call.
START_TS="$(date +%s)"

(
    bash "${SCRIPT_DIR}/run_parallel.sh" \
        --workers "${WORKERS}" \
        --total-examples "${TOTAL_SAMPLES}" \
        --start "${START}" \
        -- \
        --component-id "${COMPONENT_ID}"
) || echo "[rjob] run_parallel.sh exited non-zero; continuing to archive+aggregate."

END_TS="$(date +%s)"

# Archive per-worker stdout (the next rjob run would overwrite them).
if ls "${SCRIPT_DIR}/logs"/parallel_worker_*.log >/dev/null 2>&1; then
    cp -f "${SCRIPT_DIR}/logs"/parallel_worker_*.log "${ARCHIVE_DIR}/"
fi

# ----------------------------------------------------------------------------
# Aggregate across all worker run.jsonl files created during this sweep
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[rjob] Aggregating results"
echo "============================================================"

SUMMARY_FILE="${ARCHIVE_DIR}/summary.txt"

RJOB_ARCHIVE="${ARCHIVE_DIR}" \
RJOB_START_TS="${START_TS}" \
RJOB_END_TS="${END_TS}" \
RJOB_LOGS_ROOT="${SCRIPT_DIR}/logs" \
python3 - <<'PY' | tee "${SUMMARY_FILE}"
import json, os, re
from collections import Counter
from pathlib import Path

archive = Path(os.environ["RJOB_ARCHIVE"])
logs_root = Path(os.environ["RJOB_LOGS_ROOT"])
start_ts = int(os.environ["RJOB_START_TS"])
end_ts = int(os.environ["RJOB_END_TS"]) + 60  # small grace window

# Locate every run.jsonl touched during this sweep.
matches = []
for run_dir in logs_root.rglob("run_*_w*"):
    jsonl = run_dir / "run.jsonl"
    if not jsonl.exists():
        continue
    mtime = jsonl.stat().st_mtime
    if start_ts <= mtime <= end_ts:
        matches.append(run_dir)
matches.sort()

if not matches:
    print("(no per-worker run.jsonl files found in this time window; check run_parallel output)")
    raise SystemExit(0)

print(f"Discovered {len(matches)} worker run dirs:")
for p in matches:
    print(f"  {p}")
print()

total = correct_initial = correct_final = debug_tried = debug_real_fixes = 0
halt_counter = Counter()
per_worker_rows = []

for run_dir in matches:
    jsonl = run_dir / "run.jsonl"
    q_map = {}
    d_map = {}
    halts = []
    with jsonl.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ev = rec.get("event")
            if ev == "question_result":
                q_map[rec.get("index")] = rec
            elif ev == "debug_result":
                d_map[rec.get("index")] = rec
            elif ev == "debug_halt":
                halts.append(rec)

    w_total = len(q_map)
    w_init = sum(1 for r in q_map.values() if r.get("is_correct"))
    w_tried = len(d_map)
    w_fix = sum(1 for r in d_map.values() if r.get("fixed"))
    w_final = 0
    for idx, qr in q_map.items():
        if idx in d_map:
            if d_map[idx].get("fixed"):
                w_final += 1
        elif qr.get("is_correct"):
            w_final += 1

    for h in halts:
        reason = h.get("reason") or "?"
        halt_counter[reason] += 1
    for dr in d_map.values():
        hr = dr.get("halt_reason")
        if hr:
            halt_counter[hr] += 1

    per_worker_rows.append((run_dir.name, w_total, w_init, w_tried, w_fix, w_final))
    total += w_total
    correct_initial += w_init
    correct_final += w_final
    debug_tried += w_tried
    debug_real_fixes += w_fix

print(f"{'Worker':<36} | {'Q':>4} | {'Init':>4} | {'DbgTry':>6} | {'Fix':>3} | {'Final':>5}")
print("-" * 76)
for name, q, i, t, fx, fn in per_worker_rows:
    print(f"{name:<36} | {q:>4} | {i:>4} | {t:>6} | {fx:>3} | {fn:>5}")
print("-" * 76)
print(f"{'TOTAL':<36} | {total:>4} | {correct_initial:>4} | {debug_tried:>6} | {debug_real_fixes:>3} | {correct_final:>5}")
print()
if total:
    print(f"Total samples:                 {total}")
    print(f"Initial (no-debug) correct:    {correct_initial} / {total}  ({100*correct_initial/total:.1f}%)")
    print(f"Debug attempts (incorrect):    {debug_tried}")
    if debug_tried:
        print(f"Debug real fixes:              {debug_real_fixes} / {debug_tried}  ({100*debug_real_fixes/debug_tried:.1f}% of attempts)")
    print(f"Final correct (with debug):    {correct_final} / {total}  ({100*correct_final/total:.1f}%)")
    print(f"Net gain from debug:           {correct_final - correct_initial}")

if halt_counter:
    print()
    print("Halt / debug-result halt_reason distribution:")
    for reason, n in halt_counter.most_common():
        print(f"  {reason:<42} {n}")
PY

echo ""
echo "[rjob] Archive:        ${ARCHIVE_DIR}"
echo "[rjob] Summary file:   ${SUMMARY_FILE}"
echo "[rjob] Done."
