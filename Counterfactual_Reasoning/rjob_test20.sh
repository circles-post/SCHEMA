#!/usr/bin/env bash
# Smoke test — 20 samples from component 1, 2 workers.
#
# Mirrors the earlier 20-sample baseline (comp-id=1, start=0, 2 workers ×
# 10 each) so the repair rate here is directly comparable to:
#   - 20-sample run before _infer_role fix: 5/11 real fixes (45%)
#   - 116-sample run before _infer_role fix: 3/50 real fixes (6%)
#
# Run:
#   bash rjob_test20.sh
#
# Outputs:
#   logs/full_bench_<stamp>/summary.txt     <-- Init/DbgTry/Fix/Final table
#   logs/full_bench_<stamp>/parallel_worker_*.log
#   logs/<yyyymmdd>/run_<stamp>_w{0,1}/run.jsonl

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RJOB_COMPONENT_ID=1
export RJOB_TOTAL=20
export RJOB_START=0
export RJOB_WORKERS=2

# ----------------------------------------------------------------------------
# Prefix cleanup (needed whenever the agent's original turn writes its
# choice at the very start, e.g. "**Option3** Wait ..."). Path C's multi-
# anchor anchors only cover the "wrong reasoning" span — the short bolded
# option commitment sits in the prefix and survives untouched otherwise,
# re-anchoring the rerun back to the original wrong answer.  Turning this
# on falls back to _CONCLUSION_MARKER_RE to cut the prefix at the first
# conclusion marker (`**Option N**`, `Therefore the answer is…`, etc.).
export AGDEBUGGER_DROP_WRONG_PREFIX=1

# ----------------------------------------------------------------------------
# Multi-anchor rewrite knobs (Path C, Step 2)
# ----------------------------------------------------------------------------
# AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE:
#   off             - default. Fused spans use the primary claim's
#                     corrected_claim_text as replacement (Step 1 behaviour).
#   cross_claim_only - LLM rewrites ONLY fused spans that merge multiple
#                     distinct claim_ids (highest-value, lowest-cost).
#                     Requires AGDEBUGGER_MULTI_ANCHOR_CROSS_CLAIM_FUSE=1.
#   on_fused        - LLM rewrites every fused span (>=2 anchor hits).
#   always          - LLM rewrites every span (also single-hit spans).
#
# AGDEBUGGER_REWRITE_TIMEOUT_SEC: per-span LLM timeout, default 30.
# AGDEBUGGER_MULTI_ANCHOR_CROSS_CLAIM_FUSE: 1 to fuse spans across claim_ids.
#
# Typical experimental recipe (uncomment to enable):
#   export AGDEBUGGER_MULTI_ANCHOR_LLM_REWRITE=always
#   export AGDEBUGGER_MULTI_ANCHOR_CROSS_CLAIM_FUSE=1
#   export AGDEBUGGER_REWRITE_TIMEOUT_SEC=30

bash "${SCRIPT_DIR}/rjob.sh"
