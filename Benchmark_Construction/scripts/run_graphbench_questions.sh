#!/usr/bin/env bash
# run_graphbench_questions.sh — build question_generation benchmark questions
# on top of a graph produced by ``run_graphbench_full.sh``.
#
# What this does:
#   1. validate that the graph build has produced normalized_triples.jsonl +
#      chunks.jsonl under --graph-dir (default:
#      benchmark_runs/proteinlmbench_full_graphbench)
#   2. source extraction credentials + question_generation/.env so OPENAI_*/
#      INTERN_* / GITHUB_TOKEN are all populated
#   3. run question_generation.cli with sensible defaults covering all 5
#      question types (claim_choice / boolean_support / two_hop_tail / essay /
#      experiment_code) + LLM code-selection in auto mode
#   4. print question_generation/summary.json
#
# Usage:
#   ./scripts/run_graphbench_questions.sh
#   ./scripts/run_graphbench_questions.sh --graph-dir benchmark_runs/proteinlmbench_full_v2
#   ./scripts/run_graphbench_questions.sh --max-samples 200 --experiment-difficulty mixed
#   ./scripts/run_graphbench_questions.sh --validation-mode hybrid_model --validator-enabled
#   ./scripts/run_graphbench_questions.sh --question-types claim_choice experiment_code

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SCRIPT="$ROOT_DIR/triple_extraction_env.sh"
QG_DIR="$ROOT_DIR/question_generation"
QG_ENV="$QG_DIR/.env"

# defaults — override via CLI
GRAPH_DIR=""
OUTPUT_DIR=""
MAX_SAMPLES=""
MAX_PER_UNIQUENESS_KEY=""
MIN_CONFIDENCE=""
VALIDATION_MODE="rule_only"
VALIDATOR_ENABLED="false"
VALIDATION_CACHE_DIR=""
RETRIEVAL_TOP_K=""
EXPERIMENT_DIFFICULTY="mixed"
LLM_CODE_SELECTION="auto"
GITHUB_SEARCH_LANGUAGE="Python"
GITHUB_SEARCH_PER_PAGE="3"
QUESTION_TYPES=(claim_choice boolean_support two_hop_tail essay experiment_code)
WORKERS=""
SAMPLE_TIMEOUT=""
EXPERIMENT_GENERATION_MODE=""
FORCE_OVERWRITE="false"

# proxy policy — mirror run_graphbench_full.sh: keep user's proxy intact so
# the validator LLM + PubMed evidence retrieval can reach public internet,
# but add internal embedding + sandbox hosts to NO_PROXY.
EMBEDDING_HOST="${EMBEDDING_HOST:-<embedding-host>}"
SANDBOX_HOST="${QG_SANDBOX_HOST:-<sandbox-host>}"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_graphbench_questions.sh [options]

Input (graph):
  --graph-dir DIR           Directory produced by run_graphbench_full.sh
                            (must contain normalized_triples.jsonl + chunks.jsonl).
                            Default: benchmark_runs/proteinlmbench_full_graphbench

Output:
  --output-dir DIR          Where to write question_samples.jsonl + summary.json.
                            Default: <graph-dir>/question_generation
  --force                   Overwrite existing output-dir

Generation knobs:
  --max-samples N           Hard cap on sampled subgraphs (default: 100). Raise this
                            AND --max-per-uniqueness-key together to scale up output count.
  --max-per-uniqueness-key N  How many samples to keep that share the same
                              (head,relation,tail,question_type). Default 1. Raise to e.g. 3
                              to triple the output of a small graph at the cost of paraphrase
                              repetition.
  --min-confidence F        Minimum triple confidence (default: config default)
  --experiment-difficulty X {easy,medium,hard,mixed}  (default: mixed)
  --llm-code-selection X    {auto,on,off}  (default: auto — uses LLM when creds present)
  --question-types A B C    Space-separated subset of:
                              claim_choice boolean_support two_hop_tail essay experiment_code
                            (default: all five)
  --github-search-language L  (default: Python)
  --github-search-per-page N  (default: 3)

Validation (optional):
  --validation-mode X       {rule_only,hybrid_model} (default: rule_only)
  --validator-enabled       Enable the LLM judge (only effective with hybrid_model)
  --validation-cache-dir D  Cache dir for judge verdicts
  --retrieval-top-k N       How many evidence chunks to feed the judge

Other:
  --help, -h                Show this help

Environment overrides (export before running):
  PYTHON_BIN                Python interpreter (default: agentdebug env)
  EMBEDDING_HOST            Added to NO_PROXY (default: <embedding-host>)
USAGE
}

# ---- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph-dir)              GRAPH_DIR="$2"; shift 2 ;;
    --output-dir)             OUTPUT_DIR="$2"; shift 2 ;;
    --max-samples)            MAX_SAMPLES="$2"; shift 2 ;;
    --max-per-uniqueness-key) MAX_PER_UNIQUENESS_KEY="$2"; shift 2 ;;
    --min-confidence)         MIN_CONFIDENCE="$2"; shift 2 ;;
    --validation-mode)        VALIDATION_MODE="$2"; shift 2 ;;
    --validator-enabled)      VALIDATOR_ENABLED="true"; shift ;;
    --validation-cache-dir)   VALIDATION_CACHE_DIR="$2"; shift 2 ;;
    --retrieval-top-k)        RETRIEVAL_TOP_K="$2"; shift 2 ;;
    --experiment-difficulty)  EXPERIMENT_DIFFICULTY="$2"; shift 2 ;;
    --llm-code-selection)     LLM_CODE_SELECTION="$2"; shift 2 ;;
    --github-search-language) GITHUB_SEARCH_LANGUAGE="$2"; shift 2 ;;
    --github-search-per-page) GITHUB_SEARCH_PER_PAGE="$2"; shift 2 ;;
    --question-types)
      QUESTION_TYPES=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        # Accept both `--question-types a b c` and `--question-types "a b c"`:
        # word-split each incoming arg on whitespace so quoted forms work too.
        # shellcheck disable=SC2206
        QUESTION_TYPES+=($1)
        shift
      done
      ;;
    --workers)                WORKERS="$2"; shift 2 ;;
    --sample-timeout)         SAMPLE_TIMEOUT="$2"; shift 2 ;;
    --experiment-generation-mode) EXPERIMENT_GENERATION_MODE="$2"; shift 2 ;;
    --force)                  FORCE_OVERWRITE="true"; shift ;;
    --help|-h)                usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ---- resolve paths ----------------------------------------------------------
GRAPH_DIR="${GRAPH_DIR:-$ROOT_DIR/benchmark_runs/proteinlmbench_full_graphbench}"
if [[ ! -d "$GRAPH_DIR" ]]; then
  echo "[FAIL] --graph-dir does not exist: $GRAPH_DIR" >&2
  echo "       run ./scripts/run_graphbench_full.sh first" >&2
  exit 1
fi

TRIPLES_PATH="$GRAPH_DIR/normalized_triples.jsonl"
CHUNKS_PATH="$GRAPH_DIR/chunks.jsonl"
for f in "$TRIPLES_PATH" "$CHUNKS_PATH"; do
  if [[ ! -s "$f" ]]; then
    echo "[FAIL] required graph output missing or empty: $f" >&2
    exit 1
  fi
done

OUTPUT_DIR="${OUTPUT_DIR:-$GRAPH_DIR/question_generation}"
if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" && "$FORCE_OVERWRITE" != "true" ]]; then
  echo "[FAIL] output dir is non-empty: $OUTPUT_DIR" >&2
  echo "       pass --force to overwrite, or choose a different --output-dir" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

SAMPLES_PATH="$OUTPUT_DIR/question_samples.jsonl"
SUMMARY_PATH="$OUTPUT_DIR/summary.json"
LOG_PATH="$OUTPUT_DIR/run.log"

# ---- preflight --------------------------------------------------------------
[[ -x "$PYTHON_BIN" ]] || { echo "[FAIL] python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$ENV_SCRIPT" ]] || { echo "[FAIL] missing env script: $ENV_SCRIPT" >&2; exit 1; }
[[ -d "$QG_DIR"    ]] || { echo "[FAIL] question_generation package missing: $QG_DIR" >&2; exit 1; }

# NO_PROXY: add internal hosts (embedding service + code sandbox) so they
# bypass the proxy, while keeping the existing proxy settings so GitHub /
# OpenAI / PubMed still route through pjlab egress.
EXISTING_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
for host in "$EMBEDDING_HOST" "$SANDBOX_HOST"; do
  [[ -z "$host" ]] && continue
  case ",$EXISTING_NO_PROXY," in
    *,"$host",*) : ;;  # already present
    *) EXISTING_NO_PROXY="${EXISTING_NO_PROXY:+$EXISTING_NO_PROXY,}$host" ;;
  esac
done
export NO_PROXY="$EXISTING_NO_PROXY"
export no_proxy="$NO_PROXY"

# Also export QG_SANDBOX_HOST so sandbox_client picks it up even if the
# hard-coded constant ever gets overridden upstream.
export QG_SANDBOX_HOST="$SANDBOX_HOST"

# ---- banner -----------------------------------------------------------------
echo "============================================================"
echo "Question generation on top of graphbench output"
echo "============================================================"
echo "  python interpreter : $PYTHON_BIN"
echo "  graph dir          : $GRAPH_DIR"
echo "    triples          : $TRIPLES_PATH ($(wc -l < "$TRIPLES_PATH") lines)"
echo "    chunks           : $CHUNKS_PATH ($(wc -l < "$CHUNKS_PATH") lines)"
echo "  output dir         : $OUTPUT_DIR"
echo "  log file           : $LOG_PATH"
echo "  sandbox host       : $SANDBOX_HOST (in NO_PROXY)"
echo "  question types     : ${QUESTION_TYPES[*]}"
echo "  experiment diff    : $EXPERIMENT_DIFFICULTY"
echo "  llm code selection : $LLM_CODE_SELECTION"
echo "  validation mode    : $VALIDATION_MODE (validator_enabled=$VALIDATOR_ENABLED)"
[[ -n "$MAX_SAMPLES" ]]    && echo "  max samples        : $MAX_SAMPLES"
[[ -n "$MIN_CONFIDENCE" ]] && echo "  min confidence     : $MIN_CONFIDENCE"
echo

# ---- step 1: source creds ---------------------------------------------------
echo "[step 1] sourcing extraction credentials"
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
for var in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL; do
  if [[ -z "${!var:-}" ]]; then
    echo "[FAIL] env var $var is empty after sourcing $ENV_SCRIPT" >&2
    exit 1
  fi
  echo "  $var = set"
done

# Also source question_generation/.env so GITHUB_TOKEN is available to the
# experiment_code GitHub search path. This file is intentionally kept
# separate from triple_extraction_env.sh so the GH token never lands in a
# global env script.
if [[ -f "$QG_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$QG_ENV"
  set +a
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "  GITHUB_TOKEN = set (from $QG_ENV)"
  else
    echo "  GITHUB_TOKEN = empty (experiment_code GitHub fetches will return degraded packs)"
  fi
else
  echo "  [WARN] $QG_ENV not found — experiment_code will run without a GitHub token"
fi

# ---- step 2: assemble CLI args ---------------------------------------------
echo
echo "[step 2] launching question_generation.cli"

CMD=(
  "$PYTHON_BIN" -u -m question_generation.cli
  --triples "$TRIPLES_PATH"
  --chunks  "$CHUNKS_PATH"
  --output  "$SAMPLES_PATH"
  --summary-output "$SUMMARY_PATH"
  --question-types "${QUESTION_TYPES[@]}"
  --validation-mode "$VALIDATION_MODE"
  --experiment-difficulty "$EXPERIMENT_DIFFICULTY"
  --llm-code-selection "$LLM_CODE_SELECTION"
  --github-search-language "$GITHUB_SEARCH_LANGUAGE"
  --github-search-per-page "$GITHUB_SEARCH_PER_PAGE"
  --log-file "$LOG_PATH"
  --log-level INFO
)
[[ -n "$MAX_SAMPLES" ]]              && CMD+=(--max-samples "$MAX_SAMPLES")
[[ -n "$MAX_PER_UNIQUENESS_KEY" ]]   && CMD+=(--max-per-uniqueness-key "$MAX_PER_UNIQUENESS_KEY")
[[ -n "$MIN_CONFIDENCE" ]]           && CMD+=(--min-confidence "$MIN_CONFIDENCE")
[[ -n "$WORKERS" ]]                  && CMD+=(--workers "$WORKERS")
[[ -n "$SAMPLE_TIMEOUT" ]]           && CMD+=(--sample-timeout "$SAMPLE_TIMEOUT")
[[ -n "$EXPERIMENT_GENERATION_MODE" ]] && CMD+=(--experiment-generation-mode "$EXPERIMENT_GENERATION_MODE")
[[ -n "$RETRIEVAL_TOP_K" ]]      && CMD+=(--retrieval-top-k "$RETRIEVAL_TOP_K")
[[ -n "$VALIDATION_CACHE_DIR" ]] && CMD+=(--validation-cache-dir "$VALIDATION_CACHE_DIR")
[[ "$VALIDATOR_ENABLED" == "true" ]] && CMD+=(--validator-enabled)

echo "------------------------------------------------------------"
echo "  ${CMD[*]}"
echo "------------------------------------------------------------"

# question_generation imports ``pubmed_graph`` at module load, so PYTHONPATH
# must include the parent dir. HF_HUB_OFFLINE skips a slow HuggingFace Hub
# init on agentdebug's torch stack.
#
# - PYTHONUNBUFFERED=1 + python -u: ensure stderr/stdout are line-buffered so
#   logger.info lines hit disk immediately, not after a 4KB block fills.
# - setsid + trap '' HUP: detach from the controlling terminal so that
#   disconnecting the shell (ssh drop, terminal close) does NOT SIGHUP us.
#   The process keeps running and the log keeps growing; re-attach with
#   `tail -f "$LOG_PATH"`.
# - We also wipe __pycache__ to guarantee no stale cli.py bytecode from a
#   previous import lurks in the env.
find "$ROOT_DIR/question_generation" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "  PID will be: $$  (parent script)"
echo "  log file    : $LOG_PATH"
echo "  follow with : tail -f $LOG_PATH"
echo

trap '' HUP
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  setsid --wait "${CMD[@]}"

# ---- step 3: report ---------------------------------------------------------
echo
echo "============================================================"
echo "QUESTION GENERATION COMPLETE"
echo "============================================================"
if [[ -f "$SUMMARY_PATH" ]]; then
  "$PYTHON_BIN" - <<PY "$SUMMARY_PATH" "$SAMPLES_PATH"
import json, sys
from pathlib import Path
from collections import Counter

summary = json.loads(Path(sys.argv[1]).read_text())
samples_path = Path(sys.argv[2])

print(f"  triples         : {summary.get('triples')}")
print(f"  chunks          : {summary.get('chunks')}")
print(f"  sampled graphs  : {summary.get('sampled_subgraphs')}")
print(f"  accepted        : {summary.get('accepted_questions')}")
print(f"  validation      : {summary.get('validation')}")
print(f"  q_types         : {summary.get('question_types')}")
print(f"  experiment diff : {summary.get('experiment_difficulty')}")
breakdown = summary.get('experiment_blueprint_breakdown') or {}
if breakdown:
    print("  blueprint breakdown:")
    for key, count in sorted(breakdown.items()):
        print(f"    {key:45s} {count}")

if samples_path.exists():
    by_type = Counter()
    for line in samples_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            by_type[json.loads(line)['question_type']] += 1
        except Exception:
            pass
    if by_type:
        print("  type counts (on disk):")
        for k, v in sorted(by_type.items()):
            print(f"    {k:22s} {v}")
print()
print(f"  samples  : {sys.argv[2]}")
print(f"  summary  : {sys.argv[1]}")
PY
else
  echo "[WARN] summary.json not found at $SUMMARY_PATH"
fi
echo "============================================================"
