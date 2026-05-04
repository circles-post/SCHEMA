#!/usr/bin/env bash
# run_graphbench_subset.sh — run a SUBSET of the real ProteinLMBench corpus
# through the latest ontology-aware pipeline. Faster than --full but uses
# the actual benchmark seed source (not a synthetic FGFR3 seed), so it
# exercises the same code paths as the full bench at ~1/100 the cost.
#
# What this gives you that the smoke script doesn't:
#   - real benchmark CSV → seed extraction (benchmark_seed_source)
#   - real PubMed retrieval expanding from those seeds
#   - paper_cache reuse (subsequent runs hit cache)
#   - the same retrieval / fulltext / chunking / proposer / extraction /
#     graph code paths as the full benchmark, just on a small subset
#
# Defaults (override with flags or env vars):
#   question_limit  = 8       (real benchmark questions)
#   max_seeds       = 16
#   retmax          = 3
#   related_expand  = 2
#   related_per_seed= 1
#   chunk_limit     = 24
#   proposer        = enabled (sample_size=12, evidence/doc_threshold=1)
#
# Cost target: ~10 minutes, ~80 LLM calls.
#
# Usage:
#   ./scripts/run_graphbench_subset.sh
#   ./scripts/run_graphbench_subset.sh --question-limit 20 --max-seeds 30
#   ./scripts/run_graphbench_subset.sh --no-proposer
#   ./scripts/run_graphbench_subset.sh --output-dir benchmark_runs/subset_v3
#   ./scripts/run_graphbench_subset.sh --skip-healthcheck

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/agentdebug/bin/python}"
BASE_CONFIG="$ROOT_DIR/pipeline_config.benchmark.json"
ENV_SCRIPT="$ROOT_DIR/triple_extraction_env.sh"
PIPELINE_ENTRY="$ROOT_DIR/literature_pipeline.py"

# defaults — small but real
OUTPUT_DIR=""
QUESTION_LIMIT="${QUESTION_LIMIT:-8}"
MAX_SEEDS="${MAX_SEEDS:-16}"
RETMAX_PER_KEYWORD="${RETMAX_PER_KEYWORD:-3}"
RELATED_EXPAND_LIMIT="${RELATED_EXPAND_LIMIT:-2}"
RELATED_PER_SEED="${RELATED_PER_SEED:-1}"
CHUNK_LIMIT="${CHUNK_LIMIT:-24}"
PROPOSER_ENABLED="true"
SKIP_HEALTHCHECK="false"

# proposer / canonicalizer knobs (looser than --full, tighter than smoke)
PROPOSER_SAMPLE_SIZE="${PROPOSER_SAMPLE_SIZE:-12}"
PROPOSER_EVIDENCE_THRESHOLD="${PROPOSER_EVIDENCE_THRESHOLD:-1}"
PROPOSER_DISTINCT_DOC_THRESHOLD="${PROPOSER_DISTINCT_DOC_THRESHOLD:-1}"
PROPOSER_MAX_PER_KIND="${PROPOSER_MAX_PER_KIND:-6}"
PROPOSER_CACHE_DIR="${PROPOSER_CACHE_DIR:-/tmp/datasetsa_subset_proposer_cache}"
CANON_CACHE_DIR="${CANON_CACHE_DIR:-/tmp/datasetsa_subset_canon_cache}"
CANON_MIN_HITS="${CANON_MIN_HITS:-1}"
CANON_MIN_SOURCES="${CANON_MIN_SOURCES:-1}"

EMBEDDING_HOST="${EMBEDDING_HOST:-100.99.247.97}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8765}"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_graphbench_subset.sh [options]

Options:
  --output-dir DIR        Output directory (default: benchmark_runs/proteinlmbench_subset)
  --question-limit N      Real benchmark questions used as seed source (default: 8)
  --max-seeds N           Max distinct seed keywords (default: 16)
  --chunk-limit N         Cap on chunks for triple extraction (default: 24)
  --no-proposer           Disable OntologyProposerAgent
  --skip-healthcheck      Skip preflight HTTP probes
  --help, -h              Show this help

Use this script when you want a real benchmark subset run that takes
~5-10 minutes and exercises the same code paths as the full benchmark,
including OntologyProposerAgent + sciverse/pubmed/mesh grounding.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)        OUTPUT_DIR="$2"; shift 2 ;;
    --question-limit)    QUESTION_LIMIT="$2"; shift 2 ;;
    --max-seeds)         MAX_SEEDS="$2"; shift 2 ;;
    --chunk-limit)       CHUNK_LIMIT="$2"; shift 2 ;;
    --no-proposer)       PROPOSER_ENABLED="false"; shift ;;
    --skip-healthcheck)  SKIP_HEALTHCHECK="true"; shift ;;
    --help|-h)           usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/benchmark_runs/proteinlmbench_subset}"

# Network policy:
#   - The remote embedding service (default 100.99.247.97) is on the internal
#     pjlab network and must NOT go through any HTTP proxy.
#   - PubMed (eutils.ncbi.nlm.nih.gov), MeSH (id.nlm.nih.gov), Crossref
#     (api.crossref.org), bioRxiv/medRxiv (api.biorxiv.org) are public and
#     are only reachable via the pjlab HTTP proxy on this host.
# So we PREPEND the embedding host to whatever NO_PROXY the user already
# has, but never override the user's existing http_proxy / https_proxy.
# This is the opposite of the older scripts that bypassed the proxy for
# every external service — that path silently times out on this host.
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

# preflight files
[[ -x "$PYTHON_BIN"     ]] || { echo "[FAIL] python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$BASE_CONFIG"    ]] || { echo "[FAIL] missing base config: $BASE_CONFIG"  >&2; exit 1; }
[[ -f "$ENV_SCRIPT"     ]] || { echo "[FAIL] missing env script: $ENV_SCRIPT"    >&2; exit 1; }
[[ -f "$PIPELINE_ENTRY" ]] || { echo "[FAIL] missing pipeline entry"             >&2; exit 1; }

cd "$ROOT_DIR"

echo "============================================================"
echo "ProteinLMBench SUBSET run (real benchmark, small slice)"
echo "============================================================"
echo "  python interpreter : $PYTHON_BIN"
echo "  base config        : $BASE_CONFIG"
echo "  output dir         : $OUTPUT_DIR"
echo "  question_limit     : $QUESTION_LIMIT  (subset of 944)"
echo "  max_seed_keywords  : $MAX_SEEDS"
echo "  chunk_limit        : $CHUNK_LIMIT"
echo "  retmax_per_keyword : $RETMAX_PER_KEYWORD"
echo "  related_expand     : $RELATED_EXPAND_LIMIT (per-seed=$RELATED_PER_SEED)"
echo "  ontology proposer  : $PROPOSER_ENABLED"
if [[ "$PROPOSER_ENABLED" == "true" ]]; then
  echo "    sample_size            : $PROPOSER_SAMPLE_SIZE"
  echo "    evidence_threshold     : $PROPOSER_EVIDENCE_THRESHOLD"
  echo "    distinct_doc_threshold : $PROPOSER_DISTINCT_DOC_THRESHOLD"
  echo "    max_per_kind           : $PROPOSER_MAX_PER_KIND"
fi
echo

# step 0
echo "[step 0] sourcing extraction credentials"
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
for var in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL; do
  [[ -n "${!var:-}" ]] || { echo "[FAIL] $var not set after sourcing $ENV_SCRIPT" >&2; exit 1; }
done
echo "  credentials ok"

# step 1
if [[ "$SKIP_HEALTHCHECK" == "false" ]]; then
  echo
  echo "[step 1] preflight health checks"
  echo -n "  embedding service ($EMBEDDING_HOST:$EMBEDDING_PORT) ... "
  if curl -sSf -m 10 "http://$EMBEDDING_HOST:$EMBEDDING_PORT/health" >/dev/null 2>&1; then
    echo "ok"
  else
    echo "FAIL"
    echo "[FAIL] remote embedding service is unreachable. Override with --skip-healthcheck or fix network." >&2
    exit 1
  fi
  echo -n "  pubmed eutils ... "
  if curl -sSf -m 10 "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=p53&retmax=1&retmode=json" >/dev/null 2>&1; then
    echo "ok"
  else
    echo "WARN (PubMed retrieval will fail and produce 0 papers)"
  fi
fi

# step 2: assemble per-run config (overlay on the real benchmark config)
echo
echo "[step 2] assembling subset config"
TMP_CONFIG="$(mktemp /tmp/graphbench_subset.XXXXXX.json)"
cleanup() { rm -f "$TMP_CONFIG"; }
trap cleanup EXIT

"$PYTHON_BIN" - <<PY "$BASE_CONFIG" "$TMP_CONFIG" "$QUESTION_LIMIT" "$MAX_SEEDS" "$RETMAX_PER_KEYWORD" "$RELATED_EXPAND_LIMIT" "$RELATED_PER_SEED" "$PROPOSER_ENABLED" "$PROPOSER_SAMPLE_SIZE" "$PROPOSER_EVIDENCE_THRESHOLD" "$PROPOSER_DISTINCT_DOC_THRESHOLD" "$PROPOSER_MAX_PER_KIND" "$PROPOSER_CACHE_DIR" "$CANON_CACHE_DIR" "$CANON_MIN_HITS" "$CANON_MIN_SOURCES"
import json, sys
from pathlib import Path

(base, out, qlim, mseeds, retmax, related_expand, related_per_seed,
 proposer_on, sample_size, ev_thr, doc_thr, max_kind, p_cache, c_cache,
 c_min_hits, c_min_sources) = sys.argv[1:17]

base_p = Path(base)
data = json.loads(base_p.read_text())

# resolve env_file relative to the base config (matches resolve_config_paths)
env_file = data.get("env_file", ".env")
if env_file and not Path(env_file).is_absolute():
    data["env_file"] = str((base_p.parent / env_file).resolve())

data["project_name"] = "proteinlmbench_subset"

# real benchmark seed source — KEEP everything else from the base config
data.setdefault("benchmark_seed_source", {})
data["benchmark_seed_source"]["question_limit"] = int(qlim)
data["benchmark_seed_source"]["max_seed_keywords"] = int(mseeds)

data.setdefault("retrieval", {})
data["retrieval"]["retmax_per_keyword"] = int(retmax)
data["retrieval"]["related_expand_limit"] = int(related_expand)
data["retrieval"]["related_per_seed"] = int(related_per_seed)

if proposer_on.lower() == "true":
    data["ontology"] = {
        "agent_proposer_enabled": True,
        "sample_size": int(sample_size),
        "evidence_threshold": int(ev_thr),
        "distinct_doc_threshold": int(doc_thr),
        "max_proposals_per_kind": int(max_kind),
        "require_grounding": True,
        "cache_dir": p_cache,
        "canonicalizer": {
            "min_hits": int(c_min_hits),
            "min_distinct_sources": int(c_min_sources),
            "cache_dir": c_cache,
            "sciverse": {"enabled": True, "num_results": 3, "timeout_seconds": 45},
            "pubmed":   {"enabled": True, "retmax": 2},
            "mesh":     {"enabled": True, "limit": 2},
        },
    }
else:
    data["ontology"] = {"agent_proposer_enabled": False}

Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(f"  wrote {out}")
PY

# step 3
mkdir -p "$OUTPUT_DIR"

echo
echo "[step 3] running literature_pipeline.py"
echo "------------------------------------------------------------"
START_TS=$(date +%s)
CMD=(
  "$PYTHON_BIN" "$PIPELINE_ENTRY"
  --config "$TMP_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --extract-triples
  --chunk-limit "$CHUNK_LIMIT"
  --export-graph
  --graph-output "$OUTPUT_DIR/global_graph.graphml"
)
echo "  ${CMD[*]}"
echo "------------------------------------------------------------"

set +e
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
PIPELINE_RC=${PIPESTATUS[0]}
set -e
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

if [[ $PIPELINE_RC -ne 0 ]]; then
  echo
  echo "[FAIL] pipeline exited with code $PIPELINE_RC after ${ELAPSED}s" >&2
  exit $PIPELINE_RC
fi

# step 4: report
echo
echo "============================================================"
echo "RUN COMPLETE in ${ELAPSED}s"
echo "============================================================"
SUMMARY_PATH="$OUTPUT_DIR/phase_summary.json"
if [[ ! -f "$SUMMARY_PATH" ]]; then
  echo "[FAIL] phase_summary.json not found at $SUMMARY_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - <<PY "$SUMMARY_PATH" "$OUTPUT_DIR" "$PROPOSER_ENABLED"
import json, sys
from pathlib import Path

s = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
proposer_expected = sys.argv[3] == "true"

def get(d, *ks):
    cur = d
    for k in ks:
        cur = (cur or {}).get(k)
    return cur

fail_count = 0
def fail(msg):
    global fail_count
    fail_count += 1
    print(f"  [FAIL] {msg}")

print(f"  phase 1  expanded keywords     : {get(s,'phase_1_keyword_expansion','accepted_count')}")
print(f"  phase 2  retrieved unique      : {get(s,'phase_2_retrieval_and_filtering','retrieved_unique_papers')}")
print(f"  phase 2  kept after scoring    : {get(s,'phase_2_retrieval_and_filtering','kept_papers')}")
print(f"  phase 2  source breakdown      : {get(s,'phase_2_retrieval_and_filtering','retrieval_source_breakdown')}")
print(f"  phase 3  fulltext records      : {get(s,'phase_3_fulltext','fulltext_records')}")
print(f"  phase 3  abstract-only records : {get(s,'phase_3_fulltext','abstract_only_records')}")
print(f"  phase 3  cache hits            : {get(s,'phase_3_fulltext','cache_hits')}  network avoided: {get(s,'phase_3_fulltext','network_fetches_avoided')}")
print(f"  phase 3b sapbert / bge enabled : {get(s,'phase_3b_embeddings','sapbert_enabled')} / {get(s,'phase_3b_embeddings','bge_enabled')}")

p3c = s.get("phase_3c_ontology_proposer")
if proposer_expected:
    if not p3c:
        fail("phase 3c missing — proposer never ran")
    else:
        print()
        print("  phase 3c ontology proposer:")
        print(f"    status              : {p3c.get('status')}")
        print(f"    base / run version  : {p3c.get('base_version')} -> {p3c.get('run_version')}")
        print(f"    added entity types  : {p3c.get('added_entity_types')}")
        print(f"    added relations     : {p3c.get('added_relations')}")
        print(f"    added aliases       : {p3c.get('added_aliases')}")
        proposer_dir = out / "ontology_proposer"
        psum_path = proposer_dir / "summary.json"
        if psum_path.exists():
            psum = json.loads(psum_path.read_text())
            print(f"    LLM calls (succ/att): {psum.get('llm_calls_succeeded')}/{psum.get('llm_calls_attempted')}")
            print(f"    raw / aggregated    : {psum.get('raw_proposals_total')} / {psum.get('aggregated_proposals')}")
            print(f"    rejected (evidence) : {psum.get('rejected_evidence_threshold')}")
            print(f"    rejected (dedup)    : {psum.get('rejected_dedup')}")
            print(f"    rejected (ground)   : {psum.get('rejected_grounding')}")
            print(f"    accepted            : {psum.get('accepted')}")

p4 = s.get("phase_4_triple_extraction") or {}
print()
print(f"  phase 4  chunks fed   : {p4.get('chunks')}")
print(f"  phase 4  raw triples  : {p4.get('raw_triples')}")
print(f"  phase 4  normalized   : {p4.get('normalized_triples')}")
print(f"  phase 4  errors       : {p4.get('errors')}")
if int(p4.get("normalized_triples", 0) or 0) <= 0:
    fail("phase 4 produced 0 normalized triples — check chunks.jsonl quality")

p5 = s.get("phase_5_graph_export") or {}
print()
print(f"  phase 5  doc graphs   : {p5.get('doc_graph_count')}")
print(f"  phase 5  global nodes : {p5.get('global_nodes')}")
print(f"  phase 5  global edges : {p5.get('global_edges')}")
if int(p5.get("global_nodes", 0) or 0) <= 0:
    fail("phase 5 produced empty graph")

print()
print(f"  phase summary : {sys.argv[1]}")
print(f"  graphml       : {out}/global_graph.graphml")
print(f"  raw triples   : {out}/raw_triples.jsonl")
print(f"  normalized    : {out}/normalized_triples.jsonl")
print(f"  full log      : {out}/run.log")

if fail_count:
    print()
    print(f"  *** {fail_count} fail(s); subset run is NOT healthy. ***")
    sys.exit(1)
PY
VERIFY_RC=$?

echo
if [[ $VERIFY_RC -ne 0 ]]; then
  echo "============================================================"
  echo "SUBSET RUN HAD FAILURES — see above"
  echo "============================================================"
  exit $VERIFY_RC
fi
echo "============================================================"
echo "SUBSET RUN OK"
echo "  next step: ./scripts/run_graphbench_full.sh"
echo "============================================================"
