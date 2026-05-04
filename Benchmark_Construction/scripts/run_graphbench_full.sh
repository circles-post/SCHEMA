#!/usr/bin/env bash
# run_graphbench_full.sh — full ProteinLMBench knowledge graph build using
# the latest ontology-aware pipeline (post stage-4 refactor).
#
# What this does, end to end:
#   0. preflight: source extraction credentials, set NO_PROXY for upstream
#      services, health-check the remote BGE/SapBERT embedding service +
#      sciverse toolkit
#   1. clone the base benchmark config and overlay full-run sizes
#      (question_limit=944 / max_seeds=120 / retmax=5 / related=5/2)
#   2. inject the ontology block (OntologyProposerAgent + EntityCanonicalizer
#      with sciverse + PubMed + MeSH grounding) — disable with --no-proposer
#   3. run literature_pipeline.py end-to-end (retrieval -> fulltext ->
#      chunking -> ontology proposer -> triple extraction -> graph export)
#   4. print phase_summary.json + ontology_proposer summary
#
# Usage:
#   ./scripts/run_graphbench_full.sh
#   ./scripts/run_graphbench_full.sh --output-dir benchmark_runs/proteinlmbench_full_v2
#   ./scripts/run_graphbench_full.sh --question-limit 200 --max-seeds 60
#   ./scripts/run_graphbench_full.sh --no-proposer        # disable runtime ontology extension
#   ./scripts/run_graphbench_full.sh --skip-healthcheck   # skip preflight HTTP probes

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/fengxinshun/miniconda3/miniconda3/envs/agentdebug/bin/python}"
BASE_CONFIG="$ROOT_DIR/pipeline_config.benchmark.json"
ENV_SCRIPT="$ROOT_DIR/triple_extraction_env.sh"
PIPELINE_ENTRY="$ROOT_DIR/literature_pipeline.py"

# defaults
OUTPUT_DIR=""
QUESTION_LIMIT=""
CHUNK_LIMIT=""
MAX_SEEDS=""
RETMAX_PER_KEYWORD=""
RELATED_EXPAND_LIMIT=""
RELATED_PER_SEED=""
PROPOSER_ENABLED="true"
SKIP_HEALTHCHECK="false"
FORCE_OVERWRITE="false"
SCIVERSE_FULLTEXT="${SCIVERSE_FULLTEXT:-true}"
SCIVERSE_TOOLKIT_ROOT="${SCIVERSE_TOOLKIT_ROOT:-/mnt/shared-storage-user/fengxinshun/AISci/sciverse}"
SCIVERSE_FT_SEARCH_TOP_K="${SCIVERSE_FT_SEARCH_TOP_K:-5}"
SCIVERSE_FT_MIN_TITLE_OVERLAP="${SCIVERSE_FT_MIN_TITLE_OVERLAP:-0.6}"
SCIVERSE_FT_PREFER_DOI_DIRECT="${SCIVERSE_FT_PREFER_DOI_DIRECT:-true}"

# proposer + canonicalizer knobs (override via env)
# Stage 5 tuning: the first full run accepted 0 extensions because the
# previous defaults (sample_size=50, evidence=2, docs=2) were too strict
# for a 944-question corpus. New defaults trade strictness for recall but
# still require KB grounding via EntityCanonicalizer.
PROPOSER_SAMPLE_SIZE="${PROPOSER_SAMPLE_SIZE:-120}"
PROPOSER_EVIDENCE_THRESHOLD="${PROPOSER_EVIDENCE_THRESHOLD:-1}"
PROPOSER_DISTINCT_DOC_THRESHOLD="${PROPOSER_DISTINCT_DOC_THRESHOLD:-1}"
PROPOSER_MAX_PER_KIND="${PROPOSER_MAX_PER_KIND:-8}"
PROPOSER_CACHE_DIR="${PROPOSER_CACHE_DIR:-/tmp/datasetsa_proposer_cache}"
CANON_CACHE_DIR="${CANON_CACHE_DIR:-/tmp/datasetsa_canonicalizer_cache}"
CANON_MIN_HITS="${CANON_MIN_HITS:-2}"
CANON_MIN_SOURCES="${CANON_MIN_SOURCES:-1}"

EMBEDDING_HOST="${EMBEDDING_HOST:-100.99.247.97}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8765}"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_graphbench_full.sh [options]

Options:
  --output-dir DIR        Output directory (default: benchmark_runs/proteinlmbench_full_graphbench)
  --question-limit N      Override benchmark question limit (default: 944)
  --max-seeds N           Override max seed keywords (default: 120)
  --chunk-limit N         Cap on chunks for triple extraction (default: 0 = no cap)
  --no-proposer           Disable OntologyProposerAgent (run with base ontology only)
  --no-sciverse-fulltext  Disable sciverse fulltext fetch (PMC + Crossref landing only)
  --skip-healthcheck      Skip preflight HTTP probes
  --force                 Allow running against a non-empty --output-dir (overwrites)
  --help, -h              Show this help

Environment overrides (export before running):
  PYTHON_BIN              Python interpreter path
  EMBEDDING_HOST          Remote embedding service host
  EMBEDDING_PORT          Remote embedding service port
  PROPOSER_SAMPLE_SIZE    Number of chunks the proposer samples (default 50)
  PROPOSER_EVIDENCE_THRESHOLD   Min evidence per proposal (default 2)
  PROPOSER_DISTINCT_DOC_THRESHOLD  Min distinct docs per proposal (default 2)
  PROPOSER_MAX_PER_KIND   Max accepted proposals per kind (default 8)
  PROPOSER_CACHE_DIR      Where to cache ontology.run.yaml results
  CANON_CACHE_DIR         Where to cache canonicalizer queries
  CANON_MIN_HITS          Min KB hits for grounding (default 2)
  CANON_MIN_SOURCES       Min distinct backends with hits (default 1)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)        OUTPUT_DIR="$2"; shift 2 ;;
    --question-limit)    QUESTION_LIMIT="$2"; shift 2 ;;
    --chunk-limit)       CHUNK_LIMIT="$2"; shift 2 ;;
    --max-seeds)         MAX_SEEDS="$2"; shift 2 ;;
    --no-proposer)       PROPOSER_ENABLED="false"; shift ;;
    --no-sciverse-fulltext) SCIVERSE_FULLTEXT="false"; shift ;;
    --skip-healthcheck)  SKIP_HEALTHCHECK="true"; shift ;;
    --force)             FORCE_OVERWRITE="true"; shift ;;
    --help|-h)           usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/benchmark_runs/proteinlmbench_full_graphbench}"
QUESTION_LIMIT="${QUESTION_LIMIT:-944}"
CHUNK_LIMIT="${CHUNK_LIMIT:-0}"
MAX_SEEDS="${MAX_SEEDS:-120}"
RETMAX_PER_KEYWORD="${RETMAX_PER_KEYWORD:-5}"
RELATED_EXPAND_LIMIT="${RELATED_EXPAND_LIMIT:-5}"
RELATED_PER_SEED="${RELATED_PER_SEED:-2}"

# Network policy: keep the user's http_proxy / https_proxy intact so that
# external services (PubMed eutils, MeSH, Crossref, bioRxiv) reach the
# public internet via the pjlab proxy. Only the internal embedding service
# is added to NO_PROXY because it sits on the cluster's private network.
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
[[ -x "$PYTHON_BIN"  ]] || { echo "[FAIL] python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "[FAIL] missing base config: $BASE_CONFIG"  >&2; exit 1; }
[[ -f "$ENV_SCRIPT"  ]] || { echo "[FAIL] missing env script: $ENV_SCRIPT"    >&2; exit 1; }
[[ -f "$PIPELINE_ENTRY" ]] || { echo "[FAIL] missing pipeline entry: $PIPELINE_ENTRY" >&2; exit 1; }

cd "$ROOT_DIR"

echo "============================================================"
echo "ProteinLMBench knowledge-graph build (post-refactor pipeline)"
echo "============================================================"
echo "  python interpreter : $PYTHON_BIN"
echo "  base config        : $BASE_CONFIG"
echo "  output dir         : $OUTPUT_DIR"
echo "  question_limit     : $QUESTION_LIMIT"
echo "  max_seed_keywords  : $MAX_SEEDS"
echo "  chunk_limit        : $CHUNK_LIMIT"
echo "  retmax_per_keyword : $RETMAX_PER_KEYWORD"
echo "  related_expand     : $RELATED_EXPAND_LIMIT (per-seed=$RELATED_PER_SEED)"
echo "  sciverse fulltext  : $SCIVERSE_FULLTEXT (toolkit=$SCIVERSE_TOOLKIT_ROOT, top_k=$SCIVERSE_FT_SEARCH_TOP_K, min_overlap=$SCIVERSE_FT_MIN_TITLE_OVERLAP, doi_direct=$SCIVERSE_FT_PREFER_DOI_DIRECT)"
echo "  ontology proposer  : $PROPOSER_ENABLED"
if [[ "$PROPOSER_ENABLED" == "true" ]]; then
  echo "    sample_size           : $PROPOSER_SAMPLE_SIZE"
  echo "    evidence_threshold    : $PROPOSER_EVIDENCE_THRESHOLD"
  echo "    distinct_doc_threshold: $PROPOSER_DISTINCT_DOC_THRESHOLD"
  echo "    max_per_kind          : $PROPOSER_MAX_PER_KIND"
  echo "    proposer_cache        : $PROPOSER_CACHE_DIR"
  echo "    canonicalizer_cache   : $CANON_CACHE_DIR"
  echo "    canon_min_hits        : $CANON_MIN_HITS"
  echo "    canon_min_sources     : $CANON_MIN_SOURCES"
fi
echo

# step 0: extraction credentials
echo "[step 0] sourcing extraction credentials"
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
for var in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL; do
  if [[ -z "${!var:-}" ]]; then
    echo "[FAIL] env var $var is empty after sourcing $ENV_SCRIPT" >&2
    exit 1
  fi
  echo "  $var = set"
done

# step 1: preflight HTTP probes
if [[ "$SKIP_HEALTHCHECK" == "false" ]]; then
  echo
  echo "[step 1] preflight health checks"
  # 1a. embedding service
  echo -n "  embedding service ($EMBEDDING_HOST:$EMBEDDING_PORT) ... "
  if curl -sSf -m 10 "http://$EMBEDDING_HOST:$EMBEDDING_PORT/health" >/dev/null 2>&1; then
    echo "ok"
  else
    echo "FAIL"
    echo "[FAIL] remote embedding service is unreachable. Override with --skip-healthcheck or fix network." >&2
    exit 1
  fi
  # 1b. eutils
  echo -n "  pubmed eutils ... "
  if curl -sSf -m 10 "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=p53&retmax=1&retmode=json" >/dev/null 2>&1; then
    echo "ok"
  else
    echo "WARN (PubMed grounding will silently fall back)"
  fi
  # 1c. sciverse import
  echo -n "  sciverse toolkit import ... "
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
sys.path.insert(0, "/mnt/shared-storage-user/fengxinshun/AISci/sciverse")
import sciverse_tools  # noqa
PY
  then
    echo "ok"
  else
    echo "WARN (sciverse grounding will fall back)"
  fi
fi

# step 2: assemble per-run config
echo
echo "[step 2] assembling per-run config"
TMP_CONFIG="$(mktemp /tmp/graphbench_config.XXXXXX.json)"
cleanup() { rm -f "$TMP_CONFIG"; }
trap cleanup EXIT

"$PYTHON_BIN" - <<PY "$BASE_CONFIG" "$TMP_CONFIG" "$QUESTION_LIMIT" "$MAX_SEEDS" "$RETMAX_PER_KEYWORD" "$RELATED_EXPAND_LIMIT" "$RELATED_PER_SEED" "$PROPOSER_ENABLED" "$PROPOSER_SAMPLE_SIZE" "$PROPOSER_EVIDENCE_THRESHOLD" "$PROPOSER_DISTINCT_DOC_THRESHOLD" "$PROPOSER_MAX_PER_KIND" "$PROPOSER_CACHE_DIR" "$CANON_CACHE_DIR" "$CANON_MIN_HITS" "$CANON_MIN_SOURCES" "$SCIVERSE_FULLTEXT" "$SCIVERSE_TOOLKIT_ROOT" "$SCIVERSE_FT_SEARCH_TOP_K" "$SCIVERSE_FT_MIN_TITLE_OVERLAP" "$SCIVERSE_FT_PREFER_DOI_DIRECT"
import json, sys
from pathlib import Path

(base, out, qlim, mseeds, retmax, related_expand, related_per_seed,
 proposer_on, sample_size, ev_thr, doc_thr, max_kind, p_cache, c_cache,
 c_min_hits, c_min_sources,
 sv_full, sv_root, sv_topk, sv_overlap, sv_doi_direct) = sys.argv[1:22]

base_p = Path(base)
data = json.loads(base_p.read_text())

# resolve env_file relative to the base config (matches resolve_config_paths)
env_file = data.get("env_file", ".env")
if env_file and not Path(env_file).is_absolute():
    data["env_file"] = str((base_p.parent / env_file).resolve())

# project name
data["project_name"] = "proteinlmbench_graphbench_full"

# benchmark seed source overrides
data.setdefault("benchmark_seed_source", {})
data["benchmark_seed_source"]["question_limit"] = int(qlim)
data["benchmark_seed_source"]["max_seed_keywords"] = int(mseeds)

# retrieval scale overrides
data.setdefault("retrieval", {})
data["retrieval"]["retmax_per_keyword"] = int(retmax)
data["retrieval"]["related_expand_limit"] = int(related_expand)
data["retrieval"]["related_per_seed"] = int(related_per_seed)

# top-level sciverse block — drives PMCFullTextFetcher (phase 3). This is
# SEPARATE from the sciverse config under ontology.canonicalizer, which
# only powers the OntologyProposer grounding pass. Writing here ensures
# paywalled PDFs get a real download attempt via sci-hub/elsevier channels
# before falling back to crossref landing page / abstract-only.
if sv_full.lower() == "true":
    data["sciverse"] = {
        "enabled": True,
        "toolkit_root": sv_root,
        "search_top_k": int(sv_topk),
        "language": "en",
        "require_match": True,
        "min_title_token_overlap": float(sv_overlap),
        "prefer_doi_direct": sv_doi_direct.lower() == "true",
    }
else:
    data["sciverse"] = {"enabled": False}

# ontology block (post-refactor)
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
            "sciverse": {"enabled": True, "num_results": 5, "timeout_seconds": 60},
            "pubmed":   {"enabled": True, "retmax": 3},
            "mesh":     {"enabled": True, "limit": 3},
        },
    }
else:
    data["ontology"] = {"agent_proposer_enabled": False}

Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(f"  wrote {out}")
PY

# step 3: ensure output dir is fresh enough (don't blow it away if user reused one)
# Guard: refuse to run against a non-empty output dir unless --force is set.
# This prevents silently overwriting chunks.jsonl / raw_triples.jsonl /
# normalized_triples.jsonl / global_graph.graphml from a prior run, and
# stops ontology_proposer/*.jsonl from being appended onto stale decisions.
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]]; then
  if [[ "$FORCE_OVERWRITE" != "true" ]]; then
    echo "[FAIL] output dir is not empty: $OUTPUT_DIR" >&2
    echo "       Pass --force to overwrite, or pick a fresh --output-dir." >&2
    echo "       Contents:" >&2
    ls -A "$OUTPUT_DIR" | sed 's/^/         /' >&2
    exit 1
  fi
  echo "[WARN] --force set: overwriting existing output dir $OUTPUT_DIR"
  # Clear append-mode logs so ontology_proposer decisions/rejected JSONLs
  # don't accumulate stale lines across runs.
  rm -f "$OUTPUT_DIR/ontology_proposer/ontology_decisions.jsonl" \
        "$OUTPUT_DIR/ontology_proposer/ontology.rejected.jsonl"
fi
mkdir -p "$OUTPUT_DIR"

# step 4: run pipeline
echo
echo "[step 3] running literature_pipeline.py"
echo "------------------------------------------------------------"
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
"${CMD[@]}"

# step 5: report
echo
echo "============================================================"
echo "RUN COMPLETE"
echo "============================================================"
SUMMARY_PATH="$OUTPUT_DIR/phase_summary.json"
if [[ -f "$SUMMARY_PATH" ]]; then
  "$PYTHON_BIN" - <<PY "$SUMMARY_PATH" "$OUTPUT_DIR"
import json, sys
from pathlib import Path

s = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])

def get(d, *ks):
    cur = d
    for k in ks:
        cur = (cur or {}).get(k)
    return cur

print(f"  phase 1  expanded keywords     : {get(s,'phase_1_keyword_expansion','accepted_count')}")
print(f"  phase 2  retrieved unique      : {get(s,'phase_2_retrieval_and_filtering','retrieved_unique_papers')}")
print(f"  phase 2  kept after scoring    : {get(s,'phase_2_retrieval_and_filtering','kept_papers')}")
print(f"  phase 2  source breakdown      : {get(s,'phase_2_retrieval_and_filtering','retrieval_source_breakdown')}")
print(f"  phase 3  fulltext records      : {get(s,'phase_3_fulltext','fulltext_records')}")
print(f"  phase 3  abstract-only records : {get(s,'phase_3_fulltext','abstract_only_records')}")
print(f"  phase 3  cache hits            : {get(s,'phase_3_fulltext','cache_hits')}")
print(f"  phase 3  network avoided       : {get(s,'phase_3_fulltext','network_fetches_avoided')}")
print(f"  phase 3b sapbert / bge enabled : {get(s,'phase_3b_embeddings','sapbert_enabled')} / {get(s,'phase_3b_embeddings','bge_enabled')}")

p3c = s.get("phase_3c_ontology_proposer")
if p3c:
    print()
    print("  phase 3c ontology proposer:")
    print(f"    status              : {p3c.get('status')}")
    print(f"    base/run version    : {p3c.get('base_version')} -> {p3c.get('run_version')}")
    print(f"    added entity types  : {p3c.get('added_entity_types')}")
    print(f"    added relations     : {p3c.get('added_relations')}")
    print(f"    added aliases       : {p3c.get('added_aliases')}")
    print(f"    run yaml            : {p3c.get('run_yaml')}")
    print(f"    decisions log       : {p3c.get('decisions_log')}")
else:
    print()
    print("  phase 3c ontology proposer: disabled")

p4 = s.get("phase_4_triple_extraction") or {}
print()
print(f"  phase 4  chunks fed   : {p4.get('chunks')}")
print(f"  phase 4  raw triples  : {p4.get('raw_triples')}")
print(f"  phase 4  normalized   : {p4.get('normalized_triples')}")
print(f"  phase 4  errors       : {p4.get('errors')}")

p5 = s.get("phase_5_graph_export") or {}
print()
print(f"  phase 5  doc graphs   : {p5.get('doc_graph_count')}")
print(f"  phase 5  global nodes : {p5.get('global_nodes')}")
print(f"  phase 5  global edges : {p5.get('global_edges')}")
print()
print(f"  phase summary : {sys.argv[1]}")
print(f"  graphml       : {out}/global_graph.graphml")
print(f"  raw triples   : {out}/raw_triples.jsonl")
print(f"  normalized    : {out}/normalized_triples.jsonl")
PY
else
  echo "[WARN] phase_summary.json not found at $SUMMARY_PATH"
fi
echo "============================================================"
