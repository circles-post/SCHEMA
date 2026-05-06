#!/usr/bin/env bash
# run_graphbench_smoke.sh — minimal end-to-end smoke test for the
# post-refactor pipeline. Verifies the full chain in under a minute:
#
#   retrieval -> fulltext -> chunking -> ontology proposer (+ grounding)
#   -> triple extraction -> normalize (ontology-driven) -> graphml export
#
# Designed to catch any regression in the latest pipeline before launching
# a full benchmark run. Cost target: ~30 LLM calls, < 60 s.
#
# Usage:
#   ./scripts/run_graphbench_smoke.sh
#   ./scripts/run_graphbench_smoke.sh --no-proposer        # skip ontology proposer
#   ./scripts/run_graphbench_smoke.sh --keep               # keep output dir on success
#   ./scripts/run_graphbench_smoke.sh --out /tmp/my_smoke  # custom output dir

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SCRIPT="$ROOT_DIR/triple_extraction_env.sh"
PIPELINE_ENTRY="$ROOT_DIR/literature_pipeline.py"
EMBEDDING_HOST="${EMBEDDING_HOST:-<embedding-host>}"

OUTPUT_DIR="/tmp/pubmed_graph_graphbench_smoke"
PROPOSER_ENABLED="true"
KEEP_OUTPUT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-proposer) PROPOSER_ENABLED="false"; shift ;;
    --out)         OUTPUT_DIR="$2"; shift 2 ;;
    --keep)        KEEP_OUTPUT="true"; shift ;;
    --help|-h)
      cat <<'USAGE'
Usage:
  ./scripts/run_graphbench_smoke.sh [--no-proposer] [--out DIR] [--keep]

Runs the smallest viable pipeline that exercises every phase. Default seed
is "FGFR3 urothelial carcinoma" (mechanistic, reliably produces triples).
Reports a pass/fail per phase at the end.
USAGE
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Network policy: keep user's http_proxy / https_proxy so PubMed/biorxiv/crossref
# reach the public internet via the pjlab proxy. Only the internal embedding
# host is added to NO_PROXY (it sits on the cluster's private network).
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

# preflight
[[ -x "$PYTHON_BIN"     ]] || { echo "[FAIL] python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$ENV_SCRIPT"     ]] || { echo "[FAIL] missing env script: $ENV_SCRIPT"   >&2; exit 1; }
[[ -f "$PIPELINE_ENTRY" ]] || { echo "[FAIL] missing pipeline entry"            >&2; exit 1; }

cd "$ROOT_DIR"

echo "============================================================"
echo "graphbench smoke test"
echo "  output dir       : $OUTPUT_DIR"
echo "  ontology proposer: $PROPOSER_ENABLED"
echo "============================================================"

# step 0: env
echo
echo "[step 0] sourcing extraction credentials"
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
for var in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL; do
  [[ -n "${!var:-}" ]] || { echo "[FAIL] $var not set after sourcing $ENV_SCRIPT" >&2; exit 1; }
done
echo "  credentials ok"

# step 1: build a tiny inline config
echo
echo "[step 1] writing tiny inline config"
TMP_CONFIG="$(mktemp /tmp/graphbench_smoke.XXXXXX.json)"
cleanup() { rm -f "$TMP_CONFIG"; }
trap cleanup EXIT

"$PYTHON_BIN" - <<PY "$TMP_CONFIG" "$PROPOSER_ENABLED"
import json, sys
out_path, proposer_on = sys.argv[1], sys.argv[2] == "true"
config = {
    "project_name": "pubmed_graph_graphbench_smoke",
    "env_file": ".env",
    "seed_keywords": ["FGFR3 urothelial carcinoma"],
    "keyword_expansion": {
        "iterations": 1,
        "max_terms": 6,
        "min_term_length": 3,
        "heuristic_suffixes": [],
        "manual_synonyms": {
            "fgfr3 urothelial carcinoma": ["FGFR3 bladder cancer", "erdafitinib FGFR3"]
        },
        "generic_replacements": {},
        "mesh": {"enabled": False},
        "openai": {"enabled": False},
    },
    "pubmed": {"api_key": "", "email": "graphbench_smoke@example.com"},
    "crossref": {"enabled": False},
    "biorxiv": {"enabled": False},
    "retrieval": {
        "max_workers": 2,
        "retmax_per_keyword": 3,
        "query_template": "(\"{keyword}\"[Title/Abstract]) AND english[Language]",
        "mindate": "2022/01/01",
        "maxdate": "2024/12/31",
        "sleep_seconds": 0.0,
        "journal_allowlist": [],
        "related_expand_limit": 0,
        "related_per_seed": 0,
    },
    "scoring": {
        "semantic_weight": 0.7,
        "impact_weight": 0.3,
        "score_threshold": 0.0,
        "journal_impact_factors": {},
        "sapbert": {"enabled": False},
    },
    "embedding": {"enabled": False},
    "fulltext": {
        "prefer_pmc": True,
        "abstract_fallback": True,
        "only_kept": True,
        "max_workers": 2,
        "prefer_crossref_landing_page": False,
    },
    "paper_cache": {
        "enabled": True,
        "cache_dir": "/tmp/pubmed_graph_smoke_cache",
        "store_pdf": False,
        "store_xml": True,
        "store_text": True,
        "store_landing_page": False,
    },
    "sciverse": {"enabled": False},
    "chunking": {"chunk_words": 180, "overlap_words": 40},
    "graph": {"fusion_threshold": 0.88, "min_component_size": 2},
    "pdf_parsing": {"enabled": False},
    "openai_extraction": {
        "enabled": True,
        "model": "intern-s1-pro",
        "base_url": "https://chat.intern-ai.org.cn/api/v1/",
        "confidence_threshold": 0.3,
        "thinking_mode": False,
        "temperature": 0.0,
        "max_tokens": 1000,
    },
}
if proposer_on:
    config["ontology"] = {
        "agent_proposer_enabled": True,
        "sample_size": 6,
        "evidence_threshold": 1,
        "distinct_doc_threshold": 1,
        "max_proposals_per_kind": 4,
        "require_grounding": True,
        "cache_dir": "/tmp/pubmed_graph_smoke_proposer_cache",
        "canonicalizer": {
            "min_hits": 1,
            "min_distinct_sources": 1,
            "cache_dir": "/tmp/pubmed_graph_smoke_canon_cache",
            "sciverse": {"enabled": True, "num_results": 3, "timeout_seconds": 30},
            "pubmed":   {"enabled": True, "retmax": 2},
            "mesh":     {"enabled": True, "limit": 2},
        },
    }
else:
    config["ontology"] = {"agent_proposer_enabled": False}

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(config, fh, indent=2, ensure_ascii=False)
print(f"  wrote {out_path}")
PY

# step 2: run pipeline
echo
echo "[step 2] running pipeline (seed=FGFR3 urothelial carcinoma, retmax=3, chunk-limit=10)"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

set +e
"$PYTHON_BIN" "$PIPELINE_ENTRY" \
  --config "$TMP_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --extract-triples \
  --chunk-limit 10 \
  --export-graph \
  --graph-output "$OUTPUT_DIR/global_graph.graphml" \
  > "$OUTPUT_DIR/pipeline_stdout.log" 2>&1
PIPELINE_RC=$?
set -e

if [[ $PIPELINE_RC -ne 0 ]]; then
  echo "[FAIL] pipeline exited with code $PIPELINE_RC. Tail of log:" >&2
  tail -40 "$OUTPUT_DIR/pipeline_stdout.log" >&2
  exit $PIPELINE_RC
fi
echo "  pipeline exited cleanly"

# step 3: phase-by-phase pass/fail
echo
echo "[step 3] verifying phases"
"$PYTHON_BIN" - <<PY "$OUTPUT_DIR" "$PROPOSER_ENABLED"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
proposer_expected = sys.argv[2] == "true"

ok_count = 0
warn_count = 0
fail_count = 0

def ok(msg):
    global ok_count
    ok_count += 1
    print(f"  [ok]   {msg}")

def warn(msg):
    global warn_count
    warn_count += 1
    print(f"  [warn] {msg}")

def fail(msg):
    global fail_count
    fail_count += 1
    print(f"  [FAIL] {msg}")

# Required artifacts
required = [
    "resolved_seed_keywords.json",
    "expanded_keywords.json",
    "accepted_keywords.json",
    "pubmed_candidates.jsonl",
    "pubmed_candidates_scored.jsonl",
    "pmc_fulltext.jsonl",
    "chunks.jsonl",
    "raw_triples.jsonl",
    "normalized_triples.jsonl",
    "phase_summary.json",
    "global_graph.graphml",
]
for name in required:
    p = out / name
    if not p.exists():
        fail(f"missing artifact: {name}")
    else:
        ok(f"artifact present: {name}")

summary = json.loads((out / "phase_summary.json").read_text())

p1 = summary.get("phase_1_keyword_expansion", {})
if int(p1.get("accepted_count", 0) or 0) > 0:
    ok(f"phase 1 accepted_count = {p1.get('accepted_count')}")
else:
    fail(f"phase 1 produced no accepted keywords ({p1})")

p2 = summary.get("phase_2_retrieval_and_filtering", {})
if int(p2.get("kept_papers", 0) or 0) > 0:
    ok(f"phase 2 kept_papers = {p2.get('kept_papers')} / unique = {p2.get('retrieved_unique_papers')}")
else:
    fail(f"phase 2 retrieved 0 papers — likely network issue ({p2})")

p3 = summary.get("phase_3_fulltext", {})
total_fulltext = int(p3.get("fulltext_records", 0) or 0) + int(p3.get("abstract_only_records", 0) or 0)
if total_fulltext > 0:
    ok(f"phase 3 fulltext+abstract records = {total_fulltext}")
else:
    fail(f"phase 3 produced 0 fulltext or abstract records")

chunks_count = sum(1 for _ in (out / "chunks.jsonl").open())
if chunks_count > 0:
    ok(f"phase 3 chunks.jsonl rows = {chunks_count}")
else:
    fail("phase 3 chunks.jsonl is empty")

# phase 3c: ontology proposer (optional)
p3c = summary.get("phase_3c_ontology_proposer")
if proposer_expected:
    if not p3c:
        fail("phase 3c ontology_proposer summary missing — proposer never ran")
    else:
        ok(f"phase 3c proposer status = {p3c.get('status')}")
        proposer_dir = out / "ontology_proposer"
        if (proposer_dir / "summary.json").exists():
            psum = json.loads((proposer_dir / "summary.json").read_text())
            ok(
                f"phase 3c proposer LLM calls = {psum.get('llm_calls_succeeded')}/"
                f"{psum.get('llm_calls_attempted')} (succeeded/attempted), "
                f"raw proposals = {psum.get('raw_proposals_total')}, "
                f"accepted = {psum.get('accepted')}"
            )
            if (psum.get("llm_calls_attempted") or 0) == 0:
                fail("phase 3c proposer made 0 LLM calls — chunks did not reach the agent")
        else:
            warn("phase 3c proposer summary.json missing")
        if (proposer_dir / "ontology.run.yaml").exists():
            ok("phase 3c ontology.run.yaml written")
        else:
            fail("phase 3c ontology.run.yaml missing")
else:
    if p3c is None:
        ok("phase 3c proposer correctly skipped (--no-proposer)")
    else:
        warn(f"phase 3c proposer summary present despite --no-proposer: {p3c}")

p4 = summary.get("phase_4_triple_extraction") or {}
if int(p4.get("normalized_triples", 0) or 0) > 0:
    ok(
        f"phase 4 chunks={p4.get('chunks')} raw_triples={p4.get('raw_triples')} "
        f"normalized={p4.get('normalized_triples')} errors={p4.get('errors')}"
    )
else:
    fail(
        f"phase 4 produced 0 normalized triples — chunks={p4.get('chunks')} "
        f"raw={p4.get('raw_triples')} errors={p4.get('errors')}"
    )

p5 = summary.get("phase_5_graph_export") or {}
if int(p5.get("global_nodes", 0) or 0) > 0 and int(p5.get("global_edges", 0) or 0) > 0:
    ok(f"phase 5 nodes={p5.get('global_nodes')} edges={p5.get('global_edges')}")
else:
    fail(
        f"phase 5 produced empty graph (nodes={p5.get('global_nodes')} "
        f"edges={p5.get('global_edges')})"
    )

# graphml roundtrip
try:
    import networkx as nx
    g = nx.read_graphml(str(out / "global_graph.graphml"))
    ok(f"graphml readable: nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
except Exception as exc:
    fail(f"graphml unreadable: {exc}")

print()
print(f"summary: ok={ok_count} warn={warn_count} fail={fail_count}")
sys.exit(1 if fail_count > 0 else 0)
PY
VERIFY_RC=$?

# step 4: print head of decisions log if proposer ran
if [[ "$PROPOSER_ENABLED" == "true" && -f "$OUTPUT_DIR/ontology_proposer/ontology_decisions.jsonl" ]]; then
  echo
  echo "[step 4] first 5 ontology proposer decisions"
  head -5 "$OUTPUT_DIR/ontology_proposer/ontology_decisions.jsonl" \
    | "$PYTHON_BIN" -c "
import json, sys
for line in sys.stdin:
    try: d = json.loads(line)
    except: continue
    if d.get('kind') == 'proposer_call_ok':
        print(f\"  call ok  doc={d.get('doc_id','?')[:30]} types={d['n_new_entity_types']} rels={d['n_new_relations']} aliases={d['n_new_entity_aliases']}\")
    else:
        print(f\"  {d.get('decision','?'):8s} {d.get('kind','?'):14s} {d.get('key','?')}\")
"
fi

echo
echo "============================================================"
if [[ $VERIFY_RC -ne 0 ]]; then
  echo "SMOKE TEST FAILED — see output above and $OUTPUT_DIR/pipeline_stdout.log"
else
  echo "SMOKE TEST PASSED"
  echo "  output dir: $OUTPUT_DIR"
  echo "  inspect:    $OUTPUT_DIR/phase_summary.json"
fi
echo "============================================================"

if [[ $VERIFY_RC -eq 0 && "$KEEP_OUTPUT" == "false" ]]; then
  echo "(set --keep to retain $OUTPUT_DIR)"
fi
exit $VERIFY_RC
