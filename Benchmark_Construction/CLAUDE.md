# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Scope note: the `question_generation/` subdirectory is excluded from this guide and has its own `AGENTS.md`. Treat it as a separate project.

## Repository purpose

End-to-end "PubMed graph" pipeline that turns biomedical seed keywords (or a benchmark dataset such as ProteinLMBench) into a normalized knowledge graph (`.graphml`). Phases:

1. Seed/keyword expansion (rules + MeSH + optional LLM)
2. Multi-source literature retrieval (PubMed E-utilities, Crossref, bioRxiv/medRxiv, PubMed related-papers)
3. Scoring/filtering (SapBERT semantic + journal-impact)
4. Full-text fetch with priority `preprint JATS → preprint PDF → PMC XML → Crossref landing page → abstract fallback`, persisted in `paper_cache/`
5. Word-overlap chunking
6. Triple extraction via OpenAI-compatible Intern API (`intern-s1-pro`)
7. Entity/relation normalization + verification
8. Local-graph build → global compose → BGE-Large node fusion → small-component pruning → GraphML export

## Architecture

The driver `literature_pipeline.py` is a one-liner into `pubmed_graph.workflow.main`. Everything lives under the `pubmed_graph/` package:

- `workflow.py` — orchestrates `run_pipeline` (phases 1–3b), `run_triple_extraction`, and `run_graph_export`. CLI flags `--extract-triples` and `--export-graph` are additive on top of the base pipeline. The base pipeline always emits `chunks.jsonl`; extraction reads chunks and emits `raw_triples.jsonl` + `normalized_triples.jsonl`; graph export consumes (and re-normalizes) triples.
- `benchmark_seeds.py` — resolves seeds either from `seed_keywords` in config or from a benchmark CSV via `benchmark_seed_source`.
- `keyword_expansion.py` — rules, MeSH lookups, optional LLM expansion.
- `pubmed_client.py`, `crossref_client.py`, `biorxiv_client.py` — source clients. `retrieval.py` runs them in parallel, dedupes by DOI/PMID/title, merges metadata, and records `retrieval_source_breakdown`.
- `fulltext.py` — `PMCFullTextFetcher` orchestrates the full-text priority chain and writes through `paper_cache`. **Crossref landing pages are quality-filtered in `utils.py`** (boilerplate signals like `font-family`, `window.dataLayer`, `Log in`, `Your privacy choices`, etc.) — without this, raw publisher HTML/CSS leaks into chunks and extraction silently returns zero triples. Stale dirty cache entries are revalidated, not blindly trusted.
- `chunking.py` — word-window splitter with overlap.
- `embeddings.py` — `SapBERTScorer` and `BGELargeEmbedder`. Both support local model paths **and** a remote HTTP service (`/health`, `/score`, `/embed`) configured via `service_url` + `remote_model`.
- `triple_extraction.py` + `llm.py` — Intern/OpenAI-compatible chunk-level extractor (non-thinking mode). Reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` (aliases `INTERN_*`). Prompt at `prompts/triple_extraction.txt`.
- `normalize.py`, `entity_verification.py` — canonicalize entities/relations and optionally fact-check via PubMed/MeSH/LLM.
- `graph_ops.py` — `build_local_graphs → compose_global_graph → fuse_global_graph (BGE) → prune_small_components → export_graphml`.

Single-file scripts (`expand_keywords.py`, `retrieve_pubmed.py`, `fetch_fulltext.py`, `extract_triples.py`, `embed_and_fuse.py`, `build_graphs.py`) are thin entrypoints that exercise individual phases for debugging.

## Configuration

All runtime behavior comes from JSON configs at the repo root:

- `pipeline_config.example.json` — annotated reference / oncology demo
- `pipeline_config.workflow_test.json` — three-source smoke config
- `pipeline_config.benchmark.json`, `pipeline_config.benchmark.q50.json`, `pipeline_config.benchmark.debug.json` — ProteinLMBench benchmark variants

Key sections: `seed_keywords` or `benchmark_seed_source`, `keyword_expansion`, `pubmed`, `crossref`, `biorxiv`, `retrieval`, `scoring` (with nested `sapbert`), `embedding`, `fulltext`, `paper_cache`, `chunking`, `graph`, `openai_extraction`. `env_file` (default `.env`) is resolved relative to the config path.

Recommended denser-graph parameters (already used in current configs): `chunk_words=180`, `overlap_words=40`, `fusion_threshold=0.88`, `min_component_size=2`, `score_threshold≈0.03`.

## Running

Always source the extraction credentials first when extraction will run:

```bash
source ./triple_extraction_env.sh
```

Set `PYTHON` (or `PYTHON_BIN`) in the env to the python interpreter you want the run scripts to use; otherwise they default to `python` on `$PATH`.

End-to-end run (retrieval → fulltext → chunks → triples → graph):

```bash
python literature_pipeline.py \
  --config pipeline_config.workflow_test.json \
  --output-dir pipeline_outputs_workflow_test \
  --extract-triples \
  --chunk-limit 80 \
  --export-graph \
  --graph-output pipeline_outputs_workflow_test/global_graph.graphml
```

Other useful flags: `--dry-run` (phase 1 only — incompatible with `--extract-triples`), `--triples-output`, `--raw-triples-output`, `--graph-triples` (run graph export against a pre-existing JSONL).

ProteinLMBench full / sample runs are wrapped by `run_proteinlmbench_full_graph.sh [--sample] [--output-dir DIR] [--question-limit N] [--chunk-limit N] [--max-seeds N]`. It rewrites the base benchmark config into a tmp file with overridden `benchmark_seed_source` / `retrieval` knobs, health-checks the remote embedding service, then invokes the pipeline.

There is no test suite, linter, or build step. The "tests" are the smoke runs under `pipeline_outputs_*/` and `benchmark_runs/`, validated by inspecting `phase_summary.json`.

## Important gotchas

- **Landing-page quality filter is load-bearing.** If extraction returns 0 triples on a config that previously worked, first check `chunks.jsonl` for HTML/CSS noise (`html{`, `font-family`, `window.dataLayer`, `Log in`, `Your privacy choices`, `Springer Nature`). If present, the cleaning logic in `pubmed_graph/utils.py` regressed.
- **Remote embedding service** at `http://<embedding-host>:8765` may need `NO_PROXY=<host> no_proxy=<host>` to bypass an HTTP proxy. If the service is down, set `scoring.sapbert.enabled=false` and `embedding.enabled=false` to validate the rest of the chain.
- **`paper_cache/`** is keyed by DOI → PMCID → PMID → title-hash. `phase_summary.json` reports `cache_hits / cache_misses / pdf_cache_hits / xml_cache_hits / text_cache_hits / network_fetches_avoided`. Don't blow away the whole cache to fix a stale entry — invalid entries are revalidated automatically.
- **Output directories proliferate** (`pipeline_outputs_*`, `benchmark_runs/*`). Each run is a self-contained snapshot — `phase_summary.json` is the canonical entrypoint for inspecting it.
- **`question_generation/` is out of scope** for this guide; do not modify it as a side-effect of pipeline work.

## External references

- Workflow design notes (Chinese): `workflow_mds/pubmed_graph_workflow.md` — most current source of truth for phase semantics, recent fixes, and recommended configs.
- Pipeline overview: `README_literature_pipeline.md`.
- Local model paths: SapBERT and BGE-Large under `<embeddings-dir>/` — set the configs' `embedding.model_path` and `scoring.sapbert.model_path` to your local download.
