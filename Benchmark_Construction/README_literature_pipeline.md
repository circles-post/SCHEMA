# PubMed Graph Workflow

This folder now contains a reusable workflow scaffold for:

1. keyword expansion
2. PubMed retrieval and scoring
3. PMC full-text acquisition with abstract fallback
4. local SapBERT / BGE-Large loading hooks
5. chunk generation for LLM extraction
6. triple normalization
7. graph construction and GraphML export

## Current implementation status

Implemented now:
- keyword expansion through local heuristics and live MeSH lookup
- PubMed `ESearch`, `ESummary`, `EFetch`, and `ELink`
- PMC mapping and PMC XML full-text fetching
- abstract fallback when PMC full text is unavailable
- local SapBERT scorer interface with lexical fallback
- local BGE-Large embedding interface and node-fusion logic
- chunk generation
- Intern-compatible chunk-level triple extraction in non-thinking mode
- ontology-style postprocessing for entity aliases, entity types, and relation labels
- local/global graph build + GraphML export

Still remaining for future extension:
- richer ontology dictionaries for larger disease areas
- publisher PDF download orchestration
- stronger document-level alias resolution

## Local model paths already configured

- SapBERT: `/mnt/shared-storage-user/ai4good2-share/fengxinshun/embedding_models/SapBERT-from-PubMedBERT-fulltext`
- BGE-Large: `/mnt/shared-storage-user/ai4good2-share/fengxinshun/embedding_models/bge-large-en`

## Main entrypoints

- `literature_pipeline.py`: end-to-end workflow entrypoint
- `expand_keywords.py`: only keyword expansion
- `retrieve_pubmed.py`: PubMed retrieval + scoring
- `fetch_fulltext.py`: PMC full text fetch + abstract fallback
- `extract_triples.py`: Intern extraction + normalization
- `embed_and_fuse.py`: local embedding / fusion smoke test
- `build_graphs.py`: GraphML export from triples
- `pubmed_api.py`: low-level PubMed demo client

## Quick usage

Keyword expansion only:

```bash
python literature_pipeline.py --config pipeline_config.example.json --output-dir pipeline_outputs --dry-run
```

Run retrieval + PMC full text:

```bash
python literature_pipeline.py --config pipeline_config.workflow_test.json --output-dir pipeline_outputs_workflow_test
```

Run extraction and graph export in one command:

```bash
python literature_pipeline.py \
  --config pipeline_config.workflow_test.json \
  --output-dir pipeline_outputs_workflow_test \
  --extract-triples \
  --export-graph
```

Build GraphML from triples:

```bash
python build_graphs.py --config pipeline_config.example.json --triples pipeline_outputs/normalized_triples.jsonl --output graphs/global_graph.graphml
```

## Important notes

- The `new_graphcompass` environment already has the key runtime packages needed for embeddings and graph export.
- PubMed calls can run without `NCBI_API_KEY`, but using the key is still recommended.
- Intern-compatible API credentials are loaded from `.env`; see `.env.example`.
- Graph export now re-normalizes triples before node fusion, so it can consume either raw or normalized JSONL input.
