from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .biorxiv_client import BiorxivClient
from .crossref_client import CrossrefClient
from .chunking import split_documents_into_chunks
from .benchmark_seeds import resolve_seed_keywords
from .embeddings import BGELargeEmbedder, SapBERTScorer
from .fulltext import PMCFullTextFetcher
from .graph_ops import build_local_graphs, compose_global_graph, export_graphml, fuse_global_graph, load_triples, prune_small_components
from .keyword_expansion import KeywordExpansionEngine
from . import normalize
from .normalize import normalize_triple_records
from .ontology import Ontology
from .pubmed_client import PubMedClient
from .retrieval import LiteratureRetrievalEngine, LiteratureScorer
from .triple_extraction import run_triple_extraction
from .utils import ensure_dir, load_env_file, load_json, resolve_config_paths, write_json, write_jsonl


def default_phase_summary(
    config: dict[str, Any],
    keyword_stats: dict[str, Any],
    retrieval_stats: dict[str, Any],
    fulltext_stats: dict[str, Any],
    scored_records: list[Any],
    embedding_stats: dict[str, Any],
) -> dict[str, Any]:
    kept_records = [record for record in scored_records if record.kept]
    return {
        "project_name": config.get("project_name", "pubmed_graph_workflow"),
        "phase_1_keyword_expansion": keyword_stats,
        "phase_2_retrieval_and_filtering": {
            **retrieval_stats,
            "kept_papers": len(kept_records),
            "filtered_out_papers": len(scored_records) - len(kept_records),
        },
        "phase_3_fulltext": fulltext_stats,
        "phase_3b_embeddings": embedding_stats,
    }


def run_embedding_smoke_test(config: dict[str, Any], scored_records: list[Any], output_dir: Path) -> dict[str, Any]:
    sample_texts = []
    for record in scored_records[:3]:
        text = " ".join(part for part in [record.title, record.abstract] if part).strip()
        if text:
            sample_texts.append(text[:1000])
    stats: dict[str, Any] = {
        "sapbert_enabled": bool(config.get("scoring", {}).get("sapbert", {}).get("enabled", False)),
        "bge_enabled": bool(config.get("embedding", {}).get("enabled", False)),
    }
    if config.get("scoring", {}).get("sapbert", {}).get("enabled", False) and sample_texts:
        scorer = SapBERTScorer(config.get("scoring", {}).get("sapbert", {}))
        try:
            demo_score = scorer.score("cancer biomarker", scored_records[0]) if scored_records else 0.0
            stats["sapbert_demo_score"] = round(float(demo_score), 4)
        except Exception as exc:
            stats["sapbert_error"] = str(exc)
    if config.get("embedding", {}).get("enabled", False) and sample_texts:
        embedder = BGELargeEmbedder(config.get("embedding", {}))
        try:
            vectors = embedder.embed_texts(sample_texts[:2])
            stats["bge_vector_count"] = len(vectors)
            stats["bge_vector_dim"] = len(vectors[0]) if vectors else 0
        except Exception as exc:
            stats["bge_error"] = str(exc)
    write_json(output_dir / "embedding_smoke_test.json", stats)
    return stats


def run_pipeline(config: dict[str, Any], output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    load_env_file(config.get("env_file"))
    seed_keywords, benchmark_seed_summary = resolve_seed_keywords(config)
    if not seed_keywords:
        raise ValueError("Config must include a non-empty 'seed_keywords' list.")
    output_dir = ensure_dir(output_dir)
    write_json(output_dir / "resolved_seed_keywords.json", seed_keywords)
    if benchmark_seed_summary is not None:
        write_json(output_dir / "benchmark_seed_summary.json", benchmark_seed_summary)
    expansion_engine = KeywordExpansionEngine(config.get("keyword_expansion", {}))
    keyword_records, keyword_stats = expansion_engine.expand(seed_keywords)
    if benchmark_seed_summary is not None:
        keyword_stats = {**keyword_stats, "benchmark_seed_source": benchmark_seed_summary}
    accepted_keywords = [record.term for record in keyword_records if record.accepted]
    write_json(output_dir / "expanded_keywords.json", [asdict(record) for record in keyword_records])
    write_json(output_dir / "accepted_keywords.json", accepted_keywords)
    if dry_run:
        summary = {
            "project_name": config.get("project_name", "pubmed_graph_workflow"),
            "dry_run": True,
            "phase_1_keyword_expansion": keyword_stats,
        }
        write_json(output_dir / "phase_summary.json", summary)
        return summary
    pubmed_config = config.get("pubmed", {})
    biorxiv_config = config.get("biorxiv", {})
    crossref_config = config.get("crossref", {})
    pubmed_client = PubMedClient(api_key=pubmed_config.get("api_key"), email=pubmed_config.get("email"))
    biorxiv_client = BiorxivClient(biorxiv_config)
    crossref_client = CrossrefClient(crossref_config)
    retrieval_engine = LiteratureRetrievalEngine(
        config.get("retrieval", {}),
        pubmed_client,
        biorxiv_client=biorxiv_client,
        biorxiv_config=biorxiv_config,
        crossref_client=crossref_client,
        crossref_config=crossref_config,
    )
    papers, retrieval_stats = retrieval_engine.retrieve(accepted_keywords)
    scorer = LiteratureScorer(config.get("scoring", {}))
    scored_records = [scorer.score_paper(record.matched_keywords, record) for record in papers]
    write_json(output_dir / "pubmed_candidates.json", [asdict(record) for record in papers])
    write_json(output_dir / "pubmed_candidates_scored.json", [asdict(record) for record in scored_records])
    write_jsonl(output_dir / "pubmed_candidates.jsonl", [asdict(record) for record in papers])
    write_jsonl(output_dir / "pubmed_candidates_scored.jsonl", [asdict(record) for record in scored_records])
    fulltext_input = [record for record in scored_records if record.kept] if config.get("fulltext", {}).get("only_kept", True) else scored_records
    fulltext_cfg = dict(config.get("fulltext", {}))
    fulltext_cfg["pdf_parsing"] = dict(config.get("pdf_parsing", {}))
    fulltext_cfg["paper_cache"] = dict(config.get("paper_cache", {}))
    fulltext_cfg["sciverse"] = dict(config.get("sciverse", {}))
    fetcher = PMCFullTextFetcher(fulltext_cfg, pubmed_client, biorxiv_client=biorxiv_client)
    fulltext_records, fulltext_stats = fetcher.fetch(fulltext_input)
    write_jsonl(output_dir / "pmc_fulltext.jsonl", [asdict(record) for record in fulltext_records])
    chunk_records = split_documents_into_chunks(
        fulltext_records,
        chunk_words=config.get("chunking", {}).get("chunk_words", 250),
        overlap_words=config.get("chunking", {}).get("overlap_words", 40),
    )
    write_jsonl(output_dir / "chunks.jsonl", [asdict(record) for record in chunk_records])
    embedding_stats = run_embedding_smoke_test(config, scored_records, output_dir)

    ontology_stats = run_ontology_proposer_phase(
        config,
        chunk_records=[asdict(c) for c in chunk_records],
        output_dir=output_dir,
    )

    summary = default_phase_summary(
        config,
        keyword_stats,
        retrieval_stats,
        fulltext_stats,
        scored_records,
        embedding_stats,
    )
    if ontology_stats:
        summary["phase_3c_ontology_proposer"] = ontology_stats
    write_json(output_dir / "phase_summary.json", summary)
    return summary


def run_ontology_proposer_phase(
    config: dict[str, Any],
    chunk_records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Phase 3c: optional OntologyProposerAgent that produces ontology.run.yaml.

    Disabled by default. To enable, set in pipeline config:
        "ontology": {
            "agent_proposer_enabled": true,
            "sample_size": 30,
            "evidence_threshold": 2,
            "distinct_doc_threshold": 2,
            "require_grounding": true
        }
    Once enabled, the per-run ontology is hot-swapped into normalize.py via
    `normalize._set_active_ontology(...)`, so phase 4 (extraction) and the
    subsequent normalization use the extended tables. Phase 5 graph export
    in run_graph_export() will pick the same ontology automatically because
    normalize_triple_records reads the module-level constants.
    """
    onto_cfg = dict(config.get("ontology") or {})
    if not onto_cfg.get("agent_proposer_enabled", False):
        return None
    if not chunk_records:
        return {"status": "skipped", "reason": "no chunks"}

    from .external_kb import EntityCanonicalizer
    from .llm import InternChatClient
    from .ontology_proposer import OntologyProposerAgent

    extraction_cfg = dict(config.get("openai_extraction") or {})
    extraction_cfg.setdefault("model", "intern-s1-pro")
    extraction_cfg.setdefault("thinking_mode", False)
    extraction_cfg.setdefault("max_tokens", 1500)
    llm_client = InternChatClient(extraction_cfg)

    pubmed_cfg = dict(config.get("pubmed") or {})
    pubmed_client = PubMedClient(api_key=pubmed_cfg.get("api_key"), email=pubmed_cfg.get("email"))

    canonicalizer = None
    if onto_cfg.get("require_grounding", True):
        canonicalizer = EntityCanonicalizer(
            onto_cfg.get("canonicalizer") or {},
            pubmed_client=pubmed_client,
        )

    base_path = onto_cfg.get("base_path")
    base_ontology = Ontology.load(base_path) if base_path else Ontology.default()

    agent = OntologyProposerAgent(
        config=onto_cfg,
        llm_client=llm_client,
        canonicalizer=canonicalizer,
    )
    run_ontology = agent.propose(chunks=chunk_records, base=base_ontology, output_dir=output_dir)

    # hot-swap into normalize so phase 4 extraction + normalization see the extended ontology
    normalize._set_active_ontology(run_ontology)

    proposer_dir = output_dir / "ontology_proposer"
    return {
        "status": "ok",
        "base_version": base_ontology.version,
        "run_version": run_ontology.version,
        "added_entity_types": list(run_ontology._data.get("extensions_metadata", {}).get("added_entity_types", []) or []),
        "added_relations": list(run_ontology._data.get("extensions_metadata", {}).get("added_relations", []) or []),
        "added_aliases": list(run_ontology._data.get("extensions_metadata", {}).get("added_aliases", []) or []),
        "run_yaml": str(proposer_dir / "ontology.run.yaml"),
        "extensions_yaml": str(proposer_dir / "ontology.extensions.yaml"),
        "decisions_log": str(proposer_dir / "ontology_decisions.jsonl"),
        "rejected_log": str(proposer_dir / "ontology.rejected.jsonl"),
    }


def run_graph_export(triples_path: Path, output_graph_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    raw_triples = load_triples(triples_path)
    triples = normalize_triple_records(raw_triples, confidence_threshold=0.0)
    local_graphs = build_local_graphs(triples)
    global_graph = compose_global_graph(local_graphs)
    embedder = BGELargeEmbedder(config.get("embedding", {}))
    fused_graph = fuse_global_graph(
        global_graph,
        embedder=embedder,
        threshold=float(config.get("graph", {}).get("fusion_threshold", 0.9)),
    )
    pruned_graph = prune_small_components(
        fused_graph,
        min_component_size=int(config.get("graph", {}).get("min_component_size", 3)),
    )
    export_graphml(pruned_graph, output_graph_path)
    return {
        "input_triples": len(raw_triples),
        "normalized_triples": len(triples),
        "doc_graph_count": len(local_graphs),
        "global_nodes": int(pruned_graph.number_of_nodes()),
        "global_edges": int(pruned_graph.number_of_edges()),
        "graphml_path": str(output_graph_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PubMed graph workflow: keyword expansion, retrieval, PMC full text, chunking, extraction, and graph export."
    )
    parser.add_argument("--config", default="pipeline_config.example.json", help="Path to JSON config file.")
    parser.add_argument("--output-dir", default="pipeline_outputs", help="Directory for generated outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Only run keyword expansion.")
    parser.add_argument("--extract-triples", action="store_true", help="Run Intern extraction on generated chunks.")
    parser.add_argument("--triples-output", default="", help="Output JSONL path for normalized triples.")
    parser.add_argument("--raw-triples-output", default="", help="Optional JSONL path for raw LLM triples.")
    parser.add_argument("--chunk-limit", type=int, default=0, help="Optional chunk limit for extraction.")
    parser.add_argument("--export-graph", action="store_true", help="Build GraphML after extraction or from --graph-triples.")
    parser.add_argument(
        "--graph-triples",
        default="",
        help="Optional triples JSONL used for graph export. Defaults to extracted triples when available.",
    )
    parser.add_argument(
        "--graph-output",
        default="graphs/global_graph.graphml",
        help="GraphML output path when --export-graph is set.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = resolve_config_paths(load_json(args.config), args.config)
    output_dir = ensure_dir(args.output_dir)
    summary = run_pipeline(config, output_dir, dry_run=args.dry_run)

    extracted_triples_path: Path | None = None
    if args.extract_triples:
        if args.dry_run:
            raise ValueError("--extract-triples cannot be used with --dry-run because no chunks are generated.")
        extracted_triples_path = Path(args.triples_output) if args.triples_output else output_dir / "normalized_triples.jsonl"
        raw_output_path = Path(args.raw_triples_output) if args.raw_triples_output else output_dir / "raw_triples.jsonl"
        extraction_summary = run_triple_extraction(
            config,
            output_dir / "chunks.jsonl",
            extracted_triples_path,
            limit=args.chunk_limit,
            raw_output_path=raw_output_path,
        )
        summary["phase_4_triple_extraction"] = extraction_summary

    if args.export_graph:
        graph_triples_path = Path(args.graph_triples) if args.graph_triples else extracted_triples_path
        if graph_triples_path is None:
            graph_triples_path = output_dir / "normalized_triples.jsonl"
        graph_summary = run_graph_export(graph_triples_path, Path(args.graph_output), config)
        summary["phase_5_graph_export"] = graph_summary

    write_json(output_dir / "phase_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
