from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from pubmed_graph.keyword_expansion import KeywordExpansionEngine
from pubmed_graph.pubmed_client import PubMedClient
from pubmed_graph.retrieval import LiteratureRetrievalEngine, LiteratureScorer
from pubmed_graph.utils import ensure_dir, load_env_file, load_json, resolve_config_paths, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve PubMed papers and score them with SapBERT or lexical fallback.")
    parser.add_argument("--config", default="pipeline_config.example.json")
    parser.add_argument("--output-dir", default="pipeline_outputs")
    args = parser.parse_args()
    config = resolve_config_paths(load_json(args.config), args.config)
    load_env_file(config.get("env_file"))
    output_dir = ensure_dir(args.output_dir)
    keyword_records, _ = KeywordExpansionEngine(config.get("keyword_expansion", {})).expand(config.get("seed_keywords", []))
    keywords = [record.term for record in keyword_records if record.accepted]
    client = PubMedClient(api_key=config.get("pubmed", {}).get("api_key"), email=config.get("pubmed", {}).get("email"))
    papers, stats = LiteratureRetrievalEngine(config.get("retrieval", {}), client).retrieve(keywords)
    scorer = LiteratureScorer(config.get("scoring", {}))
    scored = [scorer.score_paper(record.matched_keywords, record) for record in papers]
    write_json(Path(output_dir) / "pubmed_candidates.json", [asdict(record) for record in papers])
    write_json(Path(output_dir) / "pubmed_candidates_scored.json", [asdict(record) for record in scored])
    write_jsonl(Path(output_dir) / "pubmed_candidates_scored.jsonl", [asdict(record) for record in scored])
    write_json(Path(output_dir) / "retrieval_stats.json", stats)


if __name__ == "__main__":
    main()
