from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from pubmed_graph.fulltext import PMCFullTextFetcher
from pubmed_graph.models import PaperRecord
from pubmed_graph.pubmed_client import PubMedClient
from pubmed_graph.utils import ensure_dir, load_env_file, load_json, read_jsonl, resolve_config_paths, write_json, write_jsonl


def load_scored_records(path: str) -> list[PaperRecord]:
    data = read_jsonl(path) if path.endswith(".jsonl") else load_json(path)
    return [PaperRecord(**row) for row in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PMC full text when available and fall back to PubMed abstract.")
    parser.add_argument("--config", default="pipeline_config.example.json")
    parser.add_argument("--input", default="pipeline_outputs/pubmed_candidates_scored.json")
    parser.add_argument("--output-dir", default="pipeline_outputs")
    args = parser.parse_args()
    config = resolve_config_paths(load_json(args.config), args.config)
    load_env_file(config.get("env_file"))
    output_dir = ensure_dir(args.output_dir)
    records = load_scored_records(args.input)
    if config.get("fulltext", {}).get("only_kept", True):
        records = [record for record in records if record.kept]
    client = PubMedClient(api_key=config.get("pubmed", {}).get("api_key"), email=config.get("pubmed", {}).get("email"))
    fulltexts, stats = PMCFullTextFetcher(config.get("fulltext", {}), client).fetch(records)
    write_jsonl(Path(output_dir) / "pmc_fulltext.jsonl", [asdict(record) for record in fulltexts])
    write_json(Path(output_dir) / "fulltext_stats.json", stats)


if __name__ == "__main__":
    main()
