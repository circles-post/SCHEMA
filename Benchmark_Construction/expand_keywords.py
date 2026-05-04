from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from pubmed_graph.benchmark_seeds import resolve_seed_keywords
from pubmed_graph.keyword_expansion import KeywordExpansionEngine
from pubmed_graph.utils import ensure_dir, load_env_file, load_json, resolve_config_paths, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand seed biomedical keywords using local, MeSH, and reserved OpenAI hooks.")
    parser.add_argument("--config", default="pipeline_config.example.json")
    parser.add_argument("--output-dir", default="pipeline_outputs")
    args = parser.parse_args()
    config = resolve_config_paths(load_json(args.config), args.config)
    load_env_file(config.get("env_file"))
    output_dir = ensure_dir(args.output_dir)
    seed_keywords, benchmark_seed_summary = resolve_seed_keywords(config)
    write_json(Path(output_dir) / "resolved_seed_keywords.json", seed_keywords)
    if benchmark_seed_summary is not None:
        write_json(Path(output_dir) / "benchmark_seed_summary.json", benchmark_seed_summary)
    engine = KeywordExpansionEngine(config.get("keyword_expansion", {}))
    records, stats = engine.expand(seed_keywords)
    if benchmark_seed_summary is not None:
        stats = {**stats, "benchmark_seed_source": benchmark_seed_summary}
    write_json(Path(output_dir) / "expanded_keywords.json", [asdict(record) for record in records])
    write_json(Path(output_dir) / "keyword_stats.json", stats)


if __name__ == "__main__":
    main()
