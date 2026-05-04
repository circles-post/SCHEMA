from __future__ import annotations

import argparse
import json

from pubmed_graph.triple_extraction import run_triple_extraction
from pubmed_graph.utils import load_json, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk-level triple extraction with Intern non-thinking mode.")
    parser.add_argument("--config", default="pipeline_config.example.json")
    parser.add_argument("--chunks", default="pipeline_outputs/chunks.jsonl")
    parser.add_argument("--output", default="pipeline_outputs/normalized_triples.jsonl")
    parser.add_argument("--raw-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = resolve_config_paths(load_json(args.config), args.config)
    result = run_triple_extraction(
        config,
        args.chunks,
        args.output,
        limit=args.limit,
        raw_output_path=args.raw_output or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
