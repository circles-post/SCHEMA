from __future__ import annotations

import argparse
import json

from pubmed_graph.utils import load_json, resolve_config_paths
from pubmed_graph.workflow import run_graph_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local graphs, fuse nodes, prune components, and export GraphML.")
    parser.add_argument("--config", default="pipeline_config.example.json")
    parser.add_argument("--triples", required=True)
    parser.add_argument("--output", default="graphs/global_graph.graphml")
    args = parser.parse_args()
    config = resolve_config_paths(load_json(args.config), args.config)
    result = run_graph_export(args.triples, args.output, config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
