from __future__ import annotations

import argparse
import json

from pubmed_graph.embeddings import BGELargeEmbedder, NodeFusionEngine
from pubmed_graph.utils import load_env_file, load_json, resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test local BGE embedding loading and simple node fusion.")
    parser.add_argument("--config", default="pipeline_config.example.json")
    parser.add_argument("--text", nargs="*", default=["TP53", "tumor protein p53", "bladder cancer"])
    args = parser.parse_args()
    config = resolve_config_paths(load_json(args.config), args.config)
    load_env_file(config.get("env_file"))
    embedder = BGELargeEmbedder(config.get("embedding", {}))
    nodes = [{"text": text, "type": "Entity"} for text in args.text]
    groups = NodeFusionEngine(embedder=embedder, threshold=float(config.get("graph", {}).get("fusion_threshold", 0.9))).fuse(nodes)
    print(json.dumps([group.__dict__ for group in groups], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
