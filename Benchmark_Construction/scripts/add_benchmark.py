"""Add a new benchmark on top of an existing paper-KG run.

Flow:
  1. Read base run's normalized_triples.jsonl + accepted_keywords.json
  2. Read the new benchmark → list[BenchmarkItem]
  3. Derive seeds for the benchmark via resolve_seed_keywords
  4. Diff against base seeds → new_seeds
  5. Run a MINI pipeline for the new_seeds (Phase 2 retrieval → fulltext → chunks → triples)
     reusing the global paper_cache so overlap is free
  6. Run benchmark triple overlay on each BenchmarkItem
     (self-loop / real-semantic clique / skip) — using LLM per item
  7. Materialize images for items that contributed triples
  8. Union base + new-paper + benchmark triples → global_graph.graphml
  9. Write phase_summary.json with ablation-friendly counts

The base run's files are NEVER modified; everything lands under --output-dir.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pubmed_graph.benchmark_images import materialize_used_images
from pubmed_graph.benchmark_readers import READERS, list_readers
from pubmed_graph.benchmark_triples import run_benchmark_extraction
from pubmed_graph.graph_ops import (
    build_local_graphs,
    compose_global_graph,
    export_graphml,
    fuse_global_graph,
    load_triples,
    prune_small_components,
)
from pubmed_graph.embeddings import BGELargeEmbedder
from pubmed_graph.utils import ensure_dir, load_json, read_jsonl, resolve_config_paths, write_json
from pubmed_graph.workflow import run_pipeline, run_triple_extraction


DATASET_ROOTS = {
    "ProteinLMBench":  "<benchmark-data>",
    "PathVQA":         "<benchmark-data>/path-vqa",
    "MedXpertQA_MM":   "<benchmark-data>/MedXpertQA",
    "SLAKE_EN":        "<benchmark-data>/SLAKE-vqa-english",
    "SLAKE_Bilingual": "<benchmark-data>/SLAKE-bilingual",
    "MedQBench":       "<benchmark-data>/MedQ-Bench",
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extend an existing paper-KG run with a new benchmark overlay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--add", required=True, choices=list_readers(),
                   help="Benchmark name to add")
    p.add_argument("--base-run", required=True,
                   help="Directory of the base run (must contain normalized_triples.jsonl)")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the extended run")
    p.add_argument("--config", default=str(ROOT / "pipeline_config.benchmark.json"),
                   help="Pipeline config for the incremental retrieval phase")
    p.add_argument("--dataset-root", default="",
                   help="Override default dataset root (see DATASET_ROOTS)")
    p.add_argument("--split", default="test")
    p.add_argument("--question-limit", type=int, default=0,
                   help="Cap benchmark items processed (0=all)")
    p.add_argument("--chunk-limit", type=int, default=0,
                   help="Cap chunks fed to Phase-4 extraction on new seeds (0=all)")
    p.add_argument("--skip-retrieval", action="store_true",
                   help="Do NOT run paper retrieval for new seeds (overlay-only mode)")
    p.add_argument("--skip-overlay", action="store_true",
                   help="Do NOT run the benchmark overlay (new-seed retrieval only)")
    p.add_argument("--skip-images", action="store_true",
                   help="Skip image materialization")
    p.add_argument("--no-llm-entities", action="store_true",
                   help="Use regex-only entity extraction on the QA (no LLM seed pass)")
    p.add_argument("--confidence-threshold", type=float, default=0.6,
                   help="Min confidence for LLM-extracted benchmark triples")
    p.add_argument("--max-workers", type=int, default=6,
                   help="Number of concurrent LLM workers for benchmark overlay "
                        "(each item uses 2 API calls; 6 ≈ 12 inflight)")
    p.add_argument("--no-llm-seed-derivation", action="store_true",
                   help="Skip the LLM pass when deriving seeds from the benchmark "
                        "for incremental retrieval (regex-only, fastest but misses "
                        "lowercase/casual answers like PathVQA's 'adenocarcinoma').")
    p.add_argument("--seed-derivation-items", type=int, default=300,
                   help="How many benchmark items to scan for seed derivation "
                        "(the LLM pass batches this by --seed-derivation-batch-size).")
    p.add_argument("--seed-derivation-batch-size", type=int, default=20,
                   help="Batch size for LLM seed derivation calls.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite non-empty --output-dir")
    p.add_argument("--fusion-threshold", type=float, default=-1.0,
                   help="Override graph fusion threshold (default: from config)")
    p.add_argument("--min-component-size", type=int, default=-1,
                   help="Override min_component_size (default: from config)")
    return p


def _guard_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SystemExit(f"[FAIL] output dir is not empty: {path}\n"
                             f"       Pass --force to overwrite, or pick a fresh --output-dir.")
        print(f"[WARN] --force: overwriting existing output dir {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_base(base: Path) -> tuple[Path, set[str]]:
    triples_path = base / "normalized_triples.jsonl"
    if not triples_path.exists():
        raise SystemExit(f"[FAIL] base run missing normalized_triples.jsonl: {triples_path}")
    seeds_path = base / "accepted_keywords.json"
    if seeds_path.exists():
        accepted = set(load_json(seeds_path))
    else:
        accepted = set()
    return triples_path, accepted


def _resolve_dataset_root(benchmark: str, override: str) -> Path:
    if override:
        return Path(override)
    root = DATASET_ROOTS.get(benchmark)
    if not root:
        raise SystemExit(f"[FAIL] no default dataset root for {benchmark}; pass --dataset-root")
    return Path(root)


def _derive_benchmark_seeds(
    items,
    limit_for_seeds: int = 300,
    use_llm: bool = True,
    llm_batch_size: int = 20,
    llm_max_terms: int = 15,
) -> list[str]:
    """Regex + LLM seed extraction from QA text for incremental retrieval.

    Mirrors the hybrid path that benchmark_seeds.resolve_seed_keywords uses
    for ProteinLMBench: regex gives +1 frequency weight, LLM gives +2. This
    is necessary because benchmarks with casual phrasing (PathVQA's
    "What is this?" → "adenocarcinoma") don't match ProteinLMBench-tuned
    regex patterns, so without the LLM pass benchmark_seeds_new.json comes
    back empty and incremental retrieval is silently skipped.
    """
    from pubmed_graph.benchmark_seeds import (
        _canonical_seed_key,
        _canonicalize_seed_candidate,
        _extract_from_question,
        _is_useful_candidate,
        _llm_extract_batch_seed_keywords,
        _split_compound_candidate,
    )

    sliced = items[:limit_for_seeds] if limit_for_seeds > 0 else items
    qa_texts = [(item.question + "\n" + item.answer).strip() for item in sliced]

    counts: dict[str, int] = {}
    examples: dict[str, str] = {}

    def _accept(candidate_raw: str, weight: int) -> None:
        for candidate in _split_compound_candidate(candidate_raw):
            canonical = _canonicalize_seed_candidate(candidate)
            if not _is_useful_candidate(canonical):
                continue
            key = _canonical_seed_key(canonical)
            if not key:
                return
            counts[key] = counts.get(key, 0) + weight
            if key not in examples or len(canonical) < len(examples[key]):
                examples[key] = canonical

    for text in qa_texts:
        for candidate in _extract_from_question(text):
            _accept(candidate, weight=1)

    if use_llm and qa_texts:
        from pubmed_graph.llm import InternChatClient
        try:
            client = InternChatClient({
                "model": "intern-s1-pro",
                "thinking_mode": False,
                "temperature": 0.0,
                "max_tokens": 1200,
            })
        except Exception as exc:
            print(f"[WARN] LLM seed derivation disabled ({exc}); regex-only.")
            client = None
        if client is not None:
            batches_ok = 0
            batches_total = 0
            for start in range(0, len(qa_texts), llm_batch_size):
                batches_total += 1
                batch = qa_texts[start:start + llm_batch_size]
                try:
                    raw = _llm_extract_batch_seed_keywords(client, batch, max_terms=llm_max_terms)
                except Exception as exc:
                    print(f"[WARN] LLM seed batch {start}-{start+len(batch)} failed: {exc}")
                    continue
                batches_ok += 1
                for term in raw:
                    _accept(term, weight=2)
            print(f"  [seed-derive] regex over {len(qa_texts)} items; "
                  f"LLM batches: {batches_ok}/{batches_total} ok "
                  f"(batch_size={llm_batch_size}, max_terms={llm_max_terms})")

    ranked = sorted(examples.items(), key=lambda kv: (-counts[kv[0]], len(kv[1]), kv[1].lower()))
    return [term for _, term in ranked]


def _build_incremental_config(base_config_path: Path, new_seeds: list[str],
                               inherit_limits: dict | None = None) -> dict:
    cfg = resolve_config_paths(load_json(base_config_path), str(base_config_path))
    cfg["seed_keywords"] = new_seeds
    if cfg.get("benchmark_seed_source"):
        cfg["benchmark_seed_source"] = dict(cfg["benchmark_seed_source"])
        cfg["benchmark_seed_source"]["enabled"] = False
    cfg["ontology"] = {"agent_proposer_enabled": False}
    if inherit_limits:
        cfg.setdefault("retrieval", {}).update(inherit_limits)
    return cfg


def _run_incremental_retrieval(config: dict, inc_dir: Path, chunk_limit: int) -> Path:
    inc_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(config, inc_dir, dry_run=False)
    triples_out = inc_dir / "normalized_triples.jsonl"
    raw_out = inc_dir / "raw_triples.jsonl"
    run_triple_extraction(
        config,
        inc_dir / "chunks.jsonl",
        triples_out,
        limit=chunk_limit,
        raw_output_path=raw_out,
    )
    return triples_out


def _concat_jsonl(sources: list[Path], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with destination.open("w", encoding="utf-8") as out:
        for src in sources:
            if not src or not src.exists():
                continue
            with src.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        total += 1
    return total


def _build_and_export_graph(triples_path: Path, output_graphml: Path,
                             graph_cfg: dict, embedding_cfg: dict) -> dict:
    triples = load_triples(triples_path)
    local_graphs = build_local_graphs(triples)
    global_graph = compose_global_graph(local_graphs)
    embedder = BGELargeEmbedder(embedding_cfg or {})
    fused = fuse_global_graph(
        global_graph,
        embedder=embedder,
        threshold=float(graph_cfg.get("fusion_threshold", 0.9)),
    )
    pruned = prune_small_components(
        fused,
        min_component_size=int(graph_cfg.get("min_component_size", 3)),
    )
    export_graphml(pruned, output_graphml)
    return {
        "input_triples": len(triples),
        "doc_graph_count": len(local_graphs),
        "global_nodes": int(pruned.number_of_nodes()),
        "global_edges": int(pruned.number_of_edges()),
        "graphml_path": str(output_graphml),
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    t0 = time.time()

    base = Path(args.base_run).resolve()
    out = Path(args.output_dir).resolve()
    _guard_output_dir(out, args.force)

    base_triples_path, base_seeds = _load_base(base)
    dataset_root = _resolve_dataset_root(args.add, args.dataset_root)

    print("=" * 60)
    print(f"base run        : {base}")
    print(f"base triples    : {base_triples_path}")
    print(f"base seeds      : {len(base_seeds)}")
    print(f"benchmark       : {args.add}   split={args.split}   root={dataset_root}")
    print(f"output dir      : {out}")
    print(f"question_limit  : {args.question_limit or 'all'}")
    print(f"skip_retrieval  : {args.skip_retrieval}   skip_overlay={args.skip_overlay}")
    print("=" * 60)

    reader = READERS[args.add]
    items = list(reader(dataset_root, split=args.split, question_limit=args.question_limit))
    print(f"[step 1] loaded {len(items)} benchmark items")
    if not items:
        raise SystemExit("[FAIL] no benchmark items loaded")

    cfg = resolve_config_paths(load_json(args.config), args.config)
    if args.fusion_threshold >= 0:
        cfg.setdefault("graph", {})["fusion_threshold"] = args.fusion_threshold
    if args.min_component_size >= 0:
        cfg.setdefault("graph", {})["min_component_size"] = args.min_component_size

    new_paper_triples_path: Path | None = None
    if not args.skip_retrieval:
        bench_seeds = _derive_benchmark_seeds(
            items,
            limit_for_seeds=args.seed_derivation_items,
            use_llm=not args.no_llm_seed_derivation,
            llm_batch_size=args.seed_derivation_batch_size,
        )
        new_seeds = sorted(set(bench_seeds) - base_seeds)
        (out / "benchmark_seeds_all.json").write_text(
            json.dumps(bench_seeds[:500], ensure_ascii=False, indent=2))
        (out / "benchmark_seeds_new.json").write_text(
            json.dumps(new_seeds, ensure_ascii=False, indent=2))
        print(f"[step 2] benchmark seeds: total={len(bench_seeds)}  new={len(new_seeds)}")
        if new_seeds:
            inc_cfg = _build_incremental_config(Path(args.config), new_seeds)
            inc_dir = out / "_incremental"
            new_paper_triples_path = _run_incremental_retrieval(inc_cfg, inc_dir, args.chunk_limit)
            if new_paper_triples_path.exists():
                shutil.copy(new_paper_triples_path, out / "new_paper_triples.jsonl")
                new_paper_triples_path = out / "new_paper_triples.jsonl"
        else:
            print("[step 2] no new seeds; skipping incremental retrieval")

    benchmark_summary: dict = {}
    benchmark_triples_path: Path | None = None
    used_items: list = []
    if not args.skip_overlay:
        overlay_cfg = dict(cfg.get("benchmark_overlay") or {})
        if args.no_llm_entities:
            overlay_cfg["llm_entity_extraction"] = False
        overlay_cfg["relation_confidence_threshold"] = args.confidence_threshold
        overlay_cfg.setdefault("verbose", False)
        cfg = {**cfg, "benchmark_overlay": overlay_cfg}
        print(f"[step 3] running benchmark overlay on {len(items)} items "
              f"(max_workers={args.max_workers})")
        benchmark_summary = run_benchmark_extraction(
            items, cfg, out, max_workers=args.max_workers,
        )
        benchmark_triples_path = Path(benchmark_summary["triples_path"])
        used_items = benchmark_summary.pop("used_items", [])
        print(f"  contributed {benchmark_summary['items_contributing']} items, "
              f"{benchmark_summary['triples_emitted']} triples "
              f"(self_loops={benchmark_summary['self_loops']}, "
              f"semantic={benchmark_summary['semantic_edges']})")

    if used_items and not args.skip_images:
        img_root = out / "benchmark_images"
        print(f"[step 4] materializing {len(used_items)} images → {img_root}")
        resolved = materialize_used_images(used_items, img_root)
        (out / "benchmark_image_index.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2))
        _attach_image_paths(benchmark_triples_path, resolved)

    merged_path = out / "merged_triples.jsonl"
    merged_count = _concat_jsonl(
        [base_triples_path, new_paper_triples_path, benchmark_triples_path],
        merged_path,
    )
    print(f"[step 5] merged triples: {merged_count} → {merged_path}")

    graph_out = out / "global_graph.graphml"
    print(f"[step 6] building graph → {graph_out}")
    graph_summary = _build_and_export_graph(
        merged_path, graph_out,
        graph_cfg=cfg.get("graph", {}),
        embedding_cfg=cfg.get("embedding", {}),
    )

    summary = {
        "benchmark": args.add,
        "split": args.split,
        "base_run": str(base),
        "output_dir": str(out),
        "base_triples":       _count_jsonl(base_triples_path),
        "new_paper_triples":  _count_jsonl(new_paper_triples_path),
        "benchmark_triples":  _count_jsonl(benchmark_triples_path),
        "merged_triples":     merged_count,
        "benchmark_overlay":  benchmark_summary,
        "graph":              graph_summary,
        "elapsed_seconds":    round(time.time() - t0, 1),
    }
    write_json(out / "phase_summary.json", summary)
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 60)
    return 0


def _count_jsonl(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _attach_image_paths(triples_path: Path, resolved: dict[str, str]) -> None:
    if not triples_path or not triples_path.exists():
        return
    lines: list[str] = []
    with triples_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            meta = t.get("meta") or {}
            qid = meta.get("question_id")
            if qid and qid in resolved:
                meta["image_path"] = resolved[qid]
                t["meta"] = meta
            lines.append(json.dumps(t, ensure_ascii=False))
    triples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
