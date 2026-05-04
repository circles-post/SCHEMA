"""Hallucination-detection pipeline CLI.

Per-model flow:
  1. Load joined SampleRecords (dataset ⋈ trajectory ⋈ answers ⋈ scored_results)
     via ``halu.io.load_joined``.
  2. For each sample, project its trajectory → list[Step] via
     ``halu.trajectory.steps_from_trajectory``.
  3. Concurrently extract claims per step (async semaphore = --extractor-concurrency).
  4. Bucket claims by normalized concept (``halu.extractor.ConceptClusterer``).
  5. For each bucket: gather evidence via short-circuit chain, then judge the
     bucket with a semaphore = --judge-concurrency. Evidence-gathering is
     async; judge is sync behind ``asyncio.to_thread``.
  6. Compute per-sample HR/HS/HF, aggregate, write halu_results.jsonl + halu_summary.json.

Phase-1 caveat: layers 2-4 of the evidence chain are stubbed to return []; only
``supporting_chunk`` matches. Concepts not mentioned in any supporting chunk
end up with empty evidence and are verdict'd as "unverifiable" (short-circuit
in ``BucketJudge``, no LLM call).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
_PARENT = _EVAL_DIR.parent
for p in (str(_PARENT), str(_EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pubmed_graph.llm import InternChatClient

from evaluation.halu.aggregator import aggregate_halu, compute_sample_metrics
from evaluation.halu.evidence import gather_evidence_for_bucket
from evaluation.halu.extractor import ClaimExtractor, ConceptClusterer
from evaluation.halu.graph_kb import GraphKB
from evaluation.halu.io import discover_model_dirs, load_joined, write_results
from evaluation.halu.judge import BucketJudge
from evaluation.halu.trajectory import steps_from_trajectory
from evaluation.halu.types import HaluResult, SampleRecord


# ---------------------------------------------------------------------------
# Proxy fix-up (same as runner.py — don't let the Boyue host get bypassed)
# ---------------------------------------------------------------------------
def _fix_no_proxy_for_host(host: str) -> None:
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        if not current:
            continue
        cleaned = re.sub(r"(\d+\.\d+\.\d+\.\d+)/\d+", r"\1", current)
        entries = [e.strip() for e in cleaned.split(",") if e.strip()]
        entries = [e for e in entries if e != host]
        os.environ[var] = ",".join(entries)


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------
async def process_sample(
    record: SampleRecord,
    *,
    extractor: ClaimExtractor,
    clusterer: ConceptClusterer,
    judge: BucketJudge,
    judge_model: str,
    evidence_chain: list[str],
    extractor_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
    graph_kb: GraphKB | None = None,
) -> HaluResult:
    bw = (record.sample.get("metadata") or {}).get("benchmark_weight") or {}
    grounding = record.sample.get("grounding") or {}
    result = HaluResult(
        sample_id=record.sample_id,
        model=record.model,
        question_type=record.question_type,
        tier=str(bw.get("tier") or "not_tagged"),
        weight=float(bw.get("weight") or 1.0),
        corroboration_status=str(grounding.get("corroboration_status") or "not_requested"),
        evidence_strength=str(grounding.get("evidence_strength") or "unknown"),
        is_correct=record.is_correct,
    )

    steps = steps_from_trajectory(record.sample_id, record.trajectory_messages)
    if not steps:
        result.error = "no_agent_steps"
        return result

    question = str(record.sample.get("question", ""))

    # ---- Phase A: extract claims per step (concurrent) ----
    async def _extract(step):
        async with extractor_sem:
            return await extractor.extract(
                record.sample_id,
                step,
                question=question,
                model_being_tested=record.model,
            )

    claims_per_step = await asyncio.gather(*(_extract(s) for s in steps))
    all_claims = [c for group in claims_per_step for c in group]
    if not all_claims:
        result.error = "no_factual_claims_extracted"
        return result

    # ---- Phase B: bucket by concept ----
    buckets = clusterer.bucket(record.sample_id, all_claims)

    # ---- Phase C: gather evidence + judge (concurrent per bucket) ----
    async def _judge_one(bucket):
        ev, src = await gather_evidence_for_bucket(
            bucket, record.sample, chain=evidence_chain, graph_kb=graph_kb
        )
        bucket.evidence = ev
        bucket.evidence_source_used = src
        if graph_kb is not None:
            try:
                bucket.concept_weight, bucket.concept_type = await asyncio.to_thread(
                    graph_kb.concept_info, bucket.canonical_concept
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[halu.cli] concept_info failed on {bucket.canonical_concept}: {type(exc).__name__}: {exc}")
        async with judge_sem:
            judged = await judge.judge(bucket, judge_model)
        # Propagate bucket attributes to each JudgedClaim for downstream slicing.
        for jc in judged:
            jc.concept_weight = bucket.concept_weight
            jc.concept_type = bucket.concept_type
        return judged

    judged_per_bucket = await asyncio.gather(*(_judge_one(b) for b in buckets))
    result.concept_buckets = buckets
    result.judged_claims = [jc for group in judged_per_bucket for jc in group]

    # ---- Phase D: per-sample metrics ----
    compute_sample_metrics(result)
    return result


# ---------------------------------------------------------------------------
# Per-model orchestrator
# ---------------------------------------------------------------------------
async def process_model(
    model_dir: Path,
    dataset: list[dict[str, Any]],
    args: argparse.Namespace,
    extractor_client: InternChatClient,
    judge_client: InternChatClient,
    cache_dir: Path,
    graph_kb: GraphKB | None = None,
) -> dict[str, Any]:
    model = model_dir.name
    print(f"\n=== halu: model={model} ===")
    records = load_joined(model_dir, dataset, only_errors=not args.include_correct)
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    if not records:
        print(f"  [{model}] no samples to analyze (all correct under --include-correct=False?)")
        return {"model": model, "n_samples": 0, "aggregate": {}}

    extractor = ClaimExtractor(
        extractor_client,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
    )
    clusterer = ConceptClusterer(cluster_threshold=args.concept_cluster_threshold)
    judge = BucketJudge(
        judge_client,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
    )

    ext_sem = asyncio.Semaphore(args.extractor_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    chain = [c.strip() for c in args.evidence_chain.split(",") if c.strip()]
    chain = ["supporting_chunk"] + chain  # always start with layer 1

    async def _run(rec):
        try:
            return await process_sample(
                rec,
                extractor=extractor,
                clusterer=clusterer,
                judge=judge,
                judge_model=args.judge_model,
                evidence_chain=chain,
                extractor_sem=ext_sem,
                judge_sem=judge_sem,
                graph_kb=graph_kb,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{model}] {rec.sample_id} FAILED: {type(exc).__name__}: {exc}")
            return HaluResult(
                sample_id=rec.sample_id,
                model=model,
                question_type=rec.question_type,
                is_correct=rec.is_correct,
                error=f"{type(exc).__name__}:{exc}",
            )

    results = await asyncio.gather(*(_run(r) for r in records))

    # Print a quick per-sample summary line
    for i, r in enumerate(results, 1):
        marker = "✗" if r.HF else "✓"
        print(
            f"  [{model}] {i}/{len(results)} {marker} {r.sample_id}  "
            f"type={r.question_type} tier={r.tier} n_claims={r.n_claims} "
            f"HR={r.HR:.2f} HS={r.HS:.2f} HF={r.HF}"
            + (f"  [{r.error}]" if r.error else "")
        )

    stats = aggregate_halu(results)
    model_out_dir = Path(args.output_dir) / model
    write_results(model_out_dir, results, {
        "model": model,
        "n_samples": len(results),
        "include_correct": args.include_correct,
        "evidence_chain": chain,
        "extractor_model": args.extractor_model,
        "judge_model": args.judge_model,
        "aggregate": stats,
    })
    ov = stats.get("overall", {})
    print(
        f"  -> {model}  n={ov.get('n_samples',0)}  "
        f"HR_macro={ov.get('HR_macro',0):.3f}  "
        f"HS_macro={ov.get('HS_macro',0):.3f}  "
        f"HS_w_micro={ov.get('HS_weighted_micro',0):.3f}  "
        f"HF_rate={ov.get('HF_rate',0):.3f}  "
        f"n_refuted={ov.get('n_refuted',0)}/{ov.get('n_claims',0)}"
    )
    return {"model": model, "n_samples": len(results), "aggregate": stats}


# ---------------------------------------------------------------------------
# Argparse + main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Claim-level hallucination detection on evaluation.runner outputs."
    )
    ap.add_argument("--runs-dir", required=True, help="Directory produced by evaluation.runner (contains <model>/trajectory.jsonl).")
    ap.add_argument("--dataset", required=True, help="Dataset JSONL (same file passed to runner).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--models", default="", help="Comma-separated subdir names; default = all model dirs with trajectories.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-correct", action="store_true", help="Default off: only analyze error trajectories.")
    ap.add_argument("--env-file", default=str(_EVAL_DIR / ".env"))

    ap.add_argument("--extractor-model", default="",
                    help="LLM model id for claim extraction. Default = $HALU_EXTRACTOR_MODEL or intern-s1-pro.")
    ap.add_argument("--extractor-base-url", default="",
                    help="OpenAI-compatible base URL for extractor. Default = $INTERN_BASE_URL.")
    ap.add_argument("--extractor-api-key-env", default="INTERN_API_KEY",
                    help="Env var holding the extractor API key. Default = INTERN_API_KEY.")
    ap.add_argument("--judge-model", default="",
                    help="LLM model id for the bucket judge. Default = $HALU_JUDGE_MODEL or intern-s1-pro.")
    ap.add_argument("--judge-base-url", default="",
                    help="OpenAI-compatible base URL for judge. Default = $INTERN_BASE_URL.")
    ap.add_argument("--judge-api-key-env", default="INTERN_API_KEY",
                    help="Env var holding the judge API key. Default = INTERN_API_KEY.")
    ap.add_argument("--extractor-concurrency", type=int, default=4)
    ap.add_argument("--judge-concurrency", type=int, default=10)
    ap.add_argument("--evidence-chain", default="graph,web,literature",
                    help="Layers tried AFTER supporting_chunk, in order (phase-1: stubs).")
    ap.add_argument("--concept-cluster-threshold", type=float, default=0.85)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--allow-self-judge", action="store_true",
                    help="Override the guard that blocks judging a model with itself.")

    # Layer-2 graph knowledge base.
    ap.add_argument("--graph", default="",
                    help="Path to global_graph.graphml. Auto-detected from --dataset parent if omitted.")
    ap.add_argument("--bge-service-url", default="",
                    help="Remote BGE embedding service URL. Defaults to $EMBEDDING_SERVICE_URL.")
    ap.add_argument("--bge-remote-model", default="bge")
    ap.add_argument("--bge-cosine-floor", type=float, default=0.6,
                    help="Below this cosine the nearest-node lookup returns None (graph miss).")
    return ap.parse_args()


def _make_client(
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float = 120.0,
) -> InternChatClient:
    return InternChatClient(
        {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": 0.0,
            "max_tokens": 1500,
            "request_timeout": timeout,
        }
    )


async def _main_async() -> int:
    args = _parse_args()
    load_dotenv(args.env_file, override=True)

    # InternS1 is the default LLM endpoint for hallucination extractor + judge.
    intern_base = os.environ.get("INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/").strip()
    ext_base = (args.extractor_base_url or intern_base).strip()
    jdg_base = (args.judge_base_url or intern_base).strip()
    ext_key = os.environ.get(args.extractor_api_key_env, "").strip()
    jdg_key = os.environ.get(args.judge_api_key_env, "").strip()
    if not ext_key:
        print(
            f"ERROR: extractor api key env {args.extractor_api_key_env} is empty. "
            f"Set it in {args.env_file} (e.g. INTERN_API_KEY=eyJ0eXBl...).",
            file=sys.stderr,
        )
        return 2
    if not jdg_key:
        print(
            f"ERROR: judge api key env {args.judge_api_key_env} is empty. "
            f"Set it in {args.env_file} (e.g. INTERN_API_KEY=eyJ0eXBl...).",
            file=sys.stderr,
        )
        return 2
    ext_model = (args.extractor_model or os.environ.get("HALU_EXTRACTOR_MODEL", "intern-s1-pro")).strip()
    jdg_model = (args.judge_model or os.environ.get("HALU_JUDGE_MODEL", "intern-s1-pro")).strip()

    # Proxy fix for Boyue host (same as runner.py)
    for url in {ext_base, jdg_base}:
        m = re.search(r"//([^:/]+)", url)
        if m:
            _fix_no_proxy_for_host(m.group(1))

    runs_dir = Path(args.runs_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Discover model dirs
    all_model_dirs = discover_model_dirs(runs_dir)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        model_dirs = [d for d in all_model_dirs if d.name in wanted]
        if not model_dirs:
            raise FileNotFoundError(
                f"No model dir matching --models={args.models} under {runs_dir}. "
                f"Available: {[d.name for d in all_model_dirs]}"
            )
    else:
        model_dirs = all_model_dirs

    # Self-judge guard
    if not args.allow_self_judge:
        evaluated = {d.name for d in model_dirs}
        # Normalize judge_model the same way runner safe_name does to compare
        safe_jdg = re.sub(r"[^A-Za-z0-9._-]+", "_", jdg_model).strip("_")
        if safe_jdg in evaluated:
            print(
                f"ERROR: judge_model={jdg_model} would be judging itself (it's one of the "
                f"evaluated models under {runs_dir}). Pass --allow-self-judge to override.",
                file=sys.stderr,
            )
            return 3

    # Load dataset ONCE
    dataset: list[dict[str, Any]] = []
    with open(args.dataset, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    print(
        f"[halu] runs_dir={runs_dir}  dataset_n={len(dataset)}  models={[d.name for d in model_dirs]}  "
        f"extractor={ext_model}  judge={jdg_model}  include_correct={args.include_correct}  "
        f"chain=supporting_chunk,{args.evidence_chain}"
    )

    ext_client = _make_client(model=ext_model, base_url=ext_base, api_key=ext_key)
    jdg_client = _make_client(model=jdg_model, base_url=jdg_base, api_key=jdg_key)

    # Resolve graph_kb if the chain includes "graph".
    graph_kb: GraphKB | None = None
    chain_preview = [c.strip().lower() for c in args.evidence_chain.split(",") if c.strip()]
    if "graph" in chain_preview:
        graphml = Path(args.graph).resolve() if args.graph else None
        if graphml is None:
            # Auto-detect: sibling of dataset, or parent/parent.
            ds = Path(args.dataset).resolve()
            for candidate in (ds.parent / "global_graph.graphml", ds.parent.parent / "global_graph.graphml"):
                if candidate.is_file():
                    graphml = candidate
                    break
        if graphml is None or not graphml.is_file():
            print(
                "WARN: evidence chain includes 'graph' but no --graph path was found. "
                "Skipping graph layer.",
                file=sys.stderr,
            )
        else:
            bge_url = (args.bge_service_url or os.environ.get("EMBEDDING_SERVICE_URL", "")).strip()
            bge_model_path = os.environ.get("BGE_MODEL_PATH", "").strip() or None
            print(f"[halu] graph_kb: {graphml}  bge_service_url={bge_url or '(local)'}")
            graph_kb = GraphKB(
                graphml,
                cache_dir=cache_dir,
                bge_service_url=bge_url or None,
                bge_remote_model=args.bge_remote_model,
                bge_model_path=bge_model_path,
                cosine_floor=args.bge_cosine_floor,
            )

    all_summaries = []
    for d in model_dirs:
        summary = await process_model(
            d,
            dataset,
            args,
            extractor_client=ext_client,
            judge_client=jdg_client,
            cache_dir=cache_dir,
            graph_kb=graph_kb,
        )
        all_summaries.append(summary)

    (output_dir / "combined_halu_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2)
    )
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
