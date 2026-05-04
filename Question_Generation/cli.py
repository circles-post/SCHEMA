from __future__ import annotations

# Early heartbeat — printed BEFORE heavy imports so that if the parent shell
# dies or imports hang, the user can tell from the log whether Python even
# reached this file. We flush explicitly because stderr may still be block
# buffered at this point (before --log-file handler is attached).
import sys as _sys, os as _os, time as _time
print(
    f"[{_time.strftime('%Y-%m-%d %H:%M:%S')} BOOT] cli.py reached "
    f"(pid={_os.getpid()}, python={_sys.executable})",
    file=_sys.stderr,
    flush=True,
)

import argparse
import logging
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path

print(
    f"[{_time.strftime('%Y-%m-%d %H:%M:%S')} BOOT] importing pubmed_graph.pubmed_client",
    file=_sys.stderr,
    flush=True,
)
from pubmed_graph.pubmed_client import PubMedClient
print(
    f"[{_time.strftime('%Y-%m-%d %H:%M:%S')} BOOT] pubmed_graph imported",
    file=_sys.stderr,
    flush=True,
)

from collections import Counter

from .config import (
    DEFAULT_CORROBORATION_MODE,
    DEFAULT_CORROBORATION_TOOL_TIMEOUT,
    DEFAULT_COVERAGE_MODE,
    DEFAULT_GITHUB_SEARCH_LANGUAGE,
    DEFAULT_GITHUB_SEARCH_PER_PAGE,
    DEFAULT_MAX_PER_UNIQUENESS_KEY,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_EXTERNAL_SOURCES,
    DEFAULT_MIN_LOCAL_SUPPORT,
    DEFAULT_NODE_QUOTA,
    DEFAULT_RATIO_POOL_MULTIPLIER,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_SUPPORTED_QUESTION_TYPES,
    DEFAULT_VALIDATION_MODE,
    DEFAULT_VALIDATOR_MODEL_CONFIG,
)
from .sampling_plan import (
    allocate_samples,
    allocate_samples_node_based,
    load_graph_coverage,
    parse_ratio_spec,
)
from .benchmark_weight import build_benchmark_weight_block, load_graph_weights
from .corroboration_agent import CorroborationAgent, TOOLS_AVAILABLE as CORROB_TOOLS_AVAILABLE, self_check as corroboration_self_check
from .dedup import deduplicate_against, deduplicate_by_question, load_seen_questions
from .experiments import DEFAULT_DIFFICULTY, VALID_DIFFICULTIES
from .exporter import export_samples, export_summary
from .generator import build_question_sample
from .indexing import build_index
from .io import load_chunks, load_triples
from .sampler import sample_single_hop_subgraphs, sample_two_hop_subgraphs, sample_vqa_subgraphs
from .vqa_source import load_vqa_source
from .validation_cache import ValidationCache
from .validator import summarize_validation, validate_sample

logger = logging.getLogger("question_generation")


_EXPERIMENT_DIFFICULTY_CHOICES = (*VALID_DIFFICULTIES, "mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fact-grounded questions from scientific KG triples")
    parser.add_argument("--triples", required=True, help="Path to normalized_triples.jsonl")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--output", required=True, help="Path to output question_samples.jsonl")
    parser.add_argument("--summary-output", help="Optional path to summary json")
    parser.add_argument(
        "--graph",
        default=None,
        help=(
            "Optional path to global_graph.graphml produced by pubmed_graph. "
            "When supplied, every accepted QuestionSample gets a "
            "metadata.benchmark_weight block (PageRank-based importance tier "
            "of the answer-target entity, for a downstream weighted evaluator)."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument(
        "--max-per-uniqueness-key",
        type=int,
        default=DEFAULT_MAX_PER_UNIQUENESS_KEY,
        help=(
            "How many samples to keep that share the same (head,relation,tail,question_type) "
            f"key. Default {DEFAULT_MAX_PER_UNIQUENESS_KEY}. Raise to allow multiple paraphrases "
            "per triple, e.g. 3 to triple your output count when the underlying graph is small."
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument(
        "--min-local-support",
        type=int,
        default=DEFAULT_MIN_LOCAL_SUPPORT,
        help=(
            "Minimum number of local chunks that must independently support a "
            "(head, relation, tail) group before it is eligible. Default 2 "
            "(multi-source gate). When --corroboration-mode=required, you can "
            "drop this to 1 to let single-source triples enter the pipeline "
            "and rely on runtime external search for corroboration."
        ),
    )
    parser.add_argument(
        "--corroboration-mode",
        choices=["off", "required"],
        default=DEFAULT_CORROBORATION_MODE,
        help=(
            "When 'required', every rule-only-passing sample must also be "
            "backed by at least one external source found at runtime via "
            "literature_search/web_search. Fails closed: tool errors → "
            "reject. Meant to replace the static min_support=2 multi-source "
            "gate with an agent-search-based one."
        ),
    )
    parser.add_argument(
        "--min-external-sources",
        type=int,
        default=DEFAULT_MIN_EXTERNAL_SOURCES,
        help="How many distinct external sources are required to call a sample corroborated.",
    )
    parser.add_argument(
        "--corroboration-tool-timeout",
        type=float,
        default=DEFAULT_CORROBORATION_TOOL_TIMEOUT,
        help="Per-tool timeout in seconds for literature_search / web_search.",
    )
    parser.add_argument(
        "--corroboration-cache-dir",
        default=None,
        help="Cache dir for corroboration_agent results (claim → external sources).",
    )
    parser.add_argument(
        "--corroboration-self-check",
        action="store_true",
        help="Probe literature_search + web_search at startup and exit (diagnostic).",
    )
    parser.add_argument("--validation-mode", choices=["rule_only", "hybrid_model"], default=DEFAULT_VALIDATION_MODE)
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    parser.add_argument("--validation-cache-dir", help="Optional cache dir for model validation")
    parser.add_argument("--validator-model", default=DEFAULT_VALIDATOR_MODEL_CONFIG["model"])
    parser.add_argument("--validator-base-url", default=DEFAULT_VALIDATOR_MODEL_CONFIG["base_url"])
    parser.add_argument("--validator-api-key", default=DEFAULT_VALIDATOR_MODEL_CONFIG["api_key"])
    parser.add_argument("--validator-enabled", action="store_true")
    parser.add_argument("--github-search-language", default=DEFAULT_GITHUB_SEARCH_LANGUAGE)
    parser.add_argument("--github-search-per-page", type=int, default=DEFAULT_GITHUB_SEARCH_PER_PAGE)
    parser.add_argument(
        "--llm-code-selection",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Control LLM-based function-level relevance extraction for experiment_code "
            "GitHub references. 'auto' (default) uses the LLM when OPENAI_*/INTERN_* "
            "credentials are set, otherwise falls back to keyword scoring. 'on' forces "
            "the LLM path (errors fall back to keywords). 'off' disables LLM entirely "
            "and always uses the keyword scorer."
        ),
    )
    parser.add_argument(
        "--experiment-generation-mode",
        choices=["template", "llm", "hybrid"],
        default="template",
        help=(
            "How experiment_code samples are produced. 'template' (default, "
            "legacy) uses hardcoded blueprints. 'llm' asks intern-s1-pro to "
            "synthesize a bespoke (data_code, main_code, unit_tests) per "
            "triple and sandbox-gates the spec; if generation fails the "
            "sample is rejected (no template fallback). 'hybrid' tries the "
            "LLM first and silently falls back to the blueprint template on "
            "failure. The LLM path requires INTERN_API_KEY / OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--experiment-difficulty",
        choices=list(_EXPERIMENT_DIFFICULTY_CHOICES),
        default=DEFAULT_DIFFICULTY,
        help=(
            "Blank-out strategy for experiment_code questions. "
            "'easy' blanks one function, 'medium' blanks the listed helpers, "
            "'hard' additionally blanks the orchestration function. "
            "'mixed' produces one sample per difficulty for every eligible subgraph."
        ),
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=list(DEFAULT_SUPPORTED_QUESTION_TYPES),
        choices=list(DEFAULT_SUPPORTED_QUESTION_TYPES),
    )
    parser.add_argument(
        "--vqa-triples",
        default=None,
        help=(
            "Optional path to benchmark_triples.jsonl (self-loop QA overlay, "
            "e.g. PathVQA). Required to produce `vqa` samples."
        ),
    )
    parser.add_argument(
        "--vqa-image-index",
        default=None,
        help=(
            "Optional path to benchmark_image_index.json mapping short doc IDs "
            "to image file paths. Required to produce `vqa` samples."
        ),
    )
    parser.add_argument(
        "--max-vqa-samples",
        type=int,
        default=None,
        help=(
            "Cap on VQA samples. Defaults to len(vqa_source). Independent "
            "of --max-samples (which caps non-VQA sampling)."
        ),
    )
    parser.add_argument(
        "--ratio",
        nargs="+",
        default=None,
        help=(
            "Per-question-type ratio of candidate samples. Pass as "
            "key=value pairs, e.g. --ratio two_hop_tail=0.3 experiment_code=0.2. "
            "Declared types keep their stated weight; undeclared-but-enabled "
            "types split the remainder evenly. Weights are normalized to 1. "
            "Quota per type = round(--max-samples × weight). When a type's "
            "candidate pool is smaller than its quota we cap at pool and warn "
            "(no redistribution). Without this flag every enabled type gets "
            "equal share."
        ),
    )
    parser.add_argument(
        "--dedup-against",
        nargs="+",
        default=None,
        help=(
            "Paths to one or more previously produced samples.jsonl files. "
            "Every normalized question text appearing in those files is "
            "excluded from this run's output — lets you run two graphs "
            "back-to-back without re-emitting identical questions where "
            "their triple sets overlap."
        ),
    )
    parser.add_argument(
        "--node-quota",
        nargs="+",
        default=None,
        help=(
            "Per-tier sample quota for the node-based allocator. Pass as "
            "T1=3 T2=2 T3=1 (default). Each graphml node gets its tier's "
            "quota worth of samples (covering that node). Active whenever "
            "--graph is given; disables --ratio / --coverage-priority."
        ),
    )
    parser.add_argument(
        "--node-based",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Use node-based allocator (one quota per graphml node by tier). "
            "'auto' (default) = on when --graph given, off otherwise. 'on' "
            "requires --graph. 'off' falls back to the ratio-based allocator."
        ),
    )
    parser.add_argument(
        "--coverage-priority",
        type=int,
        default=0,
        help=(
            "Reserve K of --max-samples for a cross-type greedy pass that "
            "picks samples purely to maximize new graph-node+edge coverage, "
            "before ratio quotas apply. K=0 disables (pure ratio-first). "
            "Requires --graph. Set to a meaningful fraction of your budget "
            "(e.g. 0.3 × max-samples) to prioritize graph coverage."
        ),
    )
    parser.add_argument(
        "--coverage",
        choices=["off", "greedy"],
        default=DEFAULT_COVERAGE_MODE,
        help=(
            "How to pick candidates within each type's quota. 'off' = "
            "random shuffle. 'greedy' = at each step, pick the candidate "
            "that adds the most new graph nodes+edges to the running "
            "covered-set; requires --graph. If 'greedy' is requested but "
            "--graph is missing, we warn and fall back to 'off'."
        ),
    )
    parser.add_argument(
        "--log-file",
        help="Path to write a detailed progress log. If omitted, logs go to stderr only.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Number of parallel worker threads for build+validate. "
            "Each worker does I/O (LLM judge, GitHub API, sandbox) so threads "
            "are fine. Set to 1 for serial execution. Default 8."
        ),
    )
    parser.add_argument(
        "--sample-timeout",
        type=float,
        default=180.0,
        help=(
            "Per-sample wall-clock budget (seconds) for build+validate. "
            "If a worker does not complete a sample in this time, the future "
            "is abandoned and counted as a reject (the thread keeps running "
            "in the background but we stop waiting for it). Default 180. "
            "Primarily protects experiment_code from hanging on GitHub "
            "API / sandbox RPCs; non-experiment workers finish in <3s."
        ),
    )
    return parser.parse_args()


def _setup_logging(args: argparse.Namespace) -> None:
    """Configure the ``question_generation`` logger.

    Always sends to stderr (so the user's terminal shows progress) and,
    when ``--log-file`` is given, also appends to that file so the run
    can be monitored from another terminal via ``tail -f``.
    """
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger("question_generation")
    root.setLevel(level)
    # stderr handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    # optional file handler
    if getattr(args, "log_file", None):
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("logging to %s", args.log_file)


def main() -> None:
    args = parse_args()
    _setup_logging(args)
    t_start = time.time()

    # Early diag branch: --corroboration-self-check runs a probe and exits
    # without touching triples/chunks. Useful for validating the tool chain
    # before committing to a long batch run.
    if getattr(args, "corroboration_self_check", False):
        diag = corroboration_self_check()
        import json as _json
        print(_json.dumps(diag, indent=2, ensure_ascii=False))
        return

    logger.info("=== question_generation start ===")
    logger.info("triples=%s  chunks=%s  max_samples=%d  max_per_key=%d",
                args.triples, args.chunks, args.max_samples, args.max_per_uniqueness_key)
    logger.info("question_types=%s  validation_mode=%s  validator_enabled=%s",
                args.question_types, args.validation_mode, args.validator_enabled)
    logger.info("experiment_difficulty=%s  llm_code_selection=%s",
                args.experiment_difficulty, args.llm_code_selection)

    logger.info("[phase 1/6] loading triples")
    triples = load_triples(args.triples)
    logger.info("[phase 1/6] loaded %d triples", len(triples))

    logger.info("[phase 2/6] loading chunks")
    chunks = load_chunks(args.chunks)
    logger.info("[phase 2/6] loaded %d chunks", len(chunks))

    logger.info("[phase 3/6] building index")
    index = build_index(triples, chunks)
    logger.info("[phase 3/6] index ready (triples=%d, entity_types=%d)",
                len(index.triples), len(index.entities_by_type))

    pubmed_client = PubMedClient()
    cache = ValidationCache(args.validation_cache_dir) if args.validation_cache_dir else None
    # Resolve validator credentials: CLI flag wins, else fall back to env
    # vars (INTERN_* preferred, OPENAI_* as alias). Without this the
    # model_config gets empty strings and model_validator.judge_claim()
    # short-circuits to a degraded fallback — no LLM call ever happens.
    import os as _os
    resolved_api_key = (
        args.validator_api_key
        or _os.environ.get("INTERN_API_KEY")
        or _os.environ.get("OPENAI_API_KEY")
        or ""
    )
    resolved_base_url = (
        args.validator_base_url
        or _os.environ.get("INTERN_BASE_URL")
        or _os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_VALIDATOR_MODEL_CONFIG.get("base_url", "")
    )
    resolved_model = (
        args.validator_model
        or _os.environ.get("OPENAI_MODEL")
        or DEFAULT_VALIDATOR_MODEL_CONFIG.get("model", "")
    )
    model_config = dict(DEFAULT_VALIDATOR_MODEL_CONFIG)
    model_config.update(
        {
            "enabled": bool(args.validator_enabled),
            "model": resolved_model,
            "base_url": resolved_base_url,
            "api_key": resolved_api_key,
        }
    )
    # Log what the validator will actually use (without leaking the key)
    logger.info(
        "validator: enabled=%s model=%s base_url=%s api_key=%s",
        model_config["enabled"],
        model_config["model"] or "<empty>",
        model_config["base_url"] or "<empty>",
        "set" if model_config["api_key"] else "<empty>",
    )
    if args.validator_enabled and not model_config["api_key"]:
        logger.warning(
            "validator_enabled=True but api_key is empty — every sample will "
            "fall through to 'degraded' mode. Set INTERN_API_KEY / OPENAI_API_KEY "
            "or pass --validator-api-key."
        )

    # ------------------------------------------------------------------
    # Corroboration agent (runtime external search)
    # ------------------------------------------------------------------
    corroboration_agent: CorroborationAgent | None = None
    if args.corroboration_mode == "required":
        if not CORROB_TOOLS_AVAILABLE:
            logger.error(
                "corroboration-mode=required but external tools are not "
                "importable — refusing to start. Check PYTHONPATH / env."
            )
            sys.exit(2)
        corroboration_cache = ValidationCache(args.corroboration_cache_dir) if args.corroboration_cache_dir else None
        corroboration_agent = CorroborationAgent(
            min_external_sources=args.min_external_sources,
            tool_timeout=args.corroboration_tool_timeout,
            cache=corroboration_cache,
        )
        logger.info(
            "corroboration agent ready: min_external_sources=%d timeout=%.0fs cache_dir=%s",
            args.min_external_sources,
            args.corroboration_tool_timeout,
            args.corroboration_cache_dir or "<none>",
        )

    logger.info("[phase 4/6] sampling subgraphs (collect per-type pools)")
    requested_single = tuple(q for q in args.question_types if q in {"claim_choice", "boolean_support", "one_hop_tail", "essay", "experiment_code"})
    if args.experiment_difficulty == "mixed":
        experiment_difficulties = tuple(VALID_DIFFICULTIES)
    else:
        experiment_difficulties = (args.experiment_difficulty,)

    # Per-type cap on how deep each pool is allowed to grow. When the user
    # passes --ratio or --coverage we over-sample each type so the
    # allocator has headroom to cut. Otherwise the legacy behaviour is
    # preserved: each type cap = --max-samples.
    ratio_active = bool(args.ratio) or args.coverage == "greedy"
    if ratio_active:
        pool_cap = max(args.max_samples * DEFAULT_RATIO_POOL_MULTIPLIER, args.max_samples)
    else:
        pool_cap = args.max_samples

    corroboration_will_run = args.corroboration_mode == "required"

    candidates_by_type: dict[str, list] = {}
    # Collect each requested single-hop type into its own pool
    for qtype in requested_single:
        pool = sample_single_hop_subgraphs(
            index=index,
            question_types=(qtype,),
            min_confidence=args.min_confidence,
            min_support=args.min_local_support,
            max_samples=pool_cap,
            experiment_difficulties=experiment_difficulties,
            corroboration_will_run=corroboration_will_run,
        )
        candidates_by_type[qtype] = pool
        logger.info("[phase 4/6] %s pool: %d candidates", qtype, len(pool))

    if "two_hop_tail" in args.question_types:
        two_hop = sample_two_hop_subgraphs(
            index=index,
            min_confidence=args.min_confidence,
            min_support=args.min_local_support,
            max_samples=pool_cap,
            corroboration_will_run=corroboration_will_run,
        )
        candidates_by_type["two_hop_tail"] = two_hop
        logger.info("[phase 4/6] two_hop_tail pool: %d candidates", len(two_hop))

    vqa_enabled_and_loaded = False
    if "vqa" in args.question_types:
        if not (args.vqa_triples and args.vqa_image_index):
            logger.warning(
                "vqa in --question-types but --vqa-triples / --vqa-image-index not set; "
                "skipping vqa sampling"
            )
        else:
            vqa_records = load_vqa_source(args.vqa_triples, args.vqa_image_index)
            cap = args.max_vqa_samples if args.max_vqa_samples else pool_cap
            vqa_samples = sample_vqa_subgraphs(vqa_records, max_samples=cap)
            candidates_by_type["vqa"] = vqa_samples
            vqa_enabled_and_loaded = True
            logger.info(
                "[phase 4/6] vqa pool: %d candidates (from %d records)",
                len(vqa_samples), len(vqa_records),
            )

    # Load graphml coverage universe; also compute tier map for the
    # node-based allocator when --graph is given.
    graph_nodes: set[str] | None = None
    graph_edges: set[tuple[str, str, str]] | None = None
    graph_alias_map: dict[str, str] | None = None
    graph_tier_map: dict[str, str] | None = None
    use_node_based = False
    if args.graph:
        try:
            _loaded = load_graph_coverage(args.graph, with_tiers=True)
            graph_nodes, graph_edges, graph_alias_map, graph_tier_map = _loaded
        except Exception as exc:
            logger.error("could not load --graph for coverage: %s", exc)
            graph_nodes = graph_edges = graph_alias_map = graph_tier_map = None

    # Decide allocator: node-based vs ratio-based
    if args.node_based == "on":
        use_node_based = True
    elif args.node_based == "auto":
        use_node_based = graph_tier_map is not None
    else:
        use_node_based = False
    if use_node_based and graph_tier_map is None:
        logger.warning("--node-based=on but no --graph — falling back to ratio allocator")
        use_node_based = False

    # Parse --node-quota if given, else use default
    tier_quota: dict[str, int] = dict(DEFAULT_NODE_QUOTA)
    if args.node_quota:
        for raw in args.node_quota:
            if "=" not in raw:
                logger.warning("--node-quota item %r missing '=' — skipping", raw)
                continue
            k, v = raw.split("=", 1)
            k = k.strip().upper()
            try:
                tier_quota[k] = max(0, int(v.strip()))
            except ValueError:
                logger.warning("--node-quota item %r has non-int value — skipping", raw)

    if use_node_based:
        # In node-based mode, --ratio is a SOFT global target biasing the
        # type-selection step inside each node. Types with the largest
        # (target - selected) deficit get picked first. If --ratio is not
        # given, types share equally and we degenerate to round-robin.
        enabled_types_nb = set(candidates_by_type.keys())
        type_ratios_nb = parse_ratio_spec(args.ratio, enabled_types_nb)
        if args.coverage_priority:
            logger.warning("--coverage-priority ignored because --node-based is active")
        sampled, allocation_diag = allocate_samples_node_based(
            candidates_by_type=candidates_by_type,
            total_quota=args.max_samples,
            graph_nodes=graph_nodes or set(),
            graph_edges=graph_edges or set(),
            alias_map=graph_alias_map or {},
            tier_map=graph_tier_map or {},
            tier_quota=tier_quota,
            type_ratios=type_ratios_nb,
        )
    else:
        enabled_types = set(candidates_by_type.keys())
        ratios = parse_ratio_spec(args.ratio, enabled_types)
        sampled, allocation_diag = allocate_samples(
            candidates_by_type=candidates_by_type,
            total_quota=args.max_samples,
            ratios=ratios,
            coverage_mode=args.coverage,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            alias_map=graph_alias_map,
            coverage_priority=args.coverage_priority,
        )
    logger.info(
        "[phase 4/6] allocated %d samples: per_type_selected=%s",
        len(sampled), allocation_diag.get("per_type_selected"),
    )
    if "coverage" in allocation_diag:
        cov = allocation_diag["coverage"]
        logger.info(
            "[phase 4/6] graph coverage: nodes %d/%d (%.1f%%), edges %d/%d (%.1f%%)",
            cov["covered_nodes"], cov["total_graph_nodes"], cov["node_coverage_rate"] * 100,
            cov["covered_edges"], cov["total_graph_edges"], cov["edge_coverage_rate"] * 100,
        )
    logger.info("[phase 4/6] sampled %d subgraphs total", len(sampled))

    # --- pre-filter by uniqueness key (cheap, serial) ---
    logger.info("[phase 5/6] pre-filtering by uniqueness key (max_per_key=%d)", args.max_per_uniqueness_key)
    uniqueness_counts: dict[str, int] = {}
    work_items: list[tuple[int, object]] = []  # (original_idx, subgraph)
    skipped_dup = 0
    for idx, subgraph in enumerate(sampled, start=1):
        current = uniqueness_counts.get(subgraph.uniqueness_key, 0)
        if current >= args.max_per_uniqueness_key:
            skipped_dup += 1
            continue
        uniqueness_counts[subgraph.uniqueness_key] = current + 1
        work_items.append((idx, subgraph))
    logger.info("[phase 5/6] %d candidates after key-dedup (%d skipped), workers=%d",
                len(work_items), skipped_dup, args.workers)

    # --- worker function ---
    _progress_lock = threading.Lock()
    _progress = {"done": 0, "accepted": 0, "rejected": 0}

    def _process_one(item: tuple[int, object]) -> tuple[object, bool]:
        idx, subgraph = item
        t0 = time.time()
        sample = build_question_sample(
            index,
            subgraph,
            sample_id=f"qg_{idx:06d}",
            github_search_per_page=args.github_search_per_page,
            github_search_language=args.github_search_language,
            llm_code_selection=args.llm_code_selection,
            experiment_generation_mode=args.experiment_generation_mode,
            corroboration_will_run=corroboration_will_run,
        )
        t_build = time.time() - t0

        t0 = time.time()
        sample = validate_sample(
            index,
            sample,
            validation_mode=args.validation_mode,
            model_config=model_config,
            pubmed_client=pubmed_client,
            retrieval_top_k=args.retrieval_top_k,
            cache=cache,
            corroboration_agent=corroboration_agent,
        )
        t_validate = time.time() - t0

        passed = sample.quality.validation_status == "passed" and (
            sample.question_type in {"essay", "experiment_code", "vqa"} or len(sample.options) >= 2
        )
        status_tag = "PASS" if passed else "REJECT"
        reject_detail = ""
        if not passed:
            reasons = getattr(sample.quality, "rejection_reasons", []) or []
            opts = len(sample.options)
            reject_detail = f" reasons={reasons} options={opts}"
        with _progress_lock:
            _progress["done"] += 1
            if passed:
                _progress["accepted"] += 1
            else:
                _progress["rejected"] += 1
            logger.info(
                "  [%d/%d] %s %s type=%s vmode=%s build=%.1fs validate=%.1fs  (accepted=%d rejected=%d)%s",
                _progress["done"], len(work_items), sample.sample_id, status_tag,
                sample.question_type, sample.grounding.validation_mode,
                t_build, t_validate, _progress["accepted"], _progress["rejected"],
                reject_detail,
            )
        return sample, passed

    # --- split by type: experiment_code has external I/O (GitHub API rate
    # limit, sandbox, LLM code selection) that can't tolerate high concurrency;
    # the rest are template + LLM-judge and parallelise cleanly. ---
    non_experiment = [(i, sg) for i, sg in work_items if sg.question_type != "experiment_code"]
    experiment = [(i, sg) for i, sg in work_items if sg.question_type == "experiment_code"]
    experiment_workers = min(args.workers, 2)  # cap: GitHub search API = 30 req/min
    logger.info(
        "[phase 5/6] generating + validating: %d non-experiment (workers=%d) + %d experiment_code (workers=%d)",
        len(non_experiment), args.workers, len(experiment), experiment_workers,
    )

    samples = []
    type_counter: Counter[str] = Counter()
    rejected = 0

    def _run_batch(items: list, workers: int) -> None:
        nonlocal rejected
        if not items:
            return
        if workers <= 1:
            for item in items:
                sample, passed = _process_one(item)
                if passed:
                    samples.append(sample)
                    type_counter[sample.question_type] += 1
                else:
                    rejected += 1
        else:
            # Use a daemon-threaded pool so abandoned futures (that exceeded
            # --sample-timeout) do not keep the interpreter alive after main()
            # returns. The pool is NOT context-managed because the default
            # __exit__ waits for all outstanding futures — which is exactly
            # what we want to avoid when a worker is stuck.
            pool = ThreadPoolExecutor(max_workers=workers)
            _consecutive_timeouts_ref = [0]
            try:
                futures = {pool.submit(_process_one, item): item for item in items}
                pending = set(futures)
                per_sample_timeout = max(float(args.sample_timeout), 10.0)
                # Total wall-clock budget for this batch, generous: each
                # worker gets per_sample_timeout, so N items / workers * budget
                total_budget = (len(items) / max(workers, 1)) * per_sample_timeout + per_sample_timeout
                deadline = time.time() + total_budget
                while pending:
                    remaining = max(deadline - time.time(), 1.0)
                    try:
                        for future in as_completed(list(pending), timeout=min(per_sample_timeout, remaining)):
                            pending.discard(future)
                            try:
                                sample, passed = future.result(timeout=0)
                            except Exception as exc:
                                logger.error("  worker exception: %s", exc)
                                rejected += 1
                                continue
                            if passed:
                                samples.append(sample)
                                type_counter[sample.question_type] += 1
                            else:
                                rejected += 1
                            _consecutive_timeouts_ref[0] = 0  # progress made; reset timeout counter
                            break  # re-enter outer loop to refresh timeout window
                    except FuturesTimeoutError:
                        # No future completed in the timeout window. Abandon
                        # ONLY the futures that are actively running (stuck
                        # workers) — queued futures may still complete on the
                        # next pass. Cancel queued ones first (they haven't
                        # started yet, so cancel() works) and give the
                        # running ones up as rejects.
                        stuck_running = [f for f in pending if f.running()]
                        cancellable = [f for f in pending if not f.running() and f.cancel()]
                        if not stuck_running and not cancellable:
                            # Nothing is running, nothing cancellable — give up
                            logger.warning("  [TIMEOUT] no progress possible — ending batch")
                            for fut in pending:
                                rejected += 1
                                with _progress_lock:
                                    _progress["done"] += 1
                                    _progress["rejected"] += 1
                                    logger.info(
                                        "  [%d/%d] <stuck> TIMEOUT-ABANDONED  (accepted=%d rejected=%d)",
                                        _progress["done"], len(work_items),
                                        _progress["accepted"], _progress["rejected"],
                                    )
                            pending.clear()
                            break
                        logger.warning(
                            "  [TIMEOUT] %d running future(s) exceeded per-sample budget %.0fs — abandoning running only (%d queued still waiting)",
                            len(stuck_running), per_sample_timeout,
                            len(pending) - len(stuck_running) - len(cancellable),
                        )
                        for fut in stuck_running:
                            pending.discard(fut)
                            rejected += 1
                            with _progress_lock:
                                _progress["done"] += 1
                                _progress["rejected"] += 1
                                logger.info(
                                    "  [%d/%d] <stuck> TIMEOUT-ABANDONED-RUNNING  (accepted=%d rejected=%d)",
                                    _progress["done"], len(work_items),
                                    _progress["accepted"], _progress["rejected"],
                                )
                        for fut in cancellable:
                            pending.discard(fut)
                            rejected += 1
                            with _progress_lock:
                                _progress["done"] += 1
                                _progress["rejected"] += 1
                                logger.info(
                                    "  [%d/%d] <stuck> CANCELLED-QUEUED  (accepted=%d rejected=%d)",
                                    _progress["done"], len(work_items),
                                    _progress["accepted"], _progress["rejected"],
                                )
                        # After three timeouts with no progress, bail out of
                        # this batch entirely — the external service is not
                        # going to recover and we'd just burn N*timeout wall.
                        _consecutive_timeouts_ref[0] += 1
                        if _consecutive_timeouts_ref[0] >= 3:
                            logger.error(
                                "  [TIMEOUT] 3 consecutive timeouts — aborting batch, marking %d remaining as rejected",
                                len(pending),
                            )
                            for fut in pending:
                                fut.cancel()
                                rejected += 1
                                with _progress_lock:
                                    _progress["done"] += 1
                                    _progress["rejected"] += 1
                                    logger.info(
                                        "  [%d/%d] <batch-aborted> TIMEOUT-ABANDONED  (accepted=%d rejected=%d)",
                                        _progress["done"], len(work_items),
                                        _progress["accepted"], _progress["rejected"],
                                    )
                            pending.clear()
                            break
            finally:
                # Don't block shutdown on stuck workers.
                pool.shutdown(wait=False)

    _run_batch(non_experiment, args.workers)
    _run_batch(experiment, experiment_workers)

    logger.info("[phase 5/6] done: accepted=%d rejected=%d skipped_dup=%d  types=%s",
                len(samples), rejected, skipped_dup, dict(type_counter))

    # --- optional: attach graph-connectivity benchmark weights -------------
    benchmark_weight_summary: dict | None = None
    weight_map: dict[str, float] | None = None
    alias_map: dict[str, str] | None = None
    if args.graph:
        try:
            weight_map, alias_map = load_graph_weights(args.graph)
            logger.info(
                "benchmark_weight: loaded %d node weights + %d fusion aliases from %s",
                len(weight_map), len(alias_map), args.graph,
            )
        except Exception as exc:
            logger.error("benchmark_weight: failed to load %s (%s) — skipping",
                         args.graph, exc)
            weight_map = None
        if weight_map:
            tier_counts: Counter[str] = Counter()
            strategy_counts: Counter[str] = Counter()
            for sample in samples:
                block = build_benchmark_weight_block(sample, weight_map, alias_map)
                sample.metadata["benchmark_weight"] = block
                tier_counts[block["tier"]] += 1
                strategy_counts[block["match_strategy"]] += 1
            total_weighted = sum(tier_counts.values()) or 1
            pct_not_found = 100.0 * strategy_counts.get("not_found", 0) / total_weighted
            logger.info("benchmark_weight: tiers=%s strategies=%s not_found=%.1f%%",
                        dict(tier_counts), dict(strategy_counts), pct_not_found)
            if pct_not_found > 50.0:
                logger.warning(
                    "benchmark_weight: %.1f%% of anchors were not in the graphml "
                    "(pruned as small components or fusion-merged without aliases). "
                    "Those samples are tiered as T3_not_in_graph (weight 0.5) "
                    "— structurally niche by construction. If this is not the intent, "
                    "re-run pubmed_graph with min_component_size=1 or pass a denser graph.",
                    pct_not_found,
                )
            benchmark_weight_summary = {
                "config": {
                    "metric": "pagerank_log_percentile",
                    "alpha": 0.85,
                    "anchor_policy": "two_hop_middle_else_tail",
                    "tier_cutoffs": {"T1": 0.80, "T2": 0.20},
                    "weights": {
                        "T1": 1.5, "T2": 1.0, "T3": 0.5, "T3_not_in_graph": 0.5,
                    },
                    "not_found_policy": "tier=T3_not_in_graph, weight=0.5 (pruned ≈ niche)",
                },
                "tier_breakdown": dict(tier_counts),
                "match_strategy_breakdown": dict(strategy_counts),
                "graph_path": str(args.graph),
                "graph_node_count": len(weight_map),
                "graph_alias_count": len(alias_map or {}),
            }

    logger.info("[phase 6/6] dedup + export")
    pre_dedup = len(samples)
    sample_dicts = [sample.__dict__ for sample in samples]

    # Optional cross-run dedup: drop any question whose text appeared in
    # a previous run's samples.jsonl (passed via --dedup-against).
    cross_dropped = 0
    if args.dedup_against:
        seen_ext = load_seen_questions(args.dedup_against)
        if seen_ext:
            sample_dicts, cross_dropped = deduplicate_against(sample_dicts, seen_ext)
            logger.info(
                "[phase 6/6] cross-run dedup: dropped %d samples against %d seen keys",
                cross_dropped, len(seen_ext),
            )

    # Within-run dedup (normalized question text)
    deduped_dicts = deduplicate_by_question(sample_dicts)
    deduped_ids = {row["sample_id"] for row in deduped_dicts}
    samples = [sample for sample in samples if sample.sample_id in deduped_ids]
    logger.info(
        "[phase 6/6] dedup: %d -> %d (cross_dropped=%d + within_dropped=%d)",
        pre_dedup, len(samples), cross_dropped,
        pre_dedup - cross_dropped - len(samples),
    )

    export_samples(samples, args.output)
    logger.info("[phase 6/6] wrote %s (%d samples)", args.output, len(samples))

    experiment_blueprint_breakdown: Counter[str] = Counter()
    for sample in samples:
        if sample.question_type != "experiment_code":
            continue
        blueprint_name = str(sample.metadata.get("experiment_blueprint", "unknown"))
        difficulty = str(sample.metadata.get("experiment_difficulty", DEFAULT_DIFFICULTY))
        experiment_blueprint_breakdown[f"{blueprint_name}:{difficulty}"] += 1
    summary = {
        "triples": len(triples),
        "chunks": len(chunks),
        "sampled_subgraphs": len(sampled),
        "accepted_questions": len(samples),
        "validation": summarize_validation(samples),
        "question_types": args.question_types,
        "validation_mode": args.validation_mode,
        "experiment_difficulty": args.experiment_difficulty,
        "experiment_blueprint_breakdown": dict(experiment_blueprint_breakdown),
    }
    if benchmark_weight_summary is not None:
        summary["benchmark_weight"] = benchmark_weight_summary
    # Ratio + coverage diagnostics: always present (degenerates to
    # equal-share + coverage-off when the user didn't pass the flags).
    if allocation_diag.get("mode") == "node_based":
        summary["allocation"] = {
            "mode":                 "node_based",
            "tier_quota":           allocation_diag.get("tier_quota"),
            "tier_sizes":           allocation_diag.get("tier_sizes"),
            "per_tier_selected":    allocation_diag.get("per_tier_selected"),
            "per_type_selected":    allocation_diag.get("per_type_selected"),
            "per_type_pool":        allocation_diag.get("per_type_pool"),
            "per_type_target":      allocation_diag.get("per_type_target"),
            "per_type_ratio":       allocation_diag.get("per_type_ratio"),
            "per_type_deviation":   allocation_diag.get("per_type_deviation"),
            "truncated_tier_slots": allocation_diag.get("truncated_tier_slots"),
            "initial_target_samples": allocation_diag.get("initial_target_samples"),
            "final_sampled":        allocation_diag.get("final_sampled"),
            "insufficient_nodes":   allocation_diag.get("insufficient_nodes"),
            "per_node_count_histogram": allocation_diag.get("per_node_count_histogram"),
        }
    else:
        summary["question_type_quotas"] = {
            "declared_ratios":           allocation_diag.get("declared_ratios", {}),
            "effective_quotas":          allocation_diag.get("effective_quotas", {}),
            "per_type_pool":             allocation_diag.get("per_type_pool", {}),
            "per_type_selected":         allocation_diag.get("per_type_selected", {}),
            "coverage_priority_target":  allocation_diag.get("coverage_priority_target", 0),
            "coverage_priority_picked":  allocation_diag.get("coverage_priority_picked", 0),
        }
    if "coverage" in allocation_diag:
        summary["coverage"] = {
            "graph_path":      str(args.graph) if args.graph else None,
            **allocation_diag["coverage"],
        }
    if args.summary_output:
        export_summary(summary, args.summary_output)
        logger.info("[phase 6/6] wrote %s", args.summary_output)

    elapsed = time.time() - t_start
    logger.info("=== question_generation done in %.1fs: %d samples ===", elapsed, len(samples))


if __name__ == "__main__":
    main()
