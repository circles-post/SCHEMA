from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from pubmed_graph.utils import normalize_text

from .config import DEFAULT_GITHUB_SEARCH_LANGUAGE, DEFAULT_GITHUB_SEARCH_PER_PAGE
from . import sandbox_runner
from .code_relevance import select_relevant_functions_cached
from .evidence_utils import independent_chunk_count, independent_doc_count, independent_support_count
from .experiments import (
    BlueprintContext,
    ExperimentBlueprint,
    dispatch_blueprint,
    normalize_difficulty,
    select_blank_targets,
)
from .github_tools import (
    fetch_github_file,
    list_github_repository_tree,
    search_github_code,
    search_github_code_structured,
    search_github_repositories,
)
from .models import Answer, Grounding, Provenance, Quality, QuestionSample


def _build_incomplete_code(main_code: str, incomplete_functions: tuple[str, ...]) -> str:
    updated = main_code
    for fn_name in incomplete_functions:
        marker = f"def {fn_name}("
        start = updated.find(marker)
        if start == -1:
            continue
        body_start = updated.find("\n", start)
        next_def = updated.find("\ndef ", body_start + 1)
        if next_def == -1:
            next_def = len(updated)
        snippet = updated[start:next_def]
        lines = snippet.splitlines()
        replacement: list[str] = []
        doc_delim = ""
        docstring_started = False
        docstring_closed = False
        for line in lines:
            replacement.append(line)
            stripped = line.strip()
            if not docstring_started and (stripped.startswith('"""') or stripped.startswith("'''")):
                docstring_started = True
                doc_delim = stripped[:3]
                if stripped.count(doc_delim) >= 2 and len(stripped) > 3:
                    docstring_closed = True
                    break
                continue
            if docstring_started and not docstring_closed and doc_delim and doc_delim in stripped:
                docstring_closed = True
                break
        if not replacement:
            replacement = [lines[0]]
        if not docstring_started:
            replacement = [lines[0]]
        replacement.append("    pass  # [Please complete the code]")
        replacement.append("")
        updated = updated.replace(snippet, "\n".join(replacement), 1)
    return updated


# How much of each fetched file we put into the prompt. Stays well under
# the prompt budget so a single experiment_code question stays manageable
# even with multiple excerpts.
#
# We now do a *full* fetch (up to ``_FULL_FETCH_BYTES``) so the AST-based
# function extractor in ``code_relevance`` can see complete function bodies
# instead of being clipped after a docstring or import block. The selector
# then keeps only ``_FUNCTIONS_PER_FILE`` top-relevant functions per file,
# each capped at ``_PER_FUNCTION_BYTES`` so the prompt budget stays bounded.
_FULL_FETCH_BYTES = 200_000
_DEFAULT_NUM_EXCERPTS = 2
_DEFAULT_TREE_ENTRIES = 20
_FUNCTIONS_PER_FILE = 3
_PER_FUNCTION_BYTES = 2500


@lru_cache(maxsize=128)
def _cached_repo_search(query: str, language: str, per_page: int) -> str:
    return search_github_repositories(query, language=language, per_page=per_page)


@lru_cache(maxsize=128)
def _cached_code_search(query: str, language: str, per_page: int) -> str:
    return search_github_code(query, language=language, per_page=per_page)


@lru_cache(maxsize=128)
def _cached_code_search_structured_key(query: str, language: str, per_page: int) -> tuple:
    """Return a hashable, cacheable view of structured code-search hits."""
    result = search_github_code_structured(query, language=language, per_page=per_page)
    items = result.get("items", []) or []
    # Tuple-of-tuples so lru_cache can store it.
    serialized = tuple(
        (it.get("owner", ""), it.get("repo", ""), it.get("path", ""), it.get("html_url", ""), it.get("default_branch", ""))
        for it in items
    )
    return (result.get("error", ""), serialized)


@lru_cache(maxsize=128)
def _cached_fetch_file(owner: str, repo: str, path: str, ref: str, max_bytes: int) -> tuple:
    """Cached wrapper around ``fetch_github_file`` returning a hashable tuple."""
    f = fetch_github_file(owner, repo, path, ref=ref, max_bytes=max_bytes)
    return (
        f.get("error", ""),
        f.get("size", 0),
        f.get("encoding", ""),
        f.get("content", ""),
        f.get("truncated", False),
        f.get("html_url", ""),
        f.get("ref", ""),
    )


@lru_cache(maxsize=128)
def _cached_repo_tree(owner: str, repo: str, ref: str, path_filter: str, max_entries: int) -> tuple:
    """Cached wrapper around ``list_github_repository_tree`` returning a hashable tuple."""
    t = list_github_repository_tree(
        owner, repo, ref=ref, path_filter=path_filter, max_entries=max_entries
    )
    entries = t.get("entries", []) or []
    serialized = tuple((e.get("path", ""), e.get("type", ""), int(e.get("size") or 0)) for e in entries)
    return (
        t.get("error", ""),
        t.get("ref", ""),
        t.get("default_branch", ""),
        bool(t.get("truncated", False)),
        serialized,
    )


def _truncate_function_source(source: str, max_bytes: int = _PER_FUNCTION_BYTES) -> tuple[str, bool]:
    """Clip a single function body if it exceeds the per-function budget."""
    if len(source) <= max_bytes:
        return source, False
    return source[:max_bytes], True


def _resolve_use_llm(llm_code_selection: str) -> bool:
    """Resolve the CLI tri-state flag into a boolean for code_relevance."""
    mode = (llm_code_selection or "auto").casefold()
    if mode == "off":
        return False
    if mode == "on":
        return True
    # auto: enable iff intern/openai creds are present
    import os
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("INTERN_API_KEY"))


def _build_github_reference_pack(
    blueprint: ExperimentBlueprint,
    blueprint_context: BlueprintContext,
    per_page: int = DEFAULT_GITHUB_SEARCH_PER_PAGE,
    language: str = DEFAULT_GITHUB_SEARCH_LANGUAGE,
    llm_code_selection: str = "auto",
) -> dict[str, Any]:
    """Run the GitHub search/fetch chain for a single blueprint.

    Compared with the previous version this now:

      - fetches each top-hit file in **full** (up to ``_FULL_FETCH_BYTES``)
      - uses ``code_relevance.select_relevant_functions_cached`` to extract
        only the most-relevant ``_FUNCTIONS_PER_FILE`` top-level functions
      - records the selection ``method`` (llm / keyword / all / none) and
        the LLM rationale into each excerpt's ``relevance`` sub-dict so
        downstream consumers can audit the choice
      - keeps the existing string-formatted ``repository_results`` and
        ``code_results`` fields untouched for backwards compat
    """
    # NOTE: _cached_repo_search (search_github_repositories) takes 30-50 seconds
    # because it fetches full repo metadata including READMEs. It is only used
    # to populate the <github_repository_search> prompt block — the actual code
    # extraction relies on search_github_code_structured + fetch_github_file.
    # We skip the slow repo search here and synthesise a short summary from the
    # structured code-search hits instead.
    repo_result = "(skipped — repo search deferred to speed up generation)"
    code_result = _cached_code_search(blueprint.github_code_query, language, per_page)

    status = "ok"
    if "failed" in repo_result.casefold() or "failed" in code_result.casefold():
        status = "degraded"

    # ----- structured code search → fetch top files -----
    structured_error, structured_items = _cached_code_search_structured_key(
        blueprint.github_code_query, language, per_page
    )
    code_excerpts: list[dict[str, Any]] = []
    repo_tree_summary: dict[str, Any] = {}

    if structured_error:
        status = "degraded"
    else:
        for owner, repo, path, html_url, _branch in structured_items[:_DEFAULT_NUM_EXCERPTS]:
            if not (owner and repo and path):
                continue
            file_error, file_size, encoding, content, truncated, file_html_url, ref_used = _cached_fetch_file(
                owner, repo, path, "", _FULL_FETCH_BYTES
            )
            if file_error:
                status = "degraded"
                code_excerpts.append(
                    {
                        "owner": owner, "repo": repo, "path": path,
                        "html_url": html_url or file_html_url,
                        "size": 0, "encoding": "", "content": "",
                        "truncated": False, "error": file_error,
                        "relevance": {"method": "none", "rationale": "fetch failed"},
                        "selected_functions": [],
                    }
                )
                continue

            # ----- LLM / keyword function-level relevance extraction -----
            use_llm = _resolve_use_llm(llm_code_selection)
            selection: dict[str, Any] = {"method": "none", "rationale": "", "selected": [], "error": ""}
            if encoding == "utf-8" and content:
                try:
                    selection = select_relevant_functions_cached(
                        content,
                        task_objective=blueprint.task_objective,
                        head=blueprint_context.head,
                        relation=blueprint_context.relation,
                        tail=blueprint_context.tail,
                        task_family=blueprint.task_family,
                        target_function_names=blueprint.incomplete_functions,
                        max_select=_FUNCTIONS_PER_FILE,
                        use_llm=use_llm,
                    )
                except Exception as exc:  # pragma: no cover
                    selection = {
                        "method": "none",
                        "rationale": "",
                        "selected": [],
                        "error": f"select_relevant_functions raised: {type(exc).__name__}: {exc}",
                    }

            selected_functions: list[dict[str, Any]] = []
            for fn in selection.get("selected", []) or []:
                src, fn_truncated = _truncate_function_source(fn.get("source", "") or "")
                selected_functions.append(
                    {
                        "name": fn.get("name", ""),
                        "signature": fn.get("signature", ""),
                        "docstring_first_line": fn.get("docstring_first_line", ""),
                        "line_start": fn.get("line_start"),
                        "line_end": fn.get("line_end"),
                        "source": src,
                        "truncated": fn_truncated,
                    }
                )

            code_excerpts.append(
                {
                    "owner": owner,
                    "repo": repo,
                    "path": path,
                    "html_url": html_url or file_html_url,
                    "ref": ref_used,
                    "size": file_size,
                    "encoding": encoding,
                    "fetch_truncated": bool(truncated),
                    "selected_functions": selected_functions,
                    "relevance": {
                        "method": selection.get("method", "none"),
                        "rationale": selection.get("rationale", ""),
                        "total_functions": selection.get("total_functions", 0),
                        "error": selection.get("error", ""),
                    },
                    "error": "",
                }
            )

        # ----- repo tree of the FIRST hit (gives the agent file layout) -----
        if structured_items:
            owner, repo, _path, _html_url, _branch = structured_items[0]
            if owner and repo:
                tree_error, tree_ref, default_branch, tree_truncated, tree_entries = _cached_repo_tree(
                    owner, repo, "", ".py", _DEFAULT_TREE_ENTRIES
                )
                repo_tree_summary = {
                    "owner": owner,
                    "repo": repo,
                    "ref": tree_ref,
                    "default_branch": default_branch,
                    "truncated": tree_truncated,
                    "entries": [
                        {"path": p, "type": t, "size": s} for (p, t, s) in tree_entries
                    ],
                    "error": tree_error,
                }
                if tree_error:
                    status = "degraded"

    return {
        "language": language,
        "repo_query": blueprint.github_repo_query,
        "code_query": blueprint.github_code_query,
        "repository_results": repo_result,
        "code_results": code_result,
        "code_excerpts": code_excerpts,
        "repo_tree_summary": repo_tree_summary,
        "status": status,
        "llm_code_selection_mode": llm_code_selection,
    }


def _format_code_excerpts(excerpts: list[dict[str, Any]]) -> str:
    """Render the LLM-selected (or keyword-selected) function snippets.

    The newer schema stores per-file ``selected_functions`` rather than a
    single raw blob. We render each function with its own header line so
    the model can see provenance + line numbers + the selection method.
    """
    if not excerpts:
        return "(no code excerpts fetched)"
    chunks: list[str] = []
    for ex in excerpts:
        repo_label = f"{ex.get('owner','?')}/{ex.get('repo','?')}"
        path = ex.get("path", "?")
        url = ex.get("html_url") or ""
        rel = ex.get("relevance", {}) or {}
        method = rel.get("method", "?")
        rationale = rel.get("rationale", "")
        total_fns = rel.get("total_functions", 0)

        header_lines = [f"### {repo_label} :: {path}"]
        if url:
            header_lines.append(f"# source: {url}")
        header_lines.append(
            f"# selection: method={method}, total_functions_in_file={total_fns}"
            + (f", rationale=\"{rationale}\"" if rationale else "")
        )
        header = "\n".join(header_lines)

        if ex.get("error"):
            chunks.append(f"{header}\n[fetch failed: {ex['error']}]")
            continue

        funcs = ex.get("selected_functions", []) or []
        if not funcs:
            chunks.append(f"{header}\n[no relevant top-level functions extracted]")
            continue

        for fn in funcs:
            fn_header = (
                f"# {repo_label} :: {path}  L{fn.get('line_start','?')}-{fn.get('line_end','?')}"
                f"  function: {fn.get('name','?')}"
            )
            body = fn.get("source", "") or ""
            truncated = "\n# ... [function body truncated]" if fn.get("truncated") else ""
            chunks.append(f"{header}\n{fn_header}\n```python\n{body}{truncated}\n```")
            # Show the file header only once per file by clearing it after the first function
            header = ""
    # Drop empty separators
    return "\n\n".join(c for c in chunks if c.strip())


def _format_repo_tree(tree: dict[str, Any]) -> str:
    if not tree or tree.get("error"):
        return "(no repository tree available)"
    entries = tree.get("entries", []) or []
    if not entries:
        return "(repository tree empty after filter)"
    header = f"{tree.get('owner','?')}/{tree.get('repo','?')} @ {tree.get('ref','?')} (default={tree.get('default_branch','?')})"
    lines = [header, "-" * len(header)]
    for e in entries[:30]:
        size_kb = (int(e.get("size") or 0) + 1023) // 1024
        lines.append(f"  {e.get('path','')}  ({size_kb} KB)")
    if tree.get("truncated"):
        lines.append("  ... (tree truncated by GitHub API)")
    return "\n".join(lines)


def _render_experiment_prompt(
    blueprint: ExperimentBlueprint,
    data_code: str,
    incomplete_main_code: str,
    github_references: dict[str, Any],
    head: str,
    relation: str,
    tail: str,
    evidence_snippets: list[str],
) -> str:
    evidence_text = " ".join(normalize_text(text) for text in evidence_snippets[:2] if normalize_text(text))
    github_status = github_references.get("status", "unknown")
    repo_results = str(github_references.get("repository_results", "")).strip()
    code_results = str(github_references.get("code_results", "")).strip()
    code_excerpts_text = _format_code_excerpts(github_references.get("code_excerpts", []) or [])
    repo_tree_text = _format_repo_tree(github_references.get("repo_tree_summary", {}) or {})
    return (
        "Please read the following experiment specification and complete the missing functions in "
        "`main_en.py`. The implementation should be grounded in the scientific evidence. Use the "
        "GitHub reference material below as inspiration rather than copying code verbatim — the "
        "code excerpts have been pulled from real public repositories matching the task topic.\n\n"
        f"<scientific_claim>\n{head} {relation.replace('_', ' ')} {tail}\n</scientific_claim>\n\n"
        f"<research_direction>\n{blueprint.research_focus}\n"
        f"Task objective: {blueprint.task_objective}\n"
        f"Evidence summary: {evidence_text or 'No additional evidence snippet provided.'}\n"
        "</research_direction>\n\n"
        f"<agent_workflow>\n"
        "1. Read the scientific evidence and task objective.\n"
        "2. Skim the GitHub code excerpts and repository layout to identify implementation patterns.\n"
        "3. Inspect `data_en.py` to understand the synthetic experiment inputs.\n"
        "4. Fill in only the missing functions in `main_en.py`.\n"
        "5. Ensure the implementation is numerically stable and aligned with the intended scientific computation.\n"
        f"GitHub reference status: {github_status}\n"
        "</agent_workflow>\n\n"
        f"<github_repository_search>\n{repo_results}\n</github_repository_search>\n\n"
        f"<github_code_search>\n{code_results}\n</github_code_search>\n\n"
        f"<github_code_excerpts>\n{code_excerpts_text}\n</github_code_excerpts>\n\n"
        f"<github_repo_tree>\n{repo_tree_text}\n</github_repo_tree>\n\n"
        f"<data_code>\n{data_code}\n</data_code>\n\n"
        f"<main_code>\n{incomplete_main_code}\n</main_code>\n"
    )


_EXP_LOGGER = __import__("logging").getLogger("question_generation.experiment")


def _sanitize_identifier(value: str, fallback: str = "generic") -> str:
    cleaned = normalize_text(value).replace(" ", "_").casefold()
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_").strip("_")
    return cleaned or fallback


def _blueprint_from_llm_spec(llm_spec: dict[str, Any], context: BlueprintContext) -> ExperimentBlueprint:
    """Create a prompt/search blueprint from the accepted LLM spec itself.

    The executable code and unit tests already come from ``llm_spec``. Using a
    synthetic blueprint derived from the same spec keeps the question prompt
    and GitHub reference queries aligned with the actual coding task rather
    than a fallback template family's queries.
    """
    task_family = str(llm_spec.get("task_family") or "llm_experiment")
    task_objective = str(llm_spec.get("task_objective") or "Complete the evidence-grounded experiment code.")
    research_focus = str(llm_spec.get("research_focus") or task_objective)
    incomplete_functions = tuple(str(fn) for fn in (llm_spec.get("incomplete_functions") or ()))
    github_query_seed = normalize_text(
        f"{task_family} {task_objective} {context.head} {context.tail} python"
    )
    return ExperimentBlueprint(
        name=f"llm_{_sanitize_identifier(task_family)}",
        task_family=task_family,
        relation=context.relation,
        direction=str(llm_spec.get("research_direction") or f"llm_{_sanitize_identifier(context.relation)}"),
        discipline=str(llm_spec.get("discipline") or "life"),
        function_type=str(llm_spec.get("function_type") or "Data analysis"),
        task_objective=task_objective,
        research_focus=research_focus,
        data_code_template=str(llm_spec.get("data_code", "")),
        main_code_template=str(llm_spec.get("main_code", "")),
        incomplete_functions=incomplete_functions,
        hard_extra_blanks=(),
        github_repo_query=str(llm_spec.get("github_repo_query") or github_query_seed),
        github_code_query=str(llm_spec.get("github_code_query") or github_query_seed),
        unit_tests=tuple(dict(item) for item in (llm_spec.get("unit_tests") or ())),
    )


def _build_rejected_experiment_sample(
    *,
    subgraph,
    sample_id: str,
    reason: str,
    generation_source: str,
    blueprint_name: str,
    validator_version: str = "experiment_generation_failed",
) -> QuestionSample:
    """Return a minimal experiment sample pre-marked as rejected.

    Used both for LLM-path failures and for template-dispatch failures
    (no matching blueprint for a triple). The rejection reason is logged
    and the sample is counted as rejected by the downstream validator.
    """
    support_count = independent_support_count(subgraph.supporting_triples)
    doc_support_count = independent_doc_count(subgraph.supporting_triples)
    chunk_support_count = independent_chunk_count(subgraph.supporting_triples)
    stub_quality = Quality(
        validation_status="rejected",
        difficulty="hard",
        ambiguity_score=0.0,
        uniqueness_key=subgraph.uniqueness_key,
        validator_version=validator_version,
        rejection_reasons=[reason],
    )
    stub_grounding = Grounding(
        is_fully_grounded=False,
        answer_supported=False,
        question_entities_supported=False,
        multi_hop_chain_supported=False,
        supporting_evidence_count=support_count,
        doc_support_count=doc_support_count,
        chunk_support_count=chunk_support_count,
        double_checked=False,
        support_level="multi_doc" if doc_support_count >= 2 else ("multi_chunk" if chunk_support_count >= 2 else "single_source"),
        validation_mode="rule_only",
        validation_status_detail=reason,
        evidence_strength="unknown",
        claim_strength="unknown",
        question_type_allowed_by_evidence=False,
        evidence_profile_version="v1",
    )
    return QuestionSample(
        sample_id=sample_id,
        question_type="experiment_code",
        question=f"[experiment generation rejected: {reason}]",
        answer=Answer(text="", canonical_text="", answer_type="Code"),
        options=[],
        subgraph={
            "nodes": [node.__dict__ for node in subgraph.nodes],
            "edges": [edge_item.__dict__ for edge_item in subgraph.edges],
        },
        provenance=Provenance(
            supporting_triples=subgraph.supporting_triples,
            supporting_chunks=subgraph.supporting_chunks,
            source_docs=sorted({t.doc_id for t in subgraph.supporting_triples}),
        ),
        grounding=stub_grounding,
        quality=stub_quality,
        metadata={
            "generation_source": generation_source,
            "generation_failed_reason": reason,
            "experiment_blueprint": blueprint_name,
            "experiment_difficulty": normalize_difficulty(subgraph.metadata.get("experiment_difficulty")),
        },
    )


def _build_rejected_llm_sample(*, subgraph, sample_id: str, edge, reason: str) -> QuestionSample:
    """Backwards-compatible shim — redirects to the generic rejected builder."""
    del edge
    return _build_rejected_experiment_sample(
        subgraph=subgraph,
        sample_id=sample_id,
        reason=reason,
        generation_source="llm",
        blueprint_name="llm::failed",
        validator_version="llm_generation_failed",
    )


def build_experiment_sample(
    subgraph,
    sample_id: str,
    github_search_per_page: int = DEFAULT_GITHUB_SEARCH_PER_PAGE,
    github_search_language: str = DEFAULT_GITHUB_SEARCH_LANGUAGE,
    llm_code_selection: str = "auto",
    generation_mode: str = "template",  # {"template", "llm", "hybrid"}
) -> QuestionSample:
    import time as _time
    _t_fn_start = _time.time()
    _EXP_LOGGER.info("[%s] build_experiment_sample START mode=%s", sample_id, generation_mode)
    edge = subgraph.edges[0]
    evidence_snippets = [triple.evidence for triple in subgraph.supporting_triples if normalize_text(triple.evidence)]
    head_type = ""
    tail_type = ""
    if subgraph.supporting_triples:
        head_type = subgraph.supporting_triples[0].head_type
        tail_type = subgraph.supporting_triples[0].tail_type
    difficulty = normalize_difficulty(subgraph.metadata.get("experiment_difficulty"))

    # --- Plan C: LLM-driven per-triple generation ---
    # In "llm" and "hybrid" modes we first ask the LLM to synthesize a
    # bespoke experiment for this triple. The spec is only accepted if its
    # reference solution passes its own unit tests AND the masked version
    # fails them (the sandbox gate inside generate_experiment_via_llm).
    llm_spec: dict[str, Any] | None = None
    if generation_mode in {"llm", "hybrid"}:
        _t0 = _time.time()
        from .experiment_llm_generator import generate_experiment_via_llm
        llm_spec = generate_experiment_via_llm(
            head=edge.head,
            head_type=head_type,
            relation=edge.relation,
            tail=edge.tail,
            tail_type=tail_type,
            evidence=evidence_snippets[0] if evidence_snippets else "",
            difficulty=difficulty,
            max_retries=2,
        )
        _EXP_LOGGER.info(
            "[%s] llm_generate %s in %.1fs",
            sample_id,
            "succeeded" if llm_spec else "failed",
            _time.time() - _t0,
        )
        if llm_spec is None and generation_mode == "llm":
            # Pure LLM mode — fail fast and produce a rejected sample
            # rather than silently using a template.
            return _build_rejected_llm_sample(
                subgraph=subgraph,
                sample_id=sample_id,
                edge=edge,
                reason="llm_generation_failed",
            )

    blueprint_context = BlueprintContext(
        head=edge.head,
        head_type=head_type,
        relation=edge.relation,
        tail=edge.tail,
        tail_type=tail_type,
        evidence=evidence_snippets[0] if evidence_snippets else "",
        difficulty=difficulty,
    )
    _t0 = _time.time()
    if llm_spec is not None:
        # LLM-spec path: synthesize the blueprint from the spec itself instead
        # of dispatching a hardcoded blueprint and overriding fields. Keeps the
        # prompt / GitHub queries aligned with the actual coding task.
        blueprint = _blueprint_from_llm_spec(llm_spec, blueprint_context)
        blueprint_name = f"llm::{llm_spec.get('task_family', 'generic')}"
        _EXP_LOGGER.info("[%s] blueprint=%s derived from LLM spec in %.1fs",
                         sample_id, blueprint_name, _time.time() - _t0)
        data_code_text = blueprint.data_code_template
        main_code = blueprint.main_code_template
        incomplete_main_code = str(llm_spec["incomplete_main_code"])
        unit_tests_list = [dict(ut) for ut in blueprint.unit_tests]
        blank_targets = tuple(blueprint.incomplete_functions)
    else:
        # Template path: dispatch by predicate. Failures (no matching blueprint)
        # used to crash — now emit a rejected sample so the batch continues.
        try:
            blueprint_name, blueprint = dispatch_blueprint(blueprint_context)
        except RuntimeError:
            return _build_rejected_experiment_sample(
                subgraph=subgraph,
                sample_id=sample_id,
                reason="no_matching_experiment_blueprint",
                generation_source="template",
                blueprint_name="unmatched",
            )
        _EXP_LOGGER.info("[%s] blueprint=%s dispatched in %.1fs",
                         sample_id, blueprint_name, _time.time() - _t0)
        data_code_text = blueprint.data_code_template
        main_code = blueprint.main_code_template
        blank_targets = select_blank_targets(blueprint, difficulty)
        incomplete_main_code = _build_incomplete_code(main_code, blank_targets)
        unit_tests_list = [dict(item) for item in blueprint.unit_tests]

    _t0 = _time.time()
    # LLM path already ran the sandbox gate inside generate_experiment_via_llm
    # — reuse that verdict to avoid paying for two more sandbox RPCs. Template
    # path still needs a fresh evaluation.
    if llm_spec is not None and "_sandbox_evaluation" in llm_spec:
        sandbox_evaluation = llm_spec["_sandbox_evaluation"]
    else:
        sandbox_evaluation = sandbox_runner.evaluate_experiment_sample(
            data_code=data_code_text,
            main_code=main_code,
            incomplete_main_code=incomplete_main_code,
            unit_tests=unit_tests_list,
        )
    _EXP_LOGGER.info("[%s] sandbox_eval done in %.1fs verdict=%s",
                     sample_id, _time.time() - _t0,
                     (sandbox_evaluation or {}).get("verdict", "?"))
    # When the verdict is NOT 'passed', dump the reference/incomplete sub-results
    # so we can tell whether the blueprint code is buggy, the sandbox is
    # misconfigured, or the masking was trivial. Truncated to avoid log bloat.
    if sandbox_evaluation and sandbox_evaluation.get("verdict") != "passed":
        ref = sandbox_evaluation.get("reference", {}) or {}
        inc = sandbox_evaluation.get("incomplete", {}) or {}
        _EXP_LOGGER.warning(
            "[%s] sandbox REJECT reasons=%s | ref: status=%s passed=%s failed=%s compile_error=%r stderr=%r | inc: status=%s passed=%s failed=%s",
            sample_id,
            sandbox_evaluation.get("rejection_reasons"),
            ref.get("sandbox_status"), ref.get("passed"), ref.get("failed"),
            (ref.get("compile_error") or "")[:200],
            (ref.get("stderr") or "")[:200],
            inc.get("sandbox_status"), inc.get("passed"), inc.get("failed"),
        )
        ref_tests = ref.get("test_results") or []
        if ref_tests:
            _EXP_LOGGER.warning(
                "[%s] ref first test: %s",
                sample_id,
                {k: (str(v)[:120] if v is not None else None) for k, v in ref_tests[0].items()},
            )

    _t0 = _time.time()
    github_references = _build_github_reference_pack(
        blueprint,
        blueprint_context=blueprint_context,
        per_page=github_search_per_page,
        language=github_search_language,
        llm_code_selection=llm_code_selection,
    )
    _EXP_LOGGER.info("[%s] github_refs done in %.1fs (n=%d)",
                     sample_id, _time.time() - _t0,
                     len(github_references.get("references", []) if isinstance(github_references, dict) else []))
    _EXP_LOGGER.info("[%s] build_experiment_sample TOTAL %.1fs", sample_id, _time.time() - _t_fn_start)
    question = _render_experiment_prompt(
        blueprint=blueprint,
        data_code=data_code_text,
        incomplete_main_code=incomplete_main_code,
        github_references=github_references,
        head=edge.head,
        relation=edge.relation,
        tail=edge.tail,
        evidence_snippets=evidence_snippets,
    )
    answer = Answer(
        text=main_code,
        canonical_text=normalize_text(main_code),
        answer_type="Code",
    )
    provenance = Provenance(
        supporting_triples=subgraph.supporting_triples,
        supporting_chunks=subgraph.supporting_chunks,
        source_docs=sorted({triple.doc_id for triple in subgraph.supporting_triples}),
    )
    doc_support_count = independent_doc_count(subgraph.supporting_triples)
    chunk_support_count = independent_chunk_count(subgraph.supporting_triples)
    evidence_support_count = independent_support_count(subgraph.supporting_triples)
    support_level = "multi_doc" if doc_support_count >= 2 else ("multi_chunk" if chunk_support_count >= 2 else "single_source")
    evidence_profile = copy.deepcopy(subgraph.metadata.get("evidence_profile", {}))
    grounding = Grounding(
        is_fully_grounded=True,
        answer_supported=True,
        question_entities_supported=True,
        multi_hop_chain_supported=True,
        supporting_evidence_count=evidence_support_count,
        doc_support_count=doc_support_count,
        chunk_support_count=chunk_support_count,
        double_checked=False,
        support_level=support_level,
        validation_mode="rule_only",
        validation_status_detail="experiment_spec_ready",
        evidence_strength=str(evidence_profile.get("evidence_strength", "unknown")),
        claim_strength=str(evidence_profile.get("claim_strength", "unknown")),
        question_type_allowed_by_evidence=subgraph.question_type in set(evidence_profile.get("allowed_question_types", [])),
        evidence_profile_version="v1",
    )
    quality = Quality(
        validation_status="pending",
        difficulty="hard",
        ambiguity_score=0.0,
        uniqueness_key=subgraph.uniqueness_key,
        validator_version="experiment_rule_only_v1",
    )
    metadata = dict(subgraph.metadata)
    metadata.update(
        {
            "task_family": (llm_spec.get("task_family") if llm_spec else blueprint.task_family),
            "experiment_blueprint": blueprint_name,
            "experiment_difficulty": difficulty,
            "research_direction": (llm_spec.get("research_direction") if llm_spec else blueprint.direction),
            "discipline": (llm_spec.get("discipline") if llm_spec else blueprint.discipline),
            "function_type": (llm_spec.get("function_type") if llm_spec else blueprint.function_type),
            "task_objective": (llm_spec.get("task_objective") if llm_spec else blueprint.task_objective),
            "research_focus": (llm_spec.get("research_focus") if llm_spec else blueprint.research_focus),
            "data_code": data_code_text,
            "main_code": main_code,
            "incomplete_main_code": incomplete_main_code,
            "incomplete_functions": list(blank_targets),
            "unit_tests": [dict(item) for item in unit_tests_list],
            "github_references": github_references,
            "github_search_language": github_search_language,
            "github_search_per_page": github_search_per_page,
            "sandbox_evaluation": sandbox_evaluation,
            "generation_source": ("llm" if llm_spec else "template"),
            "generation_mode": generation_mode,
            "generation_attempts": (llm_spec.get("generation_attempts") if llm_spec else 0),
            "agent_workflow": [
                "derive_scientific_task_from_grounded evidence",
                "search_github_repositories_for_reference_projects",
                "search_github_code_for_reusable_patterns",
                "draft_data_and_main_code_scaffolds",
                "blank_key_functions_and_export_unit_tests",
            ],
        }
    )
    return QuestionSample(
        sample_id=sample_id,
        question_type="experiment_code",
        question=question,
        answer=answer,
        options=[],
        subgraph={
            "nodes": [node.__dict__ for node in subgraph.nodes],
            "edges": [edge_item.__dict__ for edge_item in subgraph.edges],
        },
        provenance=provenance,
        grounding=grounding,
        quality=quality,
        metadata=metadata,
    )
