"""Function-level relevance extraction for fetched GitHub source files.

The default ``experiment_code`` flow used to clip each downloaded GitHub
file to the first ``_DEFAULT_EXCERPT_BYTES`` and stuff the result into the
prompt. That hits docstrings / imports / unrelated functions before
reaching anything useful (e.g. ``snap-stanford/Biomni`` returned
``analyze_circular_dichroism_spectra`` for a dose-response query).

This module replaces that with:

  1. ``ast.parse`` the full file → list of top-level function defs.
  2. Score the candidates against the task context — first via LLM
     (``intern-s1-pro`` through the existing ``InternChatClient``) and,
     when credentials are missing or the call fails, via a pure-Python
     keyword overlap fallback.
  3. Return only the top N function bodies, ready to be embedded in the
     prompt's ``<github_code_excerpts>`` block.

The whole module is **best-effort and never raises**: any exception
degrades to the keyword fallback or, in the worst case, returns an empty
list of selections — the caller can then fall back to its old behaviour.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


# ---------------------------------------------------------------------------
# Stop-word list for the keyword fallback. We only need to remove the most
# generic English filler — the keyword scorer is just a tie-breaker.
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "have", "has",
    "are", "was", "were", "will", "would", "could", "should", "may", "might",
    "of", "in", "on", "at", "is", "an", "a", "to", "by", "as", "or", "if",
    "use", "used", "using", "any", "all", "not", "but", "be", "it", "its",
    "task", "code", "function", "functions", "implement", "implementation",
    "given", "based", "between", "such", "data", "input", "output", "value",
    "values", "compute", "calculate",  # too generic; we DO want "calculate" out
    "return", "returns",
}


@dataclass
class FunctionExtract:
    name: str
    signature: str
    docstring_first_line: str
    source: str            # full source spanning the def
    line_start: int
    line_end: int


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------
def extract_top_level_functions(source: str) -> list[FunctionExtract]:
    """Return callable definitions from a Python file.

    This now extracts both:
      - top-level ``def`` / ``async def`` functions
      - methods inside top-level classes (rendered as ``ClassName.method``)

    Private dunder methods (``__init__``, ``__repr__``, etc.) are skipped
    so the candidate list stays focused on "real" logic. Many real-world
    scientific Python files are class-based (e.g. ``GDSCDataAcquisition``),
    so ignoring class methods meant the selector got back an empty
    candidate list and fell through to ``method=none``.
    """
    if not source:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out: list[FunctionExtract] = []

    def _make(name: str, node: ast.AST) -> FunctionExtract | None:
        try:
            args_text = ast.unparse(node.args)
        except Exception:
            args_text = ""
        return_text = ""
        if getattr(node, "returns", None) is not None:
            try:
                return_text = " -> " + ast.unparse(node.returns)
            except Exception:
                return_text = ""
        signature = f"def {name}({args_text}){return_text}"
        ds_text = ast.get_docstring(node) or ""
        ds_first_line = ds_text.splitlines()[0] if ds_text else ""
        line_start = node.lineno
        line_end = getattr(node, "end_lineno", line_start) or line_start
        body_src = "\n".join(lines[line_start - 1 : line_end])
        return FunctionExtract(
            name=name,
            signature=signature.strip(),
            docstring_first_line=ds_first_line.strip(),
            source=body_src,
            line_start=line_start,
            line_end=line_end,
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = _make(node.name, node)
            if fn is not None:
                out.append(fn)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Skip dunder methods — they're almost never useful references
                if member.name.startswith("__") and member.name.endswith("__"):
                    continue
                fn = _make(f"{node.name}.{member.name}", member)
                if fn is not None:
                    out.append(fn)
    return out


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> set[str]:
    return {
        w.casefold()
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or "")
        if w.casefold() not in _STOPWORDS
    }


def _select_via_keywords(
    functions: list[FunctionExtract],
    task_objective: str,
    head: str,
    relation: str,
    tail: str,
    task_family: str,
    target_function_names: tuple[str, ...],
    max_select: int,
) -> list[int]:
    needles = (
        _tokenize(task_objective)
        | _tokenize(task_family)
        | _tokenize(head)
        | _tokenize(tail)
        | {n.casefold() for n in target_function_names}
        # Split snake_case into individual tokens too: "estimate_ic50" → {estimate, ic50}
        | {part for n in target_function_names for part in n.casefold().split("_") if len(part) >= 2}
        | {w.casefold() for w in re.findall(r"[A-Za-z]+", relation or "") if len(w) >= 3}
    )
    needles -= _STOPWORDS
    if not needles:
        return list(range(min(max_select, len(functions))))

    scored: list[tuple[int, int, int]] = []  # (score, original_index, tie_breaker)
    for idx, fn in enumerate(functions):
        haystack = " ".join(
            [
                fn.name,
                fn.name.replace("_", " "),
                fn.signature,
                fn.docstring_first_line,
            ]
        ).casefold()
        score = sum(1 for kw in needles if kw and kw in haystack)
        scored.append((-score, idx, idx))
    scored.sort()
    return [idx for _, idx, _ in scored[:max_select]]


# ---------------------------------------------------------------------------
# LLM selection
# ---------------------------------------------------------------------------
def _resolve_llm_config(llm_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user-supplied config with defaults read from the environment.

    Mirrors the resolution order used by ``model_validator`` so the same
    ``OPENAI_*`` / ``INTERN_*`` env vars (set via ``triple_extraction_env.sh``)
    work without any extra wiring.
    """
    cfg = dict(llm_config or {})
    cfg.setdefault("api_key", os.getenv("OPENAI_API_KEY") or os.getenv("INTERN_API_KEY", ""))
    cfg.setdefault("base_url", os.getenv("OPENAI_BASE_URL") or os.getenv("INTERN_BASE_URL") or "https://chat.intern-ai.org.cn/api/v1/")
    cfg.setdefault("model", os.getenv("OPENAI_MODEL") or "intern-s1-pro")
    cfg.setdefault("temperature", 0.0)
    cfg.setdefault("max_tokens", 300)
    cfg.setdefault("max_retries", 1)
    cfg.setdefault("request_timeout", 30.0)
    cfg.setdefault("retry_sleep_seconds", 2.0)
    cfg.setdefault("retry_backoff", 1.5)
    return cfg


def _llm_available(llm_config: dict[str, Any] | None) -> bool:
    cfg = _resolve_llm_config(llm_config)
    return bool(cfg.get("api_key")) and bool(cfg.get("base_url")) and bool(cfg.get("model"))


def _build_selection_prompt(
    functions: list[FunctionExtract],
    task_objective: str,
    head: str,
    relation: str,
    tail: str,
    task_family: str,
    target_function_names: tuple[str, ...],
    max_select: int,
) -> list[dict[str, str]]:
    catalog_lines = []
    for idx, fn in enumerate(functions):
        snippet = fn.docstring_first_line or "(no docstring)"
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        catalog_lines.append(f"  [{idx}] {fn.signature[:160]}    # {snippet}")
    catalog = "\n".join(catalog_lines) if catalog_lines else "  (no functions)"

    target_block = ""
    if target_function_names:
        target_block = (
            "TARGET HELPERS WE'RE IMPLEMENTING (use these as a strong hint for "
            "what kind of function we want a reference for):\n  - "
            + "\n  - ".join(target_function_names)
            + "\n\n"
        )

    user_msg = (
        f"You are picking reference Python functions to help a code-generation "
        f"benchmark. The model under test will use the selected functions as "
        f"inspiration to fill in missing helpers in a small biology task. Your "
        f"job is to build a diverse, useful reference set — not to find a single "
        f"perfect match.\n\n"
        f"TASK CONTEXT:\n"
        f"- Scientific claim: {head} {relation.replace('_', ' ')} {tail}\n"
        f"- Task family: {task_family}\n"
        f"- Task objective: {task_objective}\n\n"
        f"{target_block}"
        f"CANDIDATE FUNCTIONS (choose by index):\n{catalog}\n\n"
        f"SELECTION GUIDELINES:\n"
        f"- Aim for {max_select} picks. Prefer {max_select} diverse relevant "
        f"functions over 1 near-perfect match; the downstream model benefits "
        f"from multiple angles (e.g. one metric calculator + one fitter + one "
        f"data-prep helper).\n"
        f"- A function is 'relevant' if its core computation, math, or data "
        f"transformation overlaps with ANY of the target helpers. Partial "
        f"overlap counts.\n"
        f"- Only return fewer than {max_select} if the remaining functions are "
        f"clearly off-topic (plotting, CLI parsing, I/O boilerplate, config "
        f"loaders, logging, unrelated domains).\n"
        f"- Do NOT filter purely on naming: a function called `_helper` can "
        f"still be the best match if its body computes the right thing.\n\n"
        f"Reply with ONLY a JSON object on a single line:\n"
        f'  {{"selected": [<indices>], "rationale": "<one short sentence>"}}'
    )
    return [
        {
            "role": "system",
            "content": "You are a precise code-selection assistant. Reply ONLY with valid JSON.",
        },
        {"role": "user", "content": user_msg},
    ]


def _select_via_llm(
    functions: list[FunctionExtract],
    task_objective: str,
    head: str,
    relation: str,
    tail: str,
    task_family: str,
    target_function_names: tuple[str, ...],
    max_select: int,
    llm_config: dict[str, Any] | None,
) -> tuple[list[int], str]:
    """Try the LLM. Return ``(selected_indices, rationale)``.

    Raises on any failure so the caller knows to fall back. The rationale
    is logged into ``code_excerpts`` so you can debug why a function was
    picked.
    """
    from pubmed_graph.llm import InternChatClient

    cfg = _resolve_llm_config(llm_config)
    client = InternChatClient(cfg)
    messages = _build_selection_prompt(
        functions,
        task_objective=task_objective,
        head=head,
        relation=relation,
        tail=tail,
        task_family=task_family,
        target_function_names=target_function_names,
        max_select=max_select,
    )
    response = client.chat_json(
        messages,
        model=cfg.get("model"),
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens", 300)),
    )
    raw = response.get("selected", [])
    rationale = str(response.get("rationale", "")).strip()
    if not isinstance(raw, list):
        return [], rationale
    cleaned: list[int] = []
    for value in raw:
        try:
            i = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(functions) and i not in cleaned:
            cleaned.append(i)
        if len(cleaned) >= max_select:
            break
    return cleaned, rationale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def select_relevant_functions(
    source: str,
    *,
    task_objective: str,
    head: str,
    relation: str,
    tail: str,
    task_family: str,
    target_function_names: tuple[str, ...] = (),
    max_select: int = 3,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick the most-relevant top-level functions inside a Python source file.

    Returns a structured dict so the caller can both render the snippets
    into a prompt and serialize the selection rationale into ``metadata``::

        {
            "method":      "llm" | "keyword" | "all" | "none",
            "rationale":   str,                # only set when method=="llm"
            "total_functions": int,            # how many were AST-extracted
            "selected":    list[FunctionExtract dicts],
            "error":       str,                # set when both pipelines failed
        }
    """
    funcs = extract_top_level_functions(source)
    if not funcs:
        return {
            "method": "none",
            "rationale": "",
            "total_functions": 0,
            "selected": [],
            "error": "ast.parse returned no top-level functions",
        }
    if len(funcs) <= max_select:
        return {
            "method": "all",
            "rationale": f"file only contains {len(funcs)} top-level function(s); kept all",
            "total_functions": len(funcs),
            "selected": [_to_dict(fn) for fn in funcs],
            "error": "",
        }

    method = "keyword"
    rationale = ""
    indices: list[int] = []
    error = ""

    if _llm_available(llm_config):
        try:
            indices, rationale = _select_via_llm(
                funcs,
                task_objective=task_objective,
                head=head,
                relation=relation,
                tail=tail,
                task_family=task_family,
                target_function_names=target_function_names,
                max_select=max_select,
                llm_config=llm_config,
            )
            method = "llm" if indices else "keyword"
        except Exception as exc:
            error = f"llm_selection_failed: {type(exc).__name__}: {exc}"

    if not indices:
        indices = _select_via_keywords(
            funcs,
            task_objective=task_objective,
            head=head,
            relation=relation,
            tail=tail,
            task_family=task_family,
            target_function_names=target_function_names,
            max_select=max_select,
        )
        if method != "llm":
            method = "keyword"

    selected_dicts = [_to_dict(funcs[i]) for i in indices]
    return {
        "method": method,
        "rationale": rationale,
        "total_functions": len(funcs),
        "selected": selected_dicts,
        "error": error,
    }


def _to_dict(fn: FunctionExtract) -> dict[str, Any]:
    return {
        "name": fn.name,
        "signature": fn.signature,
        "docstring_first_line": fn.docstring_first_line,
        "source": fn.source,
        "line_start": fn.line_start,
        "line_end": fn.line_end,
    }


# ---------------------------------------------------------------------------
# Cached wrapper used by experiment_generator
# ---------------------------------------------------------------------------
def _content_hash(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8", errors="replace")).hexdigest()[:16]


@lru_cache(maxsize=256)
def _cached_select(
    content_hash: str,
    source: str,
    task_objective: str,
    head: str,
    relation: str,
    tail: str,
    task_family: str,
    target_function_names: tuple[str, ...],
    max_select: int,
    llm_enabled: bool,
) -> str:
    """Cache wrapper. Returns the JSON-serialized selection result.

    When ``llm_enabled`` is False we pass an explicit ``api_key=""`` to force
    ``_llm_available`` → False even if ``OPENAI_API_KEY`` / ``INTERN_API_KEY``
    are set in the environment; otherwise the "off" CLI mode would silently
    still call the LLM.
    """
    if llm_enabled:
        cfg: dict[str, Any] | None = None  # let env vars populate via _resolve_llm_config
    else:
        cfg = {"api_key": "", "base_url": "", "model": ""}
    result = select_relevant_functions(
        source,
        task_objective=task_objective,
        head=head,
        relation=relation,
        tail=tail,
        task_family=task_family,
        target_function_names=target_function_names,
        max_select=max_select,
        llm_config=cfg,
    )
    return json.dumps(result, default=str)


def select_relevant_functions_cached(
    source: str,
    *,
    task_objective: str,
    head: str,
    relation: str,
    tail: str,
    task_family: str,
    target_function_names: tuple[str, ...] = (),
    max_select: int = 3,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Cached entry point used by ``experiment_generator``.

    The cache key includes ``content_hash + task_objective + entities``, so
    a single GitHub file fetched once is classified once across all
    blueprint dispatches.
    """
    h = _content_hash(source)
    raw = _cached_select(
        h,
        source,
        task_objective,
        head,
        relation,
        tail,
        task_family,
        tuple(target_function_names),
        max_select,
        bool(use_llm),
    )
    return json.loads(raw)


__all__ = [
    "FunctionExtract",
    "extract_top_level_functions",
    "select_relevant_functions",
    "select_relevant_functions_cached",
]
