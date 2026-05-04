"""Full agent workflow with retrieval tools + ToolUniverse MCP.

This mirrors ``test_agent_debug.get_agent_team`` from
``/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger/test_agent_debug.py``,
but strips the intern-s1 specific logic (rate-limit retries, action-start
marker sanitizer, reflection-coerce salvage) — we target the Boyue-hosted
OpenAI-compatible models (Qwen*, gpt-*, deepseek-*), which don't exhibit
those pathologies.

Provides the same 4 retrieval tools (`web_search`, `web_fetch`,
`literature_search`, `literature_fetch`) plus an optional ToolUniverse MCP
workbench. Use ``BoyueFullAgentConfig.use_tooluniverse=False`` or
``AGDEBUGGER_TOOLUNIVERSE_URL=disabled`` to skip MCP bootstrap.

``build_agent_team_full(cfg)`` returns a coroutine; awaiting it yields a
ready-to-run ``RoundRobinGroupChat``. The caller is responsible for
``close`` on the returned model client when the run is finished.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool, StaticWorkbench
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, StreamableHttpServerParams

# Install custom autogen transformer for thinking models (deepseek-v4, glm-5,
# kimi-k2, intern-s1) so multi-turn requests carry reasoning_content. Safe /
# idempotent for non-thinking models. See evaluation/thinking_model_patch.py.
from evaluation import thinking_model_patch  # noqa: F401  (import side effect)


# ---------------------------------------------------------------------------
# sys.path hacks — AgentDebug and sciverse are sibling repos, not installed
# packages. We insert their dirs so `from websearch_tools import web_search`
# and `from sciverse_tools import literature_search` resolve.
# ---------------------------------------------------------------------------
_AGENTDEBUG_DIR = Path(
    os.environ.get(
        "AGENTDEBUG_DIR",
        "/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger",
    )
)
_SCIVERSE_DIR = Path(
    os.environ.get(
        "SCIVERSE_DIR",
        "/mnt/shared-storage-user/fengxinshun/AISci/sciverse",
    )
)

for _p in (_AGENTDEBUG_DIR, _SCIVERSE_DIR):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)

# sciverse_tools reads SCIVERSE_API_TOKEN at import-time via load_dotenv(),
# and dotenv looks at CWD, not at sciverse/'s own dir. Load its .env first
# so the token is in the env before import.
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore

    _sv_env = _SCIVERSE_DIR / ".env"
    if _sv_env.is_file():
        _load_dotenv(dotenv_path=_sv_env, override=False)
except ImportError:
    pass

from sciverse_tools import literature_search as _literature_search_raw  # noqa: E402
from sciverse_tools import sciverse_fetch_markdown  # noqa: E402
from websearch_tools import web_fetch, web_search  # noqa: E402


# ---------------------------------------------------------------------------
# Wrap literature_search to drop the Optional `language` param. autogen's
# FunctionTool derives a JSON schema from the signature; ``language: str | None``
# becomes ``"anyOf": [{"type":"string"},{"type":"null"}]`` with no top-level
# ``type`` field, which Gemini's strict schema validator rejects with
# ``functionDeclaration ... didn't specify the schema type field``. We never
# pass language anyway — sciverse defaults to None when omitted.
# ---------------------------------------------------------------------------
async def literature_search(query: str, num_results: int = 5) -> str:
    """Search scholarly literature; return paper metadata (title/authors/DOI/snippet)."""
    return await _literature_search_raw(query=query, num_results=num_results)


# ---------------------------------------------------------------------------
# literature_fetch wrapper: clamps num_results / max_success / markdown budget
# and guards parallelism with a per-process semaphore so a herd of MinerU
# subprocesses can't swamp the machine.
# ---------------------------------------------------------------------------
_LITERATURE_FETCH_CONCURRENCY = max(
    1, int(os.environ.get("EVAL_LITERATURE_FETCH_CONCURRENCY", "1"))
)
_LITERATURE_FETCH_SEMAPHORE: "asyncio.Semaphore | None" = None


def _get_literature_fetch_semaphore() -> "asyncio.Semaphore":
    global _LITERATURE_FETCH_SEMAPHORE
    if _LITERATURE_FETCH_SEMAPHORE is None:
        _LITERATURE_FETCH_SEMAPHORE = asyncio.Semaphore(_LITERATURE_FETCH_CONCURRENCY)
    return _LITERATURE_FETCH_SEMAPHORE


async def literature_fetch(
    query: str,
    num_results: int = 3,
    max_success: int = 2,
    max_markdown_chars: int = 8000,
) -> str:
    """Search, download, and parse paper PDFs to markdown via sciverse+MinerU.

    Thin wrapper over ``sciverse_fetch_markdown`` with safe defaults:
    ``convert_to_md=True``, ``include_markdown_content=True``, ``max_workers=4``.
    Use for paper BODY text (methods, figures, equations); for
    title/author/abstract metadata use ``literature_search`` instead.

    SLOW: seconds-to-minutes per paper. Individual papers may fail silently.
    Call at most once per question.

    Parameters
    ----------
    query: Title-like or topic-like query; prefer 3-8 entity tokens.
    num_results: candidates to consider from search (clamped [1, 10]).
    max_success: papers to download+parse (clamped [1, num_results]).
    max_markdown_chars: per-paper body cap (clamped [1000, 20000]).
    """
    num_results = max(1, min(10, int(num_results)))
    max_success = max(1, min(num_results, int(max_success)))
    max_markdown_chars = max(1000, min(20000, int(max_markdown_chars)))
    async with _get_literature_fetch_semaphore():
        return await sciverse_fetch_markdown(
            query=query,
            num_results=num_results,
            max_success=max_success,
            convert_to_md=True,
            include_markdown_content=True,
            max_markdown_chars=max_markdown_chars,
            max_workers=4,
        )


# ---------------------------------------------------------------------------
# System prompt. Same answer-format contract as the closed-book workflow, but
# with the tool-selection strategy from test_agent_debug.py so the agent
# actually uses search/literature tools when they would help.
# ---------------------------------------------------------------------------
_SYSTEM_MESSAGE = """\
You are a scientific-evaluation assistant answering one graded question per
turn. You have access to external retrieval tools and the ToolUniverse MCP
toolbox. Use them when they would help; otherwise reason from parametric
knowledge.

## Tool-use strategy

1. For scientific/technical questions, first try `find_tools` to discover a
   relevant ToolUniverse tool. If a suitable one exists, call it with the
   exact parameter names from its schema.
2. For paper-backed evidence, call `literature_search` to discover candidate
   papers. If the snippets/abstracts are enough, stop there; otherwise call
   `literature_fetch` at most ONCE to pull full-text markdown of 1-2 papers.
3. For general web facts, call `web_search`. If the snippets suffice, stop;
   otherwise call `web_fetch` with 1-3 URLs from the results.
4. Cap each kind of search at 2-3 attempts per question. Don't loop.
5. If tools aren't suitable, fall back to reasoning with available knowledge.

## Answer format by question type

- **multichoice** (claim_choice / one_hop_tail / two_hop_tail / vqa):
  Options are labelled A, B, C, .... Output the SINGLE option letter inside
  `<answer>` tags, e.g. `<answer>A</answer>`. Do NOT include option text.
- **boolean_support**: `<answer>Supported</answer>` or
  `<answer>Not Supported</answer>`.
- **essay**: concise factual answer inside `<answer>` tags (a few sentences).
- **experiment_code**: the completed Python main_code inside `<answer>`
  tags, no markdown fences, no commentary.

## Output discipline

1. Keep reasoning short and decision-oriented. Explain tool choices briefly.
2. Use EXACTLY ONE `<answer>` tag per response, in the format above for the
   sample's question type.
3. On the line AFTER the answer tag, output the token `TERMINATE` alone so
   the conversation ends. Without it the run wastes resources.

Example (multichoice)::

    literature_search confirmed a TAF2 / Microencephaly association, which
    matches option A; B-D contradict the cited evidence.

    <answer>A</answer>
    TERMINATE
"""


# ---------------------------------------------------------------------------
# ToolUniverse MCP bootstrap (lazy, process-global)
# ---------------------------------------------------------------------------
_TOOLUNIVERSE_WORKBENCH: "McpWorkbench | None" = None


def _tooluniverse_env(cfg: "BoyueFullAgentConfig") -> dict[str, str]:
    """Build the env passed to the ToolUniverse stdio subprocess."""
    env = {
        "OPENAI_API_KEY": cfg.api_key,
        "OPENAI_BASE_URL": cfg.base_url,
        "OPENAI_API_BASE": cfg.base_url,
        # ToolUniverse agentic tools need a default LLM; reuse the same Boyue
        # model so we don't fan out to a second provider.
        "TOOLUNIVERSE_LLM_CONFIG_MODE": "env_override",
        "TOOLUNIVERSE_LLM_DEFAULT_PROVIDER": "VLLM",
        "TOOLUNIVERSE_LLM_MODEL_DEFAULT": cfg.model,
        "VLLM_SERVER_URL": cfg.base_url,
        "AGENTIC_TOOL_FALLBACK_CHAIN": f'[{{"api_type": "VLLM", "model_id": "{cfg.model}"}}]',
        "FASTMCP_SHOW_SERVER_BANNER": "false",
        "TOOLUNIVERSE_SUPPRESS_STDIO_BANNER": "1",
    }
    # Preserve cache + PATH so the child can find binaries / sockets.
    for key in ("UV_CACHE_DIR", "XDG_CACHE_HOME", "HOME", "PATH", "TOOLUNIVERSE_CACHE_DIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    # Preserve proxy env so the MCP subprocess can reach external services
    # (same local proxy the runner already uses).
    for key in ("http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


def _build_tooluniverse_server_params(
    cfg: "BoyueFullAgentConfig",
) -> "StreamableHttpServerParams | StdioServerParams":
    """Return either an HTTP-mode or stdio-mode MCP server params object."""
    url = (os.environ.get("AGDEBUGGER_TOOLUNIVERSE_URL") or "").strip()
    if url and url.lower() not in {"disabled", "none", "off"}:
        return StreamableHttpServerParams(
            url=url,
            timeout=30.0,
            sse_read_timeout=300.0,
            terminate_on_close=False,
        )

    # Stdio: use the shim in scripts/run_tooluniverse_stdio.py (we don't have
    # `uv` on this host, so we can't use the upstream `uv run …` command).
    launcher = Path(__file__).resolve().parent / "scripts" / "run_tooluniverse_stdio.py"
    python_bin = sys.executable
    return StdioServerParams(
        command=python_bin,
        args=[
            str(launcher),
            "--exclude-tool-types",
            "PackageTool",
            "--compact-mode",
        ],
        env=_tooluniverse_env(cfg),
        read_timeout_seconds=120,
    )


async def _get_tooluniverse_workbench(cfg: "BoyueFullAgentConfig") -> "McpWorkbench | None":
    """Start (or return cached) the ToolUniverse MCP workbench.

    Returns ``None`` if the caller disabled ToolUniverse. Prints a one-line
    status summary so the runner log is self-explanatory.
    """
    if not cfg.use_tooluniverse:
        return None
    global _TOOLUNIVERSE_WORKBENCH
    if _TOOLUNIVERSE_WORKBENCH is not None:
        return _TOOLUNIVERSE_WORKBENCH

    server_params = _build_tooluniverse_server_params(cfg)
    wb = McpWorkbench(server_params=server_params)
    await wb.start()
    try:
        # asyncio.shield so a timeout doesn't cancel the underlying actor
        # future (which would crash the MCP subprocess).
        tools = await asyncio.wait_for(asyncio.shield(wb.list_tools()), timeout=60)
        kind = "http" if isinstance(server_params, StreamableHttpServerParams) else "stdio"
        print(f"[ToolUniverse MCP] connected via {kind}; tool count: {len(tools)}")
    except asyncio.TimeoutError:
        print(
            "[ToolUniverse MCP] started but list_tools timed out; MCP actor is "
            "alive and will serve requests once the server finishes loading."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ToolUniverse MCP] list_tools failed: {type(exc).__name__}: {exc}")

    _TOOLUNIVERSE_WORKBENCH = wb
    return wb


# ---------------------------------------------------------------------------
# Public config + team builder
# ---------------------------------------------------------------------------
@dataclass
class BoyueFullAgentConfig:
    """Config for the full tool-enabled agent team."""

    model: str
    base_url: str
    api_key: str
    timeout: float = 600.0
    max_retries: int = 2
    max_messages: int = 20
    max_tool_iterations: int = 5
    use_tooluniverse: bool = True
    # ``reflect_on_tool_use=True`` makes autogen do a SECOND no-tools model call
    # after every tool round to produce a final text response. Thinking models
    # (glm-5.1, deepseek-v4-flash, intern-s1) emit empty content together with
    # tool_calls and the reflection call returns "" → autogen RuntimeError. For
    # those, set False; the agent's NEXT turn (round-robin) will produce text.
    reflect_on_tool_use: bool = True
    # Per-agent flag lets callers toggle which retrieval tools are exposed.
    # Defaults mirror test_agent_debug.py.
    enabled_retrieval_tools: tuple[str, ...] = field(
        default_factory=lambda: ("web_search", "web_fetch", "literature_search", "literature_fetch")
    )


def create_model_client_full(cfg: BoyueFullAgentConfig) -> OpenAIChatCompletionClient:
    """OpenAI-compatible client with function-calling enabled."""
    return OpenAIChatCompletionClient(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )


def _retrieval_workbench(enabled: tuple[str, ...]) -> StaticWorkbench:
    registry: dict[str, FunctionTool] = {
        "web_search": FunctionTool(
            web_search,
            description=(
                "Search general web sources and return result snippets with URLs. "
                "Use for websites, news, docs. If snippets are insufficient, follow "
                "up with web_fetch."
            ),
        ),
        "web_fetch": FunctionTool(
            web_fetch,
            description=(
                "Fetch the full text of 1-3 web pages or PDFs. Use after web_search "
                "when snippets are not enough, or when you already have a specific "
                "URL. Automatically handles PDFs."
            ),
        ),
        "literature_search": FunctionTool(
            literature_search,
            description=(
                "Search scholarly literature and return candidate papers with "
                "citation-style metadata (title, authors, year, DOI, venue, "
                "snippet). METADATA ONLY — if you need body text, call "
                "literature_fetch instead."
            ),
        ),
        "literature_fetch": FunctionTool(
            literature_fetch,
            description=(
                "Download and parse full-text papers into markdown. Multiple "
                "channels: CrossRef, Unpaywall, arXiv, PMC, bioRxiv. SLOW "
                "(tens of seconds to minutes per call). Use ONLY when body "
                "text is required. Defaults: num_results=3, max_success=2, "
                "max_markdown_chars=8000. Call AT MOST ONCE per question."
            ),
        ),
    }
    return StaticWorkbench([registry[name] for name in enabled if name in registry])


async def build_agent_team_full(cfg: BoyueFullAgentConfig) -> RoundRobinGroupChat:
    """Build the full tool-enabled RoundRobinGroupChat team.

    Retains a reference to the model client on the returned team as
    ``team._eval_model_client`` so the runner can ``close`` it later.
    """
    model_client = create_model_client_full(cfg)
    retrieval_wb = _retrieval_workbench(cfg.enabled_retrieval_tools)
    workbench: list[Any] = [retrieval_wb]
    tu_wb = await _get_tooluniverse_workbench(cfg)
    if tu_wb is not None:
        # Prepend ToolUniverse so `find_tools` appears first in the agent's
        # tool list, matching test_agent_debug.py ordering.
        workbench.insert(0, tu_wb)

    agent = AssistantAgent(
        name="EvalAgent",
        model_client=model_client,
        workbench=workbench,
        reflect_on_tool_use=cfg.reflect_on_tool_use,
        max_tool_iterations=cfg.max_tool_iterations,
        description=f"Tool-enabled evaluation agent backed by {cfg.model}.",
        system_message=_SYSTEM_MESSAGE,
    )

    termination = TextMentionTermination(
        "TERMINATE", sources=["EvalAgent"]
    ) | MaxMessageTermination(cfg.max_messages)
    team = RoundRobinGroupChat([agent], termination_condition=termination)
    # Stash the client so runner can close() it cleanly without groping into
    # private team internals.
    team._eval_model_client = model_client  # type: ignore[attr-defined]
    return team
