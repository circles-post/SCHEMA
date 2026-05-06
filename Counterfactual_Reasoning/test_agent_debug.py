import asyncio
import json
import os
import re
import sys
import time
from itertools import count
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool, StaticWorkbench
from autogen_core.models import AssistantMessage, FunctionExecutionResultMessage, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai import _openai_client as _autogen_openai_client_module
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, StreamableHttpServerParams

from model_routing import (
    build_openai_extra_body,
    merge_openai_extra_create_args,
    resolve_base_url_for_model,
    resolve_value_for_model,
)

_AISCI_ROOT = Path(__file__).resolve().parents[2]
_SCIVERSE_DIR = _AISCI_ROOT / "sciverse"
if str(_SCIVERSE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCIVERSE_DIR))

# sciverse_client.py reads SCIVERSE_API_TOKEN at module-import time via
# load_dotenv(); but dotenv looks in cwd, not in sciverse/'s own dir. If the
# shell env doesn't already have SCIVERSE_API_TOKEN we'd get 401 from
# api.opendatalab.org.cn. Load the sciverse .env explicitly BEFORE importing
# sciverse_tools so the token is already in os.environ when the module loads.
try:
    from dotenv import load_dotenv as _load_dotenv

    _SCIVERSE_ENV_FILE = _SCIVERSE_DIR / ".env"
    if _SCIVERSE_ENV_FILE.is_file():
        _load_dotenv(dotenv_path=_SCIVERSE_ENV_FILE, override=False)
except ImportError:
    pass

from sciverse_tools import literature_search, sciverse_fetch_markdown
from websearch_tools import web_search, web_fetch


# Per-process semaphore guarding sciverse_fetch_markdown so multiple parallel
# workers (or multiple in-flight tool calls within one worker) don't pile a
# herd of mineru subprocesses + PDF downloads onto the same machine. Default
# is 1, override with AGDEBUGGER_LITERATURE_FETCH_CONCURRENCY.
_LITERATURE_FETCH_CONCURRENCY = max(
    1, int(os.environ.get("AGDEBUGGER_LITERATURE_FETCH_CONCURRENCY", "1"))
)
_LITERATURE_FETCH_SEMAPHORE: "asyncio.Semaphore | None" = None


def _get_literature_fetch_semaphore() -> "asyncio.Semaphore":
    """Lazy-initialize the semaphore so it binds to the running event loop.

    Creating asyncio.Semaphore at module-import time can race with
    multi-loop test setups; bind on first use instead.
    """
    global _LITERATURE_FETCH_SEMAPHORE
    if _LITERATURE_FETCH_SEMAPHORE is None:
        import asyncio as _asyncio
        _LITERATURE_FETCH_SEMAPHORE = _asyncio.Semaphore(_LITERATURE_FETCH_CONCURRENCY)
    return _LITERATURE_FETCH_SEMAPHORE


async def literature_fetch(
    query: str,
    num_results: int = 3,
    max_success: int = 2,
    max_markdown_chars: int = 8000,
) -> str:
    """Search scholarly literature, download PDFs, parse them to markdown, and
    return the parsed full text for the top results.

    This is a thin wrapper over ``sciverse_fetch_markdown`` that:
      * bakes in sane defaults (``convert_to_md=True``, ``include_markdown_content=True``)
      * fixes ``max_workers=4`` so the LLM can't accidentally spawn a storm
      * clamps inputs to the same safety ranges used by ``sciverse_fetch_markdown``

    Compared to ``literature_search`` which only returns citation-style metadata,
    this tool actually downloads the paper PDFs (via multi-source fallback:
    CrossRef, Unpaywall, arXiv, PMC, bioRxiv, PubMed/PMC, …) and runs them
    through MinerU to produce LLM-readable markdown body text.

    It is SLOW (seconds-to-minutes per paper) and each call can fail individual
    papers when the download channel or MinerU stage has issues. Use it only
    when the agent specifically needs paper **body text** (methods, figures,
    equations, full discussion) — for title/author/abstract snippets prefer
    ``literature_search``.

    Parameters
    ----------
    query:
        Title-like or topic-like search query. Prefer 3-8 entity-centric tokens.
    num_results:
        How many candidate papers to consider from the search. Clamped to [1, 10].
    max_success:
        How many papers to actually download + parse. Clamped to [1, num_results].
    max_markdown_chars:
        Per-paper cap on returned markdown body text. Clamped to [1000, 20000].
    """
    num_results = max(1, min(10, int(num_results)))
    max_success = max(1, min(num_results, int(max_success)))
    max_markdown_chars = max(1000, min(20000, int(max_markdown_chars)))
    sem = _get_literature_fetch_semaphore()
    async with sem:
        return await sciverse_fetch_markdown(
            query=query,
            num_results=num_results,
            max_success=max_success,
            convert_to_md=True,
            include_markdown_content=True,
            max_markdown_chars=max_markdown_chars,
            max_workers=4,
        )

# Ensure HTTP proxy is configured so external LLM endpoints are reachable.
# Set http_proxy / https_proxy in your shell before launching if your network
# requires an upstream proxy to reach the LLM gateway.
_PROXY_URL = os.environ.get("http_proxy", "")
if _PROXY_URL:
    os.environ.setdefault("http_proxy", _PROXY_URL)
    os.environ.setdefault("https_proxy", _PROXY_URL)

# Fix no_proxy so the LLM endpoint goes through the proxy:
# 1) httpx does not support CIDR notation — strip subnet masks
# 2) Remove the LLM endpoint host (set via AGENTDEBUG_LLM_HOST) so it is NOT
#    bypassed by no_proxy.
_no_proxy = os.environ.get("no_proxy", "")
_fixed_no_proxy = re.sub(r"(\d+\.\d+\.\d+\.\d+)/\d+", r"\1", _no_proxy)
_LLM_HOST = os.environ.get("AGENTDEBUG_LLM_HOST", "").strip()
if _LLM_HOST:
    _fixed_no_proxy = ",".join(
        e for e in _fixed_no_proxy.split(",") if e.strip() != _LLM_HOST
    )
os.environ["no_proxy"] = _fixed_no_proxy
os.environ["NO_PROXY"] = _fixed_no_proxy

_MODEL_NAME = os.environ.get(
    "AGENTDEBUG_MODEL_AGENT",
    os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini"),
)
_MCP_MODEL_NAME = os.environ.get(
    "AGENTDEBUG_MODEL_MCP",
    os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini"),
)
_DEFAULT_MODEL_API_KEY = os.environ.get("AGENTDEBUG_OPENAI_API_KEY", "")
_INTERN_MODEL_API_KEY = os.environ.get("AGENTDEBUG_INTERN_API_KEY")
_MODEL_API_KEY = os.environ.get("AGENTDEBUG_OPENAI_API_KEY_AGENT") or resolve_value_for_model(
    _MODEL_NAME,
    _DEFAULT_MODEL_API_KEY,
    intern_value=_INTERN_MODEL_API_KEY,
)
_MCP_MODEL_API_KEY = os.environ.get("AGENTDEBUG_OPENAI_API_KEY_MCP") or resolve_value_for_model(
    _MCP_MODEL_NAME,
    _DEFAULT_MODEL_API_KEY,
    intern_value=_INTERN_MODEL_API_KEY,
)
_DEFAULT_MODEL_BASE_URL = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "")
_MODEL_BASE_URL = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL_AGENT") or resolve_base_url_for_model(
    _MODEL_NAME,
    _DEFAULT_MODEL_BASE_URL,
)
_MCP_MODEL_BASE_URL = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL_MCP") or resolve_base_url_for_model(
    _MCP_MODEL_NAME,
    _DEFAULT_MODEL_BASE_URL,
)
_MODEL_TIMEOUT_SEC = float(os.environ.get("AGENTDEBUG_MODEL_TIMEOUT_SEC", "300"))
_MODEL_MAX_RETRIES = int(os.environ.get("AGENTDEBUG_MODEL_MAX_RETRIES", "2"))
_MODEL_EXTRA_BODY = build_openai_extra_body()

_TOOLUNIVERSE_DIR = os.environ.get("TOOLUNIVERSE_DIR", "")
_TOOLUNIVERSE_URL = os.environ.get("AGDEBUGGER_TOOLUNIVERSE_URL", "").strip()

_TOOLUNIVERSE_WORKBENCH: McpWorkbench | None = None
_LLM_CALL_SEQ = count(1)


def _safe_add_usage(usage1, usage2):
    usage_type = type(usage1)
    return usage_type(
        prompt_tokens=(getattr(usage1, "prompt_tokens", 0) or 0) + (getattr(usage2, "prompt_tokens", 0) or 0),
        completion_tokens=(getattr(usage1, "completion_tokens", 0) or 0)
        + (getattr(usage2, "completion_tokens", 0) or 0),
    )


_autogen_openai_client_module._add_usage = _safe_add_usage


def _message_preview(messages: Sequence[SystemMessage | UserMessage | AssistantMessage | FunctionExecutionResultMessage]) -> str:
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            compact = " ".join(content.split())
            return compact[:160]
    return ""


def _log_llm_event(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, **payload},
            ensure_ascii=False,
            default=str,
        )
    )


_RATE_LIMIT_MAX_RETRIES = int(os.environ.get("AGENTDEBUG_RATE_LIMIT_MAX_RETRIES", "5"))
_RATE_LIMIT_BASE_DELAY = float(os.environ.get("AGENTDEBUG_RATE_LIMIT_BASE_DELAY", "10"))
_FC_PARSE_MAX_RETRIES = int(os.environ.get("AGENTDEBUG_FC_PARSE_MAX_RETRIES", "1"))


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if the exception is an intern-s1 token rate limit error (-20053)."""
    msg = str(exc)
    return "-20053" in msg or "限流" in msg


def _is_fc_parse_error(exc: Exception) -> bool:
    """Check if the exception is an intern-s1 FC format parse error (-20009)."""
    return "-20009" in str(exc)


_REFLECTION_SANITIZE_COUNT = 0
_REFLECTION_SANITIZE_MAX = int(os.environ.get("AGDEBUGGER_MAX_REFLECTION_SANITIZE", "2"))


def reset_reflection_sanitize_counter():
    """Reset the reflection sanitize counter.  Called by the backend's
    team_reset endpoint before each new question so the convergence
    guard starts fresh."""
    global _REFLECTION_SANITIZE_COUNT
    _REFLECTION_SANITIZE_COUNT = 0


class LoggingOpenAIChatCompletionClient(OpenAIChatCompletionClient):
    async def create(
        self,
        messages: Sequence[SystemMessage | UserMessage | AssistantMessage | FunctionExecutionResultMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Any = None,
    ):
        call_id = next(_LLM_CALL_SEQ)
        started = time.monotonic()
        _log_llm_event(
            "tooluniverse_llm_call_start",
            call_id=call_id,
            model=_MODEL_NAME,
            base_url=_MODEL_BASE_URL,
            timeout_sec=_MODEL_TIMEOUT_SEC,
            extra_body=_MODEL_EXTRA_BODY or None,
            message_count=len(messages),
            tool_count=len(tools),
            preview=_message_preview(messages),
        )

        merged_extra_create_args = merge_openai_extra_create_args(extra_create_args)
        last_exc: Exception | None = None
        max_attempts = 1 + _RATE_LIMIT_MAX_RETRIES  # 1 initial + retries

        for attempt in range(max_attempts):
            try:
                result = await super().create(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    json_output=json_output,
                    extra_create_args=merged_extra_create_args,
                    cancellation_token=cancellation_token,
                )
                if attempt > 0:
                    _log_llm_event(
                        "tooluniverse_llm_call_retry_success",
                        call_id=call_id,
                        model=_MODEL_NAME,
                        attempt=attempt + 1,
                        elapsed_sec=round(time.monotonic() - started, 2),
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                retryable = False
                if _is_rate_limit_error(exc):
                    delay = _RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                    retryable = attempt < _RATE_LIMIT_MAX_RETRIES
                    label = "rate_limit"
                elif _is_fc_parse_error(exc):
                    delay = 1.0
                    retryable = attempt < _FC_PARSE_MAX_RETRIES
                    label = "fc_parse"
                else:
                    retryable = False

                if retryable:
                    _log_llm_event(
                        "tooluniverse_llm_call_retry",
                        call_id=call_id,
                        model=_MODEL_NAME,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        reason=label,
                        delay_sec=delay,
                        error=str(exc)[:200],
                    )
                    await asyncio.sleep(delay)
                    continue

                # Not retryable — log and raise
                _log_llm_event(
                    "tooluniverse_llm_call_error",
                    call_id=call_id,
                    model=_MODEL_NAME,
                    elapsed_sec=round(time.monotonic() - started, 2),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
        else:
            # Exhausted all retries
            _log_llm_event(
                "tooluniverse_llm_call_error",
                call_id=call_id,
                model=_MODEL_NAME,
                elapsed_sec=round(time.monotonic() - started, 2),
                error_type=type(last_exc).__name__ if last_exc else "Unknown",
                error=str(last_exc) if last_exc else "max retries exhausted",
                retries_exhausted=True,
            )
            if last_exc:
                raise last_exc

        usage = getattr(result, "usage", None)
        _log_llm_event(
            "tooluniverse_llm_call_end",
            call_id=call_id,
            model=_MODEL_NAME,
            elapsed_sec=round(time.monotonic() - started, 2),
            finish_reason=getattr(result, "finish_reason", None),
            usage=usage,
        )
        # --- intern-s1 reflection sanitizer ---
        # When autogen's reflect_on_tool_use fires (tools=(), tool_choice="none"),
        # intern-s1 sometimes ignores tool_choice and emits its private
        # <|action_start|><|plugin|> markup inside the text content.  autogen sees
        # tool_calls=null and wraps the raw markup as a TextMessage, which then
        # poisons the conversation history and cascades into malformed follow-ups
        # ending in an empty completion.
        #
        # Fix: strip the markup and keep only the preceding natural-language text.
        # This runs *before* the empty-completion check so that a cleaned-up
        # response is no longer mistakenly flagged as empty.
        #
        # Convergence guard: if the sanitizer fires more than
        # _REFLECTION_SANITIZE_MAX times in the same session, the model is
        # stuck in a loop where every reflection attempt tries to call tools.
        # In that case, append a strong nudge telling the agent to provide its
        # final answer on the next turn instead of searching again.
        global _REFLECTION_SANITIZE_COUNT

        # --- intern-s1 reflection coercion (structured tool_calls variant) ---
        # Sibling case of the <|action_start|> sanitizer below: when autogen's
        # reflect_on_tool_use fires (tools=(), tool_choice="none"), intern-s1
        # sometimes ignores tool_choice and returns structured tool_calls
        # (finish_reason="function_calls", content=List[FunctionCall]) instead
        # of a text summary. autogen's _reflect_on_tool_use_flow then raises
        # `Reflect on tool use produced no valid text response.` because
        # `isinstance(result.content, str)` is False, killing the whole run.
        #
        # Fix: coerce the tool_calls payload into a short natural-language
        # salvage string so reflection can finish. Reuses the same convergence
        # counter as the text-marker sanitizer, so repeated offenders still
        # trigger the "use <answer> now" nudge.
        if not tools and not isinstance(getattr(result, "content", None), str):
            raw_calls = result.content if isinstance(result.content, list) else []

            # Build a DIAGNOSTIC-ONLY dump of the tool-call payload. This never
            # enters the trajectory; it only lands in _log_llm_event so we can
            # inspect what intern-s1 tried to do after the fact.
            diag_parts: list[str] = []
            merged_args_chars = 0
            for tc in raw_calls:
                tc_name = getattr(tc, "name", None) or "unknown_tool"
                tc_args = getattr(tc, "arguments", None) or ""
                if not isinstance(tc_args, str):
                    tc_args = str(tc_args)
                merged_args_chars += len(tc_args)
                if len(diag_parts) < 10:  # cap diagnostic detail
                    snippet = tc_args if len(tc_args) <= 120 else tc_args[:120] + "...(truncated)"
                    diag_parts.append(f"{tc_name}:{snippet}")
            diag_dump = " | ".join(diag_parts) or "(no tool_calls)"

            # TRAJECTORY-SAFE salvage: a single short, fixed sentence. Never
            # include per-call fragments, arguments, names, or URLs here —
            # intern-s1 sometimes returns token-level partial FunctionCall
            # entries (name=None, arguments=<single token>), and echoing them
            # back into the assistant message poisons downstream reasoning and
            # planner-generated replacement_text.
            salvage_text = (
                "(The previous turn attempted a tool call instead of providing a "
                "text summary. Disregard that attempt. On the next turn, commit to "
                "one of the options from the question's option list and return it "
                "wrapped in answer tags — the content inside the tags must be the "
                "concrete identifier of the option you choose, for example "
                "<answer>option2</answer> or <answer>optionA</answer>. Do not "
                "echo this instruction, do not emit a literal placeholder such "
                "as optionN, and do not leave the answer tags empty.)"
            )
            _REFLECTION_SANITIZE_COUNT += 1
            _log_llm_event(
                "tooluniverse_llm_reflection_coerced",
                call_id=call_id,
                model=_MODEL_NAME,
                tool_call_count=len(raw_calls),
                merged_args_chars=merged_args_chars,
                sanitize_count=_REFLECTION_SANITIZE_COUNT,
                original_finish_reason=getattr(result, "finish_reason", None),
                diagnostic_dump=diag_dump[:1000],
            )
            if _REFLECTION_SANITIZE_COUNT >= _REFLECTION_SANITIZE_MAX:
                salvage_text += (
                    "\n\n[SYSTEM NOTE] You have used all available tool-calling rounds. "
                    "Do NOT search or fetch again. Based on the information already gathered, "
                    "commit to one specific option from the question's option list and return it "
                    "wrapped in answer tags, then emit TERMINATE on the next line. The content "
                    "inside the answer tags must be the concrete identifier of the option you "
                    "choose (for example optionA, option2, or whatever form the question uses) — "
                    "do not echo this instruction verbatim, do not emit a literal placeholder, "
                    "and do not leave the tags empty."
                )
            result = result.model_copy(update={"content": salvage_text, "finish_reason": "stop"})

        _ACTION_START_MARKER = "<|action_start|>"
        if (
            not tools
            and isinstance(getattr(result, "content", None), str)
            and _ACTION_START_MARKER in result.content
        ):
            _REFLECTION_SANITIZE_COUNT += 1
            clean_text = result.content[: result.content.index(_ACTION_START_MARKER)].rstrip()
            _log_llm_event(
                "tooluniverse_llm_reflection_sanitized",
                call_id=call_id,
                model=_MODEL_NAME,
                original_len=len(result.content),
                clean_len=len(clean_text),
                sanitize_count=_REFLECTION_SANITIZE_COUNT,
                stripped_suffix=result.content[result.content.index(_ACTION_START_MARKER):][:200],
            )
            if not clean_text:
                clean_text = "(The model attempted a tool call instead of providing a summary.)"
            if _REFLECTION_SANITIZE_COUNT >= _REFLECTION_SANITIZE_MAX:
                clean_text += (
                    "\n\n[SYSTEM NOTE] You have used all available tool-calling rounds. "
                    "Do NOT search or fetch again. Based on the information already gathered, "
                    "commit to one specific option from the question's option list and return it "
                    "wrapped in answer tags, then emit TERMINATE on the next line. The content "
                    "inside the answer tags must be the concrete identifier of the option you "
                    "choose (for example optionA, option2, or whatever form the question uses) — "
                    "do not echo this instruction verbatim, do not emit a literal placeholder, "
                    "and do not leave the tags empty."
                )
            result = result.model_copy(update={"content": clean_text, "finish_reason": "stop"})

        if isinstance(getattr(result, "content", None), str) and not result.content.strip():
            _log_llm_event(
                "tooluniverse_llm_empty_completion",
                call_id=call_id,
                model=_MODEL_NAME,
                elapsed_sec=round(time.monotonic() - started, 2),
                finish_reason=getattr(result, "finish_reason", None),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                thought_present=bool(getattr(result, "thought", None)),
                message_count=len(messages),
                tool_count=len(tools),
                preview=_message_preview(messages),
            )
        return result


def _create_model_client() -> OpenAIChatCompletionClient:
    return LoggingOpenAIChatCompletionClient(
        model=_MODEL_NAME,
        api_key=_MODEL_API_KEY,
        base_url=_MODEL_BASE_URL,
        timeout=_MODEL_TIMEOUT_SEC,
        max_retries=_MODEL_MAX_RETRIES,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )


def _tooluniverse_env() -> dict[str, str]:
    env = {
        # OpenAI-style envs for components that inspect these directly.
        "OPENAI_API_KEY": _MCP_MODEL_API_KEY,
        "OPENAI_BASE_URL": _MCP_MODEL_BASE_URL,
        "OPENAI_API_BASE": _MCP_MODEL_BASE_URL,
        # ToolUniverse LLM tools use the VLLM provider for OpenAI-compatible endpoints.
        "TOOLUNIVERSE_LLM_CONFIG_MODE": "env_override",
        "TOOLUNIVERSE_LLM_DEFAULT_PROVIDER": "VLLM",
        "TOOLUNIVERSE_LLM_MODEL_DEFAULT": _MCP_MODEL_NAME,
        "VLLM_SERVER_URL": _MCP_MODEL_BASE_URL,
        # Keep the fallback chain on the same OpenAI-compatible endpoint.
        "AGENTIC_TOOL_FALLBACK_CHAIN": json.dumps(
            [{"api_type": "VLLM", "model_id": _MCP_MODEL_NAME}]
        ),
    }
    for key in ("UV_CACHE_DIR", "XDG_CACHE_HOME", "HOME", "PATH", "TOOLUNIVERSE_CACHE_DIR"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["FASTMCP_SHOW_SERVER_BANNER"] = "false"
    env["TOOLUNIVERSE_SUPPRESS_STDIO_BANNER"] = "1"
    thinking_mode_raw = os.environ.get("AGENTDEBUG_THINKING_MODE")
    if thinking_mode_raw is not None:
        env["AGENTDEBUG_THINKING_MODE"] = thinking_mode_raw
    if _MODEL_EXTRA_BODY:
        env["OPENAI_EXTRA_BODY"] = json.dumps(_MODEL_EXTRA_BODY)
    return env


def _build_tooluniverse_server_params() -> StreamableHttpServerParams | StdioServerParams:
    if _TOOLUNIVERSE_URL:
        return StreamableHttpServerParams(
            url=_TOOLUNIVERSE_URL,
            timeout=30.0,
            sse_read_timeout=300.0,
            terminate_on_close=False,
        )

    return StdioServerParams(
        command="uv",
        args=[
            "--directory",
            _TOOLUNIVERSE_DIR,
            "run",
            "tooluniverse-smcp-stdio",
            "--exclude-tool-types",
            "PackageTool",
            "--compact-mode",
        ],
        env=_tooluniverse_env(),
        # ToolUniverse startup is heavier than the default 5s timeout.
        read_timeout_seconds=120,
    )


async def _get_tooluniverse_workbench() -> McpWorkbench:
    global _TOOLUNIVERSE_WORKBENCH

    if _TOOLUNIVERSE_WORKBENCH is None:
        server_params = _build_tooluniverse_server_params()
        _TOOLUNIVERSE_WORKBENCH = McpWorkbench(server_params=server_params)
        await _TOOLUNIVERSE_WORKBENCH.start()
        try:
            # Use asyncio.shield() so that a timeout does NOT cancel the
            # underlying actor future.  Without shield, asyncio.wait_for
            # cancels the inner task on timeout, which puts the actor's
            # command-queue future into a cancelled state.  When the actor
            # later tries set_result() on that future it gets
            # InvalidStateError and the whole actor task crashes.
            tools = await asyncio.wait_for(
                asyncio.shield(_TOOLUNIVERSE_WORKBENCH.list_tools()),
                timeout=30,
            )
            if _TOOLUNIVERSE_URL:
                print(
                    f"[ToolUniverse MCP] connected via shared HTTP server {_TOOLUNIVERSE_URL}. "
                    f"Tool count: {len(tools)}"
                )
            else:
                print(f"[ToolUniverse MCP] connected. Tool count: {len(tools)}")
        except asyncio.TimeoutError:
            print(
                "[ToolUniverse MCP] started, but eager list_tools did not "
                "return within 30 s; the actor is still alive and will "
                "serve requests once the SMCP server finishes loading."
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[ToolUniverse MCP] connected, but eager list_tools failed; "
                f"continuing without startup tool count ({type(exc).__name__}: {exc})"
            )

    return _TOOLUNIVERSE_WORKBENCH


_SYSTEM_MESSAGE = """\
## General Policy

1. **Actively use ToolUniverse tools** when they can help answer scientific questions.
2. For scientific/technical questions, always attempt to find relevant tools before relying solely on general knowledge.
3. Use `find_tools` to discover appropriate ToolUniverse tools for the task.
4. Balance tool usage with reasoning - tools should enhance, not replace, analytical thinking.

## Tool Search Strategy

1. **When to search for tools**:
   - Scientific calculations, simulations, or data analysis
   - Domain-specific queries (biology, chemistry, physics, etc.)
   - Questions requiring specialized knowledge or databases
   - Tasks that could benefit from computational tools

2. **How to search**:
   - Build focused queries from key entities and domain terms
   - Example: `"protein structure analysis PDB"`, `"molecular dynamics simulation"`, `"gene expression analysis"`
   - If first search doesn't find suitable tools, try a broader query
   - Maximum 2-3 search attempts per question to avoid loops

3. **Tool validation**:
   - Check if the tool directly addresses the question
   - Verify required parameters are available
   - Ensure the tool output will help answer the question
   - Always use the **exact parameter names** defined in each tool's schema
   - If unsuitable, proceed with reasoning and available knowledge

## General Web Retrieval Strategy

1. **When to use general web retrieval**:
   - Questions requiring up-to-date information not covered by ToolUniverse tools
   - Verifying facts, finding references, or retrieving specific data from websites
   - When ToolUniverse tools are not available or suitable for the task

2. **How to use general web retrieval**:
   - Call `web_search` first to get an overview of available sources (snippets + URLs)
   - If the snippets contain enough information, use them directly - no further fetching needed
   - If more detail is required, call `web_fetch` with 1-3 URLs from the search results
   - `web_fetch` is for full text retrieval from known URLs and automatically handles both web pages and PDFs

3. **Search → Fetch decision**:
   - Snippets sufficient: factual lookups, simple definitions, quick verification
   - Fetch needed: detailed methodology, full page content, comprehensive data

## Literature Search Strategy

1. **When to use literature search**:
   - Questions asking for paper-backed scientific evidence
   - Requests about studies, experiments, results, authors, DOI, journals, or publication history
   - Cases where you need to identify relevant scholarly papers before reading full text

2. **How to use literature search**:
   - Call `literature_search` to retrieve candidate papers and citation-style metadata (title, authors, DOI, venue, snippet) — this is METADATA ONLY
   - Call `literature_fetch` when you need the **body text** of one or more papers (methods, figures, equations, detailed discussion). It downloads PDFs from multiple sources (CrossRef/Unpaywall/arXiv/PMC/bioRxiv/…) and runs MinerU to produce markdown
   - Prefer `literature_search` over `web_search` when the target is academic literature rather than general webpages
   - If you already have a concrete paper URL or PDF URL, use `web_fetch` to read it in full

3. **Literature vs. web**:
   - `literature_search`: discover papers and scholarly metadata (fast, cheap; use freely)
   - `literature_fetch`: download + parse paper full text to markdown (SLOW, call AT MOST ONCE per question; use only when body text is actually needed)
   - `web_search`: discover general web sources and URLs
   - `web_fetch`: retrieve full text from specific URLs

## Multiple-Choice Question Answer Format

For MCQ tasks:
1. Consider using tools to verify facts or perform calculations if needed.
2. Analyze the question and evaluate each option systematically.
3. Output complete reasoning for option evaluation.
4. **Answer format**: Use the option letter/number (A, B, C, D, etc.) in the `<answer>` tag.
   - **CORRECT**: `<answer>A</answer>` or `<answer>B</answer>`
   - **INCORRECT**: `<answer>By taking multiple images...</answer>` (do not include option content)

## Output Behavior

1. **ALWAYS output complete reasoning path**.
2. Keep reasoning concise and decision-oriented.
3. When tools are used, explain:
   - Why the tool was selected
   - What the tool output means
   - How it contributes to the answer
4. When tools are not available or suitable:
   - Analyze the question using available context/knowledge
   - Break down the problem step by step
   - Evaluate each option (if MCQ) with explicit reasoning
   - Document the logical path to the conclusion
5. **For MCQ**: Provide final answer as the option letter in `<answer>X</answer>` format (where X is A, B, C, D, etc.).
6. **For open questions**: Provide the answer content in `<answer></answer>` tags.

**CRITICAL**: Never skip reasoning steps. The reasoning path is mandatory for all responses.

## Execution Template

1. Parse question and candidate options (if MCQ).
2. **Evaluate tool usage**:
   - For scientific/technical questions: search for relevant ToolUniverse tools using `find_tools`
   - For scholarly papers and paper-backed evidence: use `literature_search`
   - For general knowledge or web-based questions: use `web_search` (and `web_fetch` if needed)
   - If tools found: validate suitability and use if appropriate
   - If no suitable tools: proceed with reasoning
3. **Generate reasoning path**:
   - Explain tool selection and results (if tools used)
   - Analyze the question/task systematically
   - For MCQ: evaluate each option with explicit logic
   - For open questions: build logical argument step by step
   - Use tool outputs and available knowledge to support reasoning
4. Return final answer:
   - **For MCQ**: Use option letter only (A, B, C, D, etc.)
   - **For open questions**: Provide answer content
5. **CRITICAL — Termination**: After providing your final answer, you **MUST** append the exact token `TERMINATE` on a new line at the very end of your response. Without this token the system cannot detect that you have finished. Every final response must end with `TERMINATE`.

## Termination Reminder

**NEVER forget to output `TERMINATE` at the end of your final answer.** Example:

```
Based on my analysis, the answer is B because ...

<answer>B</answer>

TERMINATE
```

If you do not include `TERMINATE`, the conversation will not end and resources will be wasted.
"""


async def get_agent_team():
    model_client = _create_model_client()
    workbench = await _get_tooluniverse_workbench()

    # Retrieval tools wrapped in a StaticWorkbench
    _web_search_tool = FunctionTool(
        web_search,
        description=(
            "Search general web sources and return result snippets with URLs. "
            "Use this for websites, documentation, news, and non-paper sources. "
            "If the snippets are insufficient, follow up with web_fetch."
        ),
    )
    _web_fetch_tool = FunctionTool(
        web_fetch,
        description=(
            "Fetch the full text of one or more web pages or PDFs. "
            "Use after web_search when snippets do not contain enough detail, or when you already have a specific paper or webpage URL. "
            "Pass 1-3 URLs. Automatically handles PDFs."
        ),
    )
    _literature_search_tool = FunctionTool(
        literature_search,
        description=(
            "Search scholarly literature and return candidate papers with citation-style metadata such as title, authors, year, DOI, venue, and snippets. "
            "Use this for scientific papers and literature evidence rather than general webpages. "
            "This tool returns METADATA ONLY — if you need the full body text of a paper, use literature_fetch instead."
        ),
    )
    _literature_fetch_tool = FunctionTool(
        literature_fetch,
        description=(
            "Search, download, and parse full-text scholarly papers into LLM-readable markdown. "
            "For each query, sciverse tries multiple download channels (CrossRef, Unpaywall, arXiv, PMC, bioRxiv, PubMed/PMC, …) and runs MinerU to extract the body text, figures captions, and sections. "
            "Use ONLY when you specifically need the full-text BODY (methods, results, equations, detailed discussion) — for titles/authors/abstracts prefer literature_search. "
            "This tool is SLOW (tens of seconds to minutes per call) and individual papers may fail silently — treat any failed entries as missing. "
            "Defaults: num_results=3, max_success=2, max_markdown_chars=8000. Call this AT MOST ONCE per question."
        ),
    )
    retrieval_workbench = StaticWorkbench(
        [
            _web_search_tool,
            _web_fetch_tool,
            _literature_search_tool,
            _literature_fetch_tool,
        ]
    )

    tooluniverse_agent = AssistantAgent(
        name="ToolUniverseAgent",
        model_client=model_client,
        workbench=[workbench, retrieval_workbench],
        reflect_on_tool_use=True,
        max_tool_iterations=5,
        description="Specialist agent for scientific tooling exposed by the ToolUniverse MCP server.",
        system_message=_SYSTEM_MESSAGE,
    )

    termination = TextMentionTermination("TERMINATE", sources=["ToolUniverseAgent"]) | MaxMessageTermination(20)
    team = RoundRobinGroupChat(
        [tooluniverse_agent],
        termination_condition=termination,
    )
    return team
