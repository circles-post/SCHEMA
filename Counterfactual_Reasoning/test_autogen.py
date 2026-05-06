"""Standalone runner for the ToolUniverse + WebSearch agent (no agdebugger).

Usage:
    # Interactive mode (type questions, Ctrl+D / 'exit' to quit)
    python test_autogen.py

    # Single question mode
    python test_autogen.py --task "What pathways involve UniProt protein P04637?"

    # Read question from file
    python test_autogen.py --task-file question.txt
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool, StaticWorkbench
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


from websearch_tools import web_search, web_fetch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_MODEL_NAME = os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o")
_MODEL_API_KEY = os.environ.get("AGENTDEBUG_OPENAI_API_KEY", "")
_MODEL_BASE_URL = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "")
_TOOLUNIVERSE_DIR = os.environ.get("TOOLUNIVERSE_DIR", "")


def _mask_secret(value: str, *, visible_prefix: int = 6, visible_suffix: int = 4) -> str:
    if not value:
        return "(empty)"
    if len(value) <= visible_prefix + visible_suffix:
        return value[0] + "***" if len(value) > 1 else "*"
    return f"{value[:visible_prefix]}***{value[-visible_suffix:]}"


def get_runtime_model_config() -> dict[str, str]:
    return {
        "model_name": _MODEL_NAME,
        "api_base_url": _MODEL_BASE_URL,
        "api_key_masked": _mask_secret(_MODEL_API_KEY),
    }

# ---------------------------------------------------------------------------
# System prompt (same as test_agent_debug.py)
# ---------------------------------------------------------------------------
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

## Web Search Strategy

1. **When to use web search**:
   - Questions requiring up-to-date information not covered by ToolUniverse tools
   - Verifying facts, finding references, or retrieving specific data from the web
   - When ToolUniverse tools are not available or suitable for the task

2. **How to use web search**:
   - Call `web_search` first to get an overview of available sources (snippets + URLs)
   - If the snippets contain enough information, use them directly - no further fetching needed
   - If more detail is required, call `web_fetch` with 1-3 URLs from the search results
   - `web_fetch` automatically handles both web pages and PDFs

3. **Search → Fetch decision**:
   - Snippets sufficient: factual lookups, simple definitions, quick verification
   - Fetch needed: detailed methodology, full paper content, comprehensive data

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


# ---------------------------------------------------------------------------
# Build agent team
# ---------------------------------------------------------------------------
def _tooluniverse_env() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": _MODEL_API_KEY,
        "OPENAI_BASE_URL": _MODEL_BASE_URL,
        "OPENAI_API_BASE": _MODEL_BASE_URL,
        "VLLM_API_KEY": _MODEL_API_KEY,
        "TOOLUNIVERSE_LLM_CONFIG_MODE": "env_override",
        "TOOLUNIVERSE_LLM_DEFAULT_PROVIDER": "VLLM",
        "TOOLUNIVERSE_LLM_MODEL_DEFAULT": "gpt-4o-mini",
        "VLLM_SERVER_URL": _MODEL_BASE_URL,
        "AGENTIC_TOOL_FALLBACK_CHAIN": '[{"api_type":"VLLM","model_id":"gpt-4o-mini"}]',
    }


async def build_team() -> tuple[RoundRobinGroupChat, McpWorkbench, OpenAIChatCompletionClient]:
    """Build and return (team, workbench, model_client) for manual lifecycle management."""

    # MCP workbench
    server_params = StdioServerParams(
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
        read_timeout_seconds=120,
    )
    workbench = McpWorkbench(server_params=server_params)
    await workbench.start()
    tools = await workbench.list_tools()
    print(f"[ToolUniverse MCP] connected. Tool count: {len(tools)}")

    # Web search workbench
    _web_search_tool = FunctionTool(
        web_search,
        description=(
            "Search the web and return result snippets with URLs. "
            "Call this first to get an overview of available sources. "
            "If the snippets are insufficient, follow up with web_fetch."
        ),
    )
    _web_fetch_tool = FunctionTool(
        web_fetch,
        description=(
            "Fetch the full text of one or more web pages or PDFs. "
            "Use after web_search when snippets do not contain enough detail. "
            "Pass 1-3 URLs from the search results. Automatically handles PDFs."
        ),
    )
    web_workbench = StaticWorkbench([_web_search_tool, _web_fetch_tool])

    # Model client
    model_client = OpenAIChatCompletionClient(
        model=_MODEL_NAME,
        api_key=_MODEL_API_KEY,
        base_url=_MODEL_BASE_URL,
    )
    runtime_config = get_runtime_model_config()
    print(
        "[AutoGen] Effective model config: "
        f"model={runtime_config['model_name']}, "
        f"base_url={runtime_config['api_base_url']}, "
        f"api_key={runtime_config['api_key_masked']}"
    )

    # Agent + team
    agent = AssistantAgent(
        name="ToolUniverseAgent",
        model_client=model_client,
        workbench=[workbench, web_workbench],
        reflect_on_tool_use=True,
        max_tool_iterations=5,
        description="Specialist agent for scientific tooling exposed by the ToolUniverse MCP server.",
        system_message=_SYSTEM_MESSAGE,
    )

    termination = (
        TextMentionTermination("TERMINATE", sources=["ToolUniverseAgent"])
        | MaxMessageTermination(20)
    )
    team = RoundRobinGroupChat(
        [agent],
        termination_condition=termination,
    )
    return team, workbench, model_client


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------
async def run_task(team: RoundRobinGroupChat, task: str) -> str:
    """Run a single task through the team, streaming each step to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Task: {task[:200]}")
    print("=" * 60)

    final_text = ""
    async for message in team.run_stream(task=task):
        # Each message is an event from the agent (tool call, tool result, text, etc.)
        if hasattr(message, "to_text"):
            text = message.to_text()
        else:
            text = str(message)

        # Identify message type for clear labeling
        msg_type = type(message).__name__
        source = getattr(message, "source", "")
        header = f"[{msg_type}]" + (f" ({source})" if source else "")

        print(f"\n{'─' * 60}")
        print(header)
        print("─" * 60)
        print(text[:3000] if text else "(empty)")

        # Keep track of the last substantive text for return value
        if text and "TaskResult" not in msg_type:
            final_text = text

    return final_text


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
async def interactive_loop(team: RoundRobinGroupChat) -> None:
    """Read questions from stdin and run them one at a time."""
    print("\n" + "=" * 60)
    print("Interactive mode. Type your question and press Enter.")
    print("Type 'exit' or Ctrl+D to quit.")
    print("=" * 60)

    while True:
        try:
            task = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not task or task.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        try:
            # Reset team state so each question starts fresh
            await team.reset()
            await run_task(team, task)
        except Exception as e:
            print(f"\n[ERROR] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(task: str | None = None, task_file: str | None = None) -> None:
    team, workbench, model_client = await build_team()
    try:
        if task_file:
            with open(task_file, "r") as f:
                task = f.read().strip()

        if task:
            await run_task(team, task)
        else:
            await interactive_loop(team)
    finally:
        await model_client.close()
        await workbench.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ToolUniverse + WebSearch agent")
    parser.add_argument(
        "--task", type=str, default=None,
        help="Single question to answer (non-interactive mode).",
    )
    parser.add_argument(
        "--task-file", type=str, default=None,
        help="Read the task from a text file.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(task=args.task, task_file=args.task_file))


if __name__ == "__main__":
    main()
