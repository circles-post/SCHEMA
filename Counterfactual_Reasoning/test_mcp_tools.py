"""
MCP Tool call test for ToolUniverseAgent.

Tests four layers in order:
  Stage 1 - MCP connectivity:  connect to the stdio server and list tools
  Stage 2 - Direct tool call:  invoke tools via workbench.call_tool() without LLM
  Stage 3 - End-to-end agent:  let the LLM pick and call tools for a real task
  Stage 4 - All-tool validation: enumerate all ToolUniverse tools in compact mode,
            execute them via execute_tool, and use a model judge to assess whether
            each call appears to have succeeded.

Run:
    conda activate agentdebug
    cd /mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger
    python test_mcp_tools.py

Optional: run only specific stages
    python test_mcp_tools.py --stages 1 2
    python test_mcp_tools.py --stages 3
    python test_mcp_tools.py --stages 4 --all-tools-limit 20
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# ---------------------------------------------------------------------------
# Config (mirrors test_agent_debug.py so both files stay in sync)
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini")
MODEL_API_KEY = os.environ.get(
    "AGENTDEBUG_OPENAI_API_KEY",
    "sk-ZnvhxhwyXok91ezpbDBcObLWa8GehlZtMaqnYT3ziVwhnBzC",
)
MODEL_BASE_URL = os.environ.get(
    "AGENTDEBUG_OPENAI_BASE_URL",
    "http://34.13.73.248:3888/v1",
)
JUDGE_MODEL_NAME = os.environ.get("AGENTDEBUG_JUDGE_MODEL_NAME", "gpt-4o-mini")
TOOLUNIVERSE_DIR = os.environ.get(
    "TOOLUNIVERSE_DIR",
    "/mnt/shared-storage-user/fengxinshun/AISci/ToolUniverse/",
)
TOOLUNIVERSE_UV_PATH = os.environ.get("TOOLUNIVERSE_UV_PATH", "uv")

CORE_COMPACT_TOOLS = {
    "list_tools",
    "grep_tools",
    "get_tool_info",
    "execute_tool",
    "find_tools",
}

# ---------------------------------------------------------------------------
# Shared MCP server params (same args as test_agent_debug.py)
# ---------------------------------------------------------------------------
def _make_server_params() -> StdioServerParams:
    return StdioServerParams(
        command=TOOLUNIVERSE_UV_PATH,
        args=[
            "--directory",
            TOOLUNIVERSE_DIR,
            "run",
            "tooluniverse-smcp-stdio",
            "--exclude-tool-types",
            "PackageTool",
            "--compact-mode",
        ],
        env={
            "OPENAI_API_KEY": MODEL_API_KEY,
            "OPENAI_BASE_URL": MODEL_BASE_URL,
            "OPENAI_API_BASE": MODEL_BASE_URL,
            "TOOLUNIVERSE_LLM_CONFIG_MODE": "env_override",
            "TOOLUNIVERSE_LLM_DEFAULT_PROVIDER": "VLLM",
            "TOOLUNIVERSE_LLM_MODEL_DEFAULT": "gpt-4o",
            "VLLM_SERVER_URL": MODEL_BASE_URL,
            "AGENTIC_TOOL_FALLBACK_CHAIN": '[{"api_type":"VLLM","model_id":"gpt-4o"}]',
        },
        read_timeout_seconds=120,
    )


# ---------------------------------------------------------------------------
# Stage 1: MCP connectivity
# ---------------------------------------------------------------------------
async def stage1_mcp_connectivity(workbench: McpWorkbench) -> list:
    """Connect to the MCP server and list available tools."""
    print("\n" + "=" * 60)
    print("STAGE 1: MCP Connectivity")
    print("=" * 60)

    tools = await workbench.list_tools()
    print(f"[OK] Connected. Tool count: {len(tools)}")

    print("\nFirst 10 tools:")
    for t in tools[:10]:
        name = t["name"] if isinstance(t, dict) else t.name
        desc = t["description"] if isinstance(t, dict) else t.description
        print(f"  - {name}: {str(desc)[:70]}")

    print("\n[PASS] Stage 1 passed.\n")
    return tools


# ---------------------------------------------------------------------------
# Stage 2: Direct tool calls via workbench (no LLM)
# ---------------------------------------------------------------------------
# Each case: (display_name, tool_name, arguments, check_fn)
# check_fn(result) -> (passed: bool, detail: str)
def _extract_tool_result_payload(result: Any) -> Any:
    """Unwrap AutoGen ToolResult objects into a plain payload for validation."""
    payload = getattr(result, "result", result)
    if isinstance(payload, list):
        normalized = [getattr(item, "content", item) for item in payload]
        if len(normalized) == 1:
            return normalized[0]
        return normalized
    return payload


def _default_result_check(result: Any) -> tuple[bool, str]:
    payload = _extract_tool_result_payload(result)
    ok = bool(payload)
    preview = str(payload)[:240]
    return ok, f"type={type(result).__name__}, payload_type={type(payload).__name__}, preview={preview}"


def _payload_to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        return str(payload)


async def _call_json_tool(
    workbench: McpWorkbench, tool_name: str, arguments: dict[str, Any]
) -> Any:
    payload = _extract_tool_result_payload(await workbench.call_tool(tool_name, arguments))
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _trim_text(value: str, limit: int = 5000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} chars]"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Judge returned empty content")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _heuristic_judge(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if payload.get("error") or payload.get("error_type"):
            return {
                "ok": False,
                "verdict": "failure",
                "reason": "Payload contains explicit error fields.",
            }
    text = _payload_to_text(payload).strip()
    lowered = text.lower()
    if not text:
        return {
            "ok": False,
            "verdict": "failure",
            "reason": "Payload is empty.",
        }
    if "traceback" in lowered or "exception" in lowered or '"error"' in lowered:
        return {
            "ok": False,
            "verdict": "failure",
            "reason": "Payload looks like an execution error.",
        }
    return {
        "ok": True,
        "verdict": "success",
        "reason": "Fallback heuristic accepted a non-empty, non-error payload.",
    }


def _judge_tool_result_sync(
    tool_info: dict[str, Any],
    arguments: dict[str, Any],
    payload: Any,
    judge_model_name: str,
) -> dict[str, Any]:
    payload_preview = _trim_text(_payload_to_text(payload))
    prompt = (
        "Decide whether this tool call should count as a successful, normal tool usage.\n"
        "Count as success if the tool appears to execute correctly and returns a plausible, tool-relevant result, "
        "including legitimate 'no results found' style responses.\n"
        "Count as failure if it shows validation/auth/network/server/schema errors, malformed output, or an obviously irrelevant result.\n"
        "Return JSON only with keys: verdict, confidence, reason.\n\n"
        f"Tool name: {tool_info.get('name')}\n"
        f"Description: {_trim_text(str(tool_info.get('description') or ''), 1000)}\n"
        f"Arguments used: {_trim_text(json.dumps(arguments, ensure_ascii=False), 1200)}\n"
        f"Parameter schema: {_trim_text(json.dumps(tool_info.get('parameter') or {}, ensure_ascii=False), 1800)}\n"
        f"Tool response preview: {payload_preview}\n"
    )
    body = {
        "model": judge_model_name,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator for MCP scientific tool executions. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib_request.Request(
        MODEL_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MODEL_API_KEY}",
        },
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)
    verdict = str(parsed.get("verdict", "failure")).strip().lower()
    return {
        "ok": verdict == "success",
        "verdict": verdict,
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason", ""),
        "raw_judge_response": content,
    }


async def _judge_tool_result(
    tool_info: dict[str, Any],
    arguments: dict[str, Any],
    payload: Any,
    judge_model_name: str,
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            _judge_tool_result_sync,
            tool_info,
            arguments,
            payload,
            judge_model_name,
        )
    except (urllib_error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        fallback = _heuristic_judge(payload)
        fallback["reason"] = f"{fallback['reason']} Judge fallback used because judge call failed: {exc}"
        fallback["verdict"] = fallback.get("verdict", "failure")
        fallback["confidence"] = "fallback"
        fallback["raw_judge_response"] = str(exc)
        return fallback


def _special_case_args(tool_name: str) -> dict[str, Any] | None:
    if tool_name == "list_tools":
        return {"mode": "names"}
    if tool_name == "grep_tools":
        return {"query": "APP protein", "limit": 5}
    if tool_name == "find_tools":
        return {"query": "APP protein", "limit": 5}
    if tool_name == "get_tool_info":
        return {"tool_names": ["UniProt_get_function_by_accession"]}
    if tool_name == "execute_tool":
        return {
            "tool_name": "UniProt_get_function_by_accession",
            "arguments": {"accession": "P05067"},
        }
    return None


def _generate_value(prop_name: str, schema: dict[str, Any]) -> Any:
    if schema.get("enum"):
        return schema["enum"][0]
    for key in ("anyOf", "oneOf"):
        options = schema.get(key) or []
        for option in options:
            if option.get("type") != "null":
                return _generate_value(prop_name, option)
    prop_name_lower = prop_name.lower()
    prop_type = schema.get("type")

    if prop_type == "array":
        item_schema = schema.get("items", {})
        return [_generate_value(prop_name, item_schema)]
    if prop_type == "object":
        properties = schema.get("properties") or {}
        required = schema.get("required") or list(properties.keys())[:1]
        return {
            nested_name: _generate_value(nested_name, properties.get(nested_name, {}))
            for nested_name in required
        }
    if prop_type == "integer":
        return 1
    if prop_type == "number":
        return 1
    if prop_type == "boolean":
        return False

    if "accession" in prop_name_lower:
        return "P05067"
    if "ensembl" in prop_name_lower:
        return "ENSG00000142192"
    if "efo" in prop_name_lower:
        return "EFO_0000384"
    if "chembl" in prop_name_lower:
        return "CHEMBL25"
    if "drug" in prop_name_lower or "medicinalproduct" in prop_name_lower:
        return "aspirin"
    if "gene" in prop_name_lower:
        return "APP"
    if "disease" in prop_name_lower:
        return "Crohn's disease"
    if "uniprot" in prop_name_lower:
        return "P05067"
    if "goid" in prop_name_lower or "go_id" in prop_name_lower:
        return "GO:0006915"
    if "pmid" in prop_name_lower:
        return "31452104"
    if "doi" in prop_name_lower:
        return "10.1038/s41586-020-2649-2"
    if "smiles" in prop_name_lower:
        return "CC(=O)OC1=CC=CC=C1C(=O)O"
    if "inchi" in prop_name_lower:
        return "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
    if "sequence" in prop_name_lower:
        return "MKWVTFISLLFLFSSAYSRGVFRR"
    if "query" in prop_name_lower or "text" in prop_name_lower or "question" in prop_name_lower:
        return "APP protein function"
    if "name" in prop_name_lower:
        return "APP"
    if "id" in prop_name_lower:
        return "1"

    return "test"


def _normalize_test_example(example: Any) -> dict[str, Any] | None:
    if isinstance(example, dict):
        if isinstance(example.get("input"), dict):
            return example["input"]
        if "expected_output_type" in example or "description" in example:
            return None
        return example
    return None


def _build_args(tool_info: dict[str, Any]) -> tuple[dict[str, Any], str]:
    special_args = _special_case_args(tool_info.get("name", ""))
    if special_args is not None:
        return special_args, "special_case"

    examples = tool_info.get("test_examples") or []
    for example in examples:
        normalized = _normalize_test_example(example)
        if normalized:
            return normalized, "test_examples"

    parameter = tool_info.get("parameter") or {}
    properties = parameter.get("properties") or {}
    required = parameter.get("required") or []

    args: dict[str, Any] = {}
    for name in required:
        args[name] = _generate_value(name, properties.get(name, {}))
    return args, "schema_fallback"


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "api key" in text or "credential" in text or "auth" in text or "forbidden" in text:
        return "auth_error"
    if "validation" in text or "required" in text or "missing" in text or "invalid" in text:
        return "input_error"
    if "timeout" in text:
        return "timeout"
    if "connection" in text or "network" in text or "http" in text:
        return "network_error"
    return "execution_error"


async def _load_all_tool_info(workbench: McpWorkbench) -> list[dict[str, Any]]:
    names_payload = await _call_json_tool(workbench, "list_tools", {"mode": "names"})
    tool_names_raw = names_payload.get("tools", []) if isinstance(names_payload, dict) else []
    tool_names: list[str] = []
    for item in tool_names_raw:
        if isinstance(item, dict):
            if item.get("name"):
                tool_names.append(item["name"])
        elif isinstance(item, str):
            tool_names.append(item)

    all_info: list[dict[str, Any]] = []
    for batch in _chunked(tool_names, 50):
        info_payload = await _call_json_tool(
            workbench,
            "get_tool_info",
            {"tool_names": batch},
        )
        batch_tools = info_payload.get("tools", []) if isinstance(info_payload, dict) else []
        for tool in batch_tools:
            if isinstance(tool, dict) and tool.get("name"):
                all_info.append(tool)
    return all_info


async def _invoke_tool_for_validation(
    workbench: McpWorkbench,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    if tool_name in CORE_COMPACT_TOOLS:
        return await workbench.call_tool(tool_name, arguments)
    return await workbench.call_tool(
        "execute_tool",
        {"tool_name": tool_name, "arguments": arguments},
    )


DIRECT_TOOL_CASES = [
    (
        "UniProt protein lookup (P05067 = APP)",
        "UniProt_get_function_by_accession",
        {"accession": "P05067"},
        _default_result_check,
    ),
    (
        "OpenTargets disease-target associations (Crohn's disease)",
        "OpenTargets_get_associated_targets_by_disease_efoId",
        {"efoId": "EFO_0000384"},
        _default_result_check,
    ),
    (
        "FAERS adverse event count (aspirin)",
        "FAERS_count_reactions_by_drug_event",
        {"medicinalproduct": "aspirin"},
        _default_result_check,
    ),
]


async def stage2_direct_tool_calls(workbench: McpWorkbench) -> None:
    """Call tools directly via workbench.call_tool() without any LLM."""
    print("=" * 60)
    print("STAGE 2: Direct Tool Calls (no LLM)")
    print("=" * 60)

    passed = 0
    failed = 0
    for display_name, tool_name, arguments, check_fn in DIRECT_TOOL_CASES:
        print(f"\n  Testing: {display_name}")
        print(f"  Tool   : {tool_name}")
        print(f"  Args   : {arguments}")
        try:
            result = await workbench.call_tool(tool_name, arguments)
            ok, detail = check_fn(result)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {detail}")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\nStage 2 summary: {passed} passed, {failed} failed.")
    if failed:
        print("[WARN] Some direct tool calls failed — check tool names and network.\n")
    else:
        print("[PASS] Stage 2 passed.\n")


# ---------------------------------------------------------------------------
# Stage 3: End-to-end agent run
# ---------------------------------------------------------------------------
E2E_TASK = (
    "Use ToolUniverse tools to answer the following question: "
    "What are the top 3 biological functions of human APP protein (UniProt accession P05067)? "
    "Please cite the tool you used. "
    "If you need to call the compact-mode wrapper tool `execute_tool`, "
    "pass the target tool inputs under the key `arguments`, not `parameters`."
)


async def stage3_agent_e2e(workbench: McpWorkbench) -> None:
    """Full agent run: LLM selects and calls a tool autonomously."""
    print("=" * 60)
    print("STAGE 3: End-to-End Agent Tool Call")
    print("=" * 60)
    print(f"Task: {E2E_TASK}\n")

    model_client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=MODEL_API_KEY,
        base_url=MODEL_BASE_URL,
    )
    try:
        agent = AssistantAgent(
            name="ToolUniverseAgent",
            model_client=model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
            max_tool_iterations=5,
            description="Specialist agent for scientific tooling exposed by the ToolUniverse MCP server.",
            system_message=(
                "You can use ToolUniverse MCP tools for scientific and technical tasks. "
                "Prefer MCP tools when the task needs specialized computation, retrieval, or domain workflows. "
                "In compact mode, if you call `execute_tool`, use the exact argument names "
                "`tool_name` and `arguments`. Never use the field name `parameters` for that tool. "
                "When you believe the user's request has been fully answered, append the token 'TERMINATE' "
                "at the end of your final response."
            ),
        )
        termination = (
            TextMentionTermination("TERMINATE", sources=["ToolUniverseAgent"])
            | MaxMessageTermination(10)
        )
        team = RoundRobinGroupChat([agent], termination_condition=termination)

        print("--- Agent output (streaming) ---")
        await Console(team.run_stream(task=E2E_TASK))
        print("--- End of agent output ---")
        print("\n[PASS] Stage 3 completed.\n")
    except Exception as exc:
        print(f"[ERROR] Stage 3 failed: {exc}")
        traceback.print_exc()
    finally:
        await model_client.close()


async def stage4_validate_all_tools(
    workbench: McpWorkbench,
    limit: int | None,
    output_dir: Path,
    judge_model_name: str,
) -> None:
    """Enumerate all tools in compact mode and judge whether each call worked."""
    print("=" * 60)
    print("STAGE 4: All-Tool Validation With LLM Judge")
    print("=" * 60)
    print(f"Judge model: {judge_model_name}")
    if limit is not None:
        print(f"Tool limit : {limit}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "all_tool_results.jsonl"
    summary_path = output_dir / "all_tool_summary.json"
    results_path.write_text("", encoding="utf-8")

    started_at = time.time()
    tools = await _load_all_tool_info(workbench)
    tools = sorted(tools, key=lambda item: item.get("name", ""))
    if limit is not None:
        tools = tools[:limit]

    total_tools = len(tools)
    print(f"Discovered {total_tools} tools to validate.\n")

    passed = 0
    failed = 0
    skipped = 0
    failed_tool_names: list[str] = []
    skipped_tool_names: list[str] = []
    failure_buckets: dict[str, int] = {}

    for index, tool_info in enumerate(tools, start=1):
        tool_name = tool_info["name"]
        arguments, argument_source = _build_args(tool_info)
        required = (tool_info.get("parameter") or {}).get("required") or []

        if required and not arguments:
            skipped += 1
            skipped_tool_names.append(tool_name)
            record = {
                "tool_name": tool_name,
                "status": "skipped",
                "reason": "no_arguments_generated",
                "argument_source": argument_source,
                "category": tool_info.get("category"),
                "type": tool_info.get("type"),
            }
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{total_tools}] SKIP {tool_name} | no arguments generated")
            continue

        try:
            result = await _invoke_tool_for_validation(workbench, tool_name, arguments)
            payload = _extract_tool_result_payload(result)
            judge = await _judge_tool_result(
                tool_info=tool_info,
                arguments=arguments,
                payload=payload,
                judge_model_name=judge_model_name,
            )
            ok = bool(judge["ok"])
            record = {
                "tool_name": tool_name,
                "status": "passed" if ok else "failed",
                "argument_source": argument_source,
                "arguments": arguments,
                "category": tool_info.get("category"),
                "type": tool_info.get("type"),
                "judge_model": judge_model_name,
                "judge_verdict": judge.get("verdict"),
                "judge_confidence": judge.get("confidence"),
                "judge_reason": judge.get("reason"),
                "payload_preview": _trim_text(_payload_to_text(payload), 2000),
            }
            if ok:
                passed += 1
                print(f"[{index}/{total_tools}] PASS {tool_name} | {judge.get('reason', '')[:120]}")
            else:
                failed += 1
                failed_tool_names.append(tool_name)
                failure_buckets["judge_failed"] = failure_buckets.get("judge_failed", 0) + 1
                record["failure_type"] = "judge_failed"
                print(f"[{index}/{total_tools}] FAIL {tool_name} | {judge.get('reason', '')[:120]}")
        except Exception as exc:
            failed += 1
            failed_tool_names.append(tool_name)
            failure_type = _classify_error(exc)
            failure_buckets[failure_type] = failure_buckets.get(failure_type, 0) + 1
            record = {
                "tool_name": tool_name,
                "status": "failed",
                "failure_type": failure_type,
                "argument_source": argument_source,
                "arguments": arguments,
                "category": tool_info.get("category"),
                "type": tool_info.get("type"),
                "error": str(exc),
            }
            print(f"[{index}/{total_tools}] FAIL {tool_name} | {failure_type}: {str(exc)[:120]}")

        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    attempted = passed + failed
    usable_ratio = (passed / attempted) if attempted else 0.0
    elapsed_seconds = time.time() - started_at
    summary = {
        "tool_count_total": total_tools,
        "attempted": attempted,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "usable_ratio": usable_ratio,
        "judge_model": judge_model_name,
        "failure_buckets": failure_buckets,
        "failed_tool_names": failed_tool_names,
        "skipped_tool_names": skipped_tool_names,
        "elapsed_seconds": elapsed_seconds,
        "results_file": str(results_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nStage 4 summary:")
    print(f"  attempted      : {attempted}")
    print(f"  passed         : {passed}")
    print(f"  failed         : {failed}")
    print(f"  skipped        : {skipped}")
    print(f"  usable ratio   : {usable_ratio:.2%}")
    print(f"  summary file   : {summary_path}")
    print(f"  results file   : {results_path}")
    if failed_tool_names:
        print("\nFailed tools:")
        for name in failed_tool_names:
            print(f"  - {name}")
    if skipped_tool_names:
        print("\nSkipped tools:")
        for name in skipped_tool_names:
            print(f"  - {name}")
    print("\n[PASS] Stage 4 completed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(
    stages: list[int],
    all_tools_limit: int | None,
    all_tools_output_dir: Path,
    judge_model_name: str,
) -> None:
    print(f"\nRunning stages: {stages}")
    print(f"Model      : {MODEL_NAME}  @  {MODEL_BASE_URL}")
    print(f"Judge model: {judge_model_name}")
    print(f"ToolUniverse: {TOOLUNIVERSE_DIR}")

    server_params = _make_server_params()
    print("\nStarting MCP server (may take up to 120s on first run)...")

    async with McpWorkbench(server_params=server_params) as workbench:
        if 1 in stages:
            await stage1_mcp_connectivity(workbench)
        if 2 in stages:
            await stage2_direct_tool_calls(workbench)
        if 3 in stages:
            await stage3_agent_e2e(workbench)
        if 4 in stages:
            await stage4_validate_all_tools(
                workbench=workbench,
                limit=all_tools_limit,
                output_dir=all_tools_output_dir,
                judge_model_name=judge_model_name,
            )

    print("All requested stages complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ToolUniverse MCP tool calls")
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3],
        metavar="N",
        help="Which stages to run (1=MCP connect, 2=direct calls, 3=agent e2e, 4=all-tools validation). Default: 1 2 3.",
    )
    parser.add_argument(
        "--all-tools-limit",
        type=int,
        default=None,
        help="Optional maximum number of tools to validate in Stage 4.",
    )
    parser.add_argument(
        "--all-tools-output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mcp_all_tools_validation",
        help="Directory for Stage 4 result files.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=JUDGE_MODEL_NAME,
        help="Model name used by the Stage 4 judge.",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            stages=args.stages,
            all_tools_limit=args.all_tools_limit,
            all_tools_output_dir=args.all_tools_output_dir,
            judge_model_name=args.judge_model,
        )
    )


if __name__ == "__main__":
    main()
