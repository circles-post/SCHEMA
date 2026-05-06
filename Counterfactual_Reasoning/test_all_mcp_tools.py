"""Smoke test all ToolUniverse tools reachable through the MCP discovery layer.

The ToolUniverse MCP server runs in compact mode, so only a small set of core
discovery tools is exposed directly. This script:

1. Connects to the MCP server.
2. Uses the core discovery tools to enumerate all tool names.
3. Fetches tool metadata in batches via `get_tool_info`.
4. Generates arguments from `test_examples` where available, otherwise falls
   back to simple schema-based defaults.
5. Executes each tool and records pass/fail statistics.

This is a best-effort smoke test. A failure does not necessarily mean the tool
is broken; it may require unavailable credentials, nontrivial input values, or
external services.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


MODEL_API_KEY = os.environ.get("AGENTDEBUG_OPENAI_API_KEY", "")
MODEL_BASE_URL = os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "")
TOOLUNIVERSE_DIR = os.environ.get("TOOLUNIVERSE_DIR", "")
UV_PATH = os.environ.get("TOOLUNIVERSE_UV_PATH", "uv")

CORE_COMPACT_TOOLS = {
    "list_tools",
    "grep_tools",
    "get_tool_info",
    "execute_tool",
    "find_tools",
}


def _make_server_params() -> StdioServerParams:
    return StdioServerParams(
        command=UV_PATH,
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
            "VLLM_API_KEY": MODEL_API_KEY,
            "TOOLUNIVERSE_LLM_CONFIG_MODE": "env_override",
            "TOOLUNIVERSE_LLM_DEFAULT_PROVIDER": "VLLM",
            "TOOLUNIVERSE_LLM_MODEL_DEFAULT": "gpt-4o-mini",
            "VLLM_SERVER_URL": MODEL_BASE_URL,
            "AGENTIC_TOOL_FALLBACK_CHAIN": '[{"api_type":"VLLM","model_id":"gpt-4o-mini"}]',
        },
        read_timeout_seconds=120,
    )


def _unwrap_tool_result(result: Any) -> Any:
    payload = getattr(result, "result", result)
    if isinstance(payload, list):
        unwrapped = [getattr(item, "content", item) for item in payload]
        if len(unwrapped) == 1:
            return unwrapped[0]
        return unwrapped
    return payload


def _payload_to_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return str(payload)


async def _call_json_tool(workbench: McpWorkbench, tool_name: str, arguments: dict[str, Any]) -> Any:
    payload = _unwrap_tool_result(await workbench.call_tool(tool_name, arguments))
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _generate_value(prop_name: str, schema: dict[str, Any]) -> Any:
    prop_name_lower = prop_name.lower()
    prop_type = schema.get("type")

    if prop_type == "array":
        item_schema = schema.get("items", {})
        return [_generate_value(prop_name, item_schema)]
    if prop_type == "integer":
        return 1
    if prop_type == "number":
        return 1
    if prop_type == "boolean":
        return False

    if "accession" in prop_name_lower:
        return "P05067"
    if "ensembl" in prop_name_lower:
        return "ENSG00000146648"
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
    if "text" in prop_name_lower or "query" in prop_name_lower or "question" in prop_name_lower:
        return "APP protein function"
    if "name" in prop_name_lower:
        return "APP"
    if "id" in prop_name_lower:
        return "1"

    return "test"


def _build_args(tool_info: dict[str, Any]) -> tuple[dict[str, Any], str]:
    examples = tool_info.get("test_examples") or []
    if examples:
        example = examples[0]
        if isinstance(example, dict):
            return example, "test_examples"

    parameter = tool_info.get("parameter") or {}
    properties = parameter.get("properties") or {}
    required = parameter.get("required") or []

    args: dict[str, Any] = {}
    for name in required:
        schema = properties.get(name, {})
        args[name] = _generate_value(name, schema)
    return args, "schema_fallback"


def _classify_error(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if "api key" in lower or "credential" in lower or "auth" in lower or "forbidden" in lower:
        return "auth_error"
    if "validation" in lower or "required" in lower or "missing" in lower or "invalid" in lower:
        return "input_error"
    if "timeout" in lower:
        return "timeout"
    if "connection" in lower or "network" in lower or "http" in lower:
        return "network_error"
    return "execution_error"


def _flatten_exception_group(exc: BaseException, path: str = "root") -> list[str]:
    lines = [f"{path}: {type(exc).__name__}: {exc}"]
    nested = getattr(exc, "exceptions", None)
    if nested:
        for idx, child in enumerate(nested, start=1):
            lines.extend(_flatten_exception_group(child, path=f"{path}.{idx}"))
    return lines


def _format_exception_details(exc: BaseException) -> str:
    flattened = _flatten_exception_group(exc)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    parts = [
        " | ".join(flattened),
        tb.strip(),
    ]
    return "\n".join(part for part in parts if part)


def _load_existing_records(results_path: Path) -> dict[str, dict[str, Any]]:
    if not results_path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tool_name = record.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                records[tool_name] = record
    return records


def _failure_reason_from_record(record: dict[str, Any]) -> str:
    for key in (
        "error_detail",
        "error",
        "validation_detail",
        "reason",
        "failure_type",
    ):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "Unknown failure reason"


def _build_status_index(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed_tool_names = sorted(
        name for name, record in records.items() if record.get("status") == "passed"
    )
    failed_tools = {
        name: _failure_reason_from_record(record)
        for name, record in records.items()
        if record.get("status") == "failed"
    }
    skipped_tools = {
        name: str(record.get("reason", "skipped"))
        for name, record in records.items()
        if record.get("status") == "skipped"
    }
    return {
        "passed_tool_names": passed_tool_names,
        "failed_tools": failed_tools,
        "skipped_tools": skipped_tools,
    }


def _validate_uniprot_function(payload: Any, args: dict[str, Any]) -> tuple[bool, str]:
    text = _payload_to_text(payload).lower()
    accession = str(args.get("accession", "")).upper()
    if accession == "P04637":
        ok = "tumor suppressor" in text or "cell cycle" in text
        return ok, "expected TP53 function keywords such as 'tumor suppressor' or 'cell cycle'"
    if accession == "P05067":
        ok = "amyloid" in text or "cell surface receptor" in text or "neurite" in text
        return ok, "expected APP function keywords such as 'amyloid', 'cell surface receptor', or 'neurite'"
    ok = bool(text.strip())
    return ok, "fallback non-empty UniProt function output"


def _validate_opentargets_targets(payload: Any, args: dict[str, Any]) -> tuple[bool, str]:
    text = _payload_to_text(payload)
    ok = "ENSG" in text or "score" in text or "target" in text.lower()
    return ok, "expected target-like output containing ENSG ids, scores, or target fields"


def _validate_ensembl_lookup_gene(payload: Any, args: dict[str, Any]) -> tuple[bool, str]:
    text = _payload_to_text(payload)
    gene = str(args.get("symbol") or args.get("gene") or args.get("id") or "").upper()
    ok = "ENSG" in text and (not gene or gene in text.upper() or "display_name" in text)
    return ok, "expected Ensembl gene record containing ENSG identifier and gene/display metadata"


def _validate_mygene_annotation(payload: Any, args: dict[str, Any]) -> tuple[bool, str]:
    text = _payload_to_text(payload).upper()
    query_gene = str(args.get("gene") or args.get("symbol") or args.get("q") or "APP").upper()
    ok = query_gene in text and ("ENTREZ" in text or "TAXID" in text or "SYMBOL" in text)
    return ok, "expected MyGene annotation containing the query gene plus key annotation fields"


def _validate_clinical_trials_search(payload: Any, args: dict[str, Any]) -> tuple[bool, str]:
    text = _payload_to_text(payload)
    ok = "NCT" in text or "clinicaltrials" in text.lower() or "study" in text.lower()
    return ok, "expected clinical trial search output containing NCT ids or study records"


Validator = Callable[[Any, dict[str, Any]], tuple[bool, str]]

VALIDATORS: dict[str, Validator] = {
    "UniProt_get_function_by_accession": _validate_uniprot_function,
    "OpenTargets_get_associated_targets_by_disease_efoId": _validate_opentargets_targets,
    "ensembl_lookup_gene": _validate_ensembl_lookup_gene,
    "MyGene_get_gene_annotation": _validate_mygene_annotation,
    "clinical_trials_search": _validate_clinical_trials_search,
}


async def load_all_tool_info(workbench: McpWorkbench) -> list[dict[str, Any]]:
    names_payload = await _call_json_tool(workbench, "list_tools", {"mode": "names"})
    tool_names = names_payload["tools"]
    all_info: list[dict[str, Any]] = []
    for batch in _chunked(tool_names, 50):
        info_payload = await _call_json_tool(workbench, "get_tool_info", {"tool_names": batch})
        all_info.extend(info_payload.get("tools", []))
    return all_info


async def _invoke_tool(
    workbench: McpWorkbench,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[Any, str]:
    if tool_name in CORE_COMPACT_TOOLS:
        result = await workbench.call_tool(tool_name, arguments)
        return result, "direct"

    result = await workbench.call_tool(
        "execute_tool",
        {"tool_name": tool_name, "arguments": arguments},
    )
    return result, "execute_tool"


async def run_smoke_test(limit: int | None, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "tool_results.jsonl"
    summary_path = output_dir / "summary.json"
    tested_tools_path = output_dir / "tested_tools_index.json"

    start_time = time.time()
    existing_records = _load_existing_records(results_path)
    async with McpWorkbench(server_params=_make_server_params()) as workbench:
        tools = await load_all_tool_info(workbench)
        if limit is not None:
            tools = tools[:limit]

        resumed_skips = 0
        newly_tested = 0
        passed = 0
        failed = 0
        skipped = 0
        validated_tools = 0
        validated_tools_passed = 0
        validated_tools_failed = 0
        failure_buckets: dict[str, int] = {}

        for idx, tool in enumerate(tools, start=1):
            name = tool["name"]

            if name in existing_records:
                resumed_skips += 1
                existing_status = existing_records[name].get("status", "unknown")
                print(
                    f"[{idx}/{len(tools)}] SKIP {name} | already tested previously ({existing_status})"
                )
                continue

            args, arg_source = _build_args(tool)

            if tool.get("parameter", {}).get("required") and not args:
                newly_tested += 1
                skipped += 1
                record = {
                    "tool_name": name,
                    "status": "skipped",
                    "reason": "no_args_generated",
                    "arg_source": arg_source,
                    "category": tool.get("category"),
                    "type": tool.get("type"),
                }
                existing_records[name] = record
                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{idx}/{len(tools)}] SKIP {name} | no args generated")
                continue

            try:
                newly_tested += 1
                result, execution_path = await _invoke_tool(workbench, name, args)
                payload = _unwrap_tool_result(result)
                validator = VALIDATORS.get(name)
                validation_detail = "non-empty payload"
                validation_mode = "presence_only"
                if validator is not None:
                    validated_tools += 1
                    validation_mode = "content_rule"
                    ok, validation_detail = validator(payload, args)
                    if ok:
                        validated_tools_passed += 1
                    else:
                        validated_tools_failed += 1
                else:
                    ok = bool(payload)
                record = {
                    "tool_name": name,
                    "status": "passed" if ok else "failed",
                    "arg_source": arg_source,
                    "args": args,
                    "execution_path": execution_path,
                    "category": tool.get("category"),
                    "type": tool.get("type"),
                    "validation_mode": validation_mode,
                    "validation_detail": validation_detail,
                    "payload_preview": str(payload)[:500],
                }
                if ok:
                    passed += 1
                    print(f"[{idx}/{len(tools)}] PASS {name}")
                else:
                    failed += 1
                    if validator is not None:
                        failure_buckets["content_validation_failed"] = failure_buckets.get("content_validation_failed", 0) + 1
                        record["failure_type"] = "content_validation_failed"
                        print(f"[{idx}/{len(tools)}] FAIL {name} | content validation failed")
                    else:
                        failure_buckets["empty_result"] = failure_buckets.get("empty_result", 0) + 1
                        record["failure_type"] = "empty_result"
                        print(f"[{idx}/{len(tools)}] FAIL {name} | empty result")
            except Exception as exc:
                newly_tested += 1
                failed += 1
                failure_type = _classify_error(exc)
                failure_buckets[failure_type] = failure_buckets.get(failure_type, 0) + 1
                error_detail = _format_exception_details(exc)
                record = {
                    "tool_name": name,
                    "status": "failed",
                    "failure_type": failure_type,
                    "arg_source": arg_source,
                    "args": args,
                    "execution_path": "direct" if name in CORE_COMPACT_TOOLS else "execute_tool",
                    "category": tool.get("category"),
                    "type": tool.get("type"),
                    "error": str(exc),
                    "error_detail": error_detail,
                }
                print(f"[{idx}/{len(tools)}] FAIL {name} | {failure_type}: {str(exc)[:180]}")
                flattened_lines = _flatten_exception_group(exc)
                if len(flattened_lines) > 1:
                    print("  Nested exceptions:")
                    for line in flattened_lines[1:]:
                        print(f"    - {line[:240]}")

            existing_records[name] = record
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    status_index = _build_status_index(existing_records)
    tested_tools_path.write_text(
        json.dumps(
            {
                "total_recorded_tools": len(existing_records),
                "passed_count": len(status_index["passed_tool_names"]),
                "failed_count": len(status_index["failed_tools"]),
                "skipped_count": len(status_index["skipped_tools"]),
                **status_index,
                "results_file": str(results_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "tool_count_discovered": len(tools),
        "already_tested_skipped": resumed_skips,
        "tool_count_tested_this_run": newly_tested,
        "passed_this_run": passed,
        "failed_this_run": failed,
        "skipped_this_run": skipped,
        "passed_total": len(status_index["passed_tool_names"]),
        "failed_total": len(status_index["failed_tools"]),
        "skipped_total": len(status_index["skipped_tools"]),
        "validated_tools": validated_tools,
        "validated_tools_passed": validated_tools_passed,
        "validated_tools_failed": validated_tools_failed,
        "validator_examples": sorted(VALIDATORS.keys()),
        "failure_buckets": failure_buckets,
        "elapsed_seconds": elapsed,
        "results_file": str(results_path),
        "tested_tools_index_file": str(tested_tools_path),
        "passed_tool_names": status_index["passed_tool_names"],
        "failed_tools": status_index["failed_tools"],
        "skipped_tools": status_index["skipped_tools"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test all ToolUniverse tools exposed through the MCP discovery layer.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of tools to test.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mcp_tool_smoke_test",
        help="Directory for detailed result logs.",
    )
    args = parser.parse_args()

    summary = asyncio.run(run_smoke_test(limit=args.limit, output_dir=args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
