"""Run the Boyue evaluation agent workflow over a dataset.

Pipeline per (model, sample):
    1. Render sample → user prompt via ``prompt_builder.build_prompt``.
    2. Run the single-agent RoundRobinGroupChat (see ``agent_workflow``).
    3. Extract the last ``<answer>...</answer>`` payload from the agent's
       final message.
    4. Hand ``{sample_id: answer}`` to ``evaluation.score_many`` and write
       per-sample results + aggregate stats to ``<output_dir>/<model>/``.

CLI::

    python -m evaluation.runner \\
        --dataset /path/to/samples.jsonl \\
        --models intern-s1-pro,gpt-4o-mini \\
        --output-dir ./eval_runs/smoke \\
        --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

# Make the parent dir importable so evaluation/, pubmed_graph/,
# question_generation/ all resolve as sibling packages regardless of CWD.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


# Hosts that historically were force-bypassed through ``no_proxy`` because
# the local HTTP proxy was a body-mangling Clash. Empty by default now —
# kubernetes-pod environments can't direct-connect external IPs and must go
# through the cluster proxy. Set ``EVAL_DIRECT_CONNECT_HOSTS=ip1,ip2`` to
# re-enable forced direct connection for specific hosts (e.g. when running
# from a laptop with a body-clean local proxy).
_DIRECT_CONNECT_HOSTS = tuple(
    h.strip() for h in os.environ.get("EVAL_DIRECT_CONNECT_HOSTS", "").split(",") if h.strip()
)

# Per-model base URL overrides. Some models hit intermittent
# connection drops on the default Boyue gateway; route them to the more
# stable HTTPS endpoint instead. Override or extend at runtime via
# ``BOYUE_MODEL_OVERRIDES="model1=url1,model2=url2"`` env var.
# Per-model endpoint overrides: each entry can specify a different
# ``base_url`` and/or a different ``api_key_env`` (env var name to read).
# Used for two scenarios:
#   * Models that share Boyue's gateway but hit intermittent connection drops
#     on the IP endpoint — route to the more stable HTTPS endpoint
#     (doubao, llama, grok).
#   * Models hosted on an entirely different vendor — different base URL AND
#     different API key (intern-s1-pro on chat.intern-ai.org.cn).
_DEFAULT_MODEL_OVERRIDES: dict[str, dict[str, str]] = {
    "doubao-seed-2-0-pro-260215": {"base_url": "<boyue-https-base-url>"},
    "llama-4-scout":              {"base_url": "<boyue-https-base-url>"},
    "grok-4-1-fast-reasoning":    {"base_url": "<boyue-https-base-url>"},
    "intern-s1-pro":              {"base_url": "https://chat.intern-ai.org.cn/api/v1",
                                   "api_key_env": "INTERN_API_KEY"},
}


def _resolve_model_endpoint(model: str, default_base_url: str, default_api_key: str) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for ``model`` after applying overrides.

    Precedence: env-var ``BOYUE_MODEL_OVERRIDES`` (comma-separated
    ``model=url`` pairs) overrides only the URL; for full host swaps that
    also need a different api_key_env, edit ``_DEFAULT_MODEL_OVERRIDES``.
    """
    overrides = {k: dict(v) for k, v in _DEFAULT_MODEL_OVERRIDES.items()}
    raw = os.environ.get("BOYUE_MODEL_OVERRIDES", "").strip()
    if raw:
        for pair in raw.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                overrides.setdefault(k.strip(), {})["base_url"] = v.strip()

    cfg = overrides.get(model, {})
    base_url = cfg.get("base_url") or default_base_url
    api_key_env = cfg.get("api_key_env")
    api_key = (os.environ.get(api_key_env, "").strip() if api_key_env else "") or default_api_key
    return base_url, api_key


# Legacy shim — some callers may still expect URL-only resolution.
def _resolve_model_base_url(model: str, default_base_url: str) -> str:
    base_url, _ = _resolve_model_endpoint(model, default_base_url, "")
    return base_url


def _fix_no_proxy_for_host(host: str) -> None:
    """For known direct-connect hosts, ensure ``host`` IS listed in
    ``no_proxy`` / ``NO_PROXY`` so httpx connects directly and does NOT route
    through ``$http_proxy``.

    For domain-name hosts (the HTTPS Boyue gateway), this is a
    no-op so httpx goes through the proxy as configured by the env.

    Some Boyue gateways are reachable directly while others require an
    upstream HTTP proxy; for direct-connect hosts the local proxy can mangle
    tool-call POST bodies and return 404s, so we add them to ``no_proxy``.

    httpx doesn't understand CIDR notation — CIDR entries like ``ip/32``
    are matched as a literal substring, missing the bare host. We strip CIDR
    masks to plain IPs and ensure the bare host is present.
    """
    if host not in _DIRECT_CONNECT_HOSTS:
        # Domain endpoints need the proxy; leave env alone.
        return
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        cleaned = re.sub(r"(\d+\.\d+\.\d+\.\d+)/\d+", r"\1", current) if current else ""
        entries = [e.strip() for e in cleaned.split(",") if e.strip()]
        if host not in entries:
            entries.append(host)
        os.environ[var] = ",".join(entries)

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken

from evaluation import aggregate, score_many
from evaluation.agent_workflow import BoyueModelConfig, build_agent_team
from evaluation.agent_workflow_full import BoyueFullAgentConfig, build_agent_team_full
from evaluation.prompt_builder import build_prompt


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TOOL_ARG_TRUNCATE = 200


def _truncate_tool_args(arguments: Any) -> str:
    """Truncate a tool-call arguments string to prevent retrieved-content
    contamination when downstream consumers (e.g. halu extractor) read the
    trajectory — models sometimes echo long retrieved text inside the next
    call's ``query``.

    Returns at most ``_TOOL_ARG_TRUNCATE`` chars. Non-string inputs are
    coerced via ``json.dumps`` first.
    """
    if arguments is None:
        return ""
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            arguments = str(arguments)
    if len(arguments) > _TOOL_ARG_TRUNCATE:
        return arguments[:_TOOL_ARG_TRUNCATE] + "...[truncated]"
    return arguments


def _serialize_trajectory(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Project autogen message objects to a lean, halu-consumer-friendly JSONL
    record. Keeps ToolCallExecutionEvent entries as stubs so consumers see the
    call_id but not the body (avoids dumping long tool results we'd have to
    filter out anyway). All tool arguments are truncated — see
    ``_truncate_tool_args``.
    """
    out: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        base: dict[str, Any] = {
            "idx": idx,
            "source": getattr(msg, "source", ""),
            "created_at": str(getattr(msg, "created_at", "") or ""),
        }
        if isinstance(msg, TextMessage):
            base.update({"type": "text", "text": str(msg.content or "")})
        elif isinstance(msg, ToolCallRequestEvent):
            calls = []
            for fc in msg.content or []:
                calls.append(
                    {
                        "tool_call_id": getattr(fc, "id", None),
                        "tool_name": getattr(fc, "name", None),
                        "tool_args": _truncate_tool_args(getattr(fc, "arguments", None)),
                    }
                )
            base.update({"type": "tool_call_request", "calls": calls})
        elif isinstance(msg, ToolCallExecutionEvent):
            results = []
            for fer in msg.content or []:
                results.append(
                    {
                        "tool_call_id": getattr(fer, "call_id", None),
                        "tool_name": getattr(fer, "name", None),
                        "is_error": bool(getattr(fer, "is_error", False)),
                        "text_elided": True,
                    }
                )
            base.update({"type": "tool_call_execution", "results": results})
        elif isinstance(msg, ToolCallSummaryMessage):
            base.update({"type": "tool_call_summary", "text": str(msg.content or "")})
        else:
            # Fallback: preserve class name + stringified content so unknown
            # message types don't silently disappear from the trajectory.
            base.update(
                {
                    "type": type(msg).__name__,
                    "text": str(getattr(msg, "content", "") or "")[:2000],
                }
            )
        out.append(base)
    return out


def _load_samples(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _extract_answer(messages: Iterable[Any]) -> tuple[str, str]:
    """Return (parsed_answer, concatenated_agent_texts).

    Scan EVERY ``EvalAgent`` text message in forward order, collect their
    payloads, and return the LAST ``<answer>…</answer>`` hit anywhere in that
    concatenated transcript. Small models (e.g. Qwen-7B) often emit the real
    answer in the pre-tool turn, then autogen's reflection step rewrites it
    to something useless (``"End of Answer\\nTERMINATE"``, ``"rrha"``) — if
    we only look at the last message we'd record that garbage as the answer.

    If no `<answer>` tag exists anywhere, fall back to the trimmed text of
    the last agent message (minus the ``TERMINATE`` sentinel).
    """
    agent_texts: list[str] = []
    for msg in messages:
        if not isinstance(msg, TextMessage):
            continue
        if getattr(msg, "source", "") != "EvalAgent":
            continue
        text = msg.to_text() if hasattr(msg, "to_text") else str(msg.content)
        if text:
            agent_texts.append(text)

    if not agent_texts:
        return "", ""

    joined = "\n\n".join(agent_texts)
    matches = _ANSWER_RE.findall(joined)
    if matches:
        return matches[-1].strip(), joined
    return agent_texts[-1].replace("TERMINATE", "").strip(), joined


async def _run_one_sample(
    team: Any,
    sample: dict[str, Any],
    *,
    per_sample_timeout: float,
    emit_trajectory: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one sample and return (answer_record, trajectory_record|None).

    The trajectory record is ``None`` when ``emit_trajectory=False`` OR when
    the run errored before producing any messages (timeout before first turn).
    """
    prompt = build_prompt(sample)
    await team.reset()
    started = time.monotonic()
    try:
        result: TaskResult = await asyncio.wait_for(
            team.run(task=prompt, cancellation_token=CancellationToken()),
            timeout=per_sample_timeout,
        )
    except asyncio.TimeoutError:
        return {
            "sample_id": sample.get("sample_id", ""),
            "parsed_answer": "",
            "raw_text": "",
            "error": f"timeout:{per_sample_timeout}s",
            "elapsed_sec": round(time.monotonic() - started, 2),
        }, None
    except Exception as exc:  # noqa: BLE001
        return {
            "sample_id": sample.get("sample_id", ""),
            "parsed_answer": "",
            "raw_text": "",
            "error": f"{type(exc).__name__}:{exc}",
            "elapsed_sec": round(time.monotonic() - started, 2),
        }, None

    parsed, raw = _extract_answer(result.messages)
    answer = {
        "sample_id": sample.get("sample_id", ""),
        "parsed_answer": parsed,
        "raw_text": raw,
        "error": "",
        "elapsed_sec": round(time.monotonic() - started, 2),
        "stop_reason": getattr(result, "stop_reason", ""),
    }
    trajectory: dict[str, Any] | None = None
    if emit_trajectory:
        trajectory = {
            "sample_id": sample.get("sample_id", ""),
            "prompt": prompt,
            "stop_reason": getattr(result, "stop_reason", ""),
            "messages": _serialize_trajectory(result.messages),
        }
    return answer, trajectory


# Model families with known agent-mode quirks. Detected by case-insensitive
# substring match on the model id.
#   * NO_TOOLUNIVERSE — backends that can't run with the ToolUniverse MCP
#     workbench attached. Three causes lumped together because the workaround
#     is identical (strip ToolUniverse, keep the 4 retrieval tools):
#       - Gemini, Moonshot Kimi: reject JSON-schema function declarations with
#         ``anyOf`` items missing ``type`` — ToolUniverse's find_tools.* params
#         hit this and 400.
#       - Grok-4-fast-reasoning: hangs >600s on first turn when ToolUniverse
#         is attached (root cause unconfirmed; could be schema-inflated context
#         pushing reasoning budget past timeout). Without TU it answers in
#         ~140s.
#   * THINKING — models that emit empty ``content`` alongside ``tool_calls``.
#     autogen's reflect_on_tool_use=True makes a no-tools follow-up call that
#     returns "" and then errors out, so we disable reflection for those
#     families. The agent still gets a turn after tool execution via
#     group-chat round-robin.
_NO_TOOLUNIVERSE_PATTERNS = ("gemini", "kimi", "moonshot", "grok")
# Match thinking / reasoning model families. Wildcard substrings catch any
# model whose name advertises chain-of-thought; exact ids cover families that
# don't self-label (deepseek-v4, glm-5).
_THINKING_TOOL_PATTERNS = (
    "glm-5", "glm-5.1", "deepseek-v4", "intern-s1",
    "thinking",   # e.g. gemini-3-flash-preview-thinking, claude-*-thinking
    "reasoning",  # e.g. grok-4-1-fast-reasoning
)


def _model_quirks(model: str, *, requested_tooluniverse: bool) -> tuple[bool, bool]:
    """Return (use_tooluniverse, reflect_on_tool_use) for ``model``.

    ``reflect_on_tool_use`` is **forced False for every tool-mode model**.
    autogen's reflection step makes a no-tools follow-up model call after each
    tool round; if the model returns empty content (common for any thinking /
    reasoning model under any prompt), autogen raises ``RuntimeError("Reflect
    on tool use produced no valid text response.")`` and kills the sample.
    With reflection off, the agent's NEXT turn (via group-chat round-robin)
    produces the final text answer naturally.
    """
    lower = model.lower()
    use_tu = requested_tooluniverse and not any(p in lower for p in _NO_TOOLUNIVERSE_PATTERNS)
    reflect = False  # was: not any(p in lower for p in _THINKING_TOOL_PATTERNS)
    return use_tu, reflect


async def _build_team(
    model: str, base_url: str, api_key: str,
    *, per_sample_timeout: float, use_tools: bool, use_tooluniverse: bool,
    max_messages: int, max_tool_iterations: int,
) -> Any:
    if use_tools:
        eff_use_tu, reflect = _model_quirks(model, requested_tooluniverse=use_tooluniverse)
        if eff_use_tu != use_tooluniverse:
            print(f"  [{model}] auto-disabling ToolUniverse (model is in _NO_TOOLUNIVERSE_PATTERNS).", flush=True)
        if not reflect:
            # Now true for every model — see _model_quirks() docstring.
            pass  # silenced; the disable is universal
        full_cfg = BoyueFullAgentConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=max(per_sample_timeout, 300.0),
            max_messages=max_messages,
            max_tool_iterations=max_tool_iterations,
            use_tooluniverse=eff_use_tu,
            reflect_on_tool_use=reflect,
        )
        return await build_agent_team_full(full_cfg)
    simple_cfg = BoyueModelConfig(model=model, base_url=base_url, api_key=api_key)
    return build_agent_team(simple_cfg)


async def _close_team(team: Any) -> None:
    """Close the underlying httpx client so the process exits cleanly."""
    client = getattr(team, "_eval_model_client", None)
    if client is None:
        try:
            client = team._participants[0]._model_client  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            client = None
    if client is not None:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


def _load_resume_state(
    answers_path: Path,
    trajectory_path: Path,
    sample_id_set: set[str],
    *,
    emit_trajectory: bool,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Read existing answers.jsonl + trajectory.jsonl and return the rows worth
    keeping (sample_id present in current dataset, no error, non-empty parsed
    answer). Errored / orphan rows are dropped — the caller will re-attempt
    those samples. Last occurrence wins for duplicates.
    """
    kept_answer: dict[str, dict] = {}
    kept_traj: dict[str, dict] = {}

    if answers_path.is_file():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id", "")
            if not sid or sid not in sample_id_set:
                continue
            if rec.get("error"):
                continue
            if not str(rec.get("parsed_answer") or "").strip():
                continue
            kept_answer[sid] = rec  # last-wins on duplicate

    if emit_trajectory and trajectory_path.is_file():
        for line in trajectory_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id", "")
            if sid in kept_answer:
                kept_traj[sid] = rec

    return kept_answer, kept_traj


async def _run_model(
    model: str,
    base_url: str,
    api_key: str,
    samples: list[dict[str, Any]],
    *,
    output_dir: Path,
    per_sample_timeout: float,
    judge_model_config: dict[str, Any] | None,
    use_tools: bool,
    use_tooluniverse: bool,
    max_messages: int,
    max_tool_iterations: int,
    emit_trajectory: bool,
    sample_concurrency: int = 1,
    force_rerun: bool = False,
) -> dict[str, Any]:
    sample_concurrency = max(1, int(sample_concurrency))

    model_dir = output_dir / _safe_name(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    answers_path = model_dir / "answers.jsonl"
    results_path = model_dir / "scored_results.jsonl"
    summary_path = model_dir / "summary.json"
    trajectory_path = model_dir / "trajectory.jsonl"

    # ----- Idempotent fast path: skip when summary already matches -----
    if not force_rerun and summary_path.is_file():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("n_samples") == len(samples):
                print(f"  [{model}] already DONE (summary.json reports {len(samples)} samples). "
                      f"Pass --force-rerun to redo from scratch.")
                return existing
        except Exception:  # noqa: BLE001
            pass  # fall through to resume

    # ----- Resume: keep good rows from prior runs, drop errored / orphan -----
    sample_id_set = {s.get("sample_id") for s in samples}
    kept_answer_rows: dict[str, dict] = {}
    kept_traj_rows: dict[str, dict] = {}
    if not force_rerun:
        kept_answer_rows, kept_traj_rows = _load_resume_state(
            answers_path, trajectory_path, sample_id_set, emit_trajectory=emit_trajectory
        )

    # Rewrite both files with ONLY the kept rows. This compacts away earlier
    # errored / corrupted lines and makes the file consistent before we append.
    with open(answers_path, "w", encoding="utf-8") as fh:
        for row in kept_answer_rows.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if emit_trajectory:
        with open(trajectory_path, "w", encoding="utf-8") as fh:
            for row in kept_traj_rows.values():
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    answers: dict[str, Any] = {sid: rec["parsed_answer"] for sid, rec in kept_answer_rows.items()}
    pending_samples = [s for s in samples if s.get("sample_id") not in kept_answer_rows]

    if kept_answer_rows:
        print(f"  [{model}] resume: {len(kept_answer_rows)} samples already done, "
              f"{len(pending_samples)} pending out of {len(samples)} total.", flush=True)

    n_workers = min(sample_concurrency, max(1, len(pending_samples) or 1))
    teams: list[Any] = []
    if pending_samples:
        if n_workers == 1:
            teams = [await _build_team(
                model, base_url, api_key,
                per_sample_timeout=per_sample_timeout,
                use_tools=use_tools, use_tooluniverse=use_tooluniverse,
                max_messages=max_messages, max_tool_iterations=max_tool_iterations,
            )]
        else:
            print(f"  [{model}] building {n_workers} concurrent teams...", flush=True)
            teams = await asyncio.gather(*[
                _build_team(
                    model, base_url, api_key,
                    per_sample_timeout=per_sample_timeout,
                    use_tools=use_tools, use_tooluniverse=use_tooluniverse,
                    max_messages=max_messages, max_tool_iterations=max_tool_iterations,
                )
                for _ in range(n_workers)
            ])

    # Open files in append mode so resume preserves earlier kept rows.
    answers_fh = open(answers_path, "a", encoding="utf-8")
    trajectory_fh = open(trajectory_path, "a", encoding="utf-8") if emit_trajectory else None

    if pending_samples:
        team_pool: asyncio.Queue = asyncio.Queue()
        for t in teams:
            team_pool.put_nowait(t)
        write_lock = asyncio.Lock()
        completed = [len(kept_answer_rows)]  # progress counter starts where resume left off

        async def _worker(idx: int, sample: dict[str, Any]) -> None:
            team = await team_pool.get()
            try:
                rec, traj = await _run_one_sample(
                    team,
                    sample,
                    per_sample_timeout=per_sample_timeout,
                    emit_trajectory=emit_trajectory,
                )
            finally:
                team_pool.put_nowait(team)
            async with write_lock:
                answers_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                answers_fh.flush()
                if traj is not None and trajectory_fh is not None:
                    trajectory_fh.write(json.dumps(traj, ensure_ascii=False, default=str) + "\n")
                    trajectory_fh.flush()
                if not rec["error"]:
                    answers[rec["sample_id"]] = rec["parsed_answer"]
                completed[0] += 1
                marker = "✓" if not rec["error"] else "✗"
                print(
                    f"  [{model}] {completed[0]}/{len(samples)} {marker} {rec['sample_id']}  "
                    f"({rec['elapsed_sec']}s) -> {rec['parsed_answer'][:60]!r}"
                    + (f"  [{rec['error']}]" if rec["error"] else "")
                )

        try:
            await asyncio.gather(*[_worker(i, s) for i, s in enumerate(pending_samples)])
        finally:
            answers_fh.close()
            if trajectory_fh is not None:
                trajectory_fh.close()
            for t in teams:
                await _close_team(t)
    else:
        print(f"  [{model}] no pending samples; jumping straight to scoring.", flush=True)
        answers_fh.close()
        if trajectory_fh is not None:
            trajectory_fh.close()

    # score_many can be slow (essay judge calls ~240 LLM round-trips for an
    # 800-sample bench) and may hit transient network errors. Wrap it so a
    # crash here doesn't lose the answers.jsonl + trajectory.jsonl that we
    # already wrote, and so the user gets a visible traceback instead of a
    # silent exit.
    try:
        results = score_many(
            samples,
            answers,
            judge_model_config=judge_model_config,
            skip_missing=False,
        )
        with open(results_path, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n")
        stats = aggregate(results)
        summary = {
            "model": model,
            "base_url": base_url,
            "n_samples": len(samples),
            "n_answered": len(answers),
            "aggregate": stats,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        print(
            f"\n!!! [{model}] score_many crashed AFTER all answers were written. "
            f"answers.jsonl + trajectory.jsonl are intact. "
            f"Re-run the same command — runner will resume + re-score.\n"
            f"!!! Exception: {type(exc).__name__}: {exc}\n",
            flush=True,
        )
        _tb.print_exc()
        # Return a placeholder summary so per-model loops can keep going.
        return {
            "model": model,
            "base_url": base_url,
            "n_samples": len(samples),
            "n_answered": len(answers),
            "aggregate": {"overall": {"acc": 0.0, "weighted_acc": 0.0, "n": 0, "errors": 0,
                                       "scoring_failed": True, "scoring_error": f"{type(exc).__name__}:{exc}"}},
        }


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Boyue models on a QA dataset.")
    ap.add_argument("--dataset", required=True, help="Path to JSONL of samples.")
    ap.add_argument("--models", required=True, help="Comma-separated list of Boyue model names.")
    ap.add_argument("--output-dir", required=True, help="Directory for per-model outputs.")
    ap.add_argument("--limit", type=int, default=0, help="If > 0, evaluate only the first N samples.")
    ap.add_argument(
        "--per-sample-timeout",
        type=float,
        default=None,
        help="Per-sample wall-clock cap. Default 180s for closed-book, 600s with --use-tools.",
    )
    ap.add_argument(
        "--env-file",
        default=str(_HERE / ".env"),
        help="dotenv file with BOYUE_API_KEY / BOYUE_BASE_URL.",
    )
    tools_group = ap.add_mutually_exclusive_group()
    tools_group.add_argument(
        "--use-tools",
        dest="use_tools",
        action="store_true",
        help="Run the full agent workflow with retrieval tools + ToolUniverse MCP (default).",
    )
    tools_group.add_argument(
        "--no-tools",
        dest="use_tools",
        action="store_false",
        help="Closed-book: one LLM call per sample, no tools.",
    )
    ap.set_defaults(use_tools=True)
    ap.add_argument(
        "--no-tooluniverse",
        dest="use_tooluniverse",
        action="store_false",
        help="Still run retrieval tools but skip the heavy ToolUniverse MCP bootstrap.",
    )
    ap.set_defaults(use_tooluniverse=True)
    ap.add_argument("--max-messages", type=int, default=20, help="Max agent turns per sample (tool mode only).")
    ap.add_argument("--max-tool-iterations", type=int, default=5, help="Max tool rounds per turn (tool mode only).")
    ap.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help="Number of samples to evaluate concurrently per model. >1 builds N independent agent teams that share the ToolUniverse MCP server. Default 1 = strict serial.",
    )
    ap.add_argument(
        "--no-emit-trajectory",
        dest="emit_trajectory",
        action="store_false",
        help="Skip the per-sample trajectory.jsonl dump (downstream hallucination pipeline needs it).",
    )
    ap.set_defaults(emit_trajectory=True)
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing answers.jsonl / summary.json and redo every sample. "
             "Default behavior is to RESUME: keep already-good rows from prior runs and "
             "only re-attempt samples missing or with errors.",
    )
    return ap.parse_args()


async def _main_async() -> int:
    args = _parse_args()
    # override=True so re-running after editing .env actually picks up changes.
    load_dotenv(args.env_file, override=True)

    base_url = os.environ.get("BOYUE_BASE_URL", "").strip()
    if not base_url:
        print(f"ERROR: BOYUE_BASE_URL is empty. Set it in {args.env_file}.", file=sys.stderr)
        return 2
    api_key = os.environ.get("BOYUE_API_KEY", "").strip()
    if not api_key:
        print(f"ERROR: BOYUE_API_KEY is empty. Set it in {args.env_file}.", file=sys.stderr)
        return 2

    # Make the Boyue host route through whatever proxy $http_proxy specifies
    # instead of being bypassed by a stale CIDR entry in $no_proxy.
    host_match = re.search(r"//([^:/]+)", base_url)
    if host_match:
        _fix_no_proxy_for_host(host_match.group(1))

    samples = _load_samples(args.dataset)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    per_sample_timeout = args.per_sample_timeout
    if per_sample_timeout is None:
        per_sample_timeout = 600.0 if args.use_tools else 180.0

    mode = (
        "tools+tooluniverse" if args.use_tools and args.use_tooluniverse
        else "tools" if args.use_tools
        else "closed-book"
    )
    print(
        f"[eval] dataset={args.dataset}  samples={len(samples)}  "
        f"base_url={base_url}  mode={mode}  per_sample_timeout={per_sample_timeout}s  "
        f"sample_concurrency={args.sample_concurrency}"
    )

    judge_cfg: dict[str, Any] | None = None
    if os.environ.get("JUDGE_MODEL") and os.environ.get("JUDGE_BASE_URL") and os.environ.get("JUDGE_API_KEY"):
        judge_cfg = {
            "model": os.environ["JUDGE_MODEL"],
            "base_url": os.environ["JUDGE_BASE_URL"],
            "api_key": os.environ["JUDGE_API_KEY"],
        }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_summaries = []
    for model in models:
        print(f"\n=== model: {model} ===")
        # Resolve per-model overrides: different base_url and/or api_key for
        # models hosted on alternate gateways (e.g. doubao routes via the
        # HTTPS endpoint, intern-s1-pro lives on chat.intern-ai.org.cn).
        eff_base_url, eff_api_key = _resolve_model_endpoint(model, base_url, api_key)
        if eff_base_url != base_url:
            print(f"  [{model}] using model-specific base URL: {eff_base_url}")
            host_match = re.search(r"//([^:/]+)", eff_base_url)
            if host_match:
                _fix_no_proxy_for_host(host_match.group(1))  # no-op if it needs proxy
        if eff_api_key != api_key:
            print(f"  [{model}] using model-specific api_key (env override).")
        summary = await _run_model(
            model,
            eff_base_url,
            eff_api_key,
            samples,
            output_dir=output_dir,
            per_sample_timeout=per_sample_timeout,
            judge_model_config=judge_cfg,
            use_tools=args.use_tools,
            use_tooluniverse=args.use_tooluniverse,
            max_messages=args.max_messages,
            max_tool_iterations=args.max_tool_iterations,
            emit_trajectory=args.emit_trajectory,
            sample_concurrency=args.sample_concurrency,
            force_rerun=args.force_rerun,
        )
        all_summaries.append(summary)
        overall = summary["aggregate"]["overall"]
        print(
            f"  -> {model}  acc={overall['acc']:.3f}  weighted_acc={overall['weighted_acc']:.3f}  "
            f"n={overall['n']}  errors={overall['errors']}"
        )

    (output_dir / "combined_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2)
    )
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
