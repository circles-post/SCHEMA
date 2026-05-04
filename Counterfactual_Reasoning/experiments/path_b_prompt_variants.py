"""Path B experiment — compare rewriter prompt variants on stuck cases.

Usage (from repo root, with the agentdebug conda env active):
    SCIVERSE_API_TOKEN=...  # optional, only if --use-real-evidence
    python experiments/path_b_prompt_variants.py \
        --run-dir logs/20260421/run_20260421_185626_w0 \
        --run-dir logs/20260421/run_20260421_185631_w1 \
        --cases 4 \
        --output experiments/path_b_out

The harness does NOT rerun the agent — it only compares what each
variant's rewriter would produce for the same debug-step input. That
means the experiment can run in minutes against the stored logs rather
than requiring a full 100-sample benchmark. For the verdict step (does
the agent actually flip its answer?) the user will pick one or two
winners and rerun a small real benchmark with
``AGDEBUGGER_REWRITER_PROMPT_VARIANT=<name>`` set.

The harness prints a side-by-side table and writes per-case txt files
into --output for offline inspection.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from external_agent.llm import OpenAICompatibleLLM
from external_agent.rewriter import _leak_guard
from external_agent.rewriter_prompts_experimental import VARIANTS
from external_agent.strategies import (
    REWRITER_SYSTEM_PROMPT,
    build_rewriter_user_prompt,
)


# ---------------------------------------------------------------------------
# Log parsing — extract stuck cases and reconstruct rewriter inputs
# ---------------------------------------------------------------------------


@dataclass
class StuckCase:
    run_dir: str
    index: int
    question_text: str
    ground_truth: str
    init_answer: str
    target_turn_number: int
    original_span_content: str
    prefix_context: str
    suffix_context: str
    contributing_claims: list[dict]


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def _concat_messages(messages: list[dict]) -> list[tuple[int, str, str]]:
    """Return a list of (turn_number, role, content) tuples. Missing fields
    are coerced to empty strings. Content is joined if it arrives as a list
    of blocks (tool-call messages)."""
    out: list[tuple[int, str, str]] = []
    for i, m in enumerate(messages or []):
        role = str(m.get("role") or m.get("type") or "")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("content") or block.get("text") or ""))
                else:
                    parts.append(str(block))
            content = "\n".join(p for p in parts if p)
        content = str(content or "")
        out.append((i, role, content))
    return out


def _extract_stuck_cases(run_dir: Path, limit: int) -> list[StuckCase]:
    run_jsonl = run_dir / "run.jsonl"
    if not run_jsonl.exists():
        return []

    # Single-pass load — every signal we need is inside run.jsonl. The
    # per-index analysis_trace events with phase=original_result carry the
    # full claims / judgments / planner_judgment triple, so we don't have
    # to cross-reference analysis_detail.jsonl (which is keyed by
    # conversation_id=0 across ALL questions in a worker run and would
    # require time-order matching).
    questions: dict[int, dict[str, Any]] = {}
    for rec in _load_jsonl(run_jsonl):
        idx = rec.get("index")
        if idx is None:
            continue
        q = questions.setdefault(idx, {"steps": [], "halts": [], "analyses": []})
        ev = rec.get("event")
        if ev == "question_start":
            q["ground_truth"] = rec.get("ground_truth")
            task = rec.get("task") or {}
            if isinstance(task, dict):
                q["question_text"] = task.get("question") or ""
        elif ev == "question_result":
            if "init_answer_norm" not in q:
                q["init_answer_norm"] = rec.get("answer_norm") or rec.get("answer_raw") or ""
            q["messages"] = rec.get("messages") or []
        elif ev == "debug_step":
            q["steps"].append(rec)
        elif ev == "debug_halt":
            q["halts"].append(rec.get("reason"))
        elif ev == "question_final":
            q["final_correct"] = rec.get("correct")
            q["debug_fixed"] = rec.get("debug_fixed")
        elif ev == "analysis_trace":
            summary = rec.get("analysis_summary") or {}
            if summary.get("has_analysis") and rec.get("analysis"):
                q["analyses"].append(rec)

    stuck: list[StuckCase] = []
    for idx, q in sorted(questions.items()):
        if len(stuck) >= limit:
            break
        if q.get("debug_fixed"):
            continue
        if "max_concept_repair_attempts_exhausted" not in q.get("halts", []):
            continue
        if not q.get("steps") or not q.get("analyses"):
            continue
        # Pick the first debug_step and the matching analysis_trace (the
        # first analysis that produced the repair the first step acts on).
        step0 = q["steps"][0]
        action = step0.get("action") or {}
        target_turn = action.get("target_turn")
        selected_claim_id = action.get("claim_id")
        if target_turn is None:
            continue
        messages = q.get("messages") or []
        turns = _concat_messages(messages)

        # Assistant turn at target_turn = the original span (approximation).
        # In the real pipeline the span is a fused sub-region; for this
        # offline experiment the full turn content is the upper bound of
        # what the rewriter sees.
        span = ""
        prefix_parts: list[str] = []
        suffix_parts: list[str] = []
        for tn, role, content in turns:
            if tn < target_turn:
                prefix_parts.append(f"[{role} turn {tn}] {content}")
            elif tn == target_turn:
                span = content
            else:
                suffix_parts.append(f"[{role} turn {tn}] {content}")

        # Find the analysis_trace whose planner_judgment names the same
        # selected_claim_id as this debug_step.
        contributing: list[dict] = []
        for tr in q["analyses"]:
            analysis = tr.get("analysis") or {}
            pj = analysis.get("planner_judgment") or {}
            if (
                selected_claim_id
                and pj.get("selected_claim_id") == selected_claim_id
            ):
                concept = pj.get("repair_concept_name") or pj.get("selected_claim_reason") or ""
                incorrect = pj.get("incorrect_understanding") or ""
                correct = pj.get("correct_understanding") or ""
                # Evidence basis — take the reason from the matching
                # judgment. Judgment entries are claim-aligned with the
                # claims list, both keyed by claim_id.
                evidence: list[str] = []
                for jud in analysis.get("judgments", []):
                    if jud.get("claim_id") == selected_claim_id:
                        r = str(jud.get("reason") or "").strip()
                        if r:
                            evidence.append(r[:300])
                        break
                contributing.append(
                    {
                        "concept_name": concept,
                        "incorrect_understanding": incorrect,
                        "correct_understanding": correct,
                        "evidence_basis": evidence,
                    }
                )
                break
        if not contributing:
            # Fallback — just use the first analysis with any planner_judgment.
            tr = q["analyses"][0]
            analysis = tr.get("analysis") or {}
            pj = analysis.get("planner_judgment") or {}
            concept = pj.get("repair_concept_name") or pj.get("selected_claim_reason") or ""
            incorrect = pj.get("incorrect_understanding") or ""
            correct = pj.get("correct_understanding") or ""
            evidence = []
            for jud in analysis.get("judgments", []):
                r = str(jud.get("reason") or "").strip()
                if r:
                    evidence.append(r[:300])
                    break
            if concept or incorrect or correct:
                contributing.append(
                    {
                        "concept_name": concept,
                        "incorrect_understanding": incorrect,
                        "correct_understanding": correct,
                        "evidence_basis": evidence,
                    }
                )
        if not contributing:
            continue

        prefix_ctx = "\n".join(prefix_parts)[-1600:]
        suffix_ctx = "\n".join(suffix_parts)[:800]
        question_text = (q.get("question_text") or "")[:2000]

        stuck.append(
            StuckCase(
                run_dir=run_dir.name,
                index=idx,
                question_text=question_text,
                ground_truth=str(q.get("ground_truth") or ""),
                init_answer=str(q.get("init_answer_norm") or ""),
                target_turn_number=int(target_turn),
                original_span_content=span[:3000],
                prefix_context=prefix_ctx,
                suffix_context=suffix_ctx,
                contributing_claims=contributing,
            )
        )
    return stuck


# ---------------------------------------------------------------------------
# LLM runner
# ---------------------------------------------------------------------------


async def _run_variant(
    llm: OpenAICompatibleLLM,
    *,
    variant_name: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: float,
) -> dict:
    try:
        payload = await asyncio.wait_for(
            llm.complete_json(system_prompt, user_prompt), timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        return {"variant": variant_name, "outcome": "timeout", "rewritten_text": ""}
    except Exception as exc:  # noqa: BLE001
        return {
            "variant": variant_name,
            "outcome": "llm_error",
            "error": f"{type(exc).__name__}: {exc}",
            "rewritten_text": "",
        }
    if not isinstance(payload, dict):
        return {
            "variant": variant_name,
            "outcome": "json_parse_error",
            "rewritten_text": "",
            "raw": str(payload)[:400],
        }
    rewritten = str(payload.get("rewritten_text", "")).strip()
    if not rewritten:
        return {"variant": variant_name, "outcome": "empty", "rewritten_text": ""}
    leak = _leak_guard(rewritten)
    if leak:
        return {
            "variant": variant_name,
            "outcome": leak,
            "rewritten_text": "",
            "rejected_text": rewritten,
        }
    return {"variant": variant_name, "outcome": "ok", "rewritten_text": rewritten}


def _build_llm() -> OpenAICompatibleLLM:
    model = os.environ.get("MODEL_PLANNER") or os.environ.get("AGENTDEBUG_MODEL_NAME") or "intern-s1"
    api_key = os.environ.get("AGENTDEBUG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = (
        os.environ.get("AGENTDEBUG_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://chat.intern-ai.org.cn/api/v1/"
    )
    return OpenAICompatibleLLM(model=model, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _shorten(text: str, limit: int = 240) -> str:
    s = " ".join(text.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


async def _main_async(args: argparse.Namespace) -> int:
    run_dirs = [Path(p) for p in args.run_dir]
    cases: list[StuckCase] = []
    for rd in run_dirs:
        cases.extend(_extract_stuck_cases(rd, args.cases - len(cases)))
        if len(cases) >= args.cases:
            break
    if not cases:
        print("[harness] No stuck cases found in the given run dirs.", file=sys.stderr)
        return 1

    print(f"[harness] Loaded {len(cases)} stuck cases.")
    for c in cases:
        print(
            f"  idx={c.index:3d} run={c.run_dir}  init={c.init_answer} gt={c.ground_truth} "
            f"turn={c.target_turn_number}  claims={len(c.contributing_claims)}"
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = _build_llm()
    variants = {"V0_baseline": REWRITER_SYSTEM_PROMPT, **VARIANTS}
    summary_rows: list[dict] = []

    for case in cases:
        user_prompt = build_rewriter_user_prompt(
            question_text=case.question_text,
            target_turn_number=case.target_turn_number,
            original_span_content=case.original_span_content,
            prefix_context=case.prefix_context,
            suffix_context=case.suffix_context,
            contributing_claims=case.contributing_claims,
        )
        case_slug = f"{case.run_dir}_idx{case.index}"
        case_summary = {
            "case": case_slug,
            "init": case.init_answer,
            "gt": case.ground_truth,
            "variants": {},
        }
        print(f"\n=== Case {case_slug} (init={case.init_answer}, gt={case.ground_truth}) ===")
        for variant_name, sys_prompt in variants.items():
            result = await _run_variant(
                llm,
                variant_name=variant_name,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                timeout_sec=args.timeout_sec,
            )
            case_summary["variants"][variant_name] = result
            outcome = result.get("outcome")
            rewrite = result.get("rewritten_text", "")
            rejected = result.get("rejected_text", "")
            preview = rewrite if rewrite else (rejected and f"[REJECTED] {rejected}") or ""
            print(f"  [{variant_name:20s}] outcome={outcome:<18s} | {_shorten(preview)}")
            out_path = out_dir / f"{case_slug}__{variant_name}.txt"
            with out_path.open("w") as f:
                f.write(f"=== {case_slug} / {variant_name} / outcome={outcome} ===\n\n")
                f.write(f"[init_answer] {case.init_answer}\n")
                f.write(f"[ground_truth] {case.ground_truth}\n")
                f.write(f"[target_turn] {case.target_turn_number}\n\n")
                f.write("[rewritten_text]\n")
                f.write(rewrite + "\n" if rewrite else "(none)\n")
                if rejected:
                    f.write(f"\n[rejected_text]\n{rejected}\n")
                if result.get("error"):
                    f.write(f"\n[error]\n{result['error']}\n")
        summary_rows.append(case_summary)

    # Aggregate — per-variant outcome counts.
    print("\n=== Aggregate (outcome counts per variant) ===")
    from collections import Counter

    variant_outcomes: dict[str, Counter] = {v: Counter() for v in variants}
    for cs in summary_rows:
        for vn, res in cs["variants"].items():
            variant_outcomes[vn][res.get("outcome") or "?"] += 1
    col_w = 22
    oc_keys = sorted({k for counter in variant_outcomes.values() for k in counter.keys()})
    header = f"{'variant':<{col_w}} " + " ".join(f"{k:>14s}" for k in oc_keys)
    print(header)
    print("-" * len(header))
    for vn, counter in variant_outcomes.items():
        row = f"{vn:<{col_w}} " + " ".join(f"{counter.get(k, 0):>14d}" for k in oc_keys)
        print(row)

    json_path = out_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n[harness] Wrote per-variant outputs to {out_dir}")
    print(f"[harness] Structured summary: {json_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Worker run dir (containing run.jsonl and analysis_detail.jsonl). Repeat for multiple.",
    )
    parser.add_argument("--cases", type=int, default=4, help="Number of stuck cases to test.")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/path_b_out",
        help="Directory to write per-variant output files and summary.json.",
    )
    parser.add_argument(
        "--timeout-sec", type=float, default=60.0, help="Per-variant LLM timeout."
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
