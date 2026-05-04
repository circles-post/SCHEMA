"""Trajectory filtering: turn the raw messages list from trajectory.jsonl
into a list of ``Step`` objects ready for claim extraction.

Drop policy (from plan step 2):
  * ``tool_call_execution`` entries — we never show tool RESULTS to the
    extractor (user's explicit choice). We keep their IDs implicit: the
    filtered step list simply omits them.
  * Entries with empty text / empty calls.

Keep policy:
  * ``text`` messages whose source is EvalAgent (or any agent).
  * ``tool_call_request`` messages — we render them as a short synthetic
    sentence "I will call <tool> with arguments: <truncated-args>" so the
    extractor can pick up the concept being queried. (Runner already truncates
    tool args to 200 chars, so this is safe.)
  * ``tool_call_summary`` messages — some autogen configs emit a
    natural-language summary AFTER tool calls; keep as plain text.
"""

from __future__ import annotations

from typing import Iterable

from .types import Step


def steps_from_trajectory(
    sample_id: str,
    messages: Iterable[dict],
    *,
    agent_sources: tuple[str, ...] = ("EvalAgent",),
) -> list[Step]:
    """Project a list[dict] of serialized autogen messages to Step objects.

    Any message whose source is not in ``agent_sources`` but whose type is
    ``tool_call_request`` is still included — the agent issued it.
    """
    out: list[Step] = []
    for msg in messages:
        mtype = msg.get("type", "")
        source = msg.get("source", "")
        if mtype == "tool_call_execution":
            continue

        if mtype == "text":
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            if source and agent_sources and source not in agent_sources:
                # Non-agent text (user prompt, system) — skip
                continue
            out.append(
                Step(
                    sample_id=sample_id,
                    step_idx=int(msg.get("idx", len(out))),
                    source=source,
                    msg_type="text",
                    text=text,
                )
            )
        elif mtype == "tool_call_request":
            calls = msg.get("calls") or []
            if not calls:
                continue
            # Render a stable synthetic sentence per tool call, joined by "; ".
            parts: list[str] = []
            for c in calls:
                tool = c.get("tool_name") or "unknown_tool"
                args = c.get("tool_args") or ""
                parts.append(f"[tool_call:{tool}] {args}")
            text = " ; ".join(parts)
            out.append(
                Step(
                    sample_id=sample_id,
                    step_idx=int(msg.get("idx", len(out))),
                    source=source,
                    msg_type="tool_call_request",
                    text=text,
                )
            )
        elif mtype == "tool_call_summary":
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            out.append(
                Step(
                    sample_id=sample_id,
                    step_idx=int(msg.get("idx", len(out))),
                    source=source,
                    msg_type="tool_call_summary",
                    text=text,
                )
            )
        else:
            # Unknown message type — keep if it has text, skip otherwise.
            text = (msg.get("text") or "").strip()
            if text:
                out.append(
                    Step(
                        sample_id=sample_id,
                        step_idx=int(msg.get("idx", len(out))),
                        source=source,
                        msg_type=mtype or "unknown",
                        text=text,
                    )
                )
    return out
