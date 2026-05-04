from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from external_agent.schemas import ConversationTurn


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _prepare_analysis_content(text: str) -> str:
    cleaned = text.replace("\r\n", "\n")

    # ---- Strip embedded tool-result blocks ----
    # literature_fetch / sciverse / ToolUniverse results are often inlined
    # as structured text blocks. The extractor LLM confuses them with agent
    # reasoning and produces garbage claims from JSON snippets.
    cleaned = re.sub(
        r"(?:Sciverse workflow result for|Literature results for)[^\n]*\n"
        r"(?:.*\n)*?(?=\n[A-Z]|\Z)",
        "",
        cleaned,
    )
    # Inline JSON tool-result objects: {"status": "success", "query": ...}
    cleaned = re.sub(
        r'\{"status"\s*:\s*"(?:success|error)"[^}]*(?:\{[^}]*\}[^}]*)*\}',
        "[tool result omitted]",
        cleaned,
    )
    # JSON arrays of search results: [{"title": ..., "url": ..., "snippet": ...}, ...]
    cleaned = re.sub(
        r'\[\s*\{"(?:title|url|rank|snippet)"[^\]]*\]',
        "[search results omitted]",
        cleaned,
    )
    # Tool call payloads: [{'id': 'chatcmpl-...', 'arguments': '...', 'name': '...'}]
    # AutoGen / intern-s1 serializes function calls into assistant message content.
    cleaned = re.sub(
        r"\[\s*\{['\"]id['\"]\s*:\s*['\"]chatcmpl-[^\]]*\]",
        "[tool call omitted]",
        cleaned,
    )
    # Tool *result* payloads serialized as a list-of-dict with 'content' first and
    # 'name': '<tool_name>' somewhere in the same dict. Example leak (observed in
    # comp1/w1 runs) that escaped every other regex above:
    #     [{'content': 'Literature results for: "CTPS ..." ... ', 'name':
    #      'literature_search', 'call_id': 'chatcmpl-...', 'is_error': False}]
    # The 'content' field can span thousands of characters of real paper
    # abstracts; without this strip the ClaimExtractor treats each sentence as a
    # scientific_concept claim, the judge then confirms them all as grounded,
    # and the repair pipeline halts with ``no_repairable_concepts``.
    cleaned = re.sub(
        r"\[\s*\{['\"]content['\"]\s*:\s*['\"][\s\S]*?['\"]\s*,\s*"
        r"['\"]name['\"]\s*:\s*['\"][a-zA-Z0-9_]+['\"][^\]]*\]",
        "[tool result omitted]",
        cleaned,
    )
    # Same shape but dict-first form: {'content': ..., 'name': '<tool>'}
    cleaned = re.sub(
        r"\{\s*['\"]content['\"]\s*:\s*['\"][\s\S]*?['\"]\s*,\s*"
        r"['\"]name['\"]\s*:\s*['\"][a-zA-Z0-9_]+['\"][^}]*\}",
        "[tool result omitted]",
        cleaned,
    )
    # Tool enumeration JSON: {"total_tools": N, ..., "tools": [...]}
    cleaned = re.sub(
        r'\{\s*"total_tools"\s*:\s*\d+[^}]*"tools"\s*:\s*\[[^\]]*\]\s*\}',
        "[tool list omitted]",
        cleaned,
    )
    # Long comma-separated API-style tool name lists (>5 snake_case names in quotes)
    cleaned = re.sub(
        r'(?:"\w+(?:_\w+){1,}"\s*,\s*){5,}"\w+(?:_\w+){1,}"',
        "[tool names omitted]",
        cleaned,
    )

    # ---- Strip injected SYSTEM NOTE / answer-tag meta-instructions ----
    # The dataset runner appends a [SYSTEM NOTE] block on the final tool-budget
    # turn telling the agent to commit to an option and wrap it in <answer> tags.
    # The extractor mistakes this meta-instruction for an agent claim and labels
    # it ``mapping_claim`` / ``answer_grounding``, producing junk repairs like
    # "The evidence does not support the previous mapping involving Answer
    # grounding."
    cleaned = re.sub(
        r"\[SYSTEM NOTE\][\s\S]*?(?=\n\n|\Z)",
        "",
        cleaned,
    )
    # Stray parenthetical reminder block from the same instruction template.
    cleaned = re.sub(
        r"\([^)]*?(?:wrapped in answer tags|literal placeholder|optionN)[^)]*\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Loose lines that mention answer-tag formatting rules (no parentheses).
    cleaned = re.sub(
        r"(?im)^[^\n]*\b(?:wrapped in answer tags|wrap [^\n]{0,40}answer tags|"
        r"do not echo this instruction|literal placeholder|optionN|"
        r"emit\s+TERMINATE)[^\n]*$",
        "",
        cleaned,
    )

    cleaned = re.sub(r"(?im)^\s*(?:[-*]\s*)?option\s*\d+\s*:\s.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:[-*]\s*)?option\s*[A-F]\s*:\s.*$", "", cleaned)
    cleaned = re.sub(r"(?is)<answer>.*?</answer>", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*terminate\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    max_chars = max(200, int(os.environ.get("AGDEBUGGER_ANALYSIS_MAX_CONTENT_CHARS", "3000")))
    if len(cleaned) <= max_chars:
        return cleaned

    head = max_chars // 2
    tail = max_chars - head
    return cleaned[:head].rstrip() + "\n...\n" + cleaned[-tail:].lstrip()


def _extract_content(inner: Any) -> str:
    if isinstance(inner, str):
        return inner
    if not isinstance(inner, dict):
        return str(inner)

    message_type = inner.get("type")
    if message_type == "GroupChatStart":
        messages = inner.get("messages", [])
        parts = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg["content"]))
        return _prepare_analysis_content("\n".join(parts).strip())
    if "content" in inner:
        return _prepare_analysis_content(str(inner["content"]))
    if "message" in inner and isinstance(inner["message"], dict):
        nested = inner["message"]
        if "content" in nested:
            return _prepare_analysis_content(str(nested["content"]))
    if "response" in inner and isinstance(inner["response"], dict):
        chat_message = inner["response"].get("chat_message")
        if isinstance(chat_message, dict) and "content" in chat_message:
            return _prepare_analysis_content(str(chat_message["content"]))
    return _prepare_analysis_content(json.dumps(inner, ensure_ascii=False))


def _inner_type(message: Dict[str, Any]) -> str:
    inner = message.get("message")
    if isinstance(inner, dict):
        return str(inner.get("type", ""))
    return ""


def _message_source(message: Dict[str, Any]) -> str:
    inner = message.get("message")
    if isinstance(inner, dict):
        if isinstance(inner.get("source"), str):
            return str(inner["source"])
        nested = inner.get("message")
        if isinstance(nested, dict) and isinstance(nested.get("source"), str):
            return str(nested["source"])
        response = inner.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("source"), str):
                return str(chat_message["source"])
    return ""


def _infer_role(message: Dict[str, Any]) -> str:
    outer_type = str(message.get("type", ""))
    sender = str(message.get("sender") or "")
    if outer_type == "ResponseMessageEnvelope":
        return "assistant"
    if sender and sender.lower() != "user":
        return "assistant"
    # AGDebugger's edit_and_revert re-publish path drops the outer envelope
    # sender (ends up as ""), but the inner chat_message.source is preserved.
    # Without this fallback, post-edit messages get misclassified as user
    # input and filtered out by assistant-only analysis — step>=2 then sees
    # zero turns and the repair loop halts with `no_repairable_concepts`.
    inner_source = _message_source(message)
    if inner_source and inner_source.lower() != "user":
        return "assistant"
    return "user"


def _should_skip_message(
    message: Dict[str, Any],
    *,
    role: str,
    content: str,
    seen_group_start_contents: set[str],
    last_assistant_signature: str | None,
) -> tuple[bool, str | None]:
    inner_type = _inner_type(message)
    normalized_content = _normalize_text(content)

    if inner_type in {"GroupChatRequestPublish", "GroupChatTermination"}:
        return True, last_assistant_signature

    if inner_type == "GroupChatStart":
        if not normalized_content:
            return True, last_assistant_signature
        if normalized_content in seen_group_start_contents:
            return True, last_assistant_signature
        seen_group_start_contents.add(normalized_content)
        return role != "user", last_assistant_signature

    if inner_type in {"None", ""} and not normalized_content:
        return True, last_assistant_signature

    if normalized_content in {'{"type": "None"}', "{'type': 'None'}"}:
        return True, last_assistant_signature

    if not normalized_content:
        return True, last_assistant_signature

    if role == "assistant":
        source = _message_source(message)
        signature = f"{source}::{normalized_content}"
        if signature == last_assistant_signature:
            return True, last_assistant_signature
        return False, signature

    return False, last_assistant_signature


def _prune_analysis_turns(turns: List[ConversationTurn], *, assistant_only: bool) -> List[ConversationTurn]:
    max_assistant_turns = max(1, int(os.environ.get("AGDEBUGGER_ANALYSIS_MAX_ASSISTANT_TURNS", "1")))
    long_turn_chars = max(1, int(os.environ.get("AGDEBUGGER_ANALYSIS_LONG_TURN_CHARS", "4000")))
    short_turn_chars = max(1, int(os.environ.get("AGDEBUGGER_ANALYSIS_SHORT_TURN_CHARS", "2000")))

    assistant_turns = [turn for turn in turns if turn.role == "assistant"]
    if not assistant_turns:
        return turns

    selected_assistant_turns = assistant_turns[-max_assistant_turns:]
    if assistant_only and max_assistant_turns == 1 and len(assistant_turns) >= 2:
        latest = assistant_turns[-1]
        previous = assistant_turns[-2]
        latest_text = latest.content.strip()
        latest_lower = latest_text.lower()
        # Walk backwards past any "empty" / giving-up / terminal turns (TERMINATE,
        # very short tails, explicit "could not fetch" surrenders, bare <answer>
        # commit turns) until we find a substantive reasoning/tool-summary turn.
        # Without this, a trajectory whose final assistant message is "TERMINATE"
        # or "I could not fetch the full text" yields zero extractable claims and
        # the strict-concept-repair loop halts with `no_repairable_concepts`.
        def _is_empty_signal(turn: "ConversationTurn") -> bool:
            text = turn.content.strip()
            lowered = text.lower()
            if len(text) < 40:
                return True
            if "terminate" in lowered and len(text) < 200:
                return True
            if "could not fetch" in lowered[:400] or "could not retrieve" in lowered[:400]:
                return True
            if "<answer>" in lowered and len(text) < 400:
                return True
            return False

        if _is_empty_signal(latest):
            for candidate in reversed(assistant_turns[:-1]):
                if not _is_empty_signal(candidate):
                    selected_assistant_turns = [candidate]
                    break
            else:
                # Every assistant turn is an empty signal — keep default.
                pass
        elif "<answer>" in latest_lower and "<answer>" not in previous.content.lower():
            selected_assistant_turns = [previous]
    if (
        len(selected_assistant_turns) >= 2
        and len(selected_assistant_turns[0].content) > long_turn_chars
        and len(selected_assistant_turns[-1].content) <= short_turn_chars
    ):
        selected_assistant_turns = selected_assistant_turns[1:]

    selected_ids = {id(turn) for turn in selected_assistant_turns}
    if assistant_only:
        return [turn for turn in turns if id(turn) in selected_ids]
    return [turn for turn in turns if turn.role != "assistant" or id(turn) in selected_ids]


def load_turns_from_agdebugger_state(
    state: Dict[str, Any],
    *,
    session_id: int | None = None,
    assistant_only: bool = False,
) -> List[ConversationTurn]:
    current_session = session_id if session_id is not None else state.get("current_session", 0)
    history_map = state.get("message_history", {})
    session = history_map.get(str(current_session), history_map.get(current_session, {}))
    messages = session.get("messages", [])

    turns: List[ConversationTurn] = []
    seen_group_start_contents: set[str] = set()
    last_assistant_signature: str | None = None
    assistant_turn_counter = 0
    for index, message in enumerate(messages):
        role = _infer_role(message)
        if assistant_only and role != "assistant":
            continue
        content = _extract_content(message.get("message"))
        should_skip, last_assistant_signature = _should_skip_message(
            message,
            role=role,
            content=content,
            seen_group_start_contents=seen_group_start_contents,
            last_assistant_signature=last_assistant_signature,
        )
        if should_skip:
            continue
        metadata = dict(message)
        if role == "assistant":
            assistant_turn_counter += 1
            metadata["analysis_assistant_turn"] = assistant_turn_counter
        turns.append(
            ConversationTurn(
                role=role,
                content=content,
                turn_number=index,
                metadata=metadata,
            )
        )
    return _prune_analysis_turns(turns, assistant_only=assistant_only)
