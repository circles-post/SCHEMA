#!/usr/bin/env python3
"""External controller for AGDebugger.

This script lets an external agent replace manual UI operations by calling
AGDebugger backend APIs directly:
  - send/publish new messages
  - edit queued messages
  - edit history + revert
  - insert a new message after a history timestamp (via revert + resend)

Two planner modes are supported:
  1) rule: deterministic bootstrap/step behavior (no LLM needed)
  2) llm: OpenAI-compatible planner that emits JSON actions
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib import error, request

from external_agent.integration import analyze_session_state
from external_agent.rate_limiter import (
    estimate_request_tokens,
    get_shared_limiter,
    response_total_tokens,
    with_retry_sync,
)
from model_routing import (
    build_openai_extra_body,
    resolve_base_url_for_model,
    resolve_value_for_model,
)


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from raw text."""
    text = text.strip()
    if not text:
        raise ValueError("Planner returned empty text.")

    # Handle fenced code blocks first.
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)

    # Fallback: first balanced {...} object.
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in planner output: {text[:200]}")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError(f"Unbalanced JSON object in planner output: {text[:200]}")


def _configure_external_llm_proxy(base_url: str) -> None:
    proxy_url = os.environ.get("http_proxy", "")
    if proxy_url:
        os.environ.setdefault("http_proxy", proxy_url)
        os.environ.setdefault("https_proxy", proxy_url)

    no_proxy = os.environ.get("no_proxy", "")
    fixed_no_proxy = re.sub(r"(\d+\.\d+\.\d+\.\d+)/\d+", r"\1", no_proxy)
    llm_host = urlparse(base_url).hostname
    if llm_host:
        fixed_no_proxy = ",".join(
            entry for entry in fixed_no_proxy.split(",") if entry.strip() != llm_host
        )
    os.environ["no_proxy"] = fixed_no_proxy
    os.environ["NO_PROXY"] = fixed_no_proxy


def _strict_concept_repair_only_enabled() -> bool:
    """Mirror of run_dataset_autodebug.py's strict-mode env flag.

    When strict mode is on we must NOT auto-inject <answer>option N</answer>
    or TERMINATE into messages we are about to push back through the
    backend, or the rerun will be locked to the very answer the repair was
    supposed to invalidate.
    """
    value = os.environ.get("AGDEBUGGER_STRICT_CONCEPT_REPAIR_ONLY", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _post_edit_critic_enabled() -> bool:
    value = os.environ.get("AGDEBUGGER_POST_EDIT_CRITIC", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _deliver_post_edit_critic(client: "AGDebuggerClient", critic_text: str) -> None:
    """Deliver the critic turn after an ``edit_and_revert`` has been applied.

    Default delivery is ``manager_start``: wait for the replaced-turn rollout
    to settle (the agent produces its — still-likely-wrong — answer and the
    chat TERMINATEs), then send a fresh ``GroupChatStart`` carrying the
    critic as a new user message to the manager topic. This piggy-backs on
    the exact pathway ``run_question`` uses to start each question, so the
    manager's termination state is reset cleanly and AutoGen re-runs round-
    robin with the critic appended to the conversation.

    Alternate delivery is ``broadcast``: publish a ``GroupChatMessage`` to
    ``group_topic_<uuid>``. The agent buffers it but may not emit a response
    if the manager has already latched termination — kept around as a
    diagnostic/control option, not as the default.
    """
    delivery = (
        os.environ.get("AGDEBUGGER_POST_EDIT_CRITIC_DELIVERY", "manager_start").strip().lower()
    )
    critic_source = os.environ.get("AGDEBUGGER_POST_EDIT_CRITIC_SOURCE", "user")
    wait_timeout = float(os.environ.get("AGDEBUGGER_POST_EDIT_CRITIC_WAIT_TIMEOUT", "120"))

    topics = client.get_topics()
    if not topics:
        print("[warn] post-edit critic: backend returned no topics, skipping")
        return

    if delivery == "broadcast":
        # Fire-and-hope: let the replaced-turn rollout race with the critic.
        # No explicit wait — the caller already called start_loop before us.
        broadcast_topic = _guess_group_chat_broadcast_topic(topics)
        if not broadcast_topic:
            print("[warn] post-edit critic: no broadcast topic found, skipping")
            return
        body = _make_group_chat_message_from_raw_text(critic_text, source=critic_source)
        client.publish(topic=broadcast_topic, body=body)
        print(f"[post-edit-critic] broadcast to topic={broadcast_topic}")
        return

    # delivery == "manager_start" (default)
    manager_topic = _guess_manager_topic(topics)
    if not manager_topic:
        print("[warn] post-edit critic: no manager topic found, skipping")
        return
    # Wait for the current rollout (triggered by edit_and_revert) to settle
    # — we want the agent to land on its post-rewrite answer BEFORE we push
    # back. Sending GroupChatStart mid-rollout can race with in-flight turns.
    try:
        client.wait_until_idle(timeout_sec=wait_timeout)
    except TimeoutError:
        print("[warn] post-edit critic: wait_until_idle timed out, sending GroupChatStart anyway")
    start_body = {
        "type": "GroupChatStart",
        "messages": [
            {"type": "TextMessage", "source": critic_source, "content": critic_text}
        ],
    }
    # The loop may have stopped when the rollout hit TextMentionTermination; make
    # sure it's running so the backend can process the new GroupChatStart.
    try:
        client.start_loop()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] post-edit critic: start_loop before send failed: {exc}")
    client.send(recipient=manager_topic, body=start_body)
    print(f"[post-edit-critic] GroupChatStart sent to manager={manager_topic}")


def _compose_post_edit_critic_message(critic: Any) -> Optional[str]:
    """Build the critic follow-up shown to the agent after a rewrite replaces
    a faulty turn.

    Returns ``None`` when the payload is too thin to write something the agent
    can act on — an empty or vague critic would just burn an extra turn.
    """
    if not isinstance(critic, dict):
        return None
    correct = str(critic.get("correct_understanding") or "").strip()
    if not correct:
        return None
    incorrect = str(critic.get("incorrect_understanding") or "").strip()
    concept = str(critic.get("concept_name") or "").strip()
    header_tail = f" about '{concept}'" if concept else ""
    lines = [f"[Critic] Your previous reasoning{header_tail} has been challenged by external evidence."]
    if incorrect:
        lines.append(f"You asserted: {incorrect}")
    lines.append(f"The evidence actually supports: {correct}")
    lines.append(
        "Before giving your final answer, explicitly re-evaluate each listed mechanism "
        "against this corrected evidence and cite which substantive description (NOT a "
        "label such as 'option N') it now matches. Do not restate the previous conclusion."
    )
    return "\n\n".join(lines)


def _normalize_answer_text(content: str) -> str:
    text = content.strip()
    if not text:
        return text

    # In strict concept-repair mode the controller must never fabricate an
    # answer label or a TERMINATE marker — that is the agent's job, and any
    # synthetic injection here would defeat the entire repair pipeline.
    if _strict_concept_repair_only_enabled():
        return text

    if "<answer>" not in text.lower():
        matches = re.findall(r"\boption\s*([0-9]+)\b", text, flags=re.IGNORECASE)
        if matches:
            text = f"{text}\n\n<answer>option{matches[-1]}</answer>"

    if "terminate" not in text.lower():
        text = f"{text}\n\nTERMINATE"

    return text


def _make_group_chat_message_from_text(content: str, *, source: str = "ToolUniverseAgent") -> Dict[str, Any]:
    return {
        "type": "GroupChatMessage",
        "message": {
            "type": "TextMessage",
            "source": source,
            "content": _normalize_answer_text(content),
        },
    }


def _make_group_chat_message_from_raw_text(content: str, *, source: str = "ToolUniverseAgent") -> Dict[str, Any]:
    return {
        "type": "GroupChatMessage",
        "message": {
            "type": "TextMessage",
            "source": source,
            "content": content,
        },
    }


def _looks_like_valid_message_type(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is not None


def _coerce_message_body(body: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(body)
    preserve_content = bool(payload.pop("_preserve_content", False))

    payload_type = payload.get("type")
    if payload_type is not None and not _looks_like_valid_message_type(payload_type):
        return _make_group_chat_message_from_text(str(payload_type))

    if "type" not in payload:
        if "messages" in payload:
            payload["type"] = "GroupChatStart"
        elif "message" in payload:
            payload["type"] = "GroupChatMessage"
        elif "response" in payload or "agent_response" in payload:
            payload["type"] = "GroupChatAgentResponse"
        elif "source" in payload and "content" in payload:
            payload = {"type": "GroupChatMessage", "message": payload}

    if payload.get("type") == "GroupChatMessage" and isinstance(payload.get("message"), dict):
        inner = dict(payload["message"])
        content = inner.get("content")
        if (
            not preserve_content
            and isinstance(content, str)
            and inner.get("source") == "ToolUniverseAgent"
            and inner.get("type") == "TextMessage"
        ):
            inner["content"] = _normalize_answer_text(content)
        payload["message"] = inner

    return payload


def _canonicalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(action)
    name = normalized.get("action")
    if name in {"send", "publish", "edit_queue", "edit_and_revert", "insert_after"}:
        body = normalized.get("body")
        if isinstance(body, str):
            normalized["body"] = _make_group_chat_message_from_text(body)
        elif isinstance(body, dict):
            normalized["body"] = _coerce_message_body(body)
    return normalized


def _iter_history_messages(state: Dict[str, Any]):
    history = state.get("message_history", {})
    for session in history.values():
        if not isinstance(session, dict):
            continue
        for message in session.get("messages", []):
            if isinstance(message, dict):
                yield message


def _session_messages(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    current_session = state.get("current_session")
    history = state.get("message_history", {})
    if str(current_session) in history:
        session = history[str(current_session)]
        if isinstance(session, dict):
            return session.get("messages", [])
    if current_session in history:
        session = history[current_session]
        if isinstance(session, dict):
            return session.get("messages", [])
    return []


def _valid_history_timestamps(state: Dict[str, Any]) -> set[int]:
    timestamps: set[int] = set()
    for message in _iter_history_messages(state):
        ts = message.get("timestamp")
        if isinstance(ts, int):
            timestamps.add(ts)
    return timestamps


def _message_source(entry: Dict[str, Any]) -> str:
    if isinstance(entry.get("source"), str):
        return str(entry["source"])
    nested = entry.get("message")
    if isinstance(nested, dict) and isinstance(nested.get("source"), str):
        return str(nested["source"])
    if isinstance(nested, dict):
        inner_message = nested.get("message")
        if isinstance(inner_message, dict) and isinstance(inner_message.get("source"), str):
            return str(inner_message["source"])
        response = nested.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("source"), str):
                return str(chat_message["source"])
    response = entry.get("response")
    if isinstance(response, dict):
        chat_message = response.get("chat_message")
        if isinstance(chat_message, dict) and isinstance(chat_message.get("source"), str):
            return str(chat_message["source"])
    return ""


def _message_created_at(entry: Dict[str, Any]) -> Optional[str]:
    created_at = entry.get("created_at")
    if isinstance(created_at, str):
        return created_at
    nested = entry.get("message")
    if isinstance(nested, dict) and isinstance(nested.get("created_at"), str):
        return str(nested["created_at"])
    return None


def _message_id(entry: Dict[str, Any]) -> Optional[str]:
    value = entry.get("id")
    if isinstance(value, str):
        return value
    nested = entry.get("message")
    if isinstance(nested, dict):
        if isinstance(nested.get("id"), str):
            return str(nested["id"])
        inner_message = nested.get("message")
        if isinstance(inner_message, dict) and isinstance(inner_message.get("id"), str):
            return str(inner_message["id"])
        response = nested.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("id"), str):
                return str(chat_message["id"])
    return None


def _message_content(entry: Dict[str, Any]) -> str:
    if isinstance(entry.get("content"), str):
        return str(entry["content"])
    nested = entry.get("message")
    if isinstance(nested, dict) and isinstance(nested.get("content"), str):
        return str(nested["content"])
    if isinstance(nested, dict):
        inner_message = nested.get("message")
        if isinstance(inner_message, dict) and isinstance(inner_message.get("content"), str):
            return str(inner_message["content"])
        response = nested.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("content"), str):
                return str(chat_message["content"])
    response = entry.get("response")
    if isinstance(response, dict):
        chat_message = response.get("chat_message")
        if isinstance(chat_message, dict) and isinstance(chat_message.get("content"), str):
            return str(chat_message["content"])
    return ""


def _message_type(entry: Dict[str, Any]) -> str:
    message = entry.get("message")
    if isinstance(message, dict):
        inner = message.get("message")
        if isinstance(inner, dict) and isinstance(inner.get("type"), str):
            return str(inner["type"])
        response = message.get("response")
        if isinstance(response, dict):
            chat_message = response.get("chat_message")
            if isinstance(chat_message, dict) and isinstance(chat_message.get("type"), str):
                return str(chat_message["type"])
        if isinstance(message.get("type"), str):
            return str(message["type"])
    if isinstance(entry.get("type"), str):
        return str(entry["type"])
    return ""


def _compact_text_preview(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _redact_planner_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    text = re.sub(r"<answer>\s*.*?\s*</answer>", "[answer redacted]", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bTERMINATE\b", "[terminate redacted]", text, flags=re.IGNORECASE)
    text = re.sub(r"\boption\s*[a-z0-9]+\b", "[option redacted]", text, flags=re.IGNORECASE)
    return text


def _sanitize_for_planner(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_planner_text(value)
    if isinstance(value, list):
        return [_sanitize_for_planner(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_for_planner(val) for key, val in value.items()}
    return value


def _concept_only_planner_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    text = _redact_planner_text(text)
    text = re.sub(r"(?i)\boption analysis\b", "Concept analysis", text)
    text = re.sub(r"(?i)\blooking back at our candidate explanations\b", "Looking back at the candidate explanations", text)
    text = re.sub(r"(?i)\blooking at the options\b", "Looking at the candidate explanations", text)
    text = re.sub(r"(?i)\bthe answer is\s+\[option redacted\]\b", "the trajectory selected a candidate explanation", text)
    text = re.sub(
        r"(?i)\bthe\s+(?:most\s+scientifically\s+plausible|best|correct)\s+(?:explanation|choice|answer)\s+is\s+\[option redacted\]\s*:?",
        "the trajectory selected the following candidate explanation: ",
        text,
    )
    text = re.sub(r"(?i)\bthis aligns with\s+\[option redacted\]\s*:?", "This aligns with the following concept claim:", text)
    text = re.sub(r"(?i)\bwhile other\s+\[option redacted\]\b", "while other candidate explanations", text)
    text = re.sub(r"(?i)\bthe trajectory selected\s+\[option redacted\]\s+too early\b", "the trajectory selected a candidate explanation too early", text)
    text = re.sub(r"(?i)(^|\n)(\s*[-*]?\s*)\*\*\[option redacted\]\*\*\s*:", r"\1\2Candidate explanation:", text)
    text = re.sub(r"(?i)(^|\n)(\s*[-*]?\s*)\[option redacted\]\s*:", r"\1\2Candidate explanation:", text)
    text = re.sub(r"(?i)\boptions\b", "candidate explanations", text)
    text = text.replace("[option redacted]", "candidate explanation")
    text = re.sub(r"(?i)\bother candidate explanation\b", "other candidate explanations", text)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"[answer redacted]", "[terminate redacted]"}:
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


_CONCEPT_ONLY_ANALYSIS_KEYS = {
    "text",
    "source_ref",
    "reference_name",
    "concept_name",
    "concept_true_understanding",
    "incorrect_understanding",
    "correct_understanding",
    "original_context",
    "reason",
    "content_preview",
    "replacement_guidance",
    "repair_constraint",
}


def _sanitize_analysis_for_planner(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str):
        if key in _CONCEPT_ONLY_ANALYSIS_KEYS:
            return _concept_only_planner_text(value)
        return _redact_planner_text(value)
    if isinstance(value, list):
        return [_sanitize_analysis_for_planner(item, key=key) for item in value]
    if isinstance(value, dict):
        return {
            item_key: _sanitize_analysis_for_planner(item_val, key=item_key)
            for item_key, item_val in value.items()
        }
    return value


def _contains_answer_or_terminate(text: str) -> bool:
    lowered = text.lower()
    return "<answer>" in lowered or "terminate" in lowered


def _is_editable_assistant_entry(entry: Dict[str, Any]) -> bool:
    source = _message_source(entry)
    if not source or source == "user":
        return False
    if "GroupChatManager" in source or source.endswith("Termination"):
        return False

    message_type = _message_type(entry)
    if message_type in {"ThoughtEvent", "TextMessage", "ToolCallRequestEvent", "ToolCallExecutionEvent"}:
        return True
    return _entry_is_group_chat_agent_response(entry)


def _entry_is_final_answer_like(entry: Dict[str, Any]) -> bool:
    content = _message_content(entry)
    if not isinstance(content, str) or not content.strip():
        return False
    return _contains_answer_or_terminate(content)


def _entry_is_group_chat_agent_response(entry: Dict[str, Any]) -> bool:
    message = entry.get("message")
    return isinstance(message, dict) and message.get("type") == "GroupChatAgentResponse"


def _response_envelope_mentions_entry(response_entry: Dict[str, Any], target_entry: Dict[str, Any]) -> bool:
    message = response_entry.get("message")
    if not isinstance(message, dict):
        return False
    response = message.get("response")
    if not isinstance(response, dict):
        return False

    target_id = _message_id(target_entry)
    target_created_at = _message_created_at(target_entry)
    target_content = _message_content(target_entry)

    def _matches(message_dict: Any) -> bool:
        if not isinstance(message_dict, dict):
            return False
        if target_id and message_dict.get("id") == target_id:
            return True
        if target_created_at and message_dict.get("created_at") == target_created_at:
            return True
        if target_content and message_dict.get("content") == target_content:
            return True
        return False

    chat_message = response.get("chat_message")
    if _matches(chat_message):
        return True

    inner_messages = response.get("inner_messages")
    if isinstance(inner_messages, list):
        for item in inner_messages:
            if _matches(item):
                return True
    return False


def _find_enclosing_agent_response_entry(messages: list[Dict[str, Any]], target_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target_ts = target_entry.get("timestamp")
    for entry in messages:
        if not isinstance(entry, dict):
            continue
        ts = entry.get("timestamp")
        if isinstance(target_ts, int) and isinstance(ts, int) and ts < target_ts:
            continue
        if _entry_is_group_chat_agent_response(entry) and _response_envelope_mentions_entry(entry, target_entry):
            return entry
    return None


def _history_entry_by_timestamp(messages: list[Dict[str, Any]], timestamp: int) -> Optional[Dict[str, Any]]:
    for message in messages:
        if isinstance(message, dict) and message.get("timestamp") == timestamp:
            return message
    return None


def _is_reasoning_like_entry(entry: Dict[str, Any]) -> bool:
    if _message_source(entry) == "user":
        return False
    if _message_type(entry) not in {"ThoughtEvent", "TextMessage"}:
        return False
    content = _message_content(entry).strip()
    if not content:
        return False
    return not _contains_answer_or_terminate(content)


def _prefer_reasoning_timestamp_for_repair(messages: list[Dict[str, Any]], timestamp: int) -> int:
    target_idx = next(
        (idx for idx, message in enumerate(messages) if isinstance(message, dict) and message.get("timestamp") == timestamp),
        -1,
    )
    if target_idx < 0:
        return timestamp

    target = messages[target_idx]
    if isinstance(target, dict) and _is_reasoning_like_entry(target):
        return timestamp

    preferred_source = _message_source(target) if isinstance(target, dict) else ""

    for idx in range(target_idx - 1, -1, -1):
        candidate = messages[idx]
        if not isinstance(candidate, dict):
            continue
        if preferred_source and _message_source(candidate) != preferred_source:
            continue
        if _is_reasoning_like_entry(candidate):
            candidate_ts = candidate.get("timestamp")
            if isinstance(candidate_ts, int):
                return candidate_ts

    for idx in range(target_idx - 1, -1, -1):
        candidate = messages[idx]
        if not isinstance(candidate, dict):
            continue
        if _is_reasoning_like_entry(candidate):
            candidate_ts = candidate.get("timestamp")
            if isinstance(candidate_ts, int):
                return candidate_ts

    return timestamp


def _replace_content_in_payload(payload: Dict[str, Any], replacement_text: str) -> Optional[Dict[str, Any]]:
    updated = dict(payload)

    if isinstance(updated.get("content"), str):
        updated["content"] = replacement_text
        return updated

    if updated.get("type") == "GroupChatMessage" and isinstance(updated.get("message"), dict):
        inner = dict(updated["message"])
        if isinstance(inner.get("content"), str):
            inner["content"] = replacement_text
            updated["message"] = inner
            return updated

    if updated.get("type") == "GroupChatAgentResponse" and isinstance(updated.get("response"), dict):
        response = dict(updated["response"])
        chat_message = response.get("chat_message")
        if isinstance(chat_message, dict) and isinstance(chat_message.get("content"), str):
            new_chat_message = dict(chat_message)
            new_chat_message["content"] = replacement_text
            response["chat_message"] = new_chat_message
            updated["response"] = response
            return updated

    return None


def _build_agent_response_repair_body(entry: Dict[str, Any], replacement_text: str) -> Optional[Dict[str, Any]]:
    message = entry.get("message")
    if not isinstance(message, dict) or message.get("type") != "GroupChatAgentResponse":
        return None
    response = message.get("response")
    if not isinstance(response, dict):
        return None

    chat_message = response.get("chat_message")
    chat_source = "ToolUniverseAgent"
    if isinstance(chat_message, dict) and isinstance(chat_message.get("source"), str):
        chat_source = str(chat_message["source"])

    updated_response = dict(response)
    updated_response["chat_message"] = {
        "type": "TextMessage",
        "source": chat_source,
        "content": replacement_text,
    }
    updated_response["inner_messages"] = [
        {
            "type": "ThoughtEvent",
            "source": chat_source,
            "content": replacement_text,
        }
    ]

    return {
        "type": "GroupChatAgentResponse",
        "response": updated_response,
        "name": message.get("name"),
        "_preserve_content": True,
    }


def _build_replacement_body_from_history_entry(entry: Dict[str, Any], replacement_text: str) -> Dict[str, Any]:
    agent_response_repair = _build_agent_response_repair_body(entry, replacement_text)
    if agent_response_repair is not None:
        return agent_response_repair

    if _entry_is_final_answer_like(entry):
        source = _message_source(entry) or "ToolUniverseAgent"
        updated = {
            "type": "GroupChatMessage",
            "message": {
                "type": "ThoughtEvent",
                "source": source,
                "content": replacement_text,
            },
            "_preserve_content": True,
        }
        return updated

    message = entry.get("message")
    if isinstance(message, dict):
        updated = _replace_content_in_payload(message, replacement_text)
        if updated is not None:
            updated["_preserve_content"] = True
            return updated

    updated = _replace_content_in_payload(entry, replacement_text)
    if updated is not None:
        updated["_preserve_content"] = True
        return updated

    updated = _make_group_chat_message_from_raw_text(replacement_text, source=_message_source(entry) or "ToolUniverseAgent")
    updated["_preserve_content"] = True
    return updated


def _resolve_history_timestamp(client: "AGDebuggerClient", raw_timestamp: Any) -> int:
    if isinstance(raw_timestamp, int):
        return raw_timestamp
    if isinstance(raw_timestamp, str) and raw_timestamp.isdigit():
        return int(raw_timestamp)

    state = client.get_session_history()
    for message in _iter_history_messages(state):
        ts = message.get("timestamp")
        if not isinstance(ts, int):
            continue
        if raw_timestamp == _message_created_at(message):
            return ts
        if isinstance(raw_timestamp, str) and raw_timestamp == str(ts):
            return ts
        if isinstance(raw_timestamp, str) and _message_contains_string(message, raw_timestamp):
            return ts

    raise RuntimeError(f"Unable to resolve history timestamp from planner payload: {raw_timestamp!r}")


def _message_contains_string(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return value == needle
    if isinstance(value, dict):
        return any(_message_contains_string(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(_message_contains_string(v, needle) for v in value)
    return False


def _latest_assistant_timestamp_from_compact(compact: Dict[str, Any]) -> Optional[int]:
    for message in reversed(compact.get("history_tail", [])):
        if not isinstance(message, dict):
            continue
        ts = message.get("timestamp")
        if not isinstance(ts, int):
            continue
        if _is_editable_assistant_entry(message):
            return ts
    return None


def _assistant_timestamp_for_turn(compact: Dict[str, Any], turn_number: Any) -> Optional[int]:
    if isinstance(turn_number, str) and turn_number.isdigit():
        turn_number = int(turn_number)
    if not isinstance(turn_number, (int, float)):
        return None
    fallback_turn = 0
    for message in compact.get("history_tail", []):
        if not isinstance(message, dict):
            continue
        if not _is_editable_assistant_entry(message):
            continue
        ts = message.get("timestamp")
        if not isinstance(ts, int):
            continue
        assistant_turn = message.get("assistant_turn")
        if not isinstance(assistant_turn, int):
            fallback_turn += 1
            assistant_turn = fallback_turn
        if isinstance(assistant_turn, int) and assistant_turn == int(turn_number):
            return ts
    return None


def _latest_assistant_turn_from_compact(compact: Dict[str, Any]) -> Optional[int]:
    latest_turn = None
    fallback_turn = 0
    for message in compact.get("history_tail", []):
        if not isinstance(message, dict):
            continue
        if not _is_editable_assistant_entry(message):
            continue
        assistant_turn = message.get("assistant_turn")
        if not isinstance(assistant_turn, int):
            fallback_turn += 1
            assistant_turn = fallback_turn
        latest_turn = assistant_turn
    return latest_turn


def _assistant_timestamp_for_turn_in_messages(messages: list[Dict[str, Any]], turn_number: Any) -> Optional[int]:
    if isinstance(turn_number, str) and turn_number.isdigit():
        turn_number = int(turn_number)
    if not isinstance(turn_number, (int, float)):
        return None
    assistant_turn = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if not _is_editable_assistant_entry(message):
            continue
        ts = message.get("timestamp")
        if not isinstance(ts, int):
            continue
        assistant_turn += 1
        if assistant_turn == int(turn_number):
            return ts
    return None


def _repair_target_summaries(compact: Dict[str, Any]) -> list[Dict[str, Any]]:
    targets: list[Dict[str, Any]] = []
    fallback_turn = 0
    for message in compact.get("history_tail", []):
        if not isinstance(message, dict):
            continue
        if not _is_editable_assistant_entry(message):
            continue
        ts = message.get("timestamp")
        if not isinstance(ts, int):
            continue
        source = _message_source(message)
        assistant_turn = message.get("assistant_turn")
        if not isinstance(assistant_turn, int):
            fallback_turn += 1
            assistant_turn = fallback_turn
        targets.append(
            {
                "assistant_turn": assistant_turn,
                "timestamp": ts,
                "source": source,
                "preview": _compact_text_preview(_concept_only_planner_text(_message_content(message))),
            }
        )
    return targets


def _output_topic_from_compact(compact: Dict[str, Any]) -> Optional[str]:
    for topic in compact.get("topics", []):
        if isinstance(topic, str) and topic.startswith("output_topic_"):
            return topic
    return None


def _group_topic_from_compact(compact: Dict[str, Any]) -> Optional[str]:
    for topic in compact.get("topics", []):
        if isinstance(topic, str) and topic.startswith("group_topic_"):
            return topic
    return None


def _target_timestamp_from_analysis(compact: Dict[str, Any], analysis_context: Dict[str, Any] | None) -> Optional[int]:
    if not isinstance(analysis_context, dict):
        return None
    repair = analysis_context.get("concept_repair")
    if not isinstance(repair, dict):
        return None
    concepts = repair.get("hallucinated_concepts")
    if not isinstance(concepts, list):
        return None

    preferred_claim_id = analysis_context.get("selected_claim_id")
    debugger_feedback = analysis_context.get("debugger_feedback")
    if preferred_claim_id is None and isinstance(debugger_feedback, dict):
        preferred_claim_id = debugger_feedback.get("selected_claim_id")

    repair_targets = list(_repair_target_summaries(compact))
    known_timestamps = {
        int(item["timestamp"])
        for item in repair_targets
        if isinstance(item.get("timestamp"), int)
    }

    # PR-3: if the planner explicitly named the assistant turn it wants to
    # repair via ``target_turn_number``, honour it before falling back to
    # the earliest hallucinated concept's source_timestamp. Without this the
    # controller would re-target the oldest flagged turn even when the
    # planner has reasoned about a specific later step.
    planner_judgment = analysis_context.get("planner_judgment")
    planner_target_turn: int | None = None
    if isinstance(planner_judgment, dict):
        raw = planner_judgment.get("target_turn_number")
        if isinstance(raw, int):
            planner_target_turn = raw
        elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
            planner_target_turn = int(raw.strip())
    if planner_target_turn is not None:
        for item in repair_targets:
            if (
                isinstance(item.get("assistant_turn"), int)
                and item["assistant_turn"] == planner_target_turn
                and isinstance(item.get("timestamp"), int)
            ):
                return int(item["timestamp"])

    sorted_concepts = sorted(
        (item for item in concepts if isinstance(item, dict)),
        key=lambda item: float(item.get("turn_number")) if isinstance(item.get("turn_number"), (int, float)) else float("inf"),
    )
    if preferred_claim_id is not None:
        preferred_matches = [item for item in sorted_concepts if item.get("claim_id") == preferred_claim_id]
        if preferred_matches:
            sorted_concepts = preferred_matches
    for item in sorted_concepts:
        source_timestamp = item.get("source_timestamp")
        if isinstance(source_timestamp, int) and source_timestamp in known_timestamps:
            return source_timestamp
    return None


def _target_turn_from_analysis(compact: Dict[str, Any], analysis_context: Dict[str, Any] | None) -> Optional[int]:
    ts = _target_timestamp_from_analysis(compact, analysis_context)
    if ts is None:
        return None
    for item in _repair_target_summaries(compact):
        if item.get("timestamp") == ts and isinstance(item.get("assistant_turn"), int):
            return int(item["assistant_turn"])
    return None


def _action_template_block(
    compact: Dict[str, Any],
    analysis_context: Dict[str, Any] | None,
    *,
    strict_concept_repair_only: bool = False,
) -> str:
    repair_targets = _repair_target_summaries(compact)
    preferred_turn = _target_turn_from_analysis(compact, analysis_context)
    if preferred_turn is None:
        preferred_turn = _latest_assistant_turn_from_compact(compact)

    target_lines = []
    for item in repair_targets[:8]:
        target_lines.append(
            f'- assistant_turn={item["assistant_turn"]}, timestamp={item["timestamp"]}, '
            f'source={item["source"]}, preview="{item["preview"]}"'
        )
    target_block = "\n".join(target_lines) if target_lines else "- (no assistant repair targets found)"

    template_lines = [
        "Concrete action templates for THIS snapshot. Copy one template and replace only placeholder values.",
        "Never output XML like <action>...</action>.",
        "For edit_and_revert, do NOT output `timestamp` directly.",
        "Prefer this schema: {\"action\":\"edit_and_revert\",\"claim_id\":\"...\",\"target_turn\":N}. target_turn is optional fallback only.",
        "Do not generate broad repair prose. Select the claim_id to repair and let the controller synthesize the local replacement_text.",
        "Do not use replacement_text to rank answer candidates, restate the final conclusion, or infer which answer becomes correct.",
        "Never use message `id` as `timestamp`; target_turn must come from the valid assistant turns below.",
        "Valid editable assistant turns:",
        target_block,
    ]
    if preferred_turn is not None:
        template_lines.extend(
            [
                "",
                "Preferred repair template:",
                "{",
                '  "action": "edit_and_revert",',
                '  "claim_id": "<repairable claim_id>",',
                f'  "target_turn": {preferred_turn},',
                '  "wrong_reasoning_span": {"anchor_start": "<optional: verbatim substring where wrong reasoning begins>"}',
                "}",
                "(Omit wrong_reasoning_span entirely if the target message has no clear wrong-reasoning block to truncate.)",
            ]
        )
    template_lines.extend(
        [
            "",
            "Repair rules:",
            "- Only use edit_and_revert for repairs.",
            "- Prefer claim_id as the primary repair target key; use target_turn only as a fallback.",
            "- Let the controller synthesize the local replacement_text from the selected claim.",
            "- If you cannot confidently select a repairable claim_id, do not invent replacement text.",
            "- Do not include <answer> tags.",
            "- Do not include TERMINATE.",
            "- Do not state a final answer choice directly.",
            "- Do not say 'the best explanation is', 'the correct choice is', or similar answer-selection language.",
            "- Do not use publish, send, or insert_after to force a final answer.",
        ]
    )
    template_lines.extend(
        [
            "",
            "Do not invent recipient/topic/timestamp values outside the templates above.",
        ]
    )
    return "\n".join(template_lines)


def _normalize_planner_output(action: Dict[str, Any], compact: Dict[str, Any]) -> Dict[str, Any]:
    if "action" in action:
        return action

    if (
        "type" in action
        or "message" in action
        or ("source" in action and "content" in action)
        or "messages" in action
    ):
        timestamp = _latest_assistant_timestamp_from_compact(compact)
        if timestamp is None:
            raise ValueError(f"Planner returned a message payload but no editable assistant timestamp was found: {action}")
        return {
            "action": "edit_and_revert",
            "timestamp": timestamp,
            "body": _coerce_message_body(action),
        }

    return action


def _choose_fallback_edit_target(
    compact: Dict[str, Any],
    analysis_context: Dict[str, Any] | None,
) -> tuple[Optional[int], Optional[int]]:
    target_turn = _target_turn_from_analysis(compact, analysis_context)
    target_timestamp = _target_timestamp_from_analysis(compact, analysis_context)
    if target_turn is None and isinstance(target_timestamp, int):
        for item in _repair_target_summaries(compact):
            if item.get("timestamp") == target_timestamp and isinstance(item.get("assistant_turn"), int):
                target_turn = int(item["assistant_turn"])
                break
    if target_turn is None:
        target_turn = _latest_assistant_turn_from_compact(compact)
    if target_timestamp is None and target_turn is not None:
        target_timestamp = _assistant_timestamp_for_turn(compact, target_turn)
    return target_turn, target_timestamp


def _sanitize_edit_and_revert_target(
    action: Dict[str, Any],
    compact: Dict[str, Any],
    analysis_context: Dict[str, Any] | None,
) -> Dict[str, Any]:
    normalized = dict(action)
    valid_targets = _repair_target_summaries(compact)
    valid_turns = {
        int(item["assistant_turn"])
        for item in valid_targets
        if isinstance(item.get("assistant_turn"), int)
    }
    valid_timestamps = {
        int(item["timestamp"])
        for item in valid_targets
        if isinstance(item.get("timestamp"), int)
    }

    target_turn = normalized.get("target_turn")
    if isinstance(target_turn, str) and target_turn.isdigit():
        target_turn = int(target_turn)
        normalized["target_turn"] = target_turn
    timestamp = normalized.get("timestamp")
    if isinstance(timestamp, str) and timestamp.isdigit():
        timestamp = int(timestamp)
        normalized["timestamp"] = timestamp

    fallback_turn, fallback_timestamp = _choose_fallback_edit_target(compact, analysis_context)

    if not isinstance(target_turn, (int, float)) or int(target_turn) not in valid_turns:
        if fallback_turn is not None:
            normalized["target_turn"] = fallback_turn
            target_turn = fallback_turn
        else:
            normalized.pop("target_turn", None)
            target_turn = None

    if not isinstance(timestamp, int) or timestamp not in valid_timestamps:
        if isinstance(target_turn, (int, float)):
            resolved = _assistant_timestamp_for_turn(compact, int(target_turn))
            if resolved is not None:
                normalized["timestamp"] = resolved
                timestamp = resolved
        elif fallback_timestamp is not None:
            normalized["timestamp"] = fallback_timestamp
            timestamp = fallback_timestamp

    return normalized


def _fill_missing_action_fields(action: Dict[str, Any], compact: Dict[str, Any], analysis_context: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = dict(action)
    if normalized.get("action") == "edit_and_revert":
        claim_id = normalized.get("claim_id")
        if claim_id is not None and isinstance(analysis_context, dict):
            normalized_analysis_context = dict(analysis_context)
            normalized_analysis_context["selected_claim_id"] = claim_id
        else:
            normalized_analysis_context = analysis_context
        if "replacement_text" not in normalized and isinstance(normalized.get("body"), str):
            normalized["replacement_text"] = normalized["body"]
        if "timestamp" not in normalized:
            target_timestamp = _target_timestamp_from_analysis(compact, normalized_analysis_context)
            if target_timestamp is not None:
                normalized["timestamp"] = target_timestamp
        if "target_turn" not in normalized and "timestamp" not in normalized:
            target_turn = _target_turn_from_analysis(compact, normalized_analysis_context)
            if target_turn is None:
                target_turn = _latest_assistant_turn_from_compact(compact)
            if target_turn is not None:
                normalized["target_turn"] = target_turn
        normalized = _sanitize_edit_and_revert_target(normalized, compact, normalized_analysis_context)
    return normalized


def _resolve_recipient_alias(client: "AGDebuggerClient", recipient: str) -> str:
    if "/" in recipient:
        recipient = recipient.split("/", 1)[0]
    if recipient not in {"manager", "group_manager", "RoundRobinGroupChatManager"}:
        return recipient
    topics = client.get_topics()
    manager_topic = _guess_manager_topic(topics)
    if not manager_topic:
        raise RuntimeError("Unable to resolve manager recipient from current topics.")
    return manager_topic


class AGDebuggerClient:
    def __init__(self, base_url: str, timeout_sec: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._opener = self._build_opener()

    def _build_opener(self):
        parsed = urlparse(self.base_url)
        hostname = parsed.hostname or ""
        if hostname == "localhost":
            return request.build_opener(request.ProxyHandler({}))

        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            return request.build_opener()

        if ip.is_loopback:
            return request.build_opener(request.ProxyHandler({}))
        return request.build_opener()

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout_sec: Optional[float] = None,
    ) -> Any:
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = _compact_json(payload).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}{path}",
            method=method.upper(),
            data=body,
            headers=headers,
        )
        try:
            with self._opener.open(req, timeout=self.timeout_sec if timeout_sec is None else timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("status") == "error":
                    raise RuntimeError(f"Backend {method} {path} error: {parsed.get('message', '(unknown error)')}")
                return parsed
        except error.HTTPError as e:
            details = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code} {method} {path}: {details}") from e
        except error.URLError as e:
            raise RuntimeError(f"Failed to connect to {self.base_url}: {e}") from e

    def get_topics(self) -> list[str]:
        return self._request("GET", "/topics")

    def get_message_types(self) -> Dict[str, Any]:
        return self._request("GET", "/message_types")

    def get_session_history(self) -> Dict[str, Any]:
        return self._request("GET", "/getSessionHistory")

    def get_queue(self) -> list[Dict[str, Any]]:
        return self._request("GET", "/getMessageQueue")

    def get_num_tasks(self) -> int:
        return int(self._request("GET", "/num_tasks"))

    def get_loop_status(self) -> bool:
        return bool(self._request("GET", "/loop_status"))

    def get_agents(self) -> list[str]:
        return self._request("GET", "/agents")

    def get_logs(self) -> list[Dict[str, Any]]:
        return self._request("GET", "/logs")

    def step(self) -> Any:
        return self._request("POST", "/step", {})

    def drop(self) -> Any:
        return self._request("POST", "/drop", {})

    def start_loop(self) -> Any:
        return self._request("POST", "/start_loop", {})

    def stop_loop(self, force: bool = False, timeout_sec: Optional[float] = None) -> Any:
        suffix = "?force=true" if force else ""
        return self._request("POST", f"/stop_loop{suffix}", {}, timeout_sec=timeout_sec)

    def team_reset(self, timeout_sec: Optional[float] = None) -> Any:
        """Full team-level reset: clears participant model_context, resets
        the manager, and drains the output queue."""
        return self._request("POST", "/team_reset", {}, timeout_sec=timeout_sec)

    def publish(self, topic: str, body: Dict[str, Any]) -> Any:
        return self._request("POST", "/publish", {"type": body.get("type", ""), "topic": topic, "body": body})

    def send(self, recipient: str, body: Dict[str, Any]) -> Any:
        return self._request("POST", "/send", {"recipient": recipient, "type": body.get("type", ""), "body": body})

    def edit_queue(self, idx: int, body: Dict[str, Any]) -> Any:
        return self._request("POST", "/editQueue", {"idx": idx, "body": body})

    def edit_and_revert(self, timestamp: int, body: Optional[Dict[str, Any]]) -> Any:
        payload: Dict[str, Any] = {"timestamp": timestamp}
        if body is not None:
            payload["body"] = body
        return self._request("POST", "/editAndRevertHistoryMessage", payload)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "agents": self.get_agents(),
            "topics": self.get_topics(),
            "message_types": list(self.get_message_types().keys()),
            "num_tasks": self.get_num_tasks(),
            "loop_running": self.get_loop_status(),
            "queue": self.get_queue(),
            "session_history": self.get_session_history(),
        }

    def wait_until_idle(self, timeout_sec: float = 60.0, poll_sec: float = 0.25, warmup_sec: float = 5.0) -> None:
        """Wait until the backend has no pending tasks.

        Wait for a stable idle state rather than a single ``num_tasks == 0``
        observation. This reduces the chance of missing very short-lived work
        that starts and finishes between two polls.
        """
        deadline = time.time() + timeout_sec
        warmup_end = time.time() + min(warmup_sec, timeout_sec)
        saw_queued_or_running_work = False
        consecutive_idle_polls = 0

        while time.time() < deadline:
            num_tasks = self.get_num_tasks()
            loop_running = self.get_loop_status()
            queue_nonempty = bool(self.get_queue())

            if num_tasks > 0 or queue_nonempty:
                saw_queued_or_running_work = True
                consecutive_idle_polls = 0
                time.sleep(poll_sec)
                continue

            if time.time() < warmup_end and loop_running and not saw_queued_or_running_work:
                consecutive_idle_polls = 0
                time.sleep(poll_sec)
                continue

            if saw_queued_or_running_work or time.time() >= warmup_end:
                consecutive_idle_polls += 1
                if consecutive_idle_polls >= 2:
                    return
            time.sleep(poll_sec)
        raise TimeoutError("Timed out waiting for backend to become idle.")

    def insert_after(
        self,
        timestamp: int,
        *,
        mode: str,
        body: Dict[str, Any],
        topic: Optional[str] = None,
        recipient: Optional[str] = None,
    ) -> None:
        # "Insert in middle" is implemented as:
        # 1) revert to timestamp
        # 2) add a new message
        self.stop_loop()
        self.edit_and_revert(timestamp, None)
        if mode == "publish":
            if not topic:
                raise ValueError("insert_after with mode=publish requires topic.")
            self.publish(topic, body)
        elif mode == "send":
            if not recipient:
                raise ValueError("insert_after with mode=send requires recipient.")
            self.send(recipient, body)
        else:
            raise ValueError(f"Unsupported insert mode: {mode}")


def _guess_manager_topic(topics: list[str]) -> Optional[str]:
    for t in topics:
        if "manager" in t.lower():
            return t
    return topics[0] if topics else None


def _guess_group_chat_broadcast_topic(topics: list[str]) -> Optional[str]:
    """Pick the broadcast topic where chat turns should appear in the transcript.

    ``_guess_manager_topic`` picks the manager's private inbox, which is wrong
    for injecting a turn that should show up in the group-chat history (the
    manager receives it but does not relay it to participants). The broadcast
    topic is conventionally named ``group_topic_<uuid>`` in AutoGen's
    RoundRobinGroupChat; fall back to the first non-output, non-manager,
    non-agent topic if that exact prefix is missing.
    """
    for t in topics:
        if t.startswith("group_topic_"):
            return t
    for t in topics:
        lowered = t.lower()
        if "manager" in lowered or "output_topic" in lowered:
            continue
        # Skip per-agent inbox topics like "ToolUniverseAgent_<uuid>".
        if re.match(r"^[A-Z][A-Za-z0-9]*_[0-9a-f-]{8,}$", t):
            continue
        return t
    return topics[0] if topics else None


def _chat_text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


class RulePlanner:
    """Simple non-LLM planner for local validation."""

    def __init__(self, task: str):
        self.task = task
        self.started = False

    def plan(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        topics: list[str] = snapshot.get("topics", [])
        history_state: Dict[str, Any] = snapshot.get("session_history", {})
        message_history = history_state.get("message_history", {})

        manager_topic = _guess_manager_topic(topics)
        if not self.started and manager_topic:
            self.started = True
            return {
                "action": "send",
                "recipient": manager_topic,
                "body": {
                    "type": "GroupChatStart",
                    "messages": [{"type": "TextMessage", "source": "user", "content": self.task}],
                },
            }

        if snapshot.get("num_tasks", 0) > 0:
            return {"action": "step"}

        # If queue empty and we have some history, stop.
        if message_history:
            return {"action": "finish", "reason": "no pending tasks"}

        return {"action": "finish", "reason": "nothing to do"}


class LLMPlanner:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        user_goal: str,
        strict_concept_repair_only: bool = False,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package is required for --planner llm mode.") from e

        _configure_external_llm_proxy(base_url)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.environ.get("AGDEBUGGER_PLANNER_TIMEOUT_SEC", "45")),
            max_retries=2,
        )
        self.model = model
        self.user_goal = user_goal
        self.strict_concept_repair_only = strict_concept_repair_only
        # Wire AGENTDEBUG_THINKING_MODE so the planner LLM picks up
        # extra_body={"thinking_mode": ...}. Previously this was hard-coded to
        # None, silently dropping the env var on the planner request path.
        self._extra_body = build_openai_extra_body() or None
        self.system_prompt = (
            "You control AGDebugger via JSON actions.\n"
            "Return exactly one JSON object with key `action`.\n"
            "You are a concept-repair controller, not a question-solving agent.\n"
            "You are not allowed to provide, guess, or encode the final answer choice.\n"
            "For scientific-question debugging, follow this workflow strictly:\n"
            "1) inspect the scientific concept analysis block first;\n"
            "2) identify each incorrect or unsupported concept, or incorrect concept interaction, that the assistant relied on;\n"
            "3) select the earliest assistant step where that hallucinated concept or interaction appears;\n"
            "4) replace only that local concept understanding with the fact-checked understanding or grounded repair guidance;\n"
            "5) then rerun the trajectory from the earliest affected assistant turn.\n"
            "Your job is to repair local concept understanding inside the trace, not to choose an option.\n"
            "Your actions must obey AGDebugger's message schema.\n"
            "Allowed actions:\n"
            "- step\n"
            "- drop\n"
            "- start_loop\n"
            "- stop_loop\n"
            "- publish: {action, topic, body}\n"
            "- send: {action, recipient, body}\n"
            "- edit_queue: {action, idx, body}\n"
            "- edit_and_revert: {action, claim_id, target_turn?}\n"
            "- insert_after: {action, timestamp, mode: publish|send, topic?/recipient?, body}\n"
            "- finish: {action, reason}\n"
            "If you need to start a conversation, send GroupChatStart to manager topic using send.\n"
            "Do NOT send a bare TextMessage or a body wrapped only as {message: {...}}.\n"
            "Bodies for send/publish/edit actions must be valid top-level messages such as "
            "GroupChatStart or GroupChatMessage.\n"
            "Use exact topic/recipient ids from the runtime snapshot; do not use aliases like 'manager'.\n"
            "For edit_and_revert, select the repair target by claim_id and optionally provide target_turn as a fallback. "
            "Do not emit timestamp for edit_and_revert; the controller will resolve it.\n"
            "If you use insert_after, the timestamp must exactly match an existing "
            "history message timestamp from the runtime snapshot.\n"
            "If concept_repair.hallucinated_concepts is non-empty, your first repair action should target "
            "the earliest repairable claim_id and its assistant turn.\n"
            "Do not invent a replacement concept: use concept_repair.hallucinated_concepts[].corrected_claim_text.\n"
            "Respect concept_repair.hallucinated_concepts[].error_type: fact_error repairs facts, mapping_error repairs evidence-to-answer links, "
            "constraint_error repairs domain/subject mismatches, and alignment_error repairs final-answer misalignment.\n"
            "Preserve correct parts of the trajectory; change only the incorrect or unsupported claim span and the minimum dependent wording needed for coherence.\n"
            "The controller will synthesize replacement_text by replacing the selected faulty claim span with corrected_claim_text while preserving surrounding reasoning.\n"
            "OPTIONAL: alongside claim_id you may return `wrong_reasoning_span` "
            "to tell the controller where the agent's unsupported reasoning "
            "begins in the target message, so the prefix of that message can "
            "be truncated and only the factual setup before it is kept. "
            "Use this when the target message contains a long block of "
            "option-by-option analysis, 'Let me evaluate each option', "
            "'I will re-evaluate', 'Given the lack of...', 'Based on my "
            "understanding', or similar self-argumentation patterns that "
            "argued for the wrong answer. Do NOT use it for short factual "
            "messages or messages whose content is correct up to the faulty "
            "span. Format: "
            '{"wrong_reasoning_span": {"anchor_start": "<verbatim substring from the target message where the wrong reasoning starts>"}}. '
            "`anchor_start` MUST be a literal substring copy from the target "
            "message (the controller will locate it with str.find); at least "
            "8 characters; never a paraphrase. If you are not confident, omit "
            "this field and the controller will fall back to its default.\n"
            "If you cannot identify a repairable claim_id, return finish rather than inventing replacement text.\n"
            "Do not publish or send a final answer.\n"
            "If debugger feedback says a previous action failed or repeated without effect, choose a different action."
        )

    def _request_action_text(
        self,
        user_prompt: str,
        *,
        invalid_output: str | None = None,
        action_templates: str = "",
    ) -> str:
        user_content = f"{user_prompt}\n\n{action_templates}" if action_templates else user_prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if invalid_output is not None:
            messages.extend(
                [
                    {"role": "assistant", "content": invalid_output},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was invalid because it was not a single JSON object. "
                            "Return exactly one JSON object with key `action` and no extra prose, tags, or code fences.\n\n"
                            f"{action_templates}"
                        ),
                    },
                ]
            )
        # Estimate tokens for the RPM/TPM limiter. When a retry message is
        # included the full message list is larger than the initial prompt;
        # join their contents for a tighter estimate.
        joined_user = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
        limiter = get_shared_limiter()
        limiter.acquire_sync(
            estimate_request_tokens(system_prompt=self.system_prompt, user_prompt=joined_user)
        )

        def _call() -> str:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
                **({"extra_body": self._extra_body} if self._extra_body else {}),
            )
            actual = response_total_tokens(completion)
            if actual is not None:
                limiter.record_actual(actual)
            return completion.choices[0].message.content or ""

        return with_retry_sync(_call, label=f"planner({self.model})")

    def _compact_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        queue = _sanitize_for_planner(snapshot.get("queue", [])[-6:])
        history_state = snapshot.get("session_history", {})
        curr = history_state.get("current_session")
        sessions = history_state.get("message_history", {})
        current_messages = []
        if isinstance(curr, int) or (isinstance(curr, str) and curr.isdigit()):
            key = str(curr)
            if key in sessions:
                current_messages = sessions[key].get("messages", [])
            elif curr in sessions:  # type: ignore[operator]
                current_messages = sessions[curr].get("messages", [])  # type: ignore[index]

        history_tail = []
        assistant_turn = 0
        for message in current_messages:
            if not isinstance(message, dict):
                continue
            source = _message_source(message)
            message_type = _message_type(message)
            if _is_editable_assistant_entry(message):
                assistant_turn += 1
            sanitized_content = _concept_only_planner_text(_message_content(message))
            if source == "user":
                sanitized_content = "[user prompt redacted for concept-level repair]"
            enriched = {
                "timestamp": message.get("timestamp"),
                "source": source,
                "type": message_type,
                "content": sanitized_content,
            }
            if _is_editable_assistant_entry(message):
                enriched["assistant_turn"] = assistant_turn
            history_tail.append(enriched)
        history_tail = history_tail[-12:]

        return {
            "agents": snapshot.get("agents", []),
            "topics": snapshot.get("topics", []),
            "message_types": snapshot.get("message_types", []),
            "num_tasks": snapshot.get("num_tasks", 0),
            "loop_running": snapshot.get("loop_running", False),
            "queue_tail": queue,
            "current_session": curr,
            "history_tail": history_tail,
        }

    def plan(self, snapshot: Dict[str, Any], analysis_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        compact = self._compact_snapshot(snapshot)
        action_templates = _action_template_block(
            compact,
            analysis_context,
            strict_concept_repair_only=self.strict_concept_repair_only,
        )
        analysis_block = ""
        if analysis_context is not None:
            sanitized_analysis_context = _sanitize_analysis_for_planner(analysis_context)
            analysis_block = (
                "Scientific concept extraction and fact-check analysis:\n"
                f"{json.dumps(sanitized_analysis_context, ensure_ascii=False, indent=2)}\n\n"
            )
        mode_block = (
            "Planner mode:\n"
            f"- strict_concept_repair_only: {self.strict_concept_repair_only}\n"
            "- Repairs must stay concept-level and let the downstream rerun derive the answer.\n"
            "- If true, halt instead of repairing when no repairable concepts are available.\n\n"
        )
        user_prompt = (
            f"Goal:\n{self.user_goal}\n\n"
            f"{mode_block}"
            f"{analysis_block}"
            f"Runtime snapshot:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
            "Return the next best action as JSON only."
        )
        invalid_output: str | None = None
        last_error: Exception | None = None
        for _ in range(2):
            text = self._request_action_text(
                user_prompt,
                invalid_output=invalid_output,
                action_templates=action_templates,
            )
            try:
                action = _extract_first_json_object(text)
                action = _normalize_planner_output(action, compact)
                action = _fill_missing_action_fields(action, compact, analysis_context)
                return action
            except Exception as exc:  # noqa: BLE001
                invalid_output = text
                last_error = exc

        raise ValueError(
            f"Planner failed to return valid JSON action after retry. "
            f"Last error: {last_error}. Last output: {invalid_output[:400] if invalid_output else '(empty)'}"
        )


def _ensure_body_dict(action: Dict[str, Any], key: str = "body") -> Dict[str, Any]:
    """Normalise *body* to a dict — LLM sometimes returns a plain string."""
    val = action.get(key)
    if val is None:
        return {}
    if isinstance(val, str):
        return _make_group_chat_message_from_text(val)
    if isinstance(val, dict):
        return _coerce_message_body(val)
    return val


def execute_action(client: AGDebuggerClient, action: Dict[str, Any]) -> bool:
    action = dict(action)
    if action.get("action") == "edit_and_revert":
        current_messages = _session_messages(client.get_session_history())
        if action.get("timestamp") is None and action.get("target_turn") is not None:
            timestamp = _assistant_timestamp_for_turn_in_messages(current_messages, action.get("target_turn"))
            if timestamp is None:
                raise RuntimeError(f"Invalid edit_and_revert target_turn: {action.get('target_turn')}")
            action["timestamp"] = timestamp
        replacement_text = action.get("replacement_text")
        if isinstance(replacement_text, str):
            timestamp = action.get("timestamp")
            target_entry = None
            if isinstance(timestamp, int):
                target_entry = _history_entry_by_timestamp(current_messages, timestamp)
            if isinstance(timestamp, int) and not _contains_answer_or_terminate(replacement_text):
                preferred_timestamp = _prefer_reasoning_timestamp_for_repair(current_messages, timestamp)
                if preferred_timestamp != timestamp:
                    timestamp = preferred_timestamp
                    action["timestamp"] = preferred_timestamp
                    target_entry = _history_entry_by_timestamp(current_messages, preferred_timestamp)
                if isinstance(target_entry, dict):
                    response_entry = _find_enclosing_agent_response_entry(current_messages, target_entry)
                    if response_entry is not None and isinstance(response_entry.get("timestamp"), int):
                        action["timestamp"] = int(response_entry["timestamp"])
                        target_entry = response_entry
            if action.get("body") is None and isinstance(action.get("timestamp"), int):
                if target_entry is None:
                    target_entry = _history_entry_by_timestamp(current_messages, int(action["timestamp"]))
                if isinstance(target_entry, dict):
                    action["body"] = _build_replacement_body_from_history_entry(target_entry, replacement_text)
        if action.get("replacement_text") is not None and action.get("body") is None:
            action["body"] = action["replacement_text"]
    action = _canonicalize_action(action)
    name = action.get("action")
    if not isinstance(name, str):
        raise ValueError(f"Invalid action payload (missing action): {action}")

    if name == "step":
        client.step()
        return True
    if name == "drop":
        client.drop()
        return True
    if name == "start_loop":
        client.start_loop()
        return True
    if name == "stop_loop":
        client.stop_loop()
        return True
    if name == "publish":
        client.publish(topic=action["topic"], body=_ensure_body_dict(action))
        return True
    if name == "send":
        recipient = _resolve_recipient_alias(client, str(action["recipient"]))
        client.send(recipient=recipient, body=_ensure_body_dict(action))
        return True
    if name == "edit_queue":
        client.edit_queue(idx=int(action["idx"]), body=_ensure_body_dict(action))
        return True
    if name == "edit_and_revert":
        timestamp = _resolve_history_timestamp(client, action["timestamp"])
        if timestamp not in _valid_history_timestamps(client.get_session_history()):
            raise RuntimeError(f"Invalid edit_and_revert timestamp: {timestamp}")
        body = None if action.get("body") is None else _ensure_body_dict(action)
        client.edit_and_revert(timestamp=timestamp, body=body)

        if _post_edit_critic_enabled():
            critic_text = _compose_post_edit_critic_message(action.get("critic_followup"))
            if critic_text:
                try:
                    _deliver_post_edit_critic(client, critic_text)
                except Exception as exc:  # noqa: BLE001
                    # Best-effort — never block the core repair on a critic failure.
                    print(f"[warn] post-edit critic publish failed: {type(exc).__name__}: {exc}")
        return True
    if name == "insert_after":
        timestamp = _resolve_history_timestamp(client, action["timestamp"])
        if timestamp not in _valid_history_timestamps(client.get_session_history()):
            raise RuntimeError(f"Invalid insert_after timestamp: {timestamp}")
        recipient = action.get("recipient")
        if isinstance(recipient, str):
            recipient = _resolve_recipient_alias(client, recipient)
        client.insert_after(
            timestamp=timestamp,
            mode=str(action["mode"]),
            body=_ensure_body_dict(action),
            topic=action.get("topic"),
            recipient=recipient,
        )
        return True
    if name == "finish":
        return False

    raise ValueError(f"Unknown action: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External AGDebugger controller")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AGDEBUGGER_API_BASE", "http://127.0.0.1:8081/api"),
        help="AGDebugger API base URL",
    )
    parser.add_argument(
        "--planner",
        choices=["rule", "llm"],
        default="rule",
        help="Planner backend",
    )
    parser.add_argument(
        "--goal",
        default="Answer the user question.",
        help="High-level debug goal for planner",
    )
    parser.add_argument(
        "--task",
        default="What is the weather in Seattle?",
        help="Initial user task when planner needs to bootstrap GroupChatStart",
    )
    parser.add_argument("--max-steps", type=int, default=60, help="Max control iterations")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between iterations")
    parser.add_argument("--wait-idle", action="store_true", help="Wait until idle after each action")
    parser.add_argument(
        "--claim-task",
        choices=[
            "research_questions",
            "medical_guidelines",
            "legal_cases",
            "coding",
            "scientific_concept_discovery",
        ],
        default=None,
        help="Optional claim extraction/judging task used to analyze session history for planner context.",
    )
    parser.add_argument(
        "--strict-concept-repair-only",
        dest="strict_concept_repair_only",
        action="store_true",
        help=(
            "Mirror of run_dataset_autodebug's strict mode. When set, the LLM "
            "planner refuses any answer-rewriting action and only allows local "
            "concept-repair edit_and_revert actions."
        ),
    )
    parser.set_defaults(
        strict_concept_repair_only=os.environ.get(
            "AGDEBUGGER_STRICT_CONCEPT_REPAIR_ONLY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--claim-evidence",
        default="",
        help="Inline evidence text passed into the claim judge.",
    )
    parser.add_argument(
        "--claim-evidence-file",
        default=None,
        help="Path to evidence text passed into the claim judge.",
    )
    parser.add_argument(
        "--claim-analysis-assistant-only",
        action="store_true",
        help="Analyze only assistant turns from AGDebugger session history.",
    )
    parser.add_argument(
        "--claim-use-websearch",
        action="store_true",
        help="Use the local websearch library to build evidence automatically for each extracted claim.",
    )
    parser.add_argument(
        "--claim-search-backend",
        choices=["bright_data", "serper"],
        default="bright_data",
        help="Search backend for automatic claim evidence. Default forces Bright Data client.",
    )
    parser.add_argument("--claim-search-max-searches", type=int, default=3)
    parser.add_argument("--claim-search-num-results", type=int, default=5)
    parser.add_argument("--claim-search-fetch-top-n", type=int, default=2)
    parser.add_argument("--claim-search-max-output-words", type=int, default=1500)

    # LLM settings (used only in --planner llm)
    parser.add_argument("--model", default=os.environ.get("AGENTDEBUG_MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument("--api-key", default=os.environ.get("AGENTDEBUG_OPENAI_API_KEY", ""))
    parser.add_argument(
        "--api-base",
        default=os.environ.get("AGENTDEBUG_OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    return parser.parse_args()


def _load_claim_evidence_text(args: argparse.Namespace) -> str:
    if args.claim_evidence_file:
        with open(args.claim_evidence_file, "r", encoding="utf-8") as f:
            return f.read()
    return args.claim_evidence or ""


def _build_claim_analysis(
    *,
    args: argparse.Namespace,
    snapshot: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not args.claim_task:
        return None
    evidence_text = _load_claim_evidence_text(args)
    claim_api_key = resolve_value_for_model(
        args.model,
        args.api_key or None,
        intern_value=os.environ.get("AGENTDEBUG_INTERN_API_KEY"),
    )
    try:
        return asyncio.run(
            analyze_session_state(
                task=args.claim_task,
                state=snapshot["session_history"],
                model=args.model,
                api_key=claim_api_key,
                base_url=resolve_base_url_for_model(args.model, args.api_base),
                evidence_text=evidence_text,
                use_websearch=args.claim_use_websearch,
                search_backend=args.claim_search_backend,
                search_max_searches=args.claim_search_max_searches,
                search_num_results=args.claim_search_num_results,
                search_fetch_top_n=args.claim_search_fetch_top_n,
                search_max_output_words=args.claim_search_max_output_words,
                assistant_only=args.claim_analysis_assistant_only,
            )
        )
    except Exception as exc:
        return {"analysis_error": str(exc)}


def main() -> None:
    args = parse_args()
    client = AGDebuggerClient(args.base_url)

    if args.planner == "rule":
        planner: Any = RulePlanner(task=args.task)
    else:
        if not args.api_key:
            raise RuntimeError("Missing API key for llm planner. Set --api-key or AGENTDEBUG_OPENAI_API_KEY.")
        planner_api_key = resolve_value_for_model(
            args.model,
            args.api_key,
            intern_value=os.environ.get("AGENTDEBUG_INTERN_API_KEY"),
        )
        planner_base_url = resolve_base_url_for_model(args.model, args.api_base)
        planner = LLMPlanner(
            model=args.model,
            api_key=planner_api_key,
            base_url=planner_base_url,
            user_goal=args.goal,
            strict_concept_repair_only=bool(getattr(args, "strict_concept_repair_only", False)),
        )

    print(f"[controller] base_url={args.base_url}")
    print(f"[controller] planner={args.planner}, max_steps={args.max_steps}")

    for i in range(args.max_steps):
        snapshot = client.snapshot()
        analysis_context = None
        if args.planner == "llm" and args.claim_task:
            analysis_context = _build_claim_analysis(args=args, snapshot=snapshot)
        action = planner.plan(snapshot, analysis_context=analysis_context) if args.planner == "llm" else planner.plan(snapshot)
        print(f"[step {i:03d}] action={_compact_json(action)}")

        should_continue = execute_action(client, action)
        if args.wait_idle:
            client.wait_until_idle()
        if not should_continue:
            print("[controller] finished.")
            return
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("[controller] reached max steps.")


if __name__ == "__main__":
    main()
