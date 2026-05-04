"""Preserve `reasoning_content` across multi-turn requests for thinking models.

Problem: DeepSeek-v4 / GLM-5 / Kimi-k2.5 / InternS1 backends emit
``reasoning_content`` alongside ``content`` (or ``tool_calls``) and **require
the prior assistant turn's reasoning_content to appear in the next-turn
message history**, otherwise the next request 400s with messages like
``reasoning_content must be passed back to the API`` /
``reasoning_content is missing in assistant tool call message at index N``.

Autogen's pipeline already captures reasoning_content into
``AssistantMessage.thought`` (via ``CreateResult.thought`` in
``_openai_client.py``), but its message-→-OpenAI dict transformer for the
tool-call branch (``tools_assistant_transformer_funcs``) only emits
``tool_calls`` + ``content=null``. The ``thought`` field is silently dropped.

Fix: register a custom transformer for these specific model ids that ALSO
emits ``reasoning_content`` on the assistant dict. OpenAI's
``ChatCompletionAssistantMessageParam`` is a TypedDict at runtime → extra
fields pass through unchanged onto the wire request, so OpenAI / Anthropic /
non-thinking gateways tolerate it.

Apply by importing ``evaluation.thinking_model_patch`` once before the OpenAI
client is constructed; ``install_thinking_transformer()`` runs automatically.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai._message_transform import (
    _set_thought_as_content,
    _set_tool_calls,
    assistant_condition,
    assistant_transformer_constructors,
    base_assistant_transformer_funcs,
    function_execution_result_message,
    single_assistant_transformer_funcs,
    system_message_transformers,
    user_condition,
    user_transformer_constructors,
    user_transformer_funcs,
)
from autogen_ext.models.openai._transformation.registry import (
    MESSAGE_TRANSFORMERS,
    build_conditional_transformer_func,
    build_transformer_func,
    register_transformer,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


# Models we're applying the patch to. Substring prefix-match per autogen's
# `_find_model_family` (longest prefix wins). Add new thinking models here.
THINKING_MODEL_IDS = (
    "deepseek-v4",                    # deepseek-v4-flash etc.
    "bailian/deepseek-v4",            # Boyue's bailian-routed alias
    "glm-5",                          # glm-5, glm-5.1, ...
    "kimi-k2",                        # kimi-k2.5
    "intern-s1",                      # intern-s1, intern-s1-pro
    "thinking",                       # gemini-*-thinking, claude-*-thinking — fallback
    "reasoning",                      # grok-*-reasoning — fallback (no-op for grok since TU off)
)


def _set_reasoning_content_from_thought(
    message: LLMMessage, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Emit ``reasoning_content`` on the assistant dict iff thought is preserved."""
    assert isinstance(message, AssistantMessage)
    if message.thought:
        return {"reasoning_content": message.thought}
    return {}


# Tool-call branch: keep tool_calls + null content, ALSO add reasoning_content.
# NOTE we omit _set_null_content_for_tool_calls because some thinking backends
# (e.g. DeepSeek) require ``content`` to be a string (possibly empty) rather
# than ``null`` when reasoning_content is present. We send ``content=""``
# instead of ``content: null`` — slightly safer for fussy backends.
def _set_empty_string_content(
    message: LLMMessage, context: Dict[str, Any]
) -> Dict[str, Any]:
    return {"content": ""}


_thinking_tools_funcs: List[Callable[[LLMMessage, Dict[str, Any]], Dict[str, Any]]] = (
    base_assistant_transformer_funcs
    + [
        _set_tool_calls,
        _set_empty_string_content,
        _set_reasoning_content_from_thought,
    ]
)

# Thought branch (tool_calls + thought-as-content): keep existing behavior
# but also emit reasoning_content explicitly so the backend has both.
_thinking_thought_funcs: List[Callable[[LLMMessage, Dict[str, Any]], Dict[str, Any]]] = (
    base_assistant_transformer_funcs
    + [
        _set_tool_calls,
        _set_thought_as_content,
        _set_reasoning_content_from_thought,
    ]
)

# Text-only branch: assistant message without tool_calls. Some backends still
# want reasoning_content if the model emitted it; preserve via the same hook.
_thinking_text_funcs: List[Callable[[LLMMessage, Dict[str, Any]], Dict[str, Any]]] = (
    single_assistant_transformer_funcs + [_set_reasoning_content_from_thought]
)


_thinking_assistant_funcs_map = {
    "text": _thinking_text_funcs,
    "tools": _thinking_tools_funcs,
    "thought": _thinking_thought_funcs,
}


_THINKING_TRANSFORMER_MAP = {
    SystemMessage: build_transformer_func(
        funcs=system_message_transformers,
        message_param_func=ChatCompletionSystemMessageParam,
    ),
    UserMessage: build_conditional_transformer_func(
        funcs_map=user_transformer_funcs,
        message_param_func_map=user_transformer_constructors,
        condition_func=user_condition,
    ),
    AssistantMessage: build_conditional_transformer_func(
        funcs_map=_thinking_assistant_funcs_map,
        message_param_func_map=assistant_transformer_constructors,
        condition_func=assistant_condition,
    ),
    FunctionExecutionResultMessage: function_execution_result_message,
}


_INSTALLED = False


def install_thinking_transformer(model_ids: tuple[str, ...] = THINKING_MODEL_IDS) -> None:
    """Idempotent. Registers the thinking-aware transformer for each id."""
    global _INSTALLED
    for mid in model_ids:
        register_transformer("openai", mid, _THINKING_TRANSFORMER_MAP)
    _INSTALLED = True


# ---------------------------------------------------------------------------
# Second patch: lift `reasoning_content` into `content` on tool-call responses
#
# Autogen's _openai_client.py at lines ~744-754 has a tool_calls branch that
# reads ``choice.message.content`` to populate ``CreateResult.thought`` but
# DOES NOT inspect ``reasoning_content``. The reasoning_content extraction is
# only in the text-only ``else`` branch (~line 779). Thinking models like
# deepseek-v4-flash with tool_calls return ``content=""`` + non-empty
# ``reasoning_content``, so autogen sets ``thought=None``, the thought field
# never carries the reasoning, and our transformer above (which conditions on
# message.thought) emits no reasoning_content on the next-turn request.
# DeepSeek's backend then 400s with "reasoning_content must be passed back".
#
# Fix: monkey-patch the openai SDK's async chat completion call. After the raw
# response arrives, if a choice has both tool_calls AND non-empty
# reasoning_content AND empty content, copy reasoning_content into content.
# Autogen's existing tool_calls branch then captures it as thought, our
# message transformer emits it on next-turn requests, and the backend is
# happy. This is a no-op for OpenAI gpt-4o, Anthropic Claude, and any model
# that doesn't send reasoning_content alongside tool_calls.
# ---------------------------------------------------------------------------
_RESPONSE_PATCH_INSTALLED = False


def _install_response_lifter() -> None:
    global _RESPONSE_PATCH_INSTALLED
    if _RESPONSE_PATCH_INSTALLED:
        return
    try:
        import openai.resources.chat.completions as _oai_cc
        _AsyncCompletions = _oai_cc.AsyncCompletions
    except Exception:  # noqa: BLE001
        return

    _orig_create = _AsyncCompletions.create

    async def _patched_create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        response = await _orig_create(self, *args, **kwargs)
        # Only act on non-streamed ChatCompletion (has .choices). Streamed
        # responses are async iterables; a separate streaming patch would be
        # needed but autogen's tool_call path uses the non-streamed create().
        try:
            choices = getattr(response, "choices", None)
            if not choices:
                return response
            for choice in choices:
                msg = getattr(choice, "message", None)
                if msg is None:
                    continue
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls:
                    continue
                content = getattr(msg, "content", None)
                if content:
                    continue  # autogen's existing path will pick this up
                model_extra = getattr(msg, "model_extra", None)
                if not model_extra:
                    continue
                rc = model_extra.get("reasoning_content")
                if rc:
                    # pydantic v2 lets us assign on the model directly
                    msg.content = rc
        except Exception:  # noqa: BLE001
            pass  # best-effort patch; never break user requests
        return response

    _AsyncCompletions.create = _patched_create
    _RESPONSE_PATCH_INSTALLED = True


# Auto-install on import. Safe / idempotent — register_transformer is just a
# dict assignment, and the SDK monkey-patch is a single attribute swap guarded
# by an installed flag.
install_thinking_transformer()
_install_response_lifter()
