import json
from dataclasses import dataclass
from typing import Any, Dict, List

from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    HandoffMessage,
    MemoryQueryEvent,
    MessageFactory,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    UserInputRequestedEvent,
)
from autogen_agentchat.teams._group_chat._events import (
    GroupChatAgentResponse,
    GroupChatMessage,
    GroupChatRequestPublish,
    GroupChatReset,
    GroupChatStart,
    GroupChatTermination,
)
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

__message_factory = MessageFactory()


@dataclass
class FieldInfo:
    name: str
    type: str
    required: bool


@dataclass
class MessageTypeDescription:
    name: str
    fields: List[FieldInfo] | None = None


def get_message_type_descriptions() -> Dict[str, MessageTypeDescription]:
    """
    Gets the message type descriptions for user-sendable messages for agentchat:
    - TextMessage, MultiModalMessage, StopMessage, HandoffMessage
    """

    return {
        # "TextMessage": MessageTypeDescription(
        #     name="TextMessage",
        #     fields=[
        #         FieldInfo(name="source", type="str", required=True),
        #         FieldInfo(name="content", type="str", required=True),
        #         FieldInfo(name="type", type="str", required=True),
        #     ],
        # ),
        # "MultiModalMessage": MessageTypeDescription(
        #     name="MultiModalMessage",
        #     fields=[
        #         FieldInfo(name="source", type="str", required=True),
        #         FieldInfo(name="content", type="List[str]", required=True),
        #         FieldInfo(name="type", type="str", required=True),
        #     ],
        # ),
        # "StopMessage": MessageTypeDescription(
        #     name="StopMessage",
        #     fields=[
        #         FieldInfo(name="source", type="str", required=True),
        #         FieldInfo(name="content", type="str", required=True),
        #         FieldInfo(name="type", type="str", required=True),
        #     ],
        # ),
        # "HandoffMessage": MessageTypeDescription(
        #     name="HandoffMessage",
        #     fields=[
        #         FieldInfo(name="source", type="str", required=True),
        #         FieldInfo(name="content", type="str", required=True),
        #         FieldInfo(name="target", type="str", required=True),
        #         FieldInfo(name="context", type="List[LLMMessage]", required=False),
        #         FieldInfo(name="type", type="str", required=True),
        #     ],
        # ),
        "GroupChatStart": MessageTypeDescription(
            name="GroupChatStart",
            fields=[
                FieldInfo(name="messages", type="List[ChatMessage]", required=False),
            ],
        ),
        "GroupChatAgentResponse": MessageTypeDescription(
            name="GroupChatAgentResponse",
            fields=[
                FieldInfo(name="agent_response", type="Response", required=True),
            ],
        ),
        "GroupChatRequestPublish": MessageTypeDescription(
            name="GroupChatRequestPublish",
            fields=None,
        ),
        "GroupChatMessage": MessageTypeDescription(
            name="GroupChatMessage",
            fields=[
                FieldInfo(name="message", type="ChatMessage", required=True),
            ],
        ),
        "GroupChatTermination": MessageTypeDescription(
            name="GroupChatTermination",
            fields=[
                FieldInfo(name="message", type="StopMessage", required=True),
            ],
        ),
        "GroupChatReset": MessageTypeDescription(
            name="GroupChatReset",
            fields=None,
        ),
    }


# ### Serialization ### -- maybe should be a class?

__message_map = {
    # agentchat messages
    "TextMessage": TextMessage,
    "MultiModalMessage": MultiModalMessage,
    "StopMessage": StopMessage,
    "HandoffMessage": HandoffMessage,
    # agentchat events
    "ToolCallRequestEvent": ToolCallRequestEvent,
    "ToolCallExecutionEvent": ToolCallExecutionEvent,
    "ToolCallSummaryMessage": ToolCallSummaryMessage,
    "UserInputRequestedEvent": UserInputRequestedEvent,
    "MemoryQueryEvent": MemoryQueryEvent,
    # group chat messages
    "GroupChatAgentResponse": GroupChatAgentResponse,
    "GroupChatMessage": GroupChatMessage,
    "GroupChatRequestPublish": GroupChatRequestPublish,
    "GroupChatReset": GroupChatReset,
    "GroupChatStart": GroupChatStart,
    "GroupChatTermination": GroupChatTermination,
    # core messages
    "AssistantMessage": AssistantMessage,
    "FunctionExecutionResult": FunctionExecutionResult,
    "FunctionExecutionResultMessage": FunctionExecutionResultMessage,
    "SystemMessage": SystemMessage,
    "UserMessage": UserMessage,
}


def serialize(message: ChatMessage | AgentEvent | LLMMessage | None) -> dict:
    try:
        if message is None:
            return {"type": "None"}

        serialized_message = message.model_dump(mode="json")

        # get name in case doesnt exist
        type_name = type(message).__name__
        serialized_message["type"] = type_name
        return serialized_message
    except Exception:
        print("[WARN] Unable to serialize message: ", message)
        return {}


def _deserialize_message_like(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    message_type = message.get("type")
    if message_type in __message_factory._message_types:
        return __message_factory.create(message)

    return message


def _deserialize_response(response: Any) -> Any:
    if not isinstance(response, dict):
        return response

    response_dict = dict(response)
    chat_message = response_dict.get("chat_message")
    if chat_message is not None:
        response_dict["chat_message"] = _deserialize_message_like(chat_message)

    inner_messages = response_dict.get("inner_messages")
    if inner_messages is not None:
        response_dict["inner_messages"] = [_deserialize_message_like(message) for message in inner_messages]

    if hasattr(Response, "model_validate"):
        return Response.model_validate(response_dict)
    return Response(**response_dict)


def _deserialize_group_chat_message(message_type: str, message_dict: Dict[str, Any]) -> Any:
    payload = dict(message_dict)

    if message_type == "GroupChatStart":
        messages = payload.get("messages")
        if messages is not None:
            payload["messages"] = [_deserialize_message_like(message) for message in messages]
        return GroupChatStart.model_validate(payload)

    if message_type == "GroupChatMessage":
        payload["message"] = _deserialize_message_like(payload.get("message"))
        return GroupChatMessage.model_validate(payload)

    if message_type == "GroupChatTermination":
        payload["message"] = _deserialize_message_like(payload.get("message"))
        return GroupChatTermination.model_validate(payload)

    if message_type == "GroupChatAgentResponse":
        response_fields = GroupChatAgentResponse.model_fields
        response_key = "response" if "response" in response_fields else "agent_response"
        legacy_response_key = "agent_response" if response_key == "response" else "response"

        if response_key not in payload and legacy_response_key in payload:
            payload[response_key] = payload.pop(legacy_response_key)

        if response_key in payload:
            payload[response_key] = _deserialize_response(payload[response_key])

        if "name" in response_fields and "name" not in payload:
            response = payload.get(response_key)
            payload["name"] = getattr(getattr(response, "chat_message", None), "source", "")

        return GroupChatAgentResponse.model_validate(payload)

    new_message_class = __message_map[message_type]
    return new_message_class.model_validate(payload)


def deserialize(
    message_dict: Dict | str,
) -> ChatMessage | AgentEvent | LLMMessage | None:
    try:
        if isinstance(message_dict, str):
            message_dict = json.loads(message_dict)

        message_type = message_dict["type"]  # type: ignore

        if message_type == "None":
            return None

        new_message = _deserialize_group_chat_message(message_type, message_dict)
        return new_message
    except Exception as e:
        print(
            f"[WARN] Unable to deserialize message dict into Pydantic class. Error: {str(e)}.\nMessage dict: ",
            message_dict,
        )
        return None
