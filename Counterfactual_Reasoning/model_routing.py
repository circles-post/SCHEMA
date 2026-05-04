from __future__ import annotations

import os
from typing import Any, Mapping

INTERN_MODEL_NAMES = frozenset(
    {
        "intern-s1-pro",
        "intern-s1",
        "intern-s1-mini",
    }
)

DEFAULT_INTERN_BASE_URL = "https://chat.intern-ai.org.cn/api/v1/"


def normalize_model_name(model: str | None) -> str:
    return (model or "").strip().lower()


def is_intern_model(model: str | None) -> bool:
    return normalize_model_name(model) in INTERN_MODEL_NAMES


def get_intern_base_url() -> str:
    return os.environ.get("AGENTDEBUG_INTERN_BASE_URL", DEFAULT_INTERN_BASE_URL)


def resolve_base_url_for_model(
    model: str | None,
    default_base_url: str | None,
    *,
    intern_base_url: str | None = None,
) -> str | None:
    if is_intern_model(model):
        return intern_base_url or get_intern_base_url()
    return default_base_url


def resolve_value_for_model(
    model: str | None,
    default_value: str | None,
    *,
    intern_value: str | None = None,
) -> str | None:
    if is_intern_model(model):
        return intern_value if intern_value is not None else default_value
    return default_value


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "auto", "default", "none", "null"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "Invalid boolean value. Expected one of "
        "{true,false,1,0,yes,no,on,off,auto}, "
        f"got: {value!r}"
    )


def get_thinking_mode_from_env(env_name: str = "AGENTDEBUG_THINKING_MODE") -> bool | None:
    return parse_optional_bool(os.environ.get(env_name))


def build_openai_extra_body(*, env_name: str = "AGENTDEBUG_THINKING_MODE") -> dict[str, Any]:
    thinking_mode = get_thinking_mode_from_env(env_name)
    if thinking_mode is None:
        return {}
    return {"thinking_mode": thinking_mode}


def merge_openai_extra_create_args(
    extra_create_args: Mapping[str, Any] | None = None,
    *,
    env_name: str = "AGENTDEBUG_THINKING_MODE",
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(extra_create_args or {})
    default_extra_body = build_openai_extra_body(env_name=env_name)
    if not default_extra_body:
        return merged

    current_extra_body = merged.get("extra_body")
    if current_extra_body is None:
        merged["extra_body"] = default_extra_body
        return merged
    if not isinstance(current_extra_body, Mapping):
        raise ValueError("extra_body must be a mapping when provided.")

    merged["extra_body"] = {
        **default_extra_body,
        **dict(current_extra_body),
    }
    return merged
