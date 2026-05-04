"""Multi-provider VLM (Vision-Language Model) call router.

Public entry point::

    answer: str = run_vlm(
        question=sample['metadata']['question_q'],
        image_path=sample['metadata']['image_path'],
        model="gpt-4o",                     # any supported model name
        api_keys={                          # pass keys explicitly …
            "OPENAI_API_KEY":    "sk-…",
            "ANTHROPIC_API_KEY": "sk-…",
            "GOOGLE_API_KEY":    "…",
            "INTERN_API_KEY":    "…",
            "DASHSCOPE_API_KEY": "…",
        },
        base_url=None,                      # … or override the endpoint
    )

``model`` gets routed to a provider by a name-prefix regex (see
``_MODEL_ROUTES``). Each provider has a lightweight adapter; the
OpenAI-compatible adapter is the fallback for unknown / user-hosted
model names, and handles Gemini (via a Google-supplied OpenAI-compat
endpoint) and any vLLM/TGI deployment.

Image handling: the image file is read, base64-encoded, and inlined in
the request. For OpenAI/Gemini/Intern the ``image_url`` with
``data:image/...;base64,...`` convention is used. Anthropic needs an
explicit ``image`` content block with ``source.base64``.

Failure modes never raise — on any error we return the string
``"__VLM_ERROR__:<reason>"`` which the downstream scorer will treat as
a wrong answer.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger("evaluation.vlm")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
_MODEL_ROUTES: list[tuple[str, str]] = [
    # (regex, provider_id)
    (r"^(claude-|anthropic/|anthropic\.)",             "anthropic"),
    (r"^(gemini-|google/)",                            "openai_compat_google"),
    (r"^(gpt-4o|gpt-4-vision|gpt-4\.\d*-vision|o1|o3|chatgpt-4o)", "openai"),
    (r"^(intern-?vl|internvl|intern-s1-vision)",       "openai_compat_intern"),
    (r"^(qwen2?\.?5?-vl|qwen-vl|dashscope/)",          "openai_compat_dashscope"),
    (r".*",                                            "openai_compat_generic"),  # fallback
]


_PROVIDER_ENV_DEFAULTS: dict[str, dict[str, str]] = {
    "openai":                 {"api_key_env": "OPENAI_API_KEY",    "base_url": "https://api.openai.com/v1"},
    "anthropic":              {"api_key_env": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com/v1"},
    "openai_compat_google":   {"api_key_env": "GOOGLE_API_KEY",    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "openai_compat_intern":   {"api_key_env": "INTERN_API_KEY",    "base_url": "https://chat.intern-ai.org.cn/api/v1"},
    "openai_compat_dashscope":{"api_key_env": "DASHSCOPE_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    "openai_compat_generic":  {"api_key_env": "OPENAI_API_KEY",    "base_url": "https://api.openai.com/v1"},
}


def _route_provider(model: str) -> str:
    for pattern, provider in _MODEL_ROUTES:
        if re.match(pattern, model, flags=re.IGNORECASE):
            return provider
    return "openai_compat_generic"


def _resolve_key(provider: str, api_keys: dict[str, str] | None) -> str:
    env_name = _PROVIDER_ENV_DEFAULTS[provider]["api_key_env"]
    if api_keys and env_name in api_keys and api_keys[env_name]:
        return str(api_keys[env_name])
    return os.environ.get(env_name, "") or ""


def _resolve_base_url(provider: str, base_url_override: str | None) -> str:
    if base_url_override:
        return base_url_override
    return _PROVIDER_ENV_DEFAULTS[provider]["base_url"]


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
def _image_to_base64(image_path: str) -> tuple[str, str]:
    """Return ``(media_type, base64_data)``. Media type inferred from
    extension; defaults to image/jpeg.
    """
    p = Path(image_path)
    ext = p.suffix.lower().lstrip(".")
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(p, "rb") as f:
        raw = f.read()
    return media, base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Adapter: OpenAI / OpenAI-compatible (Gemini, Intern-VL, Qwen-VL, generic)
# ---------------------------------------------------------------------------
def _call_openai_compat(
    question: str,
    image_path: str,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return "__VLM_ERROR__:openai sdk not installed"
    try:
        media, b64 = _image_to_base64(image_path)
    except FileNotFoundError:
        return f"__VLM_ERROR__:image_not_found:{image_path}"
    except Exception as exc:
        return f"__VLM_ERROR__:image_read_error:{type(exc).__name__}:{exc}"
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
            ],
        },
    ]
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (rsp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"__VLM_ERROR__:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Adapter: Anthropic (Claude)
# ---------------------------------------------------------------------------
def _call_anthropic(
    question: str,
    image_path: str,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        import anthropic  # type: ignore
    except ImportError:
        return "__VLM_ERROR__:anthropic sdk not installed"
    try:
        media, b64 = _image_to_base64(image_path)
    except FileNotFoundError:
        return f"__VLM_ERROR__:image_not_found:{image_path}"
    except Exception as exc:
        return f"__VLM_ERROR__:image_read_error:{type(exc).__name__}:{exc}"
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    try:
        rsp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        # rsp.content is a list of blocks; concatenate any text blocks.
        parts = [b.text for b in rsp.content if getattr(b, "type", "") == "text"]
        return " ".join(p for p in parts if p).strip()
    except Exception as exc:
        return f"__VLM_ERROR__:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_vlm(
    question: str,
    image_path: str,
    model: str,
    *,
    api_keys: dict[str, str] | None = None,
    base_url: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Call the VLM of choice with ``(question, image)`` and return the
    model's text output.

    Returns a string prefixed with ``__VLM_ERROR__:`` on failure so the
    downstream scorer can treat the sample as wrong without special
    error plumbing.
    """
    provider = _route_provider(model)
    api_key = _resolve_key(provider, api_keys)
    url = _resolve_base_url(provider, base_url)

    if not api_key:
        return f"__VLM_ERROR__:missing_api_key:{_PROVIDER_ENV_DEFAULTS[provider]['api_key_env']}"

    if provider == "anthropic":
        return _call_anthropic(question, image_path, model, api_key, url, max_tokens, temperature)
    # All other providers go through the OpenAI-compatible adapter.
    return _call_openai_compat(question, image_path, model, api_key, url, max_tokens, temperature)


def route_info(model: str) -> dict[str, str]:
    """Introspection helper — returns the routing decision for a model
    name without actually calling anything. Useful for config/logging.
    """
    provider = _route_provider(model)
    return {
        "model": model,
        "provider": provider,
        "api_key_env": _PROVIDER_ENV_DEFAULTS[provider]["api_key_env"],
        "base_url_default": _PROVIDER_ENV_DEFAULTS[provider]["base_url"],
    }
