from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from json import JSONDecodeError

from openai import OpenAI

from .utils import normalize_text

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class InternChatClient:
    THINKING_MODELS = {"intern-s1-pro", "intern-s1", "intern-s1-mini"}

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.api_key = (
            config.get("api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("INTERN_API_KEY")
        )
        self.base_url = (
            config.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("INTERN_BASE_URL")
            or "https://chat.intern-ai.org.cn/api/v1/"
        )
        self.model = config.get("model") or os.getenv("OPENAI_MODEL") or "intern-s1-pro"
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 1200))
        self.thinking_mode = bool(config.get("thinking_mode", False))
        self.max_retries = max(int(config.get("max_retries", 5)), 0)
        self.retry_sleep_seconds = max(float(config.get("retry_sleep_seconds", 5.0)), 0.0)
        self.retry_backoff = max(float(config.get("retry_backoff", 2.0)), 1.0)
        if not self.api_key:
            raise ValueError("Missing API key for Intern/OpenAI-compatible client")
        self.request_timeout = float(config.get("request_timeout", 60.0))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )

    def _build_request_kwargs(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        chosen_model = model or self.model
        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if chosen_model in self.THINKING_MODELS:
            kwargs["extra_body"] = {"thinking_mode": self.thinking_mode}
        return kwargs

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        content = message.content or ""
        if content and str(content).strip():
            return str(content)
        reasoning = getattr(message, "reasoning_content", None) or ""
        return str(reasoning)

    @staticmethod
    def _extract_json_text(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        match = JSON_BLOCK_RE.search(stripped)
        if match:
            return match.group(1).strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        start_list = stripped.find("[")
        start_obj = stripped.find("{")
        starts = [pos for pos in [start_list, start_obj] if pos != -1]
        if starts:
            start = min(starts)
            candidate = stripped[start:].strip()
            end_list = candidate.rfind("]")
            end_obj = candidate.rfind("}")
            end = max(end_list, end_obj)
            if end != -1:
                return candidate[: end + 1]
        return stripped

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        kwargs = self._build_request_kwargs(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        delay = self.retry_sleep_seconds
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable_rate_limit_error(exc):
                    raise
                time.sleep(delay)
                delay *= self.retry_backoff
        raise RuntimeError("Unreachable retry loop in create_chat_completion")

    @staticmethod
    def _is_retryable_rate_limit_error(exc: Exception) -> bool:
        message = str(exc).lower()
        retry_markers = (
            "rate limit",
            "too many requests",
            "429",
            "tokens",
            "限流",
            "稍后再试",
            "-20053",
        )
        return any(marker in message for marker in retry_markers)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        rsp = self.create_chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return normalize_text(self._extract_message_text(rsp.choices[0].message))

    def _repair_json_via_model(
        self,
        raw_text: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        repair_messages = [
            {
                "role": "system",
                "content": "Repair malformed JSON. Return valid JSON only. Preserve the original structure and content as much as possible.",
            },
            {"role": "user", "content": raw_text[:12000]},
        ]
        rsp = self.create_chat_completion(
            repair_messages,
            model=model,
            temperature=0,
            max_tokens=max_tokens or min(self.max_tokens, 1200),
        )
        repaired_text = self._extract_json_text(self._extract_message_text(rsp.choices[0].message))
        if not repaired_text:
            raise ValueError("Empty repaired JSON text from Intern API")
        return json.loads(repaired_text)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        rsp = self.create_chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        message = rsp.choices[0].message
        raw_text = self._extract_message_text(message)
        json_text = self._extract_json_text(raw_text)
        if not json_text:
            raise ValueError("Empty response text from Intern API")
        try:
            return json.loads(json_text)
        except JSONDecodeError:
            return self._repair_json_via_model(json_text, model=model, max_tokens=max_tokens)
