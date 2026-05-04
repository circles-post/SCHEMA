from __future__ import annotations

import os
from typing import Any

from external_agent.json_utils import parse_json_response
from external_agent.rate_limiter import (
    estimate_request_tokens,
    get_shared_limiter,
    response_total_tokens,
    with_retry_async,
)
from model_routing import build_openai_extra_body


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_sec: float | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required to use external_agent.llm. "
                "Install project dependencies before running the claim/judge pipeline."
            ) from exc

        self.model = model
        self.temperature = temperature
        effective_timeout_sec = (
            float(timeout_sec)
            if timeout_sec is not None
            else float(os.environ.get("AGENTDEBUG_MODEL_TIMEOUT_SEC", "600"))
        )
        effective_max_retries = int(os.environ.get("AGENTDEBUG_MODEL_MAX_RETRIES", "2"))
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("AGENTDEBUG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("AGENTDEBUG_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
            timeout=effective_timeout_sec,
            max_retries=effective_max_retries,
        )
        # Wire AGENTDEBUG_THINKING_MODE so the external-agent LLM calls pick up
        # extra_body={"thinking_mode": ...}. Previously this was hard-coded to
        # None, silently dropping the env var on the entire claim/judge path.
        self._extra_body = build_openai_extra_body() or None
        # Shared process-wide RPM/TPM limiter. Earlier full-bench runs
        # against Intern tripped HTTP 400 "请求过于频繁" responses that
        # the openai SDK does NOT classify as 429 — its built-in
        # max_retries therefore did nothing. The limiter holds each call
        # until quota is available; ``with_retry_async`` catches any
        # over-the-line request that still slips through.
        self._limiter = get_shared_limiter()

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        estimate = estimate_request_tokens(system_prompt=system_prompt, user_prompt=user_prompt)
        await self._limiter.acquire_async(estimate)

        async def _call() -> str:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **({"extra_body": self._extra_body} if self._extra_body else {}),
            )
            actual = response_total_tokens(response)
            if actual is not None:
                self._limiter.record_actual(actual)
            return response.choices[0].message.content or ""

        return await with_retry_async(_call, label=f"llm({self.model})")

    async def complete_json(self, system_prompt: str, user_prompt: str) -> Any:
        response_text = await self.complete_text(system_prompt, user_prompt)
        try:
            return parse_json_response(response_text)
        except Exception:
            repair_prompt = (
                "Your previous response was not valid JSON.\n"
                "Return ONLY valid JSON matching the required schema.\n"
                "Do not add markdown, XML tags, prose, or explanations.\n\n"
                f"Original user prompt:\n{user_prompt}\n\n"
                f"Previous invalid response:\n{response_text}"
            )
            repaired_text = await self.complete_text(system_prompt, repair_prompt)
            return parse_json_response(repaired_text)

    async def close(self) -> None:
        await self.client.close()
