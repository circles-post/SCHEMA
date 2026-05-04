"""Client-side RPM / TPM rate limiting + retry for Intern LLM calls.

Motivation: the Intern endpoint caps the account at 100 RPM / 50 000 TPM.
Exceeding either triggers a HTTP 400 with Chinese body `请求过于频繁,
请稍后再试` (code -20048) — openai-python does NOT treat that as a 429,
so the SDK's built-in `max_retries` does not help. Earlier full-bench
runs hit this on the very first LLM call of each worker.

This module provides:
  * ``SlidingWindowLimiter`` — thread-safe 60-second sliding window that
    enforces both request-count and token-count budgets.  Used sync by
    ``LLMPlanner`` and async by ``OpenAICompatibleLLM``.
  * ``with_retry_async`` / ``with_retry_sync`` — exponential-backoff +
    jitter retry that specifically catches ``RateLimitError``, HTTP 429
    bodies, and the Intern "请求过于频繁" / `-20048` error so none of
    the LLM calls hard-fail on a transient limit.

Scope:
  Every worker process creates its own limiter (process-local). When
  running multiple workers in parallel you MUST divide the account
  budget across workers — e.g. 4 workers × 25 RPM = 100 RPM total.
  Set ``AGDEBUGGER_LLM_RPM`` / ``AGDEBUGGER_LLM_TPM`` to the
  **per-worker** slice explicitly.

  Note: this limiter covers the ``external_agent`` pipeline only
  (claim extractor, judge, planner, rewriter). The ToolUniverse MCP
  server drives its own LLM calls for the agent-under-debug and is
  NOT governed here — its concurrency is controlled via ToolUniverse's
  own ``max_workers`` knob.
"""
from __future__ import annotations

import asyncio
import os
import random
import threading
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Optional, Tuple, TypeVar


_T = TypeVar("_T")


class SlidingWindowLimiter:
    """60-second sliding window that enforces RPM + TPM jointly.

    Caller supplies a token estimate before the request; the limiter
    records the request timestamp + estimate, then on completion the
    caller may call :meth:`record_actual` to replace the estimate with
    the real usage count so the next window reflects actual spend.
    """

    def __init__(
        self,
        rpm: int,
        tpm: int,
        window_sec: float = 60.0,
        *,
        name: str = "",
        disabled: bool = False,
    ) -> None:
        self.rpm = max(1, int(rpm))
        self.tpm = max(1, int(tpm))
        self.window = float(window_sec)
        self.name = name
        # When True, acquire_sync/acquire_async return immediately without
        # tracking any events. Use when the ONLY rate-limiting signal should
        # come from the upstream 429/`请求过于频繁` responses (handled by
        # ``with_retry_sync`` / ``with_retry_async``). Set via the env var
        # ``AGDEBUGGER_LLM_RATE_LIMITER_DISABLED=1``.
        self.disabled = bool(disabled)
        self._lock = threading.Lock()
        self._events: Deque[Tuple[float, int]] = deque()

    def _purge(self, now: float) -> None:
        cutoff = now - self.window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _wait_needed_locked(self, tokens: int, now: float) -> float:
        """Return how many seconds to sleep before a request of ``tokens``
        tokens can proceed. Caller must hold ``self._lock``."""
        self._purge(now)
        used_req = len(self._events)
        used_tok = sum(t for _, t in self._events)
        rpm_exceeds = used_req >= self.rpm
        tpm_exceeds = used_tok + max(0, tokens) > self.tpm
        if not (rpm_exceeds or tpm_exceeds):
            return 0.0
        if not self._events:
            return 0.0
        oldest_ts = self._events[0][0]
        # Add a small safety margin so we don't race the window edge.
        return max(0.0, (oldest_ts + self.window) - now + 0.05)

    def acquire_sync(self, tokens: int) -> None:
        if self.disabled:
            return
        while True:
            with self._lock:
                wait = self._wait_needed_locked(tokens, time.monotonic())
                if wait <= 0:
                    self._events.append((time.monotonic(), max(0, int(tokens))))
                    return
            # Release the lock before sleeping so concurrent callers can
            # re-evaluate; they'll re-check inside _wait_needed_locked.
            time.sleep(wait)

    async def acquire_async(self, tokens: int) -> None:
        if self.disabled:
            return
        while True:
            with self._lock:
                wait = self._wait_needed_locked(tokens, time.monotonic())
                if wait <= 0:
                    self._events.append((time.monotonic(), max(0, int(tokens))))
                    return
            await asyncio.sleep(wait)

    def record_actual(self, tokens: int) -> None:
        """Replace the most recent event's token estimate with the actual
        usage reported by the LLM provider. No-op if no events recorded."""
        if tokens is None:
            return
        with self._lock:
            if self._events:
                ts, _ = self._events[-1]
                self._events[-1] = (ts, max(0, int(tokens)))

    def snapshot(self) -> Tuple[int, int]:
        """(current_request_count, current_token_count) inside the window —
        mostly for diagnostics / tests."""
        with self._lock:
            self._purge(time.monotonic())
            return len(self._events), sum(t for _, t in self._events)


_LIMITER_LOCK = threading.Lock()
_SHARED_LIMITER: Optional[SlidingWindowLimiter] = None


def get_shared_limiter() -> SlidingWindowLimiter:
    """Return the process-wide LLM rate limiter, reading budgets from
    ``AGDEBUGGER_LLM_RPM`` / ``AGDEBUGGER_LLM_TPM`` env vars the first
    time it is constructed. Defaults: 100 RPM / 50_000 TPM (full
    single-worker Intern budget)."""
    global _SHARED_LIMITER
    if _SHARED_LIMITER is not None:
        return _SHARED_LIMITER
    with _LIMITER_LOCK:
        if _SHARED_LIMITER is not None:
            return _SHARED_LIMITER
        try:
            rpm = int(os.environ.get("AGDEBUGGER_LLM_RPM", "100"))
        except ValueError:
            rpm = 100
        try:
            tpm = int(os.environ.get("AGDEBUGGER_LLM_TPM", "50000"))
        except ValueError:
            tpm = 50000
        disabled = os.environ.get("AGDEBUGGER_LLM_RATE_LIMITER_DISABLED", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        _SHARED_LIMITER = SlidingWindowLimiter(rpm, tpm, name="intern", disabled=disabled)
    return _SHARED_LIMITER


def _reset_shared_limiter_for_tests() -> None:
    """Test hook — resets the module-global singleton so monkeypatched
    env vars take effect on the next ``get_shared_limiter`` call."""
    global _SHARED_LIMITER
    with _LIMITER_LOCK:
        _SHARED_LIMITER = None


# ---------------------------------------------------------------------------
# Rate-limit-aware retry wrappers
# ---------------------------------------------------------------------------


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if ``exc`` was caused by an upstream rate-limit response.

    Catches:
      * ``openai.RateLimitError`` (standard 429 path)
      * HTTP 400 with body containing ``请求过于频繁`` or ``-20048``
        (the Intern-specific over-quota response that openai-python does
        NOT treat as a 429, so the SDK's built-in retry won't help).
      * Generic error messages like ``rate limit`` / ``too many requests``.
    """
    try:
        import openai  # type: ignore
        if isinstance(exc, openai.RateLimitError):  # pragma: no cover - optional
            return True
    except Exception:
        pass
    message = str(exc).lower()
    if "请求过于频繁" in str(exc):
        return True
    if "-20048" in message:
        return True
    if "rate limit" in message or "too many requests" in message:
        return True
    return False


def _parse_retry_after(exc: BaseException) -> Optional[float]:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        raw = None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _retry_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    backoff = base_delay * (2 ** (attempt - 1))
    jitter = random.uniform(0, min(2.0, base_delay))
    return min(max_delay, backoff) + jitter


def _retry_env_knobs() -> Tuple[int, float, float]:
    try:
        max_attempts = int(os.environ.get("AGDEBUGGER_LLM_RETRY_MAX_ATTEMPTS", "6"))
    except ValueError:
        max_attempts = 6
    try:
        base_delay = float(os.environ.get("AGDEBUGGER_LLM_RETRY_BASE_DELAY_SEC", "2"))
    except ValueError:
        base_delay = 2.0
    try:
        max_delay = float(os.environ.get("AGDEBUGGER_LLM_RETRY_MAX_DELAY_SEC", "120"))
    except ValueError:
        max_delay = 120.0
    return max_attempts, base_delay, max_delay


def with_retry_sync(func: Callable[[], _T], *, label: str = "llm") -> _T:
    max_attempts, base_delay, max_delay = _retry_env_knobs()
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = _parse_retry_after(exc) or _retry_delay(attempt, base_delay, max_delay)
            print(
                f"[{label}] rate-limited ({type(exc).__name__}); "
                f"retry {attempt}/{max_attempts - 1} in {delay:.1f}s"
            )
            time.sleep(delay)


async def with_retry_async(
    func: Callable[[], Awaitable[_T]], *, label: str = "llm"
) -> _T:
    max_attempts, base_delay, max_delay = _retry_env_knobs()
    attempt = 0
    while True:
        try:
            return await func()
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = _parse_retry_after(exc) or _retry_delay(attempt, base_delay, max_delay)
            print(
                f"[{label}] rate-limited ({type(exc).__name__}); "
                f"retry {attempt}/{max_attempts - 1} in {delay:.1f}s"
            )
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Token estimator (rough char-based fallback)
# ---------------------------------------------------------------------------

_CHAR_PER_TOKEN = 4.0


def estimate_tokens_from_text(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return max(1, int(len(text) / _CHAR_PER_TOKEN))


def estimate_request_tokens(*, system_prompt: str, user_prompt: str, output_budget: int = 600) -> int:
    """Estimate total tokens a chat-completion request will consume.

    Intentionally over-estimates slightly so the limiter leaves headroom
    when the real response is larger than the char-based heuristic."""
    prompt_tokens = estimate_tokens_from_text(system_prompt) + estimate_tokens_from_text(user_prompt)
    return prompt_tokens + max(0, int(output_budget))


def response_total_tokens(response: Any) -> Optional[int]:
    """Pull the real total token count from an OpenAI-style chat response
    when the provider returns a ``usage`` block; otherwise ``None``."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    total = getattr(usage, "total_tokens", None)
    if total is None and isinstance(usage, dict):
        total = usage.get("total_tokens")
    if total is None:
        prompt = getattr(usage, "prompt_tokens", None) or (
            usage.get("prompt_tokens") if isinstance(usage, dict) else None
        )
        completion = getattr(usage, "completion_tokens", None) or (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        if prompt is not None or completion is not None:
            total = (prompt or 0) + (completion or 0)
    try:
        return int(total) if total is not None else None
    except (TypeError, ValueError):
        return None
