"""Base types shared across the websearch package."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, ClassVar, Dict, List

Message = dict[str, Any]
MessageList = list[Message]


@dataclass
class SamplerResponse:
    """Response from a sampler (LLM call)."""
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    })


class SamplerBase:
    """Abstract base class for any LLM sampler used by the search planner."""

    async def __call__(self, message_list: MessageList) -> SamplerResponse:
        raise NotImplementedError


@dataclass
class UsageStats:
    """Accumulate token / request usage across multiple LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    _display_names: ClassVar[Dict[str, str]] = {}

    def accumulate(self, other: Dict[str, Any] | "UsageStats") -> None:
        if isinstance(other, UsageStats):
            other = asdict(other)
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            setattr(self, f.name, getattr(self, f.name) + other.get(f.name, 0))

    def to_dict(self) -> Dict[str, int]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}
