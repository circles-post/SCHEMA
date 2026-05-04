"""Autogen agent team pointed at the Boyue OpenAI-compatible endpoint.

Mirrors the structure of ``test_agent_debug.get_agent_team`` but without MCP /
ToolUniverse / sciverse — this module is only responsible for producing a
textual answer for a single prompt, which the runner then parses and hands to
``evaluation.score_many``.

The returned team is a single-agent ``RoundRobinGroupChat`` whose agent is
instructed to emit ``<answer>...</answer>`` followed by ``TERMINATE``. The
runner extracts the contents of the last ``<answer>`` tag.
"""

from __future__ import annotations

from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


_SYSTEM_MESSAGE = """\
You are a scientific-evaluation assistant. For each user turn you will receive
ONE question plus, depending on the question type, a list of options or a
reference claim to judge. Read the question carefully and answer using the
format described below.

## Answer format by question type

- **multichoice** (claim_choice / one_hop_tail / two_hop_tail)
  The user turn presents options labelled A, B, C, ... Output the SINGLE
  option letter inside `<answer>` tags. Example: `<answer>A</answer>`.
  Do NOT include the option text.

- **boolean_support**
  Decide whether the claim is supported by the provided evidence. Output
  exactly `<answer>Supported</answer>` or `<answer>Not Supported</answer>`.

- **essay**
  Output a concise free-form answer (at most a few sentences) inside
  `<answer>` tags. The grader judges scientific content overlap, so be
  specific and factual; do not pad with filler.

- **experiment_code**
  Output the completed Python main_code (no markdown fences, no commentary)
  inside `<answer>` tags.

## Output behaviour

1. Briefly reason step-by-step above the answer tag when it helps — keep it
   short and decision-oriented.
2. Use EXACTLY ONE `<answer>` tag per response, containing the final answer
   in the format specified above for the question's type.
3. On the line AFTER the answer tag, output the token `TERMINATE` by itself.
   Without this the conversation will not end.

Example (multichoice)::

    Option A aligns with the cited evidence, options B-D contradict it.

    <answer>A</answer>
    TERMINATE
"""


@dataclass
class BoyueModelConfig:
    """OpenAI-compatible model config for a single Boyue model."""

    model: str
    base_url: str
    api_key: str
    timeout: float = 300.0
    max_retries: int = 2


def create_model_client(cfg: BoyueModelConfig) -> OpenAIChatCompletionClient:
    """Build an OpenAIChatCompletionClient for a Boyue model.

    ``model_info`` is required by autogen when the model name is not a known
    OpenAI model (Boyue hosts many custom models). We declare the minimal
    capabilities — text in, text out — because the evaluation workflow does
    not use tools, vision, or structured output.
    """
    return OpenAIChatCompletionClient(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )


def build_agent_team(
    cfg: BoyueModelConfig,
    *,
    max_messages: int = 6,
) -> RoundRobinGroupChat:
    """Single-agent RoundRobinGroupChat that terminates on TERMINATE."""
    model_client = create_model_client(cfg)
    agent = AssistantAgent(
        name="EvalAgent",
        model_client=model_client,
        description=f"Evaluation agent backed by {cfg.model}.",
        system_message=_SYSTEM_MESSAGE,
    )
    termination = TextMentionTermination("TERMINATE", sources=["EvalAgent"]) | MaxMessageTermination(max_messages)
    return RoundRobinGroupChat([agent], termination_condition=termination)
