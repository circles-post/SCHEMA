"""Convenience drivers that chain model-calling + scoring.

``score_one`` / ``score_many`` stay model-agnostic: they expect the
caller to have already produced ``model_answer`` strings. These drivers
wrap a concrete model (text LLM or VLM) call for the common case where
you just want to "evaluate model X on these samples".

Provider routing is done by model name — see ``evaluation.vlm``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .core import score_one
from .scorers import _normalize
from .types import EvalResult
from .vlm import run_vlm, route_info


logger = logging.getLogger("evaluation.drivers")


# ---------------------------------------------------------------------------
# Text LLM (non-VQA samples)
# ---------------------------------------------------------------------------
def _call_text_llm(
    prompt: str,
    model: str,
    *,
    api_keys: dict[str, str] | None = None,
    base_url: str | None = None,
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> str:
    """Route to the same provider family as the VLM router, but text-only.

    For Anthropic (claude) we use anthropic.messages; for everything
    else we go through OpenAI-compatible chat.completions.
    """
    info = route_info(model)
    provider = info["provider"]
    env_name = info["api_key_env"]
    api_key = ""
    if api_keys and env_name in api_keys and api_keys[env_name]:
        api_key = str(api_keys[env_name])
    else:
        api_key = os.environ.get(env_name, "")
    if not api_key:
        return f"__LLM_ERROR__:missing_api_key:{env_name}"
    url = base_url or info["base_url_default"]

    if provider == "anthropic":
        try:
            import anthropic  # type: ignore
        except ImportError:
            return "__LLM_ERROR__:anthropic sdk not installed"
        try:
            client = anthropic.Anthropic(api_key=api_key, base_url=url)
            rsp = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [b.text for b in rsp.content if getattr(b, "type", "") == "text"]
            return " ".join(parts).strip()
        except Exception as exc:
            return f"__LLM_ERROR__:{type(exc).__name__}:{exc}"

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return "__LLM_ERROR__:openai sdk not installed"
    try:
        client = OpenAI(api_key=api_key, base_url=url)
        rsp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (rsp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"__LLM_ERROR__:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Prompt assembly for non-VQA types
# ---------------------------------------------------------------------------
_LETTERS = "ABCDEFGHIJKL"


def _build_multichoice_prompt(sample: dict) -> str:
    opts = sample.get("options") or []
    lines = [sample.get("question", ""), "", "Options:"]
    for i, o in enumerate(opts):
        lines.append(f"  {_LETTERS[i]}. {o.get('text', '')}")
    lines.append("")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def _build_boolean_prompt(sample: dict) -> str:
    return f"{sample.get('question', '')}\n\nAnswer with Supported or Not supported."


def _build_essay_prompt(sample: dict) -> str:
    return f"{sample.get('question', '')}\n\nProvide a concise scientific answer (≤ 5 sentences)."


def _build_experiment_prompt(sample: dict) -> str:
    meta = sample.get("metadata") or {}
    return (
        sample.get("question", "")
        + "\n\n# Data setup (do not modify):\n"
        + str(meta.get("data_code", ""))
        + "\n\n# Fill in the TODOs to make the unit tests pass:\n"
        + str(meta.get("incomplete_main_code", ""))
        + "\n\nReturn ONLY the completed Python code for the main module."
    )


def _extract_letter(raw: str, options: list) -> int:
    m = re.search(r"[A-H]", (raw or "").upper())
    if m:
        idx = _LETTERS.index(m.group(0))
        if 0 <= idx < len(options):
            return idx
    return -1


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_and_score(
    sample: dict[str, Any],
    model: str,
    *,
    vlm_api_keys: dict[str, str] | None = None,
    vlm_base_url: str | None = None,
    text_api_keys: dict[str, str] | None = None,
    text_base_url: str | None = None,
    judge_model_config: dict[str, Any] | None = None,
    sandbox_timeout: float | None = None,
    pass_threshold: float = 0.5,
) -> EvalResult:
    """Run ``model`` on ``sample`` and score the response.

    * ``vqa`` samples → ``run_vlm`` (multimodal) with image + question_q
    * multichoice    → text LLM, pass letter prompt
    * boolean_support → text LLM, "Supported/Not supported" prompt
    * essay          → text LLM (free text), then LLM-as-judge scoring
    * experiment_code → text LLM with incomplete_main_code prompt, then sandbox

    ``judge_model_config`` is required for essay / open-ended vqa scoring.
    Keys fall back to env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).
    """
    qtype = sample.get("question_type", "")
    meta = sample.get("metadata") or {}

    if qtype == "vqa":
        question_q = str(meta.get("question_q") or sample.get("question", ""))
        image_path = str(meta.get("image_path", ""))
        if not image_path:
            return score_one(sample, "__VLM_ERROR__:no_image_path",
                             judge_model_config=judge_model_config,
                             sandbox_timeout=sandbox_timeout, pass_threshold=pass_threshold)
        suffix = ""
        if meta.get("vqa_format") == "yesno":
            suffix = " Answer Yes or No."
        answer = run_vlm(
            question=(question_q + suffix).strip(),
            image_path=image_path,
            model=model,
            api_keys=vlm_api_keys,
            base_url=vlm_base_url,
        )
        return score_one(sample, answer,
                         judge_model_config=judge_model_config,
                         sandbox_timeout=sandbox_timeout, pass_threshold=pass_threshold)

    # Text-only paths
    if qtype in {"claim_choice", "one_hop_tail", "two_hop_tail"}:
        prompt = _build_multichoice_prompt(sample)
        raw = _call_text_llm(prompt, model, api_keys=text_api_keys, base_url=text_base_url, max_tokens=16)
        idx = _extract_letter(raw, sample.get("options") or [])
        answer = idx if idx >= 0 else raw
    elif qtype == "boolean_support":
        answer = _call_text_llm(_build_boolean_prompt(sample), model,
                                api_keys=text_api_keys, base_url=text_base_url, max_tokens=16)
    elif qtype == "essay":
        answer = _call_text_llm(_build_essay_prompt(sample), model,
                                api_keys=text_api_keys, base_url=text_base_url, max_tokens=500)
    elif qtype == "experiment_code":
        answer = _call_text_llm(_build_experiment_prompt(sample), model,
                                api_keys=text_api_keys, base_url=text_base_url, max_tokens=2000)
    else:
        answer = ""

    return score_one(sample, answer,
                     judge_model_config=judge_model_config,
                     sandbox_timeout=sandbox_timeout, pass_threshold=pass_threshold)
