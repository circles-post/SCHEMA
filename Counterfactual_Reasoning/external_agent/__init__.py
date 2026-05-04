"""Claim extraction and judging framework for AGDebugger external agents."""

from external_agent.agdebugger_adapter import load_turns_from_agdebugger_state
from external_agent.claim_extractor import ClaimExtractor
from external_agent.evidence_provider import WebSearchEvidenceProvider
from external_agent.integration import analyze_session_state, analyze_text
from external_agent.judge import ClaimJudge
from external_agent.llm import OpenAICompatibleLLM
from external_agent.pipeline import ClaimJudgePipeline
from external_agent.schemas import Claim, ConversationTurn, EvidenceBundle, Judgment, PipelineResult
from external_agent.strategies import get_strategy

__all__ = [
    "Claim",
    "ClaimExtractor",
    "ClaimJudge",
    "ClaimJudgePipeline",
    "ConversationTurn",
    "EvidenceBundle",
    "Judgment",
    "OpenAICompatibleLLM",
    "PipelineResult",
    "WebSearchEvidenceProvider",
    "analyze_session_state",
    "analyze_text",
    "get_strategy",
    "load_turns_from_agdebugger_state",
]
