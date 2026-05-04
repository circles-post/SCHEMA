from __future__ import annotations

DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MIN_SUPPORT = 2
DEFAULT_MIN_DOUBLE_CHECK_SUPPORT = 2
DEFAULT_MAX_SAMPLES = 100
DEFAULT_MAX_PER_UNIQUENESS_KEY = 1
DEFAULT_DISTRACTOR_COUNT = 3
DEFAULT_RANDOM_SEED = 7
DEFAULT_SUPPORTED_QUESTION_TYPES = (
    "claim_choice",
    "boolean_support",
    "two_hop_tail",
    "essay",
    "experiment_code",
    "vqa",
)
DEFAULT_VALIDATION_MODE = "rule_only"
DEFAULT_RETRIEVAL_TOP_K = 3
DEFAULT_GITHUB_SEARCH_LANGUAGE = "Python"
DEFAULT_GITHUB_SEARCH_PER_PAGE = 3
DEFAULT_VALIDATOR_MODEL_CONFIG = {
    "enabled": False,
    "model": "",
    "base_url": "",
    "api_key": "",
    "temperature": 0.0,
    "max_tokens": 800,
}
DEFAULT_CORROBORATION_MODE = "off"              # "off" | "required"
DEFAULT_MIN_EXTERNAL_SOURCES = 1
DEFAULT_CORROBORATION_TOOL_TIMEOUT = 60.0
DEFAULT_MIN_LOCAL_SUPPORT = 2                   # mirrors sampler DEFAULT_MIN_SUPPORT
DEFAULT_COVERAGE_MODE = "off"                   # "off" | "greedy" (requires --graph)
DEFAULT_RATIO_POOL_MULTIPLIER = 10              # how many× max_samples to over-sample per type
DEFAULT_NODE_QUOTA = {"T1": 3, "T2": 2, "T3": 1}  # node-based allocator: per-tier sample count
