#!/usr/bin/env bash
# Fill in your extraction LLM credentials, then `source` this file from the
# repo root, e.g.:
#   source ./triple_extraction_env.sh

export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://chat.intern-ai.org.cn/api/v1/"
export OPENAI_MODEL="intern-s1-pro"

# Optional compatibility aliases used by the codebase.
export INTERN_API_KEY="$OPENAI_API_KEY"
export INTERN_BASE_URL="$OPENAI_BASE_URL"
