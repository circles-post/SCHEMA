#!/usr/bin/env bash
# Fill in your extraction LLM credentials, then run:
#   source /mnt/shared-storage-user/ai4good2-share/fengxinshun/datasetsa/triple_extraction_env.sh

export OPENAI_API_KEY="sk-g6D6kNNW9eCNzucPWI7HwDaCKkToYx9ZQ422h7XP60qHIvyv"
export OPENAI_BASE_URL="https://chat.intern-ai.org.cn/api/v1/"
export OPENAI_MODEL="intern-s1-pro"

# Optional compatibility aliases used by the codebase.
export INTERN_API_KEY="$OPENAI_API_KEY"
export INTERN_BASE_URL="$OPENAI_BASE_URL"
