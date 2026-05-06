"""Quick smoke test: send a hello message to an OpenAI-compatible endpoint.

Set AGENTDEBUG_INTERN_API_KEY (and optionally AGENTDEBUG_INTERN_BASE_URL) in
your shell before running.
"""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("AGENTDEBUG_INTERN_API_KEY", ""),
    base_url=os.environ.get("AGENTDEBUG_INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/"),
)

chat_rsp = client.chat.completions.create(
    model="intern-s1-pro",
    messages=[{"role": "user", "content": "hello"}],
)

for choice in chat_rsp.choices:
    print(choice.message.content)
