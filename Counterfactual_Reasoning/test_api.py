from openai import OpenAI

client = OpenAI(
    api_key="sk-g6D6kNNW9eCNzucPWI7HwDaCKkToYx9ZQ422h7XP60qHIvyv",  # 此处传token，不带Bearer
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

chat_rsp = client.chat.completions.create(
    model="intern-s1-pro",
    messages=[{"role": "user", "content": "hello"}],
)

for choice in chat_rsp.choices:
    print(choice.message.content)