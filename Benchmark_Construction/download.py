import json
import os

import requests

release_id = "2023-10-31"
api_key = os.getenv("AI4SCHOLAR_API_KEY")

if not api_key:
    raise RuntimeError("请先设置环境变量 AI4SCHOLAR_API_KEY")

url = f"https://ai4scholar.net/datasets/v1/release/{release_id}"
headers = {
    "Authorization": f"Bearer {api_key}",
}

response = requests.get(url, headers=headers, timeout=30)

try:
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
except ValueError:
    print(response.text)
