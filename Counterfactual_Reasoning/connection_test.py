import requests

def test_openai_api(api_url, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用OpenAI的一个基本API请求（比如chat GPT-3）
    data = {
        "model": "gpt-3.5-turbo", 
        "messages": [{"role": "system", "content": "You are a helpful assistant."}]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        
        # 判断请求是否成功
        if response.status_code == 200:
            print("API Key and URL are valid. Response:")
            print(response.json())
        else:
            print(f"Failed to connect. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# 输入你的 OpenAI API URL 和 Key
api_url = "http://34.13.73.248:3888/v1/chat/completions"  # 根据实际URL修改
api_key = "sk-ZnvhxhwyXok91ezpbDBcObLWa8GehlZtMaqnYT3ziVwhnBzC"

# 调用测试函数
test_openai_api(api_url, api_key)