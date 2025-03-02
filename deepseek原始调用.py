import os  
# 直接使用 OpenAI 的库  
from openai import OpenAI 

# 配置 DeepSeek  
deep_api_key = os.getenv("DEEPSEEK_API_KEY")  



client = OpenAI(  
    api_key=deep_api_key,  
    base_url="https://api.deepseek.com/v1"  # DeepSeek 的 API 地址  
)  

# 多轮对话示例  
messages = [  
    {"role": "system", "content": "你是一个擅长解释复杂科技概念的助手"},  
    {"role": "user", "content": "量子计算"},  
    {"role": "assistant", "content": "...(上一个响应)"},  
    {"role": "user", "content": "能详细解释量子计算的实际应用吗？"}  
]  

response = client.chat.completions.create(  
    model="deepseek-chat",  
    messages=messages  
) 

print(response.choices[0].message.content)  