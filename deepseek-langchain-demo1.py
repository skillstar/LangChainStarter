import os  
from langchain_openai import ChatOpenAI  
from langchain_core.messages import HumanMessage, SystemMessage  

# 配置 DeepSeek API  
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")  
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"  

def main():  
    try:  
        # 创建 ChatOpenAI 实例  
        chat = ChatOpenAI(  
            model="deepseek-chat",  # DeepSeek 模型  
            temperature=0.7,        # 控制生成文本的随机性  
            max_tokens=10         # 最大生成token数  
        )  

        # 定义消息  
        messages = [  
            SystemMessage(content="你是一个擅长解释复杂科技概念的专业助手"),  
            HumanMessage(content="详细解释量子计算的实际应用")  
        ]  

        # 调用 DeepSeek  
        response = chat.invoke(messages)  

        # 打印响应  
        print(response.content)  

    except Exception as e:  
        print(f"调用 DeepSeek API 时发生错误: {e}")  

if __name__ == "__main__":  
    main()  