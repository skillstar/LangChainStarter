# 导入必要的模块  
import os  
from langchain_openai import ChatOpenAI  # 导入 ChatOpenAI 模型  
from langchain_core.prompts import ChatPromptTemplate  # 导入聊天提示模板  
from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器  

# 配置 DeepSeek API 的环境变量  
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")  
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"  

def main():  
    try:  
        # 1. 创建大语言模型实例   
        # 就像选择一个超级聪明的助手，设置它的"创造力"和"话痨程度"  
        model = ChatOpenAI(  
            model="deepseek-chat",      # 选择 DeepSeek 模型  
            temperature=0.7,            # 控制回答的随机性和创造性  
                                        # 0 = 非常严谨和确定  
                                        # 1 = 非常自由和发散  
            max_tokens=25             # 限制回答的最大长度  
        )  

        # 2. 创建提示模板   
        # 就像给助手准备一个问题的标准模板  
        prompt = ChatPromptTemplate.from_template(  
            "你是一个擅长解释复杂科技概念的专业助手。请用幽默的风格解释{topic}的实际应用和最新发展。"  
        )  

        # 3. 创建输出解析器  
        # 负责把模型的复杂输出转换成简单的文本  
        output_parser = StrOutputParser()  

        # 4. 构建处理链   
        # 把模型、提示和解析器串联起来，形成一个完整的对话流程  
        # 就像搭建一条从"问题"到"答案"的流水线  
        chain = prompt | model | output_parser  

        # 5. 调用链并传入具体的主题  
        response = chain.invoke({"topic": "量子计算"})  

        # 6. 打印响应  
        print(response)  

    except Exception as e:  
        # 捕获并打印任何可能发生的错误  
        print(f"调用 DeepSeek API 时发生错误: {e}")  

# 程序入口  
if __name__ == "__main__":  
    main()  