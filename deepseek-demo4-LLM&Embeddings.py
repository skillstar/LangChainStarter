import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔐 配置 DeepSeek API
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

class RAGAssistant:
    def __init__(self):
        # 1. 初始化嵌入模型
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. 准备知识库文档
        self.docs = [
            Document(page_content="人工智能是一种模仿人类智能的技术，可以学习和解决复杂问题"),
            Document(page_content="机器学习是人工智能的重要分支，通过数据训练模型来提高性能"),
            Document(page_content="深度学习是机器学习的子领域，使用神经网络处理复杂任务"),
            Document(page_content="大语言模型如GPT通过海量文本学习，能够生成人类类似的文本"),
            Document(page_content="计算机视觉使机器能够理解和分析图像，在多个领域有广泛应用")
        ]
        
        # 3. 创建向量存储
        self.vector_store = FAISS.from_documents(self.docs, self.embeddings)
        
        # 4. 初始化大语言模型
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=300
        )
        
        # 5. 设计 RAG 提示模板
        self.rag_prompt = ChatPromptTemplate.from_template(
            "你是一个AI技术专家。请根据以下上下文回答问题。\n"
            "上下文：{context}\n"
            "问题：{question}\n"
            "如果上下文中没有相关信息，请诚实地说明。请用通俗易懂的语言回答。"
        )
    
    def retrieve_relevant_docs(self, query):
        """文档检索 - 找最相关的文档 🔍"""
        # 确保传入的是字符串
        if not isinstance(query, str):
            query = str(query)
        
        # 进行相似性搜索
        results = self.vector_store.similarity_search(query, k=2)
        
        # 格式化文档内容
        context = "\n".join([doc.page_content for doc in results])
        return context
    
    def generate_response(self, query):
        """生成回复的核心方法"""
        # 检索相关文档
        context = self.retrieve_relevant_docs(query)
        
        # 构建完整的提示
        prompt = self.rag_prompt.format(
            context=context, 
            question=query
        )
        
        # 调用语言模型
        response = self.llm.invoke(prompt)
        
        # 返回文本内容
        return response.content

def main():
    # 创建 RAG 助手
    rag_assistant = RAGAssistant()
    
    # 测试问题列表
    questions = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "大语言模型是如何工作的？"
    ]
    
    # 逐个回答问题
    for q in questions:
        print(f"问题: {q}")
        print("回答:", rag_assistant.generate_response(q))
        print("-" * 40)

if __name__ == "__main__":
    main()