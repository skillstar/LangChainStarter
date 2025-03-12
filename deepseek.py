import os  
import pdfplumber  
from typing import List, Optional, Dict, Any  

from langchain_community.embeddings import HuggingFaceBgeEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_core.documents import Document  
from langchain_openai import ChatOpenAI  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_text_splitters import RecursiveCharacterTextSplitter  

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  
# 🔐 配置 DeepSeek API  
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")  
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"  

class PDFRAGAssistant:  
    def __init__(  
        self,   
        pdf_path: Optional[str] = None,   
        embedding_model: str = "BAAI/bge-large-zh-v1.5",  
        chunk_size: int = 300,  
        chunk_overlap: int = 50  
    ):  
        # 1. 初始化文本分割器  
        self.text_splitter = RecursiveCharacterTextSplitter(  
            chunk_size=chunk_size,  
            chunk_overlap=chunk_overlap,  
            length_function=len,  
            separators=["\n\n", "\n", "。", "！", "？"]  
        )  
        
        # 2. 初始化嵌入模型  
        self.embeddings = HuggingFaceBgeEmbeddings(  
            model_name=embedding_model,  
            model_kwargs={'device': 'cpu'}  
        )  
        
        # 3. 初始化文档和向量存储  
        self.docs = []  
        self.vector_store = None  
        
        # 4. 初始化大语言模型  
        self.llm = ChatOpenAI(  
            model="deepseek-chat",  
            temperature=0.7,  
            max_tokens=500,
            streaming=True  
        )  
        
        # 5. RAG 提示模板  
        self.rag_prompt = ChatPromptTemplate.from_template(  
            "你是一个专业的文档分析助手。请根据以下上下文，用专业且清晰的语言回答问题。\n"  
            "文档上下文：{context}\n"  
            "具体问题：{question}\n"  
            "要求：\n"  
            "1. 直接引用文档原文\n"  
            "2. 如上下文不足，说明原因\n"  
            "3. 保持回答的准确性和专业性"  
        )  
        
        # 如果提供了 PDF 路径，则加载文档  
        if pdf_path:  
            self.load_pdf(pdf_path)  
    
    def load_pdf(self, pdf_path: str) -> List[Document]:  
        """从 PDF 文件加载和分割文档"""  
        if not os.path.exists(pdf_path):  
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")  
        
        with pdfplumber.open(pdf_path) as pdf:  
            # 提取所有页面的文本  
            full_text = ""  
            for page in pdf.pages:  
                page_text = page.extract_text() or ""  
                full_text += page_text + "\n\n"  
            
            # 使用文本分割器处理文档  
            docs = self.text_splitter.create_documents(  
                [full_text],   
                metadatas=[{"source": pdf_path}]  
            )  
        
        # 更新文档和向量存储  
        self.docs = docs  
        self.vector_store = FAISS.from_documents(docs, self.embeddings)  
        
        return docs  
    
    def retrieve_relevant_docs(  
        self,   
        query: str,   
        k: int = 3,  
        score_threshold: float = 0.5  # 降低阈值  
    ) -> Dict[str, Any]:  
        """检索相关文档"""  
        if not self.vector_store:  
            raise ValueError("请先加载 PDF 文档")  
        
        # 进行相似性搜索  
        results = self.vector_store.similarity_search_with_score(query, k=k)  
        
        # 格式化文档内容  
        context = "\n\n".join([  
            f"[相似度 {score:.2f}] {doc.page_content}"  
            for doc, score in results  
        ])  
        
        return {  
            "context": context,  
            "details": results  
        }  
    
    def generate_response(self, query: str) -> str:  
        """生成问题回复"""  
        try:  
            # 检索相关文档  
            retrieval_result = self.retrieve_relevant_docs(query)  
            context = retrieval_result['context']  
            
            # 如果没有找到相关文档  
            if not context:  
                return "抱歉，未找到与问题相关的文档内容。请尝试重新表述问题。"  
            
            # 构建完整的提示  
            prompt = self.rag_prompt.format(  
                context=context,   
                question=query  
            )  
            
              # 流式调用语言模型  
            full_response = ""  
            for chunk in self.llm.stream(prompt):  
                chunk_content = chunk.content  
                full_response += chunk_content  
                print(chunk_content, end='', flush=True)  
            
            print()  # 换行  
            return full_response 
        
        except Exception as e:  
            return f"生成回复时发生错误：{e}"  

def main():  
    # PDF 文件路径  
    pdf_path = "./test.pdf"  
    
    # 创建 PDF RAG 助手  
    rag_assistant = PDFRAGAssistant(pdf_path)  
    
    # 测试问题列表  
    questions = [  
        "这份文档的主要内容是什么？",  
        "文档中提到了人工智能的哪些关键观点？",  
        "文档如何描述人机协作？"  
    ]  
    
    # 逐个回答问题  
    for q in questions:  
        print(f"问题: {q}")  
        print("回答:", rag_assistant.generate_response(q))  
        print("-" * 40)  

if __name__ == "__main__":  
    main()  