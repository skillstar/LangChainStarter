#pip install sentence-transformers  -i https://pypi.org/simple

from langchain_community.embeddings import HuggingFaceBgeEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_core.documents import Document  

# 初始化嵌入模型  
embeddings = HuggingFaceBgeEmbeddings(  
    model_name="BAAI/bge-large-zh-v1.5",  
    model_kwargs={'device': 'cpu'}  # 使用CPU  
)  

# 准备文档  
docs = [  
    Document(page_content="人工智能是未来"),  
    Document(page_content="机器学习很重要")  
]  

# 生成嵌入  
doc_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])  

# 创建向量存储  
vector_store = FAISS.from_documents(docs, embeddings)  

# 相似性搜索  
query = "AI技术"  
results = vector_store.similarity_search(query, k=2)  

# 打印结果  
for doc in results:  
    print(f"相似文档: {doc.page_content}")  