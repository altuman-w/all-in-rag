from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]

docs = [Document(page_content=text) for text in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

print('=' * 60)
print(docs)

vectorstore = FAISS.from_documents(docs, embeddings)

local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

print(f"FAISS索引已保存到: {local_faiss_path}")

# 加载FAISS索引
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 验证加载的索引
query = "什么是FAISS？"

results = loaded_vectorstore.similarity_search(query, k=2)

print(f"查询: {query}")
print("最相似的文档:")
for doc in results:
    print(doc.page_content)
