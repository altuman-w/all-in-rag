import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 配置LLM模型，使用AIHubmix服务
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# 备选配置，使用DeepSeek
# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 配置嵌入模型，使用HuggingFace的中文嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载文档数据，从指定markdown文件读取
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 创建向量存储索引
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎
query_engine = index.as_query_engine()

# 打印查询引擎的提示词
print(query_engine.get_prompts())

# 执行查询并打印结果
print(query_engine.query("文中举了哪些例子?"))